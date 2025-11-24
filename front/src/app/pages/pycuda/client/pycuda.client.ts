// pycuda.service.ts
import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';

// ==================== INTERFACES ====================

export interface HealthResponse {
    status: string;
    cuda_available: boolean;
    memory_free: number;
    memory_total: number;
    filters_loaded: number;
}

export interface FilterInfo {
    name: string;
    description: string;
    parameters: {
        kernel_size: {
            type: string;
            default: number;
            min: number;
            max: number;
            description: string;
        };
    };
    recommended_block_sizes: number[];
}

export interface FiltersResponse {
    success: boolean;
    total_filters: number;
    filters: FilterInfo[];
}

export interface ProcessGPUResponse {
    success: boolean;
    processor: string;
    filter: string;
    image_size: {
        width: number;
        height: number;
    };
    kernel_size: number;
    block_config: {
        x: number;
        y: number;
    };
    grid_config: {
        x: number;
        y: number;
    };
    processing_time_ms: number;
    image_base64?: string; // Solo si return_base64=true
}

export interface CompareResponse {
    success: boolean;
    filter: string;
    image_size: {
        width: number;
        height: number;
    };
    kernel_size: number;
    cpu: {
        processing_time_ms: number;
    };
    gpu: {
        processing_time_ms: number;
        block_config: string;
        grid_config: string;
    };
    speedup: number;
    performance_gain: string;
}

export interface ProcessOptions {
    filter: 'emboss' | 'blur' | 'laplace';
    kernel_size?: number;
    block_size?: 8 | 16 | 32;
    return_base64?: boolean;
}

// ==================== SERVICE ====================

@Injectable({
    providedIn: 'root'
})
export class PyCudaService {
    private readonly apiUrl = 'http://localhost:5000';

    constructor(private http: HttpClient) {}

    // ==================== ENDPOINTS ====================

    /**
     * Health check - Verifica estado del servidor y CUDA
     */
    healthCheck(): Observable<HealthResponse> {
        return this.http.get<HealthResponse>(`${this.apiUrl}/health`);
    }

    /**
     * Lista todos los filtros disponibles
     */
    getFilters(): Observable<FiltersResponse> {
        return this.http.get<FiltersResponse>(`${this.apiUrl}/filters`);
    }

    /**
     * Procesa imagen con GPU
     * @param imageFile - Archivo de imagen
     * @param options - Opciones de procesamiento
     */
    processGPU(imageFile: File, options: ProcessOptions): Observable<ProcessGPUResponse | Blob> {
        const formData = new FormData();
        formData.append('image', imageFile);
        formData.append('filter', options.filter);

        if (options.kernel_size) {
            formData.append('kernel_size', options.kernel_size.toString());
        }

        if (options.block_size) {
            formData.append('block_size', options.block_size.toString());
        }

        const returnBase64 = options.return_base64 !== false; // Default true
        formData.append('return_base64', returnBase64.toString());

        if (returnBase64) {
            // Retorna JSON con base64
            return this.http.post<ProcessGPUResponse>(`${this.apiUrl}/process/gpu`, formData);
        } else {
            // Retorna archivo PNG directamente
            return this.http.post(`${this.apiUrl}/process/gpu`, formData, {
                responseType: 'blob'
            });
        }
    }

    /**
     * Procesa imagen con CPU
     * @param imageFile - Archivo de imagen
     * @param options - Opciones de procesamiento
     */
    processCPU(imageFile: File, options: ProcessOptions): Observable<ProcessGPUResponse | Blob> {
        const formData = new FormData();
        formData.append('image', imageFile);
        formData.append('filter', options.filter);

        if (options.kernel_size) {
            formData.append('kernel_size', options.kernel_size.toString());
        }

        const returnBase64 = options.return_base64 !== false;
        formData.append('return_base64', returnBase64.toString());

        if (returnBase64) {
            return this.http.post<ProcessGPUResponse>(`${this.apiUrl}/process/cpu`, formData);
        } else {
            return this.http.post(`${this.apiUrl}/process/cpu`, formData, {
                responseType: 'blob'
            });
        }
    }

    /**
     * Compara rendimiento CPU vs GPU
     * @param imageFile - Archivo de imagen
     * @param filter - Filtro a usar
     * @param kernelSize - Tamaño del kernel (opcional)
     */
    compare(
        imageFile: File,
        filter: 'emboss' | 'blur' | 'laplace',
        kernelSize?: number
    ): Observable<CompareResponse> {
        const formData = new FormData();
        formData.append('image', imageFile);
        formData.append('filter', filter);

        if (kernelSize) {
            formData.append('kernel_size', kernelSize.toString());
        }

        return this.http.post<CompareResponse>(`${this.apiUrl}/process/compare`, formData);
    }

    // ==================== HELPERS ====================

    /**
     * Convierte base64 a URL de imagen para mostrar en <img>
     * @param base64 - String base64
     */
    base64ToImageUrl(base64: string): string {
        return `data:image/png;base64,${base64}`;
    }

    /**
     * Descarga imagen desde Blob
     * @param blob - Blob de imagen
     * @param filename - Nombre del archivo
     */
    downloadBlob(blob: Blob, filename: string): void {
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        link.click();
        window.URL.revokeObjectURL(url);
    }

    /**
     * Crea URL temporal de un Blob para mostrar en <img>
     * @param blob - Blob de imagen
     */
    blobToImageUrl(blob: Blob): string {
        return window.URL.createObjectURL(blob);
    }
}
