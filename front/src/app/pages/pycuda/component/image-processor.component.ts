import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

import { MessageService } from 'primeng/api';

import { BlockUIModule } from 'primeng/blockui';
import { DividerModule } from 'primeng/divider';
import { TagModule } from 'primeng/tag';
import { PanelModule } from 'primeng/panel';
import { ButtonModule } from 'primeng/button';
import { TabsModule } from 'primeng/tabs';
import { ImageModule } from 'primeng/image';
import { SliderModule } from 'primeng/slider';
import { SelectModule } from 'primeng/select';
import { FileUploadModule } from 'primeng/fileupload';
import { ChipModule } from 'primeng/chip';
import { CardModule } from 'primeng/card';
import { ToastModule } from 'primeng/toast';

import {
    PyCudaService,
    ProcessOptions,
    ProcessGPUResponse,
    CompareResponse
} from '../client/pycuda.client';

interface FilterOption {
    label: string;
    value: 'emboss' | 'blur' | 'laplace' | 'gaussian' | 'sticker' | 'depth_of_field_duotone';
    icon: string;
}

interface BlockSizeOption {
    label: string;
    value: 8 | 16 | 32;
    description: string;
}

interface DefaultImage {
  name: string;
  url: string;
  file: File | null;
}

@Component({
    selector: 'app-image-processor',
    templateUrl: './image-processor.component.html',
    styleUrls: ['./image-processor.component.scss'],
    standalone: true,
    imports: [
        CommonModule,
        FormsModule,
        BlockUIModule,
        DividerModule,
        TagModule,
        PanelModule,
        ButtonModule,
        TabsModule,
        ImageModule,
        SliderModule,
        SelectModule,
        FileUploadModule,
        ChipModule,
        CardModule,
        ToastModule
    ],
    providers: [
        PyCudaService,
        MessageService
    ]
})
export class ImageProcessorComponent implements OnInit {

    defaultImages: DefaultImage[] = [
        {
            name: 'Paisaje Montaña',
            url: 'https://images.pexels.com/photos/34191214/pexels-photo-34191214.jpeg',
            file: null
        },
        {
            name: 'Ciudad Nocturna',
            url: 'https://images.pexels.com/photos/1123972/pexels-photo-1123972.jpeg',
            file: null
        },
        {
            name: 'Naturaleza',
            url: 'https://images.pexels.com/photos/158063/bellingrath-gardens-alabama-landscape-scenic-158063.jpeg',
            file: null
        },
        {
            name: 'Persona',
            url: 'https://images.pexels.com/photos/516927/pexels-photo-516927.jpeg',
            file: null
        },
        {
            name: 'Arquitectura',
            url: 'https://images.pexels.com/photos/135018/pexels-photo-135018.jpeg',
            file: null
        },
        {
            name: 'Abstract',
            url: 'https://images.pexels.com/photos/28359694/pexels-photo-28359694.jpeg',
            file: null
        }
    ];

    selectedFile: File | null = null;
    previewUrl: string = '';
    processedImageUrl: string = '';
    isProcessing = false;
    cudaAvailable = false;
    //Rutas de imagenes
    imgEmboss = '/imagenes/imgEmboss.png';
    imgBlur = '/imagenes/imgBlur.png';
    imgLaplace = '/imagenes/imgLaplace.jpg';

    filterOptions: FilterOption[] = [
        { label: ' Emboss (Relieve 3D)', value: 'emboss', icon: 'pi-image' },
        { label: ' Blur (Desenfoque)', value: 'blur', icon: 'pi-circle' },
        { label: ' Laplace (Bordes)', value: 'laplace', icon: 'pi-box' },
        { label: ' Gaussian (Suavizado)', value: 'gaussian', icon: 'pi-star' },
        { label: ' Sticker (Superposición)', value: 'sticker', icon: 'pi-image' },
        { label: ' Extra (Profundidad de Campo)', value: 'depth_of_field_duotone', icon: 'pi-image' }
    ];

    blockSizeOptions: BlockSizeOption[] = [
        { label: '8x8 (Imágenes pequeñas)', value: 8, description: 'Para imágenes < 1K pixels' },
        { label: '16x16 (Óptimo - Recomendado)', value: 16, description: 'Para imágenes HD' },
        { label: '32x32 (Imágenes grandes)', value: 32, description: 'Para imágenes > 4K' }
    ];

    // Configuración
    selectedFilter: 'emboss' | 'blur' | 'laplace' | 'gaussian' | 'sticker' | 'depth_of_field_duotone' = 'emboss';
    kernelSize = 21;
    blockSize: 8 | 16 | 32 = 16;

    processingTime: number | null = null;
    comparisonResult: CompareResponse | null = null;

    constructor(
        private pycudaService: PyCudaService,
        private messageService: MessageService
    ) {}

    ngOnInit(): void {
        this.checkHealth();
    }


    checkHealth(): void {
        this.pycudaService.healthCheck().subscribe({
            next: (response) => {
                this.cudaAvailable = response.cuda_available;
                if (response.cuda_available) {
                    const memoryGB = (response.memory_free / 1024 ** 3).toFixed(2);
                    this.showSuccess(` CUDA disponible - ${memoryGB} GB libres`);
                } else {
                }
            },
            error: (error) => {
                this.showError(' Error conectando con el servidor');
                console.error(error);
            }
        });
    }

    async selectDefaultImage(defaultImg: DefaultImage) {
        try {
        const response = await fetch(defaultImg.url);
        const blob = await response.blob();
        
        const file = new File([blob], defaultImg.name + '.jpg', { type: 'image/jpeg' });
        
        this.selectedFile = file;
        
        const reader = new FileReader();
        reader.onload = (e: any) => {
            this.previewUrl = e.target.result;
        };
        reader.readAsDataURL(file);
        
        this.processedImageUrl = '';
        this.comparisonResult = null;
        
        this.messageService.add({
            severity: 'success',
            summary: 'Imagen Seleccionada',
            detail: `${defaultImg.name} cargada correctamente`
                });
        } catch (error) {
        this.messageService.add({
            severity: 'error',
            summary: 'Error',
            detail: 'No se pudo cargar la imagen predeterminada'
        });
        console.error('Error al cargar imagen predeterminada:', error);
        }
    } 

    onFileSelected(event: any): void {
        const file = event.files[0];

        if (file && file.type.startsWith('image/')) {
            this.selectedFile = file;

            const reader = new FileReader();
            reader.onload = (e: any) => {
                this.previewUrl = e.target.result;
            };
            reader.readAsDataURL(file);

            this.processedImageUrl = '';
            this.showInfo(`Imagen cargada: ${file.name}`);
        } else {
            this.showError('Por favor selecciona una imagen válida');
        }
    }


    processWithGPU(): void {
        if (!this.selectedFile) {
            this.showWarn('Selecciona una imagen primero');
            return;
        }

        this.isProcessing = true;

        const options: ProcessOptions = {
            filter: this.selectedFilter,
            kernel_size: this.kernelSize,
            block_size: this.blockSize,
            return_base64: true
        };

        this.pycudaService.processGPU(this.selectedFile, options).subscribe({
            next: (response) => {
                const result = response as ProcessGPUResponse;
                this.processedImageUrl = this.pycudaService.base64ToImageUrl(result.image_base64!);
                this.processingTime = result.processing_time_ms;
                this.showSuccess(` Procesado con GPU en ${result.processing_time_ms.toFixed(2)} ms`);
                this.isProcessing = false;
            },
            error: (error: any) => {
                this.showError(` Error: ${error.error?.error || error.message}`);
                this.isProcessing = false;
                console.error(error);
            }
        });
    }

    processWithCPU(): void {
        if (!this.selectedFile) {
            this.showWarn('Selecciona una imagen primero');
            return;
        }

        this.isProcessing = true;

        const options: ProcessOptions = {
            filter: this.selectedFilter,            
            kernel_size: this.kernelSize,
            return_base64: true
        };

        this.pycudaService.processCPU(this.selectedFile, options).subscribe({
            next: (response) => {
                const result = response as ProcessGPUResponse;
                this.processedImageUrl = this.pycudaService.base64ToImageUrl(result.image_base64!);
                this.processingTime = result.processing_time_ms;
                this.showInfo(` Procesado con CPU en ${result.processing_time_ms.toFixed(2)} ms`);
                this.isProcessing = false;
            },
            error: (error: any) => {
                this.showError(` Error: ${error.error?.error || error.message}`);
                this.isProcessing = false;
                console.error(error);
            }
        });
    }

    comparePerformance(): void {
        if (!this.selectedFile) {
            this.showWarn('Selecciona una imagen primero');
            return;
        }

        this.isProcessing = true;

        this.pycudaService.compare(this.selectedFile, this.selectedFilter, this.kernelSize).subscribe({
            next: (response) => {
                this.comparisonResult = response;
                this.showSuccess(` GPU es ${response.speedup}x más rápido que CPU!`);
                this.isProcessing = false;
            },
            error: (error) => {
                this.showError(` Error en comparación: ${error.error?.error || error.message}`);
                this.isProcessing = false;
                console.error(error);
            }
        });
    }

    downloadProcessedImage(): void {
        if (!this.selectedFile || !this.processedImageUrl) {
            return;
        }

        if (this.processedImageUrl.startsWith('data:')) {
            const link = document.createElement('a');
            link.href = this.processedImageUrl;
            link.download = `${this.selectedFilter}_${this.selectedFile.name}`;
            link.click();
            this.showSuccess('Imagen descargada');
            return;
        }

        const options: ProcessOptions = {
            filter: this.selectedFilter,
            kernel_size: this.kernelSize,
            block_size: this.blockSize,
            return_base64: false
        };

        this.pycudaService.processGPU(this.selectedFile, options).subscribe({
            next: (blob) => {
                const filename = `${this.selectedFilter}_${this.selectedFile!.name}`;
                this.pycudaService.downloadBlob(blob as Blob, filename);
                this.showSuccess(' Imagen descargada');
            },
            error: (error: any) => {
                this.showError(' Error descargando imagen');
                console.error(error);
            }
        });
    }


    reset(): void {
        this.selectedFile = null;
        this.previewUrl = '';
        this.processedImageUrl = '';
        this.processingTime = null;
        this.comparisonResult = null;
        this.showInfo('Estado reiniciado');
    }

    validateKernelSize(): void {
        if (this.kernelSize % 2 === 0) {
            this.kernelSize++;
        }
        if (this.kernelSize < 3) {
            this.kernelSize = 3;
        }
        if (this.kernelSize > 101) {
            this.kernelSize = 101;
        }
    }


    private showSuccess(message: string): void {
        this.messageService.add({
            severity: 'success',
            summary: 'Éxito',
            detail: message,
            life: 3000
        });
    }

    private showError(message: string): void {
        this.messageService.add({
            severity: 'error',
            summary: 'Error',
            detail: message,
            life: 3000
        });
    }

    private showWarn(message: string): void {
        this.messageService.add({
            severity: 'warn',
            summary: 'Advertencia',
            detail: message,
            life: 3000
        });
    }

    private showInfo(message: string): void {
        this.messageService.add({
            severity: 'info',
            summary: 'Información',
            detail: message,
            life: 3000
        });
    }
}
