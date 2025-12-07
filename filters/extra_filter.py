"""
Implementación del Filtro Depth of Field (Duotono Condicional).
El fondo desenfocado se transforma en un degradado Azul/Amarillo fuera de la Región de Interés (ROI).
"""
import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import time
import math
from PIL import Image

class DepthOfFieldFilter:

    def __init__(self):
        self.name = "depth_of_field_duotone"
        self.description = "Aplica desenfoque Gaussiano Duotono (Azul/Amarillo) fuera de una Región de Interés."
        print(f"  {self.name.upper()} Filter - Inicializado correctamente")

        self.kernel_code = """
        // Kernel que procesa la convolución en gris pero produce salida en RGB Duotono (3 canales)
        __global__ void conditional_convolution_duotone_kernel(
            float *input_padded,    // Imagen Gris con padding (para calcular el Blur)
            float *input_original,  // Imagen RGB plana (para la ROI nítida)
            unsigned char *output,  // Salida RGB (3 canales)
            float *kernel, 
            int width, 
            int height, 
            int kernel_size, 
            int padded_width,
            int x_start, int y_start, int x_end, int y_end // Parámetros de la ROI
        )
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int total_pixels = width * height;
            
            if (idx >= total_pixels) return;
            
            int y = idx / width;
            int x = idx % width;
            
            // Índice de salida RGB (3 canales)
            int output_idx = idx * 3; 

            // --- 1. Lógica Condicional: Decidir si mantener nítido (Original) ---
            
            if (x >= x_start && x <= x_end && y >= y_start && y <= y_end) {
                // DENTRO de la ROI: Mantenemos el color RGB original nítido
                int original_idx_rgb = idx * 3;
                output[output_idx + 0] = (unsigned char)input_original[original_idx_rgb + 0]; 
                output[output_idx + 1] = (unsigned char)input_original[original_idx_rgb + 1]; 
                output[output_idx + 2] = (unsigned char)input_original[original_idx_rgb + 2]; 
                return;
            }
            
            // --- 2. Si el píxel está FUERA de la ROI, aplicamos la convolución de GRISES ---

            float sum = 0.0f;
            int k_half = kernel_size / 2;
            
            for (int ky = 0; ky < kernel_size; ky++) {
                for (int kx = 0; kx < kernel_size; kx++) {
                    
                    int img_y = y + ky;
                    int img_x = x + kx;
                    
                    int img_idx = (img_y - k_half) * padded_width + (img_x - k_half); 
                    int ker_idx = ky * kernel_size + kx;
                    
                    sum += input_padded[img_idx] * kernel[ker_idx];
                }
            }
            
            float intensity = sum; // Intensidad de gris desenfocada (0 a 255)
            
            // --- 3. CONVERSIÓN A DUOTONO AZUL-AMARILLO (Mapeo) ---
            
            float f_intensity = intensity / 255.0f; // 0.0 a 1.0
            float inv_f_intensity = 1.0f - f_intensity; // 1.0 a 0.0
            
            // Colores Base: Oscuro (Azul) y Claro (Amarillo)
            float blue_R = 0.0f;
            float blue_G = 0.0f;
            float blue_B = 150.0f; // Azul Profundo
            
            float yellow_R = 255.0f;
            float yellow_G = 255.0f;
            float yellow_B = 0.0f;
            
            // Interpolación lineal
            float final_R = (blue_R * inv_f_intensity) + (yellow_R * f_intensity);
            float final_G = (blue_G * inv_f_intensity) + (yellow_G * f_intensity);
            float final_B = (blue_B * inv_f_intensity) + (yellow_B * f_intensity);
            
            // Escribir la salida RGB Duotono
            output[output_idx + 0] = (unsigned char)min(255.0f, final_R); 
            output[output_idx + 1] = (unsigned char)min(255.0f, final_G); 
            output[output_idx + 2] = (unsigned char)min(255.0f, final_B);
        }
        """
        
        try:
            self.module = SourceModule(self.kernel_code, options=['--use_fast_math'], no_extern_c=False)
            self.cuda_function = self.module.get_function("conditional_convolution_duotone_kernel") 
            print(f"    ✓ Kernel CUDA Duotono compilado exitosamente")
        except Exception as e:
            print(f"    ✗ Error compilando kernel Duotono: {e}")
            raise
    
    
    def generate_kernel(self, kernel_size, sigma=None): 
        if kernel_size % 2 == 0:
            raise ValueError("El tamaño del kernel debe ser impar")
        
        if sigma is None:
            sigma = 3.0 
        
        if sigma <= 0:
            raise ValueError("Sigma debe ser mayor que cero")
        
        center = kernel_size // 2
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        sum_val = 0.0
        sigma_sq = sigma**2
        
        for i in range(kernel_size):
            for j in range(kernel_size):
                x = i - center
                y = j - center
                
                val = np.exp(-((x**2 + y**2) / (2.0 * sigma_sq)))
                
                kernel[i, j] = val
                sum_val += val
        
        if sum_val == 0:
            raise ValueError("Suma del kernel es cero.")
            
        kernel /= sum_val
        
        print(f"    Kernel Gaussiano {kernel_size}x{kernel_size} (Sigma={sigma:.2f}) generado para DOF.")
        return kernel

    
    def process_gpu(self, image, kernel, block_config, grid_config, roi_coords=None):
        
        if image.ndim != 3 or image.shape[2] != 3:
             raise ValueError("La imagen de entrada debe ser RGB (3 canales) para el Duotono Condicional.")

        image_float_rgb = image.astype(np.float32)
        
        image_gray_float = np.mean(image_float_rgb, axis=2)
        
        imagen_float = image_gray_float
        height, width = imagen_float.shape
        
        kernel_size = kernel.shape[0]
        pad = kernel_size // 2
        
        imagen_padded = np.pad(imagen_float, ((pad, pad), (pad, pad)), 
                               mode='constant', constant_values=0.0)
        imagen_padded = np.ascontiguousarray(imagen_padded, dtype=np.float32)

        imagen_original_flat = np.ascontiguousarray(image.flatten(), dtype=np.float32)
        
        
        if roi_coords is None or all(c == 0 for c in roi_coords):
            x_start = width // 4
            x_end = width * 3 // 4
            y_start = height // 4
            y_end = height * 3 // 4
            roi_coords = (x_start, y_start, x_end, y_end)

        x_start, y_start, x_end, y_end = map(np.int32, roi_coords)

        padded_width = np.int32(imagen_padded.shape[1])
        
        output = np.zeros((height, width, 3), dtype=np.uint8, order='C') 
        kernel_flat = np.ascontiguousarray(kernel.flatten(), dtype=np.float32)
        
        threads_per_block = block_config[0] * block_config[1] if len(block_config) > 1 else block_config[0]
        total_pixels = height * width
        num_blocks = (total_pixels + threads_per_block - 1) // threads_per_block
        
        print(f"    ROI Duotono activa: ({x_start},{y_start}) a ({x_end},{y_end})")
        start_time = time.perf_counter()
        
        try:
            d_input_padded = cuda.mem_alloc(imagen_padded.nbytes)
            d_input_original = cuda.mem_alloc(imagen_original_flat.nbytes)
            d_kernel = cuda.mem_alloc(kernel_flat.nbytes)
            d_output = cuda.mem_alloc(output.nbytes)
            
            cuda.memcpy_htod(d_input_padded, imagen_padded)
            cuda.memcpy_htod(d_input_original, imagen_original_flat)
            cuda.memcpy_htod(d_kernel, kernel_flat)
            
            self.cuda_function(
                d_input_padded,
                d_input_original,
                d_output,
                d_kernel,
                np.int32(width),
                np.int32(height),
                np.int32(kernel_size),
                padded_width,
                x_start, y_start, x_end, y_end,
                block=(threads_per_block, 1, 1),
                grid=(num_blocks, 1)
            )
            
            cuda.memcpy_dtoh(output, d_output)
            
            d_input_padded.free()
            d_input_original.free()
            d_kernel.free()
            d_output.free()
            
        except Exception as e:
            print(f"    ✗ Error en GPU: {e}")
            raise
        
        elapsed_time = time.perf_counter() - start_time
        
        return output, elapsed_time
    
    
    def get_recommended_block_sizes(self):
        return [
            {"name": "128x1 (1D)", "config": (128, 1, 1)},
            {"name": "256x1 (1D)", "config": (256, 1, 1)},
        ]
    
    def get_parameters(self):
        return {
            "kernel_size": {
                "type": "int",
                "default": 15,
                "min": 3,
                "max": 65,
                "step": 2,
                "description": "Tamaño del kernel Gaussiano para el desenfoque de fondo (debe ser impar)."
            },
            "sigma": {
                "type": "float",
                "default": 3.0,
                "min": 1.0,
                "max": 10.0,
                "step": 0.1,
                "description": "Desviación estándar (Sigma) del blur."
            },
            "roi_x_start": {"type": "int", "default": 0, "description": "Coordenada X de inicio de la Región Nítida (0=Calculado)."},
            "roi_y_start": {"type": "int", "default": 0, "description": "Coordenada Y de inicio de la Región Nítida (0=Calculado)."},
            "roi_x_end": {"type": "int", "default": 0, "description": "Coordenada X de fin de la Región Nítida (0=Calculado)."},
            "roi_y_end": {"type": "int", "default": 0, "description": "Coordenada Y de fin de la Región Nítida (0=Calculado)."}
        }