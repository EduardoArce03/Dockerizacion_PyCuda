"""
Implementación del Filtro Gaussiano (Gaussian Blur) en PyCUDA.
Suaviza la imagen y reduce el ruido usando un kernel Gaussiano 2D.
Incluye cálculo automático de Sigma y corrección de offset para la convolución.
"""
import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import time
import math


class GaussianFilter:

    def __init__(self):
        self.name = "gaussian"
        self.description = "Suaviza la imagen y reduce el ruido usando un kernel Gaussiano"
        print(f"  {self.name.upper()} Filter - Inicializado correctamente")
        
        # Kernel CUDA genérico de convolución (con corrección de offset)
        self.kernel_code = """
        __global__ void convolution_kernel(float *input, float *output, float *kernel, 
                                            int width, int height, int kernel_size, int padded_width)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int total_pixels = width * height;
            
            if (idx >= total_pixels) return;
            
            int y = idx / width;
            int x = idx % width;
            
            float sum = 0.0f;
            int k_half = kernel_size / 2;
            
            // Aplicar convolución
            for (int ky = 0; ky < kernel_size; ky++) {
                for (int kx = 0; kx < kernel_size; kx++) {
                    
                    // CORRECCIÓN DE OFFSET: 
                    int img_y = y + ky;
                    int img_x = x + kx;
                    
                    int img_idx = (img_y - k_half) * padded_width + (img_x - k_half); 
                    int ker_idx = ky * kernel_size + kx;
                    
                    sum += input[img_idx] * kernel[ker_idx];
                }
            }
            
            output[idx] = sum; 
        }
        """
        
        try:
            self.module = SourceModule(self.kernel_code, 
                                     options=['--use_fast_math'],
                                     no_extern_c=False)
            self.cuda_function = self.module.get_function("convolution_kernel") 
            print(f"    ✓ Kernel CUDA compilado exitosamente")
        except Exception as e:
            print(f"    ✗ Error compilando kernel: {e}")
            try:
                self.module = SourceModule(self.kernel_code, no_extern_c=True)
                self.cuda_function = self.module.get_function("convolution_kernel")
                print(f"    ✓ Kernel compilado (sin optimizaciones)")
            except Exception as e2:
                print(f"    ✗ Error crítico: {e2}")
                raise
    
    
    def generate_kernel(self, kernel_size, sigma=None): 
        if kernel_size % 2 == 0:
            raise ValueError("El tamaño del kernel debe ser impar")
        
        if sigma is None:
            sigma = (kernel_size - 1) / 6.0
            print(f"    ⚠️ Sigma no proporcionado. Calculado automáticamente: sigma = {sigma:.2f}")

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
             raise ValueError("Suma del kernel es cero, revise kernel_size y sigma.")
             
        kernel /= sum_val
        
        print(f"    Kernel Gaussiano {kernel_size}x{kernel_size} (Sigma={sigma:.2f}) generado.")
        return kernel

    
    def process_gpu(self, image, kernel, block_config, grid_config):
        imagen_float = image.astype(np.float32)
        kernel_size = kernel.shape[0]
        pad = kernel_size // 2
        height, width = imagen_float.shape
        
        imagen_padded = np.pad(imagen_float, ((pad, pad), (pad, pad)), 
                               mode='constant', constant_values=0.0)
        imagen_padded = np.ascontiguousarray(imagen_padded, dtype=np.float32)
        
        padded_height, padded_width = imagen_padded.shape
        output = np.zeros((height, width), dtype=np.float32, order='C')
        kernel_flat = np.ascontiguousarray(kernel.flatten(), dtype=np.float32)
        
        threads_per_block = block_config[0] * block_config[1] if len(block_config) > 1 else block_config[0]
        total_pixels = height * width
        num_blocks = (total_pixels + threads_per_block - 1) // threads_per_block
        
        start_time = time.perf_counter()
        
        try:
            imagen_gpu = cuda.mem_alloc(imagen_padded.nbytes)
            kernel_gpu = cuda.mem_alloc(kernel_flat.nbytes)
            output_gpu = cuda.mem_alloc(output.nbytes)
            
            cuda.memcpy_htod(imagen_gpu, imagen_padded)
            cuda.memcpy_htod(kernel_gpu, kernel_flat)
            
            self.cuda_function(
                imagen_gpu,
                output_gpu,
                kernel_gpu,
                np.int32(width),
                np.int32(height),
                np.int32(kernel_size),
                np.int32(padded_width),
                block=(threads_per_block, 1, 1),
                grid=(num_blocks, 1)
            )
            
            cuda.memcpy_dtoh(output, output_gpu)
            
            imagen_gpu.free()
            kernel_gpu.free()
            output_gpu.free()
            
        except Exception as e:
            print(f"    ✗ Error en GPU: {e}")
            raise
        
        elapsed_time = time.perf_counter() - start_time
        output = np.clip(output, 0, 255).astype(np.uint8)
        
        return output, elapsed_time
    
    def get_recommended_block_sizes(self):
        return [
            {"name": "12x1 (1D)", "config": (12, 1, 1)},
            {"name": "32x1 (1D)", "config": (32, 1, 1)},
            {"name": "64x1 (1D)", "config": (64, 1, 1)},
            {"name": "128x1 (1D)", "config": (128, 1, 1)},
            {"name": "16x16 (2D)", "config": (16, 16, 1)},
            {"name": "32x32 (2D)", "config": (32, 32, 1)},
        ]
    
    def get_parameters(self):
        return {
            "kernel_size": {
                "type": "int",
                "default": 5,
                "min": 3,
                "max": 65,
                "step": 2,
                "description": "Tamaño del kernel Gaussiano (si no se especifica sigma, se calcula automáticamente)."
            },
            "sigma": {
                "type": "float",
                "default": None, 
                "min": 0.1,
                "max": 10.0,
                "step": 0.1,
                "description": "Desviación estándar (Sigma): Controla la intensidad del desenfoque."
            }
        }