
import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import time
import math


class gaussianoFilter:

    def __init__(self):
        self.name = "gaussiano"
        self.description = "Suaviza la imagen y reduce el ruido usando un kernel gaussiano"
        print(f"  {self.name.upper()} Filter - Inicializado correctamente")
        
        # Kernel CUDA genérico de convolución
        self.kernel_code = """
        __global__ void convolution_kernel(float *input, float *output, float *kernel, 
                                            int width, int height, int kernel_size, int padded_width)
        {
            // Indexación 1D para mejor eficiencia
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int total_pixels = width * height;
            
            if (idx >= total_pixels) return;
            
            // Convertir índice 1D a coordenadas 2D
            int y = idx / width;
            int x = idx % width;
            
            float sum = 0.0f;
            int k_half = kernel_size / 2; // k_half es el tamaño del padding (pad)
            
            // Aplicar convolución
            for (int ky = 0; ky < kernel_size; ky++) {
                for (int kx = 0; kx < kernel_size; kx++) {
                    
                    // 🛑 CORRECCIÓN DE OFFSET: 
                    // El píxel de salida (y, x) se centra en (y + k_half, x + k_half) 
                    // de la imagen de entrada con padding.
                    // Ajustamos las coordenadas (y, x) para apuntar al píxel correcto en 'input'.
                    int img_y = y + ky;
                    int img_x = x + kx;
                    
                    // Al restar k_half, movemos el punto de inicio de la convolución
                    // de (0,0) del padding a (k_half, k_half) que es donde empieza la imagen real.
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
    
    # --- Generación del Kernel Gaussiano ---
    def generate_kernel(self, kernel_size, sigma=None): 
        """
        Genera un kernel gaussiano 2D. Calcula sigma automáticamente si no se proporciona.
        """
        if kernel_size % 2 == 0:
            raise ValueError("El tamaño del kernel debe ser impar")
        
        # 💡 Cálculo Automático de Sigma
        if sigma is None:
            # Regla práctica: sigma = (kernel_size - 1) / 6.0
            sigma = (kernel_size - 1) / 6.0
            print(f"    ⚠️ Sigma no proporcionado. Calculado automáticamente: sigma = {sigma:.2f}")

        if sigma <= 0:
            raise ValueError("Sigma debe ser mayor que cero")
        
        center = kernel_size // 2
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        sum_val = 0.0
        
        # Constante de la fórmula de Gauss
        sigma_sq = sigma**2
        
        # Fórmula del kernel gaussiano 2D
        for i in range(kernel_size):
            for j in range(kernel_size):
                x = i - center
                y = j - center
                
                # Parte exponencial de la gaussiana: exp(- (x^2 + y^2) / (2 * sigma^2))
                val = np.exp(-((x**2 + y**2) / (2.0 * sigma_sq)))
                
                kernel[i, j] = val
                sum_val += val
        
        # Normalización del kernel: la suma de todos los elementos debe ser 1
        # Esto asegura que el brillo de la imagen no cambie.
        if sum_val == 0:
             # Caso de error si sigma es muy grande o kernel_size muy pequeño
             raise ValueError("Suma del kernel es cero, revise kernel_size y sigma.")
             
        kernel /= sum_val
        
        print(f"    Kernel gaussiano {kernel_size}x{kernel_size} (Sigma={sigma:.2f}) generado.")
        if kernel_size <= 5: 
            print(f"    Suma de elementos: {kernel.sum():.4f}")
            print(f"    {kernel}")
        
        return kernel

    
    # --- El método process_gpu se mantiene sin cambios ---
    def process_gpu(self, image, kernel, block_config, grid_config):
        """
        Procesa la imagen en GPU usando el filtro gaussiano (convolución genérica).
        """
        # Preparar imagen
        imagen_float = image.astype(np.float32)
        kernel_size = kernel.shape[0]
        pad = kernel_size // 2
        height, width = imagen_float.shape
        
        print(f"    Procesando imagen: {height}x{width}")
        print(f"    Kernel: {kernel_size}x{kernel_size}, Padding: {pad}")
        
        # Aplicar padding
        imagen_padded = np.pad(imagen_float, ((pad, pad), (pad, pad)), 
                               mode='constant', constant_values=0.0)
        imagen_padded = np.ascontiguousarray(imagen_padded, dtype=np.float32)
        
        padded_height, padded_width = imagen_padded.shape
        print(f"    Imagen con padding: {padded_height}x{padded_width}")
        
        # Preparar salida y kernel
        output = np.zeros((height, width), dtype=np.float32, order='C')
        kernel_flat = np.ascontiguousarray(kernel.flatten(), dtype=np.float32)
        
        # Configuración del grid (1D)
        threads_per_block = block_config[0] * block_config[1] if len(block_config) > 1 else block_config[0]
        total_pixels = height * width
        num_blocks = (total_pixels + threads_per_block - 1) // threads_per_block
        
        print(f"    Configuración GPU:")
        print(f"      Total píxeles: {total_pixels}")
        print(f"      Hilos por bloque: {threads_per_block}")
        print(f"      Número de bloques: {num_blocks}")
        
        # Medir tiempo
        start_time = time.perf_counter()
        
        try:
            # Alocar memoria GPU
            imagen_gpu = cuda.mem_alloc(imagen_padded.nbytes)
            kernel_gpu = cuda.mem_alloc(kernel_flat.nbytes)
            output_gpu = cuda.mem_alloc(output.nbytes)
            
            # Copiar datos a GPU
            cuda.memcpy_htod(imagen_gpu, imagen_padded)
            cuda.memcpy_htod(kernel_gpu, kernel_flat)
            
            # Ejecutar kernel
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
            
            # Copiar resultado de vuelta
            cuda.memcpy_dtoh(output, output_gpu)
            
            # Liberar memoria
            imagen_gpu.free()
            kernel_gpu.free()
            output_gpu.free()
            
            print(f"    ✓ Procesamiento GPU exitoso")
            
        except Exception as e:
            print(f"    ✗ Error en GPU: {e}")
            raise
        
        # Calcular tiempo
        elapsed_time = time.perf_counter() - start_time
        print(f"    Tiempo GPU: {elapsed_time:.6f} segundos")
        
        # Diagnóstico de valores
        print(f"    Valores salida - Min: {output.min():.2f}, Max: {output.max():.2f}")
        
        # Convertir a uint8
        # Como es un filtro suavizado y normalizado, los valores deben estar entre 0 y 255
        output = np.clip(output, 0, 255).astype(np.uint8)
        
        return output, elapsed_time
    
    def get_recommended_block_sizes(self):
        """Tamaños de bloque recomendados para este filtro (convolución 1D)"""
        return [
            {"name": "12x1 (1D)", "config": (12, 1, 1)},
            {"name": "32x1 (1D)", "config": (32, 1, 1)},
            {"name": "64x1 (1D)", "config": (64, 1, 1)},
            {"name": "128x1 (1D)", "config": (128, 1, 1)},
            {"name": "16x16 (2D)", "config": (16, 16, 1)},
            {"name": "32x32 (2D)", "config": (32, 32, 1)},
        ]
    
    def get_parameters(self):
        """Parámetros configurables del filtro: tamaño de kernel y sigma"""
        return {
            "kernel_size": {
                "type": "int",
                "default": 5,
                "min": 3,
                "max": 65,
                "step": 2,  # Solo impares
                "description": "Tamaño del kernel gaussiano (si no se especifica sigma, se calcula automáticamente)."
            },
            "sigma": {
                "type": "float",
                "default": None, # Cambiado a None para indicar que el cálculo automático es posible
                "min": 0.1,
                "max": 10.0,
                "step": 0.1,
                "description": "Desviación estándar (Sigma): Controla la intensidad del desenfoque. Si es None, se calcula automáticamente."
            }
        }
