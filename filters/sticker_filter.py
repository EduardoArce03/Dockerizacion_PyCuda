"""
Implementación del Filtro de Sticker (Overlay) en PyCUDA.
Funcionalidades: 
1. Borde de Color (Amarillo/Azul) aplicado en GPU.
2. Superposición de dos imágenes PNG (stickers) realizada en CPU.
"""
import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import time
from PIL import Image
import cv2 

class StickerFilter:

    def __init__(self):
        self.name = "sticker_overlay"
        self.description = "Superpone dos stickers con un borde de color."
        print(f"  {self.name.upper()} Filter - Inicializado correctamente")
        
        # Kernel CUDA: Solo aplica el margen de color y copia la imagen base.
        self.kernel_code = """
        __global__ void sticker_overlay_kernel(
            unsigned char *input_img,   
            unsigned char *sticker_img, // Parámetro dummy para compatibilidad
            unsigned char *output_img,  
            int width, int height,
            int sticker_x, int sticker_y, int sticker_w, int sticker_h             
        )
        {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x >= width || y >= height) return;

            int output_idx = (y * width + x) * 4; 
            int input_idx = (y * width + x) * 3;  
            
            // --- CÓDIGO DE MARGEN DE COLOR (Amarillo y Azul) ---
            int line_thickness = 15;
            int spacing = 5;

            // Calcula la distancia mínima al borde
            int x_dist = min(x, width - 1 - x);
            int y_dist = min(y, height - 1 - y);
            int min_dist = min(x_dist, y_dist);

            // 1. Margen Amarillo (Externo): [5px, 20px)
            if (min_dist >= spacing && min_dist < spacing + line_thickness) 
            {
                output_img[output_idx + 0] = 255; 
                output_img[output_idx + 1] = 255; 
                output_img[output_idx + 2] = 0;   // Amarillo
                output_img[output_idx + 3] = 255; 
                return;
            } 
            // 2. Margen Azul (Interno): [25px, 40px)
            else if (min_dist >= spacing + line_thickness + spacing && min_dist < spacing + line_thickness + spacing + line_thickness)
            {
                output_img[output_idx + 0] = 0;   
                output_img[output_idx + 1] = 0;   
                output_img[output_idx + 2] = 255; // Azul
                output_img[output_idx + 3] = 255; 
                return;
            }
            // --- FIN CÓDIGO DE MARGEN ---

            // --- Copiar píxel de la imagen base (con stickers pegados en CPU) ---
            output_img[output_idx + 0] = input_img[input_idx + 0]; 
            output_img[output_idx + 1] = input_img[input_idx + 1]; 
            output_img[output_idx + 2] = input_img[input_idx + 2]; 
            output_img[output_idx + 3] = 255; 

            // Nota: La superposición de stickers en GPU ha sido eliminada.
        }
        """
        
        try:
            self.module = SourceModule(self.kernel_code, options=['--use_fast_math'], no_extern_c=False)
            self.cuda_function = self.module.get_function("sticker_overlay_kernel") 
            print(f"    ✓ Kernel CUDA compilado exitosamente")
        except Exception as e:
            print(f"    ✗ Error compilando kernel: {e}")
            try:
                self.module = SourceModule(self.kernel_code, no_extern_c=True)
                self.cuda_function = self.module.get_function("sticker_overlay_kernel")
                print(f"    ✓ Kernel compilado (sin optimizaciones)")
            except Exception as e2:
                print(f"    ✗ Error crítico: {e2}")
                raise
    
    
    def generate_kernel(self, kernel_size):
        """Método Dummy: Este filtro no usa convolución, solo se mantiene por compatibilidad."""
        print("    [!] StickerFilter no utiliza un kernel de convolución. Saltando generación de kernel.")
        return np.array([[1.0]], dtype=np.float32)

    
    # ❌ La función create_text_image se ha ELIMINADO ❌
    

    def process_gpu(self, image, kernel, sticker_img_path, footer_img_path, sticker_coords=None, block_config=(16, 16, 1), grid_config=None):
        """
        Procesa la imagen: 
        1. Pega 2 stickers (Cabecera y Footer) en CPU (usando PIL).
        2. Aplica el Margen de Color (GPU).
        """
        height, width, channels = image.shape
        if channels != 3:
             raise ValueError("La imagen de entrada debe tener 3 canales (RGB/BGR).")

        # --- 1. Preparar imagen base y superponer stickers en CPU ---
        image_pil_rgba = Image.fromarray(image, mode='RGB').convert('RGBA')
        margin = 20
        
        # Sticker Principal (Cabecera, Superior Derecha)
        sticker_pil_main = Image.open(sticker_img_path).convert("RGBA")
        
        if sticker_coords is None:
            SCALE_FACTOR = 0.30 
            sticker_w = int(width * SCALE_FACTOR)
            sticker_h = int(sticker_w * sticker_pil_main.height / sticker_pil_main.width)
            sticker_x = width - sticker_w - margin
            sticker_y = margin
            sticker_coords = {'x': sticker_x, 'y': sticker_y, 'w': sticker_w, 'h': sticker_h}
        else:
             sticker_w = sticker_coords['w']
             sticker_h = sticker_coords['h']
             sticker_x = sticker_coords['x']
             sticker_y = sticker_coords['y']
        
        sticker_resized_main = sticker_pil_main.resize((sticker_w, sticker_h), Image.Resampling.LANCZOS)
        sticker_resized_main.putalpha(int(255 * 0.80))
        image_pil_rgba.paste(sticker_resized_main, (sticker_x, sticker_y), sticker_resized_main)
        
        
        # Sticker del Footer (Inferior Izquierda)
        footer_pil = Image.open(footer_img_path).convert("RGBA")
        
        FOOTER_SCALE_FACTOR = 0.15 
        footer_w = int(width * FOOTER_SCALE_FACTOR)
        footer_h = int(footer_w * footer_pil.height / footer_pil.width)
        
        footer_x = margin 
        footer_y = height - footer_h - margin 
        
        footer_resized = footer_pil.resize((footer_w, footer_h), Image.Resampling.LANCZOS)
        footer_resized.putalpha(int(255 * 0.80))
        image_pil_rgba.paste(footer_resized, (footer_x, footer_y), footer_resized)


        # --- 2. Preparar datos para GPU (Aplicación del Margen de Color) ---
        image_with_stickers_rgb = np.ascontiguousarray(np.array(image_pil_rgba.convert('RGB')), dtype=np.uint8)
        input_gpu_data = image_with_stickers_rgb 
        output_gpu = np.zeros((height, width, 4), dtype=np.uint8, order='C')

        # Dummy data para el parámetro sticker_img en el kernel (si es necesario por la firma)
        sticker_np_dummy = np.ascontiguousarray(np.array(sticker_resized_main), dtype=np.uint8)

        if grid_config is None:
            grid_config = ( (width + block_config[0] - 1) // block_config[0],
                            (height + block_config[1] - 1) // block_config[1] )
        
        print(f"    Configuración GPU: Bloques={block_config}, Grid={grid_config}")
        
        start_time = time.perf_counter()
        
        try:
            d_input_img = cuda.mem_alloc(input_gpu_data.nbytes)
            d_sticker_img = cuda.mem_alloc(sticker_np_dummy.nbytes) # Dato Dummy
            d_output_img = cuda.mem_alloc(output_gpu.nbytes)
            
            cuda.memcpy_htod(d_input_img, input_gpu_data)
            cuda.memcpy_htod(d_sticker_img, sticker_np_dummy) 
            
            self.cuda_function(
                d_input_img, d_sticker_img, d_output_img,
                np.int32(width), np.int32(height),
                np.int32(sticker_x), np.int32(sticker_y), 
                np.int32(sticker_w), np.int32(sticker_h),
                block=block_config,
                grid=grid_config
            )
            
            cuda.memcpy_dtoh(output_gpu, d_output_img)
            
            d_input_img.free()
            d_sticker_img.free()
            d_output_img.free()
            
            print(f"    ✓ Procesamiento GPU exitoso")
            
        except Exception as e:
            print(f"    ✗ Error en GPU: {e}")
            raise
        
        elapsed_time = time.perf_counter() - start_time
        print(f"    Tiempo GPU: {elapsed_time:.6f} segundos")
        
        return output_gpu, elapsed_time
    
    def get_recommended_block_sizes(self):
        return [
            {"name": "16x16 (2D)", "config": (16, 16, 1)},
            {"name": "32x8 (2D)", "config": (32, 8, 1)},
            {"name": "32x32 (2D)", "config": (32, 32, 1)},
        ]
    
    def get_parameters(self):
        return {
            "kernel_size": {
                "type": "int",
                "default": 1,
                "min": 1,
                "max": 1,
                "step": 1,
                "description": "No se usa para este filtro de sticker (solo para compatibilidad)."
            },
            "sticker_img_path": {
                "type": "str",
                "default": "filters/logoUps.png",
                "description": "Ruta a la imagen PNG del sticker principal (Cabecera)."
            },
            "footer_img_path": {
                "type": "str",
                "default": "filters/DonBosco.png",
                "description": "Ruta a la imagen PNG del sticker inferior (Footer/Firma)."
            }
        }