import numpy as np
import time

import pycuda.driver as cuda
from pycuda.compiler import SourceModule


class BlurFilter:
    """
    Filtro de desenfoque gaussiano para imágenes en escala de grises o RGB.

    - Acepta:
        * Imagen 2D (H, W) -> escala de grises
        * Imagen 3D (H, W, 3) -> RGB
    - El kernel es cuadrado (kernel_size x kernel_size), tamaño impar.
    - Usa el contexto CUDA ya creado en la app principal (NO usa pycuda.autoinit).
    """

    def __init__(self):
        self.name = "blur"
        self.description = "Aplica desenfoque gaussiano a la imagen"
        self.supports_color = True  # <- IMPORTANTE para que el backend sepa que acepta RGB

        # Kernel CUDA: convolución 2D sobre todas las bandas (channels)
        self.kernel_code = r"""
        extern "C"
        __global__ void gaussian_blur_rgb(
            const float* input,
            float* output,
            int width,
            int height,
            int channels,
            const float* kernel,
            int kernel_size
        ) {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x >= width || y >= height) return;

            int k_half = kernel_size / 2;

            // Un hilo procesa un píxel completo (todas las bandas)
            for (int c = 0; c < channels; ++c) {
                float sum = 0.0f;

                for (int ky = -k_half; ky <= k_half; ++ky) {
                    for (int kx = -k_half; kx <= k_half; ++kx) {
                        int ix = x + kx;
                        int iy = y + ky;

                        // Padding por replicación de borde
                        if (ix < 0) ix = 0;
                        if (iy < 0) iy = 0;
                        if (ix >= width)  ix = width - 1;
                        if (iy >= height) iy = height - 1;

                        int input_idx = (iy * width + ix) * channels + c;
                        int k_idx = (ky + k_half) * kernel_size + (kx + k_half);

                        sum += input[input_idx] * kernel[k_idx];
                    }
                }

                int out_idx = (y * width + x) * channels + c;
                output[out_idx] = sum;
            }
        }
        """

        # Compilar módulo CUDA una sola vez
        self.module = SourceModule(self.kernel_code)
        self.kernel_func = self.module.get_function("gaussian_blur_rgb")

    # ------------------------------------------------------------------
    # Parámetros configurables para el front
    # ------------------------------------------------------------------
    def get_parameters(self):
        """
        Devuelve la definición de parámetros para el UI (Angular).
        """
        return {
            "kernel_size": {
                "type": "int",
                "label": "Tamaño del kernel",
                "default": 9,
                "min": 3,
                "max": 65,
                "step": 2,
                "description": "Tamaño del kernel gaussiano (debe ser impar)"
            },
            "sigma": {
                "type": "float",
                "label": "Sigma",
                "default": 0.0,
                "min": 0.0,
                "max": 25.0,
                "step": 0.1,
                "description": "Desviación estándar del kernel gaussiano (0 = valor automático)"
            }
        }

    # ------------------------------------------------------------------
    # Generación de kernel gaussiano en CPU
    # ------------------------------------------------------------------
    def generate_kernel(self, kernel_size, sigma=0.0):
        """
        Genera un kernel gaussiano 2D normalizado.

        :param kernel_size: tamaño del kernel (impar)
        :param sigma: desviación estándar; si es 0 o None, se calcula automáticamente.
        :return: kernel (kernel_size x kernel_size) en float32
        """
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size debe ser impar")

        if sigma is None or sigma == 0.0:
            # Heurística común para sigma a partir de kernel_size
            sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8

        k = kernel_size
        center = k // 2

        x = np.arange(k) - center
        y = np.arange(k) - center
        xx, yy = np.meshgrid(x, y)

        kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        kernel /= np.sum(kernel)

        return kernel.astype(np.float32)

    # ------------------------------------------------------------------
    # Procesamiento en GPU
    # ------------------------------------------------------------------
    def process_gpu(self, image, kernel, block_config, grid_config):
        """
        Aplica el blur gaussiano en GPU.

        :param image: numpy array:
                      - 2D (H, W), float32
                      - 3D (H, W, 3), float32
        :param kernel: numpy array 2D (k, k), float32
        :param block_config: (block_x, block_y, 1)
        :param grid_config: (grid_x, grid_y, 1)
        :return: (output_image, tiempo_ms)
        """
        # Normalizar dimensiones
        if image.ndim == 2:
            H, W = image.shape
            C = 1
            image_reshaped = image.reshape(H, W, 1)
        elif image.ndim == 3:
            H, W, C = image.shape
            image_reshaped = image
        else:
            raise ValueError(f"Se esperaba imagen 2D o 3D, se recibió con forma {image.shape}")

        if C not in (1, 3):
            raise ValueError(f"Solo se soportan 1 o 3 canales, se recibió: {C}")

        image_reshaped = np.ascontiguousarray(image_reshaped.astype(np.float32))
        kernel = np.ascontiguousarray(kernel.astype(np.float32))

        kH, kW = kernel.shape
        if kH != kW:
            raise ValueError("El kernel debe ser cuadrado")
        kernel_size = np.int32(kH)

        # Flatten para pasar a CUDA (H * W * C)
        image_flat = image_reshaped.reshape(-1)
        img_bytes = image_flat.nbytes
        k_bytes = kernel.nbytes

        # Asignar memoria en GPU
        d_input = cuda.mem_alloc(img_bytes)
        d_output = cuda.mem_alloc(img_bytes)
        d_kernel = cuda.mem_alloc(k_bytes)

        cuda.memcpy_htod(d_input, image_flat)
        cuda.memcpy_htod(d_kernel, kernel)

        # Eventos para medir tiempo
        start = cuda.Event()
        end = cuda.Event()

        start.record()
        self.kernel_func(
            d_input,
            d_output,
            np.int32(W),
            np.int32(H),
            np.int32(C),
            d_kernel,
            kernel_size,
            block=block_config,
            grid=grid_config
        )
        end.record()
        end.synchronize()

        gpu_time_ms = start.time_till(end)

        # Copiar resultado a host
        output_flat = np.empty_like(image_flat)
        cuda.memcpy_dtoh(output_flat, d_output)

        # Liberar memoria
        d_input.free()
        d_output.free()
        d_kernel.free()

        # Reconstruir forma original
        output = output_flat.reshape(H, W, C)
        if C == 1:
            output = output.reshape(H, W)

        return output, float(gpu_time_ms)

    # ------------------------------------------------------------------
    # (Opcional) CPU, por si luego quieres /process/cpu
    # ------------------------------------------------------------------
    def process_cpu(self, image, kernel):
        """
        Implementación sencilla en CPU (soporta 1 o 3 canales).

        :param image: numpy array 2D (H, W) o 3D (H, W, 3)
        :param kernel: numpy array 2D (k, k)
        :return: (output_image, tiempo_ms)
        """
        if image.ndim == 2:
            H, W = image.shape
            C = 1
            image_reshaped = image.reshape(H, W, 1)
        elif image.ndim == 3:
            H, W, C = image.shape
            image_reshaped = image
        else:
            raise ValueError(f"Se esperaba imagen 2D o 3D, se recibió con forma {image.shape}")

        image_reshaped = image_reshaped.astype(np.float32)
        kH, kW = kernel.shape
        k = kH
        k_half = k // 2

        padded = np.pad(
            image_reshaped,
            pad_width=((k_half, k_half), (k_half, k_half), (0, 0)),
            mode="edge"
        )

        output = np.zeros_like(image_reshaped, dtype=np.float32)

        start = time.time()
        for y in range(H):
            for x in range(W):
                region = padded[y:y + k, x:x + k, :]
                for c in range(C):
                    output[y, x, c] = np.sum(region[:, :, c] * kernel)
        end = time.time()

        if C == 1:
            output = output.reshape(H, W)

        return output, (end - start) * 1000.0  # ms


# Para que FilterFactory pueda instanciarlo:
def create_filter():
    return BlurFilter()
