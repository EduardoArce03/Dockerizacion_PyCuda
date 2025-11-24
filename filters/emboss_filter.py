
import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule



class EmbossFilter:

    def __init__(self):
        self.name = "emboss"
        self.description = "Crea un efecto 3D tipo relieve en la imagen"
        self.kernel_code = """
        __global__ void emboss_kernel(float *input, float *output, float *kernel, 
                                      int width, int height, int kernel_size)
        {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x >= width || y >= height) return;

            int r = kernel_size / 2;
            float sum = 0.0f;

            for (int ky = -r; ky <= r; ky++) {
                int yy = y + ky;
                if (yy < 0 || yy >= height) continue;

                for (int kx = -r; kx <= r; kx++) {
                    int xx = x + kx;
                    if (xx < 0 || xx >= width) continue;

                    float pixel_val = input[yy * width + xx];
                    float kernel_val = kernel[(ky + r) * kernel_size + (kx + r)];
                    sum += pixel_val * kernel_val;
                }
            }

            // Agregar offset de 128 para centrar valores
            output[y * width + x] = sum + 128.0f;
        }
        """
        self.module = SourceModule(self.kernel_code)
        self.cuda_function = self.module.get_function("emboss_kernel")

    def generate_kernel(self, kernel_size):

        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        r = kernel_size // 2

        for i in range(kernel_size):
            for j in range(kernel_size):
                y = i - r
                x = j - r

                diagonal_dist = (x + y) / (2.0 * r) if r > 0 else 0

                if diagonal_dist < -0.3:
                    weight = 1.0 / (1.0 + abs(x) + abs(y))
                    kernel[i, j] = -2.0 * weight
                elif diagonal_dist > 0.3:
                    weight = 1.0 / (1.0 + abs(x) + abs(y))
                    kernel[i, j] = 2.0 * weight
                else:
                    kernel[i, j] = diagonal_dist * 3.0

        kernel_sum = np.sum(kernel)
        kernel -= kernel_sum / (kernel_size * kernel_size)

        return kernel



    def process_gpu(self, image, kernel, block_config, grid_config):

        H, W = image.shape
        K = kernel.shape[0]

        # Allocar memoria GPU
        input_gpu = cuda.mem_alloc(image.nbytes)
        output_gpu = cuda.mem_alloc(image.nbytes)
        kernel_gpu = cuda.mem_alloc(kernel.nbytes)

        # Copiar datos a GPU
        cuda.memcpy_htod(input_gpu, image)
        cuda.memcpy_htod(kernel_gpu, kernel.flatten())

        # Medir tiempo
        start = cuda.Event()
        end = cuda.Event()
        start.record()

        # Ejecutar kernel
        self.cuda_function(
            input_gpu, output_gpu, kernel_gpu,
            np.int32(W), np.int32(H), np.int32(K),
            block=block_config, grid=grid_config
        )

        end.record()
        end.synchronize()
        gpu_time = start.time_till(end)

        # Copiar resultado de vuelta
        output = np.empty_like(image)
        cuda.memcpy_dtoh(output, output_gpu)

        # Liberar memoria
        input_gpu.free()
        output_gpu.free()
        kernel_gpu.free()

        return output, gpu_time

    def get_recommended_block_sizes(self):
        """Retorna configuraciones de bloques recomendadas para este filtro"""
        return [
            {"name": "8x8", "config": (8, 8, 1)},
            {"name": "16x16", "config": (16, 16, 1)},
            {"name": "32x32", "config": (32, 32, 1)},
            {"name": "16x8", "config": (16, 8, 1)},
        ]

    def get_parameters(self):
        """Retorna los parámetros configurables del filtro"""
        return {
            "kernel_size": {
                "type": "int",
                "default": 21,
                "min": 3,
                "max": 101,
                "description": "Tamaño del kernel de convolución (debe ser impar)"
            }
        }