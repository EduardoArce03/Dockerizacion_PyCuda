from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
from PIL import Image
import time
import io
import base64
import os
from werkzeug.utils import secure_filename
import traceback

print("="*70)
print(" Inicializando CUDA...")
print("="*70)

import pycuda.driver as cuda
import pycuda.tools

cuda.init()

device = cuda.Device(0)
print(f"✅ Dispositivo: {device.name()}")
print(f"   Compute Capability: {device.compute_capability()}")

ctx = pycuda.tools.make_default_context()
print(f"✅ Contexto CUDA creado (auto-managed)")

free_mem, total_mem = cuda.mem_get_info()
print(f" Memoria libre: {free_mem / (1024**3):.2f} GB")
print(f" Memoria total: {total_mem / (1024**3):.2f} GB")

print("\n📦 Cargando filtros...")
from filters import FilterFactory

print(" Pre-compilando kernels CUDA...")
filter_instances = {}
for filter_info in FilterFactory.get_available_filters():
    filter_name = filter_info['name']
    try:
        # Crea la instancia y la guarda solo si la inicialización (compilación CUDA) fue exitosa
        filter_instances[filter_name] = FilterFactory.create_filter(filter_name)
        print(f"   {filter_name}: Pre-compilado")
    except Exception as e:
        print(f"   ❌ {filter_name}: Error al compilar/cargar: {str(e)[:80]}")


app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Lista de filtros cargados exitosamente para la API
available_filters = [
    f for f in FilterFactory.get_available_filters() 
    if f['name'] in filter_instances
]
print(f"✅ Filtros cargados: {len(available_filters)}")
for f in available_filters:
    print(f"   • {f['name']}: {f['description']}")


# Cleanup
import atexit
def cleanup():
    print("\n🧹 Cerrando contexto CUDA...")
    try:
        ctx.detach()
        print("✅ Contexto cerrado correctamente")
    except:
        pass

atexit.register(cleanup)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# --- FUNCIONES DE CARGA MODIFICADAS ---

def load_image_grayscale(image_data):
    """Carga la imagen en escala de grises (L) y la convierte a float32 (para convolución)."""
    img = Image.open(io.BytesIO(image_data)).convert('L')
    img_array = np.array(img, dtype=np.float32)
    return img_array, img.size


def load_image_color(image_data):
    """Carga la imagen en color (RGB) y la convierte a np.uint8 (para StickerFilter y Duotono)."""
    img = Image.open(io.BytesIO(image_data)).convert('RGB')
    img_array = np.array(img, dtype=np.uint8) 
    return img_array, img.size


def array_to_image_bytes(img_array, format='PNG'):
    """Convierte el array numpy (L, RGB o RGBA) a bytes de imagen."""
    img_array = np.clip(img_array, 0, 255)
    
    # Determinar el modo (L, RGB, RGBA)
    if len(img_array.shape) == 3:
        if img_array.shape[2] == 4:
            mode = 'RGBA'
        elif img_array.shape[2] == 3:
            mode = 'RGB'
        else:
            mode = 'L'
    else:
        mode = 'L'
        
    img = Image.fromarray(img_array.astype(np.uint8), mode=mode)
    
    buffer = io.BytesIO()
    img.save(buffer, format=format)
    buffer.seek(0)
    return buffer

# --- FIN DE FUNCIONES DE CARGA MODIFICADAS ---


@app.route('/')
def index():
    return jsonify({
        "message": "🎨 Image Convolution Filter API with PyCUDA",
        "version": "2.0.0",
        "mode": "Production (no auto-reload)",
        "available_filters": [f["name"] for f in available_filters]
    })


@app.route('/health', methods=['GET'])
def health():
    ctx.push()
    try:
        fm, tm = cuda.mem_get_info()
        return jsonify({
            "status": "healthy",
            "cuda_available": True,
            "memory_free": fm,
            "memory_total": tm,
            "filters_loaded": len(available_filters)
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "cuda_available": False,
            "error": str(e)
        }), 500
    finally:
        ctx.pop()


@app.route('/filters', methods=['GET'])
def list_filters():
    return jsonify({
        "success": True,
        "total_filters": len(available_filters),
        "filters": available_filters
    })



@app.route('/process/gpu', methods=['POST'])
def process_gpu():
    ctx.push()

    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image'] 
        
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({"error": "Invalid file"}), 400

        filter_name = request.form.get('filter', '').lower()
        if not filter_name:
            return jsonify({"error": "Filter parameter is required"}), 400

        if filter_name not in filter_instances:
            return jsonify({
                "error": f"Filter '{filter_name}' not found",
                "available_filters": list(filter_instances.keys())
            }), 400

        filter_instance = filter_instances[filter_name]
        
        
        image_data = file.read()
        
        if filter_name in ['sticker_overlay', 'sticker', 'depth_of_field', 'depth_of_field_duotone']:
            image, _ = load_image_color(image_data)
        else:
            image, _ = load_image_grayscale(image_data)
        
        # 3. Determinar W, H
        if len(image.shape) == 3:
            H, W, C = image.shape
        else:
            H, W = image.shape
            C = 1

        # 4. Obtener kernel_size (Seguro)
        kernel_size = int(request.form.get('kernel_size',
                                     filter_instance.get_parameters()['kernel_size']['default']))
        
        # 5. Generar kernel (Dummy para StickerFilter, Real para Convolución/Duotono)
        kernel = filter_instance.generate_kernel(kernel_size)
        
        # 6. Configurar el Grid/Block
        block_size = int(request.form.get('block_size', 16))
        block_config = (block_size, block_size, 1)

        # Determinar Grid basado en el tipo de filtro
        if filter_name in ['sticker_overlay', 'sticker', 'depth_of_field', 'depth_of_field_duotone']:
             grid_config = ((W + block_size - 1) // block_size, (H + block_size - 1) // block_size)
        else:
             threads_per_block = block_config[0] * block_config[1]
             total_pixels = H * W
             num_blocks = (total_pixels + threads_per_block - 1) // threads_per_block
             grid_config = (num_blocks, 1) 

        # 7. Preparar argumentos para process_gpu
        process_gpu_args = {
            'image': image, 
            'kernel': kernel, 
            'block_config': block_config, 
            'grid_config': grid_config
        }
        
        # --- AÑADIR ARGUMENTOS ESPECÍFICOS ---
        if filter_name in ['sticker_overlay', 'sticker']:
            
            # Sticker Principal Path
            sticker_path_default = filter_instance.get_parameters()['sticker_img_path']['default']
            sticker_path = request.form.get('sticker_img_path', sticker_path_default)
            process_gpu_args['sticker_img_path'] = sticker_path
            
            # Footer Sticker Path
            footer_path_default = filter_instance.get_parameters()['footer_img_path']['default']
            footer_path = request.form.get('footer_img_path', footer_path_default)
            process_gpu_args['footer_img_path'] = footer_path

        if filter_name in ['depth_of_field', 'depth_of_field_duotone']:
            # Añadir coordenadas de ROI (Región de Interés)
            
            # Nota: Si el usuario no envía valores, usamos el default (que es 0)
            # El filtro se encarga de calcular el 50% central si todos son 0.
            roi_x_start = int(request.form.get('roi_x_start', filter_instance.get_parameters()['roi_x_start']['default']))
            roi_y_start = int(request.form.get('roi_y_start', filter_instance.get_parameters()['roi_y_start']['default']))
            roi_x_end = int(request.form.get('roi_x_end', filter_instance.get_parameters()['roi_x_end']['default']))
            roi_y_end = int(request.form.get('roi_y_end', filter_instance.get_parameters()['roi_y_end']['default']))
            
            process_gpu_args['roi_coords'] = (roi_x_start, roi_y_start, roi_x_end, roi_y_end)


        
        # Llamada al proceso 
        output, gpu_time = filter_instance.process_gpu(**process_gpu_args)

        
        result = {
            "success": True,
            "processor": "GPU",
            "filter": filter_name,
            "image_size": {"width": W, "height": H, "channels": output.shape[-1] if len(output.shape)>2 else 1},
            "kernel_size": kernel_size,
            "processing_time_ms": round(gpu_time, 3)
        }
        
        return_base64 = request.form.get('return_base64', 'false').lower() == 'true'

        if return_base64:
            img_buffer = array_to_image_bytes(output)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            result["image_base64"] = img_base64
            return jsonify(result)
        else:
            img_buffer = array_to_image_bytes(output)
            return send_file(
                img_buffer,
                mimetype='image/png',
                as_attachment=True,
                download_name=f'{filter_name}_gpu_{kernel_size}x{kernel_size}.png'
            )

    except Exception as e:
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500
    finally:
        ctx.pop()


@app.route('/process/compare', methods=['POST'])
def process_compare():
    ctx.push()
    try:
        # Nota: Esta ruta necesitaría la misma lógica condicional de carga de imagen para ser completa.
        return jsonify({"error": "La ruta /compare no está totalmente implementada para filtros de color/ROI."}), 501
    finally:
        ctx.pop()


if __name__ == '__main__':
    print("\n" + "="*70)
    print("INICIANDO SERVIDOR")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)