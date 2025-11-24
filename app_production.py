

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
print(f"   Compute Capability: {device.compute_capability()}")

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
        filter_instances[filter_name] = FilterFactory.create_filter(filter_name)
        print(f"   {filter_name}: Pre-compilado")
    except Exception as e:
        print(f"     {filter_name}: {str(e)[:80]}")

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

available_filters = FilterFactory.get_available_filters()
print(f"✅ Filtros cargados: {len(available_filters)}")
for f in available_filters:
    print(f"   • {f['name']}: {f['description']}")

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


def load_image_grayscale(image_data):
    img = Image.open(io.BytesIO(image_data)).convert('L')
    img_array = np.array(img, dtype=np.float32)
    return img_array, img.size


def array_to_image_bytes(img_array, format='PNG'):
    img_array = np.clip(img_array, 0, 255)
    img = Image.fromarray(img_array.astype(np.uint8))
    buffer = io.BytesIO()
    img.save(buffer, format=format)
    buffer.seek(0)
    return buffer



@app.route('/')
def index():
    return jsonify({
        "message": "🎨 Image Convolution Filter API with PyCUDA",
        "version": "2.0.0",
        "mode": "Production (no auto-reload)",
        "endpoints": {
            "/health": "GET - Health check",
            "/filters": "GET - List filters",
            "/process/cpu": "POST - Process with CPU",
            "/process/gpu": "POST - Process with GPU",
            "/process/compare": "POST - Compare CPU vs GPU",
            "/process/benchmark": "POST - Benchmark"
        },
        "gpu_info": {
            "free_memory": free_mem,
            "total_memory": total_mem
        },
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
            return jsonify({
                "error": "Filter parameter is required",
                "available_filters": list(filter_instances.keys())
            }), 400

        if filter_name not in filter_instances:
            return jsonify({
                "error": f"Filter '{filter_name}' not found",
                "available_filters": list(filter_instances.keys())
            }), 400

        filter_instance = filter_instances[filter_name]

        kernel_size = int(request.form.get('kernel_size',
                          filter_instance.get_parameters()['kernel_size']['default']))
        block_size = int(request.form.get('block_size', 16))
        return_base64 = request.form.get('return_base64', 'false').lower() == 'true'

        image_data = file.read()
        image, _ = load_image_grayscale(image_data)
        H, W = image.shape

        kernel = filter_instance.generate_kernel(kernel_size)

        block_config = (block_size, block_size, 1)
        grid_config = ((W + block_size - 1) // block_size, (H + block_size - 1) // block_size, 1)

        output, gpu_time = filter_instance.process_gpu(image, kernel, block_config, grid_config)

        result = {
            "success": True,
            "processor": "GPU",
            "filter": filter_name,
            "image_size": {"width": W, "height": H},
            "kernel_size": kernel_size,
            "block_config": {"x": block_size, "y": block_size},
            "grid_config": {"x": grid_config[0], "y": grid_config[1]},
            "processing_time_ms": round(gpu_time, 3)
        }

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
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file = request.files['image']
        filter_name = request.form.get('filter', '').lower()

        if not filter_name:
            return jsonify({"error": "Filter parameter required"}), 400

        if filter_name not in filter_instances:
            return jsonify({
                "error": f"Filter '{filter_name}' not found",
                "available_filters": list(filter_instances.keys())
            }), 400

        filter_instance = filter_instances[filter_name]
        kernel_size = int(request.form.get('kernel_size',
                          filter_instance.get_parameters()['kernel_size']['default']))

        image_data = file.read()
        image, _ = load_image_grayscale(image_data)
        H, W = image.shape

        kernel = filter_instance.generate_kernel(kernel_size)

        # GPU
        block_config = (16, 16, 1)
        grid_config = ((W + 15) // 16, (H + 15) // 16, 1)
        output_gpu, gpu_time = filter_instance.process_gpu(image, kernel, block_config, grid_config)

        speedup = cpu_time / gpu_time if gpu_time > 0 else 0

        return jsonify({
            "success": True,
            "filter": filter_name,
            "image_size": {"width": W, "height": H},
            "kernel_size": kernel_size,
            "cpu": {"processing_time_ms": round(cpu_time, 3)},
            "gpu": {
                "processing_time_ms": round(gpu_time, 3),
                "block_config": "16x16",
                "grid_config": f"{grid_config[0]}x{grid_config[1]}"
            },
            "speedup": round(speedup, 2),
            "performance_gain": f"{((speedup - 1) * 100):.1f}%" if speedup > 1 else "0%"
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500
    finally:
        ctx.pop()


if __name__ == '__main__':
    print("\n" + "="*70)
    print("INICIANDO SERVIDOR")

    # Sin reloader para evitar problemas con CUDA
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)