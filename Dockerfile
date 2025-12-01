# Imagen base con CUDA y toolchain
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Instalar Python + herramientas de compilación (para pycuda)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-dev python3-pip python3-venv \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiamos los requirements primero (para aprovechar cache)
COPY requirements.txt .

# Instalamos dependencias de Python
RUN pip3 install --no-cache-dir -r requirements.txt

# Copiamos TODO el backend (app, filtros, etc.)
COPY . .

# Variables útiles
ENV PYTHONUNBUFFERED=1

# Puerto donde escucha Flask
EXPOSE 5000

# Comando de arranque
CMD ["python3", "app_production.py"]
