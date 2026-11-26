# Imagen CUDA lista para RunPod Pod (HTTP puerto 8000) o RunPod Serverless
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
# PyTorch ya viene instalado con CUDA en la imagen base; instalamos el resto + CLIP para YOLOE
RUN grep -vE "^torch(vision)?==" requirements.txt > /tmp/req_no_torch.txt && \
    pip install --no-cache-dir -r /tmp/req_no_torch.txt && \
    pip install --no-cache-dir "git+https://github.com/ultralytics/CLIP.git"

COPY diagnostico_entorno.py pipeline.py main.py runpod_handler.py start_runpod.sh /app/
RUN chmod +x /app/start_runpod.sh && mkdir -p /workspace/models

EXPOSE 8000

# Por defecto inicia el servidor FastAPI (RunPod Pod).
# Para RunPod Serverless cambiar el comando del contenedor a: python -u runpod_handler.py
CMD ["/app/start_runpod.sh"]
