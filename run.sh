#!/usr/bin/env bash

# --- CONFIGURATION ---
IMAGE_NAME="sim-llm-benchmark-cuda"
CONTAINER_WORKDIR="/workspace"

echo "Démarrage de la configuration de l'environnement..."
chmod +x scripts/*.sh
chmod +x measure/scripts/*.sh


# 1. Construction de l'image Docker
echo "Construction de l'image Docker (cela peut prendre quelques minutes)..."
docker build -t $IMAGE_NAME .

if [ $? -ne 0 ]; then
    echo "Erreur lors du build. Vérifiez votre Dockerfile."
    exit 1
fi

# 2. Lancement du conteneur avec toutes les options
echo "Lancement du conteneur avec accès aux GPUs et SSH..."
mkdir -p /tmp/ollama_host_storage
mkdir -p "$(pwd)/save_data"
mkdir -p "$(pwd)/hf_cache"
docker run --device nvidia.com/gpu=all -it --rm \
  --shm-size=16gb \
  --net=host \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v "$(pwd)":$CONTAINER_WORKDIR \
  -v /tmp/ollama_host_storage:/tmp/ollama \
  -v "$(pwd)/hf_cache:/root/.cache/huggingface" \
  $IMAGE_NAME
