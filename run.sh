#!/bin/bash

# --- CONFIGURATION ---
IMAGE_NAME="mon-benchmark-cuda"
CONTAINER_WORKDIR="/workspace"

echo "🚀 Démarrage de la configuration de l'environnement..."

# 1. Gestion de l'agent SSH (pour le push/pull sans mot de passe)
if [ -z "$SSH_AUTH_SOCK" ]; then
    echo "🔑 Démarrage de l'agent SSH..."
    eval $(ssh-agent -s)
fi

# Vérifie si une clé est déjà chargée, sinon propose d'en ajouter une
ssh-add -l > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "🔓 Aucune clé SSH détectée. Veuillez ajouter votre clé (ex: ~/.ssh/id_rsa) :"
    ssh-add
fi

# 2. Construction de l'image Docker
echo "🛠️ Construction de l'image Docker (cela peut prendre quelques minutes)..."
docker build -t $IMAGE_NAME .

if [ $? -ne 0 ]; then
    echo "❌ Erreur lors du build. Vérifiez votre Dockerfile."
    exit 1
fi

# 3. Lancement du conteneur avec toutes les options
echo "🐳 Lancement du conteneur avec accès aux 4 GPUs et SSH..."
mkdir -p /tmp/ollama_host_storage
mkdir -p "$(pwd)/save_data"
docker run --gpus all -it --rm \
  --shm-size=16gb \
  -v $SSH_AUTH_SOCK:/ssh-agent \
  -e SSH_AUTH_SOCK=/ssh-agent \
  -v "$(pwd)":$CONTAINER_WORKDIR \
  -v /tmp/ollama_host_storage:/tmp/ollama \
  $IMAGE_NAME