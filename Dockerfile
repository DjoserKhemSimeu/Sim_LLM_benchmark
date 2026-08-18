FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# 1. Installation des outils (fusionné avec docker.io pour réduire les couches)
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    python3-venv \
    git \
    wget \
    curl \
    zstd \
    bc \
    ca-certificates \
    openssh-client \
    docker.io \
    && rm -rf /var/lib/apt/lists/*

# 2. Configuration de l'environnement virtuel Python
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 3. Installation d'Ollama
RUN curl -fsSL https://ollama.com/install.sh | OLLAMA_INSTALL_DIR=/tmp sh

# 4. Configuration SSH pour les agents
RUN mkdir -p /root/.ssh && \
    echo "Host *\n\tStrictHostKeyChecking no\n\tUserKnownHostsFile /dev/null" > /root/.ssh/config

WORKDIR /workspace

# 5. Installation des dépendances Python
# On copie d'abord UNIQUEMENT le requirements.txt. 
# Ainsi, le cache Docker n'est pas invalidé si vous modifiez juste votre code (main.py)
RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir vllm mini-swe-agent
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Désactivation des traces crewai (maintenant que Python/crewai est installé)
RUN crewai traces disable

# 6. Copie du reste du code source
COPY . .

ENV OLLAMA_MODELS=/tmp/ollama

# La commande CMD doit être simple. 
CMD ["python3", "main.py","--config","test.json"]
