FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# 1. Installation des outils (SSH est requis pour utiliser la clé passée par l'hôte)
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
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
# Installation d'Ollama
RUN curl -fsSL https://ollama.com/install.sh | OLLAMA_INSTALL_DIR=/tmp sh
# 2. Configuration SSH pour les agents
# On désactive la vérification stricte pour que l'agent ne soit pas bloqué par
# une question "Are you sure you want to continue connecting (yes/no)?"
RUN mkdir -p /root/.ssh && \
    echo "Host *\n\tStrictHostKeyChecking no\n\tUserKnownHostsFile /dev/null" > /root/.ssh/config

WORKDIR /workspace

RUN pip3 install --upgrade pip
COPY . .
RUN pip3 install --no-cache-dir -r requirements.txt
RUN crewai traces disable

COPY . .
ENV OLLAMA_MODELS=/tmp/ollama
# La commande CMD doit être simple. 
# Le montage SSH se fait lors du "docker run"
CMD ["python3", "main.py","--config","test.json"]