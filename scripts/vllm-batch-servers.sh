#!/bin/bash

# Constants
HOST="0.0.0.0"
BASE_PORT=53100
LOG_DIR="vllm-server-logs"
SLEEP_INTERVAL=1 # Réduit à 1s juste pour décaler légèrement les lancements

# Check if the number of GPUs is provided as an argument
if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <num_gpus> <model>"
  echo "Example: $0 4 mistralai/Mistral-7B-Instruct-v0.2"
  exit 1
fi

# Command-line argument
NUM_GPUS=$1
MODEL=$2

# Validate that NUM_GPUS is a positive integer
if ! [[ "$NUM_GPUS" =~ ^[0-9]+$ ]] || [[ "$NUM_GPUS" -le 0 ]]; then
  echo "Error: <num_gpus> must be a positive integer."
  exit 1
fi

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Pre-download the model
echo "Checking/pre-downloading model $MODEL via Hugging Face..."
hf download "$MODEL"

# Tableaux pour stocker les informations de chaque instance
declare -a PIDS
declare -a PORTS
declare -a LOG_FILES

# ==========================================
# PHASE 1 : LANCEMENT PARALLÈLE
# ==========================================
echo "=========================================================="
echo "Lancement de $NUM_GPUS instances vLLM en parallèle..."
echo "=========================================================="

for ((i = 0; i < NUM_GPUS; i++)); do
  PORT=$((BASE_PORT + i))
  SAFE_MODEL_NAME="${MODEL//\//_}"
  LOG_FILE="${LOG_DIR}/${PORT}_${SAFE_MODEL_NAME}.log"

  export CUDA_VISIBLE_DEVICES="$i"

  echo "Démarrage initié pour le GPU $i (Port ${PORT})..."

  # Lancement en arrière-plan
  nohup vllm serve "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --gpu-memory-utilization 0.95 \
    > "$LOG_FILE" 2>&1 &
    
  # Enregistrement des données dans les tableaux
  PIDS[$i]=$!
  PORTS[$i]=$PORT
  LOG_FILES[$i]=$LOG_FILE

  sleep "$SLEEP_INTERVAL"
done

# ==========================================
# PHASE 2 : MONITORING ET ATTENTE
# ==========================================
echo -e "En attente de l'initialisation de tous les serveurs..."

# Optionnel : Afficher les logs entremêlés de tous les serveurs
tail -f "${LOG_FILES[@]}" &
TAIL_PID=$!

# Boucle jusqu'à ce que tous soient prêts
ALL_READY=false
while [ "$ALL_READY" = false ]; do
  ALL_READY=true

  for ((i = 0; i < NUM_GPUS; i++)); do
    PID=${PIDS[$i]}
    PORT=${PORTS[$i]}

    # 1. Vérification des crashs
    if ! ps -p $PID > /dev/null; then
      echo -e "ERREUR CRITIQUE : L'instance vLLM sur le GPU $i (PID $PID) a crashé."
      kill $TAIL_PID 2>/dev/null
      
      # Optionnel : tuer les autres instances survivantes pour nettoyer
      for p in "${PIDS[@]}"; do kill -9 $p 2>/dev/null; done
      exit 1
    fi

    # 2. Vérification de la santé
    # S'il y a au moins un serveur qui ne répond pas 200 OK, on n'est pas encore prêt
    if ! curl -s -f "http://localhost:$PORT/health" > /dev/null 2>&1; then
      ALL_READY=false
    fi
  done

  # Si tout n'est pas prêt, on attend avant de revérifier
  if [ "$ALL_READY" = false ]; do
    sleep 3
  fi
done

# Nettoyage de l'affichage des logs
kill $TAIL_PID 2>/dev/null

echo "=========================================================="
echo "Toutes les instances vLLM ont démarré et sont PRÊTES !"
echo "Les logs complets sont dans le dossier $LOG_DIR."
echo "=========================================================="