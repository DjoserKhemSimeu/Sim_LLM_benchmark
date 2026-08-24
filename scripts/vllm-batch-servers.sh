#!/bin/bash

# Constants
HOST="0.0.0.0"
BASE_PORT=53100
LOG_DIR="vllm-server-logs"
SLEEP_INTERVAL=1

if [[ $# -lt 2 ]] || [[ $# -gt 3 ]]; then
  echo "Usage: $0 <num_gpus> <model_and_quant> [tokenizer]"
  exit 1
fi

NUM_GPUS=$1
MODEL=$2
TOKENIZER=$3

if ! [[ "$NUM_GPUS" =~ ^[0-9]+$ ]] || [[ "$NUM_GPUS" -le 0 ]]; then
  echo "Error: <num_gpus> must be a positive integer."
  exit 1
fi

mkdir -p "$LOG_DIR"

declare -a PIDS
declare -a PORTS
declare -a LOG_FILES

echo "=========================================================="
echo "Lancement de $NUM_GPUS instances vLLM en parallele..."
echo "=========================================================="

for ((i = 0; i < NUM_GPUS; i++)); do
  PORT=$((BASE_PORT + i))
  # Remplacement des caracteres speciaux pour le nom du fichier log
  SAFE_MODEL_NAME="${MODEL//[:\/]/_}"
  LOG_FILE="${LOG_DIR}/${PORT}_${SAFE_MODEL_NAME}.log"

  export CUDA_VISIBLE_DEVICES="$i"

  LOWER_MODEL="${MODEL,,}"
  if [[ "$LOWER_MODEL" == *"mistral"* ]] || [[ "$LOWER_MODEL" == *"mixtral"* ]] || [[ "$LOWER_MODEL" == *"ministral"* ]]; then
    PARSER="mistral"
  elif [[ "$LOWER_MODEL" == *"llama-3"* ]] || [[ "$LOWER_MODEL" == *"llama3"* ]]; then
    PARSER="llama3_json"
  elif [[ "$LOWER_MODEL" == *"qwen2"* ]]; then
    PARSER="hermes"
  elif [[ "$LOWER_MODEL" == *"qwen3"* ]]; then
    PARSER="qwen3_coder"
  else
    PARSER="hermes"
  fi

  if [ -n "$TOKENIZER" ]; then
    TOKENIZER_ARG="--tokenizer $TOKENIZER"
    echo "Demarrage GPU $i | Modele : $MODEL | Tokenizer : $TOKENIZER"
  else
    TOKENIZER_ARG=""
    echo "Demarrage GPU $i | Modele : $MODEL"
  fi

  # Lancement avec le format repo_id:quant_type natif
  nohup vllm serve "$MODEL" \
    $TOKENIZER_ARG \
    --host "$HOST" \
    --port "$PORT" \
    --enable-auto-tool-choice \
    --tool-call-parser "$PARSER" \
    --gpu-memory-utilization 0.95 \
    --max-model-len 16384 \
    --enforce-eager \
    --enable-chunked-prefill \
    > "$LOG_FILE" 2>&1 &
    
  PIDS[$i]=$!
  PORTS[$i]=$PORT
  LOG_FILES[$i]=$LOG_FILE

  sleep "$SLEEP_INTERVAL"
done

# ==========================================
# PHASE 2 : MONITORING ET ATTENTE
# ==========================================
echo -e "En attente de l'initialisation de tous les serveurs..."

tail -f "${LOG_FILES[@]}" &
TAIL_PID=$!

ALL_READY=false
while [ "$ALL_READY" = false ]; do
  ALL_READY=true

  for ((i = 0; i < NUM_GPUS; i++)); do
    PID=${PIDS[$i]}
    PORT=${PORTS[$i]}

    if ! ps -p $PID > /dev/null; then
      echo -e "ERREUR CRITIQUE : L'instance vLLM sur le GPU $i a crashe."
      kill $TAIL_PID 2>/dev/null
      for p in "${PIDS[@]}"; do kill -9 $p 2>/dev/null; done
      exit 1
    fi

    if ! curl -s -f "http://localhost:$PORT/health" > /dev/null 2>&1; then
      ALL_READY=false
    fi
  done

  if [ "$ALL_READY" = false ]; then
    sleep 3
  fi
done

kill $TAIL_PID 2>/dev/null

echo "=========================================================="
echo "Toutes les instances vLLM ont demarre et sont PRETES !"
echo "=========================================================="