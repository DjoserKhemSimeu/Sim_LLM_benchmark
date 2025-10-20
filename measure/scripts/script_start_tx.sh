#!/bin/bash
if [ $# -ne 2 ]; then
  echo "Usage: $0 <nb_users> <model>"
  exit 1
fi

LOG_DIR="/tmp/save_data"
PID_DIR="/tmp"
NB_USER=$1
MODEL=$2
mkdir -p "$LOG_DIR"

# Détecte le nombre total de GPU disponibles
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# Lancer une boucle pour chaque GPU
for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
  LOG_FILE="${LOG_DIR}/consommation_energie_gpu_${GPU_ID}_${NB_USER}_${MODEL}.csv"
  PID_FILE="${PID_DIR}/nv_measure_gpu_${GPU_ID}.pid"

  # En-tête du fichier CSV
  echo "timestamp,gpu_power" >"$LOG_FILE"

  # Boucle de mesure pour ce GPU en tâche de fondx
  (
    START_TIME=$(date +%s.%N)
    while true; do
      CURRENT_TIME=$(date +%s.%N)
      ELAPSED=$(echo "$CURRENT_TIME - $START_TIME" | bc)
      POWER=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits -i "$GPU_ID")
      echo "${ELAPSED},${POWER}" >>"$LOG_FILE"
      sleep 0.01
    done
  ) &

  # Rediriger le PID en utilisant sudo si nécessaire
  echo $! | tee "$PID_FILE" >/dev/null

  echo "Mesure de puissance GPU ${GPU_ID} lancée avec PID $(cat "$PID_FILE")"
done
