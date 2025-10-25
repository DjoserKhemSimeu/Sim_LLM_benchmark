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

# Vérifie si tegrastats est installé
TEGRASTATS_INSTALLED=false
if command -v tegrastats &>/dev/null; then
  TEGRASTATS_INSTALLED=true
fi

# Détecte le nombre total de GPU disponibles
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# Fonction pour mesurer la puissance avec tegrastats
measure_with_tegrastats() {
  local GPU_ID=$1
  local LOG_FILE="$LOG_DIR/consommation_energie_gpu_${GPU_ID}_${NB_USER}_${MODEL}.csv"
  local OUTFILE="/tmp/tegrastats_out_${GPU_ID}.txt"
  local PID_FILE="$PID_DIR/nv_measure_gpu_${GPU_ID}.pid"
  pkill -f "tegrastats" || true
  rm -f "$OUTFILE"

  echo "timestamp,gpu_power" >"$LOG_FILE"

  # Lancer tegrastats en arrière-plan
  tegrastats --interval 50 --logfile "$OUTFILE" >/dev/null 2>&1 &

  # Attendre que le fichier commence à être rempli
  while [ ! -s "$OUTFILE" ]; do
    sleep 0.1
  done

  start_time=$(date +%s.%3N)

  # Lancer la boucle de mesure en arrière-plan
  (
    tail -f "$OUTFILE" | while read -r line; do
      gpu_power=$(echo "$line" | grep -oP 'VDD_GPU_SOC \K\d+')
      current_time=$(date +%s.%3N)
      elapsed_time=$(echo "$current_time - $start_time" | bc)
      if [ -n "$gpu_power" ]; then
        echo "$elapsed_time,$gpu_power" >>"$LOG_FILE"
      fi
    done
  ) &
  LOOP_PID=$!
  echo $LOOP_PID >"$PID_FILE"
  echo "Mesure de puissance GPU ${GPU_ID} (tegrastats) lancée avec PID $LOOP_PID"
}

# Fonction pour mesurer la puissance avec nvidia-smi
measure_with_nvidia_smi() {
  local GPU_ID=$1
  local LOG_FILE="$LOG_DIR/consommation_energie_gpu_${GPU_ID}_${NB_USER}_${MODEL}.csv"
  local PID_FILE="$PID_DIR/nv_measure_gpu_${GPU_ID}.pid"

  echo "timestamp,gpu_power" >"$LOG_FILE"

  # Lancer la boucle de mesure en arrière-plan
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
  echo $! >"$PID_FILE"
  echo "Mesure de puissance GPU ${GPU_ID} (nvidia-smi) lancée avec PID $(cat "$PID_FILE")"
}

# Lancer une boucle pour chaque GPU
for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
  if $TEGRASTATS_INSTALLED; then
    measure_with_tegrastats "$GPU_ID"
    export BENCH_JETSON=1
  else
    export BENCH_JETSON=0
    measure_with_nvidia_smi "$GPU_ID"
  fi
done
