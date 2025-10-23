#!/bin/bash
PID_DIR="/tmp"
PID_PATTERN="nv_measure_gpu_*.pid"
FOUND=0

# Tuer tous les processus de mesure, même sans fichier PID
pkill -f "tegrastats --interval" || true
pkill -f "nvidia-smi --query-gpu" || true

# Puis nettoyer les fichiers PID
for PID_FILE in "$PID_DIR"/$PID_PATTERN; do
  [ -e "$PID_FILE" ] || continue
  echo "Nettoyage du fichier PID : $PID_FILE"
  rm -f "$PID_FILE"
  FOUND=1
done

if [ "$FOUND" -eq 0 ]; then
  echo "Aucun fichier PID trouvé."
fi
