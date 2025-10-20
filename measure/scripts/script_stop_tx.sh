#!/bin/bash

PID_DIR="/tmp"
PID_PATTERN="nv_measure_gpu_*.pid"
FOUND=0

for PID_FILE in "$PID_DIR"/$PID_PATTERN; do
    # Vérifie si le fichier existe (évite les erreurs si aucun ne correspond)
    [ -e "$PID_FILE" ] || continue

    PID=$(cat "$PID_FILE")
    echo "Arrêt du processus de mesure avec PID $PID (fichier: $PID_FILE)"
    kill "$PID" 2>/dev/null && echo "Processus $PID arrêté." || echo "Échec de l'arrêt du processus $PID."
    rm -f "$PID_FILE"
    FOUND=1
done

if [ "$FOUND" -eq 0 ]; then
    echo "Aucun processus de mesure trouvé."
fi
