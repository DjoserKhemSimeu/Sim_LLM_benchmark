#!/bin/bash
PID_DIR="logs/pids"

echo "Arrêt des mesures de puissance..."

# 1. Tuer les boucles nommées (très important)
pkill -9 -f "measure_loop_gpu" || true

# 2. Tuer les utilitaires système
pkill -9 -f "tegrastats" || true
pkill -9 -f "nvidia-smi --query-gpu" || true

# 3. Nettoyer les fichiers PID
rm -f "$PID_DIR"/nv_measure_gpu_*.pid

echo "Nettoyage terminé."