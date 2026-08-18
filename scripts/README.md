# scripts/

Scripts utilitaires et d'orchestration du benchmark.

Fichiers et utilités clés :
- `multi_gpu_bench.py` : script principal de lancement des benchmarks concurrents (usage multi-GPU).
- `ollama-batch-servers.sh` : démarre une instance Ollama par GPU/port configuré.
- `parse_ollama_log.py` (si présent) : parse les logs d'Ollama pour extraire `prompt` / `output`.
- `water_from_mix.py` : calcule la consommation d'eau associée à la production d'énergie depuis un CSV.

Notes d'utilisation :
- La plupart des scripts dépendent des variables d'environnement définies par `configs/config.py`.
- Lancer `python scripts/<script>.py --help` pour voir les options disponibles.
- Certains scripts nécessitent `pandas`, `numpy`, `matplotlib`.
