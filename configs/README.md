# configs/

Contient la logique de chargement et de transformation de la configuration JSON en variables d'environnement et en configuration Ollama.

Fichiers importants :
- `config.py` : lit le fichier JSON fourni (via `--config`) et :
  - définit les variables d'environnement préfixées `BENCH_*` (ex. `BENCH_PUE`, `BENCH_USERS`, `BENCH_MODELS`, `BENCH_GPU_0_NAME`, ...),
  - génère `configs/config.toml` (déclaration des instances Ollama) et lance `scripts/ollama-batch-servers.sh`.

Structure attendue du JSON : voir la section "JSON Configuration Schema" dans le README racine.

Conseils :
- Modifiez `config.py` si vous souhaitez ajouter de nouvelles variables d'environnement ou changer la façon dont sont mappés les GPU aux instances Ollama.
- Testez avec un petit fichier JSON (ex. `test.json`) avant de lancer de larges benchmarks.