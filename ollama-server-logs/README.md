# ollama-server-logs/

Répertoire contenant les logs produits par les serveurs Ollama utilisés pendant les
benchmarks. Ces logs peuvent être analysés pour récupérer les `prompt` et les sorties
de modèle.

Format attendu :
- Logs textuels contenant des traces de requêtes et des lignes marquant des tokens décodés.

Outils :
- `scripts/parse_ollama_log.py` : script utilitaire (si présent) pour extraire des paires
  `prompt` / `output` et écrire des fichiers structurés (`logs/*.jsonl`, `logs/*.json`).

Notes :
- Les logs peuvent contenir des fragments de tokens (sentencepiece) listés ligne par ligne ;
  le parseur reconstruit la sortie en regroupant ces fragments.
- Si vous modifiez le format de log d'Ollama, mettez à jour `scripts/parse_ollama_log.py`.
