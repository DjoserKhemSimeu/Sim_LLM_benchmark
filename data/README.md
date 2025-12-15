# data/

Jeux de données utilisés par les scripts d'analyse et d'estimation.

Fichiers clés présents :
- `mem_density.csv` : densités mémoire / paramètres utilisés pour estimer certaines métriques.
- `more-than-carbon-data.csv` : jeu de données détaillant l'empreinte des composants (utilisé par `bar_impact_mtc.py`).
- `prompts.csv` : exemples de prompts/testcases.
- `g5k/` : données spécifiques à Grid5000 ou machines de référence.

Format attendu :
- CSV comma-separated standard; les scripts s'attendent à des en-têtes clairs (consulter les scripts de traitement pour les noms de colonnes exacts).

Ajout de données :
- Pour ajouter de nouveaux jeux, placez-les ici et mettez à jour les scripts/fonctions qui les consomment.
