# measure/

Scripts de calcul et visualisation des impacts environnementaux.

Fichiers importants :
- `scripts/bar_impact.py` : calcul de l'impact de fabrication (graphes et CSV).
- `scripts/bar_impact_mtc.py` : variante utilisant les données "more-than-carbon" pour décomposer l'impact par composant.
- `scripts/perf_show.py` et `scripts/perf_show_mtc.py` : génèrent des visualisations combinées des impacts (consommation électrique + fabrication).

Entrées et sorties :
- Entrées : CSV contenant données électriques, manufacturing datasets, et résultats bruts de benchmark.
- Sorties : CSV synthétiques et images PNG sauvegardées dans `images/`.

Dépendances : `pandas`, `numpy`, `matplotlib`, `seaborn`.

Usage rapide :
```
python measure/scripts/perf_show_mtc.py
```
