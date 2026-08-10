#!/bin/bash

# Dossier contenant tous les sous-dossiers de tes agents
BASE_DIR="agent_env"

# On s'assure que l'environnement virtuel est activé (si tu l'utilises)
# source venv/bin/activate

# On boucle sur tous les dossiers qui commencent par "agent_env_user_"
for AGENT_DIR in "$BASE_DIR"/agent_env_user_*; do
    
    # On vérifie que c'est bien un dossier
    if [ -d "$AGENT_DIR" ]; then
        echo "======================================================"
        echo " Démarrage de l'évaluation pour : $AGENT_DIR"
        
        # On vérifie si l'agent a bien généré ses prédictions
        if [ -f "$AGENT_DIR/preds.json" ]; then
            
            # On récupère juste le nom du dossier (ex: agent_env_user_qwen3.6:27b_1_0_0)
            # Ça servira d'identifiant unique pour le run_id
            DIR_NAME=$(basename "$AGENT_DIR")
            
            # 1. On lance SWE-bench en pointant vers le preds.json de ce dossier
            python -m swebench.harness.run_evaluation \
                --dataset_name princeton-nlp/SWE-bench_Lite \
                --split test \
                --predictions_path "$AGENT_DIR/preds.json" \
                --run_id "$DIR_NAME" \
                --max_workers 4
                
            # 2. SWE-bench génère un fichier du type "modele.RUN_ID.json" à la racine.
            # On le déplace directement dans le dossier de l'agent.
            echo "Déplacement du rapport vers $AGENT_DIR/"
            mv *"$DIR_NAME".json "$AGENT_DIR/" 2>/dev/null
            
            echo "Évaluation terminée pour $DIR_NAME"
            
        else
            echo "Ignoré : Aucun fichier preds.json trouvé dans $AGENT_DIR"
        fi
    fi
done

echo "======================================================"
echo "Toutes les évaluations sont terminées et rangées !"