#!/bin/bash

# Dossier contenant tous les sous-dossiers de tes agents
BASE_DIR="agent_env"

# On s'assure que l'environnement virtuel est activé (si tu l'utilises)
# source venv/bin/activate

# On boucle sur tous les dossiers qui commencent par "agent_env_user_"
for AGENT_DIR in "$BASE_DIR"/agent_env_user_*; do
    
    if [ -d "$AGENT_DIR" ]; then
        echo "======================================================"
        echo "Démarrage de l'évaluation pour : $AGENT_DIR"
        
        if [ -f "$AGENT_DIR/preds.json" ]; then
            
            # On compte les fichiers .json qui ne s'appellent pas "preds.json"
            EVAL_FILES_COUNT=$(find "$AGENT_DIR" -maxdepth 1 -type f -name "*.json" ! -name "preds.json" 2>/dev/null | wc -l)
            
            if [ "$EVAL_FILES_COUNT" -gt 0 ]; then
                echo "Ignoré : Un rapport d'évaluation existe déjà dans ce dossier."
            else
                # On récupère le nom du dossier d'origine
                DIR_NAME=$(basename "$AGENT_DIR")
                
                # On remplace les ':' par des '_' pour que Docker accepte le nom
                SAFE_RUN_ID=$(echo "$DIR_NAME" | tr ':' '_')
                
                # 1. On lance SWE-bench avec le SAFE_RUN_ID
                python -m swebench.harness.run_evaluation \
                    --dataset_name princeton-nlp/SWE-bench_Lite \
                    --split test \
                    --predictions_path "$AGENT_DIR/preds.json" \
                    --run_id "$SAFE_RUN_ID" \
                    --max_workers 4
                    
                # 2. On déplace le rapport généré (qui porte désormais le nom sécurisé)
                echo "Déplacement du rapport vers $AGENT_DIR/"
                mv *"$SAFE_RUN_ID".json "$AGENT_DIR/" 2>/dev/null
                
                echo "Évaluation terminée pour $DIR_NAME"
            fi
            
        else
            echo "Ignoré : Aucun fichier preds.json trouvé dans $AGENT_DIR"
        fi
    fi
done

echo "======================================================"
echo "Toutes les évaluations sont terminées."