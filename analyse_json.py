import pandas as pd
import json

def load_jsonl(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return pd.DataFrame(data)

def analyze_single_run(df_run):
    #Number of interaction steps 
    interaction_steps = len(df_run)
    
    #Number of input/output tokens
    total_input_tokens = df_run['nb_input_token'].sum()
    total_output_tokens = df_run['nb_output_token'].sum()
    
    #Token processing speed (Moyenne TPS)
    mean_speed_tps = df_run['speed_tps'].mean()
    
    #Execution time (Estimation : somme de (tokens_produits / vitesse))
    df_run['step_time'] = df_run.apply(
        lambda row: row['nb_output_token'] / row['speed_tps'] if row['speed_tps'] > 0 else 0, 
        axis=1
    )
    total_execution_time = df_run['step_time'].sum()
    
    #Number of tool usages (N_tool) compte outils appelés
    tool_usages = df_run['tool_called'].notna().sum()
    
    #Agent transition occurrences : compte les changements d'agent d'une étape à l'autre
    if 'agent' in df_run.columns:
        agent_transitions = (df_run['agent'] != df_run['agent'].shift()).sum() - 1
        agent_transitions = max(0, agent_transitions) 
    else:
        agent_transitions = 0
        
    #Number of errors
    error_keywords = ['error', 'exception', 'failed', 'traceback', 'command not found']
    errors_count = df_run['output'].astype(str).str.lower().apply(
        lambda text: any(kw in text for kw in error_keywords)
    ).sum()
    

    #Tool usage frequency (f_tool = N_tool / L_tau)
    tool_usage_frequency = tool_usages / interaction_steps if interaction_steps > 0 else 0
    
    #matrice de corrélation entre les différentes métriques

def main():
    file_path = 'data/valentine_data.jsonl'
    df = load_jsonl(file_path)
    
    grouping_columns = ['model', 'user_id', 'iter']
    
    print("Analyse des métriques : ")
    results = df.groupby(grouping_columns).apply(analyze_single_run).reset_index()
    
    print("\nRésultats de l'analyse :")
    print(results.to_string(index=False))
    
    # CSV
    output_csv = 'analyse_benchmark.csv'
    results.to_csv(output_csv, index=False)
