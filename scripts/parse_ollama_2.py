#!/usr/bin/env python3
import argparse
import json
import re
from typing import List, Dict

# Regex adaptées aux logs réels observés
CONN_ID_RE = re.compile(r'ptr=(?P<id>0x[a-f0-9]+)')
# Utilisation de [^"]* pour capturer le contenu du prompt plus largement
PROMPT_RE = re.compile(r'prompt="(?P<p>.*?)"', re.DOTALL)
DECODED_RE = re.compile(r'msg=decoded.*?string="(?P<t>.*?)(?<!\\)"')
RUN_ID_RE = re.compile(r'\$\$\$RUN_ID:\s*(?P<body>.*?)\$\$\$', re.DOTALL)
METRICS_RE = re.compile(r'completion_tokens=(?P<ct>\d+)')

def _unescape(s: str) -> str:
    """Nettoyage des séquences d'échappement dans les logs."""
    try:
        return s.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t')
    except:
        return s

def _extract_run_id(prompt: str) -> dict:
    res = {"user_id": None, "nb_user": None, "model": None, "iter": None, "host": None}
    m = RUN_ID_RE.search(prompt)
    if not m: return res
    body = m.group("body")
    patterns = {'USER_ID': 'user_id', 'NB_USER': 'nb_user', 'MODEL': 'model', 'ITER': 'iter', 'HOST': 'host'}
    for label, key in patterns.items():
        match = re.search(fr'{label}=(?P<v>[^\s\$]+)', body)
        if match:
            val = match.group('v')
            res[key] = int(val) if val.isdigit() else val
    return res

def parse_log(input_path: str) -> List[dict]:
    results = []
    active_sessions: Dict[str, dict] = {}
    last_seen_ptr = None # Mémoire du dernier PTR vu pour lier les lignes orphelines

    print(f"--- Analyse de {input_path} ---")

    with open(input_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
        
        for i, line in enumerate(lines):
            lineno = i + 1
            
            # Mise à jour du PTR contextuel
            conn_match = CONN_ID_RE.search(line)
            if conn_match:
                last_seen_ptr = conn_match.group("id")

            if "msg=\"completion request\"" in line or "msg=completion request" in line:
                # Si le PTR n'est pas sur la ligne, on regarde la ligne juste avant
                current_ptr = last_seen_ptr
                
                # Capture du prompt avec une logique plus souple
                p_match = PROMPT_RE.search(line)
                if p_match and current_ptr:
                    prompt_raw = p_match.group("p")
                    prompt_clean = _unescape(prompt_raw)
                    
                    active_sessions[current_ptr] = {
                        "prompt": prompt_clean,
                        "decoded_parts": [],
                        "start_line": lineno,
                        **_extract_run_id(prompt_clean)
                    }
                continue

            # Accumulation des tokens
            if "msg=decoded" in line and last_seen_ptr in active_sessions:
                t_match = DECODED_RE.search(line)
                if t_match:
                    active_sessions[last_seen_ptr]["decoded_parts"].append(_unescape(t_match.group("t")))
                continue

            # Fin de session
            if "completion_tokens=" in line and last_seen_ptr in active_sessions:
                session = active_sessions.pop(last_seen_ptr)
                m_metrics = METRICS_RE.search(line)
                
                session["output"] = "".join(session["decoded_parts"])
                session["nb_output_token"] = int(m_metrics.group("ct")) if m_metrics else len(session["decoded_parts"])
                session["end_line"] = lineno
                
                if "decoded_parts" in session: del session["decoded_parts"]
                results.append(session)

    print(f"--- Terminé : {len(results)} requêtes extraites ---")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    
    data = parse_log(args.input)
    with open(args.output, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")