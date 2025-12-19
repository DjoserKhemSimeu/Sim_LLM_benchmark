#!/usr/bin/env python3
"""
Parse an Ollama server log and extract prompts and decoded token outputs.

Usage:
    python3 scripts/parse_ollama_log.py --input ollama-server-logs/53100_mistral:7b.log --output out.jsonl

The script groups decoded "string" lines following a prompt until the next prompt,
and writes one JSON object per request with fields: prompt, output, start_line, end_line,
and additional metadata (`nb_output_token`, `user_id`, `nb_user`, `model`, `iter`, `host`, `agent`, `tool_called`).
"""
from __future__ import annotations

import argparse
import json
import re
from typing import List, Optional
import os

# ---------------------------------------------------------------------------
# REGEX
# ---------------------------------------------------------------------------

PROMPT_RE = re.compile(
    r'level=TRACE\b.*?msg=completion request.*?prompt\s*=\s*"(?P<prompt>[^"]*)"'
)

# ROBUST decoded token matcher (Ollama / sentencepiece safe)
DECODED_RE = re.compile(
    r'msg=decoded\b.*?string\s*=\s*(?:"(?P<d1>[^"]*)"|\'(?P<d2>[^\']*)\'|(?P<d3>\S+))'
)
# RUN_ID block: $$$RUN_ID: USER_ID={ID} NB_USER={NB_USER} MODEL={MODEL} ITER={ITER} HOST={HOST}$$$
RUN_ID_RE = re.compile(r'\$\$\$RUN_ID:\s*(?P<body>.*?)\$\$\$', re.DOTALL)


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _unescape(s: str) -> str:
    """Best-effort unicode escape decoding (safe)."""
    try:
        return bytes(s, "utf-8").decode("unicode_escape")
    except Exception:
        return s


def _extract_run_id_from_prompt(prompt: str) -> dict:
    res = {"user_id": None, "nb_user": None, "model": None, "iter": None, "host": None}
    if not prompt:
        return res

    m = RUN_ID_RE.search(prompt)
    if not m:
        return res

    body = m.group("body")
    kv = {
        'USER_ID': r'USER_ID=(?P<val>[^\s]+)',
        'NB_USER': r'NB_USER=(?P<val>[^\s]+)',
        'MODEL': r'MODEL=(?P<val>[^\s]+)',
        'ITER': r'ITER=(?P<val>[^\s]+)',
        'HOST': r'HOST=(?P<val>[^\s]+)'
    }

    for k, patt in kv.items():
        mm = re.search(patt, body)
        if not mm:
            continue

        v = mm.group('val')
        try:
            if k == 'USER_ID':
                res['user_id'] = int(v)
            elif k == 'NB_USER':
                res['nb_user'] = int(v)
            elif k == 'ITER':
                res['iter'] = int(v)
            elif k == 'MODEL':
                res['model'] = v
            elif k == 'HOST':
                res['host'] = v
        except Exception:
            res[k.lower()] = v

    return res


def _detect_agent_from_prompt(prompt: str) -> Optional[str]:
    if not prompt:
        return None
    if 'ISSUE-FIXER:' in prompt:
        return 'ISSUE-FIXER'
    if 'TASK-PLANNER:' in prompt:
        return 'TASK-PLANNER'
    
    return None


def _detect_tool_from_tokens(tokens: List[str]) -> Optional[str]:
    if not tokens:
        return None

    raw = [str(t) for t in tokens if t is not None]
    action_idx = None

    for i, tok in enumerate(raw):
        if tok.strip().lower().startswith('action'):
            action_idx = i
            break

    if action_idx is None:
        return None

    joined = ''.join(s.strip().lower() for s in raw[action_idx + 1:])

    if 'git' in joined and 'clone' in joined:
        return 'git_clone'
    if 'git' in joined and 'commit' in joined:
        return 'git_commit'
    if 'git' in joined and 'push' in joined:
        return 'git_push'
    if 'git' in joined and 'branch' in joined:
        return 'git_branch'
    if 'create' in joined and 'pr' in joined:
        return 'create_pr'
    if 'pytest' in joined or ('run' in joined and 'tests' in joined):
        return 'run_tests'
    if 'write' in joined and 'file' in joined:
        return 'write_file'
    if 'read' in joined and 'file' in joined:
        return 'read_file'
    if 'repo' in joined and 'tree' in joined:
        return 'repo_tree'
    if 'fetch' in joined and 'issue' in joined:
        return 'fetch_issue'
    if 'web' in joined and 'search' in joined:
        return 'web_search'

    return None


# ---------------------------------------------------------------------------
# CORE PARSER
# ---------------------------------------------------------------------------

def parse_log(input_path: str) -> List[dict]:
    results: List[dict] = []
    current: Optional[dict] = None

    with open(input_path, "r", encoding="utf-8", errors="replace") as f:
        for lineno, raw_line in enumerate(f, start=1):
            line = raw_line.rstrip("\n")

            # ---------------------------------------------------------------
            # PROMPT DETECTION (primary)
            # ---------------------------------------------------------------
            if ("msg=\"completion request\"" in line) or ("msg=completion request" in line):
                ppos = line.find("prompt=")
                prompt_text = ""

                if ppos != -1:
                    rest = line[ppos + len("prompt="):]

                    if rest.startswith('"'):
                        buf = []
                        esc = False
                        for ch in rest[1:]:
                            if esc:
                                buf.append(ch)
                                esc = False
                            elif ch == "\\":
                                esc = True
                            elif ch == '"':
                                break
                            else:
                                buf.append(ch)
                        prompt_text = _unescape("".join(buf))
                    else:
                        prompt_text = rest.split()[0]

                if prompt_text.isdigit():
                    continue

                if current is not None:
                    _finalize_current(current, lineno - 1, results)

                current = {
                    "prompt": prompt_text,
                    "decoded_parts": [],
                    "output": "",
                    "start_line": lineno,
                    "end_line": lineno,
                }
                continue

            # ---------------------------------------------------------------
            # PROMPT DETECTION (regex fallback)
            # ---------------------------------------------------------------
            m = PROMPT_RE.search(line)
            if m:
                if current is not None:
                    _finalize_current(current, lineno - 1, results)

                current = {
                    "prompt": _unescape(m.group("prompt")),
                    "decoded_parts": [],
                    "output": "",
                    "start_line": lineno,
                    "end_line": lineno,
                }
                continue

            # ---------------------------------------------------------------
            # DECODED TOKENS
            # ---------------------------------------------------------------
            if current is None or "string=" not in line:
                continue

            m2 = DECODED_RE.search(line)
            if not m2:
                continue

            dec = (
                m2.group('d1')
                or m2.group('d2')
                or m2.group('d3')
            )

            if dec is None:
                continue

            dec = _unescape(dec)
            dec = dec.replace('\x00', '').replace('\r', '')
            current["decoded_parts"].append(dec)
            current["end_line"] = lineno

    if current is not None:
        _finalize_current(current, current["end_line"], results)

    return results


def _finalize_current(current: dict, end_line: int, results: List[dict]) -> None:
    current["output"] = "\n".join(current.get("decoded_parts", []))
    current["nb_output_token"] = len(current.get("decoded_parts", []))

    rid = _extract_run_id_from_prompt(current.get("prompt", ""))
    current.update(rid)

    current["agent"] = _detect_agent_from_prompt(current.get("prompt", ""))
    current["tool_called"] = _detect_tool_from_tokens(current.get("decoded_parts", []))
    current["end_line"] = end_line

    results.append(current)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Parse Ollama log for prompts and decoded outputs")
    p.add_argument("--input", required=True, help="Path to ollama log file")
    p.add_argument("--output", required=True, help="Path to output JSONL file")
    args = p.parse_args()

    items = parse_log(args.input)
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as out:
        for item in items:
            out.write(json.dumps(item, ensure_ascii=False) + "\n\n")

    print(f"Wrote {len(items)} records to {args.output}")

    pretty_path = args.output.replace(".jsonl", ".pretty.json")
    try:
        with open(pretty_path, "w", encoding="utf-8") as pf:
            json.dump(items, pf, ensure_ascii=False, indent=2)
        print(f"Wrote pretty JSON to {pretty_path}")
    except Exception as e:
        print(f"Failed to write pretty JSON: {e}")


if __name__ == "__main__":
    main()
