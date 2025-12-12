#!/usr/bin/env python3
"""
Parse an Ollama server log and extract prompts and decoded token outputs.

Usage:
    python3 scripts/parse_ollama_log.py --input ollama-server-logs/53100_gemma3:4b.log --output out.jsonl

The script looks for lines matching:
  level =TRACE msg=completion request prompt ="..."
and decoded token lines like:
  level = TRACE source=sentencepiece.go msg=decoded ids=[...] string "..."

It groups decoded "string" lines following a prompt until the next prompt, concatenates them,
and writes one JSON object per request with fields: prompt, output, start_line, end_line.
"""
from __future__ import annotations

import argparse
import json
import re
from typing import List, Optional

PROMPT_RE = re.compile(r'level=TRACE*?msg=completion request prompt\s*=\s*"(?P<prompt>(?:[^"\\\\]|\\\\.)*)"')
DECODED_RE = re.compile(r'string="(?P<decoded>(?:[^"\\\\]|\\\\.)*)"')


def _unescape(s: str) -> str:
    # Convert escaped sequences like \n, \" into real characters
    try:
        return bytes(s, "utf-8").decode("unicode_escape")
    except Exception:
        return s


def parse_log(input_path: str) -> List[dict]:
    results = []
    current = None  # type: Optional[dict]

    with open(input_path, "r", encoding="utf-8", errors="replace") as f:
        for lineno, raw_line in enumerate(f, start=1):
            line = raw_line.rstrip("\n")

            # Detect prompt lines: look for msg="completion request" (with or without quotes)
            if ("msg=\"completion request\"" in line) or ("msg=completion request" in line):
                # attempt to extract prompt="..." value
                ppos = line.find("prompt=")
                prompt_text = ""
                if ppos != -1:
                    rest = line[ppos + len("prompt="):]
                    # if starts with a quote, read until closing quote (allow escaped)
                    if rest.startswith('"'):
                        buf = []
                        esc = False
                        for ch in rest[1:]:
                            if esc:
                                buf.append(ch)
                                esc = False
                            elif ch == "\\":
                                buf.append(ch)
                                esc = True
                            elif ch == '"':
                                break
                            else:
                                buf.append(ch)
                        prompt_text = _unescape("".join(buf))
                    else:
                        # unquoted prompt — take until space
                        prompt_text = rest.split()[0]

                # If prompt_text is a numeric token count (e.g. prompt=12746), ignore it
                if prompt_text.isdigit():
                    # do not start a new record for numeric-only prompts
                    continue

                # flush previous
                if current is not None:
                    # join decoded token pieces with newlines for readability
                    current["output"] = "\n".join(current.get("decoded_parts", []))
                    current["end_line"] = lineno - 1
                    results.append(current)

                current = {
                    "prompt": prompt_text,
                    "decoded_parts": [],
                    "output": "",
                    "start_line": lineno,
                    "end_line": lineno,
                }
                continue

            # Detect decoded token lines from sentencepiece
            if ("source=sentencepiece.go" in line) and ("msg=decoded" in line) and ("string=" in line):
                
                if current is None:
                    continue
                spos = line.find("string=")
                
                rest = line[spos + len("string="):].lstrip()
                
                decoded_val = ""
                if not rest:
                    continue
                if rest[0] == '"':
                    # quoted
                    buf = []
                    esc = False
                    for ch in rest[1:]:
                        if esc:
                            buf.append(ch)
                            esc = False
                        elif ch == "\\":
                            buf.append(ch)
                            esc = True
                        elif ch == '"':
                            break
                        else:
                            buf.append(ch)
                    
                    decoded_val = _unescape("".join(buf))
            
                else:
                    # unquoted token: take until whitespace
                    decoded_val = rest.split()[0]

                if decoded_val:
                    current["decoded_parts"].append(decoded_val)
                    current["end_line"] = lineno

            m = PROMPT_RE.search(line)
            if m:
                # flush previous
                if current is not None:
                    # join decoded pieces to single string
                    # join decoded token pieces with newlines for readability
                    current["output"] =current.get("decoded_parts", [])

                    current["end_line"] = lineno - 1
                    results.append(current)
                prompt_text = _unescape(m.group("prompt"))
                current = {
                    "prompt": prompt_text,
                    "decoded_parts": [],
                    "output": "",
                    "start_line": lineno,
                    "end_line": lineno,
                }
                continue

            m2 = DECODED_RE.search(line)
            if m2 and current is not None:
                dec = _unescape(m2.group("decoded"))
                current["decoded_parts"].append(dec)
                current["end_line"] = lineno
            
    # flush last
    if current is not None:
        current["output"] = current.get("decoded_parts", [])
        print(current["output"])
        results.append(current)
    
    # cleanup: remove decoded_parts
    
    
    
    return results


def main() -> None:
    p = argparse.ArgumentParser(description="Parse Ollama log for prompts and decoded outputs")
    p.add_argument("--input", required=True, help="Path to ollama log file")
    p.add_argument("--output", required=True, help="Path to output JSONL file")
    args = p.parse_args()

    items = parse_log(args.input)

    with open(args.output, "w", encoding="utf-8") as out:
        for item in items:
            # write each JSON object and add an extra blank line for readability
            out.write(json.dumps(item, ensure_ascii=False) + "\n\n")
    print(f"Wrote {len(items)} records to {args.output}")

    # Also write a pretty-printed JSON array for easier reading
    pretty_path = args.output
    if pretty_path.endswith('.jsonl'):
        pretty_path = pretty_path[:-6] + '.json'
    else:
        pretty_path = args.output + '.pretty.json'

    try:
        with open(pretty_path, 'w', encoding='utf-8') as pf:
            json.dump(items, pf, ensure_ascii=False, indent=2)
        print(f"Wrote pretty JSON to {pretty_path}")
    except Exception as e:
        print(f"Failed to write pretty JSON: {e}")
    


if __name__ == "__main__":
    main()
