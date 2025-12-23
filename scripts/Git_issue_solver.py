import argparse
import json
import os
import subprocess
import threading
import requests
from pathlib import Path
from typing import Optional, Type, List, Any, Dict
import re
from bs4 import BeautifulSoup
import duckduckgo_search
from pydantic import BaseModel, Field

# CrewAI
from crewai import Agent, Task, Process, Crew, LLM
from crewai.tools import BaseTool

# --- CONFIGURATION & ARGUMENTS ---
def parse_args():
    p = argparse.ArgumentParser(description='Run Git issue solver agent')
    p.add_argument('--user-id', type=int, default=None)
    p.add_argument('--host', type=str, default="http://localhost:11432")
    p.add_argument('--n_users', type=int, default=1)
    p.add_argument('--iter', type=int, default=0)
    return p.parse_args()

args = parse_args()
ID = int(args.user_id) if args.user_id is not None else 0
HOST = args.host
ITER = args.iter
NB_USER = args.n_users
MODEL = os.environ.get("BENCH_MODEL", "mistral-nemo")

# --- GESTION DES CHEMINS ABSOLUS ---
ABS_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ABS_ROOT / "logs" / "parsed"
LOG_FILE = LOG_DIR / f"results_{MODEL.replace(':', '-')}.jsonl"
log_lock = threading.Lock()

class BenchmarkedLLM(LLM):
    """Surcharge de LLM pour capturer les métriques et l'outil appelé."""
    
    def call(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        response = super().call(messages, **kwargs)
        output_text = str(response)
        prompt_text = messages[-1]['content'] if messages else ""
        
        # Extraction de l'outil appelé via Regex
        # On cherche des patterns types : "Action: name" ou {"action": "name"}
        tool_called = self._extract_tool_name(output_text)
        
        # Estimation du nombre de tokens
        nb_tokens = len(output_text.split()) * 1.3 

        log_entry = {
            "prompt": prompt_text,
            "output": output_text,
            "tool_called": tool_called,
            "nb_output_token": int(nb_tokens),
            "user_id": ID,
            "nb_user": NB_USER,
            "model": MODEL,
            "iter": ITER,
            "host": HOST,
            "agent": "issue-fixer"
        }

        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with log_lock:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        return response

    def _extract_tool_name(self, text: str) -> Optional[str]:
        """Analyse le texte pour trouver l'outil que l'agent a décidé d'utiliser."""
        # Pattern 1: Format ReAct standard "Action: [tool_name]"
        action_match = re.search(r"Action:\s*(\w+)", text, re.IGNORECASE)
        if action_match:
            return action_match.group(1).strip()
        
        # Pattern 2: Format JSON souvent utilisé par les modèles récents
        try:
            # On cherche une structure JSON dans le texte
            json_match = re.search(r"\{.*\"action\":\s*\"(\w+)\".*\}", text, re.DOTALL)
            if json_match:
                return json_match.group(1).strip()
        except:
            pass
            
        return None
# --- INITIALISATION ENVIRONNEMENT ---
agent_env_path = os.path.join('agent_env', f'agent_env_user_{MODEL}_{NB_USER}_{ID}_{ITER}')
os.makedirs(agent_env_path, exist_ok=True)
os.chdir(agent_env_path)

# --- DÉFINITION DES OUTILS (VERSION INTÉGRALE) ---

class WebSearchInput(BaseModel):
    query: str = Field(..., description="The web search query")

class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = "Performs a free web search via DuckDuckGo and returns URLs and excerpts of results."
    args_schema: Type[BaseModel] = WebSearchInput
    def _run(self, query: str) -> str:
        try:
            with duckduckgo_search.DDGS() as ddgs:
                results = ddgs.text(query, max_results=5)
            if not results: return "No results."
            output = []
            for r in results:
                entry = f"Title: {r.get('title')}\nURL: {r.get('href')}\nExcerpt: {r.get('body')}\n"
                output.append(entry)
            return "\n---\n".join(output)
        except Exception as e: return f"Error: {e}"

class CloneRepoInput(BaseModel):
    repo_url: str = Field(..., description="SSH of the GitHub repository to clone")

class CloneRepoTool(BaseTool):
    name: str = "git_clone"
    description: str = "Clone a GitHub repository locally (uses the `git` CLI) and change into the cloned directory."
    args_schema: Type[BaseModel] = CloneRepoInput
    def _run(self, repo_url: str) -> str:
        try:
            subprocess.run(["git", "clone", repo_url], check=True)
            folder = repo_url.split('/')[-1].replace('.git','')
            os.chdir(folder)
            return f"Cloned to: {folder} and jumped into it"
        except Exception as e: return f"Clone error: {e}"

class ReadFileTool(BaseTool):
    name: str = "read_file"
    description: str = "Reads a file in the cloned repository and returns its content."
    def _run(self, path: str) -> str:
        try: return Path(path).read_text(encoding='utf-8')
        except Exception as e: return f"File read error: {e}"

class WriteFileTool(BaseTool):
    name: str = "write_file"
    description: str = "Writes content to a file in the cloned repository (creates directories if needed)."
    def _run(self, path: str, content: str) -> str:
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding='utf-8')
            return f"Wrote {p}"
        except Exception as e: return f"File write error: {e}"

class RunTestsTool(BaseTool):
    name: str = "run_tests"
    description: str = "Runs the test suite via `pytest` if present in the repository."
    def _run(self, pytest_args: Optional[str] = None) -> str:
        try:
            cmd = ["pytest", "-q"] + (pytest_args.split() if pytest_args else [])
            p = subprocess.run(cmd, capture_output=True, text=True)
            return json.dumps({"rc": p.returncode, "stdout": p.stdout, "stderr": p.stderr})
        except Exception as e: return f"Error running tests: {e}"

class GitCommitTool(BaseTool):
    name: str = "git_commit"
    description: str = "Stage and commit changes in the local repository."
    def _run(self, message: str = "update") -> str:
        try:
            subprocess.run(["git", "add", "-A"], check=True)
            subprocess.run(["git", "commit", "-m", message], check=True)
            return "Committed"
        except Exception as e: return f"Git commit error: {e}"

class GitPushTool(BaseTool):
    name: str = "git_push"
    description: str = "Pushes the local branch to the 'origin' remote."
    def _run(self, force: bool = False) -> str:
        try:
            branch = f'test_{MODEL.replace(":", "-")}_{NB_USER}_{ID}'
            subprocess.run(["git", "checkout", "-b", branch], check=True)
            cmd = ["git", "push", "-u", "origin", branch]
            if force: cmd.insert(2, "--force")
            subprocess.run(cmd, check=True)
            return f"Pushed to {branch}"
        except Exception as e: return f"Git push error: {e}"

class FetchIssueTool(BaseTool):
    name: str = "fetch_issue"
    description: str = "Fetches issues from the GitHub API."
    def _run(self, owner: str = "DjoserKhemSimeu", repo: str = "dummy_agent", issue_number: int = 1) -> str:
        try:
            url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}"
            resp = requests.get(url)
            return resp.json().get("body", "")
        except Exception as e: return f"Fetch issue error: {e}"

# --- INSTANCIATION AGENTS ET CREW (VERSION INTÉGRALE) ---

llm = BenchmarkedLLM(model=f"ollama/{MODEL}", base_url=HOST, temperature=0.0)

agent1 = Agent(
    role="issue-fixer",
    goal=(
        f"In the repository git@github.com:DjoserKhemSimeu/dummy_agent.git, resolve the GitHub issue number 1 "
        f"locally and propose PRs. Context: run by user_id={ID}. Local environment 'dummy_agent', owner='DjoserKhemSimeu', repo='dummy_agent'."
        f"$$$RUN_ID: USER_ID={ID} NB_USER={NB_USER} MODEL={MODEL} ITER={ITER} HOST={HOST}$$$"
    ),
    backstory=f"ISSUE-FIXER:Autonomous agent to diagnose, propose, and apply fixes on GitHub repositories. (invoked by user {ID})",
    verbose=True,
    memory=True,
    tools=[CloneRepoTool(), ReadFileTool(), WriteFileTool(), RunTestsTool(), GitCommitTool(), GitPushTool(), FetchIssueTool()],
    allow_delegation=False,
    llm=llm,
)

task1 = Task(
    description=(
        "TASK-PLANNER:Fix the provided issue: clone the repo, diagnose, propose a patch, and create a PR if tests pass. The update pipeline should be the following:\n"
        "1. Clone the GitHub repository locally and jump into it.\n"
        "2. Create a new branch for the changes.\n"
        "3. Analyze the issue and the code to identify the root of the issue.\n"
        "4. Make the necessary changes to the source code.\n"
        "5. Run the test suite to validate the changes.\n"
        "6. If tests pass, commit the changes and push the branch to the remote repository.\n"
        f"$$$RUN_ID: USER_ID={ID} NB_USER={NB_USER} MODEL={MODEL} ITER={ITER} HOST={HOST}$$$"
    ),
    expected_output="Report of the actions taken, including the URL of the created branch pushed.",
    agent=agent1,
)

crew = Crew(
    agents=[agent1],
    model=f"ollama/{MODEL}",
    tasks=[task1],
    process=Process.sequential,
    verbose=True,
    planning=True,
    planning_llm=llm
)

if __name__ == "__main__":
    print(f"Starting User {ID} Benchmark...")
    result = crew.kickoff()
    print(f"\n--- Final result User {ID} ---\n{result}")