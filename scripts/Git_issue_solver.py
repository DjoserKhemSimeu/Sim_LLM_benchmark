import argparse
import json
import os
from random import random
import subprocess
import threading
import requests
from pathlib import Path
from typing import Optional, Type, List, Any, Dict
import re
from ollama import Client
import litellm
from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI
import duckduckgo_search
from pydantic import BaseModel, Field
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_ollama import ChatOllama
from crewai.llm import LLM
import time
# CrewAI
from crewai import Agent, Task, Process, Crew, LLM
from crewai.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from litellm.utils import token_counter

# --- CONFIGURATION & ARGUMENTS ---
def parse_args():
    p = argparse.ArgumentParser(description='Run Git issue solver agent')
    p.add_argument('--user-id', type=int, default=None)
    p.add_argument('--host', type=str, default="http://localhost:11434")
    p.add_argument('--n_users', type=int, default=1)
    p.add_argument('--iter', type=int, default=0)
    return p.parse_args()

args = parse_args()
ID = int(args.user_id) if args.user_id is not None else 0
HOST = args.host
ITER = args.iter
NB_USER = args.n_users
MODEL = os.environ.get("BENCH_MODEL", "gemma3:4b")
GIT_SSH = os.environ.get("BENCH_GIT_SSH", "git@github.com:DjoserKhemSimeu/dummy_agent.git")
OWNER = os.environ.get("BENCH_OWNER", "DjoserKhemSimeu")
REPO_NAME = os.environ.get("BENCH_REPO_NAME", "dummy_matrix_agent")
temperature = float(os.environ.get("BENCH_TEMPERATURE", "0.0"))
topK = int(os.environ.get("BENCH_TOPK", "5"))
topP = float(os.environ.get("BENCH_TOPP", "0.9"))
gpu_model = os.environ.get("BENCH_GPU_MODEL", "Unknown")
gpu_fp32_tflops = float(os.environ.get("BENCH_FP32_TFLOPS", "0"))
gpu_memory_gib = float(os.environ.get("BENCH_GPU_MEMORY_GIB", "0"))
gpu_num = int(os.environ.get("BENCH_NUM_GPU", "1"))
#chanegr

# --- GESTION DES CHEMINS ABSOLUS ---
ABS_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ABS_ROOT / "logs" / "parsed"
#LOG_FILE = LOG_DIR / f"results_{MODEL.replace(':', '-')}.jsonl"
combo_tag = os.environ.get("BENCH_COMBO_TAG", "default")
LOG_FILE = LOG_DIR / f"results_{MODEL.replace(':', '-')}_{combo_tag}.jsonl"
log_lock = threading.Lock()


BASE = HOST

def parse_parameter_size(param_str):
    if not param_str:
        return None

    s = str(param_str).strip().upper()
    match = re.match(r"([0-9]*\.?[0-9]+)\s*([BM])", s)
    if not match:
        return None

    value = float(match.group(1))
    unit = match.group(2)

    return int(value * 1e9) if unit == "B" else int(value * 1e6)

def get_ollama_model_param_dict():
    r = requests.get(f"{BASE}/api/tags")
    r.raise_for_status()
    data = r.json()

    return {
        (m.get("name") or m.get("model")):
        parse_parameter_size(m.get("details", {}).get("parameter_size"))
        for m in data.get("models", [])
    }
            
MODEL_PARAM_DICT = get_ollama_model_param_dict()
    

class BenchmarkCallback(BaseCallbackHandler):
    def __init__(self):
        self.nb_input_token = 0
        self.nb_output_token = 0
        self.eval_duration_ns = 0
        self.tool_called = None

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.nb_input_token = 0
        self.nb_output_token = 0
        self.tool_called = None

    def on_llm_end(self, response, **kwargs):
        try:
            # Récupération des stats réelles Ollama via LangChain
            gen_info = response.generations[0][0].generation_info
            self.nb_input_token = gen_info.get("prompt_eval_count", 0)
            self.nb_output_token = gen_info.get("eval_count", 0)
            self.eval_duration_ns = gen_info.get("eval_duration") or 1
        except: pass

    def on_tool_start(self, serialized, input_str, **kwargs):
        # Capture le nom de l'outil dès qu'il est activé
        self.tool_called = serialized.get("name")

class BenchmarkedLLM_3(LLM):
    def __init__(self, model, base_url, **kwargs):
        super().__init__(model=model, base_url=base_url, **kwargs)


    def call(self, messages, **kwargs):
        

        # 2. Appel au LLM et capture du temps
        start_time = time.time()
        
        # CrewAI LLM.call() renvoie le texte, mais nous avons besoin des stats.
        # On utilise litellm via la méthode parente qui peuple souvent les metadata
        output_text = super().call(messages, **kwargs)
        
        duration = time.time() - start_time

        # 3. Extraction des tokens via LiteLLM
        # Dans CrewAI, les stats sont souvent stockées dans l'instance après l'appel
        # ou accessibles via l'objet de réponse si on appelait litellm.completion
        
        # Solution robuste pour CrewAI : On récupère les stats de l'usage global si disponibles
        # Sinon, on estime via une heuristique (ou on accède aux attributs internes)
        try:
            # Note: LiteLLM stocke parfois les stats dans des variables de classe ou de callback
            # Ici on utilise une approche de secours fiable si les compteurs sont à 0
            
            prompt_tokens = token_counter(model=self.model, messages=messages)
            completion_tokens = token_counter(model=self.model, text=output_text)
        except:
            prompt_tokens = 0
            completion_tokens = 0

        # 4. Calcul du TPS (vitesse)
        tps = completion_tokens / duration if duration > 0 else 0
        
        # 5. Format de sortie EXACT
        prompt_text = messages[-1]["content"] if messages else ""
        agent_label = "ISSUE-FIXER" if "issue-fixer" in prompt_text.lower() else "TASK-PLANNER"
        tool_called = self._extract_tool_name(output_text)

        
        model_num_params = MODEL_PARAM_DICT.get(MODEL)        

        log_entry = {
            "prompt": prompt_text,
            "output": output_text,
            "tool_called": tool_called,
            "speed_tps": round(tps, 2),
            "user_id": ID,
            "iter": ITER,
            "host": HOST,
            "agent": agent_label,
            #will use for knn
            "model": MODEL,
            "nb_input_token": prompt_tokens,
            "temperature": temperature,
            "top_p": topP,
            "top_k": topK,
            "nb_user": NB_USER,
            "gpu_model": gpu_model,
            "gpu_fp32_tflops": gpu_fp32_tflops,
            "gpu_memory_gib": gpu_memory_gib,
            "gpu_num": gpu_num,
            "model_num_params": model_num_params,
            "nb_output_token": completion_tokens, 
            "inference_time" : duration  #the target 
        }

        # 6. Logging
        with log_lock:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        print(f"User {ID} - {agent_label} - Tokens: {completion_tokens} Out / {prompt_tokens} In - {tps:.2f} tps")
        
        return output_text

    def _extract_tool_name(self, text: str) -> Optional[str]:
        if not text: return None
        action_match = re.search(r"Action:\s*(\w+)", text, re.IGNORECASE)
        if action_match:
            return action_match.group(1).strip()
        try:
            data = json.loads(text)
            return data.get('action') or data.get('tool')
        except:
            pass
        return None


# --- INITIALISATION ENVIRONNEMENT ---
agent_env_path = os.path.join('agent_env', f'agent_env_user_{MODEL}_{NB_USER}_{ID}_{ITER}')
os.makedirs(agent_env_path, exist_ok=True)
os.chdir(agent_env_path)

# --- DÉFINITION DES OUTILS (VERSION INTÉGRALE) ---


class CloneRepoInput(BaseModel):
    repo_url: str = Field(..., description="SSH of the GitHub repository to clone")
   

class CloneRepoTool(BaseTool):
    name: str = "git_clone"
    description: str = "Clone a GitHub repository locally (uses the `git` CLI) and change into the cloned directory."
    args_schema: Type[BaseModel] =CloneRepoInput

    def _run(self, repo_url: str) -> str:
        if repo_url.startswith("http://") or repo_url.startswith("https://"):
            return "Please use the SSH URL for cloning"
        try:
          
            # run git clone
            subprocess.run(["git", "clone", repo_url], check=True)
            os.chdir(repo_url.split('/')[-1].replace('.git',''))
            return f"Cloned to: {repo_url.split('/')[-1].replace('.git','')} and jumped into it"
        except Exception as e:
            return f"Clone error: {e}"


class ReadFileInput(BaseModel):
    path: str = Field(..., description="Relative path in the repo to the file to read, if you already used git clone you don't need to put the name of the repo")
    #repo_dir: Optional[str] = Field(None, description="Chemin du repo local")


class ReadFileTool(BaseTool):
    name: str = "read_file"
    description: str = "Reads a file in the cloned repository and returns its content."
    args_schema: Type[BaseModel] = ReadFileInput

    def _run(self, path: str) -> str:
        try:
            base = Path('.')
            p = base / path
            return p.read_text(encoding='utf-8')
        except Exception as e:
            return f"File read error: {e}"

class EditFileInput(BaseModel):
    path: str = Field(..., description="Chemin relatif du fichier à modifier")
    old_content: str = Field(..., description="Le texte exact à rechercher dans le fichier")
    new_content: str = Field(..., description="Le nouveau texte qui remplacera l'ancien")

class EditFileTool(BaseTool):
    name: str = "edit_file"
    description: str = "Remplace un bloc de texte spécifique par un autre dans un fichier existant."
    args_schema: Type[BaseModel] = EditFileInput

    def _run(self, path: str, old_content: str, new_content: str, repo_dir: Optional[str] = None) -> str:
        try:
            base = Path(repo_dir) if repo_dir else Path('.')
            p = base / path
            
            if not p.exists():
                return f"Erreur : Le fichier {path} n'existe pas."

            # Lecture du contenu actuel
            content = p.read_text(encoding='utf-8')

            if old_content not in content:
                return "Erreur : Le texte à remplacer n'a pas été trouvé exactement tel quel dans le fichier."

            # Remplacement
            new_full_content = content.replace(old_content, new_content)
            
            # Écriture
            p.write_text(new_full_content, encoding='utf-8')
            return f"Fichier {path} mis à jour avec succès."
            
        except Exception as e:
            return f"Erreur lors de la modification : {e}"
class WriteFileInput(BaseModel):
    path: str = Field(..., description="Relative path in the repo to the file to write")
    content: str = Field(..., description="Content to write")
    


class WriteFileTool(BaseTool):
    name: str = "write_file"
    description: str = "Writes content to a file in the cloned repository (creates directories if needed)."
    args_schema: Type[BaseModel] = WriteFileInput

    def _run(self, path: str, content: str, repo_dir: Optional[str] = None) -> str:
        try:
            base = Path(repo_dir) if repo_dir else Path('.')
            p = base / path
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding='utf-8')
            return f"Wrote {p}"
        except Exception as e:
            return f"File write error: {e}"


class RunTestsInput(BaseModel):
    pytest_args: Optional[str] = Field(None, description="Additional arguments for pytest")


class RunTestsTool(BaseTool):
    name: str = "run_tests"
    description: str = "Runs the test suite via `pytest` if present in the repository."
    args_schema: Type[BaseModel] = RunTestsInput

    def _run(self, repo_dir: Optional[str] = None, pytest_args: Optional[str] = None) -> str:
        try:
            cwd = repo_dir or '.'
            cmd = ["pytest", "-q"]
            if pytest_args:
                cmd += pytest_args.split()
            p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
            result = {
                        "rc": p.returncode,
                        "stdout": p.stdout,
                        "stderr": p.stderr
                    }
            output_path = os.path.join(cwd, "pytest_results.json")

            with open(output_path, "w") as f:
                json.dump(result, f, indent=4)

            return json.dumps(result)
        except FileNotFoundError:
            return json.dumps({"rc": 0, "stdout": "pytest not found; skipped", "stderr": ""})
        except Exception as e:
            return f"Error running tests: {e}"


class GitCommitInput(BaseModel):
    message: str = Field(...)


class GitCommitTool(BaseTool):
    name: str = "git_commit"
    description: str = "Stage and commit changes in the local repository." 
    args_schema: Type[BaseModel] = GitCommitInput

    def _run(self, repo_dir: Optional[str] = None, message: str = "update") -> str:
        try:
            cwd = repo_dir or '.'
            subprocess.run(["git", "add", "-A"], cwd=cwd, check=True)
            subprocess.run(["git", "commit", "-m", message], cwd=cwd, check=True)
            return "Committed"
        except subprocess.CalledProcessError as e:
            return f"Git commit error: {e}"
        except Exception as e:
            return f"Git commit error: {e}"


class GitPushInput(BaseModel):
    force: Optional[bool] = Field(False, description="Push with --force if necessary")


class GitPushTool(BaseTool):
    name: str = "git_push"
    description: str = "Pushes the local branch to the 'origin' remote (optionally injecting the token for auth)."
    args_schema: Type[BaseModel]  = GitPushInput

    def _run(self, repo_dir: Optional[str] = None, owner: Optional[str] = None, repo: Optional[str] = None, token_env: Optional[str] = 'GITHUB_TOKEN', force: Optional[bool] = False) -> str:
        try:
            branch=f'test_{MODEL.replace(":", "-")}_{args.n_users}_{ID}'
            cwd = repo_dir or '.'
            # Ensure branch exists locally
            check = subprocess.run(["git", "rev-parse", "--verify", branch], cwd=cwd, capture_output=True, text=True)
            if check.returncode != 0:
                # create branch from current HEAD
                subprocess.run(["git", "checkout", "-b", branch], cwd=cwd, check=True)
            else:
                subprocess.run(["git", "checkout", branch], cwd=cwd, check=True)

            push_cmd = ["git", "push", "-u", "origin", branch]
            if force:
                push_cmd.insert(2, "--force")

            p = subprocess.run(push_cmd, cwd=cwd, capture_output=True, text=True)
            if p.returncode == 0:
                url = None
                if owner and repo:
                    url = f"https://github.com/{owner}/{repo}/tree/{branch}"
                return json.dumps({"status": "pushed", "stdout": p.stdout, "url": url})

            # If push failed and token is available, try setting origin to tokened https URL
            token = os.environ.get(token_env or 'GITHUB_TOKEN')
            if token and owner and repo:
                tokened = f"https://{token}@github.com/{owner}/{repo}.git"
                # Save current origin URL
                cur = subprocess.run(["git", "remote", "get-url", "origin"], cwd=cwd, capture_output=True, text=True)
                old_url = cur.stdout.strip() if cur.returncode == 0 else None
                subprocess.run(["git", "remote", "set-url", "origin", tokened], cwd=cwd, check=True)
                try:
                    p2 = subprocess.run(push_cmd, cwd=cwd, capture_output=True, text=True)
                    if p2.returncode == 0:
                        # restore origin to non-tokened URL if we can
                        if old_url:
                            subprocess.run(["git", "remote", "set-url", "origin", old_url], cwd=cwd, check=True)
                        url = f"https://github.com/{owner}/{repo}/tree/{branch}"
                        return json.dumps({"status": "pushed", "stdout": p2.stdout, "url": url})
                    return f"Push failed after token attempt: {p2.returncode} stdout={p2.stdout} stderr={p2.stderr}"
                finally:
                    # best-effort restore
                    if old_url:
                        subprocess.run(["git", "remote", "set-url", "origin", old_url], cwd=cwd)

            return f"Push failed: {p.returncode} stdout={p.stdout} stderr={p.stderr}"
        except Exception as e:
            return f"Git push error: {e}"


class CreateBranchInput(BaseModel):
    branch: str = Field(..., description="Name of the branch to create")
    start_point: Optional[str] = Field(None, description="Start commit/branch (e.g., main)")
    checkout: Optional[bool] = Field(True, description="If True, switch to the new branch after creation")


class CreateBranchTool(BaseTool):
    name: str = "git_create_branch"
    description: str = "Create a new local branch and (optionally) checkout to it."
    args_schema: Type[BaseModel] = CreateBranchInput

    def _run(self, branch: str, repo_dir: Optional[str] = None, start_point: Optional[str] = None, checkout: Optional[bool] = True) -> str:
        try:
            branch = f"{branch}_{ID}"
            cwd = repo_dir or '.'
            # check if branch already exists
            check = subprocess.run(["git", "rev-parse", "--verify", branch], cwd=cwd, capture_output=True, text=True)
            if check.returncode == 0:
                return f"Branch '{branch}' already exists"

            cmd = ["git", "checkout", "-b", branch]
            if start_point:
                cmd.append(start_point)
            subprocess.run(cmd, cwd=cwd, check=True)

            if not checkout:
                # if user doesn't want to checkout, create branch then checkout back to previous
                # get previous branch
                prev = subprocess.run(["git", "rev-parse", "--abbrev-ref", "@{-1}"], cwd=cwd, capture_output=True, text=True)
                prev_branch = prev.stdout.strip() if prev.returncode == 0 else None
                if prev_branch:
                    subprocess.run(["git", "checkout", prev_branch], cwd=cwd, check=True)

            return f"Branch '{branch}' created"
        except subprocess.CalledProcessError as e:
            return f"Git error creating branch: {e}"
        except Exception as e:
            return f"Create branch error: {e}"


class CreatePRInput(BaseModel):
    owner: str = Field(...)
    repo: str = Field(...)
    head_branch: str = Field(...)
    base_branch: Optional[str] = Field(None)
    title: Optional[str] = Field(None)
    body: Optional[str] = Field(None)


class CreatePRTool(BaseTool):
    name: str = "create_pr"
    description: str = "Create a Pull Request on GitHub using the GITHUB_TOKEN environment variable." 
    args_schema: Type[BaseModel] = CreatePRInput

    def _run(self, owner: str, repo: str, head_branch: str, base_branch: Optional[str] = None, title: Optional[str] = None, body: Optional[str] = None) -> str:
        try:
            token = os.environ.get('GITHUB_TOKEN')
            if not token:
                return "GITHUB_TOKEN not set"
            # Use REST API to create PR
            api = f"https://api.github.com/repos/{owner}/{repo}/pulls"
            head = head_branch
            base = base_branch or 'main'
            payload = {"title": title or f"Fix: {head_branch}", "head": head, "base": base, "body": body or ""}
            resp = requests.post(api, json=payload, headers={"Authorization": f"token {token}", "Accept": "application/vnd.github+json"})
            if resp.status_code >= 200 and resp.status_code < 300:
                return resp.json().get('html_url', str(resp.json()))
            return f"PR error: {resp.status_code} {resp.text}"
        except Exception as e:
            return f"Create PR error: {e}"


class FetchIssueInput(BaseModel):
    owner: str = Field(..., description="GitHub owner of the repository")
    repo: str = Field(..., description="Name of the GitHub repository")
    issue_number: Optional[int] = Field(None, description="Issue number (1)")


class FetchIssueTool(BaseTool):
    name: str = "fetch_issue"
    description: str = "Fetches issues from the GitHub API. If the repository has only one issue, returns its JSON content 'body'."
    args_schema: Type[BaseModel] = FetchIssueInput

    def _run(self, owner: str , repo:str, issue_number: Optional[int] = 1) -> str:
        try:
            url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}"
            response = requests.get(url)
            response.raise_for_status()  # lève une erreur si la requête échoue
            data = response.json()
            return data.get("body", "")
        except Exception as e:
            return f"Fetch issue error: {e}"


class RepoTreeInput(BaseModel):
    owner: str = Field(...)
    repo: str = Field(...)
    path: Optional[str] = Field(None, description="Relative path to list (empty = root)")
    ref: Optional[str] = Field(None, description="Branch/sha (default: main/master)")


class RepoTreeTool(BaseTool):
    name: str = "repo_tree"
    description: str = "List the repository tree via the GitHub API and optionally read the content of a given file." 
    args_schema: Type[BaseModel] = RepoTreeInput

    def _run(self, owner: str, repo: str, path: Optional[str] = None, ref: Optional[str] = None) -> str:
        try:
            token = os.environ.get('GITHUB_TOKEN')
            headers = {"Accept": "application/vnd.github+json"}
            if token:
                headers["Authorization"] = f"token {token}"

            # determine branch/ref
            ref_param = ref or ''
            # use contents API
            api = f"https://api.github.com/repos/{owner}/{repo}/contents"
            if path:
                api = f"{api}/{path.lstrip('/') }"
            params = {}
            if ref_param:
                params['ref'] = ref_param
            resp = requests.get(api, headers=headers, params=params)
            if resp.status_code != 200:
                return f"GitHub contents API error: {resp.status_code} {resp.text}"
            data = resp.json()
            # If it's a list -> directory, else file
            if isinstance(data, list):
                files = [{"path": it.get('path'), "type": it.get('type')} for it in data]
                return json.dumps(files)
            else:
                # file object
                content = data.get('content')
                encoding = data.get('encoding')
                if content and encoding == 'base64':
                    import base64
                    decoded = base64.b64decode(content).decode('utf-8', errors='ignore')
                    return decoded
                return json.dumps(data)
        except Exception as e:
            return f"Repo tree error: {e}"

# --- INSTANCIATION AGENTS ET CREW (VERSION INTÉGRALE) ---

llm = BenchmarkedLLM_3(model=f"ollama/{MODEL}", base_url=HOST, temperature=temperature, top_k=topK, top_p=topP, seed=42, num_ctx=8192)

agent1 = Agent(
    role="issue-fixer",
    goal=(
        f"In the repository {GIT_SSH}, resolve the GitHub issue number 1 "
        f"locally and propose PRs. Context: run by user_id={ID}, owner='{OWNER}', repo='{REPO_NAME}'."
        f"$$$RUN_ID: USER_ID={ID} NB_USER={NB_USER} MODEL={MODEL} ITER={ITER} HOST={HOST}$$$"
    ),
    backstory=f"ISSUE-FIXER:Autonomous agent to diagnose, propose, and apply fixes on GitHub repositories. (invoked by user {ID})",
    verbose=True,
    memory=True,
    tools=[CloneRepoTool(), ReadFileTool(), EditFileTool(), RunTestsTool(), GitCommitTool(), GitPushTool(), CreatePRTool(), CreateBranchTool(), FetchIssueTool()],
    allow_delegation=False,
    llm=llm,
    reasoning=False
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
    #planning=False,
    #planning_llm=llm
)

if __name__ == "__main__":
    print(f"Starting User {ID} Benchmark...")
    result = crew.kickoff()
    print(f"\n--- Final result User {ID} ---\n{result}")