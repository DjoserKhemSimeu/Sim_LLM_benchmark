from crewai import Agent, Task, Process, Crew, LLM
import argparse
from typing import Optional,Type, List
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import requests
from bs4 import BeautifulSoup
import duckduckgo_search
import subprocess
import os
import json
from pathlib import Path
ID=0
MODEL=os.environ.get("BENCH_MODEL", "ollama/mistral-nemo")
# CLI: accept --user-id so external processes (bench) can pass the client id
def parse_args():
    p = argparse.ArgumentParser(description='Run Git issue solver agent (optionally with user id)')
    p.add_argument('--user-id', type=int, default=None, help='Optional user id to associate with this run')
    p.add_argument('--host', type=str, default="http://localhost:11432", help='Host where the LLM server is running')
    return p.parse_args()
class WebSearchInput(BaseModel):
    query: str = Field(..., description="La requête de recherche web")

class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = "Effectue une recherche web gratuite via DuckDuckGo et retourne les URLs + extraits des résultats."
    args_schema: Type[BaseModel] = WebSearchInput

    def _run(self, query: str) -> str:
        try:
            with duckduckgo_search.DDGS() as ddgs:
                results = ddgs.text(query, max_results=5)
            if not results:
                return "Pas de résultats."

            output: List[str] = []
            for r in results:
                url = r.get("href")
                title = r.get("title")
                snippet = r.get("body") or ""
                # On essaie de récupérer un peu de contenu réel
                try:
                    resp = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
                    soup = BeautifulSoup(resp.text, "html.parser")
                    # Extraire les 2-3 premiers paragraphes
                    paras = soup.find_all("p")
                    text = " ".join(p.get_text() for p in paras[:3])
                    excerpt = text if text else snippet
                except Exception:
                    excerpt = snippet

                entry = f"Title: {title}\nURL: {url}\nExcerpt: {excerpt}\n"
                output.append(entry)
            return "\n---\n".join(output)
        except Exception as e:
            return f"Erreur pendant la recherche web: {e}"

# -------------------------
# Tools pour résolution d'issues
# -------------------------


class CloneRepoInput(BaseModel):
    repo_url: str = Field(..., description="SSH du dépôt GitHub à cloner")
   

class CloneRepoTool(BaseTool):
    name: str = "git_clone"
    description: str = "Clone un dépôt GitHub localement (utilise `git` en CLI)."
    args_schema: Type[BaseModel] =CloneRepoInput

    def _run(self, repo_url: str) -> str:
        try:
          
            # run git clone
            subprocess.run(["git", "clone", repo_url], check=True)
            return f"Cloned to: {repo_url.split('/')[-1].replace('.git','')}"
        except Exception as e:
            return f"Erreur clone: {e}"


class ReadFileInput(BaseModel):
    path: str = Field(..., description="Chemin relatif dans le repo vers le fichier à lire")
    #repo_dir: Optional[str] = Field(None, description="Chemin du repo local")


class ReadFileTool(BaseTool):
    name: str = "read_file"
    description: str = "Lit un fichier dans le dépôt cloné et retourne son contenu."
    args_schema: Type[BaseModel] = ReadFileInput

    def _run(self, path: str) -> str:
        try:
            base = Path('.')
            p = base / path
            return p.read_text(encoding='utf-8')
        except Exception as e:
            return f"Erreur lecture fichier: {e}"


class WriteFileInput(BaseModel):
    path: str = Field(..., description="Chemin relatif dans le repo vers le fichier à écrire")
    content: str = Field(..., description="Contenu à écrire")
    


class WriteFileTool(BaseTool):
    name: str = "write_file"
    description: str = "Écrit du contenu dans un fichier du dépôt cloné (crée les répertoires si besoin)."
    args_schema: Type[BaseModel] = WriteFileInput

    def _run(self, path: str, content: str, repo_dir: Optional[str] = None) -> str:
        try:
            base = Path(repo_dir) if repo_dir else Path('.')
            p = base / path
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding='utf-8')
            return f"Wrote {p}"
        except Exception as e:
            return f"Erreur write fichier: {e}"


class RunTestsInput(BaseModel):
    repo_dir: Optional[str] = Field(None, description="Chemin du repo local")
    pytest_args: Optional[str] = Field(None, description="Arguments supplémentaires pour pytest")


class RunTestsTool(BaseTool):
    name: str = "run_tests"
    description: str = "Exécute la suite de tests via `pytest` si présente dans le dépôt."
    args_schema: Type[BaseModel] = RunTestsInput

    def _run(self, repo_dir: Optional[str] = None, pytest_args: Optional[str] = None) -> str:
        try:
            cwd = repo_dir or '.'
            cmd = ["pytest", "-q"]
            if pytest_args:
                cmd += pytest_args.split()
            p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
            return json.dumps({"rc": p.returncode, "stdout": p.stdout, "stderr": p.stderr})
        except FileNotFoundError:
            return json.dumps({"rc": 0, "stdout": "pytest not found; skipped", "stderr": ""})
        except Exception as e:
            return f"Erreur run tests: {e}"


class GitCommitInput(BaseModel):
    repo_dir: Optional[str] = Field(None)
    message: str = Field(...)


class GitCommitTool(BaseTool):
    name: str = "git_commit"
    description: str = "Ajoute et commit les changements dans le dépôt local." 
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
            return f"Erreur git commit: {e}"


class GitPushInput(BaseModel):
    repo_dir: Optional[str] = Field(None)
    
    owner: Optional[str] = Field(None, description="Owner GitHub (optionnel, utile si token est utilisé)")
    repo: Optional[str] = Field(None, description="Nom du repo GitHub (optionnel)")
    force: Optional[bool] = Field(False, description="Pousser avec --force si nécessaire")


class GitPushTool(BaseTool):
    name: str = "git_push"
    description: str = "Pousse la branche locale vers le remote 'origin' (optionnellement en injectant le token pour l'auth)."
    args_schema: Type[BaseModel]  = GitPushInput

    def _run(self, repo_dir: Optional[str] = None, owner: Optional[str] = None, repo: Optional[str] = None, token_env: Optional[str] = 'GITHUB_TOKEN', force: Optional[bool] = False) -> str:
        try:
            branch=f'test_{ID}'
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
            return f"Erreur git push: {e}"


class CreateBranchInput(BaseModel):
    repo_dir: Optional[str] = Field(None, description="Chemin du repo local")
    branch: str = Field(..., description="Nom de la branche à créer")
    start_point: Optional[str] = Field(None, description="Commit/branch de départ (ex: main)")
    checkout: Optional[bool] = Field(True, description="Si True, basculer sur la nouvelle branche après création")


class CreateBranchTool(BaseTool):
    name: str = "git_create_branch"
    description: str = "Crée une nouvelle branche locale et (optionnel) bascule dessus."
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
            return f"Erreur create branch: {e}"


class CreatePRInput(BaseModel):
    owner: str = Field(...)
    repo: str = Field(...)
    head_branch: str = Field(...)
    base_branch: Optional[str] = Field(None)
    title: Optional[str] = Field(None)
    body: Optional[str] = Field(None)


class CreatePRTool(BaseTool):
    name: str = "create_pr"
    description: str = "Crée une Pull Request sur GitHub en utilisant la variable d'environnement GITHUB_TOKEN." 
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
            return f"Erreur create PR: {e}"


class FetchIssueInput(BaseModel):
    owner: str = Field(..., description="Owner GitHub du dépôt")
    repo: str = Field(..., description="Nom du dépôt GitHub")
    issue_number: Optional[int] = Field(None, description="Numéro de l'issue (facultatif)")


class FetchIssueTool(BaseTool):
    name: str = "fetch_issue"
    description: str = "Récupère les issues depuis l'API GitHub. Si le dépôt n'a qu'une issue, retourne son contenu JSON."
    args_schema: Type[BaseModel] = FetchIssueInput

    def _run(self, owner: str, repo: str, issue_number: Optional[int] = None) -> str:
        try:
            token = os.environ.get('GITHUB_TOKEN')
            api = f"https://api.github.com/repos/{owner}/{repo}/issues"
            headers = {"Accept": "application/vnd.github+json"}
            if token:
                headers["Authorization"] = f"token {token}"
            resp = requests.get(api, headers=headers, params={"state": "open"})
            if resp.status_code != 200:
                return f"Erreur GitHub API: {resp.status_code} {resp.text}"
            issues = resp.json()
            if issue_number:
                for it in issues:
                    if it.get('number') == int(issue_number):
                        return json.dumps(it)
                return f"Issue #{issue_number} introuvable"
            # if only one issue present, return it directly
            if isinstance(issues, list) and len(issues) == 1:
                return json.dumps(issues[0])
            # Otherwise return a short summary list
            summary = [{"number": i.get('number'), "title": i.get('title')} for i in issues]
            return json.dumps(summary)
        except Exception as e:
            return f"Erreur fetch issue: {e}"


class RepoTreeInput(BaseModel):
    owner: str = Field(...)
    repo: str = Field(...)
    path: Optional[str] = Field(None, description="Chemin relatif à lister (vide = racine)")
    ref: Optional[str] = Field(None, description="Branch/sha (par défaut: main/master)")


class RepoTreeTool(BaseTool):
    name: str = "repo_tree"
    description: str = "Liste l'arborescence du dépôt via l'API GitHub et peut lire le contenu d'un fichier donné." 
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
                return f"Erreur GitHub contents API: {resp.status_code} {resp.text}"
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
            return f"Erreur repo tree: {e}"


# -------------------------
# Construction des agents et crew avec outils étendus
# -------------------------

# LLM Object from crewai package
args = parse_args()
if args.user_id is not None:
    try:
        ID = int(args.user_id)
    except Exception:
        ID = 0
HOST=args.host
print(f"Starting Git issue solver with user_id={ID} using model={MODEL} at host={HOST}")
# Create LLM client. Some versions of the `crewai` package attempt to import
# native providers (e.g. OpenAI) on instantiation and will raise an
# ImportError complaining about missing OPENAI_API_KEY even when we intend
# to use an Ollama server via `base_url`.
try:
    llm = LLM(model=f"ollama/{MODEL}", base_url=HOST)
except ImportError as e:
    msg = str(e)
    # If the ImportError mentions OPENAI_API_KEY, set an empty value and retry.
    if "OPENAI_API_KEY" in msg or "OPENAI" in msg:
        if not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = ""
            print("WARNING: OPENAI_API_KEY was not set. Setting empty value to avoid ImportError from native provider imports.")
        # Retry creating the LLM client now that the env var is present
        llm = LLM(model=MODEL, base_url=HOST)
    else:
        # re-raise if it's an unexpected ImportError
        raise

# instantiate tools
web_tool = WebSearchTool()
clone_tool = CloneRepoTool()
read_tool = ReadFileTool()
write_tool = WriteFileTool()
tests_tool = RunTestsTool()
commit_tool = GitCommitTool()
push_tool = GitPushTool()
pr_tool = CreatePRTool()
create_branch_tool = CreateBranchTool()


agent_goal = (
    f"Dans le repertoire git@github.com:DjoserKhemSimeu/dummy_agent.git, résoudre l'issue GitHub dummy_agent/issues_file.md "
    f"en local et proposer des PRs. Context: run by user_id={ID}. Votre environnement local est 'dummy_agent', owner='DjoserKhemSimeu', repo='dummy_agent'. L'erreur est dans 'dummy_agent/issue_file.md'."
)

agent1 = Agent(
    role="issue-fixer",
    goal=agent_goal,
    backstory=f"Agent autonome pour diagnostiquer, proposer et appliquer des correctifs sur des dépôts GitHub. (invoked by user {ID})",
    verbose=True,
    memory=True,
    tools=[clone_tool, create_branch_tool, read_tool, write_tool, tests_tool, commit_tool, push_tool],
    allow_delegation=True,
    llm=llm,
)

task1 = Task(
    description="Fixer l'issue fournie: cloner le repo, diagnostiquer, proposer un patch et créer une PR si les tests passent la pipeline de mise a jour du code doit etre la suivante:" \
    "1. Cloner le dépôt GitHub localement." \
    "2. Créer une nouvelle branche pour les modifications." \
    "3. Analyser l'issue et le code pour identifier la cause racine."\
    "4. Apporter les modifications nécessaires au code source." \
    "5. Exécuter la suite de tests pour valider les modifications." \
    "6. Si les tests passent, committer les changements et pousser la branche vers le dépôt distant.",
    expected_output="PR ouverte ou diagnostic avec étapes à suivre",
    agent=agent1,
)

crew = Crew(
    agents=[agent1],
    model="ollama/mistral-nemo",
    tasks=[task1],
    cache=True,
    verbose=True,
    process=Process.sequential,
    planning=True,
    planning_llm=llm,
)

result = crew.kickoff()

print("\n--- Résultat final ---")
print(result)