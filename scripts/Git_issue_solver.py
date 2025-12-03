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

MODEL=os.environ.get("BENCH_MODEL", "ollama/mistral-nemo")
# CLI: accept --user-id so external processes (bench) can pass the client id
def parse_args():
    p = argparse.ArgumentParser(description='Run Git issue solver agent (optionally with user id)')
    p.add_argument('--user-id', type=int, default=None, help='Optional user id to associate with this run')
    p.add_argument('--host', type=str, default="http://localhost:11432", help='Host where the LLM server is running')
    p.add_argument('--n_users', type=int, default=1, help='Number of users in the benchmark (for logging purposes)')
    return p.parse_args()

args = parse_args()
if args.user_id is not None:
    try:
        ID = int(args.user_id)
    except Exception:
        ID = 0
HOST=args.host
agent_env_path = os.path.join('agent_env', f'agent_env_user_{MODEL}_{args.n_users}_{ID}')
os.chdir(agent_env_path)

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
            if not results:
                return "No results."

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
            return f"Error during web search: {e}"

# -------------------------
# Tools for issue resolution
# -------------------------


class CloneRepoInput(BaseModel):
    repo_url: str = Field(..., description="SSH URL of the GitHub repository to clone")
   

class CloneRepoTool(BaseTool):
    name: str = "git_clone"
    description: str = "Clone a GitHub repository locally (uses the `git` CLI) and change into the cloned directory."
    args_schema: Type[BaseModel] =CloneRepoInput

    def _run(self, repo_url: str) -> str:
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
            branch=f'test_{MODEL}_{args.n_users}_{ID}'
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
    owner: str = Field(..., description="GitHub owner of the repository (DjoserKhemSimeu)")
    repo: str = Field(..., description="Name of the GitHub repository(dummy_agent)")
    issue_number: Optional[int] = Field(None, description="Issue number (1)")


class FetchIssueTool(BaseTool):
    name: str = "fetch_issue"
    description: str = "Fetches issues from the GitHub API. If the repository has only one issue, returns its JSON content 'body'."
    args_schema: Type[BaseModel] = FetchIssueInput

    def _run(self, owner: str = "DjoserKhemSimeu", repo: str = "dummy_agent", issue_number: Optional[int] = 1) -> str:
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


# -------------------------
# Construction of agents and crew with extended tools
# -------------------------

# LLM Object from crewai package

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
fetchIssue_tool = FetchIssueTool()
create_branch_tool = CreateBranchTool()


agent_goal = (
    f"In the repository git@github.com:DjoserKhemSimeu/dummy_agent.git, resolve the GitHub issue number 1 "
    f"locally and propose PRs. Context: run by user_id={ID}. Local environment 'dummy_agent', owner='DjoserKhemSimeu', repo='dummy_agent'."
)

agent1 = Agent(
    role="issue-fixer",
    goal=agent_goal,
    backstory=f"Autonomous agent to diagnose, propose, and apply fixes on GitHub repositories. (invoked by user {ID})",
    verbose=True,
    memory=True,
    tools=[clone_tool, create_branch_tool, read_tool, write_tool, tests_tool, commit_tool, push_tool, fetchIssue_tool],
    allow_delegation=True,
    llm=llm,
)

task1 = Task(
    description="Fix the provided issue: clone the repo, diagnose, propose a patch, and create a PR if tests pass. The update pipeline should be the following:" \
    "1. Clone the GitHub repository locally an jump into it." \
    "2. Create a new branch for the changes." \
    "3. Analyze the issue and the code to identify the root of the issue." \
    "4. Make the necessary changes to the source code." \
    "5. Run the test suite to validate the changes." \
    "6. If tests pass, commit the changes and push the branch to the remote repository.",
    expected_output="Report of the actions taken, including the URL of the created branch pushed.",
    agent=agent1,
)

crew = Crew(
    agents=[agent1],
    model=f"ollama/{MODEL}",
    tasks=[task1],
    cache=True,
    verbose=True,
    process=Process.sequential,
    planning=True,
    planning_llm=llm,
)

result = crew.kickoff()

print("\n--- Final result ---")
print(result)