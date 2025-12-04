<!-- Copilot instructions for AI coding agents -->
# Repository notes for AI coding agents

This repository contains a minimal workspace with two top-level folders: `backend/` and `frontend/`.
Currently both folders are empty. Use the guidance below to quickly discover the real tech stack, run common developer workflows, and make low-risk edits.

1. Quick repo summary
- **Layout:** top-level `backend/` and `frontend/` directories.
- **State:** folders are present but currently contain no files — treat the repository as a scaffold until you locate language-specific manifests.

2. First actions to discover the project
- Search for common manifest files at the repository root and inside `backend/` and `frontend/`: `package.json`, `yarn.lock`, `pnpm-lock.yaml`, `pyproject.toml`, `requirements.txt`, `Pipfile`, `go.mod`, `Cargo.toml`, `*.csproj`, `pom.xml`, `Dockerfile`, `.devcontainer/devcontainer.json`.
- Commands you can run locally (PowerShell):
  - `Get-ChildItem -Path . -Recurse -File -Include package.json,pyproject.toml,requirements.txt,Dockerfile` 
  - `git status --porcelain` (to check working tree)

3. If manifests are missing
- Ask the user clarifying questions before making assumptions: which language(s) are used, how to run locally, and where secrets/config are stored.
- If the user is unavailable, create minimal, reversible changes (README notes, TODO files) and clearly label them as guesses.

4. Editing conventions for this repo
- Keep changes minimal and focused — prefer small, single-purpose commits.
- Use the repository root `backend/` and `frontend/` as the default places to add code for server and client respectively.
- If you add files, update `README.md` in the corresponding folder describing how to run that component.

5. Discovery checklist (look for these exact file/directory names)
- `backend/` — expect `src/`, `app/`, `server/`, `requirements.txt`, `pyproject.toml`, `package.json`, `Dockerfile`.
- `frontend/` — expect `src/`, `public/`, `package.json`, `vite.config.*`, `webpack.config.*`, `tsconfig.json`.
- CI/automation — look for `.github/workflows/`, `azure-pipelines.yml`, `Jenkinsfile`.

6. Safety and rollback
- Don't change global config or add secrets. Add `.env.example` rather than `.env` if you need to demonstrate environment variables.
- Prefer non-destructive edits and include a concise explanation in commit messages.

7. Examples of helpful automated tasks to suggest to the user
- If you find a `package.json` in `frontend/`: suggest `npm install` then `npm run dev` or `npm run build` depending on scripts present.
- If you find `pyproject.toml` in `backend/`: suggest `python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt` (or use `pip` with `pyproject` tools).

8. What to do if tests or builds fail
- Report the exact error and the commands you ran. Avoid guessing fixes for failing CI unless you can reproduce locally.

9. When merging this file
- If a `.github/copilot-instructions.md` already exists, merge conservatively: preserve any project-specific commands or historical notes and add missing discovery steps above.

If anything here is unclear or you want this file adjusted to a particular stack (Node, Python, .NET, etc.), tell me which stack to prioritize and I will update this guidance with concrete commands and examples.
