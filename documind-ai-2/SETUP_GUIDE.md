# DocuMind AI: setup guide

How to get DocuMind AI running on your own computer, from nothing installed to asking your first question.

## 1. What you need to download

| What | Why | Where | Size (approx.) |
|---|---|---|---|
| **Python 3.11 or 3.12** (3.13 also works) | Runs the app | [python.org/downloads](https://www.python.org/downloads/). On Windows, tick **"Add python.exe to PATH"** in the installer | ~30 MB |
| **A Gemini API key** | Lets the app write answers | [Google AI Studio](https://aistudio.google.com/apikey) (sign in with a Google account, click *Create API key*) | free to create; check Google's current quota and pricing |
| **Python packages** | Streamlit, PyMuPDF, ChromaDB, Google Gen AI SDK, sentence-transformers (brings PyTorch), etc. | Installed automatically by one `pip` command in step 3 | ~1-2 GB on disk |
| **Embedding model** (`all-MiniLM-L6-v2`) | Turns text into searchable vectors | Downloads **by itself** the first time you upload a PDF; needs internet once | ~90 MB |
| Git (optional) | Only needed to publish to GitHub | [git-scm.com](https://git-scm.com/downloads) | ~60 MB |
| VS Code (optional) | A comfortable editor | [code.visualstudio.com](https://code.visualstudio.com/) | ~100 MB |

You do **not** need a GPU, Docker, or any database. Everything runs on CPU, in memory.

## 2. Get the project

Unzip `documind-ai.zip`, then open a terminal **inside** the `documind-ai` folder (the one containing `app.py`).

- **Windows:** open the folder in File Explorer, click the address bar, type `powershell`, press Enter.
- **macOS / Linux:** `cd path/to/documind-ai`

## 3. Install

**Windows (PowerShell)**

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
Copy-Item .env.example .env
```

If PowerShell refuses to run the activate script: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then try again.

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
```

The install takes a few minutes because of PyTorch. You will see `(.venv)` at the start of your prompt while the environment is active. Run the activate line again whenever you open a new terminal.

## 4. Add your API key

1. Open the new `.env` file in any text editor.
2. Replace `your-api-key-here` with your key:
   ```
   GEMINI_API_KEY=AIza...your-real-key...
   ```
3. Save. **Never** commit or share this file. It is already listed in `.gitignore`.

Model names change. The default is `gemini-3.8-flash`. If you ever see "model was not found", check Google's [model list](https://ai.google.dev/gemini-api/docs/models) and set `GEMINI_MODEL=...` in `.env`.

## 5. Run it

```bash
streamlit run app.py
```

Your browser opens at <http://localhost:8501>. Click **Load the sample handbook** on the first screen (or upload your own PDF in the sidebar), then ask, for example: *"How much can shading reduce output?"*

The first upload is slow (10-60 seconds) because the embedding model downloads. After that it is fast.

Stop the app with `Ctrl+C` in the terminal.

### Try it without downloading PyTorch (demo mode)

Quality is lower (matches words, not meaning), but it needs no large downloads:

- macOS / Linux: `EMBEDDING_BACKEND=hashing streamlit run app.py`
- Windows: `$env:EMBEDDING_BACKEND="hashing"; streamlit run app.py`

## 6. Check that everything works

```bash
python -m pytest -q                      # 101 tests, no API key or internet needed
python -m eval.run_eval --embedder hashing
```

Manual checklist:

- [ ] A text PDF uploads and shows "N pages, N passages"
- [ ] A question the PDF answers gives an answer with blue **S1** badges and a *Sources* box with a quote and page number
- [ ] A question the PDF does **not** cover gets "Not enough evidence" instead of a guess
- [ ] **Study tools** → Summary / Key points / FAQ / Quiz generate and the study guide downloads
- [ ] **Clear session** removes the documents

## 7. Troubleshooting

| You see | Do this |
|---|---|
| `streamlit: command not found` / `'streamlit' is not recognized` | The virtual environment is not active. Run the activate line from step 3 |
| `ModuleNotFoundError: src` or `config` | Run commands from inside the `documind-ai` folder |
| "No Gemini API key is set" | Check `.env` is in the same folder as `app.py`, has no quotes or spaces around the key, then restart the app |
| "API rejected the API key" | Key was revoked, mistyped, or has no access. Create a new one |
| "quota or rate limit was reached" | Wait a minute, or check your quota in Google AI Studio |
| "embedding model could not be loaded" | First run needs internet (huggingface.co must be reachable). Check proxy/firewall |
| "No extractable text" | Scanned PDF. Run OCR first (for example OCRmyPDF), then upload |
| Everything is refused as "not enough evidence" | Lower `MIN_SIMILARITY` in `.env` (try `0.18`) |
| Install fails on `torch` | Use Python 3.11 or 3.12, and run `python -m pip install --upgrade pip` first |

## 8. Put it on GitHub

```bash
git init
git add .
git status          # confirm .env is NOT listed
git commit -m "Initial commit: DocuMind AI"
git branch -M main
git remote add origin https://github.com/<your-username>/documind-ai.git
git push -u origin main
```

Create the empty repository on github.com first (no README, no .gitignore). If `.env` ever appears in `git status`, stop and fix `.gitignore` before committing.

## 9. Deploy publicly (Streamlit Community Cloud)

1. Push to GitHub (step 8).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **Create app** → pick the repo, branch `main`, file `app.py`; under *Advanced settings* choose Python 3.11 or 3.12.
3. Paste into *Secrets*:
   ```toml
   GEMINI_API_KEY = "your-real-key"
   ```
4. Deploy. The first boot is slow (installs PyTorch, downloads the model). Free tiers have limited memory, so if the app gets killed, switch to `EMBEDDING_BACKEND=hashing` or a host with more RAM.

A public app spends **your** Gemini quota. The built-in rate limits help, but set a quota or billing alert in Google Cloud as well.
