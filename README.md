# Meeting Memory

A small meeting search application for the Catalist Agentic AI Intern build test. It contains all 138 AMI meeting transcripts, the 25 supplied assessment questions, 25 individually source-checked answers, and a pre-meeting brief for Team TS3012. The answers in this archive were prepared by direct transcript review and validated locally. They are not outputs from a deployed inference model.

The app has three sections: Ask a question, Assessment questions, and Meeting brief. The assessment section displays the included source-checked responses without an active database. Live questions use indexed retrieval and model-based reasoning, so they require a PostgreSQL connection and a configured model API key. Without a model API key, a connected database provides source excerpts but does not pretend to generate answers.

## Stack

- Next.js App Router, TypeScript, React, Tailwind CSS
- Python, FastAPI, Pydantic and Uvicorn
- Neon PostgreSQL with native full-text search and GIN indexes
- Optional OpenAI-compatible model API for interpreting evidence
- GitHub repository, Vercel frontend and Render backend deployment configuration

There is no Docker, queue service, separate vector database or user authentication.

## How it works

The importer preserves the original meeting text, speaker letters, timestamps and chronological order. It builds overlapping text passages for retrieval while keeping original turns separate for checking quotes. PostgreSQL searches indexed passages and returns speaker turns. The reasoning service compares earlier and later evidence. If the first pass is incomplete, it performs a second search. Every generated citation must match at least six words in one original speaker turn before an answer is returned.

The initial corpus contains 138 meetings. The importer is incremental: rerunning it skips files with unchanged SHA-256 hashes. It never runs as part of the normal web server startup.

## Repository layout

- `frontend/`: Next.js interface
- `backend/app/`: FastAPI, parser, database, retrieval, reasoning and citation checks
- `backend/scripts/`: initialization, answer generation and validation commands
- `backend/tests/`: unit tests, API tests and simulated database-ingestion tests
- `data/`: original 138 meeting transcript files, questions and credits
- `evaluation/`: 15 independent source-checked questions, offline retrieval results and preprocessing benchmark
- `docs/technical_writeup.md`: architecture and limitations
- `answers.json`: completed 25 answers and pre-final TS3012 briefing

## Requirements

Python 3.11 or newer and Node.js 20 or newer. To run live queries, create a Neon PostgreSQL project and obtain its connection string. Model-based synthesis additionally requires a working OpenAI-compatible API key and an available model. Free-tier access to inference is not assumed.

## Run the backend

In Bash:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export DATABASE_URL='postgresql://USER:PASSWORD@HOST.neon.tech/DB?sslmode=require'
export FRONTEND_URL='http://localhost:3000'
export MODEL_API_KEY='YOUR_API_KEY'
export MODEL_NAME='gpt-4o-mini'
python -m scripts.initialize
uvicorn app.main:app --reload --port 8000
```

In Windows PowerShell:

```powershell
cd backend
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:DATABASE_URL = 'postgresql://USER:PASSWORD@HOST.neon.tech/DB?sslmode=require'
$env:FRONTEND_URL = 'http://localhost:3000'
$env:MODEL_API_KEY = 'YOUR_API_KEY'
$env:MODEL_NAME = 'gpt-4o-mini'
python -m scripts.initialize
uvicorn app.main:app --reload --port 8000
```

The two `YOUR` values are examples and must be replaced with credentials issued by your providers. `MODEL_API_KEY` can be omitted if only transcript retrieval is needed. PostgreSQL is still required for live search. The API health route, provided questions, saved answers and saved brief also work without Neon and without a model key.

The initialization command creates the database tables and GIN index, then ingests or updates the 138 transcripts. It is safe to run again when new meeting files are added. Data is kept in Neon rather than the Render container's local filesystem.

For an OpenAI-compatible inference provider, set `MODEL_API_BASE_URL` only if using an alternative supported API endpoint. Ensure `MODEL_NAME` is available to your account. Do not commit real keys or `.env` files to Git.

## Run the frontend

In another terminal:

```bash
cd frontend
npm install
cp .env.example .env.local
npm run dev
```

PowerShell equivalent:

```powershell
cd frontend
npm install
Copy-Item .env.example .env.local
npm run dev
```

Open `http://localhost:3000`. The default API URL is `http://localhost:8000`. Set `NEXT_PUBLIC_API_URL` in `frontend/.env.local` when needed.

## Check the API

Visit `http://localhost:8000/docs`, or run:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/api/questions
curl http://localhost:8000/api/answers
curl http://localhost:8000/api/brief
curl -X POST http://localhost:8000/api/ask -H 'Content-Type: application/json' -d '{"question":"What decisions changed in Team TS3012?","team":"TS3012"}'
```

Use `Invoke-RestMethod` on Windows PowerShell for the POST request if `curl` is aliased to another command:

```powershell
Invoke-RestMethod -Method Post -Uri 'http://localhost:8000/api/ask' -ContentType 'application/json' -Body '{"question":"What decisions changed in Team TS3012?","team":"TS3012"}'
```

## Generate and validate answers

From `backend/`:

```bash
python -m scripts.validate_answers
python -m scripts.generate_answers
```

Validation works without credentials. The generation command requires an active Neon database and model API key. It replaces `answers.json` only if all 25 generated responses have valid citations. When a provider is not configured, it refuses to replace the included checked answers with invented output.

## Tests and evaluation

From `backend/`:

```bash
python -m pytest tests -q
```

From the repository root:

```bash
python evaluation/prepare_and_run.py
python evaluation/benchmark.py
```

The offline evaluation checks whether a local lexical retrieval baseline finds a manually verified source passage in the first five results for 15 new questions. It is not a live PostgreSQL benchmark or a measure of end-to-end model answer accuracy. The benchmark measures only local parsing and passage construction. Actual results are recorded in the JSON files and discussed in the write-up.

## Deploy on Neon, Render and Vercel

1. Create a Neon PostgreSQL project and copy its pooled connection string with SSL enabled. Do not expose it in the browser.
2. On your own computer, configure `DATABASE_URL` and run `python -m scripts.initialize` from the backend directory. The script writes directly to the Neon project. It only needs to run after the database is created or source files change.
3. Push this repository to GitHub. Create a new Render Web Service from the repository. Set its root directory to `backend`, build command to `pip install -r requirements.txt`, and start command to `uvicorn app.main:app --host 0.0.0.0 --port $PORT`. The repository also includes `render.yaml`.
4. In Render, set `DATABASE_URL`, `FRONTEND_URL`, and optionally `MODEL_API_KEY`, `MODEL_API_BASE_URL`, `MODEL_NAME`. Set `FRONTEND_URL` to your exact Vercel origin, including `https://`.
5. In Vercel, import the same repository, set root directory to `frontend` and framework to Next.js. Configure `NEXT_PUBLIC_API_URL` as the public Render service URL, without a trailing slash.
6. Verify `/health` and `/docs` on the public API, then use every section of the Vercel interface. Confirm cross-origin requests succeed and the Neon table contains 138 meetings.

Render's free instance can spin down during inactivity. Allow for the initial wake-up time. Exact external deployment URLs cannot be specified before the services exist.

## Known limitations

- No Neon project or inference API credentials were provided for this local submission. PostgreSQL connectivity, real inference calls and deployed URLs have not been verified.
- A real Next.js production build could not be run in the available environment because required npm packages were not cached and package installation was unavailable.
- The current source-checked `answers.json` was assembled from direct transcript review, not from a successfully executed live agent run. The separate model generation script is included but not executed here.
- The independent 15-question retrieval proxy is not equivalent to full agent evaluation; see `evaluation/results.json` for actual misses.
- Native PostgreSQL full-text search handles exact vocabulary better than subtle conversational paraphrases. At larger scale, search calibration, ranking evaluation and selective semantic retrieval may be needed.
- A thousand-meeting database insert and GIN index timing was not measurable without an active Neon project. The report separates measured local preprocessing from unmeasured DB and model costs.

## Data attribution

The transcript dataset is from the AMI Meeting Corpus, University of Edinburgh and partners, distributed under CC BY 4.0. Original attribution and source information are preserved in `data/CREDITS.txt`.
