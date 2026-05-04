# CCSQL

Text-to-SQL agent powered by Claude Code, evaluated on the [Spider2](https://spider2-sql.github.io/) benchmark over Snowflake databases.

Given a natural-language question and a Snowflake database, CCSQL launches Claude Code with MCP tools for schema exploration and query execution, producing a SQL answer for each question.

## Setup

### 1. Install uv (if needed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Create the virtual environment and install dependencies

```bash
uv venv --python 3.14
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 3. Download the Spider2-Snow data

Download `spider2-snow.zip` from [Google Drive](https://drive.google.com/file/d/1sgc-oFqkETkLoJiu90SH1RJI70nTbRL8/view?usp=share_link), then unzip it at the project root:

```bash
unzip spider2-snow.zip -d spider2-snow
```

The resulting `spider2-snow/` directory should contain:
- `snowflake_credential.json` — Snowflake connection credentials
- `resource/databases_no_nulls_2/` — per-database JSON metadata
- `resource/documents/` — external knowledge documents (referenced by some questions)

### 4. Install Claude Code CLI

Follow the [Claude Code docs](https://docs.anthropic.com/en/docs/claude-code/overview) to install the `claude` CLI and authenticate.

## Project structure

```
.
├── CLAUDE.md              # Agent instructions (injected into every Claude Code session)
├── .mcp.json              # MCP server config — wires up the snowflake-tools server
├── ccsql_test.jsonl       # Evaluation dataset (JSONL, one question per line)
├── requirements.txt       # Python dependencies
├── spider2-snow/          # Data directory (downloaded separately)
└── src/
    ├── launcher.py        # SLURM array-job launcher — splits dataset into folds
    ├── run_fold.py        # Processes one JSONL fold sequentially
    ├── run_question.py    # Invokes Claude Code on a single question
    ├── mcp_server.py      # FastMCP server exposing DB tools to Claude Code
    ├── db_tools.py        # Schema/query helpers (Snowflake + SQLite)
    └── python_interpreter_worker.py  # Subprocess worker for the python_interpreter tool
```

## Usage

### Single question

```bash
python src/run_question.py \
    --question_json '{"instance_id":"sf_bq194","instruction":"...","db_id":"GITHUB_REPOS"}' \
    --output_dir inference_res/test \
    --db_path spider2-snow
```

### Full benchmark (SLURM)

```bash
python src/launcher.py \
    --input_file ccsql_test.jsonl \
    --num_folds 20 \
    --run_name my-run \
    --output_dir inference_res \
    --max_concurrent_jobs 10 \
    --timeout 600
```

Each question produces an `answer.sql` file under `inference_res/<run_name>/<instance_id>/`.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `sonnet` | Claude model (`sonnet`, `opus`, `haiku`) |
| `--effort` | — | Thinking effort (`low`, `medium`, `high`, `max`) |
| `--timeout` | `600` | Seconds per question before killing the process |
| `--no_skip` | off | Re-run questions that already have an `answer.sql` |
