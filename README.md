# FlexSQL: Flexible Exploration and Execution Make Better Text-to-SQL Agents

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2602.02301-b31b1b.svg)]()

</div>



---

End-to-end pipeline for **Spider2-Snowflake**: environment setup, data download, metadata preprocessing, multi-program inference, majority voting, and evaluation under both pass@k and pass@1.

> 💡 The same instructions apply to the **SQLite subset (Spider2-Lite)**. Download `spider2-lite` from the [official repo](https://github.com/xlang-ai/Spider2) and pass `--db_path datasets/Spider2/spider2-lite/resource/databases/sqlite --db_type sqlite` to the agent script in place of the Snowflake equivalents.

## 💡 Overview

Given a natural-language question and a Snowflake schema, the pipeline:

1. Cleans and enriches the released Spider2 metadata so schema linking sees accurate column casing, no all-null columns, and concrete example values.
2. Generates `k` candidate programs per question with a hierarchical schema-linking + 2-step batched planning agent, executed against Snowflake.
3. Reduces the `k` candidates to a single answer per question via table-equivalence majority voting, and reports both pass@k and pass@1.

## ⚙️ Setup

Install [uv](https://docs.astral.sh/uv/) and sync dependencies:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
. .venv/bin/activate
```

## 📥 Download Spider2-Snowflake data

Fetch the Spider2-Snowflake split from the official repo ([xlang-ai/Spider2](https://github.com/xlang-ai/Spider2)) and place it at `datasets/Spider2/spider2-snow/`:

```bash
mkdir -p datasets/Spider2
git clone --depth 1 https://github.com/xlang-ai/Spider2.git /tmp/Spider2
cp -r /tmp/Spider2/spider2-snow datasets/Spider2/
```

After this step the tree should look like:

```
datasets/Spider2/spider2-snow/
├── spider2-snow.jsonl
├── snowflake_credential.json     # fill in with your Snowflake credentials
├── resource/
│   └── databases/
└── evaluation_suite/
    ├── evaluate.py
    └── gold/
```

> ⚠️ A working Snowflake account is required. Populate `datasets/Spider2/spider2-snow/snowflake_credential.json` before running any preprocessing or evaluation step.

## 🧹 Preprocessing

The released Spider2-Snowflake metadata has casing inconsistencies, columns with only null values, and missing example values. We provide a cleaned/enriched version.

**Option A — download the prepared metadata (recommended):**

Download from [Google Drive](https://drive.google.com/file/d/1hBPo75NCOZ2istvKWixWheSB1jdXoBxa/view?usp=share_link) and extract under `datasets/Spider2/spider2-snow/`. Expected layout after extraction:

```
datasets/Spider2/spider2-snow/resource/databases_no_nulls_2/
datasets/Spider2/spider2-snow/table_similarities_report_no_nulls.json
```

**Option B — run the preprocessing pipeline yourself:**

See `[src/preprocessing/README.md](src/preprocessing/README.md)`. In short:

```bash
bash src/preprocessing/run_all.sh
```

This produces `resource/databases_no_nulls_2/` and a table-similarity report under `datasets/Spider2/spider2-snow/`.

## 🚀 Run the main experiment

The main experiments are driven by `src/main.py` which can be launched via `src/main_slurm_laucher.py` on a Slurm cluster with local vllm servers.

```bash
python src/main_slurm_laucher.py \
  --input_file datasets/Spider2/spider2-snow/spider2-snow.jsonl \
  --num_folds 600 \
  --vllm_model <path-to-your-model> \
  --gpu_name l40s \
  --vllm_server_extra_args "-tp 4 --enable-auto-tool-choice --tool-call-parser openai --gpu-memory-utilization 0.8" \
  --sbatch_args $'#SBATCH --time=3:00:00' \
  --base_dir inference_res \
  --slurm_log_dir gpt-oss-120b-logs \
  --agent_script_extra_args "--model gpt-oss-120b \
    --db_path datasets/Spider2/spider2-snow --db_type snowflake \
    --num_programs 1 --hierarchical-sl --planning_top_k 8 --planning_batch_size 4 --use_2step_batch_planning \
    --similarities_path datasets/Spider2/spider2-snow/table_similarities_report.json \
    --custom_exp_name spider2-snow-exp \
    --planning_gen_config '{\"temperature\": 1.0}' \
    --stitching_gen_config '{\"temperature\": 1.0}' \
    --eval_gen_config '{\"temperature\": 1.0}' \
    --max_self_eval_rounds 3"
```

Per-question outputs land under `inference_res/<custom_exp_name>/<question_id>/`, each containing `program_<k>.py` (or `.sql`) and a corresponding `program_output_<k>.csv`. 

Our experiment logs are available [here](https://drive.google.com/drive/folders/13Wt9aP7XChcxriFdD2Kz2d1mFvTsvASA?usp=sharing)
## 🗳️ Majority voting

Reduce the `k` candidate outputs per question to a single majority answer:

```bash
python utils/major_voting.py \
  --input_dir  inference_res/<run_name>/<custom_exp_name> \
  --output_dir inference_res/<run_name>/<custom_exp_name>_majority \
  --fuzzy_threshold 0.90 \
  --allow_superset
```

This writes one `<question_id>.csv` per question into `--output_dir`.

## 📊 Evaluation

### pass@k — `utils/evaluate_passk.py`

Pass@k groups all `k` candidates per question and counts a question correct if *any* candidate matches gold. Use the provided wrapper, which first flattens the per-question `program_output_*.csv` files into a single folder:

```bash
bash eval_spider20.sh inference_res/<run_name>/<custom_exp_name>
```

This calls:

```bash
python utils/move_csv_to_evaluate_folder.py \
    --source_dir <infer_folder> \
    --output    <infer_folder>/evaluation

python utils/evaluate_passk.py \
    --result_dir <infer_folder>/evaluation \
    --gold_dir   datasets/Spider2/spider2-snow/evaluation_suite/gold \
    --mode       exec_result
```

### pass@1 — Spider2 official evaluator

Pass@1 evaluates a single answer per question. Run majority voting first (previous section), then point the official Spider2 evaluator at the majority output:

```bash
python datasets/Spider2/spider2-snow/evaluation_suite/evaluate.py \
    --mode      exec_result \
    --result_dir inference_res/<run_name>/<custom_exp_name>_majority \
    --gold_dir  datasets/Spider2/spider2-snow/evaluation_suite/gold
```

## 🤖 Claude Code integration

A Claude Code variant of this pipeline — packaged as [Claude Code skills](https://docs.claude.com/en/docs/claude-code) and backing MCP servers — is available in `claude_harness`.


## 📜 Citation



```bibtex
@misc{pham2026flexsql,
      title={FlexSQL: Flexible Exploration and Execution Make Better Text-to-SQL Agents}, 
      author={Quang Hieu Pham and Yang He and Ping Nie and Canwen Xu and Davood Rafiei and Yuepeng Wang and Xi Ye and Jocelyn Qiaochu Chen},
      year={2026},
      eprint={2605.02815},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2605.02815}, 
}
```

