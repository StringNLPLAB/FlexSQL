# Spider2-Snowflake Metadata Preprocessing

Scripts that fix known issues in the released Spider2 (Snowflake split) metadata so
that downstream schema-linking and SQL-generation steps see clean, accurate inputs.

All scripts query Snowflake to validate / enrich the on-disk JSON + DDL files, so a
working `datasets/Spider2/spider2-snow/snowflake_credential.json` is required.

## Pipeline

The four scripts must be run in this order — each consumes the output of the
previous step:

1. **`normalize_spider2_snowflake.py`** — rewrites table / column / schema
   identifiers in the metadata JSON and DDL CSV files to match the actual
   casing returned by Snowflake's `INFORMATION_SCHEMA`. The released metadata
   is inconsistent about quoted vs. unquoted identifiers, which breaks any SQL
   that round-trips through it.

2. **`remove_null_columns.py`** — queries Snowflake to find columns that contain
   only `NULL` (or only the literal string `"nan"`) and strips them from both
   the JSON metadata and the DDL CSVs. These columns add noise to schema
   linking without contributing useful signal.

3. **`add_example_value.py`** — fills in `column_examples` on each table's JSON
   by sampling up to 10 distinct non-null values per column directly from
   Snowflake. Existing `sample_rows` are reused first; columns with fewer than
   2 non-null examples there get a follow-up query.

4. **`get_table_similarities.py`** — scans the cleaned metadata and groups
   tables within each database that share a name prefix or suffix **and** have
   identical column-name/type schemas (e.g. partitioned tables like
   `GSOD2007`, `GSOD2008`, …). Emits a JSON report consumed by `main.py` for
   downstream schema linking. Offline / no Snowflake calls.

## Run everything

From the project root:

```bash
bash src/preprocessing/run_all.sh
```

The wrapper resolves paths relative to the project root, so it works from any cwd.
Override defaults via env vars:

```bash
METADATA_ROOT=datasets/Spider2/spider2-snow/resource/databases \
OUTPUT_ROOT=datasets/Spider2/spider2-snow/resource/databases_no_nulls_2 \
CREDENTIAL_PATH=datasets/Spider2/spider2-snow/snowflake_credential.json \
MAX_WORKERS=32 \
bash src/preprocessing/run_all.sh
```

| Variable             | Default                                                          | Used by       |
| -------------------- | ---------------------------------------------------------------- | ------------- |
| `METADATA_ROOT`      | `datasets/Spider2/spider2-snow/resource/databases`               | step 1 input  |
| `OUTPUT_ROOT`        | `datasets/Spider2/spider2-snow/resource/databases_no_nulls_2`    | steps 1–4     |
| `CREDENTIAL_PATH`    | `datasets/Spider2/spider2-snow/snowflake_credential.json`        | steps 1–3     |
| `SIMILARITY_REPORT`  | `table_similarities_report.json`                                 | step 4 output |
| `MAX_WORKERS`        | `32`                                                             | step 2        |
| `PYTHON`             | `python`                                                         | interpreter   |

Steps 2 and 3 read and write the same `OUTPUT_ROOT` (step 2 cleans in place,
step 3 enriches the cleaned files), and step 4 reads from `OUTPUT_ROOT` to
produce `SIMILARITY_REPORT`. Re-running the pipeline overwrites the previous
run.

## Run a single step

Each script also runs standalone — useful when iterating on one stage:

```bash
python src/preprocessing/normalize_spider2_snowflake.py \
    --metadata-root datasets/Spider2/spider2-snow/resource/databases \
    --output-root  datasets/Spider2/spider2-snow/resource/databases_no_nulls_2 \
    --credential-path datasets/Spider2/spider2-snow/snowflake_credential.json

python src/preprocessing/remove_null_columns.py \
    --max-workers 32 \
    --metadata-root datasets/Spider2/spider2-snow/resource/databases_no_nulls_2 \
    --output-root  datasets/Spider2/spider2-snow/resource/databases_no_nulls_2 \
    --credential-path datasets/Spider2/spider2-snow/snowflake_credential.json

python src/preprocessing/add_example_value.py \
    --db_folder datasets/Spider2/spider2-snow/resource/databases_no_nulls_2 \
    --credential_path datasets/Spider2/spider2-snow/snowflake_credential.json

python src/preprocessing/get_table_similarities.py \
    --metadata-root datasets/Spider2/spider2-snow/resource/databases_no_nulls_2 \
    --output table_similarities_report.json
```

`add_example_value.py` imports `get_ddl` from `src/` and `utils.program_frame_sf`
from the project root — invoke it from the project root (or via `run_all.sh`) so
those imports resolve.
