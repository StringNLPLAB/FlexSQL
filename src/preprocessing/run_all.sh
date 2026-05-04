#!/usr/bin/env bash
# Run the full Spider2-Snowflake metadata preprocessing pipeline end-to-end.
#
# Resolves paths relative to the project root, so it can be invoked from anywhere:
#   bash src/preprocessing/run_all.sh
#   ./src/preprocessing/run_all.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

METADATA_ROOT="${METADATA_ROOT:-datasets/Spider2/spider2-snow/resource/databases}"
OUTPUT_ROOT="${OUTPUT_ROOT:-datasets/Spider2/spider2-snow/resource/databases_no_nulls_2}"
CREDENTIAL_PATH="${CREDENTIAL_PATH:-datasets/Spider2/spider2-snow/snowflake_credential.json}"
SIMILARITY_REPORT="${SIMILARITY_REPORT:-table_similarities_report.json}"
MAX_WORKERS="${MAX_WORKERS:-32}"

PY="${PYTHON:-python}"

echo "[1/4] Normalizing metadata identifier casing -> ${OUTPUT_ROOT}"
"${PY}" src/preprocessing/normalize_spider2_snowflake.py \
    --metadata-root "${METADATA_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --credential-path "${CREDENTIAL_PATH}"

echo "[2/4] Removing all-null / all-'nan' columns (in-place on ${OUTPUT_ROOT})"
"${PY}" src/preprocessing/remove_null_columns.py \
    --max-workers "${MAX_WORKERS}" \
    --metadata-root "${OUTPUT_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --credential-path "${CREDENTIAL_PATH}"

echo "[3/4] Adding column example values"
"${PY}" src/preprocessing/add_example_value.py \
    --db_folder "${OUTPUT_ROOT}" \
    --credential_path "${CREDENTIAL_PATH}"

echo "[4/4] Computing table-similarity report -> ${SIMILARITY_REPORT}"
"${PY}" src/preprocessing/get_table_similarities.py \
    --metadata-root "${OUTPUT_ROOT}" \
    --output "${SIMILARITY_REPORT}"

echo "Preprocessing complete."
echo "  Cleaned metadata:        ${OUTPUT_ROOT}"
echo "  Table similarity report: ${SIMILARITY_REPORT}"
