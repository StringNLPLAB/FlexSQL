#!/bin/bash

activate

infer_folder=$1

# Step 1: Execute missing programs and evaluate outputs
# This runs on the original folder structure (needs .py files and sub_queries.json)
# echo "Step 1: Executing missing programs and evaluating outputs..."
# python utils/evaluate_passk.py \
#     --result_dir ${infer_folder} \
#     --gold_dir datasets/Spider2/spider2-snow/evaluation_suite/gold \
#     --mode execute_programs

# Step 2: Collect CSV files to evaluation folder for easy access
echo ""
echo "Step 2: Collecting CSV files to evaluation folder..."
python utils/move_csv_to_evaluate_folder.py \
    --source_dir ${infer_folder} \
    --output ${infer_folder}/evaluation

python utils/evaluate_passk.py \
    --result_dir ${infer_folder}/evaluation \
    --gold_dir datasets/Spider2/spider2-snow/evaluation_suite/gold \
    --mode exec_result
