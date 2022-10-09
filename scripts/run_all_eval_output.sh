#!/bin/bash
# Run eval output routine
# run run_add_missing_eval_to_existing_pred_calcmetrics_v2.sh first to calculate metrics and output them into the eval_metrics/josn files in respective log directories

cd ../code

echo "Running evaluation output to txt files..."

python eval_metrics.py --eval_set default


echo Finished!

