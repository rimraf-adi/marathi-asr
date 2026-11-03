#!/bin/bash
# Fresh full evaluation of all four models. Needs HF_TOKEN in the environment.
cd "$(dirname "$0")"
mkdir -p evals/logs
export PYTHONUTF8=1 KMP_DUPLICATE_LIB_OK=TRUE
for m in ours-moe ours-dense sravaani indic-600m; do
  .venv311/Scripts/python.exe eval_offline.py --model $m > evals/logs/evals_$m.log 2>&1
done
.venv311/Scripts/python.exe spaces/build_dashboard.py > evals/logs/evals_build.log 2>&1
