#!/bin/bash
export DEEPSEEK_API_KEY="sk-33a98b12f11f4300857e5ca93bf90e24"
export DEEPSEEK_MODEL="deepseek-chat"
export DEEPSEEK_BASE_URL="https://api.deepseek.com/v1/chat/completions"

python make_tasks.py topics_CHGTGph_expanded_flirty.json tasks.tsv

mkdir -p logs out
cat tasks.tsv | parallel -j 20 --joblog logs/run.log --colsep '\t' \
'~/mindseek_seedgen/.venv/bin/python generate_subtopic.py --domain {1} --subtopic {2} --target {3} --out {4}'
