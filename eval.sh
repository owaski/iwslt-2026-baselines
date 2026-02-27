#!/usr/bin/env bash

# RUN_DIR=outputs/en-de/baseline/seg640_mss5.0_h0/ REFERENCE_FILE=/data/user_data/siqiouya/datasets/acl_6060/dev/text/txt/ACL.6060.dev.en-xx.de.txt LATENCY_UNIT=word SACREBLEU_TOKENIZER=13a bash eval.sh

set -euo pipefail

source /home/siqiouya/miniconda3/bin/activate simulstream_eval

# Override RUN_DIR to evaluate a different experiment directory.
RUN_DIR="${RUN_DIR:-/home/siqiouya/code/iwslt_2026/outputs/baseline/seg960_mss5.0_h0}"
EVAL_CONFIG="${EVAL_CONFIG:-${RUN_DIR}/speech_processor.yaml}"
LOG_FILE="${LOG_FILE:-${RUN_DIR}/metrics.jsonl}"

REFERENCE_FILE="${REFERENCE_FILE:-/data/group_data/li_lab/siqiouya/datasets/mcif-long-trans/zh.txt}"
TRANSCRIPT_FILE="${TRANSCRIPT_FILE:-/data/group_data/li_lab/siqiouya/datasets/mcif-long-trans/en.txt}"
AUDIO_DEFINITION="${AUDIO_DEFINITION:-/data/group_data/li_lab/siqiouya/datasets/mcif-long-trans/audio-segments.yaml}"
LATENCY_UNIT="${LATENCY_UNIT:-char}"
SACREBLEU_TOKENIZER="${SACREBLEU_TOKENIZER:-zh}"

if [[ ! -f "$EVAL_CONFIG" ]]; then
    echo "Missing eval config: $EVAL_CONFIG"
    exit 1
fi
if [[ ! -f "$LOG_FILE" ]]; then
    echo "Missing log file: $LOG_FILE"
    exit 1
fi

EVAL_OUT="${RUN_DIR}/eval.txt"

echo "Evaluating run: $RUN_DIR"
echo "Using config: $EVAL_CONFIG"
echo "Using metrics log: $LOG_FILE"
echo "Writing eval output to: $EVAL_OUT"

exec > >(tee "$EVAL_OUT")

PYTHONUNBUFFERED=1 uv run simulstream_score_latency \
    --scorer stream_laal \
    --eval-config "$EVAL_CONFIG" \
    --log-file "$LOG_FILE" \
    --reference "$REFERENCE_FILE" \
    --audio-definition "$AUDIO_DEFINITION" \
    --latency-unit "$LATENCY_UNIT"

PYTHONUNBUFFERED=1 uv run simulstream_score_quality \
    --scorer sacrebleu \
    --tokenizer "$SACREBLEU_TOKENIZER" \
    --eval-config "$EVAL_CONFIG" \
    --log-file "$LOG_FILE" \
    --references "$REFERENCE_FILE" \
    --transcripts "$TRANSCRIPT_FILE" \
    --audio-definition "$AUDIO_DEFINITION" \
    --latency-unit "$LATENCY_UNIT"

PYTHONUNBUFFERED=1 uv run simulstream_score_quality \
    --scorer comet \
    --model Unbabel/XCOMET-XL \
    --batch-size 8 \
    --eval-config "$EVAL_CONFIG" \
    --log-file "$LOG_FILE" \
    --references "$REFERENCE_FILE" \
    --transcripts "$TRANSCRIPT_FILE" \
    --audio-definition "$AUDIO_DEFINITION" \
    --latency-unit "$LATENCY_UNIT"

PYTHONUNBUFFERED=1 uv run simulstream_stats \
    --eval-config "$EVAL_CONFIG" \
    --log-file "$LOG_FILE" \
    --latency-unit "$LATENCY_UNIT"
