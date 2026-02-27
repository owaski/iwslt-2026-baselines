#!/usr/bin/env bash

##SBATCH --nodelist=babel-4-23
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --gres=gpu:L40S:1
#SBATCH --partition=general
#SBATCH --time=2-00:00:00
##SBATCH --dependency=afterok:job_id
#SBATCH --array=2,3,4
##SBATCH --account=siqiouya
#SBATCH --mail-type=ALL
#SBATCH --mail-user=siqiouya@andrew.cmu.edu
#SBATCH -e slurm_logs/%A-%a.err
#SBATCH -o slurm_logs/%A-%a.out

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: sbatch infer_baseline.sh <zh|de|it>" >&2
    exit 1
fi

LANG_ARG="$1"

case "$LANG_ARG" in
    zh)
        TGT_LANG=Chinese
        TGT_LANG_CODE=zh
        LATENCY_UNIT=char
        SACREBLEU_TOKENIZER=zh
        ;;
    de)
        TGT_LANG=German
        TGT_LANG_CODE=de
        LATENCY_UNIT=word
        SACREBLEU_TOKENIZER=13a
        ;;
    it)
        TGT_LANG=Italian
        TGT_LANG_CODE=it
        LATENCY_UNIT=word
        SACREBLEU_TOKENIZER=13a
        ;;
    *)
        echo "Error: unsupported language '$LANG_ARG'. Use 'zh', 'de', or 'it'." >&2
        exit 1
        ;;
esac

source /home/siqiouya/miniconda3/bin/activate iwslt2026

SEGMENT_SIZE=$((320 * SLURM_ARRAY_TASK_ID))
HISTORY_UTTERANCES=0
MIN_START_SECONDS=5.0

WAV_LIST_FILE=/data/group_data/li_lab/siqiouya/datasets/mcif-long-trans/mcif.source.simulstream
OUT_DIR=outputs/en-${TGT_LANG_CODE}/baseline/seg${SEGMENT_SIZE}_mss${MIN_START_SECONDS}_h${HISTORY_UTTERANCES}
METRICS_LOG_FILE=${OUT_DIR}/metrics.jsonl
SPEECH_CHUNK_SIZE=$(awk "BEGIN { printf \"%.3f\", ${SEGMENT_SIZE}/1000 }")

mkdir -p "$OUT_DIR"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# Build a run-specific config with the desired chunk size/policy.
cat > "${OUT_DIR}/speech_processor.yaml" <<EOF
type: "agent_simulstream.CascadeSpeechProcessor"
speech_chunk_size: ${SPEECH_CHUNK_SIZE}
latency_unit: "${LATENCY_UNIT}"
detokenizer_type: "simuleval"
asr_model_name: "Qwen/Qwen3-ASR-1.7B"
llm_model_name: "Qwen/Qwen3-4B-Instruct-2507"
source_lang: "English"
target_lang: "${TGT_LANG}"
min_start_seconds: ${MIN_START_SECONDS}
max_history_utterances: ${HISTORY_UTTERANCES}
max_new_tokens: 100
temperature: 0.0
top_p: 0.8
top_k: 20
repetition_penalty: 1.05
abstract_results_path: null
ner_results_path: null
EOF


PYTHONUNBUFFERED=1 uv run simulstream_inference \
    --speech-processor-config "${OUT_DIR}/speech_processor.yaml" \
    --wav-list-file "$WAV_LIST_FILE" \
    --src-lang English \
    --tgt-lang "${TGT_LANG}" \
    --metrics-log-file "$METRICS_LOG_FILE"


RUN_DIR=${OUT_DIR} \
REFERENCE_FILE=/data/group_data/li_lab/siqiouya/datasets/mcif-long-trans/${TGT_LANG_CODE}.txt \
TRANSCRIPT_FILE=/data/group_data/li_lab/siqiouya/datasets/mcif-long-trans/en.txt \
AUDIO_DEFINITION=/data/group_data/li_lab/siqiouya/datasets/mcif-long-trans/audio-segments.yaml \
LATENCY_UNIT=${LATENCY_UNIT} \
SACREBLEU_TOKENIZER=${SACREBLEU_TOKENIZER} \
bash eval.sh