set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate evaluation

# Override RUN_DIR to evaluate a different experiment directory.
RUN_DIR="${RUN_DIR:-/home/siqiouya/code/iwslt_2026/outputs/baseline/seg960_mss5.0_h0}"
EVAL_CONFIG="${EVAL_CONFIG:-${RUN_DIR}/speech_processor.yaml}"
LOG_FILE="${LOG_FILE:-${RUN_DIR}/metrics.jsonl}"

REFERENCE_FILE="${REFERENCE_FILE:-/data/group_data/li_lab/siqiouya/datasets/mcif-long-trans/zh.txt}"
TRANSCRIPT_FILE="${TRANSCRIPT_FILE:-/data/group_data/li_lab/siqiouya/datasets/mcif-long-trans/en.txt}"
AUDIO_DEFINITION="${AUDIO_DEFINITION:-/data/group_data/li_lab/siqiouya/datasets/mcif-long-trans/audio-segments.yaml}"
LATENCY_UNIT="${LATENCY_UNIT:-char}"
SACREBLEU_TOKENIZER="${SACREBLEU_TOKENIZER:-zh}"
MOSES_TOKENIZER="${MOSES_TOKENIZER:-zh}"

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

CHAR_LEVEL_FLAG=""
if [[ "$LATENCY_UNIT" == "char" ]]; then
    CHAR_LEVEL_FLAG="--char_level"
fi

omnisteval longform \
  --speech_segmentation "$AUDIO_DEFINITION" \
  --source_sentences_file "$TRANSCRIPT_FILE" \
  --ref_sentences_file "$REFERENCE_FILE" \
  --hypothesis_file "$LOG_FILE" \
  --simulstream_config_file "$EVAL_CONFIG" \
  --hypothesis_format simulstream \
  --comet \
  --comet_model Unbabel/XCOMET-XL \
  --lang "${MOSES_TOKENIZER}" \
  $CHAR_LEVEL_FLAG \
  --bleu_tokenizer "$SACREBLEU_TOKENIZER" \
  --output_folder "$RUN_DIR/segmentation_output"