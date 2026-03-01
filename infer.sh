set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate inference

# each line is a relative wav file path
# e.g. audio/OiqEWDVtWk.wav
# if the file is under mcif-long-trans/
WAV_LIST_FILE="${WAV_LIST_FILE:-/path/to/wav_list}"

TGT_LANG="${TGT_LANG:-Chinese}"                          # Chinese, German, or Italian
TGT_LANG_CODE="${TGT_LANG_CODE:-zh}"                     # zh, de, or it
LATENCY_UNIT="${LATENCY_UNIT:-char}"                      # char for zh, word for de/it
SEGMENT_SIZE="${SEGMENT_SIZE:-960}"                       # 640, 960, or 1280 (ms), or other values
APPROACH="${APPROACH:-with-context}"                      # baseline or with-context
NER_RESULTS_PATH="${NER_RESULTS_PATH:-data/ner_llm_results.json}" # NER JSON for with-context approach, null for baseline

# === Derived variables (no need to change) ===
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
SPEECH_CHUNK_SIZE=$(awk "BEGIN { printf \"%.3f\", ${SEGMENT_SIZE}/1000 }")
OUT_DIR=outputs/en-${TGT_LANG_CODE}/${APPROACH}/seg${SEGMENT_SIZE}_mss5.0_h0
mkdir -p "$OUT_DIR"

# Write the config
cat > "${OUT_DIR}/speech_processor.yaml" <<EOF
type: "agent_simulstream.CascadeSpeechProcessor"
speech_chunk_size: ${SPEECH_CHUNK_SIZE}
latency_unit: "${LATENCY_UNIT}"
detokenizer_type: "simuleval"
asr_model_name: "Qwen/Qwen3-ASR-1.7B"
llm_model_name: "Qwen/Qwen3-4B-Instruct-2507"
source_lang: "English"
target_lang: "${TGT_LANG}"
min_start_seconds: 5.0
max_history_utterances: 0
max_new_tokens: 100
temperature: 0.0
repetition_penalty: 1.05
ner_results_path: ${NER_RESULTS_PATH}
EOF

# Run inference
PYTHONUNBUFFERED=1 uv run simulstream_inference \
    --speech-processor-config "${OUT_DIR}/speech_processor.yaml" \
    --wav-list-file "$WAV_LIST_FILE" \
    --src-lang English \
    --tgt-lang "${TGT_LANG}" \
    --metrics-log-file "${OUT_DIR}/metrics.jsonl"
