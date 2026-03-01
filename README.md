# IWSLT 2026 Simultaneous Translation Baseline

Baseline system for the [IWSLT 2026 Simultaneous Translation Track](https://iwslt.org/2026/simultaneous), **Speech-to-Text with Extra Context** subtrack.

This system implements a cascade-based streaming pipeline: **ASR** (Qwen3-ASR-1.7B) → **Translation** (Qwen3-4B-Instruct-2507), with optional context from paper. It evaluates quality-latency tradeoffs for English → Chinese / German / Italian.

Tested on the MCIF dataset, providing additional context consistently improves translation quality across all language pairs and latencies compared to the baseline.

![Quality-Latency Tradeoff](quality_latency_tradeoff.png)

## Environment Setup

Two conda environments are needed (separate due to dependency conflicts between inference and evaluation).

```bash
# Inference environment
conda create -y -n inference python=3.12
conda activate inference
pip install uv
uv pip install qwen-asr[vllm] simulstream simuleval pymupdf-layout pymupdf4llm jupyter nvitop pycryptodome mweralign

# Evaluation environment
conda create -y -n evaluation python=3.12
conda activate evaluation
pip install uv
# if you encounter evaluation error for CJK languages, try install simulstream from source
uv pip install simulstream[eval] setuptools==80.10.2
```

## Pipeline

The full pipeline has four stages:

```
PDFs → extract_abstract.py → ner_llm.py → infer.sh → eval.sh
```

### Step 1: Extract Abstracts from PDFs

Extract title, authors, and abstract text from each paper PDF.

```bash
conda activate inference
python extract_abstract.py data/pdf_paths.txt -o data/abstract_results.json
```

- Input: a text file with one PDF path per line
- Output: JSON array of `{"path": ..., "abstract": ...}` entries

### Step 2: Extract Named Entities

Use LLM-based majority voting to extract named entities from the abstracts. Samples N times per paper and keeps entities appearing in at least k samples for robustness.

```bash
conda activate inference
python ner_llm.py data/abstract_results.json -n 16 -k 8 -o data/ner_llm_results.json
```

- `-n` — number of LLM samples per paper (default: 16)
- `-k` — minimum sample count to keep an entity (default: 8)
- Output: JSON array of `{"path": ..., "entity_count": ..., "entities": [...]}` entries
- Requires a GPU to run the NER model (Qwen3-30B-A3B-Instruct-2507-FP8 via vLLM)

### Step 3: Run Simultaneous Translation Inference

Configure via environment variables and run [infer.sh](infer.sh):

```bash
WAV_LIST_FILE=/path/to/wav_list \
TGT_LANG=Chinese \
TGT_LANG_CODE=zh \
LATENCY_UNIT=char \
SEGMENT_SIZE=960 \
APPROACH=with-context \
NER_RESULTS_PATH=data/ner_llm_results.json \
bash infer.sh

WAV_LIST_FILE=/data/group_data/li_lab/siqiouya/datasets/mcif-long-trans/mcif.source.simulstream \
TGT_LANG=Chinese \
TGT_LANG_CODE=zh \
LATENCY_UNIT=char \
SEGMENT_SIZE=640 \
APPROACH=with-context \
NER_RESULTS_PATH=data/ner_llm_results.json \
bash infer.sh
```

**Output structure:**

```
outputs/en-{zh,de,it}/{baseline,with-context}/seg{640,960,1280}_mss5.0_h0/
├── speech_processor.yaml   # full config for reproducibility
└── metrics.jsonl           # streaming inference log
```

### Step 4: Evaluation

Evaluation computes StreamLAAL (latency), SacreBLEU, COMET (XCOMET-XL), and stats (normalized erasure, real-time factor). Configure via environment variables and run [eval.sh](eval.sh):

```bash
RUN_DIR=outputs/en-zh/baseline/seg960_mss5.0_h0 \
REFERENCE_FILE=/path/to/reference.txt \
TRANSCRIPT_FILE=/path/to/source_transcript.txt \
AUDIO_DEFINITION=/path/to/audio-segments.yaml \
LATENCY_UNIT=char \
SACREBLEU_TOKENIZER=zh \
bash eval.sh
```

Results are written to `${RUN_DIR}/eval.txt`.

### Visualization

Plot quality-latency tradeoff curves across configurations:

```bash
python plot_tradeoff.py
# Generates: quality_latency_tradeoff.png
```

## System Architecture

The core streaming processor is `CascadeSpeechProcessor` in [agent_simulstream.py](agent_simulstream.py). It processes audio chunks incrementally:

1. **ASR**: Accumulates audio and runs Qwen3-ASR-1.7B, optionally injecting named entities as context. Use punctuation and Qwen3-ForcedAligner-0.6B to separate utterances.
2. **Translation**: Translates ASR output using Qwen3-4B-Instruct-2507 with [local agreement policy](https://www.isca-archive.org/interspeech_2020/liu20s_interspeech.pdf).

This method is inspired by [StreamUni](https://arxiv.org/pdf/2507.07803).

Key configuration knobs in `speech_processor.yaml`:

- `speech_chunk_size` — chunk duration in seconds (controls latency-quality tradeoff)
- `min_start_seconds` — audio buffer before first translation attempt
- `ner_results_path` — path to NER JSON for additional context injection (`null` for baseline)

