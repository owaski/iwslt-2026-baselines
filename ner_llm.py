"""
Extract named entities from paper (title, authors, abstract) using
Qwen3-30B-A3B-Instruct via vLLM.

Samples N times per paper (default 16) and keeps entities appearing
in at least half the samples (default 8) for robustness.

Usage:
    python ner_llm.py data/abstract_results.json -o data/ner_llm_results.json
    python ner_llm.py data/abstract_results.json -n 16 -k 8 -o data/ner_llm_results.json
"""
import os
import json
import argparse
import logging
from collections import Counter
from tqdm import tqdm

from vllm import LLM, SamplingParams

logging.getLogger().setLevel(logging.ERROR)

NER_MODEL = "/data/user_data/siqiouya/ckpts/pretrained/llm/Qwen3-30B-A3B-Instruct-2507-FP8/"

SYSTEM_PROMPT = """\
You are a named entity extraction system. Given a paper's title, authors, and abstract, \
extract all named entities. Include: person names, organization/affiliation names, \
location names, model/system names, method/algorithm names, dataset names, task names, \
and any other proper nouns or technical terms that a speech recognition system might struggle with.

Return ONLY a JSON array of strings, one per entity. No duplicates. No explanation.\
"""

USER_TEMPLATE = """\
Extract all named entities from the following paper metadata.

{text}

Return a JSON array of unique entity strings. Output ONLY the JSON array, nothing else."""


def build_prompts(results, tokenizer):
    """Build chat-formatted prompts from abstract results."""
    texts = []
    for data in results:
        abstract = data.get("abstract", "").strip()
        if not abstract:
            texts.append(None)
            continue

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(text=abstract)},
        ]
        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        texts.append(text)
    return texts


def parse_entity_list(response):
    """Parse a JSON array of strings from model response."""
    # Find the JSON array in the response (may be wrapped in thinking tags or markdown)
    text = response.strip()

    # Strip markdown code fences if present
    if "```" in text:
        start = text.find("```")
        end = text.rfind("```")
        if start != end:
            inner = text[start:end + 3]
            # Remove the fences
            lines = inner.split("\n")
            lines = lines[1:-1]  # drop first ``` line and last ```
            text = "\n".join(lines)

    # Find first [ and last ]
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        return []

    try:
        entities = json.loads(text[start:end + 1])
        if isinstance(entities, list):
            # Deduplicate while preserving order
            seen = set()
            unique = []
            for e in entities:
                s = str(e).strip()
                if s and s not in seen:
                    seen.add(s)
                    unique.append(s)
            return unique
    except json.JSONDecodeError:
        pass
    return []


def main():
    parser = argparse.ArgumentParser(
        description="Extract named entities from paper abstracts using Qwen3 via vLLM"
    )
    parser.add_argument(
        "input",
        help="Path to abstract_results.json (from extract_abstract.py)",
    )
    parser.add_argument(
        "-o", "--output",
        default="data/ner_llm_results.json",
        help="Output JSON file path (default: data/ner_llm_results.json)",
    )
    parser.add_argument(
        "-n", "--num-samples",
        type=int, default=16,
        help="Number of samples per paper (default: 16)",
    )
    parser.add_argument(
        "-k", "--min-count",
        type=int, default=8,
        help="Minimum number of samples an entity must appear in to be kept (default: 8)",
    )
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        results = json.load(f)

    print(f"Loaded {len(results)} papers from {args.input}")
    print(f"Loading {NER_MODEL} ...")

    llm = LLM(
        model=NER_MODEL,
        trust_remote_code=True,
        gpu_memory_utilization=0.8,
        tensor_parallel_size=1,
        max_model_len=4096,
    )
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_tokens=2048,
        n=args.num_samples,
    )

    # Build prompts
    prompts = build_prompts(results, tokenizer)

    # Collect valid indices and their prompts for batched generation
    valid_indices = [i for i, p in enumerate(prompts) if p is not None]
    valid_prompts = [prompts[i] for i in valid_indices]

    print(f"Generating NER for {len(valid_prompts)} papers "
          f"({args.num_samples} samples each, keeping entities with >= {args.min_count} hits) ...")
    outputs = llm.generate(valid_prompts, sampling_params=sampling_params, use_tqdm=True)

    # Map outputs back: each output has n samples
    output_map = {}
    for idx, out in zip(valid_indices, outputs):
        output_map[idx] = [o.text for o in out.outputs]

    # Build final results with majority-vote filtering
    final = []
    for i, data in enumerate(results):
        responses = output_map.get(i, [])
        # Count how many samples each entity appears in
        entity_counter = Counter()
        for response in responses:
            sample_entities = parse_entity_list(response)
            # Each entity counts once per sample (use set)
            for e in set(sample_entities):
                entity_counter[e] += 1

        # Keep entities appearing in >= min_count samples
        entities = [e for e, c in entity_counter.most_common() if c >= args.min_count]
        final.append({
            "path": data["path"],
            "entity_count": len(entities),
            "entities": entities,
        })

    # Save
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Results saved to: {args.output}")

    # Summary
    total = sum(d["entity_count"] for d in final)
    print(f"\nTotal entities extracted: {total} "
          f"(n={args.num_samples}, min_count={args.min_count})")
    for d in final:
        print(f"  {os.path.basename(d['path'])}: {d['entity_count']} entities")


if __name__ == "__main__":
    main()
