import json
import logging
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import List

import numpy as np
import torch
from openai import OpenAI
from qwen_asr import Qwen3ASRModel
from simulstream.server.speech_processors import SAMPLE_RATE, SpeechProcessor
from simulstream.server.speech_processors.incremental_output import IncrementalOutput
from vllm import LLM, SamplingParams

# Keep logs quiet while serving/benchmarking.
# logging.getLogger().setLevel(logging.ERROR)
# Ensure the simulstream metrics logger can still write at INFO level.
logging.getLogger("fbk_fairseq.simultaneous.metrics").setLevel(logging.INFO)


def longest_common_prefix(s1: str, s2: str) -> str:
    for i in range(min(len(s1), len(s2))):
        if s1[i] != s2[i]:
            return s1[:i]
    return s1[: min(len(s1), len(s2))]


def find_end_time(time_stamps, position: int, text: str) -> float:
    # Find the largest timestamp whose word position is <= position.
    text_no_punct = (text[:position] + text[position + 1:]).strip()
    if len(time_stamps) != len(text_no_punct.split()):
        print(f"number of time stamps and words in text do not match\ntime_stamps: {time_stamps}\ntext: {text.split()}")
        return None
    n_words_right = len(text[position + 1:].strip().split())
    return time_stamps[-n_words_right - 1].end_time


@dataclass
class CascadeState:
    speech_id: int = 0
    source: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    utt_timestamps: List[int] = field(default_factory=lambda: [0])
    utt_sources: List[str] = field(default_factory=lambda: [""])
    utt_targets: List[str] = field(default_factory=lambda: [""])
    asr_hypotheses: List[str] = field(default_factory=lambda: [""])
    translation_hypotheses: List[str] = field(default_factory=lambda: [""])
    translations: List[str] = field(default_factory=lambda: [""])
    emission_started: bool = False


class CascadeSpeechProcessor(SpeechProcessor):
    """
    SimulStream speech processor version of the original SimulEval cascade agent.

    Required config keys:
    - asr_model_name
    - llm_model_name

    Optional config keys:
    - source_lang (default: English)
    - target_lang (default: Chinese)
    - min_start_seconds (default: 2.0)
    - max_history_utterances (default: 0)
    - max_new_tokens (default: 4096)
    - temperature (default: 1.0)
    - top_p (default: 0.9)
    - top_k (default: 20)
    - repetition_penalty (default: 1.0)
    - abstract_results_path (default: null)
    - ner_results_path (default: null)
    - latency_unit (default: word)
    """

    @classmethod
    def load_model(cls, config: SimpleNamespace):
        if not hasattr(cls, "asr") or cls.asr is None:
            cls.asr = Qwen3ASRModel.LLM(
                model=config.asr_model_name,
                gpu_memory_utilization=0.2,
                max_inference_batch_size=1,
                max_model_len=1024,
                max_new_tokens=1024,
                forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
                forced_aligner_kwargs=dict(
                    dtype=torch.bfloat16,
                    device_map="cuda",
                ),
            )
        llm_base_url = getattr(config, "llm_base_url", None)
        if llm_base_url is not None:
            """
            vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
                --trust-remote-code \
                --gpu-memory-utilization 0.75 \
                --tensor-parallel-size 1 \
                --max-num-seqs 1 \
                --max-model-len 1024 \
                --enable-prefix-caching
            """
            # Use a remote vLLM server via OpenAI-compatible API.
            if not hasattr(cls, "llm_client") or cls.llm_client is None:
                cls.llm_client = OpenAI(base_url=llm_base_url, api_key="EMPTY")
                from transformers import AutoTokenizer
                cls.tokenizer = AutoTokenizer.from_pretrained(config.llm_model_name)
            cls.llm = None
        else:
            cls.llm_client = None
            if not hasattr(cls, "llm") or cls.llm is None:
                cls.llm = LLM(
                    model=config.llm_model_name,
                    trust_remote_code=True,
                    gpu_memory_utilization=0.75,
                    tensor_parallel_size=1,
                    max_num_seqs=1,
                    max_model_len=1024,
                    enable_prefix_caching=True,
                )
                cls.tokenizer = cls.llm.get_tokenizer()

    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.load_model(config)

        self.source_lang = getattr(config, "source_lang", "English")
        self.target_lang = getattr(config, "target_lang", "Chinese")
        self.target_sep = "" if self.target_lang in ["Chinese", "Japanese"] else " "
        self.latency_unit = getattr(config, "latency_unit", "word")

        self.min_start_seconds = getattr(config, "min_start_seconds", 2.0)
        self.max_history_utterances = getattr(config, "max_history_utterances", 0)

        self._temperature = getattr(config, "temperature", 1.0)
        self._top_p = getattr(config, "top_p", 0.9)
        self._top_k = getattr(config, "top_k", 20)
        self._max_tokens = getattr(config, "max_new_tokens", 512)
        self._repetition_penalty = getattr(config, "repetition_penalty", 1.05)
        self._llm_model_name = config.llm_model_name

        self.sampling_params = SamplingParams(
            temperature=self._temperature,
            top_p=self._top_p,
            top_k=self._top_k,
            max_tokens=self._max_tokens,
            repetition_penalty=self._repetition_penalty,
            stop=["\n"],
        )

        abstract_results_path = getattr(config, "abstract_results_path", None)
        self.abstract_results = (
            self._load_abstract_results(abstract_results_path)
            if abstract_results_path is not None
            else None
        )

        ner_results_path = getattr(config, "ner_results_path", None)
        self.ner_results = (
            self._load_ner_results(ner_results_path)
            if ner_results_path is not None
            else None
        )

        self._state = CascadeState()

    @staticmethod
    def _load_abstract_results(abstract_results_path: str) -> List[str]:
        with open(abstract_results_path, "r", encoding="utf-8") as f:
            abstract_results = json.load(f)
        return [result["abstract"] for result in abstract_results]

    @staticmethod
    def _load_ner_results(ner_results_path: str) -> List[str]:
        with open(ner_results_path, "r", encoding="utf-8") as f:
            ner_results = json.load(f)
        return [", ".join(result["entities"]) for result in ner_results]

    @staticmethod
    def _n_utterances(text: str) -> int:
        return text.count(".") + text.count("!") + text.count("?")

    def _transcribe_audio(self, state: CascadeState):
        audio = np.array(state.source[state.utt_timestamps[-1-self.max_history_utterances]:])
        if self.ner_results is not None:
            asr_context = self.ner_results[state.speech_id]
        elif self.abstract_results is not None:
            asr_context = self.abstract_results[state.speech_id]
        else:
            asr_context = ""

        asr_outputs = self.asr.transcribe(
            (audio, SAMPLE_RATE),
            language=self.source_lang,
            context=asr_context,
            return_time_stamps=True,
        )

        print(f"asr_hypo: {asr_outputs[0].text}")
        if asr_outputs[0].time_stamps is not None and \
            asr_outputs[0].time_stamps[-1].end_time > len(audio) / SAMPLE_RATE:
            print(f"skipping because ASR output time stamp is longer than audio length: {asr_outputs[0].time_stamps[-1].end_time} > {len(audio) / SAMPLE_RATE}")
            return None, False
        
        asr_hypo = asr_outputs[0].text
        state.asr_hypotheses.append(asr_hypo)

        asr_segment = longest_common_prefix(state.asr_hypotheses[-2], state.asr_hypotheses[-1])
        if self._n_utterances(asr_segment) >= 1:
            rightest_punct_idx = max(
                asr_segment.rfind("."),
                asr_segment.rfind("!"),
                asr_segment.rfind("?"),
            )
            find_end_time_result = find_end_time(asr_outputs[0].time_stamps, rightest_punct_idx, asr_hypo)
            if find_end_time_result is None:
                return None, False
            utt_end_time = int(find_end_time_result * SAMPLE_RATE) + state.utt_timestamps[-1]
            utt_end_time = min(utt_end_time, len(state.source))
            state.utt_timestamps.append(utt_end_time)
            state.utt_sources.append(asr_segment[: rightest_punct_idx + 1])
            state.asr_hypotheses = [asr_hypo[rightest_punct_idx + 1:].strip()]
            asr_to_translate = " ".join(state.utt_sources[-1 - self.max_history_utterances:])
            return asr_to_translate, True

        if self.max_history_utterances > 0:
            asr_to_translate = " ".join(
                state.utt_sources[-self.max_history_utterances:] + [asr_hypo]
            )
        else:
            asr_to_translate = asr_hypo
        return asr_to_translate, False

    def _prepare_llm_inputs(self, asr_segment: str, prev_translation: str, context: str) -> str:
        context_prompt = (
            f"""

[CONTEXT]

{context}"""
            if context != ""
            else ""
        )

        instruction = f"""You are a professional translator.

[TASK]
Translate the input text into {self.target_lang}. 
Preserve all named entities, such as person names, model names, dataset names, and metric names, exactly as they appear in the input text. {"\nUse the provided context only to correctly resolve named entities." if context != "" else ""}
Return only the translated text without any additional explanation.{context_prompt}

[INPUT]
{asr_segment}"""

        messages = [{"role": "user", "content": instruction}]
        text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        text += prev_translation
        return text

    def _llm_generate(self, prompt: str) -> str:
        """Generate translation using either local vLLM or remote vLLM server."""
        if self.llm_client is not None:
            response = self.llm_client.completions.create(
                model=self._llm_model_name,
                prompt=prompt,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                top_p=self._top_p,
                stop=["\n"],
                extra_body={"repetition_penalty": self._repetition_penalty},
            )
            return response.choices[0].text.replace("…", "")
        else:
            llm_outputs = self.llm.generate(
                [prompt],
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )
            return llm_outputs[0].outputs[0].text.replace("…", "")

    def _translate_segment(self, state: CascadeState, asr_segment: str, utt_finished: bool) -> str:
        if asr_segment == "":
            return ""

        prefix = (
            ""
            if self.max_history_utterances == 0
            else self.target_sep.join(state.utt_targets[-self.max_history_utterances:]) + self.target_sep
        )
        prev_translation = prefix + state.translations[-1]
        # llm_context = self.abstract_results[state.speech_id] if self.abstract_results is not None else ""
        llm_context = ""
        llm_inputs = self._prepare_llm_inputs(
            asr_segment,
            prev_translation,
            llm_context,
        )
        hypothesis = self._llm_generate(llm_inputs)

        if utt_finished:
            state.utt_targets.append(state.translations[-1] + hypothesis)
            state.translations = [""]
            state.translation_hypotheses = [""]
            if self.target_lang not in ["Chinese", "Japanese"]:
                hypothesis = hypothesis.strip()
            return hypothesis

        full_hypothesis = state.translations[-1] + hypothesis
        # print(f"full_hypothesis: {full_hypothesis}")
        state.translation_hypotheses.append(full_hypothesis)
        translation = longest_common_prefix(
            state.translation_hypotheses[-2],
            state.translation_hypotheses[-1],
        )
        translation_increment = translation[len(state.translations[-1]):]
        state.translations.append(translation)

        if self.target_lang not in ["Chinese", "Japanese"]:
            translation_increment = translation_increment.strip()
        return translation_increment

    def _text_to_tokens(self, text: str) -> List[str]:
        if text == "":
            return []
        if self.latency_unit in ["word", "spm"]:
            return text.strip().split()
        if self.latency_unit == "char":
            return list(text.strip())
        raise NotImplementedError(f"Unsupported latency_unit: {self.latency_unit}")

    def _build_incremental_output(self, text: str) -> IncrementalOutput:
        if text == "":
            return IncrementalOutput([], "", [], "")

        out_text = text
        if (
            self.latency_unit == "word"
            and self._state.emission_started
            and not out_text.startswith(" ")
        ):
            out_text = " " + out_text
        self._state.emission_started = True

        return IncrementalOutput(
            new_tokens=self._text_to_tokens(text),
            new_string=out_text,
            deleted_tokens=[],
            deleted_string="",
        )

    @torch.inference_mode()
    def process_chunk(self, waveform: np.float32) -> IncrementalOutput:
        if waveform is None or len(waveform) == 0:
            return IncrementalOutput([], "", [], "")

        self._state.source = np.concatenate(
            [self._state.source, np.asarray(waveform, dtype=np.float32)]
        )
        source_duration = len(self._state.source) / SAMPLE_RATE
        if source_duration < self.min_start_seconds:
            return IncrementalOutput([], "", [], "")

        asr_segment, utt_finished = self._transcribe_audio(self._state)
        if asr_segment is None:
            return IncrementalOutput([], "", [], "")
        translation = self._translate_segment(self._state, asr_segment, utt_finished)
        return self._build_incremental_output(translation)

    @torch.inference_mode()
    def end_of_stream(self) -> IncrementalOutput:
        translation = ""
        if len(self._state.source) > 0:
            asr_segment, utt_finished = self._transcribe_audio(self._state)
            if asr_segment is None:
                return IncrementalOutput([], "", [], "")
            translation = self._translate_segment(self._state, asr_segment, utt_finished)

            # If nothing was emitted, flush the remaining ASR hypothesis once.
            if translation == "" and self._state.asr_hypotheses[-1].strip() != "":
                trailing_asr = self._state.asr_hypotheses[-1].strip()
                if self.max_history_utterances > 0:
                    trailing_asr = " ".join(
                        self._state.utt_sources[-self.max_history_utterances:] + [trailing_asr]
                    )
                translation = self._translate_segment(self._state, trailing_asr, True)

        self._state.speech_id += 1
        return self._build_incremental_output(translation)

    def set_source_language(self, language: str) -> None:
        self.source_lang = language

    def set_target_language(self, language: str) -> None:
        self.target_lang = language
        self.target_sep = "" if language in ["Chinese", "Japanese"] else " "

    def tokens_to_string(self, tokens: List[str]) -> str:
        if self.latency_unit in ["word", "spm"]:
            return " ".join(tokens)
        if self.latency_unit == "char":
            return "".join(tokens)
        raise NotImplementedError(f"Unsupported latency_unit: {self.latency_unit}")

    def clear(self) -> None:
        self._state = CascadeState(speech_id=self._state.speech_id)