"""LLM backend abstraction supporting Transformers and Ollama."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import torch
from ollama import Client as OllamaClient
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import SETTINGS


@dataclass(slots=True)
class LLMResponse:
    """Container for LLM-generated content."""

    text: str
    metadata: dict[str, Any]


class LLMClient:
    """Unified LLM interface for generation calls."""

    def __init__(self) -> None:
        self.backend = SETTINGS.model.llm_backend
        self._tokenizer = None
        self._model = None
        self._ollama = None

    def _init_transformers(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        kwargs: dict[str, Any] = {
            "torch_dtype": SETTINGS.torch_dtype,
            "device_map": "auto" if SETTINGS.device == "cuda" else None,
        }

        if SETTINGS.device == "cuda" and SETTINGS.model.use_4bit:
            try:
                from transformers import BitsAndBytesConfig

                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
            except Exception:
                # Fall back to normal loading if bitsandbytes is unavailable.
                pass

        self._tokenizer = AutoTokenizer.from_pretrained(SETTINGS.model.hf_model_name)
        self._model = AutoModelForCausalLM.from_pretrained(SETTINGS.model.hf_model_name, **kwargs)
        self._model.eval()

    def _init_ollama(self) -> None:
        if self._ollama is None:
            self._ollama = OllamaClient()

    def generate(self, prompt: str, max_new_tokens: int | None = None, temperature: float | None = None) -> LLMResponse:
        """Generate a text completion from the configured backend."""
        if self.backend == "mock":
            return self._mock_generate(prompt)

        if self.backend == "ollama":
            self._init_ollama()
            response = self._ollama.chat(
                model=SETTINGS.model.ollama_model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": SETTINGS.model.temperature if temperature is None else temperature,
                    "num_predict": SETTINGS.model.max_new_tokens if max_new_tokens is None else max_new_tokens,
                },
            )
            return LLMResponse(text=response["message"]["content"], metadata={"backend": "ollama"})

        self._init_transformers()
        assert self._tokenizer is not None and self._model is not None
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True)
        if SETTINGS.device == "cuda":
            inputs = {key: value.to("cuda") for key, value in inputs.items()}

        outputs = self._model.generate(
            **inputs,
            max_new_tokens=SETTINGS.model.max_new_tokens if max_new_tokens is None else max_new_tokens,
            temperature=SETTINGS.model.temperature if temperature is None else temperature,
            do_sample=True,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        generated = outputs[0][inputs["input_ids"].shape[1] :]
        text = self._tokenizer.decode(generated, skip_special_tokens=True)
        return LLMResponse(text=text, metadata={"backend": "transformers"})

    def _mock_generate(self, prompt: str) -> LLMResponse:
        """Return deterministic JSON-like responses for fast smoke tests."""
        low = prompt.lower()
        if "sub_questions" in low or "qa planning agent" in low:
            text = (
                '{"sub_questions": ["Identify the first entity in the question", '
                '"Find the relationship needed to answer"], "relation_sequence": ["related_to", "related_to"]}'
            )
        elif "react reasoner" in low:
            text = (
                '{"thought": "Using highest-scoring evidence hop.", '
                '"answer": "Mock answer based on retrieved evidence.", "confidence": 0.72}'
            )
        elif "critic agent" in low:
            text = '{"approved": true, "critique": "Sufficient evidence in chain.", "confidence": 0.78}'
        else:
            text = "{}"
        return LLMResponse(text=text, metadata={"backend": "mock"})

    @staticmethod
    def extract_json(raw_text: str) -> dict[str, Any]:
        """Extract and parse a JSON object from model output text."""
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        try:
            return json.loads(raw_text[start : end + 1])
        except json.JSONDecodeError:
            return {}
