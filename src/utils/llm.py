"""LLM backend abstraction supporting Transformers and Ollama."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import torch

from src.config import SETTINGS


def ollama_healthcheck() -> tuple[bool, str]:
    """Return whether the local (or ``OLLAMA_HOST``) Ollama server responds.

    Uses a lightweight ``list()`` call; does not pull models.
    """
    try:
        OllamaClient().list()
    except Exception as exc:  # noqa: BLE001 — surface any connectivity failure
        host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
        return False, f"{type(exc).__name__}: {exc} (host: {host})"
    return True, ""


@dataclass(slots=True)
class LLMResponse:
    """Container for LLM-generated content."""

    text: str
    metadata: dict[str, Any]


class LLMClient:
    """Unified LLM interface for generation calls."""

    def __init__(self) -> None:
        self._tokenizer = None
        self._model = None
        self._ollama = None
        self._azure_client = None

    def _init_transformers(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

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
            from ollama import Client as OllamaClient

            self._ollama = OllamaClient()

    def _init_azure_openai(self) -> None:
        if self._azure_client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "Azure OpenAI backend requires the openai Python package. Install it with `pip install openai`."
                ) from exc

            kwargs: dict[str, str] = {}
            if SETTINGS.model.azure_api_key:
                kwargs["api_key"] = SETTINGS.model.azure_api_key
            if SETTINGS.model.azure_endpoint:
                kwargs["base_url"] = SETTINGS.model.azure_endpoint
            self._azure_client = OpenAI(**kwargs)

    def generate(self, prompt: str, max_new_tokens: int | None = None, temperature: float | None = None) -> LLMResponse:
        """Generate a text completion from the configured backend."""
        backend = SETTINGS.model.llm_backend
        if backend == "mock":
            return self._mock_generate(prompt)

        if backend == "ollama":
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

        if backend == "azure":
            self._init_azure_openai()
            response = self._azure_client.chat.completions.create(
                model=SETTINGS.model.azure_chat_deployment,
                messages=[{"role": "user", "content": prompt}],
                temperature=SETTINGS.model.temperature if temperature is None else temperature,
                max_completion_tokens=SETTINGS.model.max_new_tokens if max_new_tokens is None else max_new_tokens,
            )
            choice = response.choices[0]
            message = getattr(choice, "message", {})
            if isinstance(message, dict):
                text = message.get("content", "")
            else:
                text = getattr(message, "content", "")
            return LLMResponse(text=text, metadata={"backend": "azure"})

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
        if "referential role expression" in low or "referential expression" in low:
            # Anchor extraction prompt — return empty anchors; the regex fallback
            # will handle any real extraction needed during tests.
            text = '{"anchors": []}'
        elif "sub_questions" in low or "qa planning agent" in low:
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
        """Extract and parse a JSON object from model output text.

        Handles:
        - Clean JSON objects
        - JSON wrapped in ```json ... ``` code fences
        - JSON preceded by prose reasoning
        Tries the last (outermost) JSON object first for completeness.
        """
        if not raw_text:
            return {}

        # Strip code fences and retry — model often wraps in ```json ... ```
        stripped = raw_text
        for fence in ("```json", "```"):
            if fence in stripped:
                parts = stripped.split(fence)
                # Grab content between the first fence pair
                for i in range(1, len(parts)):
                    candidate = parts[i].split("```")[0].strip()
                    if candidate.startswith("{"):
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            pass

        # Fall back: find the last complete { ... } block (outermost)
        end = raw_text.rfind("}")
        start = raw_text.rfind("{", 0, end + 1)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw_text[start : end + 1])
            except json.JSONDecodeError:
                pass

        # Last resort: find first { and last }
        start = raw_text.find("{")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw_text[start : end + 1])
            except json.JSONDecodeError:
                pass

        return {}
