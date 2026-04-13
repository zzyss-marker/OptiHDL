from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import requests
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


class BaseLLMClient:
    mode = "base"

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        temperature: float = 0.8,
        max_new_tokens: int = 1024,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.05,
    ) -> str:
        raise NotImplementedError

    def cleanup(self) -> None:
        return None


class LocalHFClient(BaseLLMClient):
    mode = "local"

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.tokenizer, self.model = self._load_model(model_path)

    def _load_model(self, model_path: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if os.path.exists(os.path.join(model_path, "adapter_config.json")):
            with open(os.path.join(model_path, "adapter_config.json"), encoding="utf-8") as handle:
                adapter_config = json.load(handle)
            base_model_name = adapter_config.get("base_model_name_or_path")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
            )
            model = PeftModel.from_pretrained(base_model, model_path)
            model = model.merge_and_unload()
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
            )

        model.eval()
        return tokenizer, model

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        temperature: float = 0.8,
        max_new_tokens: int = 1024,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.05,
    ) -> str:
        use_chat = hasattr(self.tokenizer, "apply_chat_template") and getattr(self.tokenizer, "chat_template", None)
        context_max = self._get_context_window()
        estimated_prompt_tokens = len(prompt) // 3

        if estimated_prompt_tokens < 2000:
            effective_new = max_new_tokens
            input_max = context_max - effective_new - 100
        elif estimated_prompt_tokens < 4000:
            effective_new = max(1024, int(max_new_tokens * 0.8))
            input_max = context_max - effective_new - 100
        else:
            effective_new = max(768, context_max - estimated_prompt_tokens - 200)
            input_max = context_max - effective_new - 100

        if use_chat:
            try:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=input_max)
            except Exception:
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=input_max)
        else:
            merged_prompt = f"{system_prompt}\n\n{prompt}".strip() if system_prompt else prompt
            inputs = self.tokenizer(merged_prompt, return_tensors="pt", truncation=True, max_length=input_max)

        input_len = inputs["input_ids"].shape[1]
        if torch.cuda.is_available():
            inputs = {key: value.cuda() for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=effective_new,
                temperature=temperature,
                do_sample=True,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][input_len:]
        if new_tokens.numel() == 0:
            return ""
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def _get_context_window(self) -> int:
        try:
            max_length = int(getattr(self.tokenizer, "model_max_length", 4096) or 4096)
        except Exception:
            max_length = 4096
        if max_length >= 1_000_000:
            max_length = 8192
        return max(2048, min(max_length, 32768))


class OpenAICompatibleClient(BaseLLMClient):
    mode = "api"

    def __init__(self, base_url: str, api_key: str, model_name: str, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        temperature: float = 0.8,
        max_new_tokens: int = 1024,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.05,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [],
            "temperature": temperature,
            "max_tokens": max_new_tokens,
            "top_p": top_p,
        }
        if system_prompt:
            payload["messages"].append({"role": "system", "content": system_prompt})
        payload["messages"].append({"role": "user", "content": prompt})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


def build_llm_client(model_path: str) -> BaseLLMClient:
    mode = os.environ.get("OPTIHDL_LLM_MODE", "auto").strip().lower()
    base_url = os.environ.get("OPTIHDL_API_BASE_URL", "").strip()
    api_key = os.environ.get("OPTIHDL_API_KEY", "").strip()
    api_model = os.environ.get("OPTIHDL_API_MODEL", "").strip()

    use_api = False
    if mode == "api":
        use_api = True
    elif mode == "auto" and base_url and api_key and api_model:
        use_api = True

    if use_api:
        if not (base_url and api_key and api_model):
            raise ValueError("API 模式下需要配置 OPTIHDL_API_BASE_URL、OPTIHDL_API_KEY、OPTIHDL_API_MODEL")
        return OpenAICompatibleClient(base_url=base_url, api_key=api_key, model_name=api_model)

    return LocalHFClient(model_path=model_path)
