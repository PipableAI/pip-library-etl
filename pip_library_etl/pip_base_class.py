import json

import requests
from transformers import AutoModelForCausalLM, AutoTokenizer

from pip_library_etl.models import Device


class PipBaseClass:
    def __init__(
        self,
        model_key: str,
        device: str,
        url: str,
    ):
        self.device = Device(device)
        self.model_key = model_key
        self.model = None
        self.tokenizer = None
        self.url = None
        if self.device == Device.CLOUD:
            self.url = url
        else:
            self._load_model()

    def _query_model(self, prompt: str, max_new_tokens: int, eos_token: str) -> str:
        if self.device == Device.CLOUD:
            payload = {
                "model_name": self.model_key,
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
            }
            response = requests.request(
                method="POST", url=self.url, data=payload, timeout=120
            )
            if response.status_code == 200:
                return json.loads(response.text)["response"]
            else:
                raise Exception(f"Error generating response using {self.url}.")
        elif self.device == Device.CUDA:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device.value)
            encoded_eos = self.tokenizer.encode(eos_token)
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, forced_eos_token_id=encoded_eos
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _load_model(self):
        if self.model is None or self.tokenizer is None:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_key).to(
                self.device.value
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_key)
