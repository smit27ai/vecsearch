"""Chat model access through the Hugging Face Inference API.

Needs a Hugging Face token in the HF_TOKEN environment variable (free tier
works). The model is set with RAG_MODEL; the default is the one the original
chatbot used.
"""

import os

DEFAULT_MODEL = "google/gemma-2-9b-it"


class HFChat:
    def __init__(self, model=None, token=None, max_tokens=400):
        from huggingface_hub import InferenceClient

        token = token or os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("set the HF_TOKEN environment variable to call the LLM")
        self.model = model or os.environ.get("RAG_MODEL", DEFAULT_MODEL)
        self.client = InferenceClient(model=self.model, token=token)
        self.max_tokens = max_tokens

    def complete(self, messages):
        """messages: [{"role": "system"|"user"|"assistant", "content": str}] -> reply text."""
        # Gemma's chat template has no system role; fold it into the first user turn.
        if messages and messages[0]["role"] == "system":
            system, rest = messages[0]["content"], messages[1:]
            rest = [{"role": rest[0]["role"], "content": system + "\n\n" + rest[0]["content"]}] + rest[1:]
            messages = rest
        out = self.client.chat_completion(messages, max_tokens=self.max_tokens, temperature=0.0)
        return out.choices[0].message.content.strip()
