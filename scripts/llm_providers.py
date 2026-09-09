"""
Thin, provider-agnostic wrapper around the Anthropic and OpenAI chat
completion APIs, so extraction scripts take a --provider flag instead of
being locked into one SDK's client shape.

Both providers are called with the same (system_prompt, user_text) inputs
and both return a single string of the model's text output. Callers handle
parsing (e.g. stripping JSON code fences) themselves -- this module only
normalizes the request/response shape, not the content.
"""

# Defaults chosen for cheap, high-volume structured extraction as of
# Aug 2026. Override with --model if quality on your data needs a step up
# (e.g. "claude-sonnet-5" or "gpt-5.6-terra").
DEFAULT_MODELS = {
    "anthropic": "claude-haiku-4-5-20251001",
    "openai": "gpt-5.6-luna",
}


class LLMProvider:
    def __init__(self, provider: str, model: str = None):
        provider = provider.lower()
        if provider not in DEFAULT_MODELS:
            raise ValueError(f"Unknown provider '{provider}'. Choose from: {list(DEFAULT_MODELS)}")
        self.provider = provider
        self.model = model or DEFAULT_MODELS[provider]
        self._client = self._build_client()

    def _build_client(self):
        if self.provider == "anthropic":
            from anthropic import Anthropic
            return Anthropic()  # reads ANTHROPIC_API_KEY from env
        from openai import OpenAI
        return OpenAI()  # reads OPENAI_API_KEY from env

    def complete(self, system_prompt: str, user_text: str, max_tokens: int = 500) -> str:
        if self.provider == "anthropic":
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_text}],
            )
            return "".join(b.text for b in resp.content if b.type == "text").strip()

        resp = self._client.chat.completions.create(
            model=self.model,
            max_completion_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
        )
        return (resp.choices[0].message.content or "").strip()

    def __repr__(self):
        return f"LLMProvider(provider={self.provider!r}, model={self.model!r})"
