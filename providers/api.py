"""API-based LLM providers for the Science Paper Summariser.

Each provider communicates with an LLM service via its official SDK or HTTP API.
Providers use a single default context size per family (no per-model lookup dicts)
and rely on auto-resolving model aliases when no explicit model is specified.
"""

import os
import base64
import logging
import requests

from .base import Provider

LOGGER = logging.getLogger(__name__)


class ClaudeAPI(Provider):
    """Anthropic Claude API provider."""

    default_model = "claude-sonnet-4-latest"
    default_context_size = 200_000

    def supports_direct_pdf(self):
        return True

    def setup(self):
        """Initialise the Anthropic client."""
        import anthropic

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")

        self.client = anthropic.Anthropic(api_key=api_key)
        if not self.model:
            self.model = self.default_model

    def process_document(self, content, is_pdf, system_prompt, user_prompt, max_tokens=12288):
        """Process document with the Claude Messages API."""
        messages = [{"role": "user", "content": [{"type": "text", "text": user_prompt}]}]

        if is_pdf and isinstance(content, bytes):
            messages[0]["content"].append({
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": base64.b64encode(content).decode(),
                },
            })

        response = self.client.messages.create(
            model=self.model,
            temperature=self.config.get("temperature", 0.2),
            max_tokens=max_tokens,
            system=system_prompt,
            messages=messages,
        )
        return response.content[0].text

    def get_max_context_size(self):
        return self.default_context_size


class OpenAIAPI(Provider):
    """OpenAI API provider (Responses API for PDFs, Chat Completions for text)."""

    default_model = "gpt-5.2"
    default_context_size = 128_000

    def supports_direct_pdf(self):
        return True

    def setup(self):
        """Initialise the OpenAI client."""
        import openai

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.client = openai.OpenAI(api_key=api_key)
        if not self.model:
            self.model = self.default_model

    def process_document(self, content, is_pdf, system_prompt, user_prompt, max_tokens=12288):
        """Process document — uses Responses API for PDFs, Chat Completions for text."""
        if is_pdf and isinstance(content, bytes):
            input_content = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": user_prompt},
                        {
                            "type": "input_file",
                            "file_data": f"data:application/pdf;base64,{base64.b64encode(content).decode()}",
                            "filename": "document.pdf",
                        },
                    ],
                },
            ]
            response = self.client.responses.create(
                model=self.model,
                input=input_content,
                temperature=self.config.get("temperature", 0.2),
                max_output_tokens=max_tokens,
            )
            return response.output[0].content[0].text
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.config.get("temperature", 0.2),
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content

    def get_max_context_size(self):
        return self.default_context_size


class GeminiAPI(Provider):
    """Google Gemini API provider using the google-genai SDK."""

    default_model = "gemini-2.5-flash"
    default_context_size = 1_000_000

    def supports_direct_pdf(self):
        return True

    def setup(self):
        """Initialise the Gemini API client."""
        try:
            from google import genai
        except ImportError:
            raise ImportError(
                "The 'google-genai' package is required. "
                "Install with: pip install google-genai"
            )

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")

        self.client = genai.Client(api_key=api_key)
        if not self.model:
            self.model = self.default_model
        logging.info(f"Gemini model: {self.model}")

    def process_document(self, content, is_pdf, system_prompt, user_prompt, max_tokens=12288):
        """Process document with the Gemini API using system_instruction."""
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=self.config.get("temperature", 0.2),
            max_output_tokens=max_tokens,
            top_p=0.95,
            top_k=40,
        )

        if is_pdf and isinstance(content, bytes):
            logging.info(f"Sending PDF directly to Gemini ({len(content)} bytes)")
            content_parts = [
                user_prompt,
                types.Part.from_bytes(data=content, mime_type="application/pdf"),
            ]
        else:
            content_parts = user_prompt

        response = self.client.models.generate_content(
            model=self.model,
            contents=content_parts,
            config=config,
        )

        result = response.text
        if not result:
            raise ValueError("No content returned from Gemini API")
        return result

    def get_max_context_size(self):
        return self.default_context_size


class PerplexityAPI(Provider):
    """Perplexity API provider using the OpenAI-compatible endpoint."""

    default_model = "sonar-pro"
    default_context_size = 128_000

    def supports_direct_pdf(self):
        return False

    def setup(self):
        """Initialise an OpenAI client pointed at the Perplexity API."""
        import openai

        api_key = os.getenv("PERPLEXITY_API_KEY")
        if not api_key:
            raise ValueError("PERPLEXITY_API_KEY environment variable not set")

        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.perplexity.ai",
        )
        if not self.model:
            self.model = self.default_model

    def process_document(self, content, is_pdf, system_prompt, user_prompt, max_tokens=12288):
        """Process document via Perplexity's OpenAI-compatible chat completions endpoint."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.config.get("temperature", 0.2),
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    def get_max_context_size(self):
        return self.default_context_size


class OllamaAPI(Provider):
    """Ollama local API provider."""

    default_model = "qwen2.5:14b-instruct-q8_0"
    default_context_size = 32_768
    # Local models consume thinking tokens inside the output budget; be generous.
    default_max_output_tokens = 32_768

    def supports_direct_pdf(self):
        return False

    def setup(self):
        """Initialise Ollama settings (no client — uses direct HTTP)."""
        self.base_url = self.config.get("base_url", "http://localhost:11434").rstrip("/")
        if not self.model:
            self.model = self.default_model

    def validate_runtime_ready(self):
        """Verify Ollama is reachable and the configured model is available."""
        timeout = min(float(self.config.get("timeout", 300)), 15.0)
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=timeout)
            response.raise_for_status()
        except requests.RequestException as error:
            raise ValueError(
                f"Ollama endpoint is not reachable at {self.base_url}: {error}"
            ) from error

        try:
            data = response.json()
            available = {m["name"] for m in data.get("models", []) if isinstance(m, dict)}
        except Exception:
            available = set()

        if available and self.model not in available:
            LOGGER.warning(
                "Configured Ollama model %r was not listed by %s/api/tags. "
                "Available models: %s",
                self.model,
                self.base_url,
                ", ".join(sorted(available)),
            )

    def get_preferred_max_tokens(self):
        return int(self.config.get("max_output_tokens", self.default_max_output_tokens))

    def process_document(self, content, is_pdf, system_prompt, user_prompt, max_tokens=12288):
        """Process document with the Ollama generate API."""
        num_ctx = int(self.config.get("num_ctx", self.default_context_size))
        data = {
            "model": self.model,
            "prompt": user_prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": self.config.get("temperature", 0.2),
                "num_ctx": num_ctx,
                "num_predict": max_tokens,
                "stop": ["</input>", "</task>"],
            },
        }

        timeout = self.config.get("timeout", 300)
        response = requests.post(f"{self.base_url}/api/generate", json=data, timeout=timeout)
        response.raise_for_status()

        response_data = response.json()
        if "error" in response_data:
            raise ValueError(f"Ollama error: {response_data['error']}")

        if response_data.get("done_reason") == "length":
            LOGGER.warning(
                "Ollama response hit the token limit (done_reason=length, num_predict=%d); "
                "output may be truncated mid-sentence.",
                max_tokens,
            )

        result = response_data.get("response", "")
        if not result:
            raise ValueError("No content received from Ollama")
        return result

    def get_max_context_size(self):
        return self.default_context_size


def _extract_openai_compatible_model_ids(response):
    """Parse model IDs from an OpenAI-compatible /models JSON response."""
    try:
        data = response.json()
        return {
            m["id"]
            for m in data.get("data", [])
            if isinstance(m, dict) and "id" in m
        }
    except Exception:
        return set()


class OpenAICompatibleAPI(Provider):
    """Generic OpenAI-compatible chat completions provider.

    Works with any local or self-hosted inference server that exposes
    /v1/chat/completions (LM Studio, llama.cpp, vLLM, LocalAI, etc.).
    Requires base_url and model in config; api_key_env is optional.
    """

    default_context_size = 128_000
    # Local/reasoning models consume thinking tokens inside the output budget.
    default_max_output_tokens = 32_768

    def supports_direct_pdf(self):
        return False

    def setup(self):
        """Initialise an OpenAI client pointed at the configured compatible endpoint."""
        import openai

        self.base_url = str(
            self.config.get("base_url") or os.getenv("OPENAI_COMPATIBLE_BASE_URL") or ""
        ).strip().rstrip("/")
        if not self.base_url:
            raise ValueError(
                "OpenAI-compatible API provider requires a base URL. "
                "Set OPENAI_COMPATIBLE_BASE_URL in .env or pass base_url in provider config, "
                'e.g. "http://127.0.0.1:1234/v1" for LM Studio.'
            )

        if not self.model:
            raise ValueError(
                "OpenAI-compatible API provider requires 'model' in config."
            )

        self.api_key_env = str(self.config.get("api_key_env") or "").strip()
        self.api_key = os.getenv(self.api_key_env) if self.api_key_env else ""

        self.client = openai.OpenAI(
            api_key=self.api_key or "not-needed",
            base_url=self.base_url,
            timeout=float(self.config.get("timeout", 300)),
        )

    def validate_runtime_ready(self):
        """Check the endpoint is reachable and the configured model is available."""
        timeout = min(float(self.config.get("timeout", 300)), 15.0)
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            response = requests.get(
                f"{self.base_url}/models", headers=headers, timeout=timeout
            )
            response.raise_for_status()
        except requests.RequestException as error:
            raise ValueError(
                f"OpenAI-compatible API endpoint is not reachable at {self.base_url}: {error}"
            ) from error

        model_ids = _extract_openai_compatible_model_ids(response)
        if model_ids and self.model not in model_ids:
            LOGGER.warning(
                "Configured model %r was not listed by %s/models. "
                "Available models: %s",
                self.model,
                self.base_url,
                ", ".join(sorted(model_ids)),
            )

    def get_preferred_max_tokens(self):
        return int(self.config.get("max_output_tokens", self.default_max_output_tokens))

    def process_document(self, content, is_pdf, system_prompt, user_prompt, max_tokens=12288):
        """Process document via the OpenAI-compatible chat completions endpoint."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.config.get("temperature", 0.2),
            max_tokens=max_tokens,
        )

        choice = response.choices[0]
        if choice.finish_reason == "length":
            LOGGER.warning(
                "OpenAI-compatible API response hit the token limit "
                "(finish_reason=length, max_tokens=%d); output may be truncated mid-sentence.",
                max_tokens,
            )

        result = choice.message.content
        if not result:
            raise ValueError("No content received from OpenAI-compatible API")
        return result

    def get_max_context_size(self):
        return self.default_context_size
