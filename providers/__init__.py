"""Provider package for the Science Paper Summariser.

Auto-detects CLI tools on PATH and falls back to API providers when unavailable.

Provider routing:
    "claude"         -> ClaudeCLI if available, else ClaudeAPI (default provider)
    "codex"          -> CodexCLI (CLI only)
    "gemini"         -> GeminiCLI if available, else GeminiAPI
    "copilot"        -> CopilotCLI (CLI only)
    "openai"         -> OpenAIAPI (API only; CLI equivalent is "codex")
    "perplexity"     -> PerplexityAPI (API only)
    "ollama"         -> OllamaAPI (local API)
    "claude-api"     -> ClaudeAPI (explicit, bypasses auto-detection)
    "openai-api"     -> OpenAIAPI (explicit)
    "gemini-api"     -> GeminiAPI (explicit)
    "perplexity-api" -> PerplexityAPI (explicit)
"""

import logging
import os
import shutil

from .api import ClaudeAPI, OpenAIAPI, GeminiAPI, PerplexityAPI, OllamaAPI
from .cli import ClaudeCLI, CodexCLI, GeminiCLI, CopilotCLI

# Explicit API-only provider names (bypass auto-detection)
_API_PROVIDERS = {
    "claude-api": ClaudeAPI,
    "openai": OpenAIAPI,
    "openai-api": OpenAIAPI,
    "gemini-api": GeminiAPI,
    "perplexity": PerplexityAPI,
    "perplexity-api": PerplexityAPI,
    "ollama": OllamaAPI,
}

# CLI-first providers: try CLI tool, fall back to API if unavailable
_CLI_FIRST_PROVIDERS = {
    "claude": {
        "cli": ClaudeCLI,
        "api": ClaudeAPI,
        "cli_command": "claude",
        "api_key_env": "ANTHROPIC_API_KEY",
    },
    "gemini": {
        "cli": GeminiCLI,
        "api": GeminiAPI,
        "cli_command": "gemini",
        "api_key_env": "GOOGLE_API_KEY",
    },
}

# CLI-only providers (no API fallback)
_CLI_ONLY_PROVIDERS = {
    "codex": CodexCLI,
    "copilot": CopilotCLI,
}


def detect_available_clis():
    """Return a dict of CLI tool names to their paths, for all detected tools."""
    tools = ["claude", "codex", "gemini", "copilot"]
    available = {}
    for tool in tools:
        path = shutil.which(tool)
        if path:
            available[tool] = path
    return available


def create_provider(provider_name, config=None):
    """Create a provider instance with CLI-first auto-detection.

    CLI-first names (claude, gemini) check for the CLI tool on PATH first,
    falling back to the corresponding API provider if the CLI is not found.
    Explicit API names (claude-api, openai-api, etc.) bypass auto-detection.

    Args:
        provider_name: Provider identifier (e.g. "claude", "gemini-api").
        config: Optional configuration dict (e.g. {"model": "claude-opus-4-6"}).

    Returns:
        A Provider instance ready to process documents.

    Raises:
        ValueError: If the provider name is unknown or required resources are missing.
    """
    provider_name = provider_name.lower().strip()

    # 1. Explicit API providers
    if provider_name in _API_PROVIDERS:
        provider_class = _API_PROVIDERS[provider_name]
        logging.info(f"Using API provider: {provider_class.__name__}")
        return provider_class(config)

    # 2. CLI-only providers
    if provider_name in _CLI_ONLY_PROVIDERS:
        provider_class = _CLI_ONLY_PROVIDERS[provider_name]
        logging.info(f"Using CLI provider: {provider_class.__name__}")
        return provider_class(config)

    # 3. CLI-first providers (try CLI, fall back to API)
    if provider_name in _CLI_FIRST_PROVIDERS:
        entry = _CLI_FIRST_PROVIDERS[provider_name]
        cli_command = entry["cli_command"]

        if shutil.which(cli_command):
            logging.info(f"'{cli_command}' CLI found on PATH — using CLI provider")
            return entry["cli"](config)
        else:
            api_key_env = entry["api_key_env"]
            if os.getenv(api_key_env):
                logging.info(
                    f"'{cli_command}' CLI not found — falling back to API provider "
                    f"({api_key_env} is set)"
                )
                return entry["api"](config)
            else:
                raise ValueError(
                    f"Cannot use '{provider_name}': '{cli_command}' CLI not found on PATH "
                    f"and {api_key_env} is not set. "
                    f"Either install the CLI tool or set the API key."
                )

    # Unknown provider
    all_names = sorted(
        set(_API_PROVIDERS.keys())
        | set(_CLI_ONLY_PROVIDERS.keys())
        | set(_CLI_FIRST_PROVIDERS.keys())
    )
    raise ValueError(
        f"Unknown provider: '{provider_name}'. "
        f"Available providers: {', '.join(all_names)}"
    )
