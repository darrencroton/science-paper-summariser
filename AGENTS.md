# AGENTS.md

This file provides guidance to AI coding CLI tools like Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Science Paper Summariser is a Python service that monitors an `input/` directory for PDFs and text files, sends them to an LLM with a strict astronomy-focused summary template, validates the output, and writes markdown summaries to `output/`. Processed papers are moved to `processed/`. It runs as a background `nohup` process via shell scripts.

The tool uses an **explicit mode/provider model**: the user chooses `cli` or `api`, then chooses a provider within that mode. The program never switches modes automatically.

## Commands

```bash
# Setup
python -m venv myenv && source myenv/bin/activate && pip install -r requirements.txt
cp .env.template .env  # then fill in API keys if you plan to use API mode

# Run as background service
./start_paper_summariser.sh                      # default: cli claude
./start_paper_summariser.sh [mode] [provider] [model]
./stop_paper_summariser.sh

# Run directly (foreground, for debugging)
source myenv/bin/activate
python3 summarise.py                              # default: cli claude
python3 summarise.py cli claude claude-sonnet-4-latest --effort high
python3 summarise.py cli copilot claude-sonnet-4.6 --effort high
OPENAI_COMPATIBLE_BASE_URL=http://localhost:1234/v1 python3 summarise.py api openai-compatible google/gemma-4-31b

# Tests
./myenv/bin/python -m unittest tests/test_effort_support.py

# Tail logs while running
tail -f logs/history.log
```

The focused unit tests cover CLI argument parsing and CLI provider command construction.

## Architecture

### Source Files

- **`summarise.py`** — Main orchestration: file monitoring loop, PDF reading (via `marker-pdf`, lazy-loaded), prompt construction, LLM call with retry (3 attempts, exponential backoff), output validation, metadata-based filename generation, file movement. Provider is created once in `main()` and reused. Uses signal handlers for graceful SIGTERM/SIGINT shutdown via a global `shutdown_requested` flag.

- **`providers/`** — Provider package with clean separation of API and CLI providers:
  - `base.py` — `Provider` base class defining the interface: `setup()`, `process_document()`, `get_max_context_size()`, `supports_direct_pdf()`, `validate_runtime_ready()`, `get_preferred_max_tokens()`.
  - `api.py` — API providers: `ClaudeAPI`, `OpenAIAPI`, `GeminiAPI`, `PerplexityAPI`, `OllamaAPI`, `OpenAICompatibleAPI`. Each uses its SDK or direct HTTP. Gemini uses `system_instruction`. Perplexity uses the OpenAI SDK pointed at api.perplexity.ai. `OpenAICompatibleAPI` is the preferred local/open-model route and targets any `/v1/chat/completions` server (LM Studio, llama.cpp, Ollama's OpenAI-compatible endpoint, vLLM, LocalAI, etc.).
  - `cli.py` — CLI providers: `CLIProvider` base class with `ClaudeCLI`, `CodexCLI`, `GeminiCLI`, `CopilotCLI`, `OpenCodeCLI`. All use subprocess invocation in non-interactive mode. OpenCode is retained as an optional CLI route, not the primary local-model path.
  - `__init__.py` — `create_provider(mode, provider_name, config)` factory with explicit registries and prerequisite validation.

### Processing Pipeline

1. `get_pending_files()` — scans `input/` excluding completed and failed files
2. `read_input_file()` — reads PDF (binary for direct upload, or marker-pdf text extraction) or text
3. `create_system_prompt()` + `create_user_prompt()` — builds prompts using `project_knowledge/` files
4. `_call_llm_with_retry()` — calls the provider with retry and exponential backoff; validates that summaries with `[^N]` footnote markers also contain a `## References` section, retrying if not
5. `strip_preamble()` — removes any text before the first `# ` heading
6. `generate_glossary()` — makes a focused summary-only post-processing call and inserts `## Glossary` before `## References`
7. `generate_tags()` — makes a focused summary-only post-processing call using arXiv-filtered science keywords and inserts `## Tags` before `## References`
8. `validate_summary()` — checks structure (title, year, authors, bullets/footnotes, glossary, tags)
9. `save_summary()` — writes to `output/` with metadata-derived filename (`Author - Year - Title.md`)
10. `move_to_done()` — moves original to `processed/` with conflict-safe renaming

Source metadata is detected once per file before prompting. Filename metadata is extracted once from the final summary and used for both `save_summary()` and `move_to_done()`.

### Provider Routing

```text
python3 summarise.py                 → cli claude
python3 summarise.py cli claude      → ClaudeCLI
python3 summarise.py cli gemini      → GeminiCLI
python3 summarise.py cli codex       → CodexCLI
python3 summarise.py cli copilot     → CopilotCLI
python3 summarise.py cli opencode    → OpenCodeCLI
python3 summarise.py api claude      → ClaudeAPI
python3 summarise.py api gemini      → GeminiAPI
python3 summarise.py api openai      → OpenAIAPI
python3 summarise.py api perplexity  → PerplexityAPI
python3 summarise.py api ollama               → OllamaAPI
python3 summarise.py api openai-compatible    → OpenAICompatibleAPI
```

If the selected mode/provider combination is invalid, the CLI binary is missing, the required API key is absent, or a local API server is unreachable, startup fails immediately.

### State Tracking

File-based, no database:
- `logs/completed.log` — one filename per line (successfully processed)
- `logs/failed.log` — pipe-separated: `filename|timestamp|error_message` (failed after 3 attempts)
- `logs/prompt.txt` — last full prompt sent to LLM (debug aid)
- `logs/process.pid` — PID of the running service

## Key Domain Conventions

- All summaries use **UK English** and **LaTeX** for equations
- Every bullet point must have a footnote with an **exact quote** from the paper (never paraphrased)
- The summary template in `project_knowledge/paper-summary-template.md` defines the exact structure — changes here affect all future summaries
- Tags must use CamelCase keywords from `project_knowledge/astronomy-keywords.txt`; arXiv category metadata filters the keyword list when available
- Author lists must be complete — never truncated with "et al." in the summary body (though filenames use "et al." for 3+ authors)

## Adding a New LLM Provider

### API Provider
1. Create a subclass of `Provider` in `providers/api.py`
2. Set `default_model` and `default_context_size` class attributes
3. Implement: `setup()`, `process_document()`, `supports_direct_pdf()`
4. Override `validate_runtime_ready()` if the provider needs a pre-run connectivity check (local/self-hosted servers)
5. Override `get_preferred_max_tokens()` if the provider needs a higher output token budget than the 12 288 base default (local/reasoning models should use 32 768 or more)
6. Add the provider name to `_API_PROVIDERS` in `providers/__init__.py`

### CLI Provider
1. Create a subclass of `CLIProvider` in `providers/cli.py`
2. Set class attributes: `cli_command`, `prompt_flag`, `extra_flags`, `model_flag`
3. Override command-building or output parsing methods when the CLI has provider-specific syntax
4. Add to `_CLI_PROVIDERS` in `providers/__init__.py`

## Environment Variables

API keys are loaded from `.env` via `python-dotenv`. They are only needed when using API mode:
- `ANTHROPIC_API_KEY` (Claude API)
- `OPENAI_API_KEY` (OpenAI API)
- `GOOGLE_API_KEY` (Gemini API)
- `PERPLEXITY_API_KEY` (Perplexity API)
- Ollama requires no key or URL config (defaults to `http://localhost:11434`)
- `openai-compatible` requires `OPENAI_COMPATIBLE_BASE_URL` set in `.env` or the launch environment; see README.md for the canonical setup examples
- `openai-compatible` with a key-protected server: set `OPENAI_COMPATIBLE_API_KEY_ENV` to the env var name containing the bearer token, and define the actual key outside the repo, preferably in `~/.llm/.env.llm`
- CLI tools (`claude`, `codex`, `gemini`, `copilot`, `opencode`) require no API keys
- OpenCode model selection uses `--model provider/model` (e.g. `ollama/llama3.2`) and remains available when specifically needed; prefer `api openai-compatible` for local/open models

## Directory Notes

`input/`, `output/`, and `processed/` are symlinks to external locations (Dropbox, Obsidian vault, etc.) and are gitignored. The `logs/` directory is also gitignored. The `myenv/` virtualenv is gitignored. The `archive/` directory holds superseded files and is gitignored.
