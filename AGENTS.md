# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Science Paper Summariser is a Python service that monitors an `input/` directory for PDFs and text files, sends them to an LLM with a strict astronomy-focused summary template, validates the output, and writes markdown summaries to `output/`. Processed papers are moved to `processed/`. It runs as a background `nohup` process via shell scripts.

The tool uses a **CLI-first provider model**: it prefers AI CLI tools (Claude Code, Codex, Gemini CLI, Copilot) when available on PATH, falling back to API providers automatically.

## Commands

```bash
# Setup
python -m venv myenv && source myenv/bin/activate && pip install -r requirements.txt
cp .env.template .env  # then fill in API keys (only needed for API fallback)

# Run as background service
./start_paper_summariser.sh [provider] [model]   # e.g. claude, gemini, codex, copilot, openai
./stop_paper_summariser.sh

# Run directly (foreground, for debugging)
source myenv/bin/activate
python3 summarise.py claude                       # default: CLI-first
python3 summarise.py claude-api                   # force API mode
python3 summarise.py gemini gemini-2.5-flash      # specific model

# Tail logs while running
tail -f logs/history.log
```

There are no automated tests.

## Architecture

### Source Files

- **`summarise.py`** — Main orchestration: file monitoring loop, PDF reading (via `marker-pdf`, lazy-loaded), prompt construction, LLM call with retry (3 attempts, exponential backoff), output validation, metadata-based filename generation, file movement. Provider is created once in `main()` and reused. Uses signal handlers for graceful SIGTERM/SIGINT shutdown via a global `shutdown_requested` flag.

- **`providers/`** — Provider package with clean separation of API and CLI providers:
  - `base.py` — `Provider` base class defining the interface: `setup()`, `process_document()`, `get_max_context_size()`, `supports_direct_pdf()`.
  - `api.py` — API providers: `ClaudeAPI`, `OpenAIAPI`, `GeminiAPI`, `PerplexityAPI`, `OllamaAPI`. Each uses its SDK directly. Gemini uses `system_instruction` parameter. Perplexity uses the OpenAI-compatible endpoint.
  - `cli.py` — CLI providers: `CLIProvider` base class with `ClaudeCLI`, `CodexCLI`, `GeminiCLI`, `CopilotCLI`. All use subprocess invocation in non-interactive mode.
  - `__init__.py` — `create_provider()` factory with auto-detection logic: CLI-first names check PATH before falling back to API.

### Processing Pipeline

1. `get_pending_files()` — scans `input/` excluding completed and failed files
2. `read_input_file()` — reads PDF (binary for direct upload, or marker-pdf text extraction) or text
3. `create_system_prompt()` + `create_user_prompt()` — builds prompts using `project_knowledge/` files
4. `_call_llm_with_retry()` — calls the provider with retry and exponential backoff
5. `strip_preamble()` — removes any text before the first `# ` heading
6. `validate_summary()` — checks structure (title, year, authors, bullets/footnotes, glossary, tags)
7. `save_summary()` — writes to `output/` with metadata-derived filename (`Author - Year - Title.md`)
8. `move_to_done()` — moves original to `processed/` with conflict-safe renaming

Metadata is extracted once per file and passed to both `save_summary()` and `move_to_done()`.

### Provider Routing

```
"claude" (default) → claude CLI on PATH? → Yes: ClaudeCLI
                                          → No: ANTHROPIC_API_KEY set? → Yes: ClaudeAPI
                                                                        → No: error
"gemini"           → gemini CLI on PATH? → Yes: GeminiCLI → No: GeminiAPI (similar)
"codex"            → CodexCLI (CLI only)
"copilot"          → CopilotCLI (CLI only)
"openai"           → OpenAIAPI (API only)
"perplexity"       → PerplexityAPI (API only)
"ollama"           → OllamaAPI (local)
"claude-api"       → ClaudeAPI (explicit, bypasses CLI check)
"gemini-api"       → GeminiAPI (explicit, bypasses CLI check)
```

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
- Tags must use keywords from `project_knowledge/astronomy-keywords.txt` (CamelCase)
- Author lists must be complete — never truncated with "et al." in the summary body (though filenames use "et al." for 3+ authors)

## Adding a New LLM Provider

### API Provider
1. Create a subclass of `Provider` in `providers/api.py`
2. Set `default_model` and `default_context_size` class attributes
3. Implement: `setup()`, `process_document()`, `supports_direct_pdf()`
4. Add the provider name to the routing dicts in `providers/__init__.py`

### CLI Provider
1. Create a subclass of `CLIProvider` in `providers/cli.py`
2. Set class attributes: `cli_command`, `prompt_flag`, `extra_flags`, `model_flag`
3. Add to `_CLI_ONLY_PROVIDERS` or `_CLI_FIRST_PROVIDERS` in `providers/__init__.py`

## Environment Variables

API keys are loaded from `.env` via `python-dotenv`. Only needed when using API providers (or as fallback when CLI tools are unavailable):
- `ANTHROPIC_API_KEY` (Claude API)
- `OPENAI_API_KEY` (OpenAI API)
- `GOOGLE_API_KEY` (Gemini API)
- `PERPLEXITY_API_KEY` (Perplexity API)
- Ollama requires no key (local at `localhost:11434`)
- CLI tools (`claude`, `codex`, `gemini`, `copilot`) require no API keys

## Directory Notes

`input/`, `output/`, and `processed/` are symlinks to external locations (Dropbox, Obsidian vault, etc.) and are gitignored. The `logs/` directory is also gitignored. The `myenv/` virtualenv is gitignored. The `archive/` directory holds superseded files and is gitignored.
