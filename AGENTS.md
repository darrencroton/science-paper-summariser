# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Science Paper Summariser is a Python service that monitors an `input/` directory for PDFs and text files, sends them to an LLM with a strict astronomy-focused summary template, validates the output, and writes markdown summaries to `output/`. Processed papers are moved to `processed/`. It runs as a background `nohup` process via shell scripts.

## Commands

```bash
# Setup
python -m venv myenv && source myenv/bin/activate && pip install -r requirements.txt
cp .env.template .env  # then fill in API keys

# Run as background service
./start_paper_summariser.sh [provider] [model]   # e.g. claude, gemini, openai, perplexity, ollama
./stop_paper_summariser.sh

# Run directly (foreground, for debugging)
source myenv/bin/activate
python3 summarise.py claude                       # default model
python3 summarise.py gemini gemini-2.5-flash      # specific model

# Tail logs while running
tail -f logs/history.log
```

There are no automated tests.

## Architecture

Two source files with clear separation:

- **`summarise.py`** — Main orchestration: file monitoring loop, PDF reading (via `marker-pdf`), prompt construction, LLM call with retry (3 attempts, exponential backoff), output validation, metadata-based filename generation, file movement. Uses signal handlers for graceful SIGTERM/SIGINT shutdown via a global `shutdown_requested` flag.

- **`llm_providers.py`** — Provider abstraction layer. Base class `LLMProvider` with subclasses: `ClaudeProvider`, `OpenAIProvider`, `GeminiProvider`, `PerplexityProvider`, `OllamaProvider`. Factory function `create_llm_provider(name, config)` instantiates them. Key method: `process_document(content, is_pdf, system_prompt, user_prompt, max_tokens)`. Providers declare `supports_direct_pdf()` — if True, raw PDF bytes are sent to the API; if False, `marker-pdf` extracts text first.

### Processing Pipeline

1. `get_pending_files()` — scans `input/` excluding completed and failed files
2. `read_input_file()` — reads PDF (binary or marker-pdf extraction) or text
3. `create_system_prompt()` + `create_user_prompt()` — builds prompts using `project_knowledge/` files
4. `llm_provider.process_document()` — calls the LLM API
5. `strip_preamble()` — removes any text before the first `# ` heading
6. `validate_summary()` — checks structure (title, year, authors, bullets/footnotes match, glossary, tags)
7. `save_summary()` — writes to `output/` with metadata-derived filename (`Author - Year - Title.md`)
8. `move_to_done()` — moves original to `processed/` with conflict-safe renaming

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

1. Create a subclass of `LLMProvider` in `llm_providers.py`
2. Implement: `setup()`, `process_document()`, `get_max_context_size()`, `get_default_model()`, `supports_direct_pdf()`
3. Add the provider name to the `providers` dict in `create_llm_provider()`

## Environment Variables

API keys are loaded from `.env` via `python-dotenv`. Only the key for the chosen provider is needed:
- `ANTHROPIC_API_KEY` (Claude)
- `OPENAI_API_KEY` (OpenAI)
- `GOOGLE_API_KEY` (Gemini)
- `PERPLEXITY_API_KEY` (Perplexity)
- Ollama requires no key (local at `localhost:11434`)

## Directory Notes

`input/`, `output/`, and `processed/` are symlinks to external locations (Dropbox, Obsidian vault, etc.) and are gitignored. The `logs/` directory is also gitignored. The `myenv/` virtualenv is gitignored.
