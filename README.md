# Science Paper Summariser

A Python tool that uses LLMs to automatically summarise scientific papers, following an exact template and hashtag list. The tool monitors an input directory for new PDFs or text files, processes them using your chosen provider, and generates markdown summaries with extensive referencing back to the source material.

## Features

- **CLI-first provider model**: prefers AI CLI tools (Claude Code, Codex, Gemini CLI, Copilot) when available, with automatic API fallback
- Monitors input directory for new PDFs/text files
- Generates detailed paper summaries in markdown format
- Includes exact quotes as footnotes for all statements
- UK English with LaTeX support for equations
- Creates glossary of technical terms
- Moves processed files to archive
- Comprehensive logging
- Tracks files that failed after 3 processing attempts in failed.log

## Directory Structure

```
science-paper-summariser/
├── input/               # Place papers here for processing
├── output/              # Generated summaries appear here
├── processed/           # Completed papers are moved here
├── logs/                # Processing history and errors
├── providers/           # LLM provider implementations
│   ├── __init__.py      # Factory with auto-detection logic
│   ├── base.py          # Provider base class
│   ├── api.py           # API providers (Claude, OpenAI, Gemini, Perplexity, Ollama)
│   └── cli.py           # CLI providers (Claude Code, Codex, Gemini CLI, Copilot)
├── project_knowledge/
│   ├── astronomy-keywords.txt
│   └── paper-summary-template.md
├── summarise.py         # Main orchestration
└── start/stop scripts
```

## Setup

1. Create virtual environment:
   ```bash
   python -m venv myenv
   source myenv/bin/activate
   pip install -r requirements.txt
   ```

2. **CLI tools (recommended):** Install one or more AI CLI tools. If a CLI tool is on your PATH, it will be used automatically:
   - [Claude Code](https://docs.anthropic.com/en/docs/claude-code) — `claude`
   - [Codex CLI](https://github.com/openai/codex) — `codex`
   - [Gemini CLI](https://github.com/google-gemini/gemini-cli) — `gemini`
   - [GitHub Copilot CLI](https://docs.github.com/en/copilot) — `copilot`

3. **API keys (optional fallback):** If you don't have CLI tools installed, or want to use explicit API providers, add your keys to `.env` (see `.env.template`):
   ```
   ANTHROPIC_API_KEY=your_key_here
   OPENAI_API_KEY=your_key_here
   PERPLEXITY_API_KEY=your_key_here
   GOOGLE_API_KEY=your_key_here
   ```

4. Create required directories:
   ```bash
   mkdir -p input output processed logs
   ```
   Tip: create symbolic links to where you want to read/write the input/output

5. Make the start and stop scripts executable:
   ```bash
   chmod +x start_paper_summariser.sh stop_paper_summariser.sh
   ```

## Usage

1. Start the summariser with your preferred provider:
   ```bash
   # Use Claude (default) — tries Claude Code CLI first, falls back to API
   ./start_paper_summariser.sh

   # Use Gemini — tries Gemini CLI first, falls back to API
   ./start_paper_summariser.sh gemini

   # Use Codex CLI
   ./start_paper_summariser.sh codex

   # Use Copilot CLI
   ./start_paper_summariser.sh copilot

   # Use OpenAI API directly
   ./start_paper_summariser.sh openai

   # Force API mode (bypass CLI auto-detection)
   ./start_paper_summariser.sh claude-api
   ./start_paper_summariser.sh gemini-api

   # Use Perplexity API
   ./start_paper_summariser.sh perplexity

   # Use Ollama (local)
   ./start_paper_summariser.sh ollama

   # Specify a model override for any provider
   ./start_paper_summariser.sh claude claude-opus-4-6
   ./start_paper_summariser.sh gemini gemini-2.5-flash
   ./start_paper_summariser.sh openai gpt-5-mini
   ```

2. Place PDF or text files in the `input/` directory

3. Monitor `output/` for generated summaries. Processed papers end up in `processed/`

4. Stop the summariser:
   ```bash
   ./stop_paper_summariser.sh
   ```

## Providers

### CLI-first (auto-detection)

| Name | CLI Tool | API Fallback | Notes |
|------|----------|-------------|-------|
| `claude` (default) | `claude` | Anthropic API | Requires Claude Code CLI or `ANTHROPIC_API_KEY` |
| `gemini` | `gemini` | Google Gemini API | Requires Gemini CLI or `GOOGLE_API_KEY` |

### CLI-only

| Name | CLI Tool | Notes |
|------|----------|-------|
| `codex` | `codex` | OpenAI Codex CLI |
| `copilot` | `copilot` | GitHub Copilot CLI |

### API-only

| Name | Notes |
|------|-------|
| `openai` / `openai-api` | Requires `OPENAI_API_KEY` |
| `perplexity` / `perplexity-api` | Requires `PERPLEXITY_API_KEY` |
| `ollama` | Local at `localhost:11434` |
| `claude-api` | Explicit API (bypasses CLI check) |
| `gemini-api` | Explicit API (bypasses CLI check) |

Model override works for all providers — pass the model name as the second argument. If no model is specified, each provider uses its own sensible default. Check each provider's documentation for available models.

NOTE: In my experience, the latest Claude Sonnet/Opus, Gemini Pro, and GPT give the best results via both CLI and API. Ollama can be hit and miss depending on the model and available memory.

## Requirements

- zsh shell
- Python 3.9+
- At least one of:
  - An AI CLI tool on PATH (`claude`, `codex`, `gemini`, `copilot`)
  - API keys for the chosen provider
- python-dotenv
- marker-pdf (for PDF text extraction when the provider doesn't support direct PDF upload)
