# Science Paper Summariser

A Python tool that watches an `input/` directory for PDFs or text files, sends each paper to an LLM using a strict astronomy-focused template, validates the response shape, writes a markdown summary to `output/`, and moves the original file to `processed/`.

Provider selection is explicit:

```bash
python3 summarise.py [mode] [provider] [model] [--effort level]
```

## Features

- Explicit `cli` and `api` modes
- Monitors `input/` for PDF and text files
- Generates markdown summaries with exact supporting quotes in footnotes
- Uses UK English and LaTeX for equations
- Includes glossary and tags sections
- Recovers arXiv metadata from filenames to enforce the `Published:` date and link
- Moves processed files with collision-safe renaming
- Logs successes, failures, and the last prompt sent to the model

## Directory Structure

```text
science-paper-summariser/
├── input/               # Place papers here for processing
├── output/              # Generated summaries appear here
├── processed/           # Completed papers are moved here
├── logs/                # Processing history and errors
├── providers/
│   ├── __init__.py      # Explicit mode/provider factory
│   ├── base.py          # Provider base class
│   ├── api.py           # API providers
│   └── cli.py           # CLI providers
├── project_knowledge/
│   ├── astronomy-keywords.txt
│   └── paper-summary-template.md
├── summarise.py         # Main orchestration loop
└── start/stop scripts
```

## Setup

1. Create a virtual environment and install dependencies:

```bash
python -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

2. Create the runtime directories:

```bash
mkdir -p input output processed logs
```

3. Choose how you want to run the summariser:

- `cli` mode requirements:
  - Install the CLI you plan to use and make sure it is on `PATH`
  - Supported CLI providers: `claude`, `copilot`, `codex`, `gemini`, `opencode`
- `api` mode requirements:
  - Add the required credentials to `.env`
  - Supported API providers: `claude`, `gemini`, `openai`, `perplexity`, `ollama`, `openai-compatible`
  - Required environment variables for cloud providers:

```bash
ANTHROPIC_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
PERPLEXITY_API_KEY=your_key_here
```

For local/open-weight models, prefer `api openai-compatible`. It works with any server that exposes an OpenAI-compatible `/v1/chat/completions` endpoint, including LM Studio, llama.cpp, Ollama's OpenAI-compatible endpoint, vLLM, and LocalAI. Set `OPENAI_COMPATIBLE_BASE_URL` in `.env` or for the launch command; both `ollama` and `openai-compatible` check connectivity at startup and fail immediately if the server is unreachable.

4. Make the scripts executable if needed:

```bash
chmod +x start_paper_summariser.sh stop_paper_summariser.sh
```

## Usage

### `summarise.py`

Defaults to Claude Code CLI:

```bash
python3 summarise.py
```

Equivalent explicit form:

```bash
python3 summarise.py cli claude
```

Recommended launch profiles:

```bash
# Claude Code with Sonnet
python3 summarise.py cli claude claude-sonnet-4-latest --effort high

# GitHub Copilot with Sonnet
python3 summarise.py cli copilot claude-sonnet-4.6 --effort high

# Local/open model through an OpenAI-compatible LM Studio server
OPENAI_COMPATIBLE_BASE_URL=http://localhost:1234/v1 python3 summarise.py api openai-compatible google/gemma-4-31b
```

Argument rules:

- No arguments: starts as `cli claude`
- Two arguments: `mode provider`
- Three arguments: `mode provider model`
- `--effort <level>` is optional in `cli` mode and supports `low`, `medium`, or `high`
- `--effort` is rejected in `api` mode
- Any invocation with exactly one positional argument or more than three positional arguments exits with usage guidance

### `start_paper_summariser.sh`

The wrapper script uses the same argument order:

```bash
./start_paper_summariser.sh
./start_paper_summariser.sh cli claude claude-sonnet-4-latest --effort high
./start_paper_summariser.sh cli copilot claude-sonnet-4.6 --effort high
OPENAI_COMPATIBLE_BASE_URL=http://localhost:1234/v1 ./start_paper_summariser.sh api openai-compatible google/gemma-4-31b
```

When no arguments are given, the script starts `python3 summarise.py` which defaults to `cli claude`.

### Monitoring and shutdown

```bash
tail -f logs/history.log
./stop_paper_summariser.sh
```

Summaries are written to `output/`. Processed papers are moved to `processed/`. Use symlinks if you want these to go elsewhere.

## Supported Providers

### CLI mode

| Provider | Requirement | Notes |
| --- | --- | --- |
| `claude` | `claude` binary on `PATH` | Supports `--effort low|medium|high` |
| `gemini` | `gemini` binary on `PATH` | Ignores `--effort` and uses Gemini defaults |
| `codex` | `codex` binary on `PATH` | Supports `--effort` via Codex config overrides |
| `copilot` | `copilot` binary on `PATH` | Supports `--effort low|medium|high` |
| `opencode` | `opencode` binary on `PATH` and a configured OpenCode model | Optional legacy path; use `api openai-compatible` for local/open models unless you specifically need OpenCode routing |

### API mode

| Provider | Requirement | Notes |
| --- | --- | --- |
| `claude` | `ANTHROPIC_API_KEY` | Anthropic API |
| `gemini` | `GOOGLE_API_KEY` | Google Gemini API |
| `openai` | `OPENAI_API_KEY` | OpenAI API |
| `perplexity` | `PERPLEXITY_API_KEY` | Perplexity API |
| `ollama` | Local Ollama server | No API key required; checks connectivity at startup |
| `openai-compatible` | Local or self-hosted `/v1/chat/completions` server | Requires `OPENAI_COMPATIBLE_BASE_URL` in `.env` (or `base_url` in provider config); `api_key_env` optional; checks connectivity at startup |

Each provider keeps its own default model or model-loading behaviour. Passing a third argument overrides that default.
In `cli` mode you can also pass `--effort low|medium|high`. Effort is currently unsupported in `api` mode.

### Local/open models

Use `api openai-compatible` for local/open-weight models served by LM Studio, llama.cpp, Ollama's OpenAI-compatible server, vLLM, LocalAI, or another `/v1/chat/completions` endpoint.

Configure the base URL in `.env`:

```bash
OPENAI_COMPATIBLE_BASE_URL=http://localhost:1234/v1
```

Then pass the exact model ID reported by your server:

```bash
python3 summarise.py api openai-compatible google/gemma-4-31b
```

For LM Studio, start a local server and use the `/v1` URL shown by LM Studio, usually `http://localhost:1234/v1`. Check `http://localhost:1234/v1/models` if you need the exact model ID. For llama.cpp, use the server's `/v1` URL, commonly `http://localhost:8080/v1`. For Ollama's OpenAI-compatible endpoint, use `http://localhost:11434/v1`.

OpenCode remains available as an optional CLI provider. If you use it, pass the model in OpenCode's `provider/model` format, for example `python3 summarise.py cli opencode ollama/llama3.2`.

## Tests

Run the focused unit tests with the project virtualenv:

```bash
./myenv/bin/python -m unittest tests/test_effort_support.py
```

## Failure Behaviour

- Invalid mode: exits immediately
- Provider unsupported in the selected mode: exits immediately
- CLI binary missing in `cli` mode: exits immediately
- API key missing in `api` mode: exits immediately
- Local API server unreachable (`ollama`, `openai-compatible`): exits immediately with a clear message
- Long noisy Markdown conversion lines from PDF extraction are removed before prompting
- Extracted-text prompts above the configured safety budget drop references, then appendices
- Summaries with footnote markers but no `## References` section trigger an automatic retry

`logs/history.log` records the selected mode, requested provider, provider backend class, and active model so CLI and API runs are easy to distinguish.

## Requirements

- zsh
- Python 3.9+
- `python-dotenv`
- `marker-pdf` for PDF text extraction when the selected provider does not support direct PDF upload
- The prerequisites for the specific mode/provider pair you choose
