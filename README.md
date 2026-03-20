# Science Paper Summariser

A Python tool that watches an `input/` directory for PDFs or text files, sends each paper to an LLM using a strict astronomy-focused template, validates the response shape, writes a markdown summary to `output/`, and moves the original file to `processed/`.

Provider selection is explicit:

```bash
python3 summarise.py [mode] [provider] [model]
```

The program never switches mode or provider automatically. If the requested mode or provider cannot run, it exits immediately with a clear error.

## Features

- Explicit `cli` and `api` modes
- No automatic provider switching
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
  - Supported CLI providers: `claude`, `gemini`, `codex`, `copilot`
- `api` mode requirements:
  - Add the required credentials to `.env`
  - Supported API providers: `claude`, `gemini`, `openai`, `perplexity`, `ollama`
  - Required environment variables:

```bash
ANTHROPIC_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
PERPLEXITY_API_KEY=your_key_here
```

`ollama` does not use an API key, but it does require a reachable local Ollama server, which defaults to `http://localhost:11434`.

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

More examples:

```bash
python3 summarise.py cli gemini
python3 summarise.py cli codex gpt-5.4
python3 summarise.py cli copilot
python3 summarise.py api claude claude-sonnet-4-latest
python3 summarise.py api openai gpt-5.2
python3 summarise.py api gemini gemini-2.5-pro
python3 summarise.py api perplexity sonar-pro
python3 summarise.py api ollama llama3.2
```

Argument rules:

- No arguments: starts as `cli claude`
- Two arguments: `mode provider`
- Three arguments: `mode provider model`
- One argument or more than three arguments: exits with usage guidance

### `start_paper_summariser.sh`

The wrapper script uses the same argument order:

```bash
./start_paper_summariser.sh
./start_paper_summariser.sh cli gemini
./start_paper_summariser.sh api openai gpt-5.2
```

When no arguments are given, the script starts `python3 summarise.py` which defaults to `cli claude`.

### Monitoring and shutdown

```bash
tail -f logs/history.log
./stop_paper_summariser.sh
```

Summaries are written to `output/`. Processed papers are moved to `processed/`.

## Supported Providers

### CLI mode

| Provider | Requirement | Notes |
| --- | --- | --- |
| `claude` | `claude` binary on `PATH` | Default mode/provider combination |
| `gemini` | `gemini` binary on `PATH` | CLI only in this mode |
| `codex` | `codex` binary on `PATH` | OpenAI Codex CLI |
| `copilot` | `copilot` binary on `PATH` | GitHub Copilot CLI |

### API mode

| Provider | Requirement | Notes |
| --- | --- | --- |
| `claude` | `ANTHROPIC_API_KEY` | Anthropic API |
| `gemini` | `GOOGLE_API_KEY` | Google Gemini API |
| `openai` | `OPENAI_API_KEY` | OpenAI API |
| `perplexity` | `PERPLEXITY_API_KEY` | Perplexity API |
| `ollama` | Local Ollama server | No API key required |

Each provider keeps its own internal default model. Passing a third argument overrides that default.

## Failure Behaviour

- Invalid mode: exits immediately
- Provider unsupported in the selected mode: exits immediately
- CLI binary missing in `cli` mode: exits immediately
- API key missing in `api` mode: exits immediately
- No automatic CLI/API switching is attempted

`logs/history.log` records the selected mode, requested provider, provider backend class, and active model so CLI and API runs are easy to distinguish.

## Requirements

- zsh
- Python 3.9+
- `python-dotenv`
- `marker-pdf` for PDF text extraction when the selected provider does not support direct PDF upload
- The prerequisites for the specific mode/provider pair you choose
