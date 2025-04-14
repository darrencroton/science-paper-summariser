# Science Paper Summariser

A Python tool that uses Claude AI to automatically summarise scientific papers. The tool monitors an input directory for new PDFs or text files, processes them using Claude, and generates markdown summaries with extensive referencing back to the source material.

## Features

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
├── input/           # Place papers here for processing
├── output/          # Generated summaries appear here
├── processed/       # Completed papers are moved here
├── logs/           # Processing history and errors
└── project_knowledge/
    ├── astronomy-keywords.txt
    ├── paper-summary-template.md
    └── prompt.txt
```

## Setup

1. Create virtual environment:
   ```bash
   python -m venv myenv
   source myenv/bin/activate
   pip install -r requirements.txt
   ```

2. Add your Anthropic API key to `.env`:
   ```
   ANTHROPIC_API_KEY=your_key_here
   ```

3. Create required directories:
   ```bash
   mkdir -p input output processed logs
   ```

4. Make the start and stop scripts executable:
   ```bash
   chmod +x start_claude_summariser.sh stop_claude_summariser.sh
   ```

## Usage

1. Start the summariser with your preferred LLM provider:
   ```bash
   # Use Claude (default)
   ./start_claude_summariser.sh
   
   # Specify a different Claude model
   ./start_claude_summariser.sh claude claude-3-5-sonnet-20241022
   
   # Use Ollama with optional model specification
   ./start_claude_summariser.sh ollama mistral:7b
   
   # Use OpenAI (GPT-4o by default for scientific papers)
   ./start_claude_summariser.sh openai
   
   # Specify a different OpenAI model
   ./start_claude_summariser.sh openai gpt-4-turbo
   
   # Use Perplexity
   ./start_claude_summariser.sh perplexity
   ```

## Supported Models

This tool supports a variety of LLM providers and models. Please check llm_providers.py and update as needed.

### Claude (Anthropic)
- `claude-3-7-sonnet-20250219` (default)
- `claude-3-5-sonnet-20241022`
- `claude-3-opus-20240229`
- `claude-3-sonnet-20240229`
- `claude-3-haiku-20240307`

### OpenAI
- `gpt-4o` (default)
- `gpt-4o-mini`

### Perplexity
- `r1-1776` (default)
- `sonar`
- `sonar-deep-research`
- `sonar-reasoning-pro`
- `sonar-reasoning`
- `sonar-pro`

### Ollama
- `qwen2.5:14b-instruct-q8_0` (default)
- Any model available in your local Ollama installation

1. Place PDF or text files in the `input/` directory

2. Monitor `output/` for generated summaries. Processed papers end up in `processed/`

3. Stop the summariser:
   ```bash
   ./stop_claude_summariser.sh
   ```


## Requirements

- zsh shell
- Python 3.9+
- API keys (depending on provider):
  - Anthropic API key (for Claude)
  - OpenAI API key (for GPT models)
  - Perplexity API key (for Perplexity)
- Ollama installed locally (for Ollama provider)
- python-dotenv
- marker-pdf
