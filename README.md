# Science Paper Summariser

A Python tool that uses LLMs (Claude, GPT, Gemini, etc.) to automatically summarise scientific papers, following an exact template and hashtag list. The tool monitors an input directory for new PDFs or text files, processes them using your chosen model, and generates markdown summaries with extensive referencing back to the source material.

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
    └── paper-summary-template.md
```

## Setup

1. Create virtual environment:
   ```bash
   python -m venv myenv
   source myenv/bin/activate
   pip install -r requirements.txt
   ```

2. Add your LLM provider API key(s) to `.env` (see `.env.template`):
   ```
   ANTHROPIC_API_KEY=your_key_here
   OPENAI_API_KEY=your_key_here
   PERPLEXITY_API_KEY=your_key_here
   GOOGLE_API_KEY=your_key_here
   ```

3. Create required directories:
   ```bash
   mkdir -p input output processed logs
   ```
   Tip: create symbolic links to where you want to read/write the input/output

4. Make the start and stop scripts executable:
   ```bash
   chmod +x start_paper_summariser.sh stop_paper_summariser.sh
   ```

## Usage

1. Start the summariser with your preferred LLM provider (see below for defaults):
   ```bash
   # Use Claude (default)
   ./start_paper_summariser.sh
   
   # Specify a different Claude model
   ./start_paper_summariser.sh claude claude-3-5-sonnet-20241022
   
   # Use Ollama with optional model specification
   ./start_paper_summariser.sh ollama llama3.1:8b-instruct-q8_0
   
   # Use OpenAI
   ./start_paper_summariser.sh openai
   
   # Specify a different OpenAI model
   ./start_paper_summariser.sh openai gpt-4o
   
   # Use Perplexity
   ./start_paper_summariser.sh perplexity
   
   # Use Google Gemini
   ./start_paper_summariser.sh gemini
   
   # Specify a different Gemini model
   ./start_paper_summariser.sh gemini gemini-1.5-pro
   ```

2. Place PDF or text files in the `input/` directory

3. Monitor `output/` for generated summaries. Processed papers end up in `processed/`

4. Stop the summariser:
   ```bash
   ./stop_paper_summariser.sh
   ```

## Supported Models

This tool supports a variety of LLM providers and models. Please check `llm_providers.py` and update as needed. 

NOTE: In my experience, Claude Sonnet 3.5/3.7 gives the best results, followed by GPT 4.1. The rest can be hit and miss. I'm yet to find an Ollama model that fits in 32GB Macbook Pro shared memory and consistantly gives good results (although Qwen 2.5 14b isn't terrible).

### Claude (Anthropic)
- `claude-3-7-sonnet-20250219` (default)
- `claude-3-5-sonnet-20241022`
- `claude-3-opus-20240229`
- `claude-3-sonnet-20240229`
- `claude-3-haiku-20240307`

### OpenAI
- `gpt-4.1` (default)
- `gpt-4.1-mini`
- `gpt-4.1-nano`
- `gpt-4o`
- `gpt-4o-mini`

### Perplexity
- `r1-1776` (default)
- `sonar`
- `sonar-deep-research`
- `sonar-reasoning-pro`
- `sonar-reasoning`
- `sonar-pro`

### Google Gemini
- `gemini-2.5-pro-exp-03-25` (default)
- `gemini-2.5-flash-preview-04-17`
- `gemini-2.0-flash`
- `gemini-1.5-pro`

### Ollama
- `qwen2.5:14b-instruct-q8_0` (default)
- `llama3.1:8b-instruct-q8_0`
- ... any model available in your local Ollama installation


## Requirements

- zsh shell
- Python 3.9+
- API keys (depending on provider):
  - Anthropic API key (for Claude)
  - OpenAI API key (for GPT models)
  - Perplexity API key (for Perplexity)
  - Google API key (for Gemini)
- Ollama installed locally (for Ollama provider)
- python-dotenv
- marker-pdf
