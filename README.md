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
   ./start_paper_summariser.sh claude claude-haiku-4-5
   
   # Use Gemini
   ./start_paper_summariser.sh gemini
   
   # Specify a different Gemini model
   ./start_paper_summariser.sh gemini gemini-2.5-flash
   
   # Use OpenAI
   ./start_paper_summariser.sh openai
   
   # Specify a different OpenAI model
   ./start_paper_summariser.sh openai gpt-5-mini
   
   # Use Perplexity
   ./start_paper_summariser.sh perplexity

   # Use Ollama with optional model specification
   ./start_paper_summariser.sh ollama llama3.1:8b-instruct-q8_0
   ```

2. Place PDF or text files in the `input/` directory

3. Monitor `output/` for generated summaries. Processed papers end up in `processed/`

4. Stop the summariser:
   ```bash
   ./stop_paper_summariser.sh
   ```

## Supported Models

This tool supports a variety of LLM providers and models. Please check `llm_providers.py` and update as needed. 

NOTE: In my experience, Claude Sonnet 4, Gemini 2.5 Pro, and GPT 5.2 give the best results. The rest can be hit and miss. I'm yet to find an Ollama model that fits in 32GB Macbook Pro shared memory and consistantly gives good results (although Qwen 2.5 14b isn't terrible).

### Claude (default)
- `claude-sonnet-4-6` (default)
- `claude-opus-4-6`
- `claude-haiku-4-5`

### Google Gemini
- `gemini-2.5-pro` (default)
- `gemini-2.5-flash`
- `gemini-2.5-flash-lite`
- `gemini-3-flash-preview` (preview)
- `gemini-3.1-pro-preview` (preview)

### OpenAI
- `gpt-5.2` (default)
- `gpt-5-mini`
- `gpt-5-nano`
- `gpt-4.1`
- `gpt-4o`
- `gpt-4o-mini`

### Perplexity
- `sonar-pro` (default)
- `sonar`
- `sonar-reasoning-pro`
- `sonar-deep-research`

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
