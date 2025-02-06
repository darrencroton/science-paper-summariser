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
   pip install anthropic pypdf2 python-dotenv
   ```

2. Add your Anthropic API key to `.env`:
   ```
   ANTHROPIC_API_KEY=your_key_here
   ```

3. Create required directories:
   ```bash
   mkdir -p input output processed logs
   ```

## Usage

1. Start the summariser:
   ```bash
   ./start_claude_summariser.sh
   ```

2. Place PDF or text files in the `input/` directory

3. Monitor `output/` for generated summaries

4. Stop the summariser:
   ```bash
   ./stop_claude_summariser.sh
   ```

## Requirements

- Python 3.8+
- Anthropic API key
- PyPDF2
- python-dotenv
