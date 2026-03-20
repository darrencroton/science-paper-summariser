#!/usr/bin/env python3
"""Test a single provider against a single PDF.

Usage: python run_test.py <provider_name> <pdf_path> <output_dir>
"""

import sys
import os
import time
import logging

# Add parent dir to path so we can import from the project
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from providers import create_provider
from summarise import (
    read_project_knowledge, read_input_file, create_system_prompt,
    create_user_prompt, strip_preamble, validate_summary, extract_metadata,
    create_base_filename,
)
from pathlib import Path

def main():
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <provider> <pdf_path> <output_dir>")
        sys.exit(1)

    provider_name = sys.argv[1]
    pdf_path = Path(sys.argv[2])
    output_dir = Path(sys.argv[3])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logging.info(f"Testing provider '{provider_name}' with '{pdf_path.name}'")

    # Create provider
    provider = create_provider(provider_name)
    logging.info(f"Provider: {provider.__class__.__name__}")

    # Load project knowledge
    keywords, template = read_project_knowledge()

    # Read file
    content, error = read_input_file(pdf_path, provider)
    if error:
        logging.error(f"Failed to read: {error}")
        sys.exit(1)

    # Build prompts
    system_prompt = create_system_prompt(keywords)
    is_pdf = pdf_path.suffix.lower() == ".pdf"
    if is_pdf and provider.supports_direct_pdf():
        paper_text = ""
    elif isinstance(content, bytes):
        paper_text = content.decode("utf-8", errors="ignore")
    else:
        paper_text = content
    user_prompt = create_user_prompt(paper_text, template, is_pdf)

    logging.info(f"Prompt size: system={len(system_prompt)}, user={len(user_prompt)} chars")

    # Call LLM
    start = time.monotonic()
    try:
        summary = provider.process_document(
            content=content, is_pdf=is_pdf,
            system_prompt=system_prompt, user_prompt=user_prompt,
            max_tokens=12288,
        )
    except Exception as e:
        logging.error(f"LLM call failed: {e}")
        sys.exit(1)
    elapsed = time.monotonic() - start
    logging.info(f"LLM returned {len(summary)} chars in {elapsed:.1f}s")

    # Post-process
    summary = strip_preamble(summary)
    validate_summary(summary)

    # Extract metadata for filename
    title, authors, year = extract_metadata(summary)
    logging.info(f"Metadata: {title[:50]}... | {authors[:2]} | {year}")

    # Save with production-style metadata filename, prefixed by provider
    base_filename = create_base_filename(title, authors, year, pdf_path)
    out_file = output_dir / f"{provider_name} - {base_filename}.md"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(summary)
    logging.info(f"Saved to: {out_file}")

if __name__ == "__main__":
    main()
