#!/usr/bin/env python3
"""Run end-to-end tests: extract PDFs once, then test each provider.

Usage: python run_all_tests.py [provider1 provider2 ...]
  If no providers given, tests all 4 CLI providers.
  Logs to both stdout and test_validation/output/test.log
"""

import sys
import os
import time
import logging
from pathlib import Path

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from providers import create_provider
from summarise import (
    read_project_knowledge, read_input_file, create_system_prompt,
    create_user_prompt, strip_preamble, validate_summary, extract_metadata,
    create_base_filename,
)

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"
OUTPUT_DIR = Path(__file__).parent / "output"

PDFS = [
    EXAMPLES_DIR / "Harikane et al.pdf",
    EXAMPLES_DIR / "Torralba et al - 2026.pdf",
]


def setup_logging():
    """Log to both stdout and a file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_file = OUTPUT_DIR / "test.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
        ],
    )
    return log_file


def extract_pdfs(provider_for_extraction):
    """Extract all PDFs once using marker-pdf, return dict of {pdf_path: text}."""
    extracted = {}
    for pdf_path in PDFS:
        logging.info(f"Extracting: {pdf_path.name}")
        start = time.monotonic()
        content, error = read_input_file(pdf_path, provider_for_extraction)
        elapsed = time.monotonic() - start
        if error:
            logging.error(f"  FAILED to extract {pdf_path.name}: {error}")
            continue
        if isinstance(content, bytes):
            text = content.decode("utf-8", errors="ignore")
        else:
            text = content
        logging.info(f"  Extracted {len(text)} chars in {elapsed:.1f}s")
        extracted[pdf_path] = text
    return extracted


def run_test(provider_name, pdf_path, extracted_text, keywords, template):
    """Run a single provider test with pre-extracted text. Returns (success, output_path, error)."""
    logging.info(f"--- Testing {provider_name} on {pdf_path.name} ---")

    try:
        provider = create_provider(provider_name)
    except Exception as e:
        logging.error(f"  Failed to create provider: {e}")
        return False, None, str(e)

    logging.info(f"  Provider class: {provider.__class__.__name__}")

    # Build prompts using pre-extracted text
    system_prompt = create_system_prompt(keywords)
    user_prompt = create_user_prompt(extracted_text, template, is_pdf=True)
    logging.info(f"  Prompt size: system={len(system_prompt)}, user={len(user_prompt)} chars")

    # Call LLM
    start = time.monotonic()
    try:
        summary = provider.process_document(
            content=extracted_text, is_pdf=False,
            system_prompt=system_prompt, user_prompt=user_prompt,
            max_tokens=12288,
        )
    except Exception as e:
        elapsed = time.monotonic() - start
        logging.error(f"  LLM call FAILED after {elapsed:.1f}s: {e}")
        return False, None, str(e)
    elapsed = time.monotonic() - start
    logging.info(f"  LLM returned {len(summary)} chars in {elapsed:.1f}s")

    # Post-process
    summary = strip_preamble(summary)
    try:
        validate_summary(summary)
        logging.info("  Validation: PASSED")
    except Exception as e:
        logging.warning(f"  Validation: FAILED - {e}")

    # Extract metadata
    try:
        title, authors, year = extract_metadata(summary)
        logging.info(f"  Metadata: {title[:50]}... | {authors[:2]} | {year}")
    except Exception as e:
        logging.warning(f"  Metadata extraction failed: {e}")

    # Save with production-style metadata filename, prefixed by provider
    base_filename = create_base_filename(title, authors, year, pdf_path)
    out_file = OUTPUT_DIR / f"{provider_name} - {base_filename}.md"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(summary)
    logging.info(f"  Saved to: {out_file}")

    return True, out_file, None


def main():
    providers_to_test = sys.argv[1:] if len(sys.argv) > 1 else ["claude", "codex", "gemini", "copilot"]

    log_file = setup_logging()
    logging.info(f"Testing providers: {providers_to_test}")
    logging.info(f"PDFs: {[p.name for p in PDFS]}")
    logging.info(f"Log file: {log_file}")

    # Load project knowledge once
    keywords, template = read_project_knowledge()
    logging.info(f"Keywords: {len(keywords)} chars, Template: {len(template)} chars")

    # Extract PDFs once (CLI providers don't support direct PDF)
    logging.info("=== Phase 1: Extracting PDFs with marker-pdf ===")
    # Use any CLI provider for extraction (they all go through marker-pdf)
    dummy_provider = create_provider(providers_to_test[0])
    extracted = extract_pdfs(dummy_provider)
    logging.info(f"Extracted {len(extracted)}/{len(PDFS)} PDFs")

    if not extracted:
        logging.error("No PDFs extracted — aborting")
        sys.exit(1)

    # Run each provider sequentially
    logging.info("=== Phase 2: Testing providers ===")
    results = []
    for provider_name in providers_to_test:
        for pdf_path, text in extracted.items():
            success, out_file, error = run_test(
                provider_name, pdf_path, text, keywords, template
            )
            results.append({
                "provider": provider_name,
                "pdf": pdf_path.name,
                "success": success,
                "output": out_file,
                "error": error,
            })

    # Summary
    logging.info("=== Results Summary ===")
    for r in results:
        status = "OK" if r["success"] else f"FAIL: {r['error']}"
        logging.info(f"  {r['provider']:10s} | {r['pdf']:30s} | {status}")

    passed = sum(1 for r in results if r["success"])
    logging.info(f"\n{passed}/{len(results)} tests passed")


if __name__ == "__main__":
    main()
