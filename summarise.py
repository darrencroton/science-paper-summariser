"""Science Paper Summariser — main orchestration module.

Monitors an input directory for PDFs and text files, sends them to an LLM
with a strict astronomy-focused summary template, validates the output,
and writes markdown summaries to the output directory. Processed papers
are moved to the processed directory. Runs as a background nohup process
via shell scripts.
"""

import os

# Required for marker-pdf on Apple Silicon — must be set before PyTorch is imported.
# Without this, unsupported MPS operations fail silently or produce empty tensors.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
import re
import time
import logging
import signal
import concurrent.futures
from dotenv import load_dotenv
from pathlib import Path
from providers import create_provider

# --- Configuration and Paths ---
load_dotenv()

# Provider and optional model from command-line arguments
LLM_PROVIDER = sys.argv[1] if len(sys.argv) > 1 else "claude"
LLM_MODEL = sys.argv[2] if len(sys.argv) > 2 else None

# Directory paths relative to the script location
SCRIPT_DIR = Path(__file__).parent.absolute()
INPUT_DIR = SCRIPT_DIR / "input"
OUTPUT_DIR = SCRIPT_DIR / "output"
LOGS_DIR = SCRIPT_DIR / "logs"
DONE_DIR = SCRIPT_DIR / "processed"
KNOWLEDGE_DIR = SCRIPT_DIR / "project_knowledge"

# Log file paths
PROGRESS_FILE = LOGS_DIR / "completed.log"
FAILED_FILE = LOGS_DIR / "failed.log"
PROMPT_DEBUG_FILE = LOGS_DIR / "prompt.txt"

# --- Global shutdown flag ---
shutdown_requested = False

# --- Lazy-loaded marker-pdf model cache ---
_marker_models = None
MARKER_TIMEOUT = 300  # seconds before marker-pdf extraction is abandoned


def _get_marker_models():
    """Lazy-load and cache marker-pdf models. Only called when text extraction is needed."""
    global _marker_models
    if _marker_models is None:
        from marker.models import create_model_dict
        logging.info("Loading marker-pdf models (one-time initialisation)...")
        _marker_models = create_model_dict()
        logging.info("marker-pdf models loaded.")
    return _marker_models


# --- Signal handling ---

def handle_shutdown_signal(signum, frame):
    """Signal handler for SIGINT and SIGTERM. Sets shutdown flag for graceful exit."""
    global shutdown_requested
    if not shutdown_requested:
        signal_name = signal.Signals(signum).name
        logging.info(f"\n--- {signal_name} received. Initiating graceful shutdown... ---")
        shutdown_requested = True
    else:
        logging.info(
            f"--- Shutdown already in progress "
            f"(Signal {signal.Signals(signum).name} received again) ---"
        )


def interruptible_sleep(seconds, interval=0.5):
    """Sleep for the given duration, checking for shutdown signal every interval.

    Returns True if shutdown was requested during the sleep.
    """
    wait_start = time.monotonic()
    while not shutdown_requested and time.monotonic() - wait_start < seconds:
        time.sleep(interval)
    return shutdown_requested


# --- State management ---

def setup_logging():
    """Configure logging to stdout (redirected to history.log by the start script)."""
    LOGS_DIR.mkdir(exist_ok=True)
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)


def load_progress():
    """Load the set of successfully processed filenames from the progress file."""
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                return {line.strip() for line in f if line.strip()}
        except Exception as e:
            logging.error(f"Error loading progress file {PROGRESS_FILE}: {e}")
            return set()
    return set()


def load_failed_files():
    """Load the set of filenames that permanently failed processing."""
    if FAILED_FILE.exists():
        try:
            with open(FAILED_FILE, "r", encoding="utf-8") as f:
                failed = set()
                for line in f:
                    if line.strip() and "|" in line:
                        failed.add(line.strip().split("|", 1)[0])
                return failed
        except Exception as e:
            logging.error(f"Error loading failed files list {FAILED_FILE}: {e}")
            return set()
    return set()


def save_progress(processed_files):
    """Save the updated set of successfully processed filenames."""
    try:
        with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
            for filename in sorted(processed_files):
                f.write(f"{filename}\n")
    except Exception as e:
        logging.error(f"Error saving progress to {PROGRESS_FILE}: {e}")


def add_to_failed_files(filename, error):
    """Append a filename and error details to the permanently failed list."""
    try:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        error_str = str(error).replace("\n", " ").replace("\r", "")
        with open(FAILED_FILE, "a", encoding="utf-8") as f:
            f.write(f"{filename}|{timestamp}|{error_str}\n")
        logging.info(f"Added {filename} to failed files list.")
    except Exception as e:
        logging.error(f"Error adding {filename} to failed list {FAILED_FILE}: {e}")


# --- File reading ---

def read_project_knowledge():
    """Read keywords and template files needed for prompts.

    Raises an exception if critical files are missing.
    """
    try:
        keywords_path = KNOWLEDGE_DIR / "astronomy-keywords.txt"
        template_path = KNOWLEDGE_DIR / "paper-summary-template.md"
        with open(keywords_path, "r", encoding="utf-8") as f:
            keywords = f.read()
        with open(template_path, "r", encoding="utf-8") as f:
            template = f.read()
        return keywords, template
    except FileNotFoundError as e:
        logging.critical(f"CRITICAL: Project knowledge file not found: {e}. Cannot continue.")
        raise
    except Exception as e:
        logging.critical(f"CRITICAL: Failed to read project knowledge files: {e}")
        raise


def read_input_file(file_path, provider):
    """Read content from an input file (PDF or TXT).

    Uses marker-pdf for text extraction when the provider does not support
    direct PDF upload. Marker-pdf models are lazy-loaded and cached.

    Returns (content, error_message). Content is bytes for direct PDF upload,
    str for extracted text or text files.
    """
    try:
        file_size = file_path.stat().st_size
        if file_size > 100 * 1024 * 1024:
            return None, f"File too large: {file_size / 1024 / 1024:.1f}MB (max 100MB)"

        file_suffix = file_path.suffix.lower()

        if file_suffix == ".pdf":
            if provider.supports_direct_pdf():
                logging.info(f"Reading PDF binary for direct upload: {file_path.name}")
                with open(file_path, "rb") as f:
                    return f.read(), None
            else:
                logging.info(f"Extracting text from PDF using marker-pdf: {file_path.name}")
                from marker.output import text_from_rendered
                from marker.config.parser import ConfigParser

                config = {
                    "output_format": "markdown",
                    "disable_image_extraction": True,
                    "use_llm": False,
                }
                config_parser = ConfigParser(config)
                converter_cls = config_parser.get_converter_cls()
                converter = converter_cls(
                    config=config_parser.generate_config_dict(),
                    artifact_dict=_get_marker_models(),
                    processor_list=config_parser.get_processors(),
                    renderer=config_parser.get_renderer(),
                    llm_service=config_parser.get_llm_service(),
                )
                logging.info(
                    f"Running marker-pdf extraction "
                    f"(timeout: {MARKER_TIMEOUT}s): {file_path.name}"
                )
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(converter, str(file_path))
                    try:
                        rendered = future.result(timeout=MARKER_TIMEOUT)
                    except concurrent.futures.TimeoutError:
                        raise RuntimeError(
                            f"marker-pdf timed out after {MARKER_TIMEOUT}s "
                            f"processing {file_path.name}"
                        )
                text, _, _ = text_from_rendered(rendered)
                logging.info(f"Extracted ~{len(text.split())} words from PDF")
                return text, None

        elif file_suffix == ".txt":
            logging.info(f"Reading text file: {file_path.name}")
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read(), None
        else:
            return None, f"Unsupported file type: {file_path.suffix}"

    except Exception as e:
        logging.error(f"Error reading input file {file_path.name}: {e}", exc_info=True)
        return None, f"File read/processing error: {e}"


# --- Prompt construction ---

def create_system_prompt(keywords):
    """Construct the system prompt defining the LLM's role, rules, and knowledge base."""
    return (
        "<role>\n"
        "You are an esteemed professor of astrophysics at Harvard University "
        "specializing in analyzing research papers. Your expertise includes:\n"
        "- Identifying key scientific results and their significance\n"
        "- Writing in clear technical UK English\n"
        "- Supporting all claims with precise quotations\n"
        "- Using LaTeX for mathematical expressions\n"
        "</role>\n\n"
        "<rules>\n"
        "1. Write only in UK English using clear technical language\n"
        "2. Use markdown formatting throughout\n"
        "3. Use LaTeX for all mathematical expressions\n"
        "4. Only include content from the provided paper\n"
        "5. Every bullet point must have a supporting footnote\n"
        "6. Footnotes must contain EXACT quotes - never paraphrase\n"
        "7. Always enclose footnote quotes in quotation marks\n"
        "8. Include page/section reference for every quote\n"
        "9. Use bold for key terms on first mention\n"
        "10. Use italics for emphasis and paper names\n"
        "11. If you cannot find an exact supporting quote, do not make the statement\n"
        "12. ALWAYS include a Glossary section with a table of technical terms\n"
        "</rules>\n\n"
        "<knowledgeBase>\n"
        f"Available astronomy keywords by category:\n{keywords}\n"
        "</knowledgeBase>"
    )


def create_user_prompt(paper_text, template, is_pdf=False):
    """Construct the user prompt with the task, template, and optionally the paper text.

    If paper_text is empty, the LLM receives the paper via other means (e.g. direct PDF upload).
    """
    base_prompt = (
        "<task>\n"
        "Summarize this research paper following these EXACT requirements:\n\n"
        "<format>\n"
        "1. THE VERY FIRST LINE must be the paper title as '# Title'\n"
        "2. NO TEXT before the title - not even a greeting\n"
        "3. Below title, exactly one blank line, then:\n"
        "   - Line starting 'Authors: ' with FULL author list\n"
        "   - Line starting 'Published: ' with month, year, and link\n"
        "   - One blank line before starting sections\n"
        "4. Include EVERY author (surname and initials with period, comma separated)\n"
        "5. Never truncate author list with 'et al.'\n"
        "6. MUST include publication month and year\n"
        "7. Follow the exact section order specified\n"
        "</format>\n\n"
        "<template>\n"
        f"Use this exact structure:\n{template}\n"
        "</template>\n\n"
        "<tags>\n"
        "The Tags section must have two parts:\n"
        "1. First line: Hashtags for telescopes, surveys, datasets, models (proper nouns only)\n"
        "2. Second line: Science area hashtags (use ONLY provided keywords, only the best ones)\n"
        "</tags>\n"
        "</task>\n\n"
    )

    if paper_text:
        base_prompt += (
            "<input>\n"
            "Paper to summarize:\n\n"
            f"---BEGIN PAPER---\n{paper_text}\n---END PAPER---\n"
            "</input>"
        )
    return base_prompt


# --- LLM call with retry ---

def _write_debug_prompt(system_prompt, user_prompt):
    """Write the full prompt to the debug file for troubleshooting."""
    try:
        full_prompt = f"SYSTEM PROMPT\n{system_prompt}\n\n---\n\nUSER PROMPT\n{user_prompt}"
        with open(PROMPT_DEBUG_FILE, "w", encoding="utf-8") as f:
            f.write(full_prompt)
        logging.info(f"Debug prompt written to {PROMPT_DEBUG_FILE}")
    except Exception as e:
        logging.warning(f"Could not write debug prompt file: {e}")


def _call_llm_with_retry(provider, content, is_pdf, system_prompt, user_prompt, max_retries=3):
    """Call the LLM with retry logic and exponential backoff.

    Returns the summary text on success.
    Raises the last exception after all retries are exhausted.
    Raises InterruptedError if shutdown is requested.
    """
    last_error = None
    for attempt in range(max_retries):
        if shutdown_requested:
            raise InterruptedError("Shutdown requested before LLM call")

        try:
            logging.info(f"Attempt {attempt + 1}/{max_retries} calling LLM...")
            summary = provider.process_document(
                content=content,
                is_pdf=is_pdf,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=12288,
            )
            if not summary or not summary.strip():
                raise ValueError("LLM returned empty or whitespace-only response")

            logging.info(f"LLM call successful (received ~{len(summary)} chars)")
            return summary

        except Exception as e:
            provider_name = provider.__class__.__name__
            last_error = e
            error_msg = (
                f"Attempt {attempt + 1} failed — {provider_name}: "
                f"{e.__class__.__name__}: {e}"
            )
            logging.error(error_msg)

            if attempt < max_retries - 1:
                wait_time = 2 ** (attempt + 1)
                logging.info(f"Waiting {wait_time}s before retry...")
                if interruptible_sleep(wait_time):
                    raise InterruptedError("Shutdown requested during retry wait")
            else:
                logging.error(f"All {max_retries} attempts failed.")

    raise last_error


# --- Post-processing ---

def strip_preamble(summary_content):
    """Remove any text before the first Markdown heading ('# ')."""
    lines = summary_content.split("\n")
    title_index = -1
    for i, line in enumerate(lines):
        if line.strip().startswith("# "):
            title_index = i
            break

    if title_index > 0:
        preamble = "\n".join(lines[:title_index])
        logging.info(
            f"Removed {len(preamble.splitlines())} lines of preamble before paper title."
        )
        return "\n".join(lines[title_index:])
    elif title_index == 0:
        logging.info("Summary starts correctly with title line.")
        return summary_content
    else:
        logging.warning("Could not find title heading ('# ') to strip preamble.")
        return summary_content


def validate_summary(summary_content):
    """Validate the generated summary structure and log results.

    This is a safety-net check — it logs warnings but does not reject summaries.
    """
    lines = [line.strip() for line in summary_content.split("\n") if line.strip()]
    if not lines:
        logging.warning("VALIDATION WARNING: Summary content is empty.")
        return

    # Structural checks
    start_with_title = lines[0].startswith("# ")
    year_found = any(re.search(r"\b(19|20)\d{2}\b", line) for line in lines[:5])
    author_complete = not any(
        "et al" in line.lower() for line in lines[:5] if line.startswith("Authors:")
    )
    bullet_count = sum(
        1 for line in lines
        if (line.startswith("- ") or line.startswith("* "))
        and not line.startswith("[^")
    )
    footnote_count = sum(1 for line in lines if line.startswith("[^"))
    footnote_lines = [line for line in lines if line.startswith("[^")]
    properly_quoted = sum(
        1 for line in footnote_lines if '"' in line and line.count('"') >= 2
    )
    all_quoted = (properly_quoted == footnote_count) if footnote_count > 0 else True

    has_glossary = any(line.startswith("## Glossary") for line in lines)
    has_tags = False
    has_two_tag_lines = False
    try:
        tags_index = next(i for i, line in enumerate(lines) if line.startswith("## Tags"))
        has_tags = True
        tag_content = [line for line in lines[tags_index + 1 :] if line][:2]
        has_two_tag_lines = len(tag_content) == 2 and all(
            l.startswith("#") for l in tag_content
        )
    except StopIteration:
        pass

    # Log results
    logging.info("Validation results:")
    logging.info(f"  Starts with title: {start_with_title}")
    logging.info(f"  Year found: {year_found}")
    logging.info(f"  Author list complete: {author_complete}")
    logging.info(f"  Bullets: {bullet_count}, Footnotes: {footnote_count}")
    logging.info(
        f"  All footnotes quoted: {all_quoted} ({properly_quoted}/{footnote_count})"
    )
    logging.info(f"  Has Glossary: {has_glossary}")
    logging.info(f"  Has Tags (two lines): {has_tags} ({has_two_tag_lines})")

    # Specific warnings
    if not start_with_title:
        logging.warning("VALIDATION WARNING: Summary does not start with title heading '# '")
    if not year_found:
        logging.warning("VALIDATION WARNING: No year found near top of summary")
    if not author_complete:
        logging.warning(
            "VALIDATION WARNING: Author list might be truncated ('et al.' found)"
        )
    if bullet_count != footnote_count:
        logging.warning(
            f"VALIDATION WARNING: Bullet/footnote mismatch "
            f"({bullet_count} vs {footnote_count})"
        )
    if not all_quoted and footnote_count > 0:
        logging.warning(
            f"VALIDATION WARNING: Not all footnotes properly quoted "
            f"({properly_quoted}/{footnote_count})"
        )
    if not has_glossary:
        logging.warning("VALIDATION WARNING: Missing '## Glossary' section")
    if not has_tags:
        logging.warning("VALIDATION WARNING: Missing '## Tags' section")
    elif not has_two_tag_lines:
        logging.warning("VALIDATION WARNING: '## Tags' section missing two hashtag lines")


# --- Metadata and filenames ---

def extract_metadata(summary_content):
    """Extract author surnames, year, and title from the summary markdown."""
    lines = [l.strip() for l in summary_content.split("\n") if l.strip()]
    if not lines:
        return "Untitled", [], None

    title = lines[0].replace("# ", "", 1).strip() if lines[0].startswith("# ") else "Untitled"
    year = None
    authors_surnames = []

    for line in lines[1:5]:
        if line.startswith("Authors: "):
            author_line = line.replace("Authors: ", "")
            for part in author_line.split(","):
                name_parts = part.strip().split()
                if name_parts:
                    surname = name_parts[0]
                    if surname and not surname.endswith(".") and len(surname) > 1:
                        authors_surnames.append(surname)
        if not year:
            if year_match := re.search(r"\b(19|20)\d{2}\b", line):
                year = year_match.group()

    return title, authors_surnames, year


def format_authors(authors_surnames):
    """Format author surnames for filenames (e.g. 'Smith et al.')."""
    count = len(authors_surnames)
    if count == 0:
        return "UnknownAuthor"
    if count == 1:
        return authors_surnames[0]
    if count == 2:
        return f"{authors_surnames[0]} and {authors_surnames[1]}"
    return f"{authors_surnames[0]} et al."


def sanitize_filename(filename_part):
    """Remove or replace characters unsafe for filenames and limit length."""
    if not isinstance(filename_part, str):
        filename_part = str(filename_part)

    unsafe_chars = r'<>:"/\\|?*' + "\n\r\t"
    for char in unsafe_chars:
        filename_part = filename_part.replace(char, "")
    filename_part = filename_part.strip().rstrip(".")

    if not filename_part:
        return "sanitized_empty"

    max_bytes = 240
    while len(filename_part.encode("utf-8")) > max_bytes:
        filename_part = filename_part[:-1]
        if not filename_part:
            return "sanitized_short"

    return filename_part.strip()


def create_base_filename(title, authors_surnames, year, input_path):
    """Create a sanitised filename stem from metadata: 'Author(s) - Year - Title'."""
    if year:
        formatted_authors = format_authors(authors_surnames)
        safe_authors = sanitize_filename(formatted_authors)
        safe_year = sanitize_filename(year)
        safe_title = sanitize_filename(title[:80])

        parts = [part for part in [safe_authors, safe_year, safe_title] if part]
        if parts:
            base_name = " - ".join(parts)
            logging.info(f"Filename stem: {base_name}")
            return base_name
        else:
            logging.warning("Could not create filename from metadata (all parts empty).")

    fallback_stem = sanitize_filename(input_path.stem)
    logging.warning(f"Using fallback filename stem: {fallback_stem}")
    return fallback_stem if fallback_stem else "fallback_filename"


# --- File output ---

def save_summary(summary_content, base_filename):
    """Save the generated summary to a markdown file in the output directory.

    Returns the output Path on success.
    Raises on failure so the caller can avoid moving the original file.
    """
    output_filename = f"{base_filename}.md"
    output_path = OUTPUT_DIR / output_filename
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(summary_content)
    logging.info(f"Summary saved to: {output_path}")
    return output_path


def move_to_done(file_path, base_filename):
    """Move the processed input file to the processed directory with a metadata-based name.

    Returns the destination Path, or None on failure.
    """
    try:
        DONE_DIR.mkdir(exist_ok=True)
        target_filename = f"{base_filename}{file_path.suffix}"
        dest_path = DONE_DIR / target_filename

        counter = 1
        original_dest = dest_path
        while dest_path.exists():
            target_filename = f"{base_filename}_{counter}{file_path.suffix}"
            dest_path = DONE_DIR / target_filename
            counter += 1
            if counter == 2:
                logging.warning(f"Destination {original_dest.name} exists, adding counter.")

        file_path.rename(dest_path)
        logging.info(f"Moved {file_path.name} -> {target_filename}")
        return dest_path

    except Exception as e:
        logging.error(f"Error moving {file_path.name} to done: {e}", exc_info=True)
        return None


# --- File scanning ---

def get_pending_files(input_dir, processed_log, failed_log):
    """Scan input directory for unprocessed files (.pdf, .txt). Skips zero-byte files."""
    pending = []
    try:
        for f in input_dir.glob("*.*"):
            if f.is_file() and f.suffix.lower() in [".pdf", ".txt"]:
                if f.name not in processed_log and f.name not in failed_log:
                    try:
                        if f.stat().st_size > 0:
                            pending.append(f)
                        else:
                            logging.warning(f"Skipping zero-byte file: {f.name}")
                    except OSError as e:
                        logging.warning(f"Could not stat {f.name}: {e}")
    except Exception as e:
        logging.error(f"Error scanning input directory {input_dir}: {e}")
    return pending


# --- Core processing ---

def process_file(file_path, keywords, template, provider):
    """Process a single input file through the full pipeline.

    Steps: read file -> build prompts -> call LLM (with retry) ->
    strip preamble -> validate -> save summary -> move original.

    Returns (success, original_filename, error_message_or_None).
    """
    is_pdf = file_path.suffix.lower() == ".pdf"
    logging.info(
        f"Using provider: {provider.__class__.__name__}"
        + (f", Model: {provider.model}" if provider.model else "")
    )

    try:
        # 1. Read file content
        content, error = read_input_file(file_path, provider)
        if error:
            return False, file_path.name, error

        logging.info(f"Content type: {type(content).__name__}")

        # 2. Build prompts
        system_prompt = create_system_prompt(keywords)

        if is_pdf and provider.supports_direct_pdf():
            paper_text = ""  # PDF binary sent directly to provider
        elif isinstance(content, bytes):
            paper_text = content.decode("utf-8", errors="ignore")
        else:
            paper_text = content

        user_prompt = create_user_prompt(paper_text, template, is_pdf)

        prompt_info = (
            f"System prompt: {len(system_prompt)} chars. "
            f"User prompt: {len(user_prompt)} chars."
        )
        if paper_text:
            prompt_info += f" Paper text: {len(paper_text)} chars."
        logging.info(prompt_info)

        _write_debug_prompt(system_prompt, user_prompt)

        # 3. Call LLM with retry
        summary = _call_llm_with_retry(provider, content, is_pdf, system_prompt, user_prompt)

        # 4. Post-process
        summary = strip_preamble(summary)
        validate_summary(summary)

        # 5. Extract metadata once, use for both save and move
        title, authors, year = extract_metadata(summary)
        base_filename = create_base_filename(title, authors, year, file_path)

        save_summary(summary, base_filename)

        moved = move_to_done(file_path, base_filename)
        if not moved:
            return (
                False,
                file_path.name,
                f"Processed but failed to move original file {file_path.name}",
            )

        return True, file_path.name, None

    except InterruptedError:
        return False, file_path.name, "Shutdown requested"
    except Exception as e:
        error_msg = f"{e.__class__.__name__}: {e}"
        logging.error(f"Processing error for {file_path.name}: {error_msg}", exc_info=True)
        return False, file_path.name, error_msg


# --- Main loop ---

def main():
    """Main entry point: set up, create provider, monitor input directory, process files."""
    signal.signal(signal.SIGINT, handle_shutdown_signal)
    signal.signal(signal.SIGTERM, handle_shutdown_signal)

    try:
        setup_logging()
        model_info = f" with model {LLM_MODEL}" if LLM_MODEL else ""
        logging.info("--- Science Paper Summariser Starting ---")
        logging.info(f"Process ID: {os.getpid()}")
        logging.info(f"Provider: {LLM_PROVIDER}{model_info}")

        # Ensure all necessary directories exist
        for dir_path in [INPUT_DIR, OUTPUT_DIR, DONE_DIR, KNOWLEDGE_DIR, LOGS_DIR]:
            dir_path.mkdir(exist_ok=True)

        # Load essential data
        keywords, template = read_project_knowledge()
        progress = load_progress()
        failed_files = load_failed_files()
        logging.info(f"Loaded {len(progress)} processed, {len(failed_files)} failed files")

        # Create provider once for the session
        provider_config = {"model": LLM_MODEL} if LLM_MODEL else {}
        provider = create_provider(LLM_PROVIDER, config=provider_config)
        logging.info(
            f"Provider ready: {provider.__class__.__name__}"
            + (f" (model: {provider.model})" if provider.model else "")
        )

        logging.info(f"Monitoring: {INPUT_DIR.absolute()}")
        logging.info("--- Waiting for files (SIGTERM or Ctrl+C to exit) ---")

        waiting_message_shown = False

        # Main processing loop
        while not shutdown_requested:
            try:
                pending = get_pending_files(INPUT_DIR, progress, failed_files)

                if pending and not shutdown_requested:
                    file_path = pending[0]
                    waiting_message_shown = False
                    logging.info(f"--- Processing: {file_path.name} ---")

                    success, filename, error_msg = process_file(
                        file_path, keywords, template, provider
                    )

                    if success:
                        progress.add(filename)
                        save_progress(progress)
                        logging.info(f"--- Done: {filename} (Success) ---")
                    elif error_msg != "Shutdown requested":
                        logging.error(f"--- Done: {filename} (FAILED: {error_msg}) ---")
                        add_to_failed_files(filename, error_msg)
                        failed_files.add(filename)
                    else:
                        logging.info(f"--- Stopped: {filename} (shutdown) ---")

                    if not shutdown_requested:
                        logging.info("--- Checking for next file ---\n")
                        time.sleep(2)

                elif not waiting_message_shown:
                    logging.info("No files in queue. Waiting...\n")
                    waiting_message_shown = True
                    interruptible_sleep(10)
                else:
                    interruptible_sleep(5)

            except Exception as e:
                logging.critical(
                    f"--- UNEXPECTED ERROR in main loop: {e} ---", exc_info=True
                )
                if not shutdown_requested:
                    logging.critical("Pausing 15s before continuing...")
                    interruptible_sleep(15)

    except Exception as e:
        logging.critical(f"--- CRITICAL STARTUP ERROR: {e} ---", exc_info=True)
        print(
            f"CRITICAL FALLBACK: {time.strftime('%Y-%m-%d %H:%M:%S')} - {e}",
            file=sys.stderr,
        )
    finally:
        logging.info("--- Science Paper Summariser Stopped ---\n")


if __name__ == "__main__":
    main()
