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
import json
import logging
import signal
import concurrent.futures
from dataclasses import dataclass
from html import unescape
from dotenv import load_dotenv
from pathlib import Path
from typing import Optional
import xml.etree.ElementTree as ET
import requests
from providers import SUPPORTED_MODES, create_provider, get_supported_provider_names

# --- Configuration and Paths ---
load_dotenv()

DEFAULT_MODE = "cli"
DEFAULT_PROVIDER = "claude"
VALID_EFFORT_LEVELS = ("low", "medium", "high")

# Directory paths relative to the script location
SCRIPT_DIR = Path(__file__).parent.absolute()
INPUT_DIR = SCRIPT_DIR / "input"
OUTPUT_DIR = SCRIPT_DIR / "output"
LOGS_DIR = SCRIPT_DIR / "logs"
DONE_DIR = SCRIPT_DIR / "processed"
KNOWLEDGE_DIR = SCRIPT_DIR / "project_knowledge"
SUMMARY_TEMPLATE_FILE = KNOWLEDGE_DIR / "paper-summary-template.md"
SUMMARY_WORKED_EXAMPLE_FILE = KNOWLEDGE_DIR / "summary-worked-example.md"
ASTRONOMY_KEYWORDS_FILE = KNOWLEDGE_DIR / "astronomy-keywords.txt"

# Log file paths
PROGRESS_FILE = LOGS_DIR / "completed.log"
FAILED_FILE = LOGS_DIR / "failed.log"
PROMPT_DEBUG_DIR = LOGS_DIR / "debug"
ARXIV_CATEGORY_CACHE_FILE = LOGS_DIR / "arxiv_categories.json"

# --- Global shutdown flag ---
shutdown_requested = False

# --- Lazy-loaded marker-pdf model cache ---
_marker_models = None
MARKER_TIMEOUT = 300  # seconds before marker-pdf extraction is abandoned
SOURCE_SCAN_CHAR_LIMIT = 4000
EXTRACTION_NOISE_LINE_CHAR_LIMIT = 1000
DEFAULT_PROMPT_CHAR_BUDGET = 300_000
REFERENCE_SECTION_TITLES = {
    "BIBLIOGRAPHY",
    "LITERATURE CITED",
    "NOTES AND REFERENCES",
    "REFERENCE",
    "REFERENCES",
    "REFERENCES AND NOTES",
    "REFERENCES CITED",
    "WORKS CITED",
}
APPENDIX_SECTION_TITLES = {
    "APPENDICES",
    "APPENDIX",
    "SUPPLEMENTARY MATERIAL",
    "SUPPLEMENTARY MATERIALS",
}

_INLINE_FOOTNOTE_RE = re.compile(r"\[\^\d+\]")
_REFERENCES_HEADING_RE = re.compile(r"^## References\s*$", re.MULTILINE)
_ARXIV_CATEGORY_RE = re.compile(r"\((?P<code>[A-Za-z0-9.-]+)\)")

ARXIV_FILENAME_RE = re.compile(
    r"(?P<id>\d{4}\.\d{4,5}(?:v\d+)?|[A-Za-z.-]+/\d{7}(?:v\d+)?)"
)
ARXIV_TEXT_RE = re.compile(
    r"(?:arxiv\s*:\s*|arxiv\.org/(?:abs|pdf)/)"
    r"(?P<id>\d{4}\.\d{4,5}(?:v\d+)?|[A-Za-z.-]+/\d{7}(?:v\d+)?)",
    re.IGNORECASE,
)
ARXIV_NEW_STYLE_RE = re.compile(
    r"^(?P<yy>\d{2})(?P<mm>0[1-9]|1[0-2])\.\d{4,5}(?:v\d+)?$"
)
DOI_URL_RE = re.compile(
    r"https?://(?:dx\.)?doi\.org/(?P<doi>10\.\d{4,9}/[-._;()/:A-Z0-9]+)",
    re.IGNORECASE,
)
DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)

ARXIV_METADATA_TIMEOUT = 5
ARXIV_CATEGORY_KEYWORD_HEADINGS = {
    "astro-ph.CO": ("COSMOLOGY", "GALAXIES"),
    "astro-ph.EP": ("PLANETARY SYSTEMS", "STARS"),
    "astro-ph.GA": ("THE GALAXY", "GALAXIES", "INTERSTELLAR MEDIUM (ISM)", "STARS"),
    "astro-ph.HE": (
        "PHYSICAL DATA AND PROCESSES",
        "STARS",
        "TRANSIENTS",
        "RESOLVED AND UNRESOLVED SOURCES AS A FUNCTION OF WAVELENGTH",
    ),
    "astro-ph.IM": (
        "ASTRONOMICAL INSTRUMENTATION, METHODS AND TECHNIQUES",
        "ASTRONOMICAL DATABASES",
        "RESOLVED AND UNRESOLVED SOURCES AS A FUNCTION OF WAVELENGTH",
    ),
    "astro-ph.SR": ("THE SUN", "STARS", "PHYSICAL DATA AND PROCESSES"),
}
DEFAULT_KEYWORD_HEADINGS = (
    "GENERAL",
    "PHYSICAL DATA AND PROCESSES",
    "ASTRONOMICAL INSTRUMENTATION, METHODS AND TECHNIQUES",
    "ASTRONOMICAL DATABASES",
)
GLOSSARY_MAX_TERMS = 12

MONTH_NAMES = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}


@dataclass(frozen=True)
class SourceMetadata:
    """Structured source metadata that can guide or repair summary top matter."""

    source_type: Optional[str] = None
    identifier: Optional[str] = None
    canonical_url: Optional[str] = None
    published_label: Optional[str] = None
    detection_method: Optional[str] = None
    primary_category: Optional[str] = None
    categories: tuple[str, ...] = ()


def format_usage():
    """Return concise CLI usage guidance for the explicit mode/provider interface."""
    return (
        "Usage: python3 summarise.py\n"
        "   or: python3 summarise.py <mode> <provider> [model] [--effort <level>]\n\n"
        "Modes and providers:\n"
        f"  cli: {', '.join(get_supported_provider_names('cli'))}\n"
        f"  api: {', '.join(get_supported_provider_names('api'))}\n\n"
        f"Effort levels (cli mode only): {', '.join(VALID_EFFORT_LEVELS)}\n\n"
        "Examples:\n"
        "  python3 summarise.py\n"
        "  python3 summarise.py cli claude\n"
        "  python3 summarise.py cli claude --effort high\n"
        "  python3 summarise.py cli codex gpt-5.4 --effort medium\n"
        "  python3 summarise.py cli copilot --effort low\n"
        "  python3 summarise.py api openai gpt-5.2\n\n"
        "Old one-argument invocations such as 'python3 summarise.py gemini' "
        "are no longer supported."
    )


def parse_cli_args(argv):
    """Parse command-line arguments for explicit mode/provider selection."""
    effort = None
    positional_args = []
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg == "--effort":
            if effort is not None:
                raise ValueError(
                    "The --effort option may only be provided once.\n\n"
                    f"{format_usage()}"
                )
            if index + 1 >= len(argv):
                raise ValueError(
                    "The --effort option requires a value.\n\n"
                    f"{format_usage()}"
                )
            effort = argv[index + 1].strip().lower()
            if not effort:
                raise ValueError(
                    "The --effort option requires a non-blank value.\n\n"
                    f"{format_usage()}"
                )
            index += 2
            continue

        positional_args.append(arg)
        index += 1

    arg_count = len(positional_args)
    if arg_count == 0:
        mode = DEFAULT_MODE
        provider_name = DEFAULT_PROVIDER
        model = None
    else:
        if arg_count not in (2, 3):
            raise ValueError(format_usage())

        mode = positional_args[0].lower().strip()
        provider_name = positional_args[1].lower().strip()
        model = positional_args[2].strip() if arg_count == 3 else None

    if not mode or not provider_name:
        raise ValueError(format_usage())
    if mode not in SUPPORTED_MODES:
        raise ValueError(
            f"Invalid mode '{mode}'. Supported modes: {', '.join(SUPPORTED_MODES)}.\n\n"
            f"{format_usage()}"
        )
    if effort is not None:
        if effort not in VALID_EFFORT_LEVELS:
            raise ValueError(
                f"Invalid effort '{effort}'. Supported effort levels: "
                f"{', '.join(VALID_EFFORT_LEVELS)}.\n\n"
                f"{format_usage()}"
            )
        if mode != "cli":
            raise ValueError(
                "The --effort option is only supported in cli mode.\n\n"
                f"{format_usage()}"
            )

    return mode, provider_name, model, effort


def build_provider_config(model_override=None, effort=None):
    """Build provider configuration for the selected startup options."""
    provider_config = {}
    if model_override:
        provider_config["model"] = model_override
    if effort:
        provider_config["effort"] = effort
    return provider_config


def validate_startup_selection(argv):
    """Parse CLI args and validate that the requested provider can be created."""
    mode, provider_name, model_override, effort = parse_cli_args(argv)
    provider = create_provider(
        mode,
        provider_name,
        config=build_provider_config(model_override=model_override, effort=effort),
    )
    provider.validate_runtime_ready()
    return mode, provider_name, model_override, effort, provider


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
    """Read project knowledge files needed for prompts.

    Raises an exception if critical files are missing.
    """
    try:
        with open(ASTRONOMY_KEYWORDS_FILE, "r", encoding="utf-8") as f:
            keywords = f.read()
        with open(SUMMARY_TEMPLATE_FILE, "r", encoding="utf-8") as f:
            template = f.read()
        with open(SUMMARY_WORKED_EXAMPLE_FILE, "r", encoding="utf-8") as f:
            worked_example = f.read()
        return keywords, template, worked_example
    except FileNotFoundError as e:
        logging.critical(f"CRITICAL: Project knowledge file not found: {e}. Cannot continue.")
        raise
    except Exception as e:
        logging.critical(f"CRITICAL: Failed to read project knowledge files: {e}")
        raise


def parse_keyword_sections(keywords):
    """Parse the grouped keyword file into heading -> lines."""
    sections = {}
    current_heading = None
    current_lines = []

    for raw_line in keywords.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if not line.startswith("#"):
            if current_heading is not None:
                sections[current_heading] = current_lines
            current_heading = line.upper()
            current_lines = []
            continue

        if current_heading is None:
            current_heading = "GENERAL"
        current_lines.append(line)

    if current_heading is not None:
        sections[current_heading] = current_lines

    return sections


def _render_keyword_sections(sections, headings):
    """Render selected keyword sections in source-file format."""
    rendered_sections = []
    for heading in headings:
        lines = sections.get(heading)
        if not lines:
            continue
        rendered_sections.append("\n".join([heading, *lines]))
    return "\n\n".join(rendered_sections)


def iter_keyword_tags(keywords):
    """Yield allowed hashtag tokens from a grouped keyword list in source order."""
    seen_tags = set()
    for match in re.finditer(r"#[A-Za-z0-9][A-Za-z0-9]*", keywords or ""):
        tag = match.group(0)
        if tag in seen_tags:
            continue
        seen_tags.add(tag)
        yield tag


def _normalise_search_text(value):
    """Normalise text for lightweight tag fallback matching."""
    return re.sub(r"[^A-Za-z0-9]+", "", value or "").lower()


def _split_tag_words(value):
    """Split a CamelCase/acronym tag into searchable words."""
    return [
        part.lower()
        for part in re.findall(
            r"[A-Z]+(?=[A-Z][a-z]|\d|$)|[A-Z]?[a-z]+|\d+",
            value or "",
        )
        if len(part) > 1
    ]


def filter_keywords_for_categories(keywords, categories):
    """Return keywords relevant to arXiv categories, falling back when unknown."""
    category_list = tuple(category for category in categories if category)
    if not category_list:
        return keywords

    selected_headings = []
    seen_headings = set()
    for category in category_list:
        for heading in ARXIV_CATEGORY_KEYWORD_HEADINGS.get(category, ()):
            if heading not in seen_headings:
                seen_headings.add(heading)
                selected_headings.append(heading)

    if not selected_headings:
        return keywords

    for heading in DEFAULT_KEYWORD_HEADINGS:
        if heading not in seen_headings:
            selected_headings.append(heading)
            seen_headings.add(heading)

    filtered = _render_keyword_sections(parse_keyword_sections(keywords), selected_headings)
    return filtered or keywords


def _is_extraction_noise_line(line):
    """Return True for long marker-pdf conversion lines with little prose value."""
    if len(line) <= EXTRACTION_NOISE_LINE_CHAR_LIMIT:
        return False

    stripped = line.strip()
    if not stripped:
        return True

    alpha_chars = sum(char.isalpha() for char in stripped)
    alpha_ratio = alpha_chars / len(stripped)
    separator_chars = sum(char in "|-_=+·. " for char in stripped)
    separator_ratio = separator_chars / len(stripped)
    html_breaks = stripped.count("<br>")

    return (
        stripped.count("|") >= 2
        or html_breaks >= 20
        or separator_ratio > 0.85
        or alpha_ratio < 0.08
    )


def normalise_extracted_text(text, source_name="document"):
    """Remove extraction artefacts that can overwhelm LLM context windows.

    Marker-pdf can occasionally convert wide PDF tables or layouts into
    thousands of characters of row/separator noise. These lines add little
    useful prose context and can make otherwise normal papers exceed CLI limits.
    """
    if not isinstance(text, str) or not text:
        return text

    kept_lines = []
    removed_lines = 0
    removed_chars = 0

    for line in text.splitlines():
        if _is_extraction_noise_line(line):
            removed_lines += 1
            removed_chars += len(line) + 1
            continue
        kept_lines.append(line)

    if removed_lines:
        logging.info(
            "Removed %d noisy extraction lines from %s (~%d chars).",
            removed_lines,
            source_name,
            removed_chars,
        )
        return "\n".join(kept_lines)

    return text


def _get_prompt_char_budget(provider):
    """Return a conservative combined-prompt character budget."""
    configured_budget = getattr(provider, "config", {}).get("max_prompt_chars")
    if configured_budget:
        return int(configured_budget)

    return DEFAULT_PROMPT_CHAR_BUDGET


def _combined_prompt_chars(system_prompt, user_prompt):
    """Return the exact combined prompt length used by CLI providers."""
    return len(f"{system_prompt}\n\n{user_prompt}")


def _normalise_section_heading(line):
    """Return a normalised Markdown/plain section heading title."""
    stripped = line.strip()
    if not stripped:
        return ""

    heading_match = re.match(r"^#{1,6}\s+(?P<title>.*?)\s*#*\s*$", stripped)
    if heading_match:
        stripped = heading_match.group("title").strip()

    stripped = stripped.strip("*_`")
    stripped = re.sub(r"\s+", " ", stripped)
    stripped = re.sub(r"[\s:.;-]+$", "", stripped)
    return stripped.upper()


def _is_matching_section_heading(line, section_titles, section_prefixes=()):
    """Return True when a line is a heading for a section to drop."""
    title = _normalise_section_heading(line)
    if title in section_titles:
        return True
    return any(title.startswith(prefix) for prefix in section_prefixes)


def _drop_sections_from_heading(paper_text, section_titles, section_prefixes=()):
    """Drop text from the first matching section heading onwards."""
    lines = paper_text.splitlines()
    output = []

    for line in lines:
        if _is_matching_section_heading(line, section_titles, section_prefixes):
            break
        output.append(line)

    return "\n".join(output)


def _drop_references_section(paper_text):
    """Drop the references section from extracted Markdown text."""
    return _drop_sections_from_heading(paper_text, REFERENCE_SECTION_TITLES)


def _drop_appendix_section(paper_text):
    """Drop the appendix and all following extracted Markdown text."""
    return _drop_sections_from_heading(
        paper_text,
        APPENDIX_SECTION_TITLES,
        section_prefixes=("APPENDIX ", "APPENDICES "),
    )


def fit_prompt_to_provider_budget(
    provider,
    system_prompt,
    paper_text,
    template,
    worked_example="",
    source_metadata=None,
    is_pdf=False,
):
    """Build a user prompt, removing low-value sections only if needed."""
    user_prompt = create_user_prompt(
        paper_text,
        template,
        worked_example=worked_example,
        source_metadata=source_metadata,
        is_pdf=is_pdf,
    )
    budget = _get_prompt_char_budget(provider)
    if not budget:
        return paper_text, user_prompt

    combined_chars = _combined_prompt_chars(system_prompt, user_prompt)
    if combined_chars <= budget:
        return paper_text, user_prompt

    logging.warning(
        "Combined prompt is %d chars, above %s budget of %d chars. "
        "Applying fallback prompt reduction.",
        combined_chars,
        getattr(provider, "provider_name", provider.__class__.__name__),
        budget,
    )

    reductions = (
        ("references", _drop_references_section),
        ("appendix", _drop_appendix_section),
    )
    reduced_text = paper_text
    for label, reducer in reductions:
        candidate_text = reducer(reduced_text)
        if candidate_text == reduced_text:
            continue

        candidate_prompt = create_user_prompt(
            candidate_text,
            template,
            worked_example=worked_example,
            source_metadata=source_metadata,
            is_pdf=is_pdf,
        )
        candidate_chars = _combined_prompt_chars(system_prompt, candidate_prompt)
        logging.info(
            "Prompt reduction dropped %s: %d -> %d combined chars.",
            label,
            combined_chars,
            candidate_chars,
        )
        reduced_text = candidate_text
        user_prompt = candidate_prompt
        combined_chars = candidate_chars

        if combined_chars <= budget:
            return reduced_text, user_prompt

    raise ValueError(
        f"Combined prompt is {combined_chars} chars after cleanup, above "
        f"{getattr(provider, 'provider_name', provider.__class__.__name__)} "
        f"budget of {budget} chars"
    )


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
                text = normalise_extracted_text(text, source_name=file_path.name)
                logging.info(f"Extracted ~{len(text.split())} words from PDF")
                return text, None

        elif file_suffix == ".txt":
            logging.info(f"Reading text file: {file_path.name}")
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                return normalise_extracted_text(text, source_name=file_path.name), None
        else:
            return None, f"Unsupported file type: {file_path.suffix}"

    except Exception as e:
        logging.error(f"Error reading input file {file_path.name}: {e}", exc_info=True)
        return None, f"File read/processing error: {e}"


# --- Source metadata detection ---

def _trim_trailing_punctuation(value):
    """Remove punctuation that commonly trails detected identifiers in prose."""
    return value.rstrip(").,;:]")


def _extract_arxiv_id_from_filename(input_path):
    """Extract an arXiv identifier from the input filename when present."""
    stem = input_path.stem.strip()
    if not stem:
        return None

    if full_match := ARXIV_FILENAME_RE.fullmatch(stem):
        return full_match.group("id")

    if search_match := ARXIV_FILENAME_RE.search(stem):
        return search_match.group("id")
    return None


def _extract_arxiv_id_from_text(paper_text):
    """Extract an arXiv identifier from the document header region when present."""
    if not paper_text:
        return None

    header_window = paper_text[:SOURCE_SCAN_CHAR_LIMIT]
    if match := ARXIV_TEXT_RE.search(header_window):
        return match.group("id")
    return None


def _extract_doi_from_text(paper_text):
    """Extract a DOI from the document header region when present."""
    if not paper_text:
        return None

    header_window = paper_text[:SOURCE_SCAN_CHAR_LIMIT]
    if url_match := DOI_URL_RE.search(header_window):
        return _trim_trailing_punctuation(url_match.group("doi"))

    if doi_match := DOI_RE.search(header_window):
        return _trim_trailing_punctuation(doi_match.group(0))
    return None


def _published_label_from_arxiv_id(arxiv_id):
    """Return a month/year label for new-style arXiv identifiers."""
    if not arxiv_id:
        return None

    if match := ARXIV_NEW_STYLE_RE.match(arxiv_id):
        year = 2000 + int(match.group("yy"))
        month = MONTH_NAMES[int(match.group("mm"))]
        return f"{month} {year}"
    return None


def _canonical_arxiv_url(arxiv_id):
    """Return the canonical arXiv abstract URL, without a version suffix when present."""
    if not arxiv_id:
        return None

    versionless_id = re.sub(r"v\d+$", "", arxiv_id, flags=re.IGNORECASE)
    return f"https://arxiv.org/abs/{versionless_id}"


def _versionless_arxiv_id(arxiv_id):
    """Return an arXiv identifier without a version suffix."""
    return re.sub(r"v\d+$", "", arxiv_id or "", flags=re.IGNORECASE)


def _strip_html_tags(value):
    """Return HTML text content with tags removed."""
    return unescape(re.sub(r"<[^>]+>", " ", value))


def _extract_category_codes(value):
    """Extract unique category codes from arXiv subject text."""
    codes = []
    seen_codes = set()
    for match in _ARXIV_CATEGORY_RE.finditer(value or ""):
        code = match.group("code")
        if code in seen_codes:
            continue
        seen_codes.add(code)
        codes.append(code)
    return tuple(codes)


def extract_arxiv_categories_from_html(html):
    """Extract primary and full arXiv category codes from an abstract page."""
    subjects_match = re.search(
        r'<td[^>]*class=["\'][^"\']*\bsubjects\b[^"\']*["\'][^>]*>(?P<body>.*?)</td>',
        html or "",
        flags=re.IGNORECASE | re.DOTALL,
    )
    subjects_html = subjects_match.group("body") if subjects_match else (html or "")

    primary_match = re.search(
        r'<span[^>]*class=["\'][^"\']*\bprimary-subject\b[^"\']*["\'][^>]*>'
        r"(?P<body>.*?)</span>",
        subjects_html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    primary_codes = _extract_category_codes(
        _strip_html_tags(primary_match.group("body")) if primary_match else ""
    )
    categories = _extract_category_codes(_strip_html_tags(subjects_html))
    primary_category = (
        primary_codes[0] if primary_codes else (categories[0] if categories else None)
    )
    if primary_category and not categories:
        categories = (primary_category,)
    return primary_category, categories


def extract_arxiv_categories_from_api_xml(xml_text):
    """Extract primary and full arXiv category codes from arXiv API XML."""
    try:
        root = ET.fromstring(xml_text or "")
    except ET.ParseError:
        return None, ()

    namespaces = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }
    entry = root.find("atom:entry", namespaces)
    search_root = entry if entry is not None else root

    primary_node = search_root.find("arxiv:primary_category", namespaces)
    primary_category = (
        primary_node.attrib.get("term") if primary_node is not None else None
    )

    categories = []
    seen_categories = set()
    for category_node in search_root.findall("atom:category", namespaces):
        category = category_node.attrib.get("term")
        if category and category not in seen_categories:
            seen_categories.add(category)
            categories.append(category)

    if primary_category and primary_category not in seen_categories:
        categories.insert(0, primary_category)
    if not primary_category and categories:
        primary_category = categories[0]

    return primary_category, tuple(categories)


def _load_arxiv_category_cache():
    """Read cached arXiv category metadata."""
    try:
        if not ARXIV_CATEGORY_CACHE_FILE.exists():
            return {}
        with open(ARXIV_CATEGORY_CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as error:
        logging.warning("Could not read arXiv category cache: %s", error)
        return {}


def _save_arxiv_category_cache(cache):
    """Persist cached arXiv category metadata without interrupting processing."""
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        with open(ARXIV_CATEGORY_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, sort_keys=True)
    except Exception as error:
        logging.warning("Could not write arXiv category cache: %s", error)


def _get_cached_arxiv_categories(arxiv_id):
    """Return cached category metadata for an arXiv identifier if present."""
    cached = _load_arxiv_category_cache().get(_versionless_arxiv_id(arxiv_id))
    if not isinstance(cached, dict):
        return None

    categories = cached.get("categories")
    if not isinstance(categories, list):
        return None

    primary_category = cached.get("primary_category")
    return primary_category, tuple(category for category in categories if category)


def _cache_arxiv_categories(arxiv_id, primary_category, categories):
    """Store successful arXiv category metadata for future offline runs."""
    category_tuple = tuple(category for category in categories if category)
    if not category_tuple:
        return

    versionless_id = _versionless_arxiv_id(arxiv_id)
    cache = _load_arxiv_category_cache()
    cache[versionless_id] = {
        "primary_category": primary_category,
        "categories": list(category_tuple),
    }
    _save_arxiv_category_cache(cache)


def fetch_arxiv_categories(arxiv_id):
    """Fetch arXiv category metadata for an identifier, using a local cache first."""
    versionless_id = _versionless_arxiv_id(arxiv_id)
    if not versionless_id:
        return None, ()

    cached = _get_cached_arxiv_categories(versionless_id)
    if cached is not None:
        return cached

    try:
        response = requests.get(
            "https://export.arxiv.org/api/query",
            params={"id_list": versionless_id},
            headers={"User-Agent": "science-paper-summariser/0.1"},
            timeout=ARXIV_METADATA_TIMEOUT,
        )
        response.raise_for_status()
    except requests.RequestException as error:
        logging.warning(
            "Could not fetch arXiv category metadata for %s: %s",
            arxiv_id,
            error,
        )
        return None, ()

    primary_category, categories = extract_arxiv_categories_from_api_xml(response.text)
    _cache_arxiv_categories(versionless_id, primary_category, categories)
    return primary_category, categories


def extract_source_metadata(input_path, paper_text):
    """Detect canonical paper identifiers that can enforce the Published line."""
    if arxiv_id := _extract_arxiv_id_from_filename(input_path):
        primary_category, categories = fetch_arxiv_categories(arxiv_id)
        return SourceMetadata(
            source_type="arxiv",
            identifier=arxiv_id,
            canonical_url=_canonical_arxiv_url(arxiv_id),
            published_label=_published_label_from_arxiv_id(arxiv_id),
            detection_method="filename",
            primary_category=primary_category,
            categories=categories,
        )

    if arxiv_id := _extract_arxiv_id_from_text(paper_text):
        primary_category, categories = fetch_arxiv_categories(arxiv_id)
        return SourceMetadata(
            source_type="arxiv",
            identifier=arxiv_id,
            canonical_url=_canonical_arxiv_url(arxiv_id),
            published_label=_published_label_from_arxiv_id(arxiv_id),
            detection_method="paper_text",
            primary_category=primary_category,
            categories=categories,
        )

    if doi := _extract_doi_from_text(paper_text):
        return SourceMetadata(
            source_type="doi",
            identifier=doi,
            canonical_url=f"https://doi.org/{doi}",
            detection_method="paper_text",
        )

    return SourceMetadata()


def _extract_existing_published_text(line):
    """Extract the free-text Published value, excluding any trailing markdown link."""
    if not line:
        return ""

    stripped = line.strip()
    if stripped.startswith("Published:"):
        stripped = stripped[len("Published:"):].strip()

    return re.sub(r"\s*\(?\[[^\]]+\]\([^)]+\)\)?\s*$", "", stripped).strip()


def build_published_line(existing_line, source_metadata):
    """Construct the canonical Published line from model output and detected metadata."""
    if not source_metadata:
        return existing_line

    published_text = source_metadata.published_label or _extract_existing_published_text(existing_line)
    if not published_text:
        return existing_line

    if source_metadata.canonical_url:
        return f"Published: {published_text} ([Link]({source_metadata.canonical_url}))"
    return f"Published: {published_text}"


def enforce_source_metadata(summary_content, source_metadata):
    """Normalise the Published line when a canonical identifier is known."""
    if not source_metadata or not (source_metadata.canonical_url or source_metadata.published_label):
        return summary_content

    lines = summary_content.split("\n")
    authors_index = None
    published_index = None

    for index, line in enumerate(lines[:12]):
        stripped = line.strip()
        if authors_index is None and stripped.startswith("Authors:"):
            authors_index = index
        if published_index is None and stripped.startswith("Published:"):
            published_index = index

    if published_index is not None:
        replacement = build_published_line(lines[published_index], source_metadata)
        if replacement and replacement != lines[published_index]:
            lines[published_index] = replacement
            logging.info(
                "Normalised Published line using %s metadata (%s).",
                source_metadata.source_type,
                source_metadata.detection_method,
            )
        return "\n".join(lines)

    replacement = build_published_line("", source_metadata)
    if replacement and authors_index is not None:
        lines.insert(authors_index + 1, replacement)
        logging.info(
            "Inserted missing Published line using %s metadata (%s).",
            source_metadata.source_type,
            source_metadata.detection_method,
        )
        return "\n".join(lines)

    return summary_content


# --- Prompt construction ---

def create_system_prompt():
    """Construct the system prompt defining the main summary task."""
    return (
        "<role>\n"
        "You are an esteemed professor of astrophysics at Harvard University "
        "specializing in analyzing research papers. You are an expert in \n"
        "identifying key scientific results and their significance.\n"
        "</role>\n\n"
        "<rules>\n"
        "1. Use only information from the provided paper; if no exact quote supports "
        "a claim, do not make the claim\n"
        "2. Follow the template and worked example exactly: concise Markdown bullets, "
        "UK English, LaTeX for mathematics, bold key technical terms on first mention, "
        "and italics for paper or model names where natural\n"
        "3. Each bullet must make one clear scientific claim and include the most "
        "important concrete detail available: a number, sample size, named method, "
        "parameter, comparison, or limitation\n"
        "4. Preserve the paper's specific names for important instruments, surveys, "
        "datasets, software, models, methods, and acronyms where central to a point\n"
        "5. Every bullet must end with one supporting footnote, and every footnote "
        "must contain an exact quote in quotation marks plus a section/page reference\n"
        "6. Always include a ## References section at the end listing every footnote "
        "definition as [^N]: \"exact quote\" (Section X.Y, p.Z)\n"
        "7. Treat Discussion and Weaknesses as critical engagement, not recap. Discussion "
        "bullets should place the paper against prior models, surveys, datasets, or observations "
        "the paper builds on or contradicts. Weaknesses bullets must be concrete and named "
        "(e.g. \"omits Population III stars\", \"dust model untested at z>7\", \"uniform "
        "ionising background ignores reionisation patchiness\"); avoid generic hedges "
        "(\"limited sample\", \"more work needed\")\n"
        "</rules>"
    )


def create_user_prompt(
    paper_text,
    template,
    source_metadata=None,
    is_pdf=False,
    worked_example="",
):
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
    )

    if source_metadata and source_metadata.canonical_url:
        base_prompt += (
            "<source_metadata>\n"
            f"Detected source identifier: {source_metadata.source_type}:{source_metadata.identifier}\n"
            f"Canonical paper link: {source_metadata.canonical_url}\n"
        )
        if source_metadata.published_label:
            base_prompt += (
                f"Published line date: {source_metadata.published_label}\n"
            )
        base_prompt += (
            "You MUST use this exact link in the Published line.\n"
        )
        if source_metadata.published_label:
            base_prompt += (
                "You MUST use this exact month and year in the Published line.\n"
            )
        base_prompt += "</source_metadata>\n\n"

    if worked_example:
        base_prompt += (
            "<worked_example>\n"
            "This fictional example shows the required density, bullet structure, "
            "footnote style, and quote format. Imitate the style and formatting, "
            "but do not copy its topic, claims, names, or references.\n\n"
            f"{worked_example.strip()}\n"
            "</worked_example>\n\n"
        )

    base_prompt += "</task>\n\n"

    if paper_text:
        base_prompt += (
            "<input>\n"
            "Paper to summarize:\n\n"
            f"---BEGIN PAPER---\n{paper_text}\n---END PAPER---\n"
            "</input>"
        )
    return base_prompt


# --- LLM call with retry ---

def _write_debug_prompt(name, system_prompt, user_prompt):
    """Write a named prompt to logs/debug/<name>.txt for troubleshooting."""
    try:
        PROMPT_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        dest = PROMPT_DEBUG_DIR / f"{name}.txt"
        full_prompt = f"SYSTEM PROMPT\n{system_prompt}\n\n---\n\nUSER PROMPT\n{user_prompt}"
        dest.write_text(full_prompt, encoding="utf-8")
        logging.info("Debug prompt written to %s", dest)
    except Exception as e:
        logging.warning("Could not write debug prompt file: %s", e)


def _is_retryable_llm_error(error):
    """Return whether an LLM/provider error is worth retrying."""
    message = str(error).lower()
    non_retryable_markers = (
        "credit balance is too low",
        "api key",
        "authentication",
        "logged out",
        "login required",
        "not found on path",
    )
    return not any(marker in message for marker in non_retryable_markers)


def _call_llm_with_retry(
    provider,
    content,
    is_pdf,
    system_prompt,
    user_prompt,
    max_retries=3,
    response_validator=None,
):
    """Call the LLM with retry logic and exponential backoff.

    Returns the summary text on success.
    Raises the last exception after all retries are exhausted.
    Raises InterruptedError if shutdown is requested.
    """
    max_tokens = provider.get_preferred_max_tokens()
    last_error = None
    for attempt in range(max_retries):
        if shutdown_requested:
            raise InterruptedError("Shutdown requested before LLM call")

        try:
            logging.info(
                f"Attempt {attempt + 1}/{max_retries} calling LLM "
                f"(max_tokens={max_tokens})..."
            )
            summary = provider.process_document(
                content=content,
                is_pdf=is_pdf,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
            )
            if not summary or not summary.strip():
                raise ValueError("LLM returned empty or whitespace-only response")

            if _INLINE_FOOTNOTE_RE.search(summary) and not _REFERENCES_HEADING_RE.search(summary):
                raise ValueError(
                    "Summary contains footnote markers but is missing the ## References section"
                )

            if response_validator is not None:
                response_validator(summary)

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

            if not _is_retryable_llm_error(e):
                logging.error("Provider error is non-retryable. Skipping remaining attempts.")
                break

            if attempt < max_retries - 1:
                wait_time = 2 ** (attempt + 1)
                logging.info(f"Waiting {wait_time}s before retry...")
                if interruptible_sleep(wait_time):
                    raise InterruptedError("Shutdown requested during retry wait")
            else:
                logging.error(f"All {max_retries} attempts failed.")

    raise last_error


def _call_glossary_llm_with_retry(provider, system_prompt, user_prompt, max_retries=3):
    """Generate a glossary, preserving the last non-empty candidate if validation fails."""
    last_error = None
    last_candidate = ""
    for attempt in range(max_retries):
        if shutdown_requested:
            raise InterruptedError("Shutdown requested before glossary LLM call")
        try:
            logging.info("Attempt %s/%s calling LLM...", attempt + 1, max_retries)
            section = provider.process_document(
                content="",
                is_pdf=False,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=provider.get_preferred_max_tokens(),
            )
            if not section or not section.strip():
                raise ValueError("LLM returned empty or whitespace-only response")
            last_candidate = section
            if _INLINE_FOOTNOTE_RE.search(section) and not _REFERENCES_HEADING_RE.search(section):
                raise ValueError("Summary has inline footnote markers but is missing the ## References section")
            validate_glossary_section(section)
            logging.info("LLM call successful (received ~%s chars)", len(section))
            return _normalise_generated_section(section, "## Glossary")
        except Exception as error:
            last_error = error
            logging.error("Attempt %s failed — %s: %s", attempt + 1, provider.__class__.__name__, error)
            if not _is_retryable_llm_error(error) or attempt == max_retries - 1:
                break
            wait_time = 2 ** (attempt + 1)
            logging.info("Waiting %ss before retry...", wait_time)
            if interruptible_sleep(wait_time):
                raise InterruptedError("Shutdown requested during glossary retry wait")

    if last_candidate.strip():
        logging.warning(
            "Glossary validation failed after %s attempt(s); preserving unvalidated glossary section: %s",
            max_retries,
            last_error or "Unknown validation failure",
        )
        try:
            return _normalise_generated_section(last_candidate, "## Glossary")
        except ValueError:
            return f"## Glossary\n\n{last_candidate.strip()}"

    raise Exception(str(last_error or "Unknown LLM failure"))


# --- Post-processing ---

def _normalise_generated_section(markdown, expected_heading):
    """Return a generated section starting at its expected heading."""
    lines = [line.rstrip() for line in markdown.strip().splitlines()]
    for index, line in enumerate(lines):
        if line.strip() == expected_heading:
            return "\n".join(lines[index:]).strip()
    raise ValueError(f"Generated section is missing '{expected_heading}' heading")


def _has_markdown_section(markdown, heading):
    """Return whether a Markdown document contains an exact section heading."""
    target = heading.strip().lower()
    return any(line.strip().lower() == target for line in markdown.splitlines())


def _non_empty_section_lines(section_markdown, heading):
    """Return non-empty lines after a generated section heading."""
    lines = [line.strip() for line in section_markdown.splitlines()]
    if not lines or lines[0] != heading:
        raise ValueError(f"Generated section must start with '{heading}'")
    return [line for line in lines[1:] if line]


def _split_markdown_table_row(line):
    r"""Return markdown table cells for a pipe-delimited row, or None.

    Handles | characters inside $...$, `...`, and \| escapes correctly.
    """
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        return None

    cells: list[str] = []
    current: list[str] = []
    escaped = False
    in_code = False
    in_math = False
    for char in stripped[1:-1]:
        if escaped:
            current.append(char)
            escaped = False
            continue
        if char == "\\":
            current.append(char)
            escaped = True
            continue
        if char == "`" and not in_math:
            in_code = not in_code
            current.append(char)
            continue
        if char == "$" and not in_code:
            in_math = not in_math
            current.append(char)
            continue
        if char == "|" and not in_code and not in_math:
            cells.append("".join(current).strip())
            current = []
            continue
        current.append(char)
    cells.append("".join(current).strip())
    return cells


def _is_markdown_separator_row(cells):
    """Return True when table cells form a markdown separator row."""
    if cells is None:
        return False
    return all(re.match(r"^:?-{3,}:?$", cell) for cell in cells)


def _extract_tags_from_line(line):
    """Extract canonical hashtags from a generated tag line."""
    tags = []
    for match in re.finditer(r"#[A-Za-z0-9][A-Za-z0-9_./-]*", line or ""):
        token = "#" + re.sub(r"[^A-Za-z0-9]+", "", match.group(0).lstrip("#"))
        if token not in tags:
            tags.append(token)
    return tags


def _append_unique_tag(tags, tag):
    """Append a tag if it is present and not already in the list."""
    if tag and tag not in tags:
        tags.append(tag)


def _render_tags_section(proper_tags, science_tags):
    """Render a canonical Tags section from classified tag lists."""
    lines = ["## Tags", ""]
    if proper_tags:
        lines.append(" ".join(proper_tags[:5]))
    if science_tags:
        if proper_tags:
            lines.append("")
        lines.append(" ".join(science_tags[:5]))
    return "\n".join(lines).rstrip()


def normalise_tags_section(section_markdown, available_keywords=None, *, reject_unknown_science_tags=False):
    """Return a canonical Tags section after classifying candidate tags."""
    section = _normalise_generated_section(section_markdown, "## Tags")
    tag_lines = _non_empty_section_lines(section, "## Tags")
    allowed_science_tags = (
        set(iter_keyword_tags(available_keywords)) if available_keywords is not None else None
    )
    proper_tags = []
    science_tags = []
    dropped_science_tags = []

    for line_index, line in enumerate(tag_lines):
        for tag in _extract_tags_from_line(line):
            if allowed_science_tags is not None and tag in allowed_science_tags:
                _append_unique_tag(science_tags, tag)
            elif allowed_science_tags is not None and line_index > 0:
                _append_unique_tag(dropped_science_tags, tag)
            elif allowed_science_tags is None and line_index > 0:
                _append_unique_tag(science_tags, tag)
            else:
                _append_unique_tag(proper_tags, tag)

    if dropped_science_tags:
        message = "Science tags outside supplied keyword list: " + ", ".join(dropped_science_tags)
        if reject_unknown_science_tags:
            raise ValueError(message)
        logging.warning("Dropped %s", message[0].lower() + message[1:])
    if len(proper_tags) > 5:
        logging.warning("Truncated proper-noun tags to 5 entries")
    if len(science_tags) > 5:
        logging.warning("Truncated science tags to 5 entries")
    if not proper_tags and not science_tags:
        raise ValueError("Tags section did not contain any parseable hashtags")

    return _render_tags_section(proper_tags, science_tags)


def validate_glossary_section(section_markdown):
    """Validate generated glossary markdown."""
    section = _normalise_generated_section(section_markdown, "## Glossary")
    content_lines = _non_empty_section_lines(section, "## Glossary")
    if len(content_lines) < 3:
        raise ValueError("Glossary section must contain a markdown table with rows")

    header_cells = _split_markdown_table_row(content_lines[0])
    separator_cells = _split_markdown_table_row(content_lines[1])
    if header_cells != ["Term", "Definition"]:
        raise ValueError("Glossary table must use columns 'Term' and 'Definition'")
    if len(separator_cells or []) != 2 or not _is_markdown_separator_row(separator_cells):
        raise ValueError("Glossary table is missing a markdown separator row")

    term_rows = content_lines[2:]
    for line in term_rows:
        cells = _split_markdown_table_row(line)
        if cells is None or len(cells) != 2:
            raise ValueError("Glossary section must contain only a two-column table")
        if not cells[0] or not cells[1]:
            raise ValueError("Glossary table rows must include a term and definition")
    if not term_rows:
        raise ValueError("Glossary table must contain at least one term row")
    if len(term_rows) > GLOSSARY_MAX_TERMS:
        raise ValueError(f"Glossary table must contain no more than {GLOSSARY_MAX_TERMS} terms")


def validate_tags_section(section_markdown, available_keywords=None, *, reject_unknown_science_tags=False):
    """Validate generated tag markdown."""
    section = normalise_tags_section(
        section_markdown,
        available_keywords,
        reject_unknown_science_tags=reject_unknown_science_tags,
    )
    tag_lines = _non_empty_section_lines(section, "## Tags")
    for line in tag_lines:
        tags = line.split()
        if len(tags) > 5:
            raise ValueError("Each Tags section line must contain no more than 5 tags")


def build_glossary_prompt(summary_text):
    """Build the focused glossary-generation prompt."""
    system_prompt = (
        "You are an astrophysics editor. Return only the requested Markdown section. "
        "Use UK English and define only specialised terms or acronyms from the summary."
    )
    user_prompt = (
        "Create a concise glossary for this completed paper summary.\n\n"
        "Return exactly this Markdown shape:\n"
        "## Glossary\n\n"
        "| Term | Definition |\n"
        "|---|---|\n"
        "| **technical term** | One clear sentence explaining the term in context. |\n\n"
        f"Use no more than {GLOSSARY_MAX_TERMS} entries. Skip common scientific terms.\n"
        "Do not include introductory text, commentary, or any other section.\n\n"
        "---BEGIN SUMMARY---\n"
        f"{summary_text}\n"
        "---END SUMMARY---"
    )
    return system_prompt, user_prompt


def build_tags_prompt(summary_text, keywords):
    """Build the focused tag-generation prompt."""
    system_prompt = (
        "You are an astronomy indexing assistant. Return only the requested Markdown "
        "section and select tags from the supplied summary and keyword list."
    )
    user_prompt = (
        "Create the tag block for this completed paper summary.\n\n"
        "Return exactly this Markdown shape:\n"
        "## Tags\n\n"
        "#ProperNoun1 #ProperNoun2 #ProperNoun3\n\n"
        "#ScienceKeyword1 #ScienceKeyword2 #ScienceKeyword3\n\n"
        "Rules:\n"
        "- First hashtag line: telescopes, surveys, datasets, missions, instruments, "
        "models, software, or named catalogues from the summary only; no more than 5.\n"
        "- Second hashtag line: choose no more than 5 science-area hashtags from the "
        "available keyword list only.\n"
        "- Copy science-area hashtags exactly as written in the list; do not invent, "
        "rename, shorten, pluralise, or create aliases for science tags.\n"
        "- If the best conceptual science tag is absent, choose the closest listed tag "
        "that is justified by the summary, or omit it.\n"
        "- Use spaces between hashtags, not commas, bullets, or labels.\n"
        "- If fewer than 5 tags are justified, use fewer.\n"
        "- Do not include introductory text, commentary, or any other section.\n\n"
        "Available science-area keywords:\n"
        f"{keywords}\n\n"
        "---BEGIN SUMMARY---\n"
        f"{summary_text}\n"
        "---END SUMMARY---"
    )
    return system_prompt, user_prompt


def generate_glossary(summary_text, provider):
    """Generate and validate a glossary section from the completed summary."""
    system_prompt, user_prompt = build_glossary_prompt(summary_text)
    _write_debug_prompt("paper-glossary", system_prompt, user_prompt)
    try:
        return _call_glossary_llm_with_retry(provider, system_prompt, user_prompt)
    except InterruptedError:
        raise
    except Exception as error:
        logging.warning("Glossary generation failed; skipping section: %s", error)
        return ""


def generate_tags(summary_text, keywords, provider):
    """Generate a best-effort tags section from the completed summary."""
    system_prompt, user_prompt = build_tags_prompt(summary_text, keywords)
    _write_debug_prompt("paper-tags", system_prompt, user_prompt)
    try:
        section = _call_llm_with_retry(
            provider,
            summary_text,
            False,
            system_prompt,
            user_prompt,
            response_validator=lambda response: validate_tags_section(
                response,
                keywords,
                reject_unknown_science_tags=True,
            ),
        )
        return normalise_tags_section(
            section,
            available_keywords=keywords,
            reject_unknown_science_tags=True,
        )
    except Exception as error:
        logging.warning("Tag generation failed; using fallback tags: %s", error)
        return build_fallback_tags(summary_text, keywords)


def build_fallback_tags(summary_text, keywords):
    """Build a minimal science-tag section from keyword matches in the summary."""
    normalised_summary = _normalise_search_text(summary_text)
    science_tags = []
    for tag in iter_keyword_tags(keywords):
        words = _split_tag_words(tag.lstrip("#"))
        if words and all(word in normalised_summary for word in words):
            science_tags.append(tag)
        if len(science_tags) == 5:
            break
    if not science_tags:
        logging.warning("Could not derive fallback tags from summary text")
    return _render_tags_section([], science_tags)


def insert_section(summary, section_markdown, before_heading="## References"):
    """Insert a generated markdown section before another heading, or append it."""
    summary_lines = summary.rstrip().splitlines()
    section = section_markdown.strip()

    insert_index = None
    for index, line in enumerate(summary_lines):
        if line.strip() == before_heading:
            insert_index = index
            break

    if insert_index is None:
        return f"{summary.rstrip()}\n\n{section}\n"

    before = "\n".join(summary_lines[:insert_index]).rstrip()
    after = "\n".join(summary_lines[insert_index:]).lstrip()
    return f"{before}\n\n{section}\n\n{after}\n"


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


def validate_summary(summary_content, source_metadata=None, require_generated_sections=True):
    """Validate the generated summary structure and log results.

    This is a safety-net check — it logs warnings but does not reject summaries.
    """
    lines = [line.strip() for line in summary_content.split("\n") if line.strip()]
    if not lines:
        logging.warning("VALIDATION WARNING: Summary content is empty.")
        return

    # Structural checks
    start_with_title = lines[0].startswith("# ")
    published_line = next((line for line in lines[:6] if line.startswith("Published:")), "")
    year_found = any(re.search(r"\b(19|20)\d{2}\b", line) for line in lines[:5])
    author_complete = not any(
        "et al" in line.lower() for line in lines[:5] if line.startswith("Authors:")
    )
    published_has_link = bool(re.search(r"\[[^\]]+\]\(https?://[^)]+\)", published_line))
    expected_link_ok = (
        source_metadata.canonical_url in published_line
        if source_metadata and source_metadata.canonical_url
        else published_has_link
    )
    expected_date_ok = (
        source_metadata.published_label in published_line
        if source_metadata and source_metadata.published_label
        else year_found
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
    logging.info(f"  Has Published line: {bool(published_line)}")
    logging.info(f"  Year found: {year_found}")
    logging.info(f"  Author list complete: {author_complete}")
    if source_metadata and source_metadata.canonical_url:
        logging.info(f"  Expected paper link present: {expected_link_ok}")
    elif published_line:
        logging.info(f"  Published line has link: {published_has_link}")
    if source_metadata and source_metadata.published_label:
        logging.info(f"  Expected publication date present: {expected_date_ok}")
    logging.info(f"  Bullets: {bullet_count}, Footnotes: {footnote_count}")
    logging.info(
        f"  All footnotes quoted: {all_quoted} ({properly_quoted}/{footnote_count})"
    )
    if require_generated_sections:
        logging.info(f"  Has Glossary: {has_glossary}")
        logging.info(f"  Has Tags (two lines): {has_tags} ({has_two_tag_lines})")

    # Specific warnings
    if not start_with_title:
        logging.warning("VALIDATION WARNING: Summary does not start with title heading '# '")
    if not published_line:
        logging.warning("VALIDATION WARNING: Missing 'Published:' line near top of summary")
    if not year_found:
        logging.warning("VALIDATION WARNING: No year found near top of summary")
    if source_metadata and source_metadata.published_label and not expected_date_ok:
        logging.warning(
            "VALIDATION WARNING: Published line does not include expected month/year '%s'",
            source_metadata.published_label,
        )
    if source_metadata and source_metadata.canonical_url and not expected_link_ok:
        logging.warning(
            "VALIDATION WARNING: Published line missing expected canonical link '%s'",
            source_metadata.canonical_url,
        )
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
    if require_generated_sections and not has_glossary:
        logging.warning("VALIDATION WARNING: Missing '## Glossary' section")
    if require_generated_sections and not has_tags:
        logging.warning("VALIDATION WARNING: Missing '## Tags' section")
    elif require_generated_sections and not has_two_tag_lines:
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
                stripped = part.strip()
                if not stripped:
                    continue
                name_parts = stripped.split()
                if not name_parts:
                    continue
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

    Appends a numeric counter (_1, _2, …) if the filename already exists,
    mirroring the collision-safe logic in move_to_done().

    Returns the output Path on success.
    Raises on failure so the caller can avoid moving the original file.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{base_filename}.md"
    original_path = output_path
    counter = 1
    while output_path.exists():
        output_path = OUTPUT_DIR / f"{base_filename}_{counter}.md"
        counter += 1
        if counter == 2:
            logging.warning(f"Output {original_path.name} exists, adding counter.")
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

def process_file(file_path, keywords, template, provider, worked_example=""):
    """Process a single input file through the full pipeline.

    Steps: read file -> build prompts -> call LLM (with retry) ->
    strip preamble -> validate -> save summary -> move original.

    Returns (success, original_filename, error_message_or_None).
    """
    is_pdf = file_path.suffix.lower() == ".pdf"
    provider_mode = getattr(provider, "mode", "unknown")
    provider_name = getattr(provider, "provider_name", provider.__class__.__name__)
    logging.info(
        f"Using provider: mode={provider_mode}, requested={provider_name}, "
        f"backend={provider.__class__.__name__}"
        + (f", Model: {provider.model}" if provider.model else "")
    )

    try:
        # 1. Read file content
        content, error = read_input_file(file_path, provider)
        if error:
            return False, file_path.name, error

        logging.info(f"Content type: {type(content).__name__}")

        # 2. Build prompts
        if is_pdf and provider.supports_direct_pdf():
            paper_text = ""  # PDF binary sent directly to provider
        elif isinstance(content, bytes):
            paper_text = content.decode("utf-8", errors="ignore")
        else:
            paper_text = content

        source_metadata = extract_source_metadata(file_path, paper_text)
        if source_metadata.canonical_url:
            logging.info(
                "Detected source metadata: type=%s, id=%s, url=%s, date=%s, via=%s",
                source_metadata.source_type,
                source_metadata.identifier,
                source_metadata.canonical_url,
                source_metadata.published_label or "n/a",
                source_metadata.detection_method,
            )
        else:
            logging.info("No canonical source metadata detected for %s", file_path.name)

        if source_metadata.categories:
            logging.info(
                "Detected arXiv categories: primary=%s, all=%s",
                source_metadata.primary_category or "n/a",
                ", ".join(source_metadata.categories),
            )
        filtered_keywords = filter_keywords_for_categories(
            keywords,
            source_metadata.categories,
        )
        logging.info(
            "Tag keyword list: %d -> %d chars.",
            len(keywords),
            len(filtered_keywords),
        )

        system_prompt = create_system_prompt()

        paper_text, user_prompt = fit_prompt_to_provider_budget(
            provider,
            system_prompt,
            paper_text,
            template,
            worked_example=worked_example,
            source_metadata=source_metadata,
            is_pdf=is_pdf,
        )

        prompt_info = (
            f"System prompt: {len(system_prompt)} chars. "
            f"User prompt: {len(user_prompt)} chars."
        )
        if paper_text:
            prompt_info += f" Paper text: {len(paper_text)} chars."
        logging.info(prompt_info)

        _write_debug_prompt("paper-summary", system_prompt, user_prompt)

        # 3. Call LLM with retry
        summary = _call_llm_with_retry(provider, content, is_pdf, system_prompt, user_prompt)

        # 4. Post-process
        summary = strip_preamble(summary)
        summary = enforce_source_metadata(summary, source_metadata)
        validate_summary(
            summary,
            source_metadata=source_metadata,
            require_generated_sections=False,
        )

        tags_section = generate_tags(summary, filtered_keywords, provider)

        if _has_markdown_section(summary, "## Glossary"):
            logging.info("Summary already includes a Glossary section; skipping generated glossary.")
            glossary_section = ""
        else:
            glossary_section = generate_glossary(summary, provider)
        if glossary_section:
            summary = insert_section(summary, glossary_section)
        if tags_section:
            summary = insert_section(summary, tags_section)
        validate_summary(summary, source_metadata=source_metadata)

        # 5. Extract metadata once, use for both save and move
        title, authors, year = extract_metadata(summary)
        base_filename = create_base_filename(title, authors, year, file_path)

        saved_path = save_summary(summary, base_filename)

        moved = move_to_done(file_path, saved_path.stem)
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

def main(argv=None):
    """Main entry point: set up, create provider, monitor input directory, process files."""
    signal.signal(signal.SIGINT, handle_shutdown_signal)
    signal.signal(signal.SIGTERM, handle_shutdown_signal)

    exit_code = 0
    argv = sys.argv[1:] if argv is None else argv
    setup_logging()

    try:
        mode, provider_name, model_override, effort, provider = validate_startup_selection(argv)
        logging.info("--- Science Paper Summariser Starting ---")
        logging.info(f"Process ID: {os.getpid()}")
        logging.info(
            f"Startup selection: mode={mode}, provider={provider_name}, "
            f"model_override={model_override or 'default'}, "
            f"effort={effort or 'default'}"
        )

        logging.info(
            f"Provider ready: mode={mode}, provider={provider_name}, "
            f"backend={provider.__class__.__name__}, "
            f"model={provider.model or 'default'}, "
            f"effort={getattr(provider, 'effort', None) or 'default'}"
        )

        # Ensure all necessary directories exist
        for dir_path in [INPUT_DIR, OUTPUT_DIR, DONE_DIR, KNOWLEDGE_DIR, LOGS_DIR]:
            dir_path.mkdir(exist_ok=True)

        # Load essential data
        keywords, template, worked_example = read_project_knowledge()
        progress = load_progress()
        failed_files = load_failed_files()
        logging.info(f"Loaded {len(progress)} processed, {len(failed_files)} failed files")

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
                        file_path, keywords, template, provider, worked_example
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

    except ValueError as e:
        exit_code = 2
        logging.critical(str(e))
        print(str(e), file=sys.stderr)
    except Exception as e:
        exit_code = 1
        logging.critical(f"--- CRITICAL STARTUP ERROR: {e} ---", exc_info=True)
        print(
            f"CRITICAL ERROR: {time.strftime('%Y-%m-%d %H:%M:%S')} - {e}",
            file=sys.stderr,
        )
    finally:
        logging.info("--- Science Paper Summariser Stopped ---\n")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
