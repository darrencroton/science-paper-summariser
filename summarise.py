import os
import sys
import re
import time
import logging
import signal # Used for graceful shutdown handling
from dotenv import load_dotenv # Loads environment variables from .env file
from pathlib import Path # For object-oriented path manipulation
from marker.converters.pdf import PdfConverter # marker-pdf for PDF text extraction
from marker.models import create_model_dict   # marker-pdf model loading
from marker.output import text_from_rendered # marker-pdf output formatting
from marker.config.parser import ConfigParser  # marker-pdf configuration
from llm_providers import create_llm_provider # Factory for creating LLM provider instances

# --- Configuration and Paths ---
# Load API keys and other environment variables from a .env file
load_dotenv()

# Determine LLM provider and optional model from command-line arguments
LLM_PROVIDER = sys.argv[1] if len(sys.argv) > 1 else 'claude' # Default to 'claude'
LLM_MODEL = sys.argv[2] if len(sys.argv) > 2 else None # Optional model override

# Define key directory paths relative to the script location
SCRIPT_DIR = Path(__file__).parent.absolute()
INPUT_DIR = SCRIPT_DIR / "input"           # Directory for incoming papers
OUTPUT_DIR = SCRIPT_DIR / "output"         # Directory for generated summaries
LOGS_DIR = SCRIPT_DIR / "logs"             # Directory for log files
DONE_DIR = SCRIPT_DIR / "processed"      # Directory for successfully processed papers
KNOWLEDGE_DIR = SCRIPT_DIR / "project_knowledge" # Directory for prompt templates/keywords

# Define specific log file paths
PROGRESS_FILE = LOGS_DIR / "completed.log" # Tracks successfully processed filenames
FAILED_FILE = LOGS_DIR / "failed.log"       # Tracks permanently failed filenames
PROMPT_DEBUG_FILE = LOGS_DIR / "prompt.txt" # File to dump the last used prompt for debugging
# --- End Configuration ---

# --- Global flag for graceful shutdown ---
shutdown_requested = False

def handle_shutdown_signal(signum, frame):
    """
    Signal handler for SIGINT (Ctrl+C) and SIGTERM (kill).
    Sets the global shutdown_requested flag to True to allow the main loop to exit gracefully.
    """
    global shutdown_requested
    if not shutdown_requested: # Prevent potential duplicate logs if signal delivered rapidly
        signal_name = signal.Signals(signum).name
        logging.info(f"\n--- {signal_name} received. Initiating graceful shutdown... ---")
        shutdown_requested = True
    else:
        logging.info(f"--- Shutdown already in progress (Signal {signal.Signals(signum).name} received again) ---")

def setup_logging():
    """
    Configure the Python logging system.
    Sets up a handler to write all logs (INFO level and above) to standard output (stdout).
    The start script redirects stdout to the 'history.log' file.
    """
    LOGS_DIR.mkdir(exist_ok=True) # Ensure the logs directory exists
    root_logger = logging.getLogger() # Get the root logger

    # Remove any handlers potentially configured by imported libraries (e.g., marker)
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    root_logger.setLevel(logging.INFO) # Set the minimum log level to INFO
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # Create a handler that writes to standard output
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler) # Add the handler to the root logger

def load_progress():
    """Load the set of successfully processed filenames from the progress file."""
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
                # Read filenames, stripping whitespace, ignore empty lines
                return {line.strip() for line in f if line.strip()}
        except Exception as e:
            logging.error(f"Error loading progress file {PROGRESS_FILE}: {e}")
            return set() # Return empty set on error to avoid reprocessing everything
    return set() # Return empty set if file doesn't exist

def load_failed_files():
    """Load the set of filenames that permanently failed processing."""
    if FAILED_FILE.exists():
        try:
            with open(FAILED_FILE, 'r', encoding='utf-8') as f:
                failed = set()
                for line in f:
                    # Expected format: filename|timestamp|error_message
                    if line.strip() and '|' in line:
                        failed.add(line.strip().split('|', 1)[0]) # Get only the filename part
                return failed
        except Exception as e:
            logging.error(f"Error loading failed files list {FAILED_FILE}: {e}")
            return set() # Return empty set on error
    return set() # Return empty set if file doesn't exist

def save_progress(processed_files):
    """Save the updated set of successfully processed filenames."""
    try:
        # Write the sorted list of filenames, one per line
        with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
            for filename in sorted(processed_files):
                f.write(f"{filename}\n")
    except Exception as e:
        logging.error(f"Error saving progress to {PROGRESS_FILE}: {e}")

def add_to_failed_files(filename, error):
    """Append a filename and error details to the permanently failed list."""
    try:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        # Sanitize error message for writing to a single line
        error_str = str(error).replace('\n', ' ').replace('\r', '')
        # Append filename|timestamp|error to the file
        with open(FAILED_FILE, 'a', encoding='utf-8') as f:
            f.write(f"{filename}|{timestamp}|{error_str}\n")
        logging.info(f"Added {filename} to failed files list.")
    except Exception as e:
        logging.error(f"Error adding {filename} to failed list {FAILED_FILE}: {e}")

def read_project_knowledge():
    """
    Read essential knowledge files (keywords, template) needed for prompts.
    Raises an exception if critical files are missing.
    """
    try:
        keywords_path = KNOWLEDGE_DIR / "astronomy-keywords.txt"
        template_path = KNOWLEDGE_DIR / "paper-summary-template.md"
        # Read keyword list
        with open(keywords_path, 'r', encoding='utf-8') as f:
            keywords = f.read()
        # Read summary template
        with open(template_path, 'r', encoding='utf-8') as f:
            template = f.read()
        return keywords, template
    except FileNotFoundError as e:
        # If these files are missing, the application cannot function correctly
        logging.critical(f"CRITICAL: Project knowledge file not found: {e}. Cannot continue.")
        raise # Stop execution
    except Exception as e:
        logging.critical(f"CRITICAL: Failed to read project knowledge files: {e}")
        raise # Stop execution

def read_input_file(file_path: Path, llm_provider):
    """
    Read content from an input file (PDF or TXT).
    Handles PDF text extraction using marker-pdf if the LLM provider
    doesn't support direct PDF uploads.
    Returns (content, error_message). content is bytes for direct PDF, str otherwise.
    """
    try:
        # Check file size limit (e.g., 100MB)
        file_size = file_path.stat().st_size
        if file_size > 100 * 1024 * 1024:
            return None, f"File too large: {file_size / 1024 / 1024:.1f}MB (max 100MB)"

        file_suffix = file_path.suffix.lower()

        if file_suffix == '.pdf':
            # If provider handles PDFs directly, read as binary
            if llm_provider.supports_direct_pdf():
                logging.info(f"Reading PDF binary for direct provider upload: {file_path.name}")
                with open(file_path, 'rb') as f:
                    return f.read(), None
            # Otherwise, extract text using marker-pdf
            else:
                logging.info(f"Extracting text from PDF using marker-pdf: {file_path.name}")
                # Configure marker-pdf (minimal config for text extraction)
                config = {"output_format": "markdown", "disable_image_extraction": True, "use_llm": False}
                # Add Ollama specific config if needed (example, may not be needed if marker handles it)
                if llm_provider.__class__.__name__ == "OllamaProvider":
                     logging.warning("Ollama provider detected for marker text extraction - ensure marker is configured if needed.")
                    # Example config, adjust if marker requires explicit Ollama setup
                    # config.update({
                    #     "llm_service": "marker.services.ollama.OllamaService",
                    #     "ollama_base_url": "http://localhost:11434", # Assuming default
                    #     "ollama_model": llm_provider.model
                    # })

                config_parser = ConfigParser(config)
                model_lst = create_model_dict() # Load default marker models
                converter_kwargs = {
                    "config": config_parser.generate_config_dict(),
                    "artifact_dict": model_lst,
                    "processor_list": config_parser.get_processors(),
                    "renderer": config_parser.get_renderer()
                }
                # Add llm_service if needed based on config
                # if llm_provider.__class__.__name__ == "OllamaProvider":
                #     converter_kwargs["llm_service"] = config_parser.get_llm_service()

                converter = PdfConverter(**converter_kwargs)
                rendered = converter(str(file_path)) # Convert PDF
                text, _, _ = text_from_rendered(rendered) # Extract text from marker output
                logging.info(f"Extracted ~{len(text.split())} words from PDF for {llm_provider.__class__.__name__}")
                return text, None # Return extracted text as string
        elif file_suffix == '.txt':
            # Read plain text files directly
            logging.info(f"Reading text file: {file_path.name}")
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read(), None # Return text as string
        else:
            # Handle unsupported file types
            return None, f"Unsupported file type: {file_path.suffix}"

    except Exception as e:
        # Log errors during file reading/processing
        logging.error(f"Error reading input file {file_path.name}: {e}", exc_info=True) # Log traceback
        return None, f"File read/processing error: {e}"

def create_system_prompt(keywords):
    """Construct the static system prompt defining the LLM's role, rules, and knowledge."""
    # This prompt sets the context and constraints for the LLM's summarization task.
    return (
        "<role>\n"
        # Role definition... 
        "You are an esteemed professor of astrophysics at Harvard University "
        "specializing in analyzing research papers. Your expertise includes:\n"
        "- Identifying key scientific results and their significance\n"
        "- Writing in clear technical UK English\n"
        "- Supporting all claims with precise quotations\n"
        "- Using LaTeX for mathematical expressions\n"
        "</role>\n\n"
        "<rules>\n"
        # Strict rules for output format and content... 
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
        # Injecting the astronomy keywords as reference data
        f"Available astronomy keywords by category:\n{keywords}\n"
        "</knowledgeBase>"
    )

def create_user_prompt(paper_text, template, is_pdf=False):
    """
    Construct the user prompt, including the task, template, and optionally the paper text.
    If paper_text is empty, it implies the LLM receives the paper via other means (e.g., direct PDF upload).
    """
    # Defines the specific task and provides the output structure template
    base_prompt = (
        "<task>\n"
        # Task instructions... 
        "Summarize this research paper following these EXACT requirements:\n\n"
        "<format>\n"
        # Formatting requirements... 
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
        # Inject the summary template structure
        f"Use this exact structure:\n{template}\n"
        "</template>\n\n"
        "<tags>\n"
        # Specific instructions for the Tags section... 
        "The Tags section must have two parts:\n"
        "1. First line: Hashtags for telescopes, surveys, datasets, models (proper nouns only)\n"
        "2. Second line: Science area hashtags (use ONLY provided keywords, only the best ones)\n"
        "</tags>\n"
        "</task>\n\n"
    )

    # If paper text was extracted (i.e., not direct PDF upload), include it in the prompt
    if paper_text:
        base_prompt += (
            "<input>\n"
            f"Paper to summarize:\n\n"
            f"---BEGIN PAPER---\n{paper_text}\n---END PAPER---\n"
            "</input>"
        )
    return base_prompt

def process_file(file_path, keywords, template):
    """
    Process a single input file: read, generate prompts, call LLM, handle retries,
    validate, save, and move the file.
    Returns (success_boolean, original_filename, error_message_or_none).
    """
    is_pdf = file_path.suffix.lower() == '.pdf'
    last_error = "Unknown processing error during LLM call or post-processing" # Default error

    try:
        # Create the appropriate LLM provider instance
        provider_config = {"model": LLM_MODEL} if LLM_MODEL else {}
        llm_provider = create_llm_provider(LLM_PROVIDER, config=provider_config)
        logging.info(f"Using provider: {llm_provider.__class__.__name__}, Model: {llm_provider.model}")

        # Read the file content (binary PDF or extracted text)
        paper_content, error = read_input_file(file_path, llm_provider)
        if error:
            # If reading fails, return immediately, don't retry
            return False, file_path.name, error

        logging.info(f"Content type for LLM: {type(paper_content).__name__}")

        # Determine if the extracted text needs to be included in the user prompt
        system_prompt = create_system_prompt(keywords)
        include_text_in_prompt = not (is_pdf and llm_provider.supports_direct_pdf())

        paper_content_text = "" # Default to empty text for prompt
        if include_text_in_prompt:
            if isinstance(paper_content, bytes):
                # This case should ideally not happen if read_input_file works correctly
                logging.warning("Inconsistency: Text needed for prompt, but content is bytes. Attempting decode.")
                try:
                    paper_content_text = paper_content.decode('utf-8', errors='ignore')
                except Exception as decode_err:
                    return False, file_path.name, f"Cannot decode binary PDF content for text-based prompt: {decode_err}"
            else:
                paper_content_text = paper_content # Use the extracted text

        # Create the final user prompt
        user_prompt = create_user_prompt(paper_content_text, template, is_pdf)

        # Log prompt size information for debugging
        prompt_info = f"System prompt: {len(system_prompt)} chars. User prompt base: {len(user_prompt) - len(paper_content_text)} chars."
        if include_text_in_prompt:
            prompt_info += f" Included text: {len(paper_content_text)} chars."
        logging.info(prompt_info)

        # Write the full prompt to a debug file
        try:
            full_prompt = f"SYSTEM PROMPT\n{system_prompt}\n\n---\n\nUSER PROMPT\n{user_prompt}"
            with open(PROMPT_DEBUG_FILE, 'w', encoding='utf-8') as f:
                f.write(full_prompt)
            logging.info(f"Full prompt for {file_path.name} written to {PROMPT_DEBUG_FILE}")
        except Exception as e:
            logging.warning(f"Could not write debug prompt file: {e}")

        # --- LLM Call with Retry Logic ---
        max_retries = 3
        for attempt in range(max_retries):
            # Check for shutdown signal before making the potentially long LLM call
            if shutdown_requested:
                logging.warning(f"Shutdown requested before LLM call (Attempt {attempt+1}) for {file_path.name}.")
                return False, file_path.name, "Shutdown requested"

            try:
                logging.info(f"Attempt {attempt + 1}/{max_retries} calling LLM for {file_path.name}...")
                # Call the LLM provider to process the document
                # Pass the original paper_content (binary or text)
                summary_content = llm_provider.process_document(
                    content=paper_content,
                    is_pdf=is_pdf,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_tokens=8192 # Max tokens for the *response* generation
                )

                # Basic check for empty response
                if not summary_content or not summary_content.strip():
                    raise ValueError("LLM returned empty or whitespace-only summary content.")

                logging.info(f"LLM processing successful for {file_path.name} (received ~{len(summary_content)} chars).")

                # Post-process the summary
                summary_content = strip_preamble(summary_content)
                validate_summary(summary_content)
                save_summary(summary_content, file_path)

                # Move the original file to the 'processed' directory
                moved_path = move_to_done(file_path, summary_content)
                if moved_path:
                    # Success! Return True and the original filename for progress tracking
                    return True, file_path.name, None
                else:
                    # If moving the file fails, consider it a failure for this attempt
                    last_error = f"Processed successfully but failed to move original file {file_path.name}"
                    logging.error(last_error)
                    break # Don't retry if only the move failed

            except Exception as e:
                # Handle errors during the LLM call or post-processing
                provider_name = llm_provider.__class__.__name__.replace("Provider", "")
                error_type = e.__class__.__name__
                last_error = f"Attempt {attempt + 1} failed - Provider: {provider_name}, ErrorType: {error_type}, Details: {str(e)}"
                logging.error(last_error, exc_info=False) # Keep traceback off by default for cleaner logs

                if attempt < max_retries - 1:
                    wait_time = 2 ** (attempt + 1) # Exponential backoff (2, 4, 8 seconds)
                    logging.info(f"Waiting {wait_time} seconds before retry...")
                    # Wait interruptibly, checking for shutdown signal
                    wait_start = time.monotonic()
                    while time.monotonic() - wait_start < wait_time:
                        if shutdown_requested:
                            logging.warning("Shutdown requested during retry wait. Aborting.")
                            return False, file_path.name, "Shutdown requested"
                        time.sleep(0.2) # Brief sleep to yield CPU
                else:
                    # Max retries reached
                    logging.error(f"Max retries reached for {file_path.name}.")
                    # last_error holds the error from the final attempt
                    break # Exit retry loop
            # --- End LLM Call with Retry Logic ---

    except Exception as e:
        # Catch errors during setup before the retry loop (e.g., provider init, initial read)
        last_error = f"Preprocessing error for {file_path.name}: {e.__class__.__name__}: {str(e)}"
        logging.error(last_error, exc_info=True) # Log full traceback for setup errors

    # If we reach here, processing failed (either max retries or setup error)
    return False, file_path.name, last_error


def strip_preamble(summary_content):
    """Remove any text before the first Markdown heading ('# ')."""
    lines = summary_content.split('\n')
    title_index = -1
    # Find the index of the first line that starts with '# ' after stripping whitespace
    for i, line in enumerate(lines):
        if line.strip().startswith('# '):
            title_index = i
            break

    if title_index > 0:
        # If preamble exists, log its removal and return the rest
        preamble = '\n'.join(lines[:title_index])
        logging.info(f"Removed {len(preamble.splitlines())} lines of preamble before paper title.")
        return '\n'.join(lines[title_index:])
    elif title_index == 0:
        # Summary correctly starts with the title
        logging.info("Summary starts correctly with title line.")
        return summary_content
    else:
        # If no title heading found, log a warning and return original content
        logging.warning("Could not find title heading ('# ') to strip preamble.")
        return summary_content


def validate_summary(summary_content):
    """Perform basic validation checks on the generated summary format and log results."""
    lines = [line.strip() for line in summary_content.split('\n') if line.strip()]
    if not lines:
        logging.warning("VALIDATION WARNING: Summary content is empty, cannot validate.")
        return

    # --- Perform Checks ---
    start_with_title = lines[0].startswith('# ')
    # Check for a 4-digit year (19xx or 20xx) within the first 5 lines
    year_found = any(re.search(r'\b(19|20)\d{2}\b', line) for line in lines[:5])
    # Check if 'et al' appears in the Authors line (first 5 lines)
    author_complete = not any('et al' in line.lower() for line in lines[:5] if line.startswith('Authors:'))
    # Count bullet points (starting with '- ' or '  - ') that are not footnotes
    bullet_count = sum(1 for line in lines if (line.startswith('- ') or line.startswith('  - ')) and not line.startswith('[^'))
    # Count footnote definition lines (starting with '[^')
    footnote_count = sum(1 for line in lines if line.startswith('[^'))
    footnote_lines = [line for line in lines if line.startswith('[^')]
    # Count footnotes that appear correctly quoted (contain at least two quote marks)
    properly_quoted_count = sum(1 for line in footnote_lines if '"' in line and line.count('"') >= 2)
    all_footnotes_quoted = (properly_quoted_count == footnote_count) if footnote_count > 0 else True # True if no footnotes

    # Check for Glossary section heading
    has_glossary = any(line.startswith('## Glossary') for line in lines)
    # Check for Tags section heading and structure
    has_tags = False
    has_two_tag_lines = False
    try:
        # Find the index of the '## Tags' line
        tags_index = next(i for i, line in enumerate(lines) if line.startswith('## Tags'))
        has_tags = True
        # Get the next two non-empty lines *after* '## Tags'
        tag_content_lines = [line for line in lines[tags_index+1:] if line][:2]
        # Check if there are exactly two lines and both start with '#'
        has_two_tag_lines = (len(tag_content_lines) == 2 and all(l.startswith('#') for l in tag_content_lines))
    except StopIteration:
        # '## Tags' heading not found
        has_tags = False

    # --- Log Validation Results ---
    logging.info("Validation results:")
    logging.info(f"- Starts with title ('# '): {start_with_title}")
    logging.info(f"- Year found near top: {year_found}")
    logging.info(f"- Author list appears complete (no 'et al'): {author_complete}")
    logging.info(f"- Bullet points found: {bullet_count}")
    logging.info(f"- Footnotes found ([^...]): {footnote_count}")
    logging.info(f"- All footnotes appear properly quoted: {all_footnotes_quoted} ({properly_quoted_count}/{footnote_count})")
    logging.info(f"- Has '## Glossary' section: {has_glossary}")
    logging.info(f"- Has '## Tags' section: {has_tags}")
    logging.info(f"- Tags section has two hashtag lines: {has_two_tag_lines}")

    # --- Log Specific Warnings ---
    if not start_with_title: logging.warning("VALIDATION WARNING: Summary does not start with title heading '# '")
    if not year_found: logging.warning("VALIDATION WARNING: No year found near top of summary")
    if not author_complete: logging.warning("VALIDATION WARNING: Author list might be truncated ('et al.' found)")
    if bullet_count != footnote_count: logging.warning(f"VALIDATION WARNING: Mismatch between bullets ({bullet_count}) and footnotes ({footnote_count})")
    if not all_footnotes_quoted and footnote_count > 0: logging.warning(f"VALIDATION WARNING: Not all footnotes appear properly quoted ({properly_quoted_count}/{footnote_count})")
    if not has_glossary: logging.warning("VALIDATION WARNING: Missing '## Glossary' section")
    if not has_tags: logging.warning("VALIDATION WARNING: Missing '## Tags' section")
    elif not has_two_tag_lines: logging.warning("VALIDATION WARNING: '## Tags' section missing two subsequent hashtag lines")
    # --- End Validation ---


def extract_metadata(summary_content):
    """Extract author surnames, year, and title from the summary markdown content."""
    lines = [l.strip() for l in summary_content.split('\n') if l.strip()]
    if not lines: return 'Untitled', [], None # Default values if summary is empty

    # Title is the first line, removing '# '
    title = lines[0].replace('# ', '', 1).strip() if lines[0].startswith('# ') else 'Untitled'
    year = None
    authors_surnames = []

    # Scan the first few lines (e.g., 2nd to 5th) for Authors and Published year
    for line in lines[1:5]:
        # Extract author surnames from the 'Authors:' line
        if line.startswith('Authors: '):
            author_line = line.replace('Authors: ', '')
            # Split by comma, then extract the first word (assumed surname)
            for part in author_line.split(','):
                name_parts = part.strip().split()
                if name_parts:
                    surname = name_parts[0]
                    # Basic check to avoid adding initials like 'A.'
                    if surname and not surname.endswith('.') and len(surname) > 1:
                        authors_surnames.append(surname)
        # Extract the first 4-digit year found (19xx or 20xx)
        if not year: # Only take the first year found
            if year_match := re.search(r'\b(19|20)\d{2}\b', line):
                year = year_match.group()

    return title, authors_surnames, year

def format_authors(authors_surnames):
    """Format the list of author surnames for use in filenames (e.g., Smith et al.)."""
    count = len(authors_surnames)
    if count == 0: return "UnknownAuthor"
    if count == 1: return authors_surnames[0]
    if count == 2: return f"{authors_surnames[0]} and {authors_surnames[1]}"
    # Use "et al." for 3 or more authors
    return f"{authors_surnames[0]} et al."

def sanitize_filename(filename_part):
    """
    Remove or replace characters unsafe for filenames and limit length.
    Applies to individual components like author, year, or title part.
    """
    if not isinstance(filename_part, str): filename_part = str(filename_part)

    # Define characters typically unsafe in filesystems
    unsafe_chars = r'<>:"/\\|?*' + '\n\r\t'
    # Remove unsafe characters
    for char in unsafe_chars:
        filename_part = filename_part.replace(char, '')
    # Remove leading/trailing whitespace and trailing dots
    filename_part = filename_part.strip().rstrip('.')

    # Handle potentially empty string after sanitization
    if not filename_part: return "sanitized_empty"

    # Enforce a maximum byte length (safer than char length for multi-byte chars)
    max_bytes = 240 # Use a safe limit below common filesystem max (e.g., 255)
    while len(filename_part.encode('utf-8')) > max_bytes:
        filename_part = filename_part[:-1] # Shorten by one character
        # Ensure it doesn't become empty during shortening
        if not filename_part: return "sanitized_short"

    return filename_part.strip() # Final strip just in case

def create_base_filename(title, authors_surnames, year, input_path):
    """
    Create a sanitized base filename (stem, no extension) using metadata.
    Format: "Author(s) - Year - TitleStart".
    Falls back to the sanitized input filename stem if year is missing.
    """
    # Use metadata only if year is present
    if year:
        formatted_authors = format_authors(authors_surnames)
        # Sanitize each component
        safe_authors = sanitize_filename(formatted_authors)
        safe_year = sanitize_filename(year)
        safe_title = sanitize_filename(title[:80]) # Use first 80 chars of title

        # Combine sanitized parts, filtering out any that became empty
        parts = [part for part in [safe_authors, safe_year, safe_title] if part]
        if parts:
            base_name = " - ".join(parts)
            logging.info(f"Created metadata-based filename stem: {base_name}")
            return base_name
        else:
            # This should be rare if sanitization fallbacks work
            logging.warning("Could not create filename from metadata (all parts empty). Falling back.")

    # Fallback: use the sanitized stem of the original input filename
    fallback_stem = sanitize_filename(input_path.stem)
    logging.warning(f"Using fallback filename stem based on input: {fallback_stem}")
    # Ensure fallback stem is not empty
    return fallback_stem if fallback_stem else "fallback_filename"


def save_summary(summary_content, input_path):
    """Save the generated summary content to a markdown file in the output directory."""
    try:
        # Extract metadata to create the filename
        title, authors, year = extract_metadata(summary_content)
        base_filename = create_base_filename(title, authors, year, input_path)
        output_filename = f"{base_filename}.md" # Add markdown extension
        output_path = OUTPUT_DIR / output_filename

        # Ensure output directory exists
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        # Write the summary content to the file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(summary_content)
        logging.info(f"Summary saved to: {output_path}")
    except Exception as e:
        # Log errors during file saving
        logging.error(f"Error saving summary for {input_path.name}: {e}", exc_info=True)

def move_to_done(file_path, summary_content):
    """
    Move the successfully processed input file to the 'processed' directory.
    Renames the file based on extracted metadata, handling potential name conflicts.
    Returns the destination Path object or None if move fails.
    """
    try:
        DONE_DIR.mkdir(exist_ok=True) # Ensure processed directory exists
        # Extract metadata to determine the new filename base
        title, authors, year = extract_metadata(summary_content)
        base_filename = create_base_filename(title, authors, year, file_path)
        # Construct the initial target filename using the original file's suffix
        target_filename = f"{base_filename}{file_path.suffix}"
        dest_path = DONE_DIR / target_filename

        # Handle potential filename collisions by adding a counter (_1, _2, etc.)
        counter = 1
        original_dest_path = dest_path # Keep original name for logging warning
        while dest_path.exists():
            # If file exists, append counter to the base filename
            target_filename = f"{base_filename}_{counter}{file_path.suffix}"
            dest_path = DONE_DIR / target_filename
            counter += 1
            # Log only the first time a conflict is detected for a file
            if counter == 2:
                logging.warning(f"Destination file {original_dest_path.name} exists, trying with counter.")
            logging.info(f"Renaming processed file to: {target_filename}")

        # Perform the file move operation
        file_path.rename(dest_path)
        logging.info(f"Moved {file_path.name} to done directory as {target_filename}")
        return dest_path # Return the final destination path

    except Exception as e:
        # Log errors during file moving
        logging.error(f"Error moving file {file_path.name} to done directory: {e}", exc_info=True)
        return None # Return None indicates failure


def get_pending_files(input_dir, processed_log, failed_log):
    """
    Scan the input directory for files (.pdf, .txt) that haven't been processed
    successfully and are not marked as permanently failed. Skips zero-byte files.
    Returns a list of Path objects for pending files.
    """
    pending = []
    try:
        # Iterate through all files in the input directory
        for f in input_dir.glob("*.*"):
            # Check if it's a file and has a supported extension
            if f.is_file() and f.suffix.lower() in ['.pdf', '.txt']:
                # Check if it's already processed or permanently failed
                if f.name not in processed_log and f.name not in failed_log:
                    # Check if file has content (size > 0 bytes)
                    try:
                        if f.stat().st_size > 0:
                            pending.append(f)
                        else:
                            logging.warning(f"Skipping zero-byte file: {f.name}")
                    except OSError as e:
                        # Handle potential errors accessing file status (e.g., permissions)
                        logging.warning(f"Could not get status for file {f.name}, skipping: {e}")
    except Exception as e:
        # Log errors during directory scanning
        logging.error(f"Error scanning input directory {input_dir}: {e}")
    return pending # Return the list of files needing processing


def main():
    """
    Main execution function.
    Sets up logging, loads state, monitors the input directory,
    processes files one by one, and handles graceful shutdown.
    """
    # --- Register Signal Handlers ---
    # Setup handlers for SIGINT (Ctrl+C) and SIGTERM (kill command)
    # These handlers will set the global 'shutdown_requested' flag.
    signal.signal(signal.SIGINT, handle_shutdown_signal)
    signal.signal(signal.SIGTERM, handle_shutdown_signal)
    # --- End Signal Registration ---

    try:
        # Initial setup
        setup_logging()
        model_info = f" with model {LLM_MODEL}" if LLM_MODEL else ""
        logging.info(f"--- Science Paper Summariser Starting ---")
        logging.info(f"Process ID: {os.getpid()}") # Log PID for easier process management
        logging.info(f"Using LLM Provider: {LLM_PROVIDER}{model_info}")

        # Ensure all necessary directories exist
        for dir_path in [INPUT_DIR, OUTPUT_DIR, DONE_DIR, KNOWLEDGE_DIR, LOGS_DIR]:
            dir_path.mkdir(exist_ok=True)
            logging.info(f"Ensured directory exists: {dir_path}")

        # Load essential data for processing and state
        keywords, template = read_project_knowledge() # Exits if critical files missing
        progress = load_progress()          # Load set of successfully processed files
        failed_files = load_failed_files() # Load set of permanently failed files

        logging.info(f"Loaded {len(progress)} processed files from {PROGRESS_FILE.name}")
        if failed_files:
            logging.info(f"Loaded {len(failed_files)} permanently failed files from {FAILED_FILE.name} (will be skipped)")

        logging.info(f"Monitoring input directory: {INPUT_DIR.absolute()}")
        logging.info("--- Waiting for files (Send SIGTERM or press Ctrl+C to exit gracefully) ---")

        waiting_message_shown = False

        # --- Main Processing Loop ---
        # Loop continues until a shutdown signal is received
        while not shutdown_requested:
            try:
                # Check for new files needing processing
                pending = get_pending_files(INPUT_DIR, progress, failed_files)

                if pending and not shutdown_requested: # Double-check signal before starting work
                    file_path = pending[0] # Process one file per loop iteration
                    waiting_message_shown = False # Reset waiting message flag
                    logging.info(f"--- Processing file: {file_path.name} ---")

                    # Call the main processing function for the file
                    success, filename, error_msg = process_file(file_path, keywords, template)

                    # --- Handle Processing Result ---
                    if success:
                        # If successful, add original filename to progress set and save
                        progress.add(filename)
                        save_progress(progress)
                        logging.info(f"--- Finished processing: {filename} (Success) ---")
                    # Check if failure was due to shutdown, not a processing error
                    elif error_msg != "Shutdown requested":
                        # If failed for other reasons, log error and add to permanent failed list
                        logging.error(f"--- Finished processing: {filename} (FAILED) ---")
                        logging.error(f"Failure reason: {error_msg}")
                        add_to_failed_files(filename, error_msg)
                        failed_files.add(filename) # Update in-memory set to prevent immediate retry
                    else:
                        # Processing was aborted due to shutdown signal
                        logging.info(f"--- Processing stopped for {filename} due to shutdown request ---")
                    # --- End Handle Processing Result ---

                    # Short pause before checking for the next file, unless shutting down
                    if not shutdown_requested:
                        logging.info("--- Checking for next file ---")
                        time.sleep(2) # Short pause

                elif not waiting_message_shown:
                    # No pending files found, show waiting message once
                    logging.info("No files in queue. Waiting...")
                    waiting_message_shown = True
                    # Wait for ~10 seconds, but check for shutdown signal periodically
                    wait_start = time.monotonic()
                    while not shutdown_requested and time.monotonic() - wait_start < 10:
                        time.sleep(0.5) # Check for signal every 0.5s
                else:
                    # Idle, waiting message already shown. Wait ~5 seconds.
                     wait_start = time.monotonic()
                     while not shutdown_requested and time.monotonic() - wait_start < 5:
                         time.sleep(0.5) # Check for signal every 0.5s

            except Exception as e:
                # Catch unexpected errors within the main loop's try block
                logging.critical(f"--- UNEXPECTED ERROR in main loop: {e} ---", exc_info=True)
                if not shutdown_requested:
                    # Pause briefly before attempting to continue the loop
                    logging.critical("Pausing for 15 seconds before attempting to continue...")
                    wait_start = time.monotonic()
                    while not shutdown_requested and time.monotonic() - wait_start < 15:
                        time.sleep(0.5)
        # --- End Main Processing Loop ---

    except Exception as e:
        # Catch critical errors during initial setup (before the main loop starts)
        logging.critical(f"--- CRITICAL STARTUP ERROR: {e} ---", exc_info=True)
        # Attempt fallback logging to stderr if logging setup failed
        print(f"CRITICAL FALLBACK: {time.strftime('%Y-%m-%d %H:%M:%S')} - {e}", file=sys.stderr)
    finally:
        # This block always executes, ensuring shutdown message is logged
        logging.info("--- Science Paper Summariser Stopped ---\n")


if __name__ == "__main__":
    # Entry point when script is executed directly
    main()