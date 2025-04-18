"""
Scientific Paper Summarizer

This script continuously monitors a directory for PDF or text files containing scientific papers,
processes them with an LLM provider, and creates structured summaries following a
specific template. The system supports multiple LLM providers through a pluggable
architecture and provides extensive validation and error handling.

Usage:
    python summarise.py [provider] [model]
    
    provider: LLM provider to use (claude, openai, perplexity, gemini, ollama)
              defaults to 'claude' if not specified
    model:    Optional model name specific to the chosen provider
              if not specified, the provider default is used

Directory Structure:
    input/       - Place papers to be summarized here (.pdf or .txt)
    output/      - Generated summaries are saved here (.md)
    processed/   - Successfully processed papers are moved here
    logs/        - Processing logs and error information
    project_knowledge/ - Templates and domain-specific knowledge
"""

import os
import sys
import re
import time
import logging
import requests
import json
import base64
from dotenv import load_dotenv
from pathlib import Path
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from marker.config.parser import ConfigParser
from llm_providers import create_llm_provider

# ===== CONFIGURATION AND SETUP =====

# Load environment variables from .env file
load_dotenv()

# Define LLM provider from command line args or use default
LLM_PROVIDER = sys.argv[1] if len(sys.argv) > 1 else 'claude'

# Define the provider-specific model (if provided)
LLM_MODEL = sys.argv[2] if len(sys.argv) > 2 else None

# Define absolute paths for directory structure
SCRIPT_DIR = Path(__file__).parent.absolute()
INPUT_DIR = SCRIPT_DIR / "input"
OUTPUT_DIR = SCRIPT_DIR / "output"
LOGS_DIR = SCRIPT_DIR / "logs"
DONE_DIR = SCRIPT_DIR / "processed"
PROGRESS_FILE = LOGS_DIR / "completed.log"
FAILED_FILE = LOGS_DIR / "failed.log"
KNOWLEDGE_DIR = SCRIPT_DIR / "project_knowledge"

# ===== LOGGING AND ENVIRONMENT FUNCTIONS =====

def check_environment():
    """
    Check that required environment variables and dependencies are available.
    Provider-specific checks are handled in the respective provider classes.
    """
    pass

def setup_logging():
    """
    Configure logging to write to both a file and the console.
    
    Creates a logging configuration that:
    - Timestamps each message
    - Logs at INFO level and above
    - Outputs to both a log file and the terminal
    """
    LOGS_DIR.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(LOGS_DIR / "history.log"),
            logging.StreamHandler()
        ]
    )

def log_message(message):
    """
    Write a message to the system log.
    
    Args:
        message (str): The message to log
    """
    logging.info(message)

# ===== PROGRESS TRACKING FUNCTIONS =====

def load_progress():
    """
    Load list of previously processed files from the completed.log file.
    
    Returns:
        set: Set of filenames that have already been processed
    """
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'r') as f:
            return {line.strip() for line in f if line.strip()}
    return set()

def load_failed_files():
    """
    Load list of files that permanently failed processing.
    
    Returns:
        set: Set of filenames that have failed processing and shouldn't be retried
    """
    if FAILED_FILE.exists():
        with open(FAILED_FILE, 'r') as f:
            return {line.strip().split('|')[0] for line in f if line.strip()}
    return set()

def save_progress(processed_files):
    """
    Save the current list of processed files to completed.log.
    
    Args:
        processed_files (set): Set of successfully processed filenames
    """
    with open(PROGRESS_FILE, 'w') as f:
        for filename in sorted(processed_files):
            f.write(f"{filename}\n")

def add_to_failed_files(filename, error):
    """
    Add a file to the permanently failed list with timestamp and error details.
    
    Args:
        filename (str): Name of the file that failed processing
        error (str): Error message or reason for failure
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(FAILED_FILE, 'a') as f:
        f.write(f"{filename}|{timestamp}|{error}\n")
    log_message(f"Added {filename} to failed files list - will be skipped in future processing attempts")

def get_pending_files(input_dir, progress, failed):
    """
    Get list of unprocessed files, excluding previously processed and failed files.
    
    Args:
        input_dir (Path): Directory to scan for new files
        progress (set): Set of already processed filenames
        failed (set): Set of permanently failed filenames
        
    Returns:
        list: List of Path objects for files that need processing
    """
    return [
        f for f in input_dir.glob("*.*") 
        if f.suffix.lower() in ['.pdf', '.txt'] 
        and f.name not in progress
        and f.name not in failed
    ]

# ===== PROJECT KNOWLEDGE FUNCTIONS =====

def read_project_knowledge():
    """
    Read domain-specific knowledge files for use in prompt construction.
    
    Reads:
    - astronomy-keywords.txt: Contains domain-specific terminology
    - paper-summary-template.md: Template for generating consistent summaries
    
    Returns:
        tuple: (keywords string, template string)
        
    Raises:
        Exception: If knowledge files cannot be read
    """
    try:
        with open(KNOWLEDGE_DIR / "astronomy-keywords.txt") as f:
            keywords = f.read()
        
        with open(KNOWLEDGE_DIR / "paper-summary-template.md") as f:
            template = f.read()
        
        return keywords, template
    except Exception as e:
        logging.error(f"Failed to read project knowledge files: {str(e)}")
        raise

# ===== FILE PROCESSING FUNCTIONS =====

def read_input_file(file_path: Path, llm_provider):
    """
    Read and process input file based on file type and provider capabilities.
    
    Handles both text files and PDFs, with special handling for providers
    that support direct PDF input vs. those requiring text extraction.
    
    Args:
        file_path (Path): Path to the input file
        llm_provider (LLMProvider): Provider instance to check capabilities
        
    Returns:
        tuple: (content, error_message)
            - content: File contents as text or binary data
            - error_message: Error message if reading failed, None if successful
    """
    try:
        # Check file size (100MB limit)
        file_size = file_path.stat().st_size
        if file_size > 100 * 1024 * 1024:  # 100MB in bytes
            return None, f"File too large: {file_size/1024/1024:.1f}MB (max 100MB)"

        if file_path.suffix.lower() == '.pdf':
            # Check if provider supports direct PDF input
            if llm_provider.supports_direct_pdf():
                # Read PDF as binary for providers that support direct PDF processing
                with open(file_path, 'rb') as f:
                    return f.read(), None
            else:
                try:
                    # For providers that need text extraction, use marker library
                    # Configure marker for text extraction
                    config = {
                        "output_format": "markdown",
                        "disable_image_extraction": True,
                        "use_llm": False
                    }
                    
                    # If using Ollama, add Ollama-specific config
                    if llm_provider.__class__.__name__ == "OllamaProvider":
                        ollama_model = llm_provider.model
                        config.update({
                            "llm_service": "marker.services.ollama.OllamaService",
                            "ollama_base_url": "http://localhost:11434",
                            "ollama_model": ollama_model
                        })
                    
                    config_parser = ConfigParser(config)
                    
                    # Initialize converter with appropriate configuration
                    converter_kwargs = {
                        "config": config_parser.generate_config_dict(),
                        "artifact_dict": create_model_dict(),
                        "processor_list": config_parser.get_processors(),
                        "renderer": config_parser.get_renderer()
                    }
                    
                    # Add llm_service if using Ollama
                    if llm_provider.__class__.__name__ == "OllamaProvider":
                        converter_kwargs["llm_service"] = config_parser.get_llm_service()
                    
                    converter = PdfConverter(**converter_kwargs)
                    
                    # Convert the PDF to text
                    rendered = converter(str(file_path))
                    text, _, _ = text_from_rendered(rendered)
                    provider_name = llm_provider.__class__.__name__.replace('Provider', '')
                    log_message(f"Extracted {len(text.split())} words from PDF for {provider_name}")
                    return text, None
                    
                except Exception as e:
                    return None, f"PDF processing error: {str(e)}"
        else:  # For .txt files
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read(), None
                
    except Exception as e:
        return None, f"File read error: {str(e)}"

# ===== PROMPT CONSTRUCTION FUNCTIONS =====
        
def create_system_prompt(keywords):
    """
    Create the system prompt defining role and expertise.
    
    Builds a comprehensive system prompt that:
    - Defines the LLM's role as an expert in astrophysics
    - Sets clear rules for output formatting and citation standards
    - Provides domain knowledge from the astronomy keywords file
    
    Args:
        keywords (str): Domain-specific terminology from knowledge file
        
    Returns:
        str: Formatted system prompt
    """
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
    """
    Create the user prompt with specific task instructions.
    
    Builds a comprehensive user prompt that:
    - Defines the exact summarization task
    - Specifies formatting requirements in detail
    - Includes the paper template structure
    - Provides guidance on tag generation
    - Optionally includes the paper text for non-PDF providers
    
    Args:
        paper_text (str or bytes): The paper content if needed in the prompt
        template (str): The summary template structure
        is_pdf (bool): Whether the input is a PDF document
        
    Returns:
        str: Formatted user prompt
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
        "2. Second line: Science area hashtags (use ONLY provided keywords)\n"
        "</tags>\n"
        "</task>\n\n"
    )
    
    # Include paper text in the prompt if provided
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
    Process a single file through the LLM pipeline.
    
    Main processing function that:
    1. Initializes the appropriate LLM provider
    2. Reads the file content based on provider capabilities
    3. Constructs appropriate prompts
    4. Sends the content to the LLM for processing
    5. Validates and saves the output
    6. Moves the processed file to the done directory
    
    Args:
        file_path (Path): Path to the file to process
        keywords (str): Domain-specific terminology for prompt construction
        template (str): Summary template structure
        
    Returns:
        tuple: (success, filename, error_message)
            - success (bool): Whether processing succeeded
            - filename (str): Name of the processed file
            - error_message (str or None): Error message if processing failed
    """
    # Determine file type
    is_pdf = file_path.suffix.lower() == '.pdf'
    
    # Initialize the LLM provider - do this first so we can use its capabilities
    provider_config = {"model": LLM_MODEL} if LLM_MODEL else {}
    llm_provider = create_llm_provider(LLM_PROVIDER, config=provider_config)
    
    # Read the file content based on provider capabilities
    paper_content, error = read_input_file(file_path, llm_provider)
    
    # Log the content type to diagnose issues
    if is_pdf:
        log_message(f"PDF content type: {type(paper_content).__name__}")
    if error:
        return False, file_path.name, error
    
    # Create appropriate prompts
    system_prompt = create_system_prompt(keywords)
    
    # Determine if we need to include text in prompt (inverse of supports_direct_pdf)
    include_text_in_prompt = not is_pdf or not llm_provider.supports_direct_pdf()
    
    # If we're including text in the prompt and the content is binary, there's a problem
    if include_text_in_prompt and is_pdf and isinstance(paper_content, bytes):
        log_message("Warning: PDF text not properly extracted for prompt inclusion")
    
    user_prompt = create_user_prompt(
        paper_text=paper_content if include_text_in_prompt else "",
        template=template,
        is_pdf=is_pdf
    )
    
    # Only log text extraction for PDFs if needed
    if is_pdf and not llm_provider.supports_direct_pdf():
        provider_name = llm_provider.__class__.__name__.replace('Provider', '')
        log_message(f"PDF text extraction for {provider_name}: text length is {len(str(paper_content)) if paper_content else 0} characters")
    
    # Write complete prompt to file for debugging
    full_prompt = f"SYSTEM PROMPT\n{system_prompt}\n\n---\n\nUSER PROMPT\n{user_prompt}"
    prompt_file = LOGS_DIR / "prompt.txt"
    with open(prompt_file, 'w') as f:
        f.write(full_prompt)
    log_message(f"Full prompt written to {prompt_file}")

    # Process with retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Use the provider to process the document
            summary_content = llm_provider.process_document(
                content=paper_content,
                is_pdf=is_pdf,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=8192  # Appropriate for summary generation across models
            )
            
            # Common post-processing
            summary_content = strip_preamble(summary_content)
            validate_summary(summary_content)
            save_summary(summary_content, file_path)
            
            moved_path = move_to_done(file_path, summary_content)
            if moved_path:
                log_message(f"Completed processing: {moved_path.name}\n")
                return True, moved_path.name, None
            else:
                return False, file_path.name, "Failed to move file"
            
        except requests.exceptions.RequestException as e:
            provider_name = llm_provider.__class__.__name__.replace('Provider', '').lower()
            error_msg = f"Attempt {attempt + 1} failed - {provider_name} API request error: {str(e)}"
            log_message(error_msg)
        except ValueError as e:
            provider_name = llm_provider.__class__.__name__.replace('Provider', '').lower()
            error_msg = f"Attempt {attempt + 1} failed - {provider_name} value error: {str(e)}"
            log_message(error_msg)
        except ImportError as e:
            provider_name = llm_provider.__class__.__name__.replace('Provider', '').lower()
            error_msg = f"Attempt {attempt + 1} failed - {provider_name} missing dependency: {str(e)}"
            log_message(error_msg)
        except Exception as e:
            provider_name = llm_provider.__class__.__name__.replace('Provider', '').lower()
            error_msg = f"Attempt {attempt + 1} failed - {provider_name} error: {e.__class__.__name__}: {str(e)}"
            log_message(error_msg)
            
        # Implement exponential backoff for retries
        if attempt < max_retries - 1:
            wait_time = 2 ** attempt
            log_message(f"Waiting {wait_time} seconds before retry...")
            time.sleep(wait_time)
        else:
            return False, file_path.name, error_msg

# ===== OUTPUT PROCESSING FUNCTIONS =====

def strip_preamble(summary_content):
    """
    Remove any text before the paper title (which must start with '# ').
    
    Some LLM providers may include preamble text like "Here's the summary..."
    that needs to be removed for clean output.
    
    Args:
        summary_content (str): Raw summary from LLM
        
    Returns:
        str: Summary with preamble removed
    """
    lines = summary_content.split('\n')
    title_index = -1
    
    # Find the first line starting with '# ' (the paper title)
    for i, line in enumerate(lines):
        if line.strip().startswith('# '):
            title_index = i
            break
    
    # If we found a title line, remove everything before it
    if title_index > 0:
        log_message(f"Removed {title_index} lines of preamble before paper title")
        return '\n'.join(lines[title_index:])
    
    return summary_content  # Return unchanged if no title marker found

def validate_summary(summary_content):
    """
    Perform validation checks on the generated summary.
    
    Checks multiple aspects of the summary against our required format:
    - Title formatting
    - Year presence
    - Author list completeness
    - Bullet/footnote correspondence
    - Quote formatting
    - Required sections (Glossary, Tags)
    
    Args:
        summary_content (str): Summary to validate
        
    Returns:
        None: Results are logged but not returned
    """
    lines = [line.strip() for line in summary_content.split('\n') if line.strip()]
    
    # Check if first line starts with # (title heading)
    start_with_title = lines[0].startswith('# ')
    
    # Check for year in first few lines
    year_found = False
    for line in lines[:5]:
        if re.search(r'\b(19|20)\d{2}\b', line):
            year_found = True
            break
    
    # Check for "et al." in authors line
    author_complete = not any('et al' in line.lower() 
                            for line in lines[:5] if line.startswith('Authors:'))

    # Check bullets match footnotes
    bullet_count = sum(1 for line in lines 
                      if (line.startswith('- ') or line.startswith('  - '))
                      and not line.startswith('[^'))
    footnote_count = sum(1 for line in lines if line.startswith('[^'))
    
    # Check for proper quote formatting
    footnote_lines = [line for line in lines if line.startswith('[^')]
    properly_quoted = sum(1 for line in footnote_lines 
                         if '"' in line and line.count('"') >= 2)
    
    # Check for Glossary section
    has_glossary = any(line.startswith('## Glossary') for line in lines)
    
    # Check for Tags section structure
    has_tags = False
    has_two_tag_lines = False
    for i, line in enumerate(lines):
        if line.startswith('## Tags'):
            has_tags = True
            # Check next two non-empty lines start with #
            tag_lines = [l for l in lines[i+1:i+4] if l.strip()][:2]
            has_two_tag_lines = (len(tag_lines) == 2 and 
                               all(l.strip().startswith('#') for l in tag_lines))
            break
    
    # Log all validation results
    log_message(f"Validation results:")
    log_message(f"- First line starts with #: {start_with_title}")
    log_message(f"- Year found: {year_found}")
    log_message(f"- Complete author list: {author_complete}")
    log_message(f"- Bullet points: {bullet_count}")
    log_message(f"- Footnotes: {footnote_count}")
    log_message(f"- Properly quoted footnotes: {properly_quoted}")
    log_message(f"- Has Glossary section: {has_glossary}")
    log_message(f"- Has Tags section: {has_tags}")
    log_message(f"- Has proper tag format: {has_two_tag_lines}")

    # Log warnings for validation failures
    if not start_with_title:
        log_message(f"WARNING: Summary does not start with title heading")
    if not year_found:
        log_message("WARNING: No year found in summary")
    if not author_complete:
        log_message("WARNING: Author list appears to be truncated with 'et al.'")
    if bullet_count != footnote_count:
        log_message(f"WARNING: Mismatch between bullets ({bullet_count}) and footnotes ({footnote_count})")
    if properly_quoted != footnote_count:
        log_message(f"WARNING: Some footnotes may not be properly quoted ({properly_quoted}/{footnote_count})")
    if not has_glossary:
        log_message("WARNING: Missing Glossary section")
    if not has_tags or not has_two_tag_lines:
        log_message("WARNING: Missing or improperly formatted Tags section")

def move_to_done(file_path, summary_content):
    """
    Move processed file to the done directory with metadata-based name.
    
    After successful processing, moves the input file to the processed directory
    with a standardized filename based on extracted metadata.
    
    Args:
        file_path (Path): Path to the original input file
        summary_content (str): Generated summary text
        
    Returns:
        Path or None: Path to moved file, or None if move failed
    """
    try:
        DONE_DIR.mkdir(exist_ok=True)
        
        title, authors, year = extract_metadata(summary_content)
        new_filename = create_summary_filename(title, authors, year)
        
        if not new_filename:
            new_filename = file_path.name
            log_message(f"Using original filename for done dir: {new_filename}")
        else:
            new_filename = new_filename.replace('.md', file_path.suffix)
        
        dest_path = DONE_DIR / new_filename
        file_path.rename(dest_path)
        log_message(f"Moved {file_path.name} to done directory as {new_filename}")
        return dest_path  # Return the full path instead of just filename
        
    except Exception as e:
        log_message(f"Warning: Could not move file to done directory: {str(e)}")
        return None
    
def extract_metadata(summary_content):
    """
    Extract author, year, and title from summary content.
    
    Parses the summary text to extract key metadata for filename generation.
    
    Args:
        summary_content (str): Generated summary text
        
    Returns:
        tuple: (title, authors, year)
            - title (str): Paper title
            - authors (list): List of author surnames
            - year (str or None): Publication year
    """
    lines = [l.strip() for l in summary_content.split('\n') if l.strip()]
    title = lines[0].replace('# ', '', 1) if lines[0].startswith('# ') else ''
    year = None
    authors = []
    
    for line in lines[:5]:
        if line.startswith('Authors: '):
            author_line = line.replace('Authors: ', '')
            for part in author_line.split(','):
                part = part.strip()
                if '.' in part:  # Look for initials which indicate an author
                    surname = part.split()[0].strip()
                    if surname:
                        authors.append(surname)
        
        # Extract year from publication line
        if year_match := re.search(r'\b(19|20)\d{2}\b', line):
            year = year_match.group()
    
    return title, authors, year

def format_authors(authors):
    """
    Format author list for filename according to specified rules.
    
    Args:
        authors (list): List of author surnames
        
    Returns:
        str: Formatted author string (e.g., "Smith", "Smith and Jones", "Smith et al.")
    """
    if not authors:
        return "Unknown"
    
    if len(authors) == 1:
        return authors[0]
    elif len(authors) == 2:
        return f"{authors[0]} and {authors[1]}"
    else:
        return f"{authors[0]} et al."

def sanitize_filename(filename):
    """
    Replace filesystem-unfriendly characters and handle length limits.
    
    Makes filenames safe for all filesystems by:
    - Removing unsafe characters
    - Limiting filename length to 255 bytes
    
    Args:
        filename (str): Raw filename to sanitize
        
    Returns:
        str: Sanitized filename safe for all filesystems
    """
    # Replace filesystem-unfriendly characters with safe alternatives
    unsafe_chars = r'<>:"/\\|?*'
    for char in unsafe_chars:
        filename = filename.replace(char, '')

    # Handle maximum length (255 bytes for most filesystems)
    max_bytes = 255
    while len(filename.encode('utf-8')) > max_bytes:
        if '.' in filename:
            name, ext = filename.rsplit('.', 1)
            filename = name[:-1] + '.' + ext
        else:
            filename = filename[:-1]
            
    return filename.strip()

def create_summary_filename(title, authors, year):
    """
    Create a standardized filename from metadata.
    
    Args:
        title (str): Paper title
        authors (list): List of author surnames
        year (str or None): Publication year
        
    Returns:
        str or None: Formatted filename, or None if year is missing
    """
    if not year:
        log_message("WARNING: No year found in summary")
        return None
    return sanitize_filename(f"{format_authors(authors)} - {year} - {title[:100].strip()}.md")

def get_filename_with_fallback(summary, input_path):
    """
    Create filename from metadata with fallback to original name.
    
    Args:
        summary (str): Generated summary text
        input_path (Path): Original input file path
        
    Returns:
        str: Best available filename for the summary
    """
    title, authors, year = extract_metadata(summary)
    filename = create_summary_filename(title, authors, year)
    
    if not filename:
        filename = f"{input_path.stem}.md"
        log_message(f"Using fallback filename: {filename}")
        
    return filename

def save_summary(summary, input_path):
    """
    Save the generated summary to a file.
    
    Args:
        summary (str): Generated summary text
        input_path (Path): Path to the original input file
    """
    filename = get_filename_with_fallback(summary, input_path)
    output_path = OUTPUT_DIR / filename
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(output_path, 'w') as f:
            f.write(summary)
        log_message(f"Summary saved to: {output_path}")
    except Exception as e:
        log_message(f"Error saving summary: {str(e)}")

# ===== MAIN EXECUTION =====

def main():
    """
    Main entry point for the paper summarization process.
    
    Initializes the system, sets up directories, and continuously monitors
    the input directory for new files to process.
    """
    try:
        # Setup environment
        setup_logging()
        check_environment()
        
        # Log startup information
        model_info = f" with model {LLM_MODEL}" if LLM_MODEL else ""
        logging.info(f"Starting paper summarisation using {LLM_PROVIDER}{model_info}")
        
        # Create required directories
        INPUT_DIR.mkdir(exist_ok=True)
        OUTPUT_DIR.mkdir(exist_ok=True)
        DONE_DIR.mkdir(exist_ok=True)
        KNOWLEDGE_DIR.mkdir(exist_ok=True)
        
        # Load project knowledge and tracking data
        keywords, template = read_project_knowledge()
        progress = load_progress()
        failed_files = load_failed_files()
        
        if failed_files:
            log_message(f"Loaded {len(failed_files)} permanently failed files that will be skipped")
        
        logging.info(f"Monitoring directory: {INPUT_DIR.absolute()}")
        
        waiting_message_shown = False
        
        # Main processing loop
        while True:
            try:
                pending_files = get_pending_files(INPUT_DIR, progress, failed_files)
                
                if pending_files:
                    file_path = pending_files[0]  # Process one file at a time
                    waiting_message_shown = False
                    logging.info(f"Processing file: {file_path.name}")
                    
                    # Process the file
                    success, filename, error = process_file(
                        file_path, keywords, template
                    )
                    
                    if success:
                        progress.add(filename)
                        save_progress(progress)
                    else:
                        logging.error(f"Failed to process {filename}: {error}")
                        
                        # After 3 failed attempts, add to failed.log regardless of error type
                        log_message(f"File {filename} failed after 3 attempts - marking as permanently failed")
                        add_to_failed_files(filename, error)
                        failed_files.add(filename)
                    
                    logging.info("Pausing before next file...")
                    time.sleep(10)
                
                elif not waiting_message_shown:
                    # Show waiting message only once when queue becomes empty
                    time.sleep(10)
                    logging.info("No files in queue. Waiting for new files...\n")
                    waiting_message_shown = True
                else:
                    # Short sleep when idle to avoid CPU usage
                    time.sleep(2)
                    
            except KeyboardInterrupt:
                logging.info("\nShutting down gracefully...")
                break
            except Exception as e:
                logging.error(f"Error in main loop: {str(e)}")
                time.sleep(5)
                    
    except Exception as e:
        logging.critical(f"Critical error in main process: {str(e)}")
        raise

if __name__ == "__main__":
    main()