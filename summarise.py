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

# Load environment variables
load_dotenv()

# Define LLM provider
LLM_PROVIDER = sys.argv[1] if len(sys.argv) > 1 else 'claude'

# Define the provider-specific model (if provided)
LLM_MODEL = sys.argv[2] if len(sys.argv) > 2 else None

# Define absolute paths
SCRIPT_DIR = Path(__file__).parent.absolute()
INPUT_DIR = SCRIPT_DIR / "input"
OUTPUT_DIR = SCRIPT_DIR / "output"
LOGS_DIR = SCRIPT_DIR / "logs"
DONE_DIR = SCRIPT_DIR / "processed"
PROGRESS_FILE = LOGS_DIR / "completed.log"
FAILED_FILE = LOGS_DIR / "failed.log"
KNOWLEDGE_DIR = SCRIPT_DIR / "project_knowledge"

def check_environment():
    """Check required environment variables are set"""
    # Provider-specific checks are now handled in the provider classes
    pass

def setup_logging():
    """Set up system-wide logging"""
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
    """Write a message to system log"""
    logging.info(message)

def load_progress():
    """Load progress of processed files"""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'r') as f:
            return {line.strip() for line in f if line.strip()}
    return set()

def load_failed_files():
    """Load list of permanently failed files"""
    if FAILED_FILE.exists():
        with open(FAILED_FILE, 'r') as f:
            return {line.strip().split('|')[0] for line in f if line.strip()}
    return set()

def save_progress(processed_files):
    """Save progress of processed files"""
    with open(PROGRESS_FILE, 'w') as f:
        for filename in sorted(processed_files):
            f.write(f"{filename}\n")

def add_to_failed_files(filename, error):
    """Add a file to the permanently failed list with timestamp and error"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(FAILED_FILE, 'a') as f:
        f.write(f"{filename}|{timestamp}|{error}\n")
    log_message(f"Added {filename} to failed files list - will be skipped in future processing attempts")

def read_project_knowledge():
    """Read all project knowledge files"""
    try:
        with open(KNOWLEDGE_DIR / "astronomy-keywords.txt") as f:
            keywords = f.read()
        
        with open(KNOWLEDGE_DIR / "paper-summary-template.md") as f:
            template = f.read()
        
        return keywords, template
    except Exception as e:
        logging.error(f"Failed to read project knowledge files: {str(e)}")
        raise

def read_input_file(file_path: Path):
    """Read content from either PDF or text file with provider-specific handling"""
    try:
        # Check file size (100MB limit)
        file_size = file_path.stat().st_size
        if file_size > 100 * 1024 * 1024:  # 100MB in bytes
            return None, f"File too large: {file_size/1024/1024:.1f}MB (max 100MB)"

        if file_path.suffix.lower() == '.pdf':
            # For providers that need text extraction (OpenAI, Perplexity)
            if LLM_PROVIDER in ["openai", "perplexity"]:
                try:
                    # Use Marker to extract text from PDF
                    config = {
                        "output_format": "markdown",
                        "disable_image_extraction": True,
                        "use_llm": False
                    }
                    config_parser = ConfigParser(config)
                    
                    # Initialize converter
                    converter = PdfConverter(
                        config=config_parser.generate_config_dict(),
                        artifact_dict=create_model_dict(),
                        processor_list=config_parser.get_processors(),
                        renderer=config_parser.get_renderer()
                    )
                    
                    # Convert the PDF to text
                    rendered = converter(str(file_path))
                    text, _, _ = text_from_rendered(rendered)
                    log_message(f"Extracted {len(text.split())} words from PDF for {LLM_PROVIDER}")
                    return text, None
                    
                except Exception as e:
                    return None, f"PDF processing error: {str(e)}"
            
            elif LLM_PROVIDER == "ollama":
                try:
                    # Configure using ConfigParser for Marker
                    ollama_model = LLM_MODEL or "mistral:7b"
                    config = {
                        "output_format": "markdown",
                        "disable_image_extraction": True,
                        "use_llm": False,
                        "llm_service": "marker.services.ollama.OllamaService",
                        "ollama_base_url": "http://localhost:11434",
                        "ollama_model": ollama_model
                    }
                    config_parser = ConfigParser(config)
                    
                    # Initialize converter with parsed config
                    converter = PdfConverter(
                        config=config_parser.generate_config_dict(),
                        artifact_dict=create_model_dict(),
                        processor_list=config_parser.get_processors(),
                        renderer=config_parser.get_renderer(),
                        llm_service=config_parser.get_llm_service()
                    )
                    
                    # Convert the PDF
                    rendered = converter(str(file_path))
                    text, _, _ = text_from_rendered(rendered)
                    return text, None
                    
                except Exception as e:
                    return None, f"PDF processing error: {str(e)}"
            else:
                # Claude's direct binary reading approach for providers that support PDFs directly
                with open(file_path, 'rb') as f:
                    return f.read(), None
        else:  # For .txt files
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read(), None
                
    except Exception as e:
        return None, f"File read error: {str(e)}"

# This function is no longer needed as each provider handles token limits
# def get_max_tokens(paper_text):
#     """Calculate appropriate max_tokens based on input length and provider"""
#     ...
    
def create_system_prompt(keywords):
    """Create the system prompt defining role and expertise"""
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
    """Create the user prompt with specific task instructions"""
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
    
    # Include paper text in the prompt for txt files or PDFs that need text extraction
    # This includes Ollama, OpenAI, and Perplexity for PDFs
    if not is_pdf or LLM_PROVIDER in ["ollama", "openai", "perplexity"]:
        base_prompt += (
            "<input>\n"
            f"Paper to summarize:\n\n"
            f"---BEGIN PAPER---\n{paper_text}\n---END PAPER---\n"
            "</input>"
        )
    
    return base_prompt

def process_file(file_path, keywords, template):
    """Process a file using the configured LLM provider"""
    # Determine file type
    is_pdf = file_path.suffix.lower() == '.pdf'
    
    # Check if we need to extract text from PDF
    needs_text_extraction = LLM_PROVIDER in ["openai", "perplexity"] and is_pdf
    
    # Read the file content
    paper_content, error = read_input_file(file_path)
    
    # Log the content type to diagnose issues
    if is_pdf:
        log_message(f"PDF content type: {type(paper_content).__name__}")
    if error:
        return False, file_path.name, error
    
    # Initialize the LLM provider
    provider_config = {"model": LLM_MODEL} if LLM_MODEL else {}
    llm_provider = create_llm_provider(LLM_PROVIDER, config=provider_config)
    
    # Create appropriate prompts
    system_prompt = create_system_prompt(keywords)
    
    # For OpenAI and Perplexity, we need to include the extracted text in the prompt
    # since they don't have native PDF handling
    include_text_in_prompt = not is_pdf or (is_pdf and (LLM_PROVIDER in ["ollama", "openai", "perplexity"]))
    
    # If we're including text in the prompt and the content is binary, there's a problem
    if include_text_in_prompt and is_pdf and isinstance(paper_content, bytes) and LLM_PROVIDER in ["openai", "perplexity"]:
        log_message("Warning: PDF text not properly extracted for prompt inclusion")
    
    user_prompt = create_user_prompt(
        paper_text=paper_content if include_text_in_prompt else "",
        template=template,
        is_pdf=is_pdf
    )
    
    if is_pdf and LLM_PROVIDER in ["openai", "perplexity"]:
        log_message(f"PDF text extraction for {LLM_PROVIDER}: text length is {len(str(paper_content)) if paper_content else 0} characters")
    
    # Write complete prompt to file for debugging
    full_prompt = f"SYSTEM PROMPT\n{system_prompt}\n\n---\n\nUSER PROMPT\n{user_prompt}"
    prompt_file = LOGS_DIR / "prompt.txt"
    with open(prompt_file, 'w') as f:
        f.write(full_prompt)
    log_message(f"Full prompt written to {prompt_file}")

    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Use the provider to process the document
            # We use a lower max_tokens value for response generation (not context window)
            # This is the maximum number of tokens the model will generate in response
            summary_content = llm_provider.process_document(
                content=paper_content,
                is_pdf=is_pdf,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=8192  # This is appropriate for summary generation across models
            )
            
            # Common post-processing
            validate_summary(summary_content)
            save_summary(summary_content, file_path)
            
            moved_path = move_to_done(file_path, summary_content)
            if moved_path:
                log_message(f"Completed processing: {moved_path.name}\n")
                return True, moved_path.name, None
            else:
                return False, file_path.name, "Failed to move file"
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Attempt {attempt + 1} failed - API request error: {str(e)}"
            log_message(error_msg)
        except ValueError as e:
            error_msg = f"Attempt {attempt + 1} failed - Value error: {str(e)}"
            log_message(error_msg)
        except ImportError as e:
            error_msg = f"Attempt {attempt + 1} failed - Missing dependency: {str(e)}"
            log_message(error_msg)
        except Exception as e:
            error_msg = f"Attempt {attempt + 1} failed - {LLM_PROVIDER} error: {e.__class__.__name__}: {str(e)}"
            log_message(error_msg)
            
        if attempt < max_retries - 1:
            wait_time = 2 ** attempt
            log_message(f"Waiting {wait_time} seconds before retry...")
            time.sleep(wait_time)
        else:
            return False, file_path.name, error_msg
    
def validate_summary(summary_content):
    """Basic validation of summary format and log results"""
    lines = [line.strip() for line in summary_content.split('\n') if line.strip()]
    
    # Check if first line starts with # 
    start_with_title = lines[0].startswith('# ')
    
    # Check for year in first few lines
    year_found = False
    for line in lines[:5]:
        if re.search(r'\b(19|20)\d{2}\b', line):
            year_found = True
            break
    
    # Check for et al in authors
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
    """Move processed file to done directory with metadata-based name"""
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
    """Extract author, year, and title from summary content"""
    lines = [l.strip() for l in summary_content.split('\n') if l.strip()]
    title = lines[0].replace('# ', '', 1) if lines[0].startswith('# ') else ''
    year = None
    authors = []
    
    for line in lines[:5]:
        if line.startswith('Authors: '):
            author_line = line.replace('Authors: ', '')
            for part in author_line.split(','):
                part = part.strip()
                if '.' in part:
                    surname = part.split()[0].strip()
                    if surname:
                        authors.append(surname)
        
        if year_match := re.search(r'\b(19|20)\d{2}\b', line):
            year = year_match.group()
    
    return title, authors, year

def format_authors(authors):
    """Format author list according to specified rules"""
    if not authors:
        return "Unknown"
    
    if len(authors) == 1:
        return authors[0]
    elif len(authors) == 2:
        return f"{authors[0]} and {authors[1]}"
    else:
        return f"{authors[0]} et al."

def sanitize_filename(filename):
    """Replace filesystem-unfriendly characters and handle length limits"""
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
    """Create a filename from metadata"""
    if not year:
        log_message("WARNING: No year found in summary")
        return None
    return sanitize_filename(f"{format_authors(authors)} - {year} - {title[:100].strip()}.md")

def get_filename_with_fallback(summary, input_path):
    """Create filename from metadata with fallback to original"""
    title, authors, year = extract_metadata(summary)
    filename = create_summary_filename(title, authors, year)
    
    if not filename:
        filename = f"{input_path.stem}.md"
        log_message(f"Using fallback filename: {filename}")
        
    return filename

def save_summary(summary, input_path):
    """Save the summary to a file"""
    filename = get_filename_with_fallback(summary, input_path)
    output_path = OUTPUT_DIR / filename
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(output_path, 'w') as f:
            f.write(summary)
        log_message(f"Summary saved to: {output_path}")
    except Exception as e:
        log_message(f"Error saving summary: {str(e)}")

def get_pending_files(input_dir, progress, failed):
    """Get list of unprocessed files, excluding permanently failed files"""
    return [
        f for f in input_dir.glob("*.*") 
        if f.suffix.lower() in ['.pdf', '.txt'] 
        and f.name not in progress
        and f.name not in failed
    ]

def main():
        
    try:
        
        setup_logging()
        check_environment()
        
        model_info = f" with model {LLM_MODEL}" if LLM_MODEL else ""
        logging.info(f"Starting paper summarisation using {LLM_PROVIDER}{model_info}")
        
        INPUT_DIR.mkdir(exist_ok=True)
        OUTPUT_DIR.mkdir(exist_ok=True)
        DONE_DIR.mkdir(exist_ok=True)
        KNOWLEDGE_DIR.mkdir(exist_ok=True)
        
        keywords, template = read_project_knowledge()
        progress = load_progress()
        failed_files = load_failed_files()
        
        if failed_files:
            log_message(f"Loaded {len(failed_files)} permanently failed files that will be skipped")
        
        logging.info(f"Monitoring directory: {INPUT_DIR.absolute()}")
        
        waiting_message_shown = False
        
        while True:
            try:
                pending_files = get_pending_files(INPUT_DIR, progress, failed_files)
                
                if pending_files:
                    file_path = pending_files[0]  # Process one file at a time
                    waiting_message_shown = False
                    logging.info(f"Processing file: {file_path.name}")
                    
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
                    time.sleep(10)
                    logging.info("No files in queue. Waiting for new files...\n")
                    waiting_message_shown = True
                else:
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