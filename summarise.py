import os
import sys
import re
import time
import logging
import requests
import json
import anthropic
import base64
from dotenv import load_dotenv
from pathlib import Path
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from marker.config.parser import ConfigParser

# Load environment variables
load_dotenv()

# Define LLM provider (either "claude" or "ollama")
LLM_PROVIDER = sys.argv[1] if len(sys.argv) > 1 else 'claude'

# Define the basse Ollama model to use
OLLAMA_MODEL = (sys.argv[2] if len(sys.argv) > 2 
                else "mistral:7b" if len(sys.argv) > 1 
                else "")

# Define absolute paths
SCRIPT_DIR = Path(__file__).parent.absolute()
INPUT_DIR = SCRIPT_DIR / "input"
OUTPUT_DIR = SCRIPT_DIR / "output"
LOGS_DIR = SCRIPT_DIR / "logs"
DONE_DIR = SCRIPT_DIR / "processed"
PROGRESS_FILE = LOGS_DIR / "completed.log"
KNOWLEDGE_DIR = SCRIPT_DIR / "project_knowledge"

def check_environment():
    """Check required environment variables are set"""
    if LLM_PROVIDER == "claude":
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")

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

def save_progress(processed_files):
    """Save progress of processed files"""
    with open(PROGRESS_FILE, 'w') as f:
        for filename in sorted(processed_files):
            f.write(f"{filename}\n")

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
            if LLM_PROVIDER == "ollama":
                try:
                    # Configure using ConfigParser for Marker
                    config = {
                        "output_format": "markdown",
                        "disable_image_extraction": True,
                        "use_llm": False,
                        "llm_service": "marker.services.ollama.OllamaService",
                        "ollama_base_url": "http://localhost:11434",
                        "ollama_model": OLLAMA_MODEL
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
                # Claude's direct binary reading approach
                with open(file_path, 'rb') as f:
                    return f.read(), None
        else:  # For .txt files
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read(), None
                
    except Exception as e:
        return None, f"File read error: {str(e)}"

def get_max_tokens(paper_text):
    """Calculate appropriate max_tokens based on input length and provider"""
    if LLM_PROVIDER == "ollama":
        estimated_tokens = len(paper_text.split()) * 0.75 + 2000  # 1000 token overhead
        if estimated_tokens > 24576:
            log_message(f"WARNING: Paper length ({estimated_tokens} tokens) may be too long for effective summarization")
        return min(24576, int(estimated_tokens))
    else:
        return 200000  # Claude's maximum tokens
    
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
    
    if not is_pdf or LLM_PROVIDER == "ollama":
        base_prompt += (
            "<input>\n"
            f"Paper to summarize:\n\n"
            f"---BEGIN PAPER---\n{paper_text}\n---END PAPER---\n"
            "</input>"
        )
    
    return base_prompt

def process_file(file_path, keywords, template):
    """Process a file using either Claude or Ollama"""
    paper_content, error = read_input_file(file_path)
    if error:
        return False, file_path.name, error
    
    is_pdf = file_path.suffix.lower() == '.pdf'
    
    if LLM_PROVIDER == "ollama":
        max_tokens = get_max_tokens(paper_content)
    
    # Create appropriate prompts
    system_prompt = create_system_prompt(keywords)
    user_prompt = create_user_prompt(
        paper_text=paper_content if not is_pdf or LLM_PROVIDER == "ollama" else "",
        template=template,
        is_pdf=is_pdf
    )
    
    # Write complete prompt to file for debugging
    full_prompt = f"SYSTEM PROMPT\n{system_prompt}\n\n---\n\nUSER PROMPT\n{user_prompt}"
    prompt_file = LOGS_DIR / "prompt.txt"
    with open(prompt_file, 'w') as f:
        f.write(full_prompt)
    log_message(f"Full prompt written to {prompt_file}")

    max_retries = 3
    for attempt in range(max_retries):
        try:
            if LLM_PROVIDER == "ollama":
                
                # Ollama API call
                data = {
                    "model": OLLAMA_MODEL,
                    "prompt": user_prompt,
                    "system": system_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                        "num_ctx": max_tokens,
                        "num_predict": 8192,
                        "stop": ["</input>", "</task>"]
                    }
                }

                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json=data
                )
                response.raise_for_status()
                                
                response_data = response.json()
                summary_content = response_data.get('response', '')

                if 'error' in response_data:
                    raise ValueError(f"Ollama error: {response_data['error']}")
                    
                if not summary_content:
                    raise ValueError("No content received from Ollama")

            else:
                # Claude API call
                client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
                
                messages = [{
                    "role": "user",
                    "content": [{"type": "text", "text": user_prompt}]
                }]
                
                if is_pdf:
                    messages[0]["content"].append({
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": base64.b64encode(paper_content).decode()
                        }
                    })
                    
                message = client.messages.create(
                    model="claude-3-7-sonnet-20250219",
                    temperature=0.2,
                    max_tokens=8192,
                    system=system_prompt,
                    messages=messages
                )
                
                summary_content = message.content[0].text
            
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
            error_msg = f"Attempt {attempt + 1} failed - Ollama API error: {str(e)}"
            log_message(error_msg)
        except anthropic.APIError as e:
            error_msg = f"Attempt {attempt + 1} failed - Claude API error: {str(e)}"
            log_message(error_msg)
        except Exception as e:
            error_msg = f"Attempt {attempt + 1} failed - Unexpected error: {e.__class__.__name__}: {str(e)}"
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

def get_pending_files(input_dir, progress):
    """Get list of unprocessed files"""
    return [
        f for f in input_dir.glob("*.*") 
        if f.suffix.lower() in ['.pdf', '.txt'] 
        and f.name not in progress
    ]

def main():
    try:
        
        setup_logging()
        check_environment()
        logging.info(f"Starting paper summarisation using {LLM_PROVIDER} {OLLAMA_MODEL}")
        
        INPUT_DIR.mkdir(exist_ok=True)
        OUTPUT_DIR.mkdir(exist_ok=True)
        DONE_DIR.mkdir(exist_ok=True)
        KNOWLEDGE_DIR.mkdir(exist_ok=True)
        
        keywords, template = read_project_knowledge()
        progress = load_progress()
        
        logging.info(f"Monitoring directory: {INPUT_DIR.absolute()}")
        
        waiting_message_shown = False
        
        while True:
            try:
                pending_files = get_pending_files(INPUT_DIR, progress)
                
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