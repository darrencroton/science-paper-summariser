import os
import re
import anthropic
import time
import logging
from datetime import datetime
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from pathlib import Path

# Load environment variables
load_dotenv()

# Define absolute paths
SCRIPT_DIR = Path(__file__).parent.absolute()
INPUT_DIR = SCRIPT_DIR / "input"
OUTPUT_DIR = SCRIPT_DIR / "output"
LOGS_DIR = SCRIPT_DIR / "logs"
DONE_DIR = SCRIPT_DIR / "processed"
PROGRESS_FILE = LOGS_DIR / "completed.log"
KNOWLEDGE_DIR = SCRIPT_DIR / "project_knowledge"

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
        
        with open(KNOWLEDGE_DIR / "prompt.txt") as f:
            prompt_template = f.read()
        
        return keywords, template, prompt_template
    except Exception as e:
        logging.error(f"Failed to read project knowledge files: {str(e)}")
        raise

def read_input_file(file_path):
    """Read content from either PDF or text file with better error handling"""
    try:
        if file_path.suffix.lower() == '.pdf':
            try:
                reader = PdfReader(file_path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                return text, None
            except Exception as e:
                return None, f"PDF processing error: {str(e)}"
        else:  # For .txt files
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read(), None
    except Exception as e:
        return None, f"File read error: {str(e)}"

def validate_input_length(paper_text):
    """Validate input length and log warnings"""
    token_estimate = len(paper_text.split())
    if token_estimate > 65536:
        log_message(f"WARNING: Paper length ({token_estimate} tokens) may be too long for effective summarization")
    return token_estimate

def get_max_tokens(paper_text):
    """Calculate appropriate max_tokens based on input length"""
    estimated_tokens = len(paper_text.split()) * 0.8 + 1000  # 1000 token overhead for formatting
    return min(65536, int(estimated_tokens))  # Cap at Sonnet's maximum

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
    filename = get_filename_with_fallback(summary, input_path)
    output_path = OUTPUT_DIR / filename
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(output_path, 'w') as f:
            f.write(summary)
        log_message(f"Summary saved to: {output_path}")
    except Exception as e:
        log_message(f"Error saving summary: {str(e)}")

def create_system_prompt(keywords, prompt_template):
    """Create the system prompt for consistent formatting"""
    return (
        "You're an esteemed professor of astrophysics at Harvard University. "
        "You will summarize research papers with these STRICT requirements:\n\n"
        "1. THE VERY FIRST LINE must be the paper title as a level 1 heading\n"
        "2. NO TEXT whatsoever before the title - not even a greeting or introduction\n"
        "3. Write in UK English using clear technical language\n"
        "4. Use markdown formatting\n"
        "5. Use latex for mathematical expressions\n"
        "6. Only include content from the provided paper\n"
        "7. Follow the exact section order and formatting specified\n"
        "8. Every bullet point must have a supporting footnote containing a verbatim quote\n"
        "9. IMPORTANT: Footnotes must contain EXACT quotes from the paper - never paraphrase\n"
        "10. Always enclose footnote quotes in quotation marks and include section/page\n"
        "11. If you cannot find an exact supporting quote for a statement, do not make the statement\n"
        "12. ALWAYS include a Glossary section with a table of technical terms\n"
        "13. Include EVERY author in the author list - never truncate with 'et al.'\n"
        "14. MUST include the paper's publication month and year in the first few lines\n\n"
        f"Available astronomy keywords:\n{keywords}\n\n"
        f"Additional instructions:\n{prompt_template}"
    )

def create_user_prompt(paper_text, template):
    """Create the user prompt with the paper and template"""
    return (
        f"Summarize this paper following these EXACT requirements:\n\n"
        f"1. THE VERY FIRST LINE of your response must be the paper title as '# Title'\n"
        f"2. Do not include ANY text before the title - no introduction, no explanation\n"
        f"3. For every bullet point, you MUST provide a supporting footnote with an exact, "
        f"verbatim quote from the paper\n"
        f"4. Never paraphrase quotes\n"
        f"5. If you cannot find an exact quote to support a statement, do not make that statement\n"
        f"6. Include EVERY author in the author list\n"
        f"7. MUST include the paper's publication month and year\n\n"
        f"Template to follow:\n\n"
        f"{template}\n\n"
        f"Paper to summarize:\n\n"
        f"---BEGIN PAPER---\n{paper_text}\n---END PAPER---"
    )

def validate_summary(summary_content):
    """Basic validation of summary format and log results"""
    lines = [line.strip() for line in summary_content.split('\n') if line.strip()]
    
    # Check if first line starts with # 
    if not lines[0].startswith('#'):
        log_message(f"WARNING: Summary does not start with title heading")
    
    # Check for year in first few lines
    year_found = False
    for line in lines[:5]:
        if re.search(r'\b(19|20)\d{2}\b', line):
            year_found = True
            break
    if not year_found:
        log_message("WARNING: No year found in summary")
    
    # Check for et al in authors
    author_complete = not any('et al' in line.lower() 
                            for line in lines[:5] if line.startswith('Authors:'))
    if not author_complete:
        log_message("WARNING: Author list appears to be truncated with 'et al.'")

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
    
    # Log all validation results
    log_message(f"Validation results:")
    log_message(f"- First line starts with #: {lines[0].startswith('#')}")
    log_message(f"- Year found: {year_found}")
    log_message(f"- Complete author list: {author_complete}")
    log_message(f"- Bullet points: {bullet_count}")
    log_message(f"- Footnotes: {footnote_count}")
    log_message(f"- Properly quoted footnotes: {properly_quoted}")
    log_message(f"- Has Glossary section: {has_glossary}")
    
    if bullet_count != footnote_count:
        log_message(f"WARNING: Mismatch between bullets ({bullet_count}) and footnotes ({footnote_count})")
    if properly_quoted != footnote_count:
        log_message(f"WARNING: Some footnotes may not be properly quoted ({properly_quoted}/{footnote_count})")
    if not has_glossary:
        log_message("WARNING: Missing Glossary section")

def process_file(file_path, keywords, template, prompt_template):
    paper_text, error = read_input_file(file_path)
    if error:
        return False, file_path.name, error
    
    validate_input_length(paper_text)
    system_prompt = create_system_prompt(keywords, prompt_template)
    user_prompt = create_user_prompt(paper_text, template)
    max_tokens = get_max_tokens(paper_text)
    
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            message = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=max_tokens,
                temperature=0.3,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            
            summary_content = message.content[0].text
            validate_summary(summary_content)
            save_summary(summary_content, file_path)
            
            moved_path = move_to_done(file_path, summary_content)
            if moved_path:
                log_message(f"Completed processing: {moved_path.name}\n")
                return True, moved_path.name, None
            else:
                return False, file_path.name, "Failed to move file"
            
        except Exception as e:
            error_msg = f"Attempt {attempt + 1} failed: {e.__class__.__name__}: {str(e)}"
            log_message(error_msg)
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                log_message(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                return False, file_path.name, error_msg

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
        logging.info("Starting paper summarisation")
        
        INPUT_DIR.mkdir(exist_ok=True)
        OUTPUT_DIR.mkdir(exist_ok=True)
        DONE_DIR.mkdir(exist_ok=True)
        KNOWLEDGE_DIR.mkdir(exist_ok=True)
        
        keywords, template, prompt_template = read_project_knowledge()
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
                        file_path, keywords, template, prompt_template
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
