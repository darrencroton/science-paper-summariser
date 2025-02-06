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