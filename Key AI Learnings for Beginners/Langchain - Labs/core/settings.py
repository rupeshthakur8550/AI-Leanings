
import os
from dotenv import load_dotenv

# Load environment variables from .env file
# Assuming .env is in the root directory (parent of core)
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_MODEL = os.getenv("GOOGLE_MODEL")
# GROK_MODEL = os.getenv("GROK_MODEL")
