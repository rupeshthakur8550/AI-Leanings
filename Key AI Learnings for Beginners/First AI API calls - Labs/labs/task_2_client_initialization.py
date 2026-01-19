"""
Task 2: Initialize the OpenAI Client
Learn how to connect to OpenAI's servers.
"""

import sys
import openai
import os 

# Add parent directory to sys.path to allow importing from core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core import settings

# The OpenAI client needs two things:
# 1. API Key - Your authentication (like a password)
# 2. Base URL - Where to send requests (like an address)

client = openai.OpenAI(
    api_key = settings.OPENAI_API_KEY,
    base_url = settings.OPENAI_API_BASE
)

print("Step 2 Complete: Connected to OpenAI!")
print(f" - API Key: {settings.OPENAI_API_KEY[:10]}...")
print(f" - Base URL: {os.getenv("OPENAI_API_BASE")}")