"""
Task 3: Making Your First API Call
Understand EVERY part of the chat completion call.
"""

import openai
import sys
import os

# Add parent directory to sys.path to allow importing from core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core import settings 

client = openai.OpenAI(
    api_key = settings.OPENAI_API_KEY,
    base_url = settings.OPENAI_API_BASE
)

# ==========================================
# UNDERSTANDING THE API CALL STRUCTURE
# ==========================================
#
# To make an API call, you MUST provide:
# 1. model - Which AI model to use (required)
# 2. messages - Your conversation with the AI (required)
#
# The messages parameter is a list of dictionaries, each with:
# - role: Who is speaking ("user", "assistant", or "system")
# - content: What they are saying
# ==========================================

# TODO: Read each line below carefully to understand what it does

try:
    response = client.chat.completions.create(
        model =  settings.OPENAI_MODEL,
        messages = [
            {
                "role": "user",
                "content": "Hello AI, please introduce yourself...!"
            }
        ],
        timeout=30.0
    )
    api_error = None
except Exception as e:
    print(f"Error during API call: {e}")
    response = None
    api_error = e

# ==========================================
# REAL RESPONSE OBJECT STRUCTURE
# This is an ACTUAL response from OpenAI:
# ==========================================
"""
ChatCompletion(
    id='chatcmpl-Cz1cUYvnkifFyZz5eV0hTUFnKIySq', 
    choices=[
        Choice(finish_reason='stop', 
        index=0, 
        logprobs=None, 
        message=ChatCompletionMessage(
            content="Hello! I'm ChatGPT, an AI language model developed by OpenAI. I'm here to help with a wide range of tasks, such as answering questions, providing explanations, brainstorming ideas, writing and editing text, and much more. How can I assist you today?", 
            refusal=None, 
            role='assistant', 
            annotations=[], 
            audio=None, 
            function_call=None, 
            tool_calls=None)
        )
    ], 
    created=1768660270, 
    model='gpt-4.1-mini-2025-04-14', 
    object='chat.completion', 
    service_tier='default', 
    system_fingerprint='fp_376a7ccef1', 
    usage=CompletionUsage(
        completion_tokens=53, 
        prompt_tokens=15, 
        total_tokens=68, 
        completion_tokens_details=CompletionTokensDetails(
            accepted_prediction_tokens=0, 
            audio_tokens=0, 
            reasoning_tokens=0, 
            rejected_prediction_tokens=0
        ), 
        prompt_tokens_details=PromptTokensDetails(
            audio_tokens=0, 
            cached_tokens=0
        )
    )
)
"""

# Once you uncomment and run the code above, this will execute:

try:
    if 'response' in locals() and response:
        print(f"\n\nResponse: {response} \n")
        ai_text = response.choices[0].message.content

        print("API call successful!!")
        print(f"AI said: {ai_text}")

        print(f"Total tokens used: {response.usage.total_tokens}")

    else:
        print("API call failed")
        if api_error:
             print(f"Reason: {api_error}")
        else:
             print("Required Parameters missing or invalid")
             print(f"1. model: {settings.OPENAI_MODEL}")
             print("2. messages: [{'role': 'user', 'content': 'your message'}]")

except NameError:
    print("\n Required values:")
    print(f" - model: {settings.OPENAI_MODEL}")
    print(" - messages: [{'role': 'user', 'content': 'your message'}]")
