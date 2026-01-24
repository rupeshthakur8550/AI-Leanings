"""
Task 2: Multi-Model Support - One Interface, Many Providers!
Test OpenAI, Google, and X.AI models using the same LangChain interface.

Learning Goal: Experience provider flexibility without code changes.
"""

import os
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from openai import APITimeoutError

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core import settings

def main():

    print("🎯 Task 2: Multi-Model Support with LangChain")
    print("=" * 50)

    print("\n🌐 Initialize Multiple AI Providers")
    print("=" * 50)

    # TODO 1: Initialize OpenAI model
    print("Setting up OpenAI GPT-4.1-mini...")

    openai_llm = ChatOpenAI(
        model = settings.OPENAI_MODEL,
        api_key = settings.OPENAI_API_KEY,
        base_url = settings.OPENAI_API_BASE
    )

    print("Setting up Google GEMINI-2.5-flash")
    google_llm = ChatGoogleGenerativeAI(
        model = settings.GOOGLE_MODEL,
        api_key = settings.GEMINI_API_KEY,
    )

    # Compare all models with the same prompt
    print("\n✅ All models initialized! Now let's compare them...")
    print("\nModel Comparison - Same Prompt, Different Models")
    print("=" * 50)

    test_prompt = "Explain cloud computing in one sentence"
    print(f"📝 Prompt: '{test_prompt}'\n")

    if openai_llm:
        try:
            response = openai_llm.invoke(test_prompt)
            print(f"\nOpen AI Response: {response.content}")
        except APITimeoutError:
            print("❌ OpenAI request timed out")
            return None

    if google_llm:
        try:
            response = google_llm.invoke(test_prompt)
            print(f"\nGoogle LLM Response: {response.content}")
        except APITimeoutError:
            print("❌ Google LLM request timed out")
            return None
    
    print("\n💡 Same code, different providers - perfect for A/B testing!")
    print("\n✅ Task 2 completed! You can now switch models at will!")
    print("🎉 You tested 2 different AI providers with identical code!")


if __name__ == "__main__":
    main()