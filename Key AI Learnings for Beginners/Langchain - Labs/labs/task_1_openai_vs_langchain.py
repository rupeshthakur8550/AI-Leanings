"""
Task 1: OpenAI SDK vs LangChain - See the Difference!
Compare the complexity of raw OpenAI SDK with LangChain's clean abstraction.

Learning Goal: Understand why LangChain simplifies AI development.
"""

import sys
import os
import openai
from openai import APITimeoutError
from langchain_openai import ChatOpenAI

# Add parent directory to sys.path to allow importing from core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core import settings

def raw_openai_approach():
    """Raw OpenAI SDK - complex and verbose"""
    print("\nRAW OPENAI SDK APPROACH")

    client = openai.OpenAI(
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_API_BASE,
        timeout=30.0,   # 👈 important
        max_retries=2   # 👈 important
    )

    try:
        response = client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "user", "content": "Explain machine learning in one sentence"}
            ],
        )

        text = response.choices[0].message.content
        print(f"Response: {text}")
        return text

    except APITimeoutError:
        print("❌ OpenAI request timed out")
        return None


def langchain_approach():
    """LangChain - clean and simple"""
    print("\nLANGCHAIN APPROACH")

    try: 
        llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_API_BASE,
            timeout=30,
            max_retries=2,
        )

        response = llm.invoke("Explain machine learning in one sentence")
        """ 
            Raw Response: content='Machine learning is a field of artificial intelligence that enables computers to learn patterns from data and make predictions or decisions without being explicitly programmed.' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 26, 'prompt_tokens': 13, 'total_tokens': 39, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_provider': 'openai', 'model_name': 'gpt-4.1-mini-2025-04-14', 'system_fingerprint': 'fp_376a7ccef1', 'id': 'chatcmpl-CzalB2cvgAB90haRhMBm7iMmhJTE2', 'service_tier': 'default', 'finish_reason': 'stop', 'logprobs': None} id='lc_run--019bd46b-2205-71b3-aa7a-a7e8ff75be12-0' tool_calls=[] invalid_tool_calls=[] usage_metadata={'input_tokens': 13, 'output_tokens': 26, 'total_tokens': 39, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}
        """
        print(f"Response: {response.content}")
        return response.content

    except APITimeoutError:
        print("❌ OpenAI request timed out")
        return None

def main():

    print("🎯 Task 1: OpenAI SDK vs LangChain Comparison")
    print("=" * 50)

    raw_result = raw_openai_approach()
    langchain_result = langchain_approach()

    # Show the difference
    if raw_result and langchain_result:
        print("\n📊 COMPARISON:")
        print("✅ Both approaches work, but LangChain is:")
        print("  - 70% less code")
        print("  - Cleaner response handling")
        print("  - Provider agnostic")

        print("\n✅ Task 1 completed!")


if __name__ == "__main__":
    main()