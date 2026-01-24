"""
Task 3: Prompt Templates - Dynamic, Reusable Prompts
Show how ONE template can be reused with different variables.

Learning Goal: Master prompt templates for consistent, reusable prompts.
"""

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core import settings

def main():

    print("🎯 Task 3: Dynamic Prompt Templates")
    print("=" * 50)

    print("\n📝 Creating a Reusable Template")
    print("=" * 50)

    # TODO 1: Create a versatile template
    template = PromptTemplate(
        input_variables = ["topic", "style"],
        template = "Explain {topic} in {style}"
    )

     # Test with actual LLM to show it works
    print("\n🤖 Testing Template with AI")
    print("=" * 50)

    openai_llm = ChatOpenAI(
        model = settings.OPENAI_MODEL,
        api_key = settings.OPENAI_API_KEY,
        base_url = settings.OPENAI_API_BASE,
        temperature = 0.8
    )

    # TODO 2: Use the template with LLM
    if template and openai_llm:
        # Format the template with specific values
        test_prompt = template.format(
            topic = "AI Engineering and LLM calls",
            style = "Exactly in 5 sentences"
        )

        print(f"\nPrompt: {test_prompt}")

        response = openai_llm.invoke(test_prompt)

        print(f"\nResponse: {response.content}")

        # Show the benefits
        print("\n💡 Template Benefits:")
        print("  ✓ ONE template, INFINITE uses")
        print("  ✓ Variables make it dynamic")
        print("  ✓ Consistent structure across all prompts")
        print("  ✓ Change inputs, not code!")

        print("\n✅ Task 3 completed! One template, endless possibilities!")

if __name__ == "__main__":
    main()
