"""
Task 4: Output Parsers - From Text to Structured Data
Transform AI responses into structured formats your application can use.

Learning Goal: Extract structured data from unstructured AI responses.
"""

import os
import sys
import json
from enum import Enum
from typing import List

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import (
    StrOutputParser,
    CommaSeparatedListOutputParser,
    JsonOutputParser,
    PydanticOutputParser,
)

from langchain_classic.output_parsers import (
    StructuredOutputParser,
    ResponseSchema,
    RegexParser,
    EnumOutputParser,
)
from langchain_classic.output_parsers.fix import OutputFixingParser
from langchain_classic.output_parsers.retry import RetryOutputParser

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core import settings

# ----- Example Models & Enums -----
class TechInfo(BaseModel):
    """Information about a technology."""
    name: str = Field(description="Name of the technology")
    year_released: int = Field(description="Year it was first released")
    creator: str = Field(description="Person or organization that created it")
    tags: List[str] = Field(description="List of keywords describing it")


class Difficulty(Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


def main():
    print("🎯 Task 4: Output Parsers Showcase")
    print("=" * 50)

    llm = ChatOpenAI(
        model=settings.OPENAI_MODEL,
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_API_BASE,
        temperature=0.3,
    )

    # --------------------------
    # Parser 1: String Output
    # --------------------------
    print("\n🧵 Parser 1: String Output")
    print("=" * 50)

    str_prompt = PromptTemplate(
        template="Analyze {technology} and provide pros and cons in 2-3 sentences",
        input_variables=["technology"]
    )
    str_parser = StrOutputParser()
    str_chain = str_prompt | llm | str_parser

    result = str_chain.invoke({"technology": "Blockchain"})
    print(f"✅ Output: {result}")

    # --------------------------
    # Parser 2: List Output
    # --------------------------
    print("\n📋 Parser 2: List Output")
    print("=" * 50)

    list_prompt = PromptTemplate(
        template="List 3 use cases for {technology} (comma-separated):",
        input_variables=["technology"]
    )
    list_parser = CommaSeparatedListOutputParser()
    list_chain = list_prompt | llm | list_parser

    result = list_chain.invoke({"technology": "Blockchain"})
    print(f"✅ Output: {result}")
    print(f"✅ Type: {type(result)} - Python list!")

    # --------------------------
    # Parser 3: JSON Output
    # --------------------------
    print("\n📦 Parser 3: JSON Output")
    print("=" * 50)

    json_parser = JsonOutputParser()
    json_prompt = PromptTemplate(
        template="Return a JSON object with 'name', 'rank', and 'attribute' for {topic}.\n{format_instructions}",
        input_variables=["topic"],
        partial_variables={"format_instructions": json_parser.get_format_instructions()}
    )
    json_chain = json_prompt | llm | json_parser

    result = json_chain.invoke({"topic": "Python Programming"})
    print(f"✅ Output: {json.dumps(result, indent=2)}")
    print(f"✅ Type: {type(result)}")

    # --------------------------
    # Parser 4: Pydantic Output
    # --------------------------
    print("\n🏗️ Parser 4: Pydantic Output")
    print("=" * 50)

    pydantic_parser = PydanticOutputParser(pydantic_object=TechInfo)
    pydantic_prompt = PromptTemplate(
        template="Provide details about {technology}.\n{format_instructions}",
        input_variables=["technology"],
        partial_variables={"format_instructions": pydantic_parser.get_format_instructions()}
    )
    pydantic_chain = pydantic_prompt | llm | pydantic_parser

    result = pydantic_chain.invoke({"technology": "React"})
    print(f"✅ Output: {result}")
    print(f"✅ Name: {result.name}, Creator: {result.creator}")

    # --------------------------
    # Parser 5: Structured Schema
    # --------------------------
    print("\n🧱 Parser 5: Structured Schema")
    print("=" * 50)

    response_schemas = [
        ResponseSchema(name="answer", description="answer to the user's question"),
        ResponseSchema(name="source", description="source used to answer the user's question, should be a website"),
    ]
    structured_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    structured_prompt = PromptTemplate(
        template="Answer the user question as best as possible.\n{format_instructions}\n{question}",
        input_variables=["question"],
        partial_variables={"format_instructions": structured_parser.get_format_instructions()}
    )
    structured_chain = structured_prompt | llm | structured_parser

    result = structured_chain.invoke({"question": "What is the capital of France?"})
    print(f"✅ Output: {result}")

    # --------------------------
    # Parser 6: Regex Extraction
    # --------------------------
    print("\n🔍 Parser 6: Regex Extraction")
    print("=" * 50)

    regex_parser = RegexParser(
        regex=r"Confidence:\s*(\d+)\s*\nReasoning:\s*(.*)",
        output_keys=["confidence", "reasoning"]
    )
    regex_prompt = PromptTemplate(
        template="Rate your confidence in {topic} from 1-100 and give reasoning.\nFormat: 'Confidence: <number>\nReasoning: <text>'",
        input_variables=["topic"]
    )
    regex_chain = regex_prompt | llm | regex_parser

    result = regex_chain.invoke({"topic": "AI Safety"})
    print(f"✅ Output: {result}")

    # --------------------------
    # Parser 7: Enum Output
    # --------------------------
    print("\n🎚️ Parser 7: Enum Output")
    print("=" * 50)

    enum_parser = EnumOutputParser(enum=Difficulty)
    enum_prompt = PromptTemplate(
        template="Rate the difficulty of learning {subject}.\nONLY return one of these choices: {choices}\nDifficulty:",
        input_variables=["subject"],
        partial_variables={"choices": [e.value for e in Difficulty]}
    )
    enum_chain = enum_prompt | llm | enum_parser

    result = enum_chain.invoke({"subject": "Quantum Physics"})
    print(f"✅ Output: {result}")

    # --------------------------
    # Parser 8: Output Fixing
    # --------------------------
    print("\n🛠️ Parser 8: Output Fixing")
    print("=" * 50)

    fixing_parser = OutputFixingParser.from_llm(parser=pydantic_parser, llm=llm)
    fixing_chain = pydantic_prompt | llm | fixing_parser

    result = fixing_chain.invoke({"technology": "TypeScript"})
    print(f"✅ Output (Fixed if needed): {result}")

    # --------------------------
    # Parser 9: Retry Logic
    # --------------------------
    print("\n🔄 Parser 9: Retry Logic")
    print("=" * 50)

    retry_parser = RetryOutputParser.from_llm(parser=pydantic_parser, llm=llm)
    simulated_bad_output = "React was created by Facebook in 2013 as a UI library. It is widely used."
    prompt_value = pydantic_prompt.format_prompt(technology="React")

    print(f"📝 Simulated Bad Input: '{simulated_bad_output}'")
    try:
        pydantic_parser.parse(simulated_bad_output)
    except Exception:
        fixed_result = retry_parser.parse_with_prompt(simulated_bad_output, prompt_value)
        print(f"✅ Successfully Fixed Output: {fixed_result}")

    print("\n💡 Parser Highlights:")
    print("  ✓ Different parsers produce different structured outputs")
    print("  ✓ Pipelines: prompt | llm | parser")
    print("  ✓ Outputs ready for direct application use")

if __name__ == "__main__":
    main()
