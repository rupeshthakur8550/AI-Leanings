"""
Task 5: Complete Chain - Combining Everything!
Build complete AI pipelines using LangChain's chain composition.

Learning Goal: Master chain composition with the pipe operator (|).
"""

import os
import sys
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import (
    CommaSeparatedListOutputParser, 
    StrOutputParser, 
    JsonOutputParser, 
    PydanticOutputParser
)

from langchain_classic.output_parsers import (
    StructuredOutputParser, 
    ResponseSchema, 
    RegexParser, 
    EnumOutputParser
)
from langchain_classic.output_parsers.fix import OutputFixingParser
from langchain_classic.output_parsers.retry import RetryOutputParser

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core import settings

# --- Data Models for Parsers ---

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
    print("🎯 Task 5: Chain Composition with |")
    print("=" * 50)

    llm = ChatOpenAI(
        api_key = settings.OPENAI_API_KEY,
        model = settings.OPENAI_MODEL,
        base_url = settings.OPENAI_API_BASE,
        temperature = 0.3
    )

    # Chain 1: Simple Analysis Chain
    print("\n⛓️ Chain 1: Simple Analysis")
    print("=" * 50)

    analysis_prompt = PromptTemplate(
        template = "Analyze {technology} and provide pros and cons in 2-3 sentences",
        input_variable = ["technology"]
    )

    str_parser = StrOutputParser()

    analysis_chain = analysis_prompt | llm | str_parser

    if analysis_chain:
        result = analysis_chain.invoke({
            "technology": "Blockchain"
        })
        print(f"📝 Input: 'Analyze blockchain'")
        print(f"✅ Output: {result}")

    # Chain 2: List Generation Chain
    print("\n⛓️ Chain 2: List Generation with Parser")
    print("=" * 50)

    list_prompt = PromptTemplate(
        template = "List 3 use cases for {technology} (comma-seperated):",
        input_variables = ["technology"]
    )

    list_parser = CommaSeparatedListOutputParser()

    list_chain = list_prompt | llm | list_parser

    if list_chain:
        result = list_chain.invoke({
            "technology": "Blockchain"
        })
        print(f"📝 Input: 'List use cases for blockchain'")
        print(f"✅ Output: {result}")
        print(f"✅ Type: {type(result)} - Python list!")

    # Chain 3: JSON Output Parser
    print("\n⛓️ Chain 3: JSON Dict Output")
    print("=" * 50)

    json_prompt = PromptTemplate(
        template = "Return a JSON object with 'name', 'rank', and 'attribute' for {topic}.\n{format_instructions}",
        input_variables = ["topic"],
        partial_variables = {"format_instructions": JsonOutputParser().get_format_instructions()}
    )

    json_parser = JsonOutputParser()

    json_chain = json_prompt | llm | json_parser

    if json_chain:
        result = json_chain.invoke({"topic": "Python Programming"})
        print(f"✅ Output: {result}")
        print(f"✅ Type: {type(result)}")

    # Chain 4: Pydantic Output Parser (Strong Typing)
    print("\n⛓️ Chain 4: Pydantic (Strong Typing)")
    print("=" * 50)

    pydantic_parser = PydanticOutputParser(pydantic_object=TechInfo)

    pydantic_prompt = PromptTemplate(
        template = "Provide details about {technology}.\n{format_instructions}",
        input_variables = ["technology"],
        partial_variables = {"format_instructions": pydantic_parser.get_format_instructions()}
    )

    pydantic_chain = pydantic_prompt | llm | pydantic_parser

    if pydantic_chain:
        # We wrap in a try-block because schema enforcement can be strict
        try:
            result = pydantic_chain.invoke({"technology": "React"})
            print(f"✅ Output: {result}")
            print(f"✅ Name: {result.name}, Creator: {result.creator}")
        except Exception as e:
            print(f"⚠️ Pydantic parsing error: {e}")

    # Chain 5: Structured Output Parser
    print("\n⛓️ Chain 5: Structured Output (Simple Schema)")
    print("=" * 50)

    response_schemas = [
        ResponseSchema(name="answer", description="answer to the user's question"),
        ResponseSchema(name="source", description="source used to answer the user's question, should be a website.")
    ]
    structured_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    structured_prompt = PromptTemplate(
        template = "Answer the user question as best as possible.\n{format_instructions}\n{question}",
        input_variables = ["question"],
        partial_variables = {"format_instructions": structured_parser.get_format_instructions()}
    )

    structured_chain = structured_prompt | llm | structured_parser

    if structured_chain:
        result = structured_chain.invoke({"question": "What is the capital of France?"})
        print(f"✅ Output: {result}")

    # Chain 6: Regex Parser
    print("\n⛓️ Chain 6: Regex Extraction")
    print("=" * 50)

    regex_parser = RegexParser(
        regex=r"Confidence:\s*(\d+)\s*\nReasoning:\s*(.*)",
        output_keys=["confidence", "reasoning"]
    )

    regex_prompt = PromptTemplate(
        template = "Rate your confidence in {topic} from 1-100 and give reasoning.\nFormat: 'Confidence: <number>\nReasoning: <text>'",
        input_variables = ["topic"]
    )

    regex_chain = regex_prompt | llm | regex_parser

    if regex_chain:
        result = regex_chain.invoke({"topic": "AI Safety"})
        print(f"✅ Output: {result}")

    # Chain 7: Enum Output Parser
    print("\n⛓️ Chain 7: Enum Output (Fixed Choices)")
    print("=" * 50)

    enum_parser = EnumOutputParser(enum=Difficulty)

    enum_prompt = PromptTemplate(
        template = "Rate the difficulty of learning {subject}.\nONLY return one of these choices: {choices}\nDifficulty:",
        input_variables = ["subject"],
        partial_variables = {"choices": [e.value for e in Difficulty]}
    )

    enum_chain = enum_prompt | llm | enum_parser

    if enum_chain:
        result = enum_chain.invoke({"subject": "Quantum Physics"})
        print(f"✅ Output: {result}")

    # Chain 8: Output Fixing Parser
    print("\n⛓️ Chain 8: Output Fixing (Auto-correction)")
    print("=" * 50)

    # Wrap the existing pydantic parser with an OutputFixingParser
    fixing_parser = OutputFixingParser.from_llm(parser=pydantic_parser, llm=llm)
    
    fixing_chain = pydantic_prompt | llm | fixing_parser

    if fixing_chain:
        result = fixing_chain.invoke({"technology": "TypeScript"})
        print(f"✅ Output (Fixed if needed): {result}")

    # Chain 9: Retry Output Parser
    print("\n⛓️ Chain 9: Retry Parser (Auto-retry logic)")
    print("=" * 50)

    # RetryOutputParser is usually used to wrap a failing parse attempt.
    # It requires passing the prompt to the parse_with_prompt method.
    retry_parser = RetryOutputParser.from_llm(parser=pydantic_parser, llm=llm)
    
    # Let's simulate a scenario where the LLM returns bad data
    bad_output = "React was created by Facebook in 2013 as a UI library. It is widely used."
    prompt_value = pydantic_prompt.format_prompt(technology="React")

    print(f"📝 Simulated Bad Input: '{bad_output}'")
    
    try:
        # This will fail as it's not valid JSON matching our TechInfo schema
        print("� Attempting to parse with standard Pydantic parser...")
        pydantic_parser.parse(bad_output)
    except Exception as e:
        print(f"❌ Initial Parse Failed: Schema mismatch")
        print("🔄 Recovering with RetryOutputParser (calling LLM to fix)...")
        
        # Retry parser uses the LLM + original prompt + bad output to fix it
        fixed_result = retry_parser.parse_with_prompt(bad_output, prompt_value)
        print(f"✅ Successfully Fixed Output: {fixed_result}")

    # Demonstrate the power of chains
    print("\n🎉 Complete Pipeline Example")
    print("=" * 50)

    test_tech = "Artificial Intelligence"

    if analysis_chain and list_chain:
        analysis = analysis_chain.invoke({"technology": test_tech})
        print(f"Analysis: {analysis}")

        use_cases = list_chain.invoke({"technology": test_tech})
        print(f"\nUse cases: \n")
        for i, use_case in enumerate(use_cases, 1):
            print(f"    {i}. {use_case}")

    # Show the magic
    print("\n💡 Chain Composition Magic:")
    print("  ✓ The pipe | operator connects everything")
    print("  ✓ prompt | llm | parser = complete pipeline")
    print("  ✓ Different parsers = different output formats")
    print("  ✓ Same LLM, infinite possibilities!")

    print("\n✅ Task 5 completed! You've mastered LangChain chains!")
    print("🏆 You can now build any AI pipeline with the | operator!")

if __name__ == "__main__":
    main()