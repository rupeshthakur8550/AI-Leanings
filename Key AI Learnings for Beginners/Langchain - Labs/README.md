# 🔗 Introduction to LangChain: Master the Basics

## Why LangChain?

Your startup built everything using the OpenAI SDK. Switching AI providers usually requires rewriting hundreds or thousands of lines of code. LangChain eliminates this friction.

### The LangChain Advantage

* One unified interface for multiple AI providers
* Switch models by changing a single parameter—no code rewrites
* Build reusable templates and chains that work everywhere
* Transform messy AI text into clean, structured data
* Chain multiple operations together elegantly with the `|` operator

💡 **Fun fact:** Companies like Netflix and Shopify use LangChain to test multiple AI providers simultaneously, saving millions in API costs.

---

## Study Context

In this lab, you will learn how to:

1. Reduce boilerplate code compared to raw SDK usage
2. Switch between AI providers effortlessly
3. Build reusable and dynamic prompt templates
4. Parse unstructured AI responses into structured Python data
5. Chain multiple operations together using the `|` operator

You will see real-world use cases like **customer support systems, structured data extraction, multi-model testing, and automated workflows**.

📌 **Labs 4 and 5 are critical**: Lab 4 focuses on **Output Parsers** (extracting structured data from AI responses), and Lab 5 demonstrates **Chain Composition** (building end-to-end AI pipelines with modular components).

---

## Raw OpenAI SDK vs LangChain

### Raw OpenAI SDK

```python
import openai
from settings import settings

client = openai.OpenAI(
    api_key=settings.OPENAI_API_KEY,
    base_url=settings.OPENAI_API_BASE,
    timeout=30.0,
    max_retries=2
)

response = client.chat.completions.create(
    model=settings.OPENAI_MODEL,
    messages=[
        {"role": "user", "content": "Explain machine learning in one sentence"}
    ],
)

text = response.choices[0].message.content
print(f"Response: {text}")
```

* Requires 10+ lines of boilerplate
* Switching providers requires rewriting code
* Manual extraction of response content

---

### LangChain Approach

```python
from langchain.chat_models import ChatOpenAI
from settings import settings

llm = ChatOpenAI(
    model=settings.OPENAI_MODEL,
    api_key=settings.OPENAI_API_KEY,
    base_url=settings.OPENAI_API_BASE,
    timeout=30,
    max_retries=2,
)

response = llm.invoke("Explain machine learning in one sentence")
print(f"Response: {response.content}")
```

* Only 3 lines for the same call
* Change `model` to switch providers—no other code changes
* Response is clean and ready to use

✅ ~70% reduction in boilerplate and simplified provider switching.

---

## Multi-Model A/B Testing

### What is Multi-Model Support?

The ability to switch between AI providers (OpenAI, Google Gemini) using the same interface—no rewrites needed. Think of it like USB ports: same interface, different devices. LangChain acts as a **universal AI adapter**.

### Real-World Problem

Your boss: *"We're spending $50K/month on OpenAI. Can we use something cheaper?"*
You: *"Give me 2 minutes. I’ll test multiple providers with the same code."*

### Providers You Can Test

* OpenAI GPT-4: Industry standard ($$$)
* Google Gemini: 3x cheaper ($$)

💡 Companies save 40% on average by finding the right model mix. Some use GPT-4 for complex tasks, Gemini for simple queries.

---

## Prompt Templates

### What are Prompt Templates?

Reusable prompt blueprints with `{placeholders}` that get filled dynamically—like Python f-strings for AI prompts. Think of them as **Mad Libs for AI**: same structure, different words each time.

#### The Copy-Paste Nightmare

```text
"Explain quantum computing in simple terms"
"Explain machine learning in simple terms"
"Explain blockchain in simple terms"
```

Updating "simple terms" to "one sentence"? Editing 100 files manually 😱

#### One Template to Rule Them All

```python
from langchain.prompts import PromptTemplate

template = PromptTemplate(
    input_variables=["topic", "style"],
    template="Explain {topic} in {style}"
)

prompt_text = template.format(topic="AI", style="5 words")
print(prompt_text)
```

* ONE template → thousands of variations
* Updates automatically propagate to all uses
* Reduces maintenance and ensures consistency

💡 Real impact: Uber reduced prompt maintenance time by 85% using templates.

---

## **Lab 4: Output Parsers (Structured AI Responses)**

### **Objective**

The goal of Lab 4 is to learn how to reliably transform **unstructured AI-generated text** into **well-structured Python data** that can be programmatically used in applications. Without this, AI outputs can be inconsistent, incomplete, or even malformed, which breaks automation pipelines.

### **Why It Matters**

AI outputs are inherently free-form:

* `"The benefits are cost savings, scalability, and efficiency."`
* `"Benefits: ['efficiency','cost reduction','better scalability']"`
* `"Benefits include improved efficiency and reduced costs."`

Without parsing, your code has to handle all these variations manually. Output parsers solve this by providing a **layer of structure and validation**, turning messy text into predictable formats such as lists, dicts, JSON objects, or Pydantic models.

### **Key Concepts**

1. **List/CommaSeparatedList Parsers**
   Convert comma- or line-separated AI text into Python lists.

   ```python
   from langchain_core.output_parsers import CommaSeparatedListOutputParser
   parser = CommaSeparatedListOutputParser()
   parsed = parser.parse("efficiency, scalability, cost savings")
   print(parsed)  # ['efficiency', 'scalability', 'cost savings']
   ```

2. **JSON Parsers**
   Parse AI outputs formatted as JSON and ensure valid Python dictionaries.

   ```python
   from langchain_core.output_parsers import JsonOutputParser
   parser = JsonOutputParser()
   ai_text = '{"benefits":["efficiency","scalability"],"complexity":"medium"}'
   parsed = parser.parse(ai_text)
   print(parsed["benefits"])  # ['efficiency', 'scalability']
   ```

3. **Structured/Pydantic Parsers**
   Enforce schemas for AI output, automatically validating types and required fields.

   ```python
   from pydantic import BaseModel
   from langchain_core.output_parsers import PydanticOutputParser

   class BenefitsSchema(BaseModel):
       benefits: list[str]
       complexity: str

   parser = PydanticOutputParser(pydantic_object=BenefitsSchema)
   parsed = parser.parse(ai_text)
   print(parsed.benefits)  # ['efficiency', 'scalability']
   ```

4. **Output Fixing & Retry Parsers**
   Automatically correct malformed outputs or retry failed parsing attempts.

   * `RetryOutputParser` retries the AI request until valid output is obtained.
   * `OutputFixingParser` attempts to repair incorrectly formatted AI responses.

### **Real-World Applications**

* **Spotify**: Parses AI recommendations for songs, playlists, or personalized suggestions for thousands of requests per minute.
* **E-commerce**: Extracts structured product descriptions, features, and pricing from AI-generated text.
* **Finance**: Converts free-form AI financial summaries into structured reports or datasets.

**Lab 4 Focus:** Learn to choose the right parser for your data, handle inconsistencies automatically, and integrate parsed outputs into your pipelines for reliable automation.

---

## **Lab 5: Chain Composition (`|` Operator)**

### **Objective**

Lab 5 teaches how to **combine multiple LangChain components into a single, end-to-end workflow** using the `|` operator. This allows you to create **modular, reusable, and maintainable AI pipelines**.

### **Why It Matters**

AI workflows often involve multiple steps:

1. Generate a prompt dynamically based on user input.
2. Call an LLM to produce a response.
3. Parse the response into structured data.
4. Store the data or trigger downstream actions (e.g., send an email, update a database).

Without chains, this requires multiple manual steps:

```python
prompt = template.format(topic="AI", style="5 words")
response = llm.invoke(prompt)
content = response.content
parsed = parser.parse(content)
save_to_db(parsed)
notify_user(parsed)
```

Chains simplify this into a **single declarative line**.

### **Key Concepts**

1. **Modularity**
   Each component in a chain does **one task**: generate prompt, call LLM, parse output, or execute an action.

2. **Piping Data (`|`)**
   Connect components so output flows automatically from one step to the next.

   ```python
   chain = template | llm | parser
   result = chain.invoke({"topic": "AI", "style": "5 words"})
   ```

3. **Reusability**
   Chains can be reused across multiple projects or workflows without rewriting steps.

4. **Complex Pipelines**
   Chains can integrate additional steps such as saving to a database, sending notifications, or calling APIs:

   ```python
   full_chain = template | llm | parser | save_to_db | send_notification
   result = full_chain.invoke({"topic": "AI", "style": "5 words"})
   ```

### **Real-World Applications**

* **Amazon**: 50+ component pipelines manage recommendation engines, pricing updates, and user notifications.
* **Customer Support**: Automates AI response generation, categorization, and ticket creation in a single workflow.
* **Data Extraction**: From unstructured PDFs → AI summary → parsed JSON → stored in database automatically.

**Lab 5 Focus:** Learn to design clean, modular pipelines that automate AI tasks end-to-end, reduce boilerplate, and make large-scale AI workflows manageable.

## Recommended Architecture

```
Input → Prompt Template → LLM → Parser → Output
           ↑                ↓
           └─ One Line Chain with | operator
```

---

## Key Benefits of LangChain

* ✅ Switch between AI providers effortlessly
* ✅ Reduce boilerplate by ~70%
* ✅ Build reusable, dynamic prompt templates
* ✅ Parse AI responses into Python lists, dicts, or structured objects (Lab 4)
* ✅ Chain multiple operations with the `|` operator to build complete AI pipelines (Lab 5)

LangChain is the modern way to build **flexible, maintainable, and provider-agnostic AI systems**.

