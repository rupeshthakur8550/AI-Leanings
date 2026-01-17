"""
Task 5: Understanding Tokens and Business Costs
Learn how tokens work and calculate real business costs for AI usage.
"""

import openai
import os
 
client = openai.OpenAI(
    api_key = os.getenv("OPENAI_API_KEY"),
    base_url = os.getenv("OPENAI_API_BASE")
)

prompt = "Explain the benefits of using AI for customer support in a business?"

response = client.chat.completions.create(
    model = os.getenv("OPENAI_MODEL"),
    messages = [{"role": "user", "context": prompt}]
)

# ==========================================
# WHAT ARE TOKENS?
# ==========================================
#
# Think of tokens as "pieces of words" that AI uses:
# - Simple words = 1 token (e.g., "cat", "run")
# - Complex words = multiple tokens (e.g., "unbelievable" = 3 tokens)
# - Rough estimate: 1 token ≈ 4 characters or 0.75 words
#
# The response.usage object tells you EXACTLY how many tokens you used:
# ┌────────────────────────────────────┐
# │ response.usage                      │
# │  ├── prompt_tokens      (input)    │ ← What you asked
# │  ├── completion_tokens  (output)   │ ← What AI answered
# │  └── total_tokens       (sum)      │ ← What you pay for
# └────────────────────────────────────┘
# ==========================================

# TODO: Extract the token counts from response.usage

input_tokens = response.usage.prompt_tokens
output_tokens = response.usage.completion_tokens
total_tokens = response.usage.total_tokens

print("📊 Token Usage Report:")
print("="*50)
print(f"  Your question used: {input_tokens} tokens")
print(f"  AI's response used: {output_tokens} tokens")
print(f"  Total tokens billed: {total_tokens} tokens")
print("="*50)

# ==========================================
# CALCULATING REAL BUSINESS COSTS
# ==========================================
#
# GPT-4.1-mini Official Pricing:
# ┌─────────────────────────────────────┐
# │ Input:  $0.80 per 1M tokens         │
# │         = $0.0008 per 1K tokens     │
# │                                      │
# │ Output: $3.20 per 1M tokens         │
# │         = $0.0032 per 1K tokens     │
# └─────────────────────────────────────┘
#
# Notice: Output costs 4x more than input!
# This is why keeping AI responses concise matters for your budget.
# ==========================================

input_price_per_1k_token = 0.0008
output_price_per_1k_token = 0.0032

input_cost = (input_tokens / 1000) * input_price_per_1k_token
output_cost = (output_tokens / 1000) * output_price_per_1k_token
total_cost = input_cost + output_cost

print("\n💰 Cost Breakdown for This Call:")
print("-"*50)
print(f"  Input cost:  ${input_cost:.6f} ({input_tokens} tokens)")
print(f"  Output cost: ${output_cost:.6f} ({output_tokens} tokens)")
print(f"  TOTAL COST:  ${total_cost:.6f}")
print("-"*50)

print("\n✅ Task 5 completed! You now understand tokens and costs!")