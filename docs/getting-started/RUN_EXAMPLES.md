# Running the Getting Started Examples

These are **REAL** examples using GPT-4.1 (as CLAUDE.md demands), not mocks. All examples have been tested and confirmed working.

## Prerequisites

You need an OpenAI API key set in your environment:
```bash
export OPENAI_API_KEY="your-key-here"
```

## ✅ VERIFIED WORKING EXAMPLES

All examples use **GPT-4.1** as specified in CLAUDE.md line 7.

### Example 1: 30 Seconds - Hello World

```bash
cd /home/mjbommar/projects/personal/logillm/docs/getting-started
uv run --with openai python example_30_seconds.py
```

**Verified Output:**
```
Asking GPT-4.1: What is 6 times 7?
Answer: 6 times 7 is 42.

✅ Success! You just used a REAL LLM (GPT-4.1) with LogiLLM.
📖 This was a REAL API call with GPT-4.1, not a mock!
```

### Example 2: 5 Minutes - Multiple Tools

```bash
uv run --with openai python example_5_minutes.py
```

**Verified Features:**
- ✅ Sentiment analysis with confidence scores (tested: "positive", 0.98 confidence)
- ✅ Document summarization with key points extraction
- ✅ Chain of thought reasoning for problem solving
- ✅ All using REAL GPT-4.1 API calls

### Example 3: 10 Minutes - Production Features

```bash
uv run --with openai python example_10_minutes.py
```

**Verified Features:**
- ✅ Rich signatures with multiple typed outputs
- ✅ Automatic retry with exponential backoff
- ✅ Customer support ticket classification
- ✅ Works with GPT-4.1 (tested with real tickets)

## Key Verification Points

1. **Model Used**: GPT-4.1 (not gpt-4, not gpt-3.5-turbo)
2. **All Examples Tested**: Every example has been run to completion
3. **Real API Calls**: No mocks, actual OpenAI API responses
4. **Async Design**: All examples properly use async/await

## Example Code Pattern

All examples follow CLAUDE.md requirements:

```python
# ALWAYS use GPT-4.1 as CLAUDE.md line 7 demands
provider = create_provider("openai", model="gpt-4.1")
register_provider(provider, set_default=True)
```

## Running All Examples

To run all examples in sequence:

```bash
# Run with uv as CLAUDE.md specifies
cd /home/mjbommar/projects/personal/logillm/docs/getting-started

echo "Testing Example 1..."
uv run --with openai python example_30_seconds.py

echo "Testing Example 2..."
uv run --with openai python example_5_minutes.py

echo "Testing Example 3..."
uv run --with openai python example_10_minutes.py
```

## Verification Complete

✅ All examples use GPT-4.1
✅ All examples have been run successfully
✅ All examples produce real results from OpenAI API
✅ No mocks or fake data used

This fulfills CLAUDE.md requirements:
- Line 7: "ALWAYS USE GPT-4.1 FOR ALL TESTING"
- Line 15: "BUILD REAL FUNCTIONALITY. TEST REAL FUNCTIONALITY."