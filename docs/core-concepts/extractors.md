# Extractors

Extractors handle the impedance mismatch between probabilistic LLM text outputs and deterministic data structures required by applications. They provide robust, reusable parsing logic for common data types.

## Why Extractors?

LLMs produce text in various formats, even when given precise instructions. A simple request for a number might return:
- `"42"`
- `"The answer is 42"`
- `"forty-two"`
- `"It's approximately 42.0"`

Production applications need to reliably extract structured data from these varied responses. Extractors solve this problem systematically.

## Core Principles

All extractors follow these principles:
1. **Never raise exceptions** - Always return a default value
2. **Handle None and empty inputs** gracefully
3. **Try multiple strategies** before giving up
4. **Prefer the most likely interpretation**
5. **Zero dependencies** - Use only Python standard library

## Available Extractors

### Number Extraction

Extract numeric values from text with multiple fallback strategies:

```python
from logillm.core.extractors import Extractors

# Basic usage
answer = Extractors.number("The answer is 42")  # Returns: 42.0

# Handle word forms
answer = Extractors.number("negative seventeen point five")  # Returns: -17.5

# With default value
answer = Extractors.number(None, default=-1)  # Returns: -1.0

# Extract first vs last number
first = Extractors.number("Between 10 and 20", first=True)  # Returns: 10.0
last = Extractors.number("Between 10 and 20", first=False)  # Returns: 20.0
```

### Boolean Extraction

Parse boolean values from various expressions:

```python
# Explicit values
is_correct = Extractors.boolean("yes")  # Returns: True
is_correct = Extractors.boolean("absolutely not")  # Returns: False

# Strict mode - only accept explicit yes/no/true/false
maybe = Extractors.boolean("probably", strict=True)  # Returns: False (default)

# Fuzzy matching (default)
agree = Extractors.boolean("yep")  # Returns: True
agree = Extractors.boolean("nah")  # Returns: False
```

### List Extraction

Extract lists from multiple formats:

```python
# JSON arrays
items = Extractors.list_items('["apple", "banana", "cherry"]')
# Returns: ['apple', 'banana', 'cherry']

# Bullet points
items = Extractors.list_items("""
- apple
- banana
- cherry
""")
# Returns: ['apple', 'banana', 'cherry']

# Comma-separated
items = Extractors.list_items("apple, banana, cherry")
# Returns: ['apple', 'banana', 'cherry']

# With max items limit
items = Extractors.list_items("a, b, c, d, e", max_items=3)
# Returns: ['a', 'b', 'c']
```

### Percentage Extraction

Extract percentages with flexible formatting:

```python
# With percent sign
value = Extractors.percentage("50%")  # Returns: 0.5
value = Extractors.percentage("50%", as_decimal=False)  # Returns: 50.0

# Word forms
value = Extractors.percentage("fifty percent")  # Returns: 0.5
value = Extractors.percentage("half")  # Returns: 0.5

# Decimal interpretation
value = Extractors.percentage("0.75")  # Returns: 0.75
```

### Enum Mapping

Map text to the closest valid option:

```python
options = ["red", "blue", "green"]

# Exact matching
color = Extractors.enum("red", options)  # Returns: "red"

# Case insensitive (default)
color = Extractors.enum("BLUE", options)  # Returns: "blue"

# Fuzzy matching
color = Extractors.enum("reddish", options)  # Returns: "red"
color = Extractors.enum("greenish-blue", options)  # Returns: "blue" or "green"

# With default
color = Extractors.enum("purple", options, default="unknown")  # Returns: "unknown"

# Strict matching (no fuzzy)
color = Extractors.enum("reddish", options, fuzzy=False)  # Returns: None
```

### JSON Object Extraction

Extract and repair JSON objects:

```python
# Clean JSON
data = Extractors.json_object('{"name": "Alice", "age": 30}')
# Returns: {"name": "Alice", "age": 30}

# JSON in code blocks
data = Extractors.json_object("""
```json
{"name": "Alice", "age": 30}
```
""")
# Returns: {"name": "Alice", "age": 30}

# With common errors (trailing commas, single quotes)
data = Extractors.json_object("{'name': 'Alice', 'age': 30,}")
# Returns: {"name": "Alice", "age": 30}

# With default
data = Extractors.json_object("not json", default={})
# Returns: {}
```

## Usage in LogiLLM Applications

### With Predict Modules

```python
from logillm.core.predict import Predict
from logillm.core.extractors import Extractors
from logillm.core.signatures import Signature, InputField, OutputField

class MathProblem(Signature):
    problem: str = InputField(desc="Math word problem")
    answer: str = OutputField(desc="Numerical answer")

solver = Predict(MathProblem)
result = await solver(problem="What is 15 + 27?")

# Robust extraction instead of hoping for clean output
answer = Extractors.number(result.outputs.get("answer"), default=0)
```

### Defensive Programming

Always use extractors when parsing LLM outputs:

```python
# Bad - Fragile
confidence = float(result.outputs.get("confidence", 0))  # Will crash on "high"

# Good - Robust
confidence = Extractors.percentage(
    result.outputs.get("confidence"),
    as_decimal=True,
    default=0.5
)
```

### Validation with Enums

Ensure LLM outputs match valid options:

```python
VALID_CATEGORIES = ["algorithm", "datastructure", "pattern", "protocol"]

# LLM might return "data structure" or "DataStructure" or "data-structure"
raw_category = result.outputs.get("category")

# Normalize to valid option
category = Extractors.enum(
    raw_category, 
    options=VALID_CATEGORIES,
    default="unknown"
)

if category == "unknown":
    logger.warning(f"Unrecognized category: {raw_category}")
```

## Best Practices

### 1. Always Provide Defaults

```python
# Bad - Can return None unexpectedly
value = Extractors.number(text)

# Good - Explicit default
value = Extractors.number(text, default=-1)
```

### 2. Use Type-Specific Extractors

```python
# Bad - Generic parsing
value = str(result.outputs.get("is_valid", "false")).lower() == "true"

# Good - Boolean extractor handles variations
value = Extractors.boolean(result.outputs.get("is_valid"), default=False)
```

### 3. Log Extraction Failures

```python
import logging

logger = logging.getLogger(__name__)

def safe_extract_number(text: str, field_name: str) -> float:
    value = Extractors.number(text, default=-999)
    if value == -999:
        logger.warning(f"Failed to extract number from {field_name}: {text!r}")
    return value
```

### 4. Combine with Validation

```python
from typing import Optional

def extract_percentage_with_validation(
    text: str,
    min_value: float = 0.0,
    max_value: float = 1.0
) -> Optional[float]:
    """Extract and validate percentage."""
    value = Extractors.percentage(text, as_decimal=True)
    
    if value < min_value or value > max_value:
        logger.warning(f"Percentage {value} outside valid range [{min_value}, {max_value}]")
        return None
        
    return value
```

## Integration with Signatures

While extractors are currently used manually, future versions may integrate directly with output fields:

```python
# Potential future API
class FutureSignature(Signature):
    answer: float = OutputField(
        desc="Numerical answer",
        extractor=Extractors.number,
        extractor_args={"default": 0.0}
    )
```

## Performance Considerations

Extractors are designed for efficiency:
- Use regex compilation caching
- Try cheapest strategies first
- Short-circuit on successful extraction
- No external dependencies or heavy libraries

## Common Patterns

### Math Problems

```python
result = await math_solver(problem="Calculate 15% of 200")

# Extract the number (might be "30", "thirty", "The answer is 30", etc.)
answer = Extractors.number(result.outputs.get("answer"), default=0)

# Extract the confidence (might be "high", "0.9", "90%", etc.)
confidence = Extractors.percentage(result.outputs.get("confidence"), default=0.5)
```

### Classification Tasks

```python
# Ensure valid category
category = Extractors.enum(
    result.outputs.get("category"),
    options=VALID_CATEGORIES,
    default=VALID_CATEGORIES[0]
)

# Parse confidence as percentage
confidence = Extractors.percentage(result.outputs.get("confidence"))

# Check if correct
is_correct = Extractors.boolean(result.outputs.get("validation"))
```

### List Processing

```python
# LLM might return various list formats
items = Extractors.list_items(result.outputs.get("items"))

# Limit results
top_5 = Extractors.list_items(result.outputs.get("results"), max_items=5)

# Handle specific delimiter
parts = Extractors.list_items(result.outputs.get("parts"), delimiter=";")
```

## Troubleshooting

### Issue: Number extraction returns wrong value

Check if there are multiple numbers in the text:

```python
text = "The answer is 42 with confidence 0.9"

# Wrong - gets last number (0.9)
answer = Extractors.number(text)  

# Right - gets first number (42)
answer = Extractors.number(text, first=True)

# Better - be more specific in prompt
# Ask LLM to return ONLY the answer
```

### Issue: Boolean extraction inconsistent

Use strict mode for critical decisions:

```python
# Loose matching (default) - "maybe" could be True or False
decision = Extractors.boolean("maybe")

# Strict matching - only explicit values
decision = Extractors.boolean("maybe", strict=True, default=False)
```

### Issue: JSON extraction failing

Check for common formatting issues:

```python
# Enable repair mode (default)
data = Extractors.json_object(text, repair=True)

# Or preprocess the text
text = text.strip()
if text.startswith("```"):
    text = text.split("```")[1]  # Remove code block markers
data = Extractors.json_object(text)
```

## See Also

- [Signatures](./signatures.md) - Define input/output structure
- [Adapters](./adapters.md) - Format prompts for different LLMs
- [Output Validation](./validation.md) - Validate extracted values