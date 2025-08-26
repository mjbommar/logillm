# Signatures in LogiLLM

Signatures are the foundation of LogiLLM's programming paradigm. They define the input/output specifications for modules, similar to function signatures in programming languages.

## What are Signatures?

**ELI5:** A signature is like a contract. It says "if you give me these inputs, I'll give you these outputs." For example, a Q&A signature says "give me a question, I'll give you an answer."

**Technical:** Signatures are declarative specifications that define:
- What inputs a module expects
- What outputs it should produce
- Optional descriptions and constraints for each field
- Instructions for the task

## Basic Usage

LogiLLM supports three ways to define signatures:

### 1. String Syntax (Simplest)

```python
from logillm import Predict

# Simple input -> output
qa = Predict("question -> answer")

# Multiple inputs/outputs
classifier = Predict("text, context -> category, confidence: float")

# With type hints
math = Predict("problem: str -> reasoning: str, solution: float")

# Complex types (NEW)
analyzer = Predict("items: list[str] -> summary: dict[str, int]")

# Optional types (NEW)
extractor = Predict("text: str -> result: Optional[dict]")

# Union types with pipe syntax (NEW)
processor = Predict("data: str | bytes -> processed: bool")
```

### 2. Class-Based Signatures

```python
from logillm import Signature, InputField, OutputField

class QASignature(Signature):
    """Answer questions accurately."""
    
    question: str = InputField(desc="The question to answer")
    answer: str = OutputField(desc="A concise, accurate answer")

# Use with a module
qa = Predict(signature=QASignature)
```

### 3. Dynamic Signatures

```python
from logillm.core.signatures import make_signature

# Create signature from dictionary
fields = {
    "question": (str, InputField(desc="Question")),
    "answer": (str, OutputField(desc="Answer"))
}
sig = make_signature(fields, instructions="Answer questions")
```

## Field Types

Based on the actual implementation in `logillm/core/signatures/fields.py`:

### Supported Python Types

```python
# Basic types
text: str
count: int
score: float
flag: bool

# Generic types (NEW)
items: list[str]
mapping: dict[str, int]
coords: tuple[float, float]
tags: set[str]

# Optional types (NEW)
result: Optional[str]
value: str | None  # Alternative syntax

# Union types (NEW)
data: Union[str, int]
input: str | bytes  # Pipe syntax

# Fractions (NEW)
ratio: Fraction
```

### InputField and OutputField

```python
from logillm import InputField, OutputField

class ExampleSignature(Signature):
    # Basic fields
    query: str = InputField()
    
    # With descriptions
    context: str = InputField(desc="Relevant context for the query")
    
    # With defaults
    max_length: int = InputField(default=100, desc="Maximum response length")
    
    # Output fields
    response: str = OutputField(desc="Generated response")
    confidence: float = OutputField(desc="Confidence score 0-1")
```

### Field Specifications

The signature system uses `FieldSpec` internally to track field metadata:

```python
# From logillm/core/signatures/spec.py
@dataclass
class FieldSpec:
    name: str
    field_type: FieldType  # INPUT or OUTPUT
    python_type: type
    desc: str = ""
    default: Any = None
    optional: bool = False
```

## Working with or without Pydantic

LogiLLM's signature system works transparently with or without Pydantic:

```python
# The same code works whether Pydantic is installed or not
class MySignature(Signature):
    input_text: str = InputField()
    output_text: str = OutputField()

# If Pydantic is available, you get additional validation
# If not, LogiLLM uses its pure Python implementation
```

## Signature Parsing

The parser (in `logillm/core/signatures/parser.py`) handles various formats:

```python
from logillm.core.signatures import parse_signature_string

# Parse simple format
sig = parse_signature_string("question -> answer")

# Parse with types
sig = parse_signature_string("text: str -> category: str, score: float")

# Parse with descriptions (using |)
sig = parse_signature_string("question|User's question -> answer|Generated response")
```

## Validation (Enhanced)

Signatures provide comprehensive validation with type coercion and constraints:

```python
class StrictSignature(Signature):
    # Required vs optional fields (NEW)
    required_field: str = InputField()  # Required, no default
    optional_field: str = InputField(default=None)  # Optional with default
    
    # Type coercion (NEW)
    number: int = InputField()  # "42" -> 42 automatically
    score: float = InputField()  # 3 -> 3.0 automatically
    
    # Validation result
    result: float = OutputField(desc="Calculated result")

# Validation methods (NEW)
validated_inputs = StrictSignature.validate_inputs(
    required_field="test",
    number="42"  # Automatically converted to int
)
# Returns: {"required_field": "test", "number": 42, "optional_field": None}

# Missing required field raises error
try:
    StrictSignature.validate_inputs()  # Missing required_field
except ValueError as e:
    print(e)  # "Required input field 'required_field' not provided"
```

### Field Constraints (NEW)

```python
from logillm.core.signatures.spec import FieldSpec
from logillm.core.signatures.utils import coerce_value_to_spec

# String constraints
spec = FieldSpec(
    name="username",
    python_type=str,
    constraints={"min_length": 3, "max_length": 20}
)

# Numeric constraints  
spec = FieldSpec(
    name="age",
    python_type=int,
    constraints={"min_value": 0, "max_value": 150}
)

# Pattern validation
spec = FieldSpec(
    name="email",
    python_type=str,
    constraints={"pattern": r"^[\w\.]+@[\w\.]+$"}
)

# Choice/enum constraints
spec = FieldSpec(
    name="color",
    python_type=str,
    constraints={"choices": ["red", "green", "blue"]}
)
```

## Multimodal Types (NEW)

LogiLLM supports multimodal inputs and outputs for LLMs that handle images, audio, and tools:

```python
from logillm.core.signatures.types import Image, Audio, Tool, History

# Image support
class VisionSignature(Signature):
    image: Image = InputField(desc="Image to analyze")
    caption: str = OutputField(desc="Generated caption")

# Load images multiple ways
img = Image.from_path("photo.jpg")
img = Image.from_url("https://example.com/image.png") 
img = Image.from_base64(b64_string)

# Audio support
class TranscriptionSignature(Signature):
    audio: Audio = InputField(desc="Audio to transcribe")
    transcript: str = OutputField(desc="Text transcription")

audio = Audio.from_path("recording.mp3")
audio = Audio.from_url("https://example.com/audio.wav")

# Tool/function calling
class ToolSignature(Signature):
    query: str = InputField()
    tool_calls: list[Tool] = OutputField(desc="Functions to execute")

tool = Tool(
    name="calculator",
    description="Performs calculations",
    parameters={"expression": "string"},
    function=lambda expression: eval(expression)
)

# Conversation history
class ChatSignature(Signature):
    history: History = InputField(desc="Previous messages")
    response: str = OutputField(desc="Assistant response")

history = History(messages=[
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"}
])
```

## Advanced Features

### Signature Modification

```python
# Add instructions
enhanced_sig = QASignature.with_instructions("Answer factually and concisely")

# Update field metadata
updated_sig = QASignature.with_updated_field(
    "answer",
    desc="Detailed answer with sources"
)
```

### Signature from Functions

```python
from logillm.core.signatures.parser import signature_from_function

def process_text(text: str, max_length: int = 100) -> str:
    """Process and summarize text."""
    pass

# Create signature from function
sig = signature_from_function(process_text)
```

### Signature from Examples (NEW)

```python
from logillm.core.signatures.parser import infer_signature_from_examples

# Infer signature from input/output examples
examples = [
    {
        "input": {"question": "What is 2+2?"},
        "output": {"answer": "4", "confidence": 0.99}
    },
    {
        "input": {"question": "Capital of France?"},
        "output": {"answer": "Paris", "confidence": 0.95}
    }
]

# Automatically infers types from values
fields = infer_signature_from_examples(examples)
# Creates: question: str -> answer: str, confidence: float
```

### Custom Type Support (NEW)

```python
from logillm.core.signatures import make_signature

# Define custom types
class Document:
    def __init__(self, text: str, metadata: dict):
        self.text = text
        self.metadata = metadata

# Use custom types in signatures
sig = make_signature(
    "doc: Document -> summary: str",
    custom_types={"Document": Document}
)
```

## Integration with Modules

Signatures are used by modules to:
1. Validate inputs before processing
2. Structure prompts for LLMs
3. Parse and validate outputs
4. Provide metadata for optimization

```python
from logillm import Predict

# Module uses signature for validation
class FactCheckSignature(Signature):
    claim: str = InputField(desc="Claim to verify")
    verdict: bool = OutputField(desc="True if accurate")
    explanation: str = OutputField(desc="Reasoning")

fact_checker = Predict(signature=FactCheckSignature)

# Signature validates inputs and structures outputs
result = fact_checker(claim="The Earth is flat")
# result.verdict: bool
# result.explanation: str
```

## Comparison with DSPy

| Feature | DSPy | LogiLLM |
|---------|------|---------|
| **Dependencies** | Requires Pydantic | Works with or without |
| **String Syntax** | Limited | Full support with types |
| **Complex Types** | AST parsing | Safe evaluation with brackets |
| **Multimodal Types** | Basic | Image, Audio, Tool, History |
| **Type Inference** | Frame introspection | Explicit custom_types |
| **Field Validation** | Pydantic-only | Built-in + constraints |
| **Required Fields** | Pydantic behavior | Proper sentinel values |
| **Metaclass** | Complex magic | Clean implementation |
| **Dynamic Creation** | Limited | Full factory support |

## Best Practices

1. **Start Simple**: Use string syntax for prototyping
   ```python
   qa = Predict("question -> answer")
   ```

2. **Add Types for Production**: Use class-based signatures with types
   ```python
   class ProductionSignature(Signature):
       input: str = InputField(desc="User input")
       output: str = OutputField(desc="Generated output")
       confidence: float = OutputField(desc="Confidence 0-1")
   ```

3. **Provide Descriptions**: Help the LLM understand fields
   ```python
   field = InputField(desc="Detailed description of what this field contains")
   ```

4. **Use Appropriate Types**: Match Python types to expected values
   ```python
   count: int = OutputField(desc="Number of items")
   probability: float = OutputField(desc="Probability 0-1")
   items: list[str] = OutputField(desc="List of items")
   ```

## Common Patterns

### Question Answering
```python
qa = Predict("question -> answer")
```

### Classification
```python
class ClassifySignature(Signature):
    text: str = InputField()
    category: str = OutputField(desc="One of: positive, negative, neutral")
    confidence: float = OutputField()
```

### Extraction
```python
class ExtractSignature(Signature):
    document: str = InputField()
    entities: list[str] = OutputField(desc="Named entities")
    summary: str = OutputField(desc="Brief summary")
```

### Multi-step Reasoning
```python
class ReasoningSignature(Signature):
    problem: str = InputField()
    steps: list[str] = OutputField(desc="Solution steps")
    answer: str = OutputField()
```

## Implementation Details

The signature system is implemented across several files:
- `signatures/base.py`: BaseSignature class
- `signatures/signature.py`: Enhanced Signature with Pydantic support
- `signatures/fields.py`: Field descriptors and types
- `signatures/parser.py`: String parsing logic
- `signatures/factory.py`: Dynamic signature creation
- `signatures/spec.py`: Field specifications
- `signatures/utils.py`: Utility functions

The system uses metaclasses to provide a clean API while maintaining compatibility with both Pydantic and pure Python implementations.