# Multimodal Signatures Cookbook

> **🎨 Working with Images, Audio, Tools, and History in LogiLLM**

This cookbook demonstrates how to use LogiLLM's multimodal types for building sophisticated AI applications that work with images, audio, tools, and conversation history.

## Table of Contents

1. [Introduction](#introduction)
2. [Image Processing](#image-processing)
3. [Audio Analysis](#audio-analysis)
4. [Tool Integration](#tool-integration)
5. [History Management](#history-management)
6. [Combined Multimodal Workflows](#combined-multimodal-workflows)
7. [Best Practices](#best-practices)

## Introduction

LogiLLM provides native multimodal types that make it easy to work with different data modalities:

```python
from logillm.core.signatures.types import Image, Audio, Tool, History
```

These types integrate seamlessly with LogiLLM's signature system and provide structured ways to handle complex data.

## Image Processing

### Basic Image Caption Generation

```python
from logillm import Signature, InputField, OutputField, Predict
from logillm.core.signatures.types import Image

class ImageCaption(Signature):
    """Generate a caption for an image."""
    image: Image = InputField(desc="Image to caption")
    style: str = InputField(desc="Caption style (descriptive, funny, technical)")
    
    caption: str = OutputField(desc="Generated caption")
    objects: 'list[str]' = OutputField(desc="Objects detected in the image")

# Usage
caption_generator = Predict(signature=ImageCaption)

# Load image from file
image = Image.from_path("vacation_photo.jpg")

result = caption_generator(
    image=image,
    style="descriptive"
)
print(f"Caption: {result.caption}")
print(f"Objects: {', '.join(result.objects)}")
```

### Image Comparison

```python
class ImageComparison(Signature):
    """Compare two images and describe differences."""
    image1: Image = InputField(desc="First image")
    image2: Image = InputField(desc="Second image")
    focus: 'Optional[str]' = InputField(desc="What to focus on (colors, objects, style)")
    
    similarities: 'list[str]' = OutputField(desc="Similar aspects")
    differences: 'list[str]' = OutputField(desc="Different aspects")
    overall_similarity: float = OutputField(desc="Similarity score 0-1")

# Compare images
comparer = Predict(signature=ImageComparison)

before = Image.from_path("room_before.jpg")
after = Image.from_path("room_after.jpg")

comparison = comparer(
    image1=before,
    image2=after,
    focus="furniture arrangement"
)
```

### Multi-Image Analysis

```python
class MultiImageAnalysis(Signature):
    """Analyze multiple images together."""
    images: 'list[Image]' = InputField(desc="Collection of images")
    task: str = InputField(desc="Analysis task (story, timeline, comparison)")
    
    analysis: str = OutputField(desc="Combined analysis")
    individual_descriptions: 'list[str]' = OutputField(desc="Description per image")
    relationships: 'dict[str, str]' = OutputField(desc="Relationships between images")

# Analyze photo sequence
analyzer = Predict(signature=MultiImageAnalysis)

images = [
    Image.from_path(f"event_photo_{i}.jpg")
    for i in range(1, 6)
]

result = analyzer(
    images=images,
    task="Create a story from these photos"
)
```

### Image with Metadata

```python
class ImageWithContext(Signature):
    """Process image with additional context."""
    image: Image = InputField(desc="Image to analyze")
    metadata: 'dict[str, Any]' = InputField(desc="Image metadata (location, time, camera)")
    previous_analysis: 'Optional[str]' = InputField(desc="Previous analysis if available")
    
    enhanced_description: str = OutputField(desc="Description using all context")
    insights: 'list[str]' = OutputField(desc="Insights from combined data")

# Use image with GPS and time data
image = Image.from_path("sunset.jpg")
metadata = {
    "location": "Grand Canyon, Arizona",
    "timestamp": "2024-08-26 18:45:00",
    "camera": "iPhone 15 Pro",
    "exposure": "1/500s",
}

contextual_analyzer = Predict(signature=ImageWithContext)
result = contextual_analyzer(
    image=image,
    metadata=metadata,
    previous_analysis=None
)
```

## Audio Analysis

### Basic Audio Transcription

```python
from logillm.core.signatures.types import Audio

class AudioTranscription(Signature):
    """Transcribe audio to text."""
    audio: Audio = InputField(desc="Audio to transcribe")
    language: 'Optional[str]' = InputField(desc="Expected language")
    
    transcript: str = OutputField(desc="Text transcript")
    confidence: float = OutputField(desc="Transcription confidence")
    detected_language: str = OutputField(desc="Detected language")

# Transcribe audio
transcriber = Predict(signature=AudioTranscription)

audio = Audio.from_path("meeting_recording.wav")
result = transcriber(
    audio=audio,
    language="en"
)
print(f"Transcript: {result.transcript}")
```

### Audio Sentiment Analysis

```python
class AudioSentiment(Signature):
    """Analyze sentiment and emotion in audio."""
    audio: Audio = InputField(desc="Audio to analyze")
    include_transcript: bool = InputField(desc="Include transcription")
    
    sentiment: str = OutputField(desc="Overall sentiment (positive/negative/neutral)")
    emotions: 'list[str]' = OutputField(desc="Detected emotions")
    energy_level: float = OutputField(desc="Energy level 0-1")
    transcript: 'Optional[str]' = OutputField(desc="Transcript if requested")

# Analyze podcast clip
audio_clip = Audio.from_path("podcast_segment.mp3")
analyzer = Predict(signature=AudioSentiment)

sentiment = analyzer(
    audio=audio_clip,
    include_transcript=True
)
```

### Multi-Speaker Audio

```python
class MultiSpeakerAnalysis(Signature):
    """Analyze audio with multiple speakers."""
    audio: Audio = InputField(desc="Multi-speaker audio")
    expected_speakers: 'Optional[int]' = InputField(desc="Expected number of speakers")
    
    speakers_detected: int = OutputField(desc="Number of speakers detected")
    speaker_segments: 'list[dict[str, Any]]' = OutputField(
        desc="Segments with speaker ID and text"
    )
    summary_per_speaker: 'dict[str, str]' = OutputField(
        desc="Summary of what each speaker said"
    )

# Analyze meeting recording
meeting_audio = Audio.from_path("team_meeting.wav")
speaker_analyzer = Predict(signature=MultiSpeakerAnalysis)

analysis = speaker_analyzer(
    audio=meeting_audio,
    expected_speakers=4
)
```

## Tool Integration

### Basic Tool Usage

```python
from logillm.core.signatures.types import Tool
import json

def search_web(query: str, max_results: int = 5) -> str:
    """Search the web for information."""
    # Implementation
    results = [
        {"title": "Result 1", "snippet": "..."},
        {"title": "Result 2", "snippet": "..."}
    ]
    return json.dumps(results)

def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except:
        return "Invalid expression"

class ToolAssisted(Signature):
    """Answer questions using available tools."""
    question: str = InputField(desc="User question")
    tools: 'list[Tool]' = InputField(desc="Available tools")
    
    tool_calls: 'list[dict[str, str]]' = OutputField(
        desc="Tools to call with arguments"
    )
    final_answer: str = OutputField(desc="Final answer after using tools")

# Create tools
search_tool = Tool(name="search", func=search_web)
calc_tool = Tool(name="calculator", func=calculate)

# Use tools
assistant = Predict(signature=ToolAssisted)
result = assistant(
    question="What is the population of Tokyo multiplied by 2?",
    tools=[search_tool, calc_tool]
)
```

### Advanced Tool Orchestration

```python
class ToolOrchestration(Signature):
    """Orchestrate multiple tools to solve complex tasks."""
    task: str = InputField(desc="Complex task to solve")
    tools: 'list[Tool]' = InputField(desc="Available tools")
    constraints: 'Optional[dict]' = InputField(desc="Constraints and preferences")
    
    plan: 'list[str]' = OutputField(desc="Step-by-step plan")
    tool_sequence: 'list[dict]' = OutputField(
        desc="Ordered tool calls with dependencies"
    )
    expected_outcome: str = OutputField(desc="Expected final outcome")

# Define specialized tools
def analyze_data(data: str, method: str = "statistical") -> str:
    """Analyze data using specified method."""
    return f"Analysis of {data} using {method}"

def generate_report(data: str, format: str = "markdown") -> str:
    """Generate a report from data."""
    return f"# Report\n\n{data}"

def send_email(to: str, subject: str, body: str) -> str:
    """Send an email."""
    return f"Email sent to {to}"

# Create tool collection
tools = [
    Tool(name="analyze", func=analyze_data),
    Tool(name="report", func=generate_report),
    Tool(name="email", func=send_email),
    search_tool,
    calc_tool
]

# Orchestrate complex task
orchestrator = Predict(signature=ToolOrchestration)
result = orchestrator(
    task="Research competitor pricing, analyze the data, create a report, and email it to the team",
    tools=tools,
    constraints={"max_time": "1 hour", "priority": "accuracy"}
)
```

### Tool with Validation

```python
class ValidatedToolUse(Signature):
    """Use tools with validation and error handling."""
    request: str = InputField(desc="User request")
    tools: 'list[Tool]' = InputField(desc="Available tools")
    safety_checks: 'list[str]' = InputField(desc="Safety checks to perform")
    
    validation_results: 'dict[str, bool]' = OutputField(
        desc="Validation check results"
    )
    safe_to_proceed: bool = OutputField(desc="Whether safe to use tools")
    tool_results: 'Optional[list[dict]]' = OutputField(
        desc="Tool results if safe"
    )
    explanation: str = OutputField(desc="Explanation of decisions")

# Safe tool usage
validator = Predict(signature=ValidatedToolUse)
result = validator(
    request="Delete all files in /temp",
    tools=[Tool(name="file_ops", func=lambda cmd: "...")],
    safety_checks=["no_destructive_ops", "path_whitelist", "user_confirmation"]
)
```

## History Management

### Basic Conversation History

```python
from logillm.core.signatures.types import History

class ContextualResponse(Signature):
    """Generate response using conversation history."""
    current_query: str = InputField(desc="Current user query")
    history: History = InputField(desc="Conversation history")
    
    response: str = OutputField(desc="Contextual response")
    referenced_turns: 'list[int]' = OutputField(
        desc="History turns referenced in response"
    )

# Build conversation
history = History()
history.add_turn(role="user", content="My name is Alice")
history.add_turn(role="assistant", content="Nice to meet you, Alice!")
history.add_turn(role="user", content="I work at OpenAI")
history.add_turn(role="assistant", content="That's interesting! OpenAI does fascinating work.")

# Generate contextual response
responder = Predict(signature=ContextualResponse)
result = responder(
    current_query="Do you remember where I work?",
    history=history
)
```

### History Summarization

```python
class HistorySummarization(Signature):
    """Summarize long conversation history."""
    history: History = InputField(desc="Full conversation history")
    max_summary_length: int = InputField(desc="Maximum summary length in words")
    focus_areas: 'Optional[list[str]]' = InputField(desc="Areas to focus on")
    
    summary: str = OutputField(desc="Conversation summary")
    key_points: 'list[str]' = OutputField(desc="Key discussion points")
    action_items: 'list[str]' = OutputField(desc="Identified action items")
    participants: 'list[str]' = OutputField(desc="Conversation participants")

# Summarize long conversation
long_history = History.from_messages([
    {"role": "user", "content": "Let's plan the product launch"},
    {"role": "assistant", "content": "I'll help you plan the launch..."},
    # ... many more messages
])

summarizer = Predict(signature=HistorySummarization)
summary = summarizer(
    history=long_history,
    max_summary_length=200,
    focus_areas=["decisions", "deadlines", "responsibilities"]
)
```

### History-Aware Planning

```python
class HistoryAwarePlanning(Signature):
    """Create plans based on conversation history."""
    objective: str = InputField(desc="Current objective")
    history: History = InputField(desc="Relevant conversation history")
    constraints: 'dict[str, Any]' = InputField(desc="Planning constraints")
    
    plan: 'list[dict[str, str]]' = OutputField(
        desc="Step-by-step plan with rationale"
    )
    assumptions: 'list[str]' = OutputField(
        desc="Assumptions based on history"
    )
    risks: 'list[str]' = OutputField(desc="Identified risks")

# Plan based on past discussions
planner = Predict(signature=HistoryAwarePlanning)
plan = planner(
    objective="Launch the mobile app",
    history=long_history,
    constraints={"timeline": "2 months", "budget": "$50k"}
)
```

## Combined Multimodal Workflows

### Image + Audio Analysis

```python
class VideoAnalysis(Signature):
    """Analyze video by processing image frames and audio."""
    key_frames: 'list[Image]' = InputField(desc="Key frames from video")
    audio_track: Audio = InputField(desc="Video audio track")
    timestamp_mapping: 'dict[int, float]' = InputField(
        desc="Frame index to timestamp mapping"
    )
    
    scene_descriptions: 'list[str]' = OutputField(desc="Description per scene")
    transcript: str = OutputField(desc="Audio transcript with timing")
    summary: str = OutputField(desc="Video summary")
    highlights: 'list[dict[str, Any]]' = OutputField(
        desc="Key moments with timestamp and description"
    )

# Analyze video presentation
frames = [Image.from_path(f"frame_{i}.jpg") for i in range(10)]
audio = Audio.from_path("presentation_audio.wav")
timestamps = {i: i * 3.0 for i in range(10)}  # Every 3 seconds

video_analyzer = Predict(signature=VideoAnalysis)
analysis = video_analyzer(
    key_frames=frames,
    audio_track=audio,
    timestamp_mapping=timestamps
)
```

### Tool + History Integration

```python
class IntelligentAssistant(Signature):
    """AI assistant using tools and conversation history."""
    user_input: str = InputField(desc="User input")
    history: History = InputField(desc="Conversation history")
    tools: 'list[Tool]' = InputField(desc="Available tools")
    user_preferences: 'dict[str, Any]' = InputField(desc="User preferences")
    
    response: str = OutputField(desc="Assistant response")
    tools_used: 'list[str]' = OutputField(desc="Tools that were used")
    history_update: History = OutputField(desc="Updated conversation history")
    learned_preferences: 'dict[str, Any]' = OutputField(
        desc="New preferences learned"
    )

# Intelligent assistant
assistant = Predict(signature=IntelligentAssistant)

# Setup
history = History()
tools = [search_tool, calc_tool]
preferences = {"style": "concise", "expertise": "intermediate"}

# Interact
result = assistant(
    user_input="What's the weather like in Tokyo?",
    history=history,
    tools=tools,
    user_preferences=preferences
)

# Update for next interaction
history = result.history_update
preferences.update(result.learned_preferences)
```

### Complete Multimodal Pipeline

```python
class MultimodalPipeline(Signature):
    """Process multiple modalities in a pipeline."""
    text: str = InputField(desc="Text input")
    images: 'Optional[list[Image]]' = InputField(desc="Associated images")
    audio: 'Optional[Audio]' = InputField(desc="Associated audio")
    history: 'Optional[History]' = InputField(desc="Conversation context")
    tools: 'Optional[list[Tool]]' = InputField(desc="Available tools")
    
    analysis: 'dict[str, Any]' = OutputField(
        desc="Combined analysis from all modalities"
    )
    recommendations: 'list[str]' = OutputField(desc="Recommendations")
    next_steps: 'list[str]' = OutputField(desc="Suggested next steps")
    confidence_scores: 'dict[str, float]' = OutputField(
        desc="Confidence per modality"
    )

# Full multimodal processing
pipeline = Predict(signature=MultimodalPipeline)

result = pipeline(
    text="Analyze this product demo",
    images=[Image.from_path("product_photo.jpg")],
    audio=Audio.from_path("demo_narration.mp3"),
    history=conversation_history,
    tools=[analyze_tool, report_tool]
)
```

## Best Practices

### 1. Memory Management

When working with large multimodal data:

```python
# Good: Load data lazily
class LazyImageProcessor(Signature):
    image_path: str = InputField(desc="Path to image")
    # Load image only when needed
    
# Avoid: Loading all data upfront
class EagerImageProcessor(Signature):
    images: 'list[Image]' = InputField()  # May use too much memory
```

### 2. Error Handling

Always handle multimodal data errors gracefully:

```python
class RobustMultimodal(Signature):
    """Process with fallbacks."""
    primary_image: 'Optional[Image]' = InputField()
    backup_description: 'Optional[str]' = InputField()
    
    result: str = OutputField()
    data_source: str = OutputField(desc="Which source was used")

# Usage with fallback
try:
    image = Image.from_path("might_not_exist.jpg")
except FileNotFoundError:
    image = None

processor = Predict(signature=RobustMultimodal)
result = processor(
    primary_image=image,
    backup_description="A red car parked outside" if not image else None
)
```

### 3. Type Validation

Use LogiLLM's validation features:

```python
class ValidatedMultimodal(Signature):
    """Multimodal with validation."""
    image: Image = InputField(
        desc="Image input",
        validator=lambda x: x.format in ["jpeg", "png"]
    )
    audio: Audio = InputField(
        desc="Audio input",
        validator=lambda x: x.sample_rate >= 16000
    )
    history: History = InputField(
        desc="Conversation history",
        validator=lambda x: len(x.turns) <= 100
    )
```

### 4. Streaming Multimodal Data

For real-time applications:

```python
class StreamingMultimodal(Signature):
    """Process streaming multimodal data."""
    stream_chunk: 'Union[Image, Audio, str]' = InputField(
        desc="Current chunk of stream"
    )
    stream_state: 'dict[str, Any]' = InputField(
        desc="Current stream state"
    )
    
    processed_chunk: 'Any' = OutputField(desc="Processed chunk")
    updated_state: 'dict[str, Any]' = OutputField(desc="Updated state")
    emit_output: bool = OutputField(desc="Whether to emit output now")
```

### 5. Caching Strategies

Implement caching for expensive operations:

```python
from functools import lru_cache

class CachedImageAnalysis(Signature):
    """Analysis with caching."""
    image_hash: str = InputField(desc="Hash of image content")
    analysis_type: str = InputField(desc="Type of analysis")
    
    @lru_cache(maxsize=100)
    def analyze(self, image_hash: str, analysis_type: str):
        # Expensive analysis only done once per unique image
        pass
```

## Advanced Examples

### Multimodal RAG System

```python
class MultimodalRAG(Signature):
    """Retrieval-augmented generation with multiple modalities."""
    query: str = InputField(desc="User query")
    text_docs: 'list[str]' = InputField(desc="Retrieved text documents")
    images: 'list[Image]' = InputField(desc="Retrieved images")
    audio_clips: 'list[Audio]' = InputField(desc="Retrieved audio clips")
    
    answer: str = OutputField(desc="Generated answer")
    sources: 'list[dict[str, str]]' = OutputField(
        desc="Sources used with modality type"
    )
    confidence: float = OutputField(desc="Answer confidence")

# Use in RAG system
rag = Predict(signature=MultimodalRAG)
result = rag(
    query="Explain the Mars rover landing procedure",
    text_docs=retrieved_texts,
    images=retrieved_images,
    audio_clips=retrieved_audio
)
```

### Multimodal Chain-of-Thought

```python
class MultimodalChainOfThought(Signature):
    """Chain-of-thought reasoning across modalities."""
    problem: str = InputField(desc="Problem to solve")
    visual_data: 'Optional[Image]' = InputField(desc="Visual information")
    audio_context: 'Optional[Audio]' = InputField(desc="Audio context")
    tools: 'Optional[list[Tool]]' = InputField(desc="Available tools")
    
    reasoning_steps: 'list[str]' = OutputField(
        desc="Step-by-step reasoning"
    )
    modality_insights: 'dict[str, str]' = OutputField(
        desc="Insights from each modality"
    )
    final_answer: str = OutputField(desc="Final answer")
    confidence_per_step: 'list[float]' = OutputField(
        desc="Confidence for each reasoning step"
    )
```

## Summary

LogiLLM's multimodal types provide a powerful foundation for building sophisticated AI applications that can process and reason across different data modalities. The type system ensures safety and structure while maintaining flexibility for complex workflows.

Key takeaways:
- **Type Safety**: All multimodal types are strongly typed and validated
- **Composability**: Types work seamlessly together in complex signatures
- **Flexibility**: Support for optional types, unions, and custom validators
- **Integration**: Native integration with LogiLLM's module system
- **Performance**: Efficient handling of large multimodal data

For more examples and patterns, check out the [examples directory](../../examples/multimodal/) in the LogiLLM repository.