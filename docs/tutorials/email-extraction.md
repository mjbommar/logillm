# Extracting Information from Emails with LogiLLM

> **📍 Tutorial Path**: [LLM Text Generation](./llms-txt-generation.md) → **Email Extraction** → [Code Generation](./code-generation.md) → [Yahoo Finance ReAct](./yahoo-finance-react.md)  
> **⏱️ Time**: 15-20 minutes | **🎯 Difficulty**: Beginner  
> **💡 Concepts**: Structured data extraction, Pydantic models, Field validation, Email classification

This tutorial demonstrates how to build an intelligent email processing system using LogiLLM. We'll create a system that can automatically extract key information from various types of emails, classify their intent, and structure the data for further processing.

**Perfect for**: Developers ready to learn structured data processing, anyone building email/communication tools, those wanting to understand Pydantic integration.

**Builds on**: [LLM Text Generation](./llms-txt-generation.md) - Now we'll take free-form text output and make it structured and validated.

## What You'll Build

By the end of this tutorial, you'll have a LogiLLM-powered email processing system that can:

- **Classify email types** (order confirmation, support request, meeting invitation, etc.)
- **Extract key entities** (dates, amounts, product names, contact info)
- **Determine urgency levels** and required actions
- **Structure extracted data** into consistent formats
- **Handle multiple email formats** robustly
- **Save and reuse optimized processors** for consistent performance

## Prerequisites

- Basic understanding of LogiLLM modules and signatures
- Python 3.9+ installed
- OpenAI or Anthropic API key
- Familiarity with async/await patterns

## Installation and Setup

```bash
# Install LogiLLM with provider support
pip install logillm[openai]
# or
pip install logillm[anthropic]

# For data validation (optional but recommended)
pip install pydantic
```

## Step 1: Define Our Data Structures

First, let's define the types of information we want to extract from emails:

```python
# models.py
from typing import List, Optional, Literal, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


class EmailType(str, Enum):
    """Types of emails we can classify."""
    ORDER_CONFIRMATION = "order_confirmation"
    SUPPORT_REQUEST = "support_request"  
    MEETING_INVITATION = "meeting_invitation"
    NEWSLETTER = "newsletter"
    PROMOTIONAL = "promotional"
    INVOICE = "invoice"
    SHIPPING_NOTIFICATION = "shipping_notification"
    URGENT_ALERT = "urgent_alert"
    OTHER = "other"


class UrgencyLevel(str, Enum):
    """Urgency levels for email classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ExtractedEntity(BaseModel):
    """Represents a key entity extracted from email content."""
    entity_type: str = Field(description="Type of entity (date, amount, name, etc.)")
    value: str = Field(description="The extracted value")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    context: Optional[str] = Field(None, description="Surrounding context")


class EmailInsight(BaseModel):
    """Complete analysis result for an email."""
    email_type: EmailType
    urgency: UrgencyLevel
    summary: str
    key_entities: List[ExtractedEntity]
    action_required: bool
    deadline: Optional[str] = None
    financial_amount: Optional[float] = None
    sender_info: Optional[str] = None
    priority_score: int = Field(ge=1, le=10, description="Priority from 1-10")
    action_items: List[str] = Field(default_factory=list)
    reasoning: Optional[str] = None
```

## Step 2: Create LogiLLM Signatures

Now let's define the signatures for our email processing pipeline:

```python
# signatures.py
from logillm.core.signatures import Signature, InputField, OutputField
from typing import List, Optional
from .models import EmailType, UrgencyLevel, ExtractedEntity


class ClassifyEmail(Signature):
    """Classify the type and urgency of an email based on its content."""

    email_subject: str = InputField(desc="The subject line of the email")
    email_body: str = InputField(desc="The main content of the email")
    sender: str = InputField(desc="Email sender information")

    email_type: EmailType = OutputField(desc="The classified type of email")
    urgency: UrgencyLevel = OutputField(desc="The urgency level of the email")
    reasoning: str = OutputField(desc="Brief explanation of the classification")


class ExtractEntities(Signature):
    """Extract key entities and information from email content with validation."""

    email_content: str = InputField(
        desc="The full email content including subject and body",
        min_length=1,  # Ensure non-empty content
        max_length=10000  # Reasonable email length limit
    )
    email_type: EmailType = InputField(desc="The classified type of email")

    key_entities: List[ExtractedEntity] = OutputField(
        desc="List of extracted entities with type, value, and confidence",
        min_items=0,  # Can be empty
        max_items=50  # Reasonable limit for entities
    )
    financial_amount: Optional[float] = OutputField(
        desc="Any monetary amounts found",
        ge=0.0  # Amount must be non-negative if present
    )
    important_dates: List[str] = OutputField(desc="List of important dates found")
    contact_info: List[str] = OutputField(desc="Relevant contact information extracted")


class GenerateActionItems(Signature):
    """Determine what actions are needed based on email content."""

    email_type: EmailType = InputField()
    urgency: UrgencyLevel = InputField()
    email_summary: str = InputField(desc="Brief summary of the email content")
    extracted_entities: List[ExtractedEntity] = InputField(desc="Key entities found")

    action_required: bool = OutputField(desc="Whether any action is required")
    action_items: List[str] = OutputField(desc="List of specific actions needed")
    deadline: Optional[str] = OutputField(desc="Deadline for action if applicable")
    priority_score: int = OutputField(desc="Priority score from 1-10")


class SummarizeEmail(Signature):
    """Create a concise summary of the email content."""

    email_subject: str = InputField()
    email_body: str = InputField()
    key_entities: List[ExtractedEntity] = InputField()

    summary: str = OutputField(desc="A 2-3 sentence summary of the email's main points")
```

## Step 3: Build the Email Processing Module

Now let's create our main email processing module:

```python
# processor.py
from typing import Dict, Any
from logillm.core.predict import Predict
from logillm.core.modules import Module
from .signatures import ClassifyEmail, ExtractEntities, GenerateActionItems, SummarizeEmail
from .models import EmailInsight, EmailType, UrgencyLevel


class EmailProcessor(Module):
    """A comprehensive email processing system using LogiLLM."""

    def __init__(self) -> None:
        super().__init__()

        # Initialize our processing components
        self.classifier = Predict(signature=ClassifyEmail)
        self.entity_extractor = Predict(signature=ExtractEntities)
        self.action_generator = Predict(signature=GenerateActionItems)
        self.summarizer = Predict(signature=SummarizeEmail)

    async def forward(
        self, 
        email_subject: str, 
        email_body: str, 
        sender: str = ""
    ) -> EmailInsight:
        """Process an email and extract structured information."""

        # Step 1: Classify the email
        classification = await self.classifier(
            email_subject=email_subject,
            email_body=email_body,
            sender=sender
        )

        # Step 2: Extract entities
        full_content = f"Subject: {email_subject}\n\nFrom: {sender}\n\n{email_body}"
        entities = await self.entity_extractor(
            email_content=full_content,
            email_type=classification.email_type
        )

        # Step 3: Generate summary
        summary = await self.summarizer(
            email_subject=email_subject,
            email_body=email_body,
            key_entities=entities.key_entities
        )

        # Step 4: Determine actions
        actions = await self.action_generator(
            email_type=classification.email_type,
            urgency=classification.urgency,
            email_summary=summary.summary,
            extracted_entities=entities.key_entities
        )

        # Step 5: Structure the results
        return EmailInsight(
            email_type=classification.email_type,
            urgency=classification.urgency,
            summary=summary.summary,
            key_entities=entities.key_entities,
            financial_amount=entities.financial_amount,
            action_required=actions.action_required,
            action_items=actions.action_items,
            deadline=actions.deadline,
            priority_score=actions.priority_score,
            reasoning=classification.reasoning,
            sender_info=sender
        )
```

## Step 4: Sample Data and Demo Application

Let's create sample emails and a demo application:

```python
# demo.py
import asyncio
import os
from typing import List, Dict, Any
from pathlib import Path

from logillm.providers import create_provider, register_provider
from .processor import EmailProcessor
from .models import EmailInsight


# Sample emails for testing
SAMPLE_EMAILS = [
    {
        "subject": "Order Confirmation #12345 - Your MacBook Pro is on the way!",
        "body": """Dear John Smith,

Thank you for your order! We're excited to confirm that your order #12345 has been processed.

Order Details:
- MacBook Pro 14-inch (Space Gray)
- Order Total: $2,399.00
- Estimated Delivery: December 15, 2024
- Tracking Number: 1Z999AA1234567890

If you have any questions, please contact our support team at support@techstore.com.

Best regards,
TechStore Team""",
        "sender": "orders@techstore.com"
    },
    {
        "subject": "URGENT: Server Outage - Immediate Action Required",
        "body": """Hi DevOps Team,

We're experiencing a critical server outage affecting our production environment.

Impact: All users unable to access the platform
Started: 2:30 PM EST

Please join the emergency call immediately: +1-555-123-4567

This is our highest priority.

Thanks,
Site Reliability Team""",
        "sender": "alerts@company.com"
    },
    {
        "subject": "Meeting Invitation: Q4 Planning Session", 
        "body": """Hello team,

You're invited to our Q4 planning session.

When: Friday, December 20, 2024 at 2:00 PM - 4:00 PM EST
Where: Conference Room A

Please confirm your attendance by December 18th.

Best,
Sarah Johnson""",
        "sender": "sarah.johnson@company.com"
    },
    {
        "subject": "Invoice #INV-2024-001 - Payment Due",
        "body": """Dear Customer,

Your invoice for consulting services is now due.

Invoice Details:
- Invoice Number: INV-2024-001
- Amount Due: $5,250.00
- Due Date: January 15, 2025
- Payment Terms: Net 30 days

Please remit payment by the due date to avoid late fees.

Payment can be made online at portal.consulting.com or by check.

Thank you,
Accounting Department""",
        "sender": "billing@consulting.com"
    }
]


async def run_email_processing_demo() -> None:
    """Demonstration of the email processing system."""
    
    # Configure LogiLLM
    model = os.environ.get("MODEL", "gpt-4o-mini")
    
    if model.startswith("gpt"):
        if not os.environ.get("OPENAI_API_KEY"):
            print("⚠️  Please set OPENAI_API_KEY environment variable")
            return
        provider = create_provider("openai", model=model)
    elif model.startswith("claude"):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("⚠️  Please set ANTHROPIC_API_KEY environment variable") 
            return
        provider = create_provider("anthropic", model=model)
    else:
        raise ValueError(f"Unsupported model: {model}")
        
    register_provider(provider, set_default=True)
    
    # Create our email processor
    processor = EmailProcessor()
    
    # Process each email and display results
    print("🚀 Email Processing Demo")
    print("=" * 50)
    
    results = []
    for i, email in enumerate(SAMPLE_EMAILS):
        print(f"\n📧 EMAIL {i+1}: {email['subject'][:50]}...")
        
        # Process the email
        result = await processor.forward(
            email_subject=email["subject"],
            email_body=email["body"],
            sender=email["sender"]
        )
        
        results.append(result)
        
        # Display key results
        print(f"   📊 Type: {result.email_type.value}")
        print(f"   🚨 Urgency: {result.urgency.value}")
        print(f"   📝 Summary: {result.summary}")
        print(f"   🎯 Priority: {result.priority_score}/10")
        
        if result.financial_amount:
            print(f"   💰 Amount: ${result.financial_amount:,.2f}")
        
        if result.action_required:
            print(f"   ✅ Action Required: Yes")
            if result.deadline:
                print(f"   ⏰ Deadline: {result.deadline}")
            if result.action_items:
                print(f"   📋 Actions: {', '.join(result.action_items[:2])}{'...' if len(result.action_items) > 2 else ''}")
        else:
            print(f"   ✅ Action Required: No")
            
        if result.key_entities:
            print(f"   🏷️  Entities: {len(result.key_entities)} found")
    
    # Summary statistics
    print(f"\n📈 SUMMARY")
    print(f"   Total emails processed: {len(results)}")
    print(f"   Requiring action: {sum(1 for r in results if r.action_required)}")
    print(f"   High/Critical urgency: {sum(1 for r in results if r.urgency.value in ['high', 'critical'])}")
    print(f"   Financial amounts found: {sum(1 for r in results if r.financial_amount)}")

async def main() -> None:
    """Main demo entry point."""
    await run_email_processing_demo()


if __name__ == "__main__":
    asyncio.run(main())
```

## Step 5: Enhanced Field Validation (New Features!)

LogiLLM now provides advanced field validation capabilities. Let's create an enhanced version with comprehensive validation:

```python
# enhanced_signatures.py
from logillm.core.signatures import Signature, InputField, OutputField
from typing import List, Optional, Literal
import re

class EnhancedEmailExtractor(Signature):
    """Extract email data with comprehensive validation constraints."""
    
    # Input validation
    email_subject: str = InputField(
        desc="Email subject line",
        min_length=1,
        max_length=200  # RFC 2822 recommends max 78 chars, but we're lenient
    )
    
    email_body: str = InputField(
        desc="Email body content",
        min_length=1,
        max_length=50000  # ~10 pages of text
    )
    
    sender_email: str = InputField(
        desc="Sender's email address",
        # Could add pattern validation when using Pydantic
        # pattern=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    )
    
    # Output validation with constraints
    urgency_level: Literal["low", "medium", "high", "critical"] = OutputField(
        desc="Email urgency classification"
    )
    
    confidence_score: float = OutputField(
        desc="Classification confidence",
        ge=0.0,  # Greater than or equal to 0
        le=1.0   # Less than or equal to 1
    )
    
    detected_language: str = OutputField(
        desc="Primary language of the email",
        default="en",  # Default to English
        max_length=5  # ISO 639-1 codes are 2-3 chars
    )
    
    phone_numbers: List[str] = OutputField(
        desc="Extracted phone numbers",
        max_items=10,  # Reasonable limit
        default_factory=list  # Empty list if none found
    )
    
    urls: List[str] = OutputField(
        desc="Extracted URLs",
        max_items=20,
        default_factory=list
    )
    
    # Optional fields with validation
    meeting_time: Optional[str] = OutputField(
        desc="Detected meeting time if present",
        default=None
    )
    
    estimated_response_time: Optional[int] = OutputField(
        desc="Estimated minutes to respond",
        default=None,
        ge=0,  # Must be non-negative
        le=10080  # Max one week in minutes
    )

class ValidatedEmailProcessor:
    """Email processor with automatic validation."""
    
    def __init__(self):
        from logillm.core.predict import Predict
        self.extractor = Predict(signature=EnhancedEmailExtractor)
    
    async def process_with_validation(self, subject: str, body: str, sender: str):
        """Process email with automatic input/output validation."""
        
        # Input validation happens automatically
        try:
            result = await self.extractor(
                email_subject=subject,
                email_body=body,
                sender_email=sender
            )
            
            # Output validation also automatic
            print(f"Urgency: {result.urgency_level}")  # Guaranteed to be valid
            print(f"Confidence: {result.confidence_score:.2%}")  # Always 0-1
            print(f"Language: {result.detected_language}")
            
            if result.phone_numbers:
                print(f"Found {len(result.phone_numbers)} phone numbers")
            
            if result.meeting_time:
                print(f"Meeting detected: {result.meeting_time}")
                
            return result
            
        except ValueError as e:
            print(f"Validation error: {e}")
            # Handle invalid inputs or outputs
            raise
```

### Using Complex Types for Richer Data

```python
from typing import Dict, List, Tuple
from logillm.core.signatures.types import History

class EmailThreadAnalyzer(Signature):
    """Analyze entire email threads with context."""
    
    # Use History type for conversation threads
    thread_history: History = InputField(
        desc="Email thread as conversation history"
    )
    
    # Complex output types
    participants: Dict[str, List[str]] = OutputField(
        desc="Map of participant emails to their roles"
    )
    
    topic_evolution: List[Tuple[int, str]] = OutputField(
        desc="How the topic changed over messages (index, topic)"
    )
    
    sentiment_timeline: List[float] = OutputField(
        desc="Sentiment score for each message in thread"
    )
    
    action_owners: Dict[str, str] = OutputField(
        desc="Map of action items to responsible persons"
    )

# Usage example
async def analyze_thread():
    from logillm.core.predict import Predict
    from logillm.core.signatures.types import History
    
    analyzer = Predict(signature=EmailThreadAnalyzer)
    
    # Build thread history
    thread = History(messages=[
        {"role": "user", "content": "Can we schedule a meeting about Q4 planning?"},
        {"role": "assistant", "content": "Yes, I have slots Tuesday or Thursday afternoon."},
        {"role": "user", "content": "Thursday 3pm works. Please send calendar invite."}
    ])
    
    result = await analyzer(thread_history=thread)
    
    # Rich structured output
    print(f"Participants: {result.participants}")
    print(f"Topic changes: {result.topic_evolution}")
    print(f"Sentiment over time: {result.sentiment_timeline}")
    print(f"Action assignments: {result.action_owners}")
```

### Validation Best Practices

1. **Use constraints for data quality**: Set reasonable min/max values
2. **Leverage Optional types**: Not all fields are always present
3. **Set defaults**: Provide sensible defaults for optional fields
4. **Use Literal types**: For fields with known values
5. **Validate patterns**: Use regex patterns for emails, URLs, phones
6. **Complex types**: Use Dict, List, Tuple for structured data
7. **Multimodal types**: Use History for threads, Image for attachments

## Step 6: Advanced Features - Optimization and Persistence

```python
# optimization.py
import asyncio
from typing import List, Dict, Any
from logillm.core.optimizers import AccuracyMetric
from logillm.optimizers import BootstrapFewShot, HybridOptimizer
from .processor import EmailProcessor
from .models import EmailType, UrgencyLevel


async def optimize_email_processor() -> EmailProcessor:
    """Demonstrate how to optimize the email processor for better results."""
    
    # Sample training data (in production, you'd have more examples)
    training_data = [
        {
            "inputs": {
                "email_subject": "Order #123 Shipped",
                "email_body": "Your order has been shipped. Tracking: XYZ123",
                "sender": "shop@store.com"
            },
            "outputs": {
                "email_type": EmailType.SHIPPING_NOTIFICATION,
                "urgency": UrgencyLevel.LOW,
                "action_required": False
            }
        },
        {
            "inputs": {
                "email_subject": "URGENT: Security Alert",
                "email_body": "Suspicious login detected. Please verify immediately.",
                "sender": "security@company.com"
            },
            "outputs": {
                "email_type": EmailType.URGENT_ALERT,
                "urgency": UrgencyLevel.CRITICAL,
                "action_required": True
            }
        }
        # Add more training examples...
    ]
    
    # Create processor and optimize
    processor = EmailProcessor()
    
    # Define quality metrics
    def classification_accuracy(prediction: Dict[str, Any], reference: Dict[str, Any]) -> float:
        """Custom metric to evaluate email classification accuracy."""
        pred_insight = prediction  # EmailInsight object
        ref_data = reference
        
        score = 0.0
        
        # Check email type classification
        if pred_insight.email_type == ref_data["email_type"]:
            score += 0.4
        
        # Check urgency classification  
        if pred_insight.urgency == ref_data["urgency"]:
            score += 0.3
            
        # Check action required
        if pred_insight.action_required == ref_data["action_required"]:
            score += 0.3
        
        return score
    
    # Optimize the processor
    metric = AccuracyMetric(key=None, metric_fn=classification_accuracy)
    optimizer = HybridOptimizer(
        metric=metric,
        strategy="alternating",
        verbose=True
    )
    
    print("🎯 Optimizing email processor...")
    result = await optimizer.optimize(
        module=processor,
        dataset=training_data,
        param_space={
            "temperature": (0.0, 0.8),
            "top_p": (0.8, 1.0)
        }
    )
    
    # Save the optimized model
    optimized_processor = result.optimized_module
    optimized_processor.save("models/optimized_email_processor.json")
    
    print(f"✅ Optimization complete! Improvement: {result.improvement:.2%}")
    print("💾 Optimized model saved to: models/optimized_email_processor.json")
    
    return optimized_processor


async def use_optimized_processor() -> None:
    """Load and use the optimized email processor."""
    
    # Load the pre-trained, optimized model  
    processor = EmailProcessor.load("models/optimized_email_processor.json")
    
    # Use it for fast, high-quality email processing
    result = await processor.forward(
        email_subject="Meeting Tomorrow",
        email_body="Don't forget our meeting at 2 PM tomorrow in Room B.",
        sender="colleague@company.com"
    )
    
    print(f"📧 Processed with optimized model:")
    print(f"   Type: {result.email_type.value}")
    print(f"   Urgency: {result.urgency.value}")
    print(f"   Summary: {result.summary}")
    
    return result
```

## Testing the Tutorial

Create a comprehensive test script:

```bash
# test_email_tutorial.py
"""
Test script for the email extraction tutorial.
Run with: uv run --with logillm[openai] --with pydantic python test_email_tutorial.py
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from examples.tutorials.email_extraction.demo import run_email_processing_demo
from examples.tutorials.email_extraction.processor import EmailProcessor


async def test_tutorial() -> None:
    """Test the email extraction tutorial."""
    
    # Check for required environment variables
    model = os.environ.get("MODEL", "gpt-4o-mini")
    
    if model.startswith("gpt") and not os.environ.get("OPENAI_API_KEY"):
        print("⚠️  Please set OPENAI_API_KEY environment variable")
        print("   export OPENAI_API_KEY='your-key-here'")
        return
    elif model.startswith("claude") and not os.environ.get("ANTHROPIC_API_KEY"):
        print("⚠️  Please set ANTHROPIC_API_KEY environment variable")
        print("   export ANTHROPIC_API_KEY='your-key-here'")
        return
    
    try:
        print("🧪 Running email processing tutorial test...")
        await run_email_processing_demo()
        print("✅ Tutorial test completed successfully!")
        
    except Exception as e:
        print(f"❌ Tutorial test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_tutorial())
```

## Expected Output

When you run the tutorial, you should see output like:

```
🚀 Email Processing Demo
==================================================

📧 EMAIL 1: Order Confirmation #12345 - Your MacBook Pro is on...
   📊 Type: order_confirmation
   🚨 Urgency: low
   📝 Summary: The email confirms John Smith's order #12345 for a MacBook Pro 14-inch...
   🎯 Priority: 3/10
   💰 Amount: $2,399.00
   ✅ Action Required: No
   🏷️  Entities: 4 found

📧 EMAIL 2: URGENT: Server Outage - Immediate Action Required...
   📊 Type: urgent_alert
   🚨 Urgency: critical
   📝 Summary: The Site Reliability Team has reported a critical server outage...
   🎯 Priority: 10/10
   ✅ Action Required: Yes
   ⏰ Deadline: Immediately
   📋 Actions: Join emergency call, Address server outage
   🏷️  Entities: 3 found

📧 EMAIL 3: Meeting Invitation: Q4 Planning Session...
   📊 Type: meeting_invitation
   🚨 Urgency: medium
   📝 Summary: Sarah Johnson has invited the team to a Q4 planning session...
   🎯 Priority: 5/10
   ✅ Action Required: Yes
   ⏰ Deadline: December 18th
   📋 Actions: Confirm attendance
   🏷️  Entities: 3 found

📈 SUMMARY
   Total emails processed: 4
   Requiring action: 3
   High/Critical urgency: 1
   Financial amounts found: 2
```

## Key LogiLLM Advantages Demonstrated

This tutorial showcases several LogiLLM advantages:

1. **Structured Output**: Native support for Pydantic models and complex types
2. **Modular Design**: Clean separation of concerns with composable modules
3. **Async/Await**: Modern Python concurrency patterns throughout
4. **Type Safety**: Complete type hints and validation
5. **Optimization**: Hybrid prompt and hyperparameter optimization
6. **Persistence**: Save and load optimized processors for production
7. **Zero Core Dependencies**: Framework works without external packages

## Next Steps

- **Add more email types** and refine classification (newsletter, promotional, etc.)
- **Integrate with email providers** (Gmail API, Outlook, IMAP)
- **Implement batch processing** for handling multiple emails efficiently
- **Add multilingual support** for international email processing
- **Create web interface** for interactive email analysis
- **Build workflow automation** based on extracted insights

## 🎓 What You've Learned

Excellent progress! You've now mastered structured data processing:

✅ **Pydantic Models**: Created type-safe data structures with validation  
✅ **Structured Extraction**: Converted free-form text to validated data models  
✅ **Field Validation**: Used descriptive fields for better LLM performance  
✅ **Classification Systems**: Built intelligent categorization logic  
✅ **Complex Data Types**: Handled dates, priorities, and nested structures

## 🚀 What's Next?

### Immediate Next Steps
**Ready for multi-step processing?** → **[Code Generation Tutorial](./code-generation.md)**  
Learn how to combine structured data extraction with iterative refinement and external API integration.

### Apply What You've Learned  
- **Process your own emails**: Export your email data and run it through the system
- **Add new email types**: Extend the classification to handle newsletters, invoices, etc.
- **Improve validation**: Add custom validators for phone numbers, addresses, etc.

### Advanced Extensions
- **Real-time processing**: Connect to email APIs (Gmail, Outlook) for live processing
- **Database integration**: Store extracted data in PostgreSQL or MongoDB  
- **Workflow automation**: Trigger actions based on extracted insights
- **Multi-language support**: Handle emails in different languages

### Tutorial Learning Path
1. ✅ **[LLM Text Generation](./llms-txt-generation.md)** - Foundation concepts
2. ✅ **Email Extraction** (You are here!)  
3. → **[Code Generation](./code-generation.md)** - Multi-step refinement
4. → **[Yahoo Finance ReAct](./yahoo-finance-react.md)** - Agent reasoning  
5. → **[AI Text Game](./ai-text-game.md)** - Interactive systems
6. → **[Memory-Enhanced ReAct Agent](./memory-enhanced-react.md)** - Persistent memory

### Concept Connections
- **From Email Extraction to Code Generation**: Take the structured processing you learned and apply it to iterative, self-improving systems
- **Validation Patterns**: The field validation techniques transfer directly to API response processing
- **Data Models**: Pydantic patterns you learned scale to complex agent state management

## 🛠️ Running the Tutorial

```bash
# With OpenAI
export OPENAI_API_KEY="your-key-here"
uv run --with logillm --with openai --with pydantic python -m examples.tutorials.email_extraction.demo

# With Anthropic
export ANTHROPIC_API_KEY="your-key-here"
uv run --with logillm --with anthropic --with pydantic python -m examples.tutorials.email_extraction.demo

# Run tests to verify your setup
uv run --with logillm --with openai --with pydantic python examples/tutorials/email_extraction/test_tutorial.py
```

---

**📚 [← LLM Text Generation](./llms-txt-generation.md) | [Tutorial Index](./README.md) | [Code Generation →](./code-generation.md)**

You've mastered structured data processing! Ready to learn multi-step refinement? Continue with **[Code Generation](./code-generation.md)** to build on these concepts.