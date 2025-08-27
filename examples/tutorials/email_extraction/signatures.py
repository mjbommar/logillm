"""
LogiLLM signatures for email processing.
"""

from typing import Optional

from logillm.core.signatures import InputField, OutputField, Signature

from .models import EmailType, ExtractedEntity, UrgencyLevel


class ClassifyEmail(Signature):
    """Classify the type and urgency of an email based on its content.

    IMPORTANT: Return response as JSON with exact field names and enum values."""

    email_subject: str = InputField(desc="The subject line of the email")
    email_body: str = InputField(desc="The main content of the email")
    sender: str = InputField(desc="Email sender information")

    email_type: str = OutputField(
        desc="Classify email type. Return ONLY ONE of these exact values: order_confirmation, support_request, meeting_invitation, newsletter, promotional, invoice, shipping_notification, urgent_alert, other"
    )
    urgency: str = OutputField(
        desc="Classify urgency level. Return ONLY ONE of these exact values: low, medium, high, critical"
    )
    reasoning: str = OutputField(desc="Brief explanation of the classification")


class ExtractEntities(Signature):
    """Extract key entities and information from email content."""

    email_content: str = InputField(desc="The full email content including subject and body")
    email_type: EmailType = InputField(desc="The classified type of email")

    key_entities: list[ExtractedEntity] = OutputField(
        desc='List of extracted entities in JSON format: [{"entity_type": "person", "value": "John", "confidence": 0.95, "context": "From John Smith"}]'
    )
    financial_amount: Optional[float] = OutputField(desc="Any monetary amounts found")
    important_dates: list[str] = OutputField(
        desc='List of important dates found. Format as JSON array: ["2024-12-15", "2024-12-20"]'
    )
    contact_info: list[str] = OutputField(
        desc='Relevant contact information extracted. Format as JSON array: ["support@example.com", "+1-555-123-4567"]'
    )


class GenerateActionItems(Signature):
    """Determine what actions are needed based on email content."""

    email_type: EmailType = InputField()
    urgency: UrgencyLevel = InputField()
    email_summary: str = InputField(desc="Brief summary of the email content")
    extracted_entities: list[ExtractedEntity] = InputField(desc="Key entities found")

    action_required: bool = OutputField(
        desc="Whether any action is required. Must be either true or false"
    )
    action_items: list[str] = OutputField(
        desc='List of specific actions needed. Format as JSON array: ["action1", "action2"]'
    )
    deadline: Optional[str] = OutputField(desc="Deadline for action if applicable")
    priority_score: int = OutputField(desc="Priority score from 1-10")


class SummarizeEmail(Signature):
    """Create a concise summary of the email content."""

    email_subject: str = InputField()
    email_body: str = InputField()
    key_entities: list[ExtractedEntity] = InputField()

    summary: str = OutputField(desc="A 2-3 sentence summary of the email's main points")
