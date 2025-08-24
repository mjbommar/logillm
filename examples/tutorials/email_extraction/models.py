"""
Data models for email extraction tutorial.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


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
    key_entities: list[ExtractedEntity]
    action_required: bool
    deadline: Optional[str] = None
    financial_amount: Optional[float] = None
    sender_info: Optional[str] = None
    priority_score: int = Field(ge=1, le=10, description="Priority from 1-10")
    action_items: list[str] = Field(default_factory=list)
    reasoning: Optional[str] = None
