"""
LogiLLM tutorial: Email information extraction.
"""

from .demo import run_email_processing_demo
from .models import EmailInsight, EmailType, ExtractedEntity, UrgencyLevel
from .processor import EmailProcessor

__all__ = [
    "EmailProcessor",
    "EmailInsight",
    "EmailType",
    "UrgencyLevel",
    "ExtractedEntity",
    "run_email_processing_demo",
]
