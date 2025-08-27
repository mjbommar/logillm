"""
Email processor module using LogiLLM.
"""

import json
import re

from logillm.core.modules import Module
from logillm.core.predict import Predict

from .models import EmailInsight, EmailType, ExtractedEntity, UrgencyLevel
from .signatures import ClassifyEmail, ExtractEntities, GenerateActionItems, SummarizeEmail


class EmailProcessor(Module):
    """A comprehensive email processing system using LogiLLM."""

    def __init__(self) -> None:
        super().__init__()

        # Initialize our processing components
        self.classifier = Predict(signature=ClassifyEmail)
        self.entity_extractor = Predict(signature=ExtractEntities)
        self.action_generator = Predict(signature=GenerateActionItems)
        self.summarizer = Predict(signature=SummarizeEmail)

    def _parse_financial_amount(self, text: str) -> float | None:
        """Extract financial amount from text like '$2,399.00' or 'No amount found'."""
        if not text or "no" in text.lower() or "none" in text.lower():
            return None

        # Find numbers with dollar signs, commas, periods
        match = re.search(r"\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)", text)
        if match:
            # Remove commas and convert to float
            amount_str = match.group(1).replace(",", "")
            try:
                return float(amount_str)
            except ValueError:
                return None
        return None

    def _parse_boolean(self, text: str) -> bool:
        """Parse boolean from text like 'Yes', 'No', 'true', 'false', etc."""
        if not text:
            return False
        text = text.lower().strip()
        return text in ["yes", "true", "1", "required", "needed"]

    def _parse_string_list(self, text: str) -> list[str]:
        """Parse list from text - try JSON first, then comma-separated."""
        if not text:
            return []

        text = text.strip()

        # Try JSON array format
        if text.startswith("[") and text.endswith("]"):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass

        # Try comma-separated values
        if "," in text:
            return [item.strip().strip("\"'") for item in text.split(",")]

        # Single item or sentence - return as single item
        return [text] if text else []

    def _parse_entities(self, text: str) -> list[ExtractedEntity]:
        """Parse entities from text - try JSON first, then fallback parsing."""
        if not text:
            return []

        text = text.strip()

        # Try JSON array format first
        if text.startswith("[") and text.endswith("]"):
            try:
                data = json.loads(text)
                entities = []
                for item in data:
                    if isinstance(item, dict):
                        entities.append(
                            ExtractedEntity(
                                entity_type=item.get("entity_type", "unknown"),
                                value=item.get("value", ""),
                                confidence=float(item.get("confidence", 0.5)),
                                context=item.get("context"),
                            )
                        )
                return entities
            except (json.JSONDecodeError, ValueError, KeyError):
                pass

        # Fallback: parse comma-separated entities with default confidence
        items = [item.strip() for item in text.split(",")]
        entities = []
        for item in items:
            if item:
                # Try to guess entity type
                entity_type = "person" if any(c.isupper() for c in item[:3]) else "other"
                entities.append(
                    ExtractedEntity(
                        entity_type=entity_type, value=item, confidence=0.7, context=None
                    )
                )
        return entities

    async def forward(self, email_subject: str, email_body: str, sender: str = "") -> EmailInsight:
        """Process an email and extract structured information."""

        # Step 1: Classify the email
        classification = await self.classifier(
            email_subject=email_subject, email_body=email_body, sender=sender
        )

        # Step 2: Extract entities
        full_content = f"Subject: {email_subject}\n\nFrom: {sender}\n\n{email_body}"
        entities = await self.entity_extractor(
            email_content=full_content, email_type=classification.email_type
        )

        # Step 3: Generate summary
        summary = await self.summarizer(
            email_subject=email_subject, email_body=email_body, key_entities=entities.key_entities
        )

        # Step 4: Determine actions
        actions = await self.action_generator(
            email_type=classification.email_type,
            urgency=classification.urgency,
            email_summary=summary.summary,
            extracted_entities=entities.key_entities,
        )

        # Step 5: Convert strings to enums and structure the results
        try:
            email_type_enum = EmailType(classification.email_type.lower().replace(" ", "_"))
        except ValueError:
            email_type_enum = EmailType.OTHER

        try:
            urgency_enum = UrgencyLevel(classification.urgency.lower())
        except ValueError:
            urgency_enum = UrgencyLevel.LOW

        # Parse all fields with proper type conversion
        parsed_entities = self._parse_entities(str(entities.key_entities))
        parsed_financial = self._parse_financial_amount(str(entities.financial_amount))
        parsed_action_required = self._parse_boolean(str(actions.action_required))
        parsed_action_items = self._parse_string_list(str(actions.action_items))

        return EmailInsight(
            email_type=email_type_enum,
            urgency=urgency_enum,
            summary=summary.summary,
            key_entities=parsed_entities,
            financial_amount=parsed_financial,
            action_required=parsed_action_required,
            action_items=parsed_action_items,
            deadline=actions.deadline,
            priority_score=actions.priority_score,
            reasoning=classification.reasoning,
            sender_info=sender,
        )
