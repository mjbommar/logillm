#!/usr/bin/env python3
"""LogiLLM Classification Example.

This example demonstrates how to build a text classifier using LogiLLM.
We'll classify text into different categories using both simple and complex signatures.

Prerequisites:
- OpenAI API key: export OPENAI_API_KEY=your_key
- Install LogiLLM with OpenAI support: pip install logillm[openai]
"""

import asyncio
import os

from logillm.core.extractors import Extractors
from logillm.core.predict import Predict
from logillm.core.signatures import InputField, OutputField, Signature
from logillm.providers import create_provider, register_provider


class SentimentClassifier(Signature):
    """Classify the sentiment of text."""

    text: str = InputField(desc="Text to analyze for sentiment")
    sentiment: str = OutputField(desc="Sentiment: positive, negative, or neutral")
    confidence: float = OutputField(desc="Confidence score from 0 to 1")


class TopicClassifier(Signature):
    """Classify text into predefined topics."""

    text: str = InputField(desc="Text to classify")
    available_topics: list[str] = InputField(desc="List of available topic categories")
    topic: str = OutputField(desc="Most relevant topic from the available topics")
    relevance_score: float = OutputField(desc="Relevance score from 0 to 1")


class MultiLabelClassifier(Signature):
    """Classify text with multiple labels."""

    text: str = InputField(desc="Text to classify")
    labels: list[str] = OutputField(desc="All applicable labels")
    primary_label: str = OutputField(desc="The most relevant label")
    explanation: str = OutputField(desc="Brief explanation of the classification")


async def main():
    """Run classification examples."""
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY=your_key")
        return

    print("=== LogiLLM Classification Examples ===\n")

    try:
        # Set up provider
        provider = create_provider("openai", model="gpt-4.1-mini")
        register_provider(provider, set_default=True)

        # 1. Simple sentiment classification
        print("1. Sentiment Classification")
        print("-" * 30)

        sentiment_classifier = Predict(SentimentClassifier)

        test_texts = [
            "I absolutely love this product! It exceeded my expectations.",
            "This is terrible. Complete waste of money.",
            "It's okay, nothing special but does the job.",
        ]

        for text in test_texts:
            result = await sentiment_classifier(text=text)
            print(f"Text: {text[:50]}...")
            print(f"Sentiment: {result.sentiment} (confidence: {result.confidence:.2f})")
            print()

        # 2. Topic classification with predefined categories
        print("\n2. Topic Classification")
        print("-" * 30)

        topic_classifier = Predict(TopicClassifier)

        topics = ["Technology", "Sports", "Politics", "Entertainment", "Science", "Business"]

        test_cases = [
            "The new quantum computer achieved breakthrough performance in cryptography tests.",
            "The team won the championship after an incredible comeback in the final quarter.",
            "Stock markets rallied today following positive earnings reports from major tech companies.",
        ]

        for text in test_cases:
            result = await topic_classifier(text=text, available_topics=topics)
            # Use enum extractor to ensure valid topic
            topic = Extractors.enum(result.topic, options=topics, default=topics[0])
            # Use percentage extractor for robust score parsing
            relevance = Extractors.percentage(
                str(result.relevance_score), as_decimal=True, default=0.5
            )
            print(f"Text: {text[:60]}...")
            print(f"Topic: {topic} (relevance: {relevance:.2f})")
            print()

        # 3. Multi-label classification
        print("\n3. Multi-Label Classification")
        print("-" * 30)

        multi_classifier = Predict(MultiLabelClassifier)

        complex_text = """
        The tech startup announced a $50M funding round led by prominent VCs,
        planning to use the capital for AI research and expanding their engineering team.
        """

        result = await multi_classifier(text=complex_text)
        print(f"Text: {complex_text.strip()[:80]}...")
        print(f"Primary Label: {result.primary_label}")
        print(f"All Labels: {', '.join(result.labels)}")
        print(f"Explanation: {result.explanation}")

        # 4. Zero-shot classification (no training examples)
        print("\n\n4. Zero-shot Classification")
        print("-" * 30)

        # Simple string signature for quick classification
        zero_shot = Predict("email_subject -> spam_or_ham")

        emails = [
            "Meeting tomorrow at 3pm to discuss Q4 projections",
            "You've won $1000000! Click here to claim your prize!!!",
            "Your order #12345 has been shipped and will arrive tomorrow",
        ]

        for email in emails:
            result = await zero_shot(email_subject=email)
            classification = result.outputs.get("spam_or_ham", "unknown")
            print(f"Email: {email[:50]}...")
            print(f"Classification: {classification}")
            print()

        print("\n=== Classification Examples Complete ===")

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have set your OPENAI_API_KEY environment variable.")


if __name__ == "__main__":
    asyncio.run(main())
