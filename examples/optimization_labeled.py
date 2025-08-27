#!/usr/bin/env python3
"""LabeledFewShot: Simple Baseline with Hand-Crafted Examples.

LabeledFewShot represents the traditional approach - manually curated
few-shot examples with human-labeled data. This serves as a baseline
for comparing against automated optimization methods.

Key features:
- Hand-crafted example selection
- Human-labeled training data
- Simple nearest-neighbor selection
- Quality-first approach without automation
"""

import asyncio
import os

from logillm.core.predict import Predict
from logillm.core.signatures import InputField, OutputField, Signature
from logillm.optimizers import LabeledFewShot
from logillm.providers import create_provider, register_provider


class CustomerSupport(Signature):
    """Classify and respond to customer support tickets."""

    customer_message: str = InputField(desc="Customer's support request")
    urgency_level: str = InputField(desc="Customer's stated urgency: low, medium, high")
    category: str = OutputField(desc="Issue category: billing, technical, account, general")
    priority: str = OutputField(desc="Support priority: p1, p2, p3, p4")
    response: str = OutputField(desc="Professional customer service response")
    escalation_needed: bool = OutputField(desc="Whether to escalate to specialist")


async def main():
    """Demonstrate traditional labeled few-shot approach as baseline."""

    if not os.getenv("OPENAI_API_KEY"):
        print("Please set: export OPENAI_API_KEY=your_key")
        return

    print("=== LabeledFewShot: Traditional Hand-Crafted Approach ===\n")

    # Use smaller model for consistent comparison
    provider = create_provider("openai", model="gpt-4.1-nano")
    register_provider(provider, set_default=True)

    support_agent = Predict(CustomerSupport)

    # Carefully hand-crafted labeled examples (traditional approach)
    labeled_data = [
        {
            "inputs": {
                "customer_message": "I can't log into my account and I have an important presentation in 20 minutes! This is urgent!",
                "urgency_level": "high",
            },
            "outputs": {
                "category": "technical",
                "priority": "p1",
                "response": "I understand this is urgent given your presentation. Let me help you regain access immediately. Can you verify your email address and try our password reset option?",
                "escalation_needed": "false",
            },
        },
        {
            "inputs": {
                "customer_message": "My credit card was charged twice for last month's subscription. I need this fixed.",
                "urgency_level": "medium",
            },
            "outputs": {
                "category": "billing",
                "priority": "p2",
                "response": "I sincerely apologize for the duplicate charge. I'll investigate this billing issue right away and ensure you're refunded for the extra charge within 24 hours.",
                "escalation_needed": "true",
            },
        },
        {
            "inputs": {
                "customer_message": "The app keeps crashing every time I try to upload a file larger than 10MB.",
                "urgency_level": "medium",
            },
            "outputs": {
                "category": "technical",
                "priority": "p2",
                "response": "Thank you for reporting this technical issue. File upload crashes are frustrating. I'll create a bug report for our engineering team and provide a workaround solution.",
                "escalation_needed": "true",
            },
        },
        {
            "inputs": {
                "customer_message": "I'd like to upgrade my plan to include more storage space. What options do you have?",
                "urgency_level": "low",
            },
            "outputs": {
                "category": "account",
                "priority": "p3",
                "response": "Great question! I'd be happy to help you upgrade. We have several plans with increased storage. Let me walk you through the options that best fit your needs.",
                "escalation_needed": "false",
            },
        },
        {
            "inputs": {
                "customer_message": "How do I export my data in case I want to switch to another service?",
                "urgency_level": "low",
            },
            "outputs": {
                "category": "general",
                "priority": "p4",
                "response": "I understand you want to know about data export options. We support data portability - I can guide you through our export tools to download your information in standard formats.",
                "escalation_needed": "false",
            },
        },
        {
            "inputs": {
                "customer_message": "SYSTEM IS DOWN! ALL MY WORK IS GONE! FIX THIS NOW!!!",
                "urgency_level": "high",
            },
            "outputs": {
                "category": "technical",
                "priority": "p1",
                "response": "I understand your concern about your work and the system issue. This requires immediate attention from our technical team. I'm escalating this right now to our senior engineers.",
                "escalation_needed": "true",
            },
        },
        {
            "inputs": {
                "customer_message": "I noticed a small UI glitch in the settings page - the button is slightly misaligned.",
                "urgency_level": "low",
            },
            "outputs": {
                "category": "technical",
                "priority": "p4",
                "response": "Thank you for the detailed bug report about the UI alignment. While this doesn't affect functionality, we appreciate users who help us maintain quality. I'll add this to our improvement backlog.",
                "escalation_needed": "false",
            },
        },
    ]

    # Test cases representing different types of support requests
    test_cases = [
        {
            "customer_message": "I can't find my invoice from last month and need it for expense reporting.",
            "urgency_level": "medium",
        },
        {
            "customer_message": "The website won't load on my mobile device but works fine on desktop.",
            "urgency_level": "low",
        },
        {
            "customer_message": "CRITICAL: Payment processing is failing for all customers!",
            "urgency_level": "high",
        },
        {
            "customer_message": "I love the new feature you added. Any plans for more improvements?",
            "urgency_level": "low",
        },
    ]

    # Baseline test (no examples, just the signature)
    print("📊 BASELINE (no examples):")
    print("-" * 40)
    baseline_quality = []

    for test in test_cases:
        result = await support_agent(**test)
        category = result.outputs.get("category", "general")
        priority = result.outputs.get("priority", "p3")
        response = result.outputs.get("response", "")
        escalation = result.outputs.get("escalation_needed", "false")

        # Simple quality scoring based on response length and appropriateness
        response_quality = min(1.0, len(response.split()) / 25)  # Reward detailed responses

        # Contextual appropriateness
        context_score = 0.5  # Base score
        if test["urgency_level"] == "high" and priority in ["p1", "p2"]:
            context_score += 0.3
        elif test["urgency_level"] == "low" and priority in ["p3", "p4"]:
            context_score += 0.3
        elif test["urgency_level"] == "medium" and priority in ["p2", "p3"]:
            context_score += 0.3

        overall_quality = (response_quality + context_score) / 2
        baseline_quality.append(overall_quality)

        print(f"Message: '{test['customer_message'][:50]}...'")
        print(f"  → {category} / {priority} / escalate: {escalation}")
        print(f"  Response: '{response[:60]}...'")

    baseline_avg = sum(baseline_quality) / len(baseline_quality)
    print(f"\nBaseline quality: {baseline_avg:.2f}")

    # Define metric focusing on support quality
    def support_quality_metric(pred, ref=None):
        """Reward appropriate categorization and helpful responses."""
        category = pred.get("category", "general")
        response = pred.get("response", "")
        escalation = pred.get("escalation_needed", "false")

        # Response quality (length and helpfulness indicators)
        response_words = response.split()
        response_score = min(0.4, len(response_words) / 40)

        # Professional language bonus
        professional_words = ["apologize", "understand", "help", "assist", "resolve", "thank"]
        professional_bonus = sum(0.02 for word in professional_words if word in response.lower())

        # Escalation appropriateness (bonus for escalating complex issues)
        escalation_bonus = 0.0
        if escalation.lower() == "true" and category in ["billing", "technical"]:
            escalation_bonus = 0.1
        elif escalation.lower() == "false" and category in ["general", "account"]:
            escalation_bonus = 0.05

        return min(1.0, response_score + professional_bonus + escalation_bonus)

    # Apply labeled few-shot optimization
    print("\n📚 LABELED FEW-SHOT (hand-crafted examples)...")

    optimizer = LabeledFewShot(
        metric=support_quality_metric,
        max_examples=5,  # Use 5 best hand-crafted examples
        selection_strategy="quality",  # Select highest quality examples
        quality_threshold=0.8,  # Only use high-quality examples
        diversity_weight=0.2,  # Some diversity, but quality first
    )

    result = await optimizer.optimize(
        module=support_agent,
        dataset=labeled_data,
    )

    labeled_optimized = result.optimized_module

    # Test with labeled examples
    print("\n📊 LABELED FEW-SHOT optimized:")
    print("-" * 40)
    labeled_quality = []

    for test in test_cases:
        result = await labeled_optimized(**test)
        category = result.outputs.get("category", "general")
        priority = result.outputs.get("priority", "p3")
        response = result.outputs.get("response", "")
        escalation = result.outputs.get("escalation_needed", "false")

        # Same quality scoring
        response_quality = min(1.0, len(response.split()) / 25)

        context_score = 0.5
        if test["urgency_level"] == "high" and priority in ["p1", "p2"]:
            context_score += 0.3
        elif test["urgency_level"] == "low" and priority in ["p3", "p4"]:
            context_score += 0.3
        elif test["urgency_level"] == "medium" and priority in ["p2", "p3"]:
            context_score += 0.3

        overall_quality = (response_quality + context_score) / 2
        labeled_quality.append(overall_quality)

        print(f"Message: '{test['customer_message'][:50]}...'")
        print(f"  → {category} / {priority} / escalate: {escalation}")
        print(f"  Response: '{response[:60]}...'")

    labeled_avg = sum(labeled_quality) / len(labeled_quality)
    improvement = ((labeled_avg - baseline_avg) / max(baseline_avg, 0.01)) * 100

    # Results
    print("\n" + "=" * 50)
    print("📈 LABELED FEW-SHOT RESULTS:")
    print(f"  Baseline quality:    {baseline_avg:.2f}")
    print(f"  Labeled few-shot:    {labeled_avg:.2f}")
    print(f"  Improvement:         {improvement:+.1f}%")

    # Show selected examples
    if hasattr(labeled_optimized, "demo_manager") and labeled_optimized.demo_manager.demos:
        print(f"\n📚 Selected labeled examples ({len(labeled_optimized.demo_manager.demos)}):")
        for i, demo in enumerate(labeled_optimized.demo_manager.demos[:3], 1):
            message = demo.inputs.get("customer_message", "")[:40]
            category = demo.outputs.get("category", "")
            priority = demo.outputs.get("priority", "")
            print(f"  {i}. '{message}...' → {category}/{priority}")

    # Show optimization details
    if hasattr(result, "metadata"):
        print("\n🎯 Selection process:")
        if "examples_evaluated" in result.metadata:
            print(f"  Examples evaluated:  {result.metadata['examples_evaluated']}")
        if "quality_threshold_met" in result.metadata:
            print(f"  Met quality threshold: {result.metadata['quality_threshold_met']}")
        if "selection_strategy" in result.metadata:
            print(f"  Selection strategy:    {result.metadata['selection_strategy']}")

    print("\n✨ Key insight: Traditional labeled few-shot provides a solid")
    print("   baseline with human expertise, but is limited by manual")
    print("   curation. It's the foundation that other methods build upon.")


if __name__ == "__main__":
    asyncio.run(main())
