#!/usr/bin/env python3
"""BootstrapFewShot: Teacher-Student Learning.

BootstrapFewShot implements teacher-student learning where a stronger
model (teacher) generates high-quality examples that are used to train
a weaker but faster model (student). This enables cost-effective deployment.

Key features:
- Teacher model generates diverse, high-quality examples
- Student model learns from teacher's demonstrations
- Automatic quality filtering and diversity selection
- Cost optimization through model hierarchy
"""

import asyncio
import os

from logillm.core.predict import Predict
from logillm.core.signatures import InputField, OutputField, Signature
from logillm.optimizers import BootstrapFewShot
from logillm.providers import create_provider, register_provider


class EmailClassification(Signature):
    """Classify emails into categories."""

    email_subject: str = InputField(desc="Email subject line")
    email_body: str = InputField(desc="Email body content")
    category: str = OutputField(desc="Category: urgent, promotional, social, or work")
    priority: str = OutputField(desc="Priority: high, medium, or low")
    reason: str = OutputField(desc="Brief explanation for classification")


async def main():
    """Demonstrate teacher-student bootstrap learning."""

    if not os.getenv("OPENAI_API_KEY"):
        print("Please set: export OPENAI_API_KEY=your_key")
        return

    print("=== BootstrapFewShot: Teacher-Student Learning ===\n")

    # Teacher: Use stronger model for generating examples
    teacher_provider = create_provider("openai", model="gpt-4.1")  # Stronger

    # Student: Use smaller/faster model for actual deployment
    student_provider = create_provider("openai", model="gpt-4.1-nano")  # Weaker but faster

    register_provider(student_provider, set_default=True)  # Student is default

    # Create student classifier
    classifier = Predict(EmailClassification)

    # Minimal seed data - teacher will bootstrap from this
    seed_data = [
        {
            "inputs": {
                "email_subject": "URGENT: Server Down - Action Required",
                "email_body": "Our main server is experiencing critical issues. Please respond immediately.",
            },
            "outputs": {
                "category": "work",
                "priority": "high",
                "reason": "Critical work emergency requiring immediate action",
            },
        },
        {
            "inputs": {
                "email_subject": "50% Off Everything - Limited Time!",
                "email_body": "Don't miss out on our biggest sale of the year. Shop now before it's too late!",
            },
            "outputs": {
                "category": "promotional",
                "priority": "low",
                "reason": "Commercial advertising with discount offer",
            },
        },
        {
            "inputs": {
                "email_subject": "Happy Birthday! 🎉",
                "email_body": "Hope you have a wonderful day celebrating with family and friends!",
            },
            "outputs": {
                "category": "social",
                "priority": "medium",
                "reason": "Personal social message with birthday wishes",
            },
        },
    ]

    # Test cases for evaluation
    test_cases = [
        {
            "email_subject": "Meeting Cancelled - Please Reschedule",
            "email_body": "Due to unforeseen circumstances, today's project meeting is cancelled. Please coordinate with your team to reschedule.",
        },
        {
            "email_subject": "Your friend tagged you in a photo",
            "email_body": "Sarah tagged you in a photo from last weekend's hiking trip. Click here to see it!",
        },
        {
            "email_subject": "Final Notice: Payment Due",
            "email_body": "This is your final notice. Your account is past due and requires immediate payment to avoid service interruption.",
        },
        {
            "email_subject": "Weekly Newsletter - Tech Updates",
            "email_body": "Here are this week's top tech news stories and product updates from our team.",
        },
    ]

    # Baseline test with minimal examples
    print("📊 BASELINE (student with minimal examples):")
    print("-" * 40)
    baseline_correct = 0

    for i, test in enumerate(test_cases, 1):
        result = await classifier(**test)
        category = result.outputs.get("category", "unknown")
        priority = result.outputs.get("priority", "medium")
        reason = result.outputs.get("reason", "")[:50]

        print(f"{i}. Subject: '{test['email_subject'][:40]}...'")
        print(f"   → {category} / {priority}")
        print(f"   Reason: {reason}...")

        # Simple heuristic scoring (in real use, you'd have labels)
        if (
            "urgent" in test["email_subject"].lower()
            or "final notice" in test["email_subject"].lower()
        ) and priority == "high":
            baseline_correct += 1
        elif "tagged" in test["email_subject"].lower() and category == "social":
            baseline_correct += 1
        elif "newsletter" in test["email_subject"].lower() and category == "work":
            baseline_correct += 1
        elif "meeting" in test["email_subject"].lower() and category == "work":
            baseline_correct += 1

    baseline_accuracy = baseline_correct / len(test_cases)
    print(f"\nBaseline accuracy: {baseline_accuracy:.2f}")

    # Define metric for classification quality
    def classification_metric(pred, ref=None):
        """Simple metric based on category consistency."""
        category = pred.get("category", "").lower()
        priority = pred.get("priority", "").lower()
        reason = pred.get("reason", "")

        # Reward detailed reasoning
        reasoning_score = min(1.0, len(reason.split()) / 10)

        # Basic consistency checks
        consistency_score = 0.5
        if category in ["urgent", "work"] and priority == "high":
            consistency_score += 0.3
        if category == "promotional" and priority == "low":
            consistency_score += 0.3
        if category == "social" and priority in ["low", "medium"]:
            consistency_score += 0.2

        return (reasoning_score + consistency_score) / 2

    # Bootstrap with teacher-student learning
    print("\n🎓 BOOTSTRAPPING with teacher-student learning...")

    optimizer = BootstrapFewShot(
        metric=classification_metric,
        teacher_lm=teacher_provider,  # Strong teacher model
        max_bootstrapped_demos=8,  # Generate 8 examples
        max_labeled_demos=3,  # Use 3 seed examples
        teacher_settings={
            "temperature": 0.7,  # Creative teacher
            "max_tokens": 200,
        },
    )

    result = await optimizer.optimize(
        module=classifier,
        dataset=seed_data,
    )

    bootstrapped_classifier = result.optimized_module

    # Test bootstrapped version
    print("\n📊 BOOTSTRAPPED (student learned from teacher):")
    print("-" * 40)
    bootstrap_correct = 0

    for i, test in enumerate(test_cases, 1):
        result = await bootstrapped_classifier(**test)
        category = result.outputs.get("category", "unknown")
        priority = result.outputs.get("priority", "medium")
        reason = result.outputs.get("reason", "")[:50]

        print(f"{i}. Subject: '{test['email_subject'][:40]}...'")
        print(f"   → {category} / {priority}")
        print(f"   Reason: {reason}...")

        # Same heuristic scoring
        if (
            "urgent" in test["email_subject"].lower()
            or "final notice" in test["email_subject"].lower()
        ) and priority == "high":
            bootstrap_correct += 1
        elif "tagged" in test["email_subject"].lower() and category == "social":
            bootstrap_correct += 1
        elif "newsletter" in test["email_subject"].lower() and category == "work":
            bootstrap_correct += 1
        elif "meeting" in test["email_subject"].lower() and category == "work":
            bootstrap_correct += 1

    bootstrap_accuracy = bootstrap_correct / len(test_cases)
    improvement = ((bootstrap_accuracy - baseline_accuracy) / max(baseline_accuracy, 0.01)) * 100

    # Results
    print("\n" + "=" * 45)
    print("📈 RESULTS:")
    print(f"  Baseline accuracy:     {baseline_accuracy:.2f}")
    print(f"  Bootstrapped accuracy: {bootstrap_accuracy:.2f}")
    print(f"  Improvement:           {improvement:+.1f}%")

    # Show bootstrap stats
    if hasattr(result, "metadata"):
        print("\n🎓 Bootstrap process:")
        if "bootstrapped_demos" in result.metadata:
            print(f"  Examples generated:    {result.metadata['bootstrapped_demos']}")
        if "teacher_calls" in result.metadata:
            print(f"  Teacher model calls:   {result.metadata['teacher_calls']}")
        if "quality_threshold" in result.metadata:
            print(f"  Quality threshold:     {result.metadata['quality_threshold']:.2f}")

    # Show learned examples
    if (
        hasattr(bootstrapped_classifier, "demo_manager")
        and bootstrapped_classifier.demo_manager.demos
    ):
        print(
            f"\n📚 Teacher generated examples ({len(bootstrapped_classifier.demo_manager.demos)}):"
        )
        for i, demo in enumerate(bootstrapped_classifier.demo_manager.demos[:2], 1):
            subject = demo.inputs.get("email_subject", "")[:30]
            category = demo.outputs.get("category", "")
            print(f"  {i}. '{subject}...' → {category}")

    print("\n✨ Key insight: Teacher-student bootstrap learning enables")
    print("   weaker models to achieve strong performance by learning")
    print("   from high-quality examples generated by stronger models.")


if __name__ == "__main__":
    asyncio.run(main())
