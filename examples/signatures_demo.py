#!/usr/bin/env python3
"""LogiLLM Signatures: From simple to powerful.
For detailed tutorial: examples/few_shot.py
"""

import asyncio

from logillm.core.predict import ChainOfThought, Predict
from logillm.core.signatures import InputField, OutputField, Signature
from logillm.optimizers import BootstrapFewShot
from logillm.providers import create_provider, register_provider

# Setup
provider = create_provider("openai", model="gpt-4.1-mini")
register_provider(provider, set_default=True)


async def main():
    # 1. Simple string signature
    qa = Predict("question -> answer")
    result = await qa(question="What is 2+2?")
    print(f"Simple: {result.outputs['answer']}")

    # 2. Typed string signature
    math = ChainOfThought("problem: str -> reasoning: str, answer: int")
    result = await math(problem="If I have 23 apples and eat 5, how many remain?")
    print(f"Typed: {result.outputs['answer']} (type: {type(result.outputs['answer'])})")

    # 3. Class-based signature with descriptions
    class EmailClassifier(Signature):
        """Classify customer emails by intent.
        Be specific: use 'billing', 'support', 'complaint', or 'other'."""

        email: str = InputField(desc="Customer email text")
        intent: str = OutputField(desc="Primary intent category")
        confidence: float = OutputField(desc="Confidence score 0.0-1.0")

    classifier = Predict(EmailClassifier)
    result = await classifier(email="My order hasn't arrived and I'm very upset!")
    print(f"Class: intent={result.outputs['intent']}, confidence={result.outputs['confidence']}")

    # 4. Multi-step reasoning with typed signature
    class DebugSignature(Signature):
        """Debug Python code and suggest fixes."""

        code: str = InputField(desc="Python code with bugs")
        error: str = InputField(desc="Error message received")
        analysis: str = OutputField(desc="What's wrong with the code")
        fix: str = OutputField(desc="Corrected code")
        explanation: str = OutputField(desc="Why this fixes it")

    debugger = ChainOfThought(DebugSignature)
    result = await debugger(
        code="def add(a, b):\n    return a + c", error="NameError: name 'c' is not defined"
    )
    print(f"Debug: {result.outputs['analysis'][:50]}...")

    # 5. Optimize any signature-based module
    data = [
        {
            "inputs": {"email": "Bill me twice?"},
            "outputs": {"intent": "billing", "confidence": 0.9},
        },
        {"inputs": {"email": "Can't login"}, "outputs": {"intent": "support", "confidence": 0.95}},
    ]

    from logillm.core.optimizers import Metric

    class IntentMetric(Metric):
        def __call__(self, pred, ref, **kwargs):
            return 1.0 if pred.get("intent") == ref.get("intent") else 0.0

        def name(self):
            return "intent_accuracy"

    optimizer = BootstrapFewShot(metric=IntentMetric(), max_bootstrapped_demos=2)
    optimized = (await optimizer.optimize(classifier, dataset=data)).optimized_module

    result = await optimized(email="Why was I charged twice?")
    print(f"Optimized: {result.outputs['intent']} (after learning from examples)")


if __name__ == "__main__":
    asyncio.run(main())
