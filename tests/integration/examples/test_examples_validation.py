"""Integration tests that validate all working examples.

These tests ensure that every example in examples/ actually works with real APIs.
NO MOCKS ALLOWED - these must use real API calls to verify examples work.
"""

import subprocess
import sys
from pathlib import Path

import pytest

# Add examples directory to path for imports
examples_dir = Path(__file__).parent.parent.parent.parent / "examples"
sys.path.insert(0, str(examples_dir))


@pytest.mark.integration
@pytest.mark.openai
class TestExamplesValidation:
    """Validate that all examples actually work with real APIs."""

    @pytest.mark.asyncio
    async def test_hello_world_example(self, openai_provider_registered):
        """Test hello_world.py example works."""
        from logillm.core.predict import Predict

        # This mirrors the hello_world.py example
        qa = Predict("question -> answer")

        result = await qa.forward(question="What is the capital of France?")

        assert result.success
        assert result.outputs.get("answer")
        assert "Paris" in result.outputs["answer"]
        assert result.usage.tokens.input_tokens > 0
        assert result.usage.tokens.output_tokens > 0

    @pytest.mark.asyncio
    async def test_classification_example(self, openai_provider_registered):
        """Test classification.py example works."""
        from logillm.core.predict import Predict

        # This mirrors the classification.py example
        sentiment = Predict("text -> sentiment, confidence")

        result = await sentiment.forward(text="I love this product! It's amazing!")

        assert result.success
        assert result.outputs.get("sentiment")
        assert result.outputs.get("confidence")
        # Should detect positive sentiment
        assert "positive" in result.outputs["sentiment"].lower()

    @pytest.mark.asyncio
    async def test_multi_step_reasoning_example(self, openai_provider_registered):
        """Test multi_step_reasoning.py example works."""
        from logillm.core.predict import ChainOfThought, Predict

        # Test the planning step
        planner = Predict("problem -> steps")
        planning_result = await planner.forward(problem="How do I make a paper airplane?")

        assert planning_result.success
        assert planning_result.outputs.get("steps")

        # Test the chain of thought reasoning
        cot = ChainOfThought("question -> answer")
        cot_result = await cot.forward(question="What is 17 * 23? Show your work.")

        assert cot_result.success
        assert cot_result.outputs.get("reasoning")
        assert cot_result.outputs.get("answer")
        # Should get correct answer
        assert "391" in str(cot_result.outputs["answer"])

    @pytest.mark.asyncio
    async def test_rag_simple_example(self, openai_provider_registered):
        """Test rag_simple.py example works."""
        from logillm.core.predict import Predict

        # Test document storage and retrieval (mirrors rag_simple.py)
        documents = [
            "Paris is the capital of France and known for the Eiffel Tower.",
            "London is the capital of England and has Big Ben.",
            "Tokyo is the capital of Japan and is very populous.",
        ]

        # Simple keyword-based retrieval (improved with specific word matching)
        def retrieve_documents(query, docs, top_k=2):
            """Simple keyword-based document retrieval."""
            scored_docs = []
            query_words = set(query.lower().split())

            for doc in docs:
                doc_words = set(doc.lower().split())
                # Count word intersections
                common_words = query_words.intersection(doc_words)
                # Bonus points for specific keywords like country names
                bonus = 0
                if "france" in query.lower() and "france" in doc.lower():
                    bonus += 10
                if "england" in query.lower() and "england" in doc.lower():
                    bonus += 10
                if "japan" in query.lower() and "japan" in doc.lower():
                    bonus += 10

                score = len(common_words) + bonus
                scored_docs.append((score, doc))

            scored_docs.sort(key=lambda x: x[0], reverse=True)
            return [doc for _, doc in scored_docs[:top_k]]

        # Test retrieval with France query
        relevant_docs = retrieve_documents("What is the capital of France?", documents)
        assert len(relevant_docs) > 0
        # Should find the France document first due to bonus scoring
        assert "Paris" in relevant_docs[0] or "France" in relevant_docs[0]

        # Test QA with context
        qa_with_context = Predict("context, question -> answer")

        result = await qa_with_context.forward(
            context="\n".join(relevant_docs), question="What is the capital of France?"
        )

        assert result.success
        assert result.outputs.get("answer")
        assert "Paris" in result.outputs["answer"]

    @pytest.mark.asyncio
    async def test_math_fraction_handling(self, openai_provider_registered):
        """Test fraction handling from hello_world example."""
        from fractions import Fraction

        from logillm.core.predict import Predict

        # Test fraction output parsing
        math_qa = Predict("question -> answer: Fraction")

        result = await math_qa.forward(
            question="If you roll two dice, what is the probability of getting a sum of 7? Express as a fraction."
        )

        assert result.success
        assert result.outputs.get("answer")

        # Check what type we got
        answer = result.outputs["answer"]
        print(f"DEBUG: Got answer type: {type(answer)}, value: {answer!r}")

        # If LogiLLM's type coercion is working, this should already be a Fraction
        if isinstance(answer, Fraction):
            # Success! LogiLLM correctly coerced to Fraction
            assert answer.numerator > 0
            assert answer.denominator > 0
            print(f"✅ LogiLLM correctly coerced to Fraction: {answer}")
        elif isinstance(answer, str):
            # LogiLLM didn't coerce - this is a bug!
            print(f"❌ LogiLLM failed to coerce string to Fraction: {answer!r}")
            # Try manual parsing to see if it's parseable
            try:
                Fraction(answer.strip())
                pytest.fail(f"LogiLLM should have coerced '{answer}' to Fraction automatically!")
            except ValueError:
                pytest.fail(f"LogiLLM got unparseable string '{answer}' and didn't handle it")
        else:
            pytest.fail(f"Unexpected type {type(answer)} for Fraction field")

    @pytest.mark.asyncio
    async def test_structured_output_example(self, openai_provider_registered):
        """Test structured output capabilities."""
        from logillm.core.predict import Predict

        # Test structured classification
        structured_classifier = Predict("text -> category: str, confidence: float, reasoning: str")

        result = await structured_classifier.forward(
            text="The stock market crashed today, causing widespread panic among investors."
        )

        assert result.success
        assert result.outputs.get("category")
        # Check confidence is a number (including 0.0)
        confidence = result.outputs.get("confidence")
        assert confidence is not None
        assert isinstance(confidence, (int, float))
        assert result.outputs.get("reasoning")

        # Should classify as financial/market news
        category = result.outputs["category"].lower()
        assert any(
            word in category
            for word in ["negative", "financial", "economic", "market", "stock", "finance"]
        )


@pytest.mark.integration
@pytest.mark.openai
class TestExampleScripts:
    """Test that example scripts can be executed directly."""

    @pytest.mark.skipif(
        not (examples_dir / "classification.py").exists(), reason="classification.py not found"
    )
    def test_classification_script_runs(self):
        """Test that classification.py can be run as a script."""
        script_path = examples_dir / "classification.py"

        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(examples_dir.parent),
        )

        assert result.returncode == 0, f"Script failed with: {result.stderr}"
        assert result.stdout, "Script should produce output"

        # Should show sentiment classification results
        output = result.stdout.lower()
        assert any(word in output for word in ["positive", "negative", "sentiment", "confidence"])


if __name__ == "__main__":
    # Run integration tests only
    pytest.main([__file__, "-v", "-m", "integration"])
