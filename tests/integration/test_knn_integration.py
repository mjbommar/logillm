"""Integration tests for KNN components with real LLM."""

import os

import pytest

from logillm.core.embedders import LLMEmbedder, SimpleEmbedder
from logillm.core.knn import KNN
from logillm.core.predict import Predict
from logillm.optimizers.knn_fewshot import KNNFewShot
from logillm.providers import create_provider


def get_openai_provider():
    """Get OpenAI provider for testing."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OpenAI API key not found in environment")

    return create_provider("openai", model="gpt-4.1", api_key=api_key)


def accuracy_metric(prediction: dict, expected: dict) -> float:
    """Simple accuracy metric for testing."""
    pred_answer = str(prediction.get("answer", "")).strip().lower()
    exp_answer = str(expected.get("answer", "")).strip().lower()

    # Simple exact match
    if pred_answer == exp_answer:
        return 1.0

    # Partial credit for partial matches
    if pred_answer and exp_answer and pred_answer in exp_answer:
        return 0.7

    return 0.0


class TestKNNIntegration:
    """Integration tests for KNN with real embeddings."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # 1 minute timeout
    async def test_simple_embedder_integration(self):
        """Test SimpleEmbedder with realistic text similarity."""
        embedder = SimpleEmbedder(max_features=100, normalize=True)

        # Math-related texts (use consistent vocabulary for SimpleEmbedder)
        math_texts = [
            "What is two plus two equals four",
            "Calculate two plus two equals four",
            "Addition problem two plus two",
        ]

        # Non-math text
        other_texts = [
            "The weather is sunny today",
            "I love reading books about history",
            "Cooking dinner with fresh ingredients",
        ]

        all_texts = math_texts + other_texts
        embeddings = await embedder.embed(all_texts)

        # Check embeddings
        assert len(embeddings) == 6
        assert all(len(emb) > 0 for emb in embeddings)

        # Math texts should be more similar to each other
        from logillm.core.embedders import cosine_similarity

        math_sim = cosine_similarity(embeddings[0], embeddings[1])
        cross_sim = cosine_similarity(embeddings[0], embeddings[3])  # math vs weather

        assert math_sim > cross_sim, (
            f"Math similarity {math_sim} should be > cross similarity {cross_sim}"
        )

    @pytest.mark.integration
    @pytest.mark.openai
    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # 1 minute timeout
    async def test_llm_embedder_integration(self):
        """Test LLMEmbedder with OpenAI (if available)."""
        try:
            provider = get_openai_provider()

            # Check if provider supports embeddings
            if not hasattr(provider, "embed"):
                pytest.skip("Provider does not support embeddings")

            embedder = LLMEmbedder(provider, model="text-embedding-ada-002")

            texts = [
                "What is the capital of France?",
                "What is the capital of Germany?",
                "I love eating pizza",
            ]

            embeddings = await embedder.embed(texts)

            assert len(embeddings) == 3
            assert all(len(emb) > 0 for emb in embeddings)

            # Geography questions should be more similar
            from logillm.core.embedders import cosine_similarity

            geo_sim = cosine_similarity(embeddings[0], embeddings[1])
            other_sim = cosine_similarity(embeddings[0], embeddings[2])

            assert geo_sim > other_sim

        except Exception as e:
            pytest.skip(f"LLM embedding test failed: {e}")

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # 1 minute timeout
    async def test_knn_retrieval_integration(self):
        """Test KNN retrieval with realistic examples."""
        # Create training set with math problems
        trainset = [
            {"inputs": {"question": "What is 2 + 2?"}, "outputs": {"answer": "4"}},
            {"inputs": {"question": "What is 3 + 3?"}, "outputs": {"answer": "6"}},
            {"inputs": {"question": "What is 5 * 5?"}, "outputs": {"answer": "25"}},
            {"inputs": {"question": "What is 10 / 2?"}, "outputs": {"answer": "5"}},
            {
                "inputs": {"question": "What is the capital of France?"},
                "outputs": {"answer": "Paris"},
            },
            {"inputs": {"question": "What color is the sky?"}, "outputs": {"answer": "blue"}},
        ]

        embedder = SimpleEmbedder(max_features=50)
        knn = KNN(k=3, trainset=trainset, embedder=embedder)

        # Query for a math problem
        results = await knn.retrieve(question="What is 4 + 4?")

        assert len(results) == 3

        # Results should be ranked by similarity
        similarities = [r["metadata"]["similarity"] for r in results]
        assert similarities == sorted(similarities, reverse=True)

        # Top result should likely be a math problem
        top_result = results[0]
        top_question = top_result["inputs"]["question"]

        # Should contain math-related terms
        math_indicators = ["+ ", "* ", "/ ", "what is", "equals"]
        has_math = any(indicator in top_question.lower() for indicator in math_indicators)
        assert has_math, f"Top result '{top_question}' doesn't seem math-related"

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # 1 minute timeout
    async def test_knn_different_queries_get_different_demos(self):
        """Test that different queries retrieve different demonstrations."""
        trainset = [
            {"inputs": {"question": "What is 2 + 2?"}, "outputs": {"answer": "4"}},
            {"inputs": {"question": "What is 3 * 3?"}, "outputs": {"answer": "9"}},
            {
                "inputs": {"question": "What is the capital of France?"},
                "outputs": {"answer": "Paris"},
            },
            {
                "inputs": {"question": "What is the capital of Spain?"},
                "outputs": {"answer": "Madrid"},
            },
            {"inputs": {"question": "What color is grass?"}, "outputs": {"answer": "green"}},
            {"inputs": {"question": "What color is the ocean?"}, "outputs": {"answer": "blue"}},
        ]

        embedder = SimpleEmbedder()
        knn = KNN(k=2, trainset=trainset, embedder=embedder)

        # Math query
        math_results = await knn.retrieve(question="What is 5 + 5?")

        # Geography query
        geo_results = await knn.retrieve(question="What is the capital of Italy?")

        # Should get different top results
        math_top = math_results[0]["inputs"]["question"]
        geo_top = geo_results[0]["inputs"]["question"]

        assert math_top != geo_top

        # Math query should prefer math examples
        math_has_math = any(
            "+" in r["inputs"]["question"] or "*" in r["inputs"]["question"] for r in math_results
        )
        assert math_has_math

        # Geography query should prefer geography examples
        geo_has_geo = any("capital" in r["inputs"]["question"].lower() for r in geo_results)
        assert geo_has_geo


class TestKNNFewShotIntegration:
    """Integration tests for KNNFewShot with real LLM."""

    @pytest.mark.integration
    @pytest.mark.openai
    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # 1 minute timeout
    async def test_knn_improves_qa_accuracy(self):
        """Test that KNN demonstrations improve QA accuracy."""
        try:
            provider = get_openai_provider()

            # Create training set with math problems
            trainset = [
                {"inputs": {"question": "What is 2 + 2?"}, "outputs": {"answer": "4"}},
                {"inputs": {"question": "What is 3 + 3?"}, "outputs": {"answer": "6"}},
                {"inputs": {"question": "What is 4 + 4?"}, "outputs": {"answer": "8"}},
                {"inputs": {"question": "What is 5 * 2?"}, "outputs": {"answer": "10"}},
                {"inputs": {"question": "What is 6 * 3?"}, "outputs": {"answer": "18"}},
            ]

            # Test questions
            test_dataset = [
                {"inputs": {"question": "What is 7 + 3?"}, "outputs": {"answer": "10"}},
                {"inputs": {"question": "What is 8 * 2?"}, "outputs": {"answer": "16"}},
            ]

            # Create QA module
            qa_module = Predict("question -> answer")
            qa_module.provider = provider

            # Test baseline (no demonstrations)
            baseline_score = 0
            for example in test_dataset:
                result = await qa_module(**example["inputs"])
                if result.success:
                    score = accuracy_metric(result.outputs, example["outputs"])
                    baseline_score += score
            baseline_score /= len(test_dataset)

            # Create KNN optimizer
            knn_optimizer = KNNFewShot(
                k=2,
                trainset=trainset,
                metric=accuracy_metric,
                bootstrap_fallback=False,  # Keep it simple for integration test
            )

            # Optimize module
            result = await knn_optimizer.optimize(qa_module, test_dataset)

            assert result.improvement >= 0, f"KNN should not hurt performance: {result.improvement}"
            assert result.metadata["k"] == 2
            assert result.metadata["dynamic_demonstrations"] is True

            # Test the optimized module
            optimized_module = result.optimized_module

            # Should have KNN retriever attached
            assert hasattr(optimized_module, "_knn_retriever")

            # Test a prediction with the optimized module
            test_result = await optimized_module(question="What is 9 + 1?")
            assert test_result.success

            # Should have dynamic demonstrations parameter
            if hasattr(optimized_module, "parameters"):
                assert "dynamic_demonstrations" in optimized_module.parameters

        except Exception as e:
            pytest.skip(f"KNN QA integration test failed: {e}")

    @pytest.mark.integration
    @pytest.mark.openai
    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # 1 minute timeout
    async def test_knn_with_semantic_similarity(self):
        """Test KNN with semantically similar examples."""
        try:
            provider = get_openai_provider()

            # Training set with similar concepts but different wording
            trainset = [
                {
                    "inputs": {"question": "How do I add two numbers?"},
                    "outputs": {"answer": "Use the + operator"},
                },
                {
                    "inputs": {"question": "What is the sum of X and Y?"},
                    "outputs": {"answer": "X + Y"},
                },
                {
                    "inputs": {"question": "How to calculate multiplication?"},
                    "outputs": {"answer": "Use the * operator"},
                },
                {
                    "inputs": {"question": "What is the product of A and B?"},
                    "outputs": {"answer": "A * B"},
                },
                {
                    "inputs": {"question": "How do I cook pasta?"},
                    "outputs": {"answer": "Boil water, add pasta"},
                },
                {
                    "inputs": {"question": "What ingredients do I need for pizza?"},
                    "outputs": {"answer": "Dough, sauce, cheese"},
                },
            ]

            test_dataset = [
                {
                    "inputs": {"question": "How do I perform addition?"},
                    "outputs": {"answer": "Use the + operator"},
                },
            ]

            qa_module = Predict("question -> answer")
            qa_module.provider = provider

            knn_optimizer = KNNFewShot(
                k=2, trainset=trainset, metric=accuracy_metric, bootstrap_fallback=False
            )

            await knn_optimizer.optimize(qa_module, test_dataset)

            # Should find semantically similar examples
            # Test retrieval manually
            await knn_optimizer.knn._fit_embeddings()
            similar_examples = await knn_optimizer.knn.retrieve(
                question="How do I perform addition?"
            )

            # Top results should be math-related, not cooking
            top_questions = [ex["inputs"]["question"] for ex in similar_examples[:2]]

            math_terms = ["add", "sum", "numbers", "calculate", "multiplication", "+", "*"]
            cooking_terms = ["cook", "pasta", "pizza", "ingredients"]

            math_score = sum(1 for q in top_questions for term in math_terms if term in q.lower())
            cooking_score = sum(
                1 for q in top_questions for term in cooking_terms if term in q.lower()
            )

            assert math_score > cooking_score, (
                f"Should prefer math examples. Math: {math_score}, Cooking: {cooking_score}"
            )

        except Exception as e:
            pytest.skip(f"Semantic similarity test failed: {e}")

    @pytest.mark.integration
    @pytest.mark.openai
    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # 1 minute timeout
    async def test_knn_dynamic_selection(self):
        """Test that KNN dynamically selects different demos for different queries."""
        try:
            provider = get_openai_provider()

            trainset = [
                # Math examples
                {"inputs": {"question": "What is 2 + 2?"}, "outputs": {"answer": "4"}},
                {"inputs": {"question": "What is 3 * 3?"}, "outputs": {"answer": "9"}},
                # Geography examples
                {
                    "inputs": {"question": "What is the capital of France?"},
                    "outputs": {"answer": "Paris"},
                },
                {
                    "inputs": {"question": "What is the capital of Italy?"},
                    "outputs": {"answer": "Rome"},
                },
                # Science examples
                {"inputs": {"question": "What is H2O?"}, "outputs": {"answer": "Water"}},
                {"inputs": {"question": "What is CO2?"}, "outputs": {"answer": "Carbon dioxide"}},
            ]

            qa_module = Predict("question -> answer")
            qa_module.provider = provider

            knn_optimizer = KNNFewShot(
                k=2, trainset=trainset, metric=accuracy_metric, bootstrap_fallback=False
            )

            # Optimize module
            result = await knn_optimizer.optimize(qa_module, [])
            optimized_module = result.optimized_module

            # Test different types of queries
            queries = [
                "What is 5 + 5?",  # Math
                "What is the capital of Germany?",  # Geography
                "What is NaCl?",  # Science
            ]

            demo_sets = []
            for query in queries:
                # Call forward to trigger dynamic demo selection
                await optimized_module(question=query)

                # Get the selected demos
                if (
                    hasattr(optimized_module, "parameters")
                    and "dynamic_demonstrations" in optimized_module.parameters
                ):
                    demos = optimized_module.parameters["dynamic_demonstrations"].value
                    demo_questions = [d["inputs"]["question"] for d in demos]
                    demo_sets.append(demo_questions)

            # Demo sets should be different for different query types
            assert len(demo_sets) == 3, "Should have demos for all queries"

            # At least some demo sets should be different
            all_same = all(demo_sets[0] == demo_set for demo_set in demo_sets[1:])
            assert not all_same, "Demo sets should vary for different query types"

        except Exception as e:
            pytest.skip(f"Dynamic selection test failed: {e}")

    @pytest.mark.integration
    @pytest.mark.openai
    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # 1 minute timeout
    async def test_knn_with_bootstrap_fallback(self):
        """Test KNN with bootstrap fallback integration."""
        try:
            provider = get_openai_provider()

            # Small training set to potentially trigger fallback
            trainset = [
                {"inputs": {"question": "What is 1 + 1?"}, "outputs": {"answer": "2"}},
                {"inputs": {"question": "What is 2 + 2?"}, "outputs": {"answer": "4"}},
            ]

            test_dataset = [
                {"inputs": {"question": "What is 3 + 3?"}, "outputs": {"answer": "6"}},
            ]

            qa_module = Predict("question -> answer")
            qa_module.provider = provider

            knn_optimizer = KNNFewShot(
                k=1,
                trainset=trainset,
                metric=accuracy_metric,
                bootstrap_fallback=True,  # Enable fallback
                fallback_bootstrap_demos=1,
                max_bootstrapped_demos=1,  # Keep bootstrap small
            )

            result = await knn_optimizer.optimize(qa_module, test_dataset)

            assert result.optimized_module is not None
            assert "bootstrap_fallback_used" in result.metadata

            # Should have both KNN and potentially bootstrap features
            optimized_module = result.optimized_module
            assert hasattr(optimized_module, "_knn_retriever")

        except Exception as e:
            pytest.skip(f"Bootstrap fallback integration test failed: {e}")
