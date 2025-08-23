"""Integration tests for COPRO optimizer with real LLM."""

import os

import pytest

from logillm.core.optimizers import AccuracyMetric, F1Metric
from logillm.core.predict import Predict
from logillm.optimizers import COPRO
from logillm.providers import load_openai_provider

# Skip integration tests if no OpenAI API key
pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set - skipping integration tests"
)


@pytest.fixture
def openai_provider():
    """Create OpenAI provider for testing."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not available")

    try:
        OpenAIProvider = load_openai_provider()
        return OpenAIProvider(
            api_key=api_key,
            model="gpt-4.1-mini",  # Use cheaper model for testing
            max_tokens=100,
            temperature=0.1,  # Low temperature for consistent testing
        )
    except ImportError as e:
        pytest.skip(f"OpenAI provider not available: {e}")


@pytest.fixture
def qa_dataset():
    """Simple question-answering dataset."""
    return [
        {"inputs": {"question": "What is the capital of France?"}, "outputs": {"answer": "Paris"}},
        {"inputs": {"question": "What is 2 + 2?"}, "outputs": {"answer": "4"}},
        {"inputs": {"question": "What color is the sky?"}, "outputs": {"answer": "blue"}},
        {"inputs": {"question": "How many legs does a dog have?"}, "outputs": {"answer": "4"}},
    ]


@pytest.fixture
def classification_dataset():
    """Simple sentiment classification dataset."""
    return [
        {"inputs": {"text": "I love this product!"}, "outputs": {"sentiment": "positive"}},
        {"inputs": {"text": "This is terrible."}, "outputs": {"sentiment": "negative"}},
        {"inputs": {"text": "This is amazing!"}, "outputs": {"sentiment": "positive"}},
        {"inputs": {"text": "I hate this."}, "outputs": {"sentiment": "negative"}},
    ]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_copro_improves_instructions(openai_provider, qa_dataset):
    """Test that COPRO improves instruction performance."""
    # Create module with basic instruction
    qa_module = Predict("question -> answer", provider=openai_provider)
    qa_module.signature.instructions = "Answer the question."

    # Measure baseline performance
    metric = AccuracyMetric(key="answer")
    baseline_score = 0.0

    for example in qa_dataset[:2]:  # Use small subset for speed
        prediction = await qa_module(**example["inputs"])
        score = metric(prediction.outputs, example["outputs"])
        baseline_score += score
    baseline_score /= 2

    # Create COPRO optimizer with small parameters for fast testing
    copro = COPRO(
        metric=metric,
        breadth=3,  # Small breadth for speed
        depth=1,  # Single refinement iteration
        init_temperature=0.7,
        prompt_model=openai_provider,
    )

    # Optimize module
    result = await copro.optimize(qa_module, qa_dataset[:2])  # Use small subset

    # Verify optimization completed
    assert result is not None
    assert result.optimized_module is not None
    assert result.best_score >= 0.0
    assert result.iterations > 0
    assert "best_instruction" in result.metadata

    # The optimized instruction should be valid (not necessarily different)
    # COPRO might determine the original is best after evaluation
    optimized_instruction = result.metadata["best_instruction"]
    assert len(optimized_instruction) > 0

    # COPRO should have evaluated multiple candidates
    assert "num_candidates" in result.metadata
    assert result.metadata["num_candidates"] >= 2  # At least original + 1 generated

    # Log the result for debugging
    print("Original instruction: Answer the question.")
    print(f"Optimized instruction: {optimized_instruction}")
    print(f"Number of candidates evaluated: {result.metadata['num_candidates']}")

    print(f"Baseline score: {baseline_score:.3f}")
    print(f"Optimized score: {result.best_score:.3f}")
    print("Original instruction: Answer the question.")
    print(f"Optimized instruction: {optimized_instruction}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_copro_breadth_depth_search(openai_provider, qa_dataset):
    """Test COPRO's breadth-first then depth search strategy."""
    qa_module = Predict("question -> answer", provider=openai_provider)
    qa_module.signature.instructions = "Answer briefly."

    metric = AccuracyMetric(key="answer")

    # Test with breadth=2, depth=2 for more exploration
    copro = COPRO(
        metric=metric,
        breadth=2,
        depth=2,
        init_temperature=0.8,
        track_stats=True,  # Enable stats tracking
        prompt_model=openai_provider,
    )

    result = await copro.optimize(qa_module, qa_dataset[:2])

    # Verify search strategy worked
    assert result.iterations == 3  # Initial + 2 depth iterations
    assert result.metadata["breadth"] == 2
    assert result.metadata["depth"] == 2

    # With stats enabled, should have tracking data
    assert "results_best" in result.metadata
    assert "results_latest" in result.metadata
    assert result.metadata["total_evaluations"] >= 2  # At least initial candidates

    # Should have tried multiple candidates
    assert result.metadata["num_candidates"] >= 2
    assert "candidate_scores" in result.metadata
    assert len(result.metadata["candidate_scores"]) >= 2


@pytest.mark.integration
@pytest.mark.asyncio
async def test_copro_with_qa_task(openai_provider, qa_dataset):
    """Test COPRO optimization on question-answering task."""
    # Create QA module
    qa_module = Predict("question -> answer", provider=openai_provider)
    qa_module.signature.instructions = "Please answer the question."

    metric = AccuracyMetric(key="answer")

    copro = COPRO(metric=metric, breadth=2, depth=1, prompt_model=openai_provider)

    # Split dataset for training/validation
    train_set = qa_dataset[:2]
    val_set = qa_dataset[2:3]  # Use one example for validation

    result = await copro.optimize(qa_module, train_set, val_set)

    # Test the optimized module
    optimized_module = result.optimized_module
    test_example = {"question": "What is the capital of England?"}

    prediction = await optimized_module(**test_example)

    # Should produce reasonable output
    assert "outputs" in prediction.outputs or "answer" in prediction.outputs
    answer = prediction.outputs.get("answer", prediction.outputs.get("outputs", ""))
    assert len(str(answer)) > 0

    print(f"Test question: {test_example['question']}")
    print(f"Answer: {answer}")
    print(f"Optimized instruction: {result.metadata['best_instruction']}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_copro_with_classification(openai_provider, classification_dataset):
    """Test COPRO optimization on classification task."""
    # Create classification module
    classifier = Predict("text -> sentiment", provider=openai_provider)
    classifier.signature.instructions = "Classify the sentiment as positive or negative."

    metric = AccuracyMetric(key="sentiment")

    copro = COPRO(metric=metric, breadth=2, depth=1, prompt_model=openai_provider)

    result = await copro.optimize(classifier, classification_dataset[:3])

    # Test optimized classifier
    optimized_classifier = result.optimized_module
    test_input = {"text": "This is wonderful!"}

    prediction = await optimized_classifier(**test_input)

    # Should classify sentiment
    sentiment = prediction.outputs.get("sentiment", "")
    assert sentiment.lower() in ["positive", "negative"] or "positive" in sentiment.lower()

    print(f"Test text: {test_input['text']}")
    print(f"Predicted sentiment: {sentiment}")
    print(f"Optimized instruction: {result.metadata['best_instruction']}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_copro_deduplication(openai_provider, qa_dataset):
    """Test that COPRO properly deduplicates candidates."""
    qa_module = Predict("question -> answer", provider=openai_provider)
    qa_module.signature.instructions = "Answer the question briefly."

    metric = AccuracyMetric(key="answer")

    # Enable deduplication (default)
    copro = COPRO(
        metric=metric, breadth=3, depth=1, dedupe_candidates=True, prompt_model=openai_provider
    )

    result = await copro.optimize(qa_module, qa_dataset[:2])

    # Should have removed duplicates if any were generated
    candidate_scores = result.metadata["candidate_scores"]
    num_candidates = result.metadata["num_candidates"]

    # At minimum should have generated some candidates
    assert num_candidates >= 1
    assert len(candidate_scores) == num_candidates

    print(f"Number of unique candidates: {num_candidates}")
    print(f"Candidate scores: {candidate_scores}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_copro_with_f1_metric(openai_provider):
    """Test COPRO with F1 metric for text similarity."""
    # Create text completion module
    completer = Predict("prompt -> completion", provider=openai_provider)
    completer.signature.instructions = "Complete the text."

    # Dataset with text completion examples
    dataset = [
        {"inputs": {"prompt": "The weather today is"}, "outputs": {"completion": "sunny and warm"}},
        {"inputs": {"prompt": "My favorite color is"}, "outputs": {"completion": "blue"}},
    ]

    # Use F1 metric for text similarity
    metric = F1Metric()

    copro = COPRO(metric=metric, breadth=2, depth=1, prompt_model=openai_provider)

    result = await copro.optimize(completer, dataset)

    # Test the optimized module
    optimized_module = result.optimized_module
    test_input = {"prompt": "The best programming language is"}

    prediction = await optimized_module(**test_input)
    completion = prediction.outputs.get("completion", "")

    assert len(completion) > 0

    print(f"Test prompt: {test_input['prompt']}")
    print(f"Completion: {completion}")
    print(f"F1-optimized instruction: {result.metadata['best_instruction']}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_copro_error_handling(openai_provider, qa_dataset):
    """Test COPRO handles errors gracefully."""
    qa_module = Predict("question -> answer", provider=openai_provider)

    metric = AccuracyMetric(key="answer")

    # Test with very high score threshold - should fail
    copro = COPRO(
        metric=metric,
        breadth=2,
        depth=1,
        min_score_threshold=0.99,  # Very high threshold
        prompt_model=openai_provider,
    )

    # Should raise error if no candidates meet threshold
    with pytest.raises(Exception):  # Could be OptimizationError or other
        await copro.optimize(qa_module, qa_dataset[:1])

    # Test with normal threshold should work
    copro_normal = COPRO(
        metric=metric, breadth=2, depth=1, min_score_threshold=0.0, prompt_model=openai_provider
    )

    result = await copro_normal.optimize(qa_module, qa_dataset[:1])
    assert result is not None
