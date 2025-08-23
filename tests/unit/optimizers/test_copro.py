"""Unit tests for COPRO optimizer."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from logillm.core.optimizers import AccuracyMetric
from logillm.core.predict import Predict
from logillm.core.types import OptimizationResult
from logillm.exceptions import OptimizationError
from logillm.optimizers.copro import (
    COPRO,
    COPROConfig,
    COPROStats,
    InstructionCandidate,
)


class TestInstructionCandidate:
    """Test InstructionCandidate class."""

    def test_init(self):
        """Test candidate initialization."""
        candidate = InstructionCandidate(
            instruction="Test instruction", prefix="Answer:", score=0.8, depth=1
        )

        assert candidate.instruction == "Test instruction"
        assert candidate.prefix == "Answer:"
        assert candidate.score == 0.8
        assert candidate.depth == 1
        assert candidate.module is None

    def test_init_strips_quotes(self):
        """Test that initialization strips quotes."""
        candidate = InstructionCandidate(
            instruction='"Test instruction"',
            prefix='"Answer:"',
        )

        assert candidate.instruction == "Test instruction"
        assert candidate.prefix == "Answer:"

    def test_equality(self):
        """Test candidate equality."""
        candidate1 = InstructionCandidate("Test", "Answer:", 0.8)
        candidate2 = InstructionCandidate("Test", "Answer:", 0.7)  # Different score
        candidate3 = InstructionCandidate("Different", "Answer:", 0.8)

        assert candidate1 == candidate2  # Same instruction and prefix
        assert candidate1 != candidate3  # Different instruction
        assert candidate1 != "not a candidate"

    def test_hash(self):
        """Test candidate hashing for deduplication."""
        candidate1 = InstructionCandidate("Test", "Answer:", 0.8)
        candidate2 = InstructionCandidate("Test", "Answer:", 0.7)
        candidate3 = InstructionCandidate("Different", "Answer:", 0.8)

        assert hash(candidate1) == hash(candidate2)
        assert hash(candidate1) != hash(candidate3)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        candidate = InstructionCandidate(
            instruction="Test instruction", prefix="Answer:", score=0.8, depth=1
        )

        expected = {
            "instruction": "Test instruction",
            "prefix": "Answer:",
            "score": 0.8,
            "depth": 1,
        }

        assert candidate.to_dict() == expected


class TestCOPROConfig:
    """Test COPRO configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = COPROConfig()

        assert config.breadth == 10
        assert config.depth == 3
        assert config.init_temperature == 1.4
        assert config.track_stats is False
        assert config.prompt_model is None
        assert config.dedupe_candidates is True
        assert config.min_score_threshold == 0.0

    def test_custom_config(self):
        """Test custom configuration values."""
        config = COPROConfig(
            breadth=5, depth=2, init_temperature=1.0, track_stats=True, min_score_threshold=0.5
        )

        assert config.breadth == 5
        assert config.depth == 2
        assert config.init_temperature == 1.0
        assert config.track_stats is True
        assert config.min_score_threshold == 0.5


class TestCOPROStats:
    """Test COPRO statistics tracking."""

    def test_init(self):
        """Test stats initialization."""
        stats = COPROStats()

        assert stats.results_best == {}
        assert stats.results_latest == {}
        assert stats.total_calls == 0


class TestCOPRO:
    """Test COPRO optimizer."""

    @pytest.fixture
    def metric(self):
        """Create test metric."""
        return AccuracyMetric()

    @pytest.fixture
    def mock_module(self):
        """Create mock module."""
        module = Mock(spec=Predict)
        module.__class__.__name__ = "Predict"
        module.parameters = {}
        module.signature = Mock()
        module.signature.instructions = "Initial instruction"
        return module

    @pytest.fixture
    def dataset(self):
        """Create test dataset."""
        return [
            {"inputs": {"question": "What is 2+2?"}, "outputs": {"answer": "4"}},
            {"inputs": {"question": "What is 3+3?"}, "outputs": {"answer": "6"}},
        ]

    def test_init_default(self, metric):
        """Test COPRO initialization with defaults."""
        copro = COPRO(metric=metric)

        assert copro.metric == metric
        assert copro.breadth == 10
        assert copro.depth == 3
        assert copro.init_temperature == 1.4
        assert copro.track_stats is False
        assert copro.prompt_model is None
        assert copro.stats is None

    def test_init_custom(self, metric):
        """Test COPRO initialization with custom values."""
        copro = COPRO(metric=metric, breadth=5, depth=2, init_temperature=1.0, track_stats=True)

        assert copro.breadth == 5
        assert copro.depth == 2
        assert copro.init_temperature == 1.0
        assert copro.track_stats is True
        assert isinstance(copro.stats, COPROStats)

    def test_init_invalid_breadth(self, metric):
        """Test that breadth <= 1 raises error."""
        with pytest.raises(ValueError, match="Breadth must be greater than 1"):
            COPRO(metric=metric, breadth=1)

    def test_get_instruction(self, metric, mock_module):
        """Test instruction extraction from module."""
        copro = COPRO(metric=metric)

        # Test with signature
        instruction = copro._get_instruction(mock_module)
        assert instruction == "Initial instruction"

        # Test with parameters
        mock_module.parameters = {"instruction": Mock(value="Param instruction")}
        instruction = copro._get_instruction(mock_module)
        assert instruction == "Param instruction"

        # Test with no signature or parameters
        mock_module.parameters = {}
        mock_module.signature = None
        instruction = copro._get_instruction(mock_module)
        assert instruction == "Complete the following task."

    def test_get_output_prefix(self, metric, mock_module):
        """Test output prefix extraction."""
        copro = COPRO(metric=metric)
        prefix = copro._get_output_prefix(mock_module)
        assert prefix == ""  # Currently returns empty string

    def test_set_instruction(self, metric, mock_module):
        """Test setting instruction on module."""
        copro = COPRO(metric=metric)

        copro._set_instruction(mock_module, "New instruction", "Answer:")

        # Check parameters
        assert "instruction" in mock_module.parameters
        param = mock_module.parameters["instruction"]
        assert param.value == "New instruction"
        assert param.learnable is True
        assert param.metadata["type"] == "instruction"
        assert param.metadata["prefix"] == "Answer:"

    def test_dedupe_candidates(self, metric):
        """Test candidate deduplication."""
        copro = COPRO(metric=metric, dedupe_candidates=True)

        candidates = [
            InstructionCandidate("Test", "Answer:", 0.8),
            InstructionCandidate("Test", "Answer:", 0.9),  # Duplicate with higher score
            InstructionCandidate("Different", "Answer:", 0.7),
        ]

        deduped = copro._dedupe_candidates(candidates)

        assert len(deduped) == 2
        # Should keep the higher score for duplicates
        test_candidate = next(c for c in deduped if c.instruction == "Test")
        assert test_candidate.score == 0.9

    def test_dedupe_disabled(self, metric):
        """Test deduplication can be disabled."""
        copro = COPRO(metric=metric, dedupe_candidates=False)

        candidates = [
            InstructionCandidate("Test", "Answer:", 0.8),
            InstructionCandidate("Test", "Answer:", 0.9),
        ]

        deduped = copro._dedupe_candidates(candidates)
        assert len(deduped) == 2  # No deduplication

    @pytest.mark.asyncio
    async def test_generate_initial_candidates(self, metric, mock_module):
        """Test initial candidate generation."""
        with patch("logillm.optimizers.copro.get_provider") as mock_get_provider:
            # Mock provider
            mock_provider = Mock()
            mock_provider.supports_n = False
            mock_get_provider.return_value = mock_provider

            copro = COPRO(metric=metric, breadth=3)

            # Mock the basic generator
            mock_prediction = Mock()
            mock_prediction.outputs = {
                "proposed_instruction": "Generated instruction",
                "proposed_prefix": "Generated prefix",
            }
            # Mock completions attribute (check if it exists first)
            mock_prediction.completions = None
            copro.basic_generator = AsyncMock(return_value=mock_prediction)

            candidates = await copro._generate_initial_candidates(
                mock_module, "Basic instruction", "Basic prefix"
            )

            # Should have generated candidate plus original
            assert len(candidates) == 2
            assert any(c.instruction == "Generated instruction" for c in candidates)
            assert any(c.instruction == "Basic instruction" for c in candidates)

    @pytest.mark.asyncio
    async def test_generate_refined_candidates(self, metric):
        """Test refined candidate generation."""
        with patch("logillm.optimizers.copro.get_provider") as mock_get_provider:
            mock_provider = Mock()
            mock_provider.supports_n = False
            mock_get_provider.return_value = mock_provider

            copro = COPRO(metric=metric, breadth=2)

            # Mock the advanced generator
            mock_prediction = Mock()
            mock_prediction.outputs = {
                "proposed_instruction": "Refined instruction",
                "proposed_prefix": "Refined prefix",
            }
            mock_prediction.completions = None
            copro.advanced_generator = AsyncMock(return_value=mock_prediction)

            # Previous candidates
            previous = [InstructionCandidate("Old instruction", "Old prefix", 0.8, 1)]

            candidates = await copro._generate_refined_candidates(previous, 2)

            assert len(candidates) == 1
            assert candidates[0].instruction == "Refined instruction"
            assert candidates[0].depth == 2

    def test_track_stats(self, metric):
        """Test statistics tracking."""
        copro = COPRO(metric=metric, track_stats=True)

        predictor_id = "test_predictor"
        depth = 1
        latest_scores = [0.8, 0.9, 0.7]
        all_scores = [0.8, 0.9, 0.7, 0.6, 0.5]

        copro._track_stats(predictor_id, depth, latest_scores, all_scores)

        # Check latest stats
        latest = copro.stats.results_latest[predictor_id]
        assert latest["depth"] == [1]
        assert latest["max"] == [0.9]
        assert latest["average"] == [0.8]  # (0.8 + 0.9 + 0.7) / 3
        assert latest["min"] == [0.7]

        # Check best stats (top 10)
        best = copro.stats.results_best[predictor_id]
        assert best["depth"] == [1]
        assert best["max"] == [0.9]
        assert best["min"] == [0.5]

    def test_track_stats_disabled(self, metric):
        """Test that stats tracking does nothing when disabled."""
        copro = COPRO(metric=metric, track_stats=False)

        # Should not raise error
        copro._track_stats("test", 1, [0.8], [0.8])
        assert copro.stats is None

    @pytest.mark.asyncio
    async def test_optimize_minimal(self, metric, mock_module, dataset):
        """Test minimal optimization flow."""
        copro = COPRO(metric=metric, breadth=2, depth=1)

        # Mock evaluate method
        async def mock_evaluate(module, eval_set):
            # Return different scores for different modules
            if hasattr(module, "parameters") and "instruction" in module.parameters:
                inst = module.parameters["instruction"].value
                if "better" in inst.lower():
                    return 0.9, []
                else:
                    return 0.7, []
            return 0.5, []  # baseline

        copro.evaluate = mock_evaluate

        # Mock candidate generation
        copro._generate_initial_candidates = AsyncMock(
            return_value=[
                InstructionCandidate("Initial instruction", "", 0.0),
                InstructionCandidate("Better initial instruction", "", 0.0),
            ]
        )

        copro._generate_refined_candidates = AsyncMock(
            return_value=[
                InstructionCandidate("Even better instruction", "", 0.0),
            ]
        )

        # Run optimization
        result = await copro.optimize(mock_module, dataset)

        assert isinstance(result, OptimizationResult)
        assert result.optimized_module is not None
        assert result.best_score > 0
        assert result.iterations == 2  # initial + 1 depth
        assert "best_instruction" in result.metadata

    @pytest.mark.asyncio
    async def test_optimize_no_candidates(self, metric, mock_module, dataset):
        """Test optimization with no valid candidates."""
        with patch("logillm.optimizers.copro.get_provider") as mock_get_provider:
            mock_provider = Mock()
            mock_get_provider.return_value = mock_provider

            copro = COPRO(
                metric=metric, min_score_threshold=0.9, depth=0
            )  # No refinement iterations

            # Mock evaluate to return low scores
            async def mock_evaluate(module, eval_set):
                return 0.1, []  # Below threshold

            copro.evaluate = mock_evaluate

            # Mock candidate generation
            copro._generate_initial_candidates = AsyncMock(
                return_value=[
                    InstructionCandidate("Low score instruction", "", 0.0),
                ]
            )

            # Should raise error due to no candidates above threshold
            with pytest.raises(OptimizationError, match="No valid candidates found"):
                await copro.optimize(mock_module, dataset)

    @pytest.mark.asyncio
    async def test_optimize_with_validation_set(self, metric, mock_module, dataset):
        """Test optimization with separate validation set."""
        validation_set = [{"inputs": {"question": "Test?"}, "outputs": {"answer": "Test"}}]

        copro = COPRO(metric=metric, breadth=2, depth=1)

        # Mock evaluate method
        copro.evaluate = AsyncMock(return_value=(0.8, []))

        # Mock candidate generation
        copro._generate_initial_candidates = AsyncMock(
            return_value=[
                InstructionCandidate("Test instruction", "", 0.0),
            ]
        )

        copro._generate_refined_candidates = AsyncMock(return_value=[])

        result = await copro.optimize(mock_module, dataset, validation_set)

        # Should have used validation set for evaluation
        assert isinstance(result, OptimizationResult)

    @pytest.mark.asyncio
    async def test_optimize_with_stats_tracking(self, metric, mock_module, dataset):
        """Test optimization with statistics tracking enabled."""
        copro = COPRO(metric=metric, breadth=2, depth=1, track_stats=True)

        # Mock evaluate method
        copro.evaluate = AsyncMock(return_value=(0.8, []))

        # Mock candidate generation
        copro._generate_initial_candidates = AsyncMock(
            return_value=[
                InstructionCandidate("Test instruction", "", 0.0),
            ]
        )

        copro._generate_refined_candidates = AsyncMock(return_value=[])

        result = await copro.optimize(mock_module, dataset)

        # Should include stats in metadata
        assert "results_best" in result.metadata
        assert "results_latest" in result.metadata
        assert result.metadata["total_evaluations"] > 0
