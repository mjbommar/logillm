"""Unit tests for MIPROv2 optimizer."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from logillm.core.predict import Predict
from logillm.core.types import Prediction
from logillm.optimizers.miprov2 import MIPROv2Config, MIPROv2Optimizer
from logillm.optimizers.proposers.base import InstructionProposal, ProposalStrategy


@pytest.fixture
def mock_module():
    """Create a mock module for testing."""
    # Mock forward method
    async def mock_forward(**kwargs):
        return Prediction(success=True, outputs={"answer": "test answer"}, metadata={})

    # Create module as AsyncMock that can be called directly
    module = AsyncMock(spec=Predict, side_effect=mock_forward)
    module.signature = MagicMock()
    module.signature.signature_str = "question -> answer"
    module.signature.input_fields = {"question": MagicMock()}
    module.signature.output_fields = {"answer": MagicMock()}
    module.forward = AsyncMock(side_effect=mock_forward)

    # Mock deepcopy to return a module with the same forward method
    def mock_deepcopy():
        copy = AsyncMock(spec=Predict, side_effect=mock_forward)
        copy.signature = module.signature
        copy.forward = AsyncMock(side_effect=mock_forward)
        copy.deepcopy = mock_deepcopy
        return copy

    module.deepcopy = MagicMock(side_effect=mock_deepcopy)

    return module


@pytest.fixture
def sample_dataset():
    """Create sample dataset."""
    return [
        {"inputs": {"question": "What is 2+2?"}, "outputs": {"answer": "4"}},
        {"inputs": {"question": "What is the capital of France?"}, "outputs": {"answer": "Paris"}},
        {
            "inputs": {"question": "Who wrote Romeo and Juliet?"},
            "outputs": {"answer": "Shakespeare"},
        },
    ]


@pytest.fixture
def mock_metric():
    """Create a mock metric."""

    # Use a simple callable that returns a float
    def metric_func(pred, ref):
        return 0.8

    return metric_func


class TestMIPROv2Config:
    """Test MIPROv2Config."""

    def test_default_config(self):
        """Test default configuration."""
        config = MIPROv2Config()
        assert config.mode == "medium"
        # Medium mode defaults (set by __post_init__)
        assert config.num_candidates == 12
        assert config.validation_size == 300
        assert config.num_trials == 30
        assert config.max_bootstrapped_demos == 4
        assert config.max_labeled_demos == 16

    def test_auto_settings_light(self):
        """Test light mode auto settings."""
        config = MIPROv2Config(mode="light")

        assert config.num_candidates == 6
        assert config.validation_size == 100
        assert config.num_trials == 20
        assert config.max_bootstrapped_demos == 4
        assert config.max_labeled_demos == 16

    def test_auto_settings_medium(self):
        """Test medium mode auto settings."""
        config = MIPROv2Config(mode="medium")

        assert config.num_candidates == 12
        assert config.validation_size == 300
        assert config.num_trials == 30
        assert config.max_bootstrapped_demos == 4
        assert config.max_labeled_demos == 16

    def test_auto_settings_heavy(self):
        """Test heavy mode auto settings."""
        config = MIPROv2Config(mode="heavy")

        assert config.num_candidates == 18
        assert config.validation_size == 1000
        assert config.num_trials == 50
        assert config.max_bootstrapped_demos == 4
        assert config.max_labeled_demos == 16

    def test_manual_override(self):
        """Test manual parameter override."""
        config = MIPROv2Config(mode="light", num_candidates=15, validation_size=300)

        # Manual values should override auto settings
        assert config.num_candidates == 15
        assert config.validation_size == 300
        # num_trials should still come from light mode (not overridden)
        assert config.num_trials == 20


class TestMIPROv2Optimizer:
    """Test MIPROv2Optimizer."""

    @pytest.mark.asyncio
    async def test_initialization(self, mock_module, sample_dataset, mock_metric):
        """Test optimizer initialization."""
        config = MIPROv2Config(mode="light")
        mock_proposer = MagicMock()
        optimizer = MIPROv2Optimizer(metric=mock_metric, config=config, proposer=mock_proposer)

        assert optimizer.config == config
        assert optimizer.proposer is not None
        assert optimizer.bootstrapper is not None
        assert optimizer.instruction_candidates == []
        assert optimizer.demo_candidates == []

    @pytest.mark.asyncio
    async def test_bootstrap_demonstrations(self, mock_module, sample_dataset, mock_metric):
        """Test bootstrap demonstration generation."""
        config = MIPROv2Config(mode="light")
        mock_proposer = MagicMock()
        optimizer = MIPROv2Optimizer(metric=mock_metric, config=config, proposer=mock_proposer)

        # Create mock optimized module with demo_manager
        optimized_module = MagicMock()
        # Mock the get_best method that's actually called
        demo_mock = MagicMock()
        demo_mock.inputs = {"q": "test"}
        demo_mock.outputs = {"a": "result"}
        demo_mock.score = 0.9
        optimized_module.demo_manager.get_best = MagicMock(return_value=[demo_mock])

        # Mock bootstrap optimizer
        optimizer.bootstrapper.optimize = AsyncMock(
            return_value=MagicMock(optimized_module=optimized_module, best_score=0.8)
        )

        demo_sets = await optimizer._bootstrap_demonstrations(mock_module, sample_dataset)

        # Should return N demo sets (N=num_candidates)
        assert len(demo_sets) == config.num_candidates
        # Each demo set should have demonstrations
        assert all(len(demos) > 0 for demos in demo_sets)

    @pytest.mark.asyncio
    async def test_propose_instructions(self, mock_module, sample_dataset, mock_metric):
        """Test instruction proposal generation."""
        config = MIPROv2Config(mode="light")
        mock_proposer = MagicMock()
        optimizer = MIPROv2Optimizer(metric=mock_metric, config=config, proposer=mock_proposer)

        # Mock proposer
        mock_proposals = [
            InstructionProposal(
                instruction=f"Instruction {i}", strategy=ProposalStrategy.ALL, score=None
            )
            for i in range(5)
        ]
        optimizer.proposer.propose = AsyncMock(return_value=mock_proposals)

        proposals = await optimizer._propose_instructions(
            mock_module, sample_dataset, demo_candidates=[]
        )

        assert len(proposals) == 5
        assert all(p.instruction.startswith("Instruction") for p in proposals)
        optimizer.proposer.propose.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_search_space(self, mock_metric):
        """Test search space creation."""
        config = MIPROv2Config(mode="light")
        mock_proposer = MagicMock()
        optimizer = MIPROv2Optimizer(metric=mock_metric, config=config, proposer=mock_proposer)

        # Prepare with some candidates
        optimizer.instruction_candidates = [
            InstructionProposal(instruction="Test 1", strategy=ProposalStrategy.ALL),
            InstructionProposal(instruction="Test 2", strategy=ProposalStrategy.ALL),
        ]
        optimizer.demo_candidates = [[], []]

        search_space = optimizer._create_search_space()

        assert search_space is not None
        assert len(search_space.param_specs) > 0

    @pytest.mark.asyncio
    async def test_evaluate_module(self, mock_module, sample_dataset, mock_metric):
        """Test module evaluation."""
        config = MIPROv2Config(mode="light")
        mock_proposer = MagicMock()
        optimizer = MIPROv2Optimizer(metric=mock_metric, config=config, proposer=mock_proposer)

        score = await optimizer._evaluate_module(mock_module, sample_dataset)

        # Should call metric for each sample
        assert isinstance(score, float)
        # Module should be called directly (not forward) after callback changes
        assert mock_module.call_count == len(sample_dataset)

    @pytest.mark.asyncio
    async def test_optimize_full_pipeline(self, mock_module, sample_dataset, mock_metric):
        """Test full optimization pipeline."""
        config = MIPROv2Config(mode="light", max_iterations=2)
        mock_proposer = MagicMock()
        optimizer = MIPROv2Optimizer(metric=mock_metric, config=config, proposer=mock_proposer)

        # Mock bootstrap
        optimizer.bootstrapper.optimize = AsyncMock(
            return_value=MagicMock(
                optimized_module=mock_module, best_score=0.7, metadata={"demonstrations": []}
            )
        )

        # Mock proposer
        mock_proposals = [
            InstructionProposal(instruction="Test instruction", strategy=ProposalStrategy.ALL)
        ]
        optimizer.proposer.propose = AsyncMock(return_value=mock_proposals)

        # Note: SimpleBayesianStrategy is created in optimize method
        # We'll need to mock it if needed during execution

        # Run optimization
        result = await optimizer.optimize(mock_module, sample_dataset, metric=mock_metric)

        assert result.optimized_module is not None
        assert result.best_score >= 0  # Allow 0 since mock evaluation might return 0
        assert "best_config" in result.metadata
        assert "trial_history" in result.metadata

    @pytest.mark.asyncio
    async def test_optimize_with_validation_set(self, mock_module, sample_dataset, mock_metric):
        """Test optimization with separate validation set."""
        config = MIPROv2Config(mode="light")
        mock_proposer = MagicMock()
        optimizer = MIPROv2Optimizer(metric=mock_metric, config=config, proposer=mock_proposer)

        # Split dataset
        train_set = sample_dataset[:2]
        val_set = sample_dataset[2:]

        # Mock components
        optimizer.bootstrapper.optimize = AsyncMock(
            return_value=MagicMock(
                optimized_module=mock_module, best_score=0.7, metadata={"demonstrations": []}
            )
        )

        optimizer.proposer.propose = AsyncMock(
            return_value=[InstructionProposal(instruction="Test", strategy=ProposalStrategy.ALL)]
        )

        # Run with validation set
        result = await optimizer.optimize(
            mock_module, train_set, validation_set=val_set, metric=mock_metric
        )

        assert result.optimized_module is not None
        assert "validation_size" in result.metadata

    @pytest.mark.asyncio
    async def test_apply_configuration(self, mock_module, mock_metric):
        """Test applying configuration to module."""
        config = MIPROv2Config()
        mock_proposer = MagicMock()
        optimizer = MIPROv2Optimizer(metric=mock_metric, config=config, proposer=mock_proposer)

        # Add some instruction candidates
        optimizer.instruction_candidates = [
            InstructionProposal(instruction="Test instruction", strategy=ProposalStrategy.ALL)
        ]
        optimizer.demo_candidates = [[]]

        # deepcopy is already mocked in the fixture

        test_config = {"instruction_idx": 0, "demo_set_idx": 0, "num_demos": 2}

        result_module = await optimizer._apply_configuration(mock_module, test_config)

        # Check that deepcopy was called
        mock_module.deepcopy.assert_called()
        assert result_module is not None
