"""KNNFewShot optimizer - dynamic demonstration selection using similarity."""

import copy
import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

from ..core.embedders import Embedder, SimpleEmbedder
from ..core.knn import KNN
from ..core.modules import Module, Parameter
from ..core.optimizers import Metric
from ..core.types import OptimizationResult, OptimizationStrategy
from .base import Demonstration, PromptOptimizationConfig, PromptOptimizer
from .bootstrap_fewshot import BootstrapFewShot

logger = logging.getLogger(__name__)


@dataclass
class KNNFewShotConfig(PromptOptimizationConfig):
    """Configuration for KNNFewShot optimizer."""

    # KNN-specific parameters
    k: int = 3  # Number of nearest neighbors to retrieve
    embedder_type: str = "simple"  # "simple" or "llm"
    input_keys: Optional[list[str]] = None  # Keys to use for similarity
    text_separator: str = " | "  # Separator for joining input fields

    # Bootstrap parameters (for fallback bootstrapping if needed)
    bootstrap_fallback: bool = True
    fallback_bootstrap_demos: int = 2

    # Similarity filtering
    min_similarity: float = 0.0  # Minimum similarity threshold
    max_similarity: float = 1.0  # Maximum similarity (for diversity)


class KNNFewShot(PromptOptimizer):
    """KNN-based few-shot optimizer with dynamic demonstration selection.

    This optimizer uses k-nearest neighbor retrieval to find the most similar
    examples from a training set and uses them as demonstrations. For each
    query, it dynamically selects the most relevant demonstrations based on
    semantic similarity.

    Unlike static few-shot approaches, KNNFewShot adapts the demonstrations
    to each specific input, potentially improving performance on diverse tasks.
    """

    def __init__(
        self,
        k: int,
        trainset: list[dict[str, Any]],
        embedder: Optional[Embedder] = None,
        metric: Optional[Metric] = None,
        input_keys: Optional[list[str]] = None,
        bootstrap_fallback: bool = True,
        config: Optional[KNNFewShotConfig] = None,
        **bootstrap_kwargs: Any,
    ):
        """Initialize KNNFewShot optimizer.

        Args:
            k: Number of nearest neighbors to retrieve
            trainset: Training examples to retrieve from
            embedder: Embedder to use (defaults to SimpleEmbedder)
            metric: Evaluation metric (required if using bootstrap fallback)
            input_keys: Keys to use from inputs for similarity
            bootstrap_fallback: Whether to use BootstrapFewShot as fallback
            config: KNN-specific optimization configuration
            **bootstrap_kwargs: Additional arguments for BootstrapFewShot
        """
        if k <= 0:
            raise ValueError("k must be positive")
        if not trainset:
            raise ValueError("trainset cannot be empty")

        # Create or update config
        if config is None:
            config = KNNFewShotConfig(
                strategy=OptimizationStrategy.BOOTSTRAP,  # Use bootstrap strategy
                max_demos=k,
                k=k,
                input_keys=input_keys,
                bootstrap_fallback=bootstrap_fallback,
            )

        # Require metric if using bootstrap fallback
        if bootstrap_fallback and metric is None:
            raise ValueError("metric is required when bootstrap_fallback=True")

        super().__init__(strategy=OptimizationStrategy.BOOTSTRAP, metric=metric, config=config)

        self.config: KNNFewShotConfig = config
        self.trainset = trainset
        self.bootstrap_kwargs = bootstrap_kwargs

        # Initialize embedder
        if embedder is None:
            embedder = SimpleEmbedder()
        self.embedder = embedder

        # Initialize KNN retriever
        self.knn = KNN(
            k=k,
            trainset=trainset,
            embedder=embedder,
            input_keys=input_keys,
        )

        # Initialize bootstrap fallback if enabled
        self.bootstrap_optimizer: Optional[BootstrapFewShot] = None
        if bootstrap_fallback and metric is not None:
            self.bootstrap_optimizer = BootstrapFewShot(
                metric=metric,
                max_bootstrapped_demos=config.fallback_bootstrap_demos,
                **bootstrap_kwargs,
            )

    async def _create_dynamic_module(self, module: Module, knn_retriever: KNN) -> Module:
        """Create a module with dynamic demonstration selection.

        Args:
            module: Original module to enhance
            knn_retriever: KNN retriever for finding similar examples

        Returns:
            Module with dynamic demo selection capability
        """
        dynamic_module = copy.deepcopy(module)

        # Store the KNN retriever for runtime use
        dynamic_module._knn_retriever = knn_retriever  # type: ignore[attr-defined]
        dynamic_module._original_forward = dynamic_module.forward  # type: ignore[attr-defined]

        async def dynamic_forward(**inputs) -> Any:
            """Forward pass with dynamic demonstration selection."""
            try:
                # Retrieve similar examples for current input
                similar_examples = await knn_retriever.retrieve(**inputs)

                # Convert to demonstrations
                demonstrations: list[Demonstration] = []
                for example in similar_examples:
                    demo = Demonstration(
                        inputs=example.get("inputs", {}),
                        outputs=example.get("outputs", {}),
                        score=example.get("metadata", {}).get("similarity", 0.0),
                        metadata={
                            **example.get("metadata", {}),
                            "source": "knn_retrieval",
                            "retrieval_rank": len(demonstrations),
                        },
                    )
                    demonstrations.append(demo)

                # Update module's demo manager with dynamic demonstrations
                if hasattr(dynamic_module, "demo_manager"):
                    demo_manager = dynamic_module.demo_manager
                    demo_manager.clear()  # type: ignore[attr-defined]
                    for demo in demonstrations:
                        demo_manager.add(  # type: ignore[attr-defined]
                            {"inputs": demo.inputs, "outputs": demo.outputs}
                        )

                # Also store as parameter for tracking
                if hasattr(dynamic_module, "parameters"):
                    dynamic_module.parameters["dynamic_demonstrations"] = Parameter(
                        value=[d.to_dict() for d in demonstrations],
                        learnable=False,
                        metadata={
                            "type": "dynamic_demonstrations",
                            "source": "knn_retrieval",
                            "query_inputs": inputs,
                        },
                    )

                # Call original forward method
                return await dynamic_module._original_forward(**inputs)  # type: ignore[attr-defined]

            except Exception as e:
                logger.warning(f"Dynamic demo selection failed: {e}")
                # Fallback to original forward without demos
                return await dynamic_module._original_forward(**inputs)  # type: ignore[attr-defined]

        # Replace forward method
        dynamic_module.forward = dynamic_forward  # type: ignore[method-assign]

        return dynamic_module

    async def optimize(
        self,
        module: Module,
        dataset: list[dict[str, Any]],
        validation_set: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        """Optimize module using KNN-based dynamic demonstration selection.

        Args:
            module: Module to optimize
            dataset: Dataset for evaluation (also used as retrieval set if trainset not provided)
            validation_set: Validation set for evaluation
            **kwargs: Additional arguments

        Returns:
            OptimizationResult with KNN-enhanced module
        """
        start_time = time.time()

        # Use dataset as trainset if trainset wasn't provided during init
        retrieval_set = self.trainset if self.trainset else dataset

        logger.info(
            f"Starting KNN few-shot optimization with k={self.config.k}, "
            f"retrieval_set_size={len(retrieval_set)}, "
            f"embedder={type(self.embedder).__name__}"
        )

        # Ensure KNN retriever is fitted
        await self.knn._fit_embeddings()
        embedding_stats = self.knn.get_embedding_stats()
        logger.info(f"KNN embedding stats: {embedding_stats}")

        # Get baseline performance
        eval_set = validation_set or dataset
        baseline_score, _ = await self.evaluate(module, eval_set)

        # Create dynamic module with KNN-based demo selection
        optimized_module = await self._create_dynamic_module(module, self.knn)

        # Evaluate with dynamic demonstrations
        dynamic_score, _ = await self.evaluate(optimized_module, eval_set)

        improvement = dynamic_score - baseline_score

        # If performance is poor and bootstrap fallback is enabled, try hybrid approach
        if (
            improvement < 0.05
            and self.config.bootstrap_fallback
            and self.bootstrap_optimizer is not None
        ):
            logger.info(
                f"KNN improvement ({improvement:.3f}) is modest, "
                f"trying bootstrap fallback for additional demonstrations"
            )

            try:
                # Use bootstrap to generate additional high-quality demos
                bootstrap_result = await self.bootstrap_optimizer.optimize(
                    module, dataset, validation_set
                )

                bootstrap_module = bootstrap_result.optimized_module

                # Create hybrid module that uses both bootstrapped and KNN demos
                hybrid_module = await self._create_dynamic_module(bootstrap_module, self.knn)

                # Evaluate hybrid approach
                hybrid_score, _ = await self.evaluate(hybrid_module, eval_set)

                if hybrid_score > dynamic_score:
                    logger.info(
                        f"Hybrid approach improved: {dynamic_score:.3f} -> {hybrid_score:.3f}"
                    )
                    optimized_module = hybrid_module
                    dynamic_score = hybrid_score
                    improvement = hybrid_score - baseline_score

            except Exception as e:
                logger.warning(f"Bootstrap fallback failed: {e}")

        logger.info(
            f"KNN few-shot optimization completed: {baseline_score:.3f} -> {dynamic_score:.3f} "
            f"(improvement: {improvement:+.3f})"
        )

        return OptimizationResult(
            optimized_module=optimized_module,
            improvement=improvement,
            iterations=1,  # KNN is single-pass
            best_score=dynamic_score,
            optimization_time=time.time() - start_time,
            metadata={
                "k": self.config.k,
                "retrieval_set_size": len(retrieval_set),
                "embedder_type": type(self.embedder).__name__,
                "embedding_stats": embedding_stats,
                "baseline_score": baseline_score,
                "bootstrap_fallback_used": (
                    improvement < 0.05
                    and self.config.bootstrap_fallback
                    and self.bootstrap_optimizer is not None
                ),
                "similarity_based": True,
                "dynamic_demonstrations": True,
                "config": {
                    "k": self.config.k,
                    "embedder_type": self.config.embedder_type,
                    "input_keys": self.config.input_keys,
                    "min_similarity": self.config.min_similarity,
                    "bootstrap_fallback": self.config.bootstrap_fallback,
                },
            },
        )


__all__ = ["KNNFewShot", "KNNFewShotConfig"]
