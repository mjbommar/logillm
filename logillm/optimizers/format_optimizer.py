"""Format Optimizer - Automatically discovers optimal prompt formats.

This optimizer tests different prompt formats (Markdown, JSON, XML) to find
what works best for each model and task combination. It treats format as a
hyperparameter that can be optimized alongside prompts and model parameters.
"""

import json
import logging
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from enum import Enum
from typing import Any, Optional

from ..core.adapters import ChatAdapter, JSONAdapter, MarkdownAdapter, XMLAdapter
from ..core.modules import Module
from ..core.optimizers import Metric, Optimizer
from ..core.signatures import FieldSpec, Signature
from ..core.types import OptimizationResult, OptimizationStrategy
from .format_adapter_base import Adapter, Formatter, Parser

logger = logging.getLogger(__name__)


class PromptFormat(Enum):
    """Available prompt formats to test."""

    MARKDOWN = "markdown"
    JSON = "json"
    XML = "xml"
    HYBRID_MD_JSON = "hybrid_md_json"  # Markdown structure with JSON output
    HYBRID_XML_JSON = "hybrid_xml_json"  # XML input with JSON output
    COGNITIVE = "cognitive"  # Adaptive based on cognitive function
    DSPy_STYLE = "dspy_style"  # Field markers like DSPy


@dataclass
class FormatPerformance:
    """Track performance of a specific format."""

    format: PromptFormat
    scores: list[float] = dataclass_field(default_factory=list)
    latencies: list[float] = dataclass_field(default_factory=list)
    parse_failures: int = 0
    total_attempts: int = 0

    @property
    def success_rate(self) -> float:
        """Calculate parsing success rate."""
        if self.total_attempts == 0:
            return 0.0
        return 1.0 - (self.parse_failures / self.total_attempts)

    @property
    def mean_score(self) -> float:
        """Average score across successful runs."""
        return statistics.mean(self.scores) if self.scores else 0.0

    @property
    def mean_latency(self) -> float:
        """Average latency across runs."""
        return statistics.mean(self.latencies) if self.latencies else float("inf")

    @property
    def stability(self) -> float:
        """Score stability (inverse of variance)."""
        if len(self.scores) < 2:
            return 0.0
        return 1.0 / (1.0 + statistics.variance(self.scores))


@dataclass
class FormatOptimizerConfig:
    """Configuration for format optimization."""

    formats_to_test: list[PromptFormat] = dataclass_field(
        default_factory=lambda: [
            PromptFormat.MARKDOWN,
            PromptFormat.JSON,
            PromptFormat.XML,
            # Hybrid formats temporarily disabled until adapters are implemented
            # PromptFormat.HYBRID_MD_JSON,
        ]
    )
    min_samples_per_format: int = 5
    max_samples_per_format: int = 20
    early_stopping_threshold: float = 0.2  # Stop if format is 20% worse
    adaptive_sampling: bool = True  # Allocate more samples to promising formats
    consider_latency: bool = True
    consider_stability: bool = True
    latency_weight: float = 0.1
    stability_weight: float = 0.2


class FormatOptimizer(Optimizer):
    """Optimizer that discovers the best prompt format for a task.

    This optimizer tests different prompt formats (Markdown, JSON, XML, hybrids)
    to find what works best for a specific model and task. It can be used
    standalone or as part of the HybridOptimizer.

    Key features:
    1. Tests multiple format types systematically
    2. Tracks parsing success rates
    3. Considers latency and stability
    4. Adapts sampling based on performance
    5. Learns format preferences per model
    """

    def __init__(
        self,
        metric: Metric,
        config: Optional[FormatOptimizerConfig] = None,
        track_by_model: bool = True,
        **kwargs: Any,
    ):
        """Initialize FormatOptimizer.

        Args:
            metric: Evaluation metric
            config: Format optimization configuration
            track_by_model: Whether to track format preferences per model
            **kwargs: Additional arguments for Optimizer
        """
        super().__init__(
            strategy=OptimizationStrategy.HYBRID,
            metric=metric,
        )

        self.config = config or FormatOptimizerConfig()
        self.track_by_model = track_by_model

        # Performance tracking
        self.format_performance: dict[str, dict[PromptFormat, FormatPerformance]] = defaultdict(
            dict
        )
        self.best_formats: dict[str, PromptFormat] = {}

    async def optimize(
        self,
        module: Module,
        dataset: list[dict[str, Any]],
        validation_set: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        """Optimize format selection for the module.

        Args:
            module: Module to optimize
            dataset: Training dataset
            validation_set: Validation dataset
            **kwargs: Additional arguments

        Returns:
            OptimizationResult with optimal format
        """
        start_time = time.time()
        eval_set = validation_set or dataset

        # Get model identifier
        model_id = self._get_model_id(module)

        # Initialize performance tracking for this model
        if model_id not in self.format_performance:
            for fmt in self.config.formats_to_test:
                self.format_performance[model_id][fmt] = FormatPerformance(format=fmt)

        # Test each format
        format_results = {}
        for fmt in self.config.formats_to_test:
            logger.info(f"Testing format: {fmt.value}")

            # Create module with this format
            test_module = self._apply_format(module, fmt)

            # Evaluate with adaptive sampling
            if self.config.adaptive_sampling:
                score = await self._adaptive_evaluate(test_module, eval_set, fmt, model_id)
            else:
                score = await self._fixed_evaluate(test_module, eval_set, fmt, model_id)

            format_results[fmt] = score

            # Early stopping if format is clearly bad
            if self._should_stop_early(score, format_results):
                logger.info(f"Early stopping for format {fmt.value} (score: {score:.3f})")
                break

        # Select best format
        best_format = self._select_best_format(model_id)
        self.best_formats[model_id] = best_format

        # Create optimized module
        optimized_module = self._apply_format(module, best_format)

        # Calculate improvement
        baseline_score, _ = await self.evaluate(module, eval_set[:20])
        best_score = format_results.get(best_format, 0.0)

        return OptimizationResult(
            optimized_module=optimized_module,
            improvement=best_score - baseline_score,
            iterations=len(format_results),
            best_score=best_score,
            optimization_time=time.time() - start_time,
            metadata={
                "best_format": best_format.value,
                "format_scores": {fmt.value: score for fmt, score in format_results.items()},
                "format_performance": self._get_performance_summary(model_id),
                "model_id": model_id,
            },
        )

    async def _adaptive_evaluate(
        self, module: Module, dataset: list[dict[str, Any]], format: PromptFormat, model_id: str
    ) -> float:
        """Adaptively evaluate format with dynamic sampling."""
        perf = self.format_performance[model_id][format]

        # Start with minimum samples
        n_samples = self.config.min_samples_per_format

        scores = []
        latencies = []

        for i in range(self.config.max_samples_per_format):
            if i >= n_samples:
                # Check if we should continue sampling
                if not self._should_continue_sampling(perf, model_id):
                    break

            # Evaluate on sample
            sample = dataset[i % len(dataset)]
            start = time.time()

            try:
                score, _ = await self.evaluate(module, [sample])
                latency = time.time() - start

                scores.append(score)
                latencies.append(latency)
                perf.total_attempts += 1

            except Exception as e:
                logger.debug(f"Format {format.value} failed: {e}")
                perf.parse_failures += 1
                perf.total_attempts += 1
                scores.append(0.0)
                latencies.append(time.time() - start)

        # Update performance tracking
        perf.scores.extend(scores)
        perf.latencies.extend(latencies)

        # Calculate weighted score
        return self._calculate_weighted_score(perf)

    async def _fixed_evaluate(
        self, module: Module, dataset: list[dict[str, Any]], format: PromptFormat, model_id: str
    ) -> float:
        """Evaluate format with fixed number of samples."""
        perf = self.format_performance[model_id][format]

        n_samples = min(self.config.min_samples_per_format, len(dataset))
        scores = []
        latencies = []

        for i in range(n_samples):
            sample = dataset[i]
            start = time.time()

            try:
                score, _ = await self.evaluate(module, [sample])
                latency = time.time() - start

                scores.append(score)
                latencies.append(latency)
                perf.total_attempts += 1

            except Exception as e:
                logger.debug(f"Format {format.value} failed: {e}")
                perf.parse_failures += 1
                perf.total_attempts += 1
                scores.append(0.0)
                latencies.append(time.time() - start)

        # Update performance tracking
        perf.scores.extend(scores)
        perf.latencies.extend(latencies)

        return self._calculate_weighted_score(perf)

    def _calculate_weighted_score(self, perf: FormatPerformance) -> float:
        """Calculate weighted score considering multiple factors."""
        score = perf.mean_score * perf.success_rate

        if self.config.consider_latency:
            # Normalize latency (lower is better)
            latency_factor = 1.0 / (1.0 + perf.mean_latency)
            score = (
                score * (1 - self.config.latency_weight)
                + latency_factor * self.config.latency_weight
            )

        if self.config.consider_stability:
            score = (
                score * (1 - self.config.stability_weight)
                + perf.stability * self.config.stability_weight
            )

        return score

    def _should_continue_sampling(self, perf: FormatPerformance, model_id: str) -> bool:
        """Decide whether to continue sampling for a format."""
        # Don't sample if parsing fails too often
        if perf.success_rate < 0.5 and perf.total_attempts >= 3:
            return False

        # Compare to best known format
        current_score = self._calculate_weighted_score(perf)
        best_score = 0.0

        for fmt, p in self.format_performance[model_id].items():
            if fmt != perf.format and p.scores:
                best_score = max(best_score, self._calculate_weighted_score(p))

        # Stop if significantly worse
        if best_score > 0 and current_score < best_score * (
            1 - self.config.early_stopping_threshold
        ):
            return False

        return True

    def _should_stop_early(self, current_score: float, results: dict[PromptFormat, float]) -> bool:
        """Check if we should stop testing formats early."""
        if not results:
            return False

        best_score = max(results.values())
        threshold = best_score * (1 - self.config.early_stopping_threshold)

        return current_score < threshold

    def _select_best_format(self, model_id: str) -> PromptFormat:
        """Select best format based on performance."""
        best_format = None
        best_score = -float("inf")

        for fmt, perf in self.format_performance[model_id].items():
            if perf.scores:
                score = self._calculate_weighted_score(perf)
                if score > best_score:
                    best_score = score
                    best_format = fmt

        return best_format or PromptFormat.MARKDOWN

    def _get_performance_summary(self, model_id: str) -> dict[str, Any]:
        """Get performance summary for all formats."""
        summary = {}

        for fmt, perf in self.format_performance[model_id].items():
            if perf.scores:
                summary[fmt.value] = {
                    "mean_score": perf.mean_score,
                    "success_rate": perf.success_rate,
                    "mean_latency": perf.mean_latency,
                    "stability": perf.stability,
                    "samples": len(perf.scores),
                }

        return summary

    def _apply_format(self, module: Module, format: PromptFormat) -> Module:
        """Apply a specific format to a module."""
        import copy

        formatted_module = copy.deepcopy(module)

        # Set the adapter based on format
        if format == PromptFormat.MARKDOWN:
            formatted_module.adapter = MarkdownAdapter()  # type: ignore[attr-defined]
        elif format == PromptFormat.JSON:
            formatted_module.adapter = JSONAdapter()  # type: ignore[attr-defined]
        elif format == PromptFormat.XML:
            formatted_module.adapter = XMLAdapter()  # type: ignore[attr-defined]
        elif format == PromptFormat.HYBRID_MD_JSON:
            # Use Markdown adapter as fallback for now
            formatted_module.adapter = MarkdownAdapter()  # type: ignore[attr-defined]
        elif format == PromptFormat.HYBRID_XML_JSON:
            # Use XML adapter as fallback for now
            formatted_module.adapter = XMLAdapter()  # type: ignore[attr-defined]
        elif format == PromptFormat.COGNITIVE:
            # Use Chat adapter as fallback for now
            formatted_module.adapter = ChatAdapter()  # type: ignore[attr-defined]
        else:  # DSPy_STYLE
            formatted_module.adapter = ChatAdapter()  # type: ignore[attr-defined]

        return formatted_module

    def _get_model_id(self, module: Module) -> str:
        """Get model identifier from module."""
        if hasattr(module, "provider") and module.provider:
            if hasattr(module.provider, "model"):
                return str(module.provider.model)
            elif hasattr(module.provider, "name"):
                return str(module.provider.name)
        return "unknown"

    def get_format_recommendations(self) -> dict[str, list[tuple[str, PromptFormat]]]:
        """Get format recommendations based on learned preferences.

        Returns:
            Dictionary mapping task types to recommended formats per model
        """
        recommendations = defaultdict(list)

        for model_id, best_format in self.best_formats.items():
            self.format_performance[model_id][best_format]
            recommendations["by_model"].append((model_id, best_format))

            # Analyze patterns
            if "gpt" in model_id.lower():
                recommendations["openai_models"].append((model_id, best_format))
            elif "claude" in model_id.lower():
                recommendations["anthropic_models"].append((model_id, best_format))

        return dict(recommendations)


# Hybrid and specialized adapter implementations for format optimization

# Note: Basic MarkdownAdapter and XMLAdapter are now in core.adapters


class HybridMarkdownJSONFormatter(Formatter):
    """Formatter for Markdown structure with JSON output."""

    def format_instruction(self, instruction: str | None) -> str:
        """Format instruction as Markdown header."""
        if instruction:
            return f"# Task\n\n{instruction}\n\n"
        return ""

    def format_field(self, field_spec: FieldSpec, value: Any | None = None) -> str:
        """Format field for Markdown display."""
        if value is not None:
            return f"**{field_spec.name}**: {value}\n"
        else:
            return f'  "{field_spec.name}": "your {field_spec.name} here"'

    def format_demo(self, inputs: dict[str, Any], outputs: dict[str, Any]) -> str:
        """Format demo in Markdown style."""
        lines = ["### Example\n"]
        for key, value in inputs.items():
            lines.append(f"**{key}**: {value}")
        lines.append("\nExpected JSON:")
        lines.append("```json")
        lines.append("{")
        for key, value in outputs.items():
            lines.append(f'  "{key}": "{value}"')
        lines.append("}")
        lines.append("```")
        return "\n".join(lines)

    def build_prompt(
        self,
        signature: Signature,
        inputs: dict[str, Any],
        demos: list[dict[str, Any]] | None = None,
    ) -> str:
        """Build Markdown prompt requesting JSON output."""
        parts = []

        # Add instruction
        parts.append(self.format_instruction(signature.instructions))

        # Add examples if provided
        if demos:
            parts.append("## Examples\n")
            for demo in demos:
                parts.append(self.format_demo(demo.get("inputs", {}), demo.get("outputs", {})))

        # Add input section
        parts.append("## Input\n")
        for field_name, _field_spec in signature.input_fields.items():
            value = inputs.get(field_name)
            if value is not None:
                parts.append(f"**{field_name}**: {value}")

        # Request JSON output
        parts.append("\n## Output Format\n")
        parts.append("Respond with a JSON object containing:\n")
        parts.append("```json")
        parts.append("{")
        for i, (_field_name, field_spec) in enumerate(signature.output_fields.items()):
            comma = "," if i < len(signature.output_fields) - 1 else ""
            parts.append(self.format_field(field_spec) + comma)
        parts.append("}")
        parts.append("```")

        return "\n".join(parts)


class HybridMarkdownJSONParser(Parser):
    """Parser for JSON from Markdown-structured responses."""

    def extract_fields(self, text: str, signature: Signature) -> dict[str, Any]:
        """Extract JSON from completion."""
        import json
        import re

        # Find JSON block
        json_match = re.search(r"```json\s*(\{.+?\})\s*```", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find raw JSON
        json_match = re.search(r"\{.+?\}", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        return {}

    def validate_output(self, extracted: dict[str, Any], signature: Signature) -> dict[str, Any]:
        """Validate extracted JSON."""
        validated = {}
        for field_name, field_spec in signature.output_fields.items():
            if field_name in extracted:
                validated[field_name] = extracted[field_name]
            elif field_spec.required:
                if field_spec.default is not None:
                    validated[field_name] = field_spec.default
                else:
                    validated[field_name] = ""
        return validated


class HybridMarkdownJSONAdapter(Adapter):
    """Markdown structure with JSON output."""

    def __init__(self, **kwargs):
        super().__init__(format_type="json", **kwargs)

    def _create_formatter(self) -> Formatter:
        """Create default formatter."""
        from .format_adapter_base import HybridMarkdownJSONFormatter

        return HybridMarkdownJSONFormatter()

    def _create_parser(self) -> Parser:
        """Create default parser."""
        from .format_adapter_base import HybridMarkdownJSONParser

        return HybridMarkdownJSONParser()


class HybridXMLJSONAdapter(Adapter):
    """XML input structure with JSON output."""

    def format(self, signature, inputs, demos=None):
        """Format inputs as XML but request JSON output."""
        lines = ["<request>"]

        # Instructions
        if signature.instructions:
            lines.append(f"  <task>{signature.instructions}</task>")

        # XML inputs
        lines.append("  <data>")
        for field_name, value in inputs.items():
            lines.append(f"    <{field_name}>{value}</{field_name}>")
        lines.append("  </data>")

        # JSON output request
        lines.append("  <output_format>json</output_format>")
        lines.append("  <expected_fields>")
        for field_name in signature.output_fields:
            lines.append(f"    <field>{field_name}</field>")
        lines.append("  </expected_fields>")
        lines.append("</request>")
        lines.append("\nRespond with a JSON object.")

        return [{"role": "user", "content": "\n".join(lines)}]

    def parse(self, completion, signature):
        """Parse JSON from completion."""
        import json
        import re

        # Try to find JSON
        json_match = re.search(r"\{.+?\}", completion, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except (json.JSONDecodeError, ValueError):
                pass

        return {}


class CognitiveAdapter(Adapter):
    """Adaptive format based on cognitive function."""

    def format(self, signature, inputs, demos=None):
        """Adaptively choose format based on task type."""
        # Detect task type
        task_type = self._detect_task_type(signature, inputs)

        if task_type == "reasoning":
            return self._format_for_reasoning(signature, inputs, demos)
        elif task_type == "extraction":
            return self._format_for_extraction(signature, inputs, demos)
        elif task_type == "classification":
            return self._format_for_classification(signature, inputs, demos)
        else:
            return self._format_default(signature, inputs, demos)

    def _detect_task_type(self, signature, inputs):
        """Detect the cognitive task type."""
        instructions = str(signature.instructions).lower()

        if any(word in instructions for word in ["think", "reason", "explain", "why"]):
            return "reasoning"
        elif any(word in instructions for word in ["extract", "find", "identify"]):
            return "extraction"
        elif any(word in instructions for word in ["classify", "categorize", "choose"]):
            return "classification"
        else:
            return "general"

    def _format_for_reasoning(self, signature, inputs, demos):
        """Format for reasoning tasks (XML thinking tags)."""
        lines = []
        lines.append(f"# {signature.instructions}\n")

        lines.append("<context>")
        for field_name, value in inputs.items():
            lines.append(f"  <{field_name}>{value}</{field_name}>")
        lines.append("</context>\n")

        lines.append("<thinking>")
        lines.append("  Analyze the problem step by step...")
        lines.append("</thinking>\n")

        lines.append("Provide your response as JSON:")
        lines.append("```json")
        lines.append("{")
        for field_name in signature.output_fields:
            lines.append(f'  "{field_name}": "..."')
        lines.append("}")
        lines.append("```")

        return [{"role": "user", "content": "\n".join(lines)}]

    def _format_for_extraction(self, signature, inputs, demos):
        """Format for extraction tasks (XML boundaries)."""
        # Use XML adapter for clear boundaries
        return XMLAdapter().format(signature, inputs, demos)

    def _format_for_classification(self, signature, inputs, demos):
        """Format for classification tasks (JSON choices)."""
        # Use JSON for clear categorical outputs
        from ..core.adapters import JSONAdapter

        return JSONAdapter().format(signature, inputs, demos)  # type: ignore[attr-defined]

    def _format_default(self, signature, inputs, demos):
        """Default format (Markdown)."""
        return MarkdownAdapter().format(signature, inputs, demos)

    def parse(self, completion, signature):
        """Try multiple parsing strategies."""
        parsers = [
            HybridMarkdownJSONAdapter().parse,
            XMLAdapter().parse,
            MarkdownAdapter().parse,
        ]

        for parser in parsers:
            try:
                result = parser(completion, signature)
                if result and all(field in result for field in signature.output_fields):
                    return result
            except (json.JSONDecodeError, ValueError, AttributeError):
                continue

        # Fallback: extract any mentioned values
        outputs = {}
        for field_name in signature.output_fields:
            if field_name in completion:
                # Simple extraction
                import re

                pattern = rf"{field_name}[:\s]+([^\n]+)"
                match = re.search(pattern, completion)
                if match:
                    outputs[field_name] = match.group(1).strip()

        return outputs


__all__ = ["FormatOptimizer", "PromptFormat", "FormatOptimizerConfig"]
