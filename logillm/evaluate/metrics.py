"""Evaluation metrics for LogiLLM."""

import re
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Union


class MetricBase(ABC):
    """Base class for all metrics."""

    @abstractmethod
    def __call__(
        self,
        prediction: Union[dict[str, Any], Any],
        reference: Union[dict[str, Any], Any],
        **kwargs: Any,
    ) -> float:
        """Compute metric score.

        Args:
            prediction: Model prediction
            reference: Ground truth reference

        Returns:
            Score between 0.0 and 1.0
        """
        pass

    @property
    def name(self) -> str:
        """Get metric name."""
        return self.__class__.__name__


class ExactMatch(MetricBase):
    """Exact match metric."""

    def __init__(self, field: Optional[str] = None, ignore_case: bool = False):
        """Initialize exact match metric.

        Args:
            field: Specific field to compare (for dict outputs)
            ignore_case: Whether to ignore case in comparison
        """
        self.field = field
        self.ignore_case = ignore_case

    def __call__(
        self,
        prediction: Union[dict[str, Any], Any],
        reference: Union[dict[str, Any], Any],
        **kwargs: Any,
    ) -> float:
        """Check exact match."""
        # Extract field if specified
        if self.field:
            if isinstance(prediction, dict):
                pred_value = prediction.get(self.field)
            else:
                pred_value = prediction

            if isinstance(reference, dict):
                ref_value = reference.get(self.field)
            else:
                ref_value = reference
        else:
            pred_value = prediction
            ref_value = reference

        # Convert to strings for comparison
        pred_str = str(pred_value) if pred_value is not None else ""
        ref_str = str(ref_value) if ref_value is not None else ""

        # Apply case normalization if needed
        if self.ignore_case:
            pred_str = pred_str.lower()
            ref_str = ref_str.lower()

        return 1.0 if pred_str == ref_str else 0.0


class F1Score(MetricBase):
    """F1 score metric for token overlap."""

    def __init__(
        self, field: Optional[str] = None, tokenizer: Optional[Callable[[str], list[str]]] = None
    ):
        """Initialize F1 score metric.

        Args:
            field: Specific field to compare
            tokenizer: Custom tokenizer function
        """
        self.field = field
        self.tokenizer = tokenizer or self._default_tokenizer

    def _default_tokenizer(self, text: str) -> list[str]:
        """Default word tokenizer."""
        # Simple word tokenization
        return re.findall(r"\b\w+\b", text.lower())

    def __call__(
        self,
        prediction: Union[dict[str, Any], Any],
        reference: Union[dict[str, Any], Any],
        **kwargs: Any,
    ) -> float:
        """Compute F1 score."""
        # Extract values
        if self.field and isinstance(prediction, dict):
            pred_value = prediction.get(self.field, "")
        else:
            pred_value = prediction

        if self.field and isinstance(reference, dict):
            ref_value = reference.get(self.field, "")
        else:
            ref_value = reference

        # Convert to strings
        pred_str = str(pred_value) if pred_value else ""
        ref_str = str(ref_value) if ref_value else ""

        # Tokenize
        pred_tokens = set(self.tokenizer(pred_str))
        ref_tokens = set(self.tokenizer(ref_str))

        # Handle empty cases
        if not pred_tokens and not ref_tokens:
            return 1.0
        if not pred_tokens or not ref_tokens:
            return 0.0

        # Calculate precision and recall
        intersection = pred_tokens & ref_tokens
        precision = len(intersection) / len(pred_tokens)
        recall = len(intersection) / len(ref_tokens)

        # Calculate F1
        if precision + recall == 0:
            return 0.0

        f1 = 2 * (precision * recall) / (precision + recall)
        return f1


class Accuracy(MetricBase):
    """Accuracy metric for classification."""

    def __init__(self, field: Optional[str] = None, normalize: bool = True):
        """Initialize accuracy metric.

        Args:
            field: Field to compare (e.g., 'label', 'class')
            normalize: Whether to normalize values before comparison
        """
        self.field = field
        self.normalize = normalize

    def __call__(
        self,
        prediction: Union[dict[str, Any], Any],
        reference: Union[dict[str, Any], Any],
        **kwargs: Any,
    ) -> float:
        """Compute accuracy."""
        # Extract values
        if self.field and isinstance(prediction, dict):
            pred_value = prediction.get(self.field)
        else:
            pred_value = prediction

        if self.field and isinstance(reference, dict):
            ref_value = reference.get(self.field)
        else:
            ref_value = reference

        # Normalize if requested
        if self.normalize and pred_value is not None and ref_value is not None:
            pred_value = str(pred_value).lower().strip()
            ref_value = str(ref_value).lower().strip()

        return 1.0 if pred_value == ref_value else 0.0


class BLEUScore(MetricBase):
    """BLEU score for text generation (simplified version)."""

    def __init__(self, n_gram: int = 4, field: Optional[str] = None):
        """Initialize BLEU score.

        Args:
            n_gram: Maximum n-gram size to consider
            field: Field to evaluate
        """
        self.n_gram = n_gram
        self.field = field

    def __call__(
        self,
        prediction: Union[dict[str, Any], Any],
        reference: Union[dict[str, Any], Any],
        **kwargs: Any,
    ) -> float:
        """Compute simplified BLEU score."""
        # Extract values
        if self.field and isinstance(prediction, dict):
            pred_text = str(prediction.get(self.field, ""))
        else:
            pred_text = str(prediction) if prediction else ""

        if self.field and isinstance(reference, dict):
            ref_text = str(reference.get(self.field, ""))
        else:
            ref_text = str(reference) if reference else ""

        # Simple tokenization
        pred_tokens = pred_text.lower().split()
        ref_tokens = ref_text.lower().split()

        if not pred_tokens or not ref_tokens:
            return 0.0

        # Calculate n-gram precisions
        precisions = []
        for n in range(1, min(self.n_gram, len(pred_tokens), len(ref_tokens)) + 1):
            pred_ngrams = self._get_ngrams(pred_tokens, n)
            ref_ngrams = self._get_ngrams(ref_tokens, n)

            if not pred_ngrams:
                continue

            matches = sum(1 for ng in pred_ngrams if ng in ref_ngrams)
            precision = matches / len(pred_ngrams)
            precisions.append(precision)

        if not precisions:
            return 0.0

        # Geometric mean of precisions (simplified BLEU)
        score = 1.0
        for p in precisions:
            score *= p
        score = score ** (1.0 / len(precisions))

        # Brevity penalty
        if len(pred_tokens) < len(ref_tokens):
            bp = min(1.0, len(pred_tokens) / len(ref_tokens))
            score *= bp

        return score

    def _get_ngrams(self, tokens: list[str], n: int) -> list[tuple]:
        """Get n-grams from tokens."""
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i : i + n]))
        return ngrams


class ROUGEScore(MetricBase):
    """ROUGE score for summarization (simplified ROUGE-L)."""

    def __init__(self, field: Optional[str] = None):
        """Initialize ROUGE score.

        Args:
            field: Field to evaluate
        """
        self.field = field

    def __call__(
        self,
        prediction: Union[dict[str, Any], Any],
        reference: Union[dict[str, Any], Any],
        **kwargs: Any,
    ) -> float:
        """Compute ROUGE-L score (longest common subsequence)."""
        # Extract values
        if self.field and isinstance(prediction, dict):
            pred_text = str(prediction.get(self.field, ""))
        else:
            pred_text = str(prediction) if prediction else ""

        if self.field and isinstance(reference, dict):
            ref_text = str(reference.get(self.field, ""))
        else:
            ref_text = str(reference) if reference else ""

        # Tokenize
        pred_tokens = pred_text.lower().split()
        ref_tokens = ref_text.lower().split()

        if not pred_tokens or not ref_tokens:
            return 0.0

        # Compute LCS length
        lcs_length = self._lcs_length(pred_tokens, ref_tokens)

        # Calculate precision and recall
        precision = lcs_length / len(pred_tokens) if pred_tokens else 0
        recall = lcs_length / len(ref_tokens) if ref_tokens else 0

        # F1 score
        if precision + recall == 0:
            return 0.0

        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

    def _lcs_length(self, seq1: list[str], seq2: list[str]) -> int:
        """Compute length of longest common subsequence."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[m][n]


def create_metric(metric_type: str, **kwargs: Any) -> MetricBase:
    """Create a metric by type.

    Args:
        metric_type: Type of metric ('exact_match', 'f1', 'accuracy', 'bleu', 'rouge')
        **kwargs: Additional arguments for the metric

    Returns:
        Metric instance
    """
    metrics = {
        "exact_match": ExactMatch,
        "f1": F1Score,
        "accuracy": Accuracy,
        "bleu": BLEUScore,
        "rouge": ROUGEScore,
    }

    metric_class = metrics.get(metric_type.lower())
    if not metric_class:
        raise ValueError(f"Unknown metric type: {metric_type}")

    return metric_class(**kwargs)
