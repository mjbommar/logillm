"""Demo selection strategies for prompt optimization."""

import random
from typing import Any, Optional

from .base import Demonstration, DemoSelector


class BestDemoSelector(DemoSelector):
    """Select the best scoring demonstrations."""

    def select(self, candidates: list[Demonstration], n: int, **kwargs: Any) -> list[Demonstration]:
        """Select top n demonstrations by score."""
        sorted_demos = sorted(candidates, key=lambda d: d.score, reverse=True)
        return sorted_demos[:n]


class RandomDemoSelector(DemoSelector):
    """Select random demonstrations."""

    def __init__(self, seed: Optional[int] = None):
        """Initialize with optional seed."""
        self.rng = random.Random(seed)

    def select(self, candidates: list[Demonstration], n: int, **kwargs: Any) -> list[Demonstration]:
        """Select n random demonstrations."""
        if len(candidates) <= n:
            return candidates
        return self.rng.sample(candidates, n)


class DiverseDemoSelector(DemoSelector):
    """Select diverse demonstrations to maximize coverage."""

    def select(self, candidates: list[Demonstration], n: int, **kwargs: Any) -> list[Demonstration]:
        """Select diverse demonstrations.

        This implementation uses a simple greedy approach:
        1. Start with the best scoring demo
        2. Iteratively add demos that are most different from selected ones
        """
        if not candidates:
            return []
        if len(candidates) <= n:
            return candidates

        selected = []
        remaining = candidates.copy()

        # Start with best scoring
        best = max(remaining, key=lambda d: d.score)
        selected.append(best)
        remaining.remove(best)

        # Add diverse demos
        while len(selected) < n and remaining:
            # Find demo most different from selected ones
            best_candidate = None
            best_diversity = -1

            for candidate in remaining:
                # Simple diversity: different input/output keys
                diversity = self._compute_diversity(candidate, selected)
                if diversity > best_diversity:
                    best_diversity = diversity
                    best_candidate = candidate

            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)

        return selected

    def _compute_diversity(self, candidate: Demonstration, selected: list[Demonstration]) -> float:
        """Compute diversity score between candidate and selected demos."""
        if not selected:
            return 1.0

        # Simple diversity based on key differences
        diversity_scores = []

        for demo in selected:
            # Compare input keys
            cand_input_keys = set(candidate.inputs.keys())
            demo_input_keys = set(demo.inputs.keys())
            input_diff = len(cand_input_keys.symmetric_difference(demo_input_keys))

            # Compare output keys
            cand_output_keys = set(candidate.outputs.keys())
            demo_output_keys = set(demo.outputs.keys())
            output_diff = len(cand_output_keys.symmetric_difference(demo_output_keys))

            # Compare values (simplified - just check if different)
            value_diff = 0
            for key in cand_input_keys.intersection(demo_input_keys):
                if candidate.inputs[key] != demo.inputs[key]:
                    value_diff += 1

            diversity = (input_diff + output_diff + value_diff) / 10.0  # Normalize
            diversity_scores.append(diversity)

        # Return average diversity from all selected
        return sum(diversity_scores) / len(diversity_scores)


class StratifiedDemoSelector(DemoSelector):
    """Select demonstrations stratified by some criteria."""

    def __init__(self, stratify_key: str = "type"):
        """Initialize with stratification key."""
        self.stratify_key = stratify_key

    def select(self, candidates: list[Demonstration], n: int, **kwargs: Any) -> list[Demonstration]:
        """Select demonstrations with stratification."""
        if not candidates or n <= 0:
            return []
        if len(candidates) <= n:
            return candidates

        # Group by stratification key
        groups = {}
        for demo in candidates:
            key = demo.metadata.get(self.stratify_key, "default")
            if key not in groups:
                groups[key] = []
            groups[key].append(demo)

        # Select proportionally from each group
        selected = []
        per_group = max(1, n // len(groups))

        for group_demos in groups.values():
            # Sort by score within group
            sorted_group = sorted(group_demos, key=lambda d: d.score, reverse=True)
            selected.extend(sorted_group[:per_group])

        # If we need more, take the best remaining
        if len(selected) < n:
            remaining = [d for d in candidates if d not in selected]
            remaining.sort(key=lambda d: d.score, reverse=True)
            selected.extend(remaining[: n - len(selected)])

        return selected[:n]


# Factory function
def create_demo_selector(strategy: str, **kwargs) -> DemoSelector:
    """Create a demo selector by strategy name."""
    selectors = {
        "best": BestDemoSelector,
        "random": RandomDemoSelector,
        "diverse": DiverseDemoSelector,
        "stratified": StratifiedDemoSelector,
    }

    selector_class = selectors.get(strategy, BestDemoSelector)
    return selector_class(**kwargs)


__all__ = [
    "BestDemoSelector",
    "RandomDemoSelector",
    "DiverseDemoSelector",
    "StratifiedDemoSelector",
    "create_demo_selector",
]
