"""Demo management for few-shot learning."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .types import Metadata


@dataclass
class Demo:
    """A demonstration example for few-shot learning."""

    inputs: dict[str, Any]
    outputs: dict[str, Any]
    score: float = 1.0  # Quality score for this demo
    source: str = "manual"  # "manual", "bootstrap", "optimized"
    metadata: Metadata = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "inputs": self.inputs,
            "outputs": self.outputs,
            "score": self.score,
            "source": self.source,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Demo:
        """Create from dictionary."""
        if "timestamp" in data and isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)

    def __str__(self) -> str:
        """String representation."""
        return f"Demo(score={self.score:.2f}, source={self.source})"


class DemoManager:
    """Manages demonstrations for a module with teacher-student architecture support."""

    def __init__(self, max_demos: int = 5, selection_strategy: str = "best"):
        self.demos: list[Demo] = []
        self.max_demos = max_demos
        self.selection_strategy = selection_strategy  # "best", "diverse", "recent", "teacher"
        self.teacher_demos: list[Demo] = []  # Separate storage for teacher examples

    def add(self, demo: Demo | dict[str, Any]) -> None:
        """Add a demonstration."""
        if isinstance(demo, dict):
            # Convert dict to Demo
            if "inputs" in demo and "outputs" in demo:
                demo = Demo(
                    inputs=demo["inputs"],
                    outputs=demo["outputs"],
                    score=demo.get("score", 1.0),
                    source=demo.get("source", "manual"),
                    metadata=demo.get("metadata", {}),
                )
            else:
                # Assume it's a complete demo dict
                demo = Demo.from_dict(demo)

        self.demos.append(demo)

        # Keep only the best demos if we exceed max
        if len(self.demos) > self.max_demos:
            self.demos.sort(key=lambda d: d.score, reverse=True)
            self.demos = self.demos[: self.max_demos]

    def clear(self) -> None:
        """Clear all demonstrations."""
        self.demos.clear()

    def get_best(self, n: int | None = None) -> list[Demo]:
        """Get the n best demonstrations."""
        sorted_demos = sorted(self.demos, key=lambda d: d.score, reverse=True)
        if n is not None:
            return sorted_demos[:n]
        return sorted_demos

    def filter_by_source(self, source: str) -> list[Demo]:
        """Get demonstrations from a specific source."""
        return [d for d in self.demos if d.source == source]

    def to_list(self) -> list[dict[str, Any]]:
        """Convert all demos to list of dicts for serialization."""
        return [demo.to_dict() for demo in self.demos]

    def from_list(self, demos: list[Demo | dict[str, Any]]) -> None:
        """Load demos from a list."""
        self.clear()
        for demo in demos:
            self.add(demo)

    def __len__(self) -> int:
        """Number of demonstrations."""
        return len(self.demos)

    def __iter__(self):
        """Iterate over demonstrations."""
        return iter(self.demos)

    def add_teacher_demo(self, demo: Demo | dict[str, Any]) -> None:
        """Add a demonstration from a teacher model."""
        if isinstance(demo, dict):
            if "inputs" in demo and "outputs" in demo:
                demo = Demo(
                    inputs=demo["inputs"],
                    outputs=demo["outputs"],
                    score=demo.get("score", 1.5),  # Teacher demos get higher default score
                    source="teacher",
                    metadata=demo.get("metadata", {}),
                )
            else:
                demo = Demo.from_dict(demo)
                demo.source = "teacher"

        self.teacher_demos.append(demo)
        # Also add to main demos with high priority
        self.add(demo)

    def get_diverse(self, n: int) -> list[Demo]:
        """Get diverse demonstrations using different selection strategies."""
        if self.selection_strategy == "teacher":
            # Prefer teacher demos
            teacher_subset = (
                self.filter_by_source("teacher")[: n // 2] if self.teacher_demos else []
            )
            other_subset = self.filter_by_source("bootstrap")[: (n - len(teacher_subset))]
            return teacher_subset + other_subset
        elif self.selection_strategy == "diverse":
            # Mix different sources
            sources = {}
            for demo in self.demos:
                if demo.source not in sources:
                    sources[demo.source] = []
                sources[demo.source].append(demo)

            diverse_demos = []
            per_source = max(1, n // len(sources)) if sources else n
            for source_demos in sources.values():
                diverse_demos.extend(source_demos[:per_source])

            return diverse_demos[:n]
        elif self.selection_strategy == "recent":
            # Return most recently added demos
            return list(reversed(self.demos))[:n]
        else:  # "best" or default
            return self.get_best(n)

    def select_for_prompt(self, n: int = 3) -> list[Demo]:
        """Select demonstrations for inclusion in a prompt."""
        return self.get_diverse(n)

    def __repr__(self) -> str:
        """String representation."""
        sources = {}
        for demo in self.demos:
            sources[demo.source] = sources.get(demo.source, 0) + 1
        source_str = ", ".join(f"{k}={v}" for k, v in sources.items())
        teacher_str = f", teacher={len(self.teacher_demos)}" if self.teacher_demos else ""
        return f"DemoManager(total={len(self.demos)}, {source_str}{teacher_str}, strategy={self.selection_strategy})"


__all__ = [
    "Demo",
    "DemoManager",
]
