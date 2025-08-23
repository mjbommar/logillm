"""Unit tests for demo management."""

from datetime import datetime

import pytest

from logillm.core.demos import Demo, DemoManager


class TestDemo:
    """Test Demo dataclass."""

    def test_demo_creation(self):
        """Test creating a demo instance."""
        demo = Demo(
            inputs={"text": "hello"}, outputs={"response": "hi"}, score=0.9, source="manual"
        )

        assert demo.inputs == {"text": "hello"}
        assert demo.outputs == {"response": "hi"}
        assert demo.score == 0.9
        assert demo.source == "manual"
        assert isinstance(demo.timestamp, datetime)

    def test_demo_to_dict(self):
        """Test converting demo to dict."""
        demo = Demo(
            inputs={"text": "test"},
            outputs={"result": "success"},
            score=0.8,
            source="bootstrap",
            metadata={"key": "value"},
        )

        demo_dict = demo.to_dict()

        assert demo_dict["inputs"] == {"text": "test"}
        assert demo_dict["outputs"] == {"result": "success"}
        assert demo_dict["score"] == 0.8
        assert demo_dict["source"] == "bootstrap"
        assert demo_dict["metadata"] == {"key": "value"}
        assert "timestamp" in demo_dict

    def test_demo_from_dict(self):
        """Test creating demo from dict."""
        demo_dict = {
            "inputs": {"x": 1},
            "outputs": {"y": 2},
            "score": 0.95,
            "source": "optimized",
            "metadata": {"test": True},
            "timestamp": "2024-01-01T12:00:00",
        }

        demo = Demo.from_dict(demo_dict)

        assert demo.inputs == {"x": 1}
        assert demo.outputs == {"y": 2}
        assert demo.score == 0.95
        assert demo.source == "optimized"
        assert demo.metadata == {"test": True}
        assert isinstance(demo.timestamp, datetime)

    def test_demo_str_representation(self):
        """Test string representation of demo."""
        demo = Demo(inputs={}, outputs={}, score=0.75, source="manual")

        assert str(demo) == "Demo(score=0.75, source=manual)"


class TestDemoManager:
    """Test DemoManager class."""

    def test_demo_manager_initialization(self):
        """Test creating demo manager."""
        manager = DemoManager(max_demos=10, selection_strategy="best")

        assert manager.max_demos == 10
        assert manager.selection_strategy == "best"
        assert len(manager.demos) == 0
        assert len(manager.teacher_demos) == 0

    def test_add_demo_object(self):
        """Test adding a Demo object."""
        manager = DemoManager()
        demo = Demo(inputs={"q": "question"}, outputs={"a": "answer"}, score=0.9)

        manager.add(demo)

        assert len(manager.demos) == 1
        assert manager.demos[0] == demo

    def test_add_demo_dict(self):
        """Test adding a demo from dict."""
        manager = DemoManager()
        demo_dict = {"inputs": {"text": "input"}, "outputs": {"result": "output"}, "score": 0.85}

        manager.add(demo_dict)

        assert len(manager.demos) == 1
        assert manager.demos[0].inputs == {"text": "input"}
        assert manager.demos[0].outputs == {"result": "output"}
        assert manager.demos[0].score == 0.85

    def test_add_demo_complete_dict(self):
        """Test adding a demo from complete dict with timestamp."""
        manager = DemoManager()
        demo_dict = {
            "inputs": {"x": 1},
            "outputs": {"y": 2},
            "score": 0.7,
            "source": "bootstrap",
            "metadata": {},
            "timestamp": "2024-01-01T00:00:00",
        }

        manager.add(demo_dict)

        assert len(manager.demos) == 1
        assert manager.demos[0].source == "bootstrap"

    def test_max_demos_enforcement(self):
        """Test that max_demos limit is enforced."""
        manager = DemoManager(max_demos=3)

        # Add 5 demos with different scores
        for i in range(5):
            manager.add(
                Demo(
                    inputs={"i": i},
                    outputs={"o": i},
                    score=i * 0.2,  # 0.0, 0.2, 0.4, 0.6, 0.8
                )
            )

        # Should only keep the 3 best
        assert len(manager.demos) == 3
        scores = [d.score for d in manager.demos]
        # Use pytest.approx for floating point comparison
        assert scores == pytest.approx([0.8, 0.6, 0.4])  # Top 3 scores

    def test_clear_demos(self):
        """Test clearing all demos."""
        manager = DemoManager()

        # Add some demos
        for _i in range(3):
            manager.add(Demo(inputs={}, outputs={}, score=0.5))

        assert len(manager.demos) == 3

        manager.clear()

        assert len(manager.demos) == 0

    def test_get_best_demos(self):
        """Test getting best demos."""
        manager = DemoManager()

        # Add demos with different scores
        demos = [
            Demo(inputs={}, outputs={}, score=0.3),
            Demo(inputs={}, outputs={}, score=0.9),
            Demo(inputs={}, outputs={}, score=0.6),
            Demo(inputs={}, outputs={}, score=0.1),
        ]
        for demo in demos:
            manager.add(demo)

        # Get best 2
        best_2 = manager.get_best(2)
        assert len(best_2) == 2
        assert best_2[0].score == 0.9
        assert best_2[1].score == 0.6

        # Get all sorted
        all_best = manager.get_best()
        assert len(all_best) == 4
        assert [d.score for d in all_best] == [0.9, 0.6, 0.3, 0.1]

    def test_filter_by_source(self):
        """Test filtering demos by source."""
        manager = DemoManager()

        # Add demos from different sources
        manager.add(Demo(inputs={}, outputs={}, source="manual"))
        manager.add(Demo(inputs={}, outputs={}, source="bootstrap"))
        manager.add(Demo(inputs={}, outputs={}, source="manual"))
        manager.add(Demo(inputs={}, outputs={}, source="optimized"))

        manual_demos = manager.filter_by_source("manual")
        assert len(manual_demos) == 2
        assert all(d.source == "manual" for d in manual_demos)

        bootstrap_demos = manager.filter_by_source("bootstrap")
        assert len(bootstrap_demos) == 1
        assert bootstrap_demos[0].source == "bootstrap"

    def test_to_list(self):
        """Test converting demos to list of dicts."""
        manager = DemoManager()

        manager.add(Demo(inputs={"a": 1}, outputs={"b": 2}, score=0.8))
        manager.add(Demo(inputs={"c": 3}, outputs={"d": 4}, score=0.7))

        demo_list = manager.to_list()

        assert len(demo_list) == 2
        assert demo_list[0]["inputs"] == {"a": 1}
        assert demo_list[0]["outputs"] == {"b": 2}
        assert demo_list[0]["score"] == 0.8
        assert demo_list[1]["inputs"] == {"c": 3}
        assert demo_list[1]["outputs"] == {"d": 4}
        assert demo_list[1]["score"] == 0.7

    def test_from_list(self):
        """Test loading demos from list."""
        manager = DemoManager()

        # Start with some demos
        manager.add(Demo(inputs={}, outputs={}))
        assert len(manager.demos) == 1

        # Load new demos from list
        demo_list = [
            {"inputs": {"x": 1}, "outputs": {"y": 2}, "score": 0.9},
            {"inputs": {"x": 2}, "outputs": {"y": 4}, "score": 0.8},
            Demo(inputs={"x": 3}, outputs={"y": 6}, score=0.7),  # Can be Demo object
        ]

        manager.from_list(demo_list)

        # Should clear old demos and load new ones
        assert len(manager.demos) == 3
        assert manager.demos[0].inputs == {"x": 1}
        assert manager.demos[1].inputs == {"x": 2}
        assert manager.demos[2].inputs == {"x": 3}

    def test_teacher_demos_storage(self):
        """Test that teacher demos are stored separately."""
        manager = DemoManager()

        # Add regular demo
        manager.add(Demo(inputs={}, outputs={}, source="manual"))

        # Add teacher demo (manually for now)
        teacher_demo = Demo(inputs={}, outputs={}, source="teacher")
        manager.teacher_demos.append(teacher_demo)

        assert len(manager.demos) == 1
        assert len(manager.teacher_demos) == 1
        assert manager.teacher_demos[0].source == "teacher"

    def test_selection_strategies(self):
        """Test different selection strategies are stored."""
        strategies = ["best", "diverse", "recent", "teacher"]

        for strategy in strategies:
            manager = DemoManager(selection_strategy=strategy)
            assert manager.selection_strategy == strategy
