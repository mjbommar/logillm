"""Unit tests for evaluation metrics."""

import pytest

from logillm.evaluate.metrics import (
    Accuracy,
    BLEUScore,
    ExactMatch,
    F1Score,
    ROUGEScore,
    create_metric,
)


class TestExactMatch:
    """Test ExactMatch metric."""

    def test_exact_match_strings(self):
        """Test exact match with strings."""
        metric = ExactMatch()

        assert metric("hello", "hello") == 1.0
        assert metric("hello", "world") == 0.0
        assert metric("", "") == 1.0

    def test_exact_match_case_sensitive(self):
        """Test case-sensitive matching."""
        metric = ExactMatch(ignore_case=False)

        assert metric("Hello", "hello") == 0.0
        assert metric("HELLO", "HELLO") == 1.0

    def test_exact_match_case_insensitive(self):
        """Test case-insensitive matching."""
        metric = ExactMatch(ignore_case=True)

        assert metric("Hello", "hello") == 1.0
        assert metric("HELLO", "hello") == 1.0
        assert metric("HeLLo", "hEllO") == 1.0

    def test_exact_match_with_field(self):
        """Test exact match with field extraction."""
        metric = ExactMatch(field="answer")

        pred = {"answer": "Paris", "confidence": 0.9}
        ref = {"answer": "Paris", "confidence": 1.0}

        assert metric(pred, ref) == 1.0

        pred2 = {"answer": "London"}
        assert metric(pred2, ref) == 0.0

    def test_exact_match_none_values(self):
        """Test handling of None values."""
        metric = ExactMatch()

        assert metric(None, None) == 1.0
        assert metric("test", None) == 0.0
        assert metric(None, "test") == 0.0


class TestF1Score:
    """Test F1Score metric."""

    def test_f1_identical_strings(self):
        """Test F1 with identical strings."""
        metric = F1Score()

        assert metric("hello world", "hello world") == 1.0
        assert metric("the quick brown fox", "the quick brown fox") == 1.0

    def test_f1_partial_overlap(self):
        """Test F1 with partial word overlap."""
        metric = F1Score()

        # "hello world" vs "hello there" - 1 word overlap out of 2
        score = metric("hello world", "hello there")
        # Precision: 1/2 = 0.5, Recall: 1/2 = 0.5
        # F1: 2 * (0.5 * 0.5) / (0.5 + 0.5) = 0.5
        assert score == pytest.approx(0.5, 0.01)

    def test_f1_no_overlap(self):
        """Test F1 with no overlap."""
        metric = F1Score()

        assert metric("hello world", "foo bar") == 0.0

    def test_f1_empty_strings(self):
        """Test F1 with empty strings."""
        metric = F1Score()

        assert metric("", "") == 1.0
        assert metric("hello", "") == 0.0
        assert metric("", "world") == 0.0

    def test_f1_with_field(self):
        """Test F1 with field extraction."""
        metric = F1Score(field="text")

        pred = {"text": "the cat sat", "score": 0.8}
        ref = {"text": "the dog sat", "score": 0.9}

        # 2 words overlap (the, sat) out of 3
        # Precision: 2/3, Recall: 2/3
        # F1: 2/3
        score = metric(pred, ref)
        assert score == pytest.approx(2 / 3, 0.01)

    def test_f1_custom_tokenizer(self):
        """Test F1 with custom tokenizer."""

        def char_tokenizer(text):
            return list(text.lower())

        metric = F1Score(tokenizer=char_tokenizer)

        # Character-level F1
        score = metric("hello", "hallo")
        # "hello" has chars: h,e,l,l,o (set: {h,e,l,o})
        # "hallo" has chars: h,a,l,l,o (set: {h,a,l,o})
        # Overlap: {h,l,o} = 3 unique chars
        # Precision: 3/4 (hello has 4 unique), Recall: 3/4 (hallo has 4 unique)
        # F1: 2 * (3/4 * 3/4) / (3/4 + 3/4) = 3/4 = 0.75
        assert score == pytest.approx(0.75, 0.01)


class TestAccuracy:
    """Test Accuracy metric."""

    def test_accuracy_basic(self):
        """Test basic accuracy."""
        metric = Accuracy()

        assert metric("cat", "cat") == 1.0
        assert metric("cat", "dog") == 0.0
        assert metric(1, 1) == 1.0
        assert metric(1, 2) == 0.0

    def test_accuracy_normalized(self):
        """Test normalized accuracy."""
        metric = Accuracy(normalize=True)

        assert metric("  Cat  ", "cat") == 1.0
        assert metric("DOG", "dog  ") == 1.0
        assert metric(" YES ", "yes") == 1.0

    def test_accuracy_not_normalized(self):
        """Test non-normalized accuracy."""
        metric = Accuracy(normalize=False)

        assert metric("Cat", "cat") == 0.0
        assert metric(" yes", "yes") == 0.0
        assert metric("yes", "yes") == 1.0

    def test_accuracy_with_field(self):
        """Test accuracy with field."""
        metric = Accuracy(field="label")

        pred = {"label": "positive", "score": 0.9}
        ref = {"label": "positive", "score": 0.8}

        assert metric(pred, ref) == 1.0

        pred2 = {"label": "negative"}
        assert metric(pred2, ref) == 0.0


class TestBLEUScore:
    """Test BLEUScore metric."""

    def test_bleu_identical(self):
        """Test BLEU with identical strings."""
        metric = BLEUScore()

        assert metric("hello world", "hello world") == 1.0

    def test_bleu_partial_match(self):
        """Test BLEU with partial match."""
        # Use unigram only to ensure non-zero score
        metric = BLEUScore(n_gram=1)

        # "the cat" vs "the dog"
        # Unigram: "the" matches (1/2 precision)
        score = metric("the cat", "the dog")
        assert score > 0  # Should have some score from unigram match
        assert score < 1.0  # But not perfect score

    def test_bleu_no_match(self):
        """Test BLEU with no match."""
        metric = BLEUScore()

        assert metric("hello world", "foo bar") == 0.0

    def test_bleu_brevity_penalty(self):
        """Test BLEU brevity penalty."""
        metric = BLEUScore()

        # Shorter prediction should be penalized
        score1 = metric("hello", "hello world test")
        score2 = metric("hello world test", "hello world test")

        assert score1 < score2

    def test_bleu_empty_strings(self):
        """Test BLEU with empty strings."""
        metric = BLEUScore()

        assert metric("", "hello") == 0.0
        assert metric("hello", "") == 0.0

    def test_bleu_ngram_sizes(self):
        """Test BLEU with different n-gram sizes."""
        metric1 = BLEUScore(n_gram=1)  # Unigram only
        metric4 = BLEUScore(n_gram=4)  # Up to 4-grams

        text1 = "the quick brown fox"
        text2 = "quick the fox brown"  # Same words, different order

        # Unigram BLEU should be perfect (same words)
        score1 = metric1(text2, text1)
        # 4-gram BLEU should be lower (different word order)
        score4 = metric4(text2, text1)

        assert score1 > score4


class TestROUGEScore:
    """Test ROUGEScore metric."""

    def test_rouge_identical(self):
        """Test ROUGE with identical strings."""
        metric = ROUGEScore()

        assert metric("hello world", "hello world") == 1.0

    def test_rouge_lcs(self):
        """Test ROUGE-L (LCS-based)."""
        metric = ROUGEScore()

        # "the cat sat" vs "the dog sat"
        # LCS = "the sat" (length 2)
        score = metric("the cat sat", "the dog sat")
        # Precision: 2/3, Recall: 2/3, F1: 2/3
        assert score == pytest.approx(2 / 3, 0.01)

    def test_rouge_no_overlap(self):
        """Test ROUGE with no overlap."""
        metric = ROUGEScore()

        assert metric("hello world", "foo bar") == 0.0

    def test_rouge_empty_strings(self):
        """Test ROUGE with empty strings."""
        metric = ROUGEScore()

        assert metric("", "hello") == 0.0
        assert metric("hello", "") == 0.0

    def test_rouge_with_field(self):
        """Test ROUGE with field extraction."""
        metric = ROUGEScore(field="summary")

        pred = {"summary": "quick brown fox"}
        ref = {"summary": "quick red fox"}

        # LCS = "quick fox" (length 2)
        # Precision: 2/3, Recall: 2/3, F1: 2/3
        score = metric(pred, ref)
        assert score == pytest.approx(2 / 3, 0.01)


class TestCreateMetric:
    """Test metric factory function."""

    def test_create_exact_match(self):
        """Test creating ExactMatch metric."""
        metric = create_metric("exact_match", ignore_case=True)
        assert isinstance(metric, ExactMatch)
        assert metric.ignore_case is True

    def test_create_f1(self):
        """Test creating F1Score metric."""
        metric = create_metric("f1", field="answer")
        assert isinstance(metric, F1Score)
        assert metric.field == "answer"

    def test_create_accuracy(self):
        """Test creating Accuracy metric."""
        metric = create_metric("accuracy", normalize=False)
        assert isinstance(metric, Accuracy)
        assert metric.normalize is False

    def test_create_bleu(self):
        """Test creating BLEUScore metric."""
        metric = create_metric("bleu", n_gram=2)
        assert isinstance(metric, BLEUScore)
        assert metric.n_gram == 2

    def test_create_rouge(self):
        """Test creating ROUGEScore metric."""
        metric = create_metric("rouge", field="text")
        assert isinstance(metric, ROUGEScore)
        assert metric.field == "text"

    def test_create_unknown_metric(self):
        """Test error for unknown metric type."""
        with pytest.raises(ValueError, match="Unknown metric type"):
            create_metric("unknown_metric")
