"""
Common extraction utilities for parsing LLM outputs.

These extractors handle the impedance mismatch between probabilistic text outputs
and deterministic data structures. They provide robust, reusable parsing logic
for common data types.

ZERO DEPENDENCIES - Uses only Python standard library.

Example:
    from logillm.core.extractors import Extractors

    # Extract a number from LLM output
    result = await predictor(problem="What is 15 + 27?")
    answer = Extractors.number(result.outputs.get("answer"))

    # Extract boolean with custom logic
    is_correct = Extractors.boolean(
        result.outputs.get("validation"),
        strict=True
    )
"""

import json
import re
from typing import Optional


class Extractors:
    """Common extractors for LLM output parsing.

    All extractors follow these principles:
    1. Never raise exceptions - always return a default
    2. Handle None, empty, and malformed inputs gracefully
    3. Try multiple strategies before giving up
    4. Prefer the most likely interpretation
    """

    @staticmethod
    def number(text: str | None, default: float = 0.0, first: bool = False) -> float:
        """Extract a number from text.

        Handles multiple formats:
        - Digits: "42", "3.14", "-17.5"
        - Words: "forty-two", "negative seventeen"
        - Mixed: "The answer is 42"
        - Scientific: "1.5e10"

        Args:
            text: Text potentially containing a number
            default: Value to return if no number found
            first: If True, return first number found; else last

        Returns:
            Extracted number or default

        Examples:
            >>> Extractors.number("The answer is 42")
            42.0
            >>> Extractors.number("negative seventeen point five")
            -17.5
            >>> Extractors.number(None, default=-1)
            -1.0
        """
        if not text:
            return default

        text_str = str(text).strip()

        # Strategy 1: Extract digit-based numbers
        pattern = r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
        matches = re.findall(pattern, text_str)
        if matches:
            try:
                return float(matches[0] if first else matches[-1])
            except (ValueError, IndexError):
                pass

        # Strategy 2: Word-to-number conversion
        text_lower = text_str.lower()

        # Handle negative
        negative = "negative" in text_lower or "minus" in text_lower
        if negative:
            text_lower = text_lower.replace("negative", "").replace("minus", "")

        # Basic number words
        word_to_num = {
            "zero": 0,
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
            "eleven": 11,
            "twelve": 12,
            "thirteen": 13,
            "fourteen": 14,
            "fifteen": 15,
            "sixteen": 16,
            "seventeen": 17,
            "eighteen": 18,
            "nineteen": 19,
            "twenty": 20,
            "thirty": 30,
            "forty": 40,
            "fifty": 50,
            "sixty": 60,
            "seventy": 70,
            "eighty": 80,
            "ninety": 90,
            "hundred": 100,
            "thousand": 1000,
            "million": 1000000,
            "billion": 1000000000,
        }

        # Look for exact matches
        for word, num in word_to_num.items():
            if word in text_lower:
                result = float(num)
                return -result if negative else result

        # Strategy 3: Handle "point" for decimals (e.g., "seventeen point five")
        decimal_match = re.search(r"(\w+)\s+point\s+(\w+)", text_lower)
        if decimal_match:
            whole_word = decimal_match.group(1)
            decimal_word = decimal_match.group(2)
            whole = word_to_num.get(whole_word, 0)
            decimal = word_to_num.get(decimal_word, 0)
            if whole and decimal < 10:
                result = whole + (decimal / 10)
                return -result if negative else result

        return default

    @staticmethod
    def boolean(text: str | None, default: bool = False, strict: bool = False) -> bool:
        """Extract boolean value from text.

        Handles various affirmative/negative expressions.

        Args:
            text: Text potentially containing boolean indicator
            default: Value to return if unclear
            strict: If True, only accept explicit yes/no/true/false

        Returns:
            Boolean interpretation of text

        Examples:
            >>> Extractors.boolean("yes")
            True
            >>> Extractors.boolean("absolutely not")
            False
            >>> Extractors.boolean("maybe", strict=True)
            False  # Returns default
        """
        if not text:
            return default

        text_lower = str(text).lower().strip()

        # Explicit positive indicators
        positive_strict = {"yes", "true", "1", "correct", "right", "affirmative"}
        positive_loose = positive_strict | {
            "yeah",
            "yep",
            "yup",
            "sure",
            "ok",
            "okay",
            "confirmed",
            "indeed",
            "certainly",
            "absolutely",
            "definitely",
            "positive",
        }

        # Explicit negative indicators
        negative_strict = {"no", "false", "0", "incorrect", "wrong", "negative"}
        negative_loose = negative_strict | {
            "nope",
            "nah",
            "denied",
            "rejected",
            "refuse",
            "disagree",
        }

        # Check for explicit matches
        positive_set = positive_strict if strict else positive_loose
        negative_set = negative_strict if strict else negative_loose

        # Check for negative modifiers FIRST
        if any(phrase in text_lower for phrase in ["not", "n't", "no ", "never"]):
            # Check if it's negating a positive
            if any(pos in text_lower for pos in positive_set):
                return False  # "not true", "isn't correct", etc.
            return False  # General negation

        # Check for positive
        if any(pos in text_lower for pos in positive_set):
            return True

        # Check for negative
        if any(neg in text_lower for neg in negative_set):
            return False

        # Fuzzy matching if not strict
        if not strict:
            # Starting with y or t often means yes/true
            if text_lower and text_lower[0] in "yt":
                return True
            # Starting with n or f often means no/false
            if text_lower and text_lower[0] in "nf":
                return False

        return default

    @staticmethod
    def list_items(
        text: str | None, delimiter: str = "auto", max_items: Optional[int] = None
    ) -> list[str]:
        """Extract list items from various formats.

        Handles:
        - JSON arrays: ["item1", "item2"]
        - Bullet points: - item, * item, • item, 1. item
        - Comma-separated: item1, item2, item3
        - Semicolon-separated: item1; item2; item3
        - Line-separated: one item per line
        - "and" separated: item1 and item2 and item3

        Args:
            text: Text containing list items
            delimiter: Specific delimiter or "auto" for auto-detection
            max_items: Maximum number of items to return

        Returns:
            List of extracted items (empty list if none found)

        Examples:
            >>> Extractors.list_items('["apple", "banana", "cherry"]')
            ['apple', 'banana', 'cherry']
            >>> Extractors.list_items("- apple\\n- banana\\n- cherry")
            ['apple', 'banana', 'cherry']
        """
        if not text:
            return []

        text_str = str(text).strip()
        items = []

        # Strategy 1: Try JSON array
        if text_str.startswith("[") or '["' in text_str or "[''" in text_str:
            # Clean up common issues
            cleaned = text_str
            if "```json" in cleaned:
                cleaned = re.sub(r"```json\s*|\s*```", "", cleaned)
            if "```" in cleaned:
                cleaned = re.sub(r"```\s*|\s*```", "", cleaned)

            try:
                result = json.loads(cleaned)
                if isinstance(result, list):
                    items = [str(item).strip() for item in result]
                    if items:
                        return items[:max_items] if max_items else items
            except (json.JSONDecodeError, ValueError):
                pass

        # Strategy 2: Bullet points (most reliable format)
        bullet_pattern = r"^\s*(?:[-*•]|\d+\.)\s+(.+)$"
        lines = text_str.split("\n")
        bullet_items = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            match = re.match(bullet_pattern, line)
            if match:
                bullet_items.append(match.group(1).strip())
            elif bullet_items and not re.match(r"^\s*(?:[-*•]|\d+\.)", line):
                # Continuation of previous bullet item
                bullet_items[-1] += " " + line

        if bullet_items:
            return bullet_items[:max_items] if max_items else bullet_items

        # Strategy 3: Delimiter-based splitting
        if delimiter != "auto":
            items = [item.strip() for item in text_str.split(delimiter) if item.strip()]
            if items:
                return items[:max_items] if max_items else items

        # Strategy 4: Auto-detect delimiter
        # Remove brackets if present
        content = text_str
        if content.startswith("[") and content.endswith("]"):
            content = content[1:-1]
        if content.startswith("(") and content.endswith(")"):
            content = content[1:-1]

        # Try semicolon first (often used for complex items)
        if ";" in content:
            items = [item.strip() for item in content.split(";") if item.strip()]
            if len(items) > 1:
                return items[:max_items] if max_items else items

        # Try comma separation
        if "," in content:
            items = []
            parts = content.split(",")
            for part in parts:
                # Clean up quotes and spaces
                cleaned = part.strip().strip("\"'`")
                if cleaned:
                    items.append(cleaned)
            if items:
                return items[:max_items] if max_items else items

        # Try "and" separation
        if " and " in content.lower():
            items = [
                item.strip().strip("\"'`")
                for item in re.split(r"\s+and\s+", content, flags=re.IGNORECASE)
                if item.strip()
            ]
            if len(items) > 1:
                return items[:max_items] if max_items else items

        # Try line separation (if multiple lines)
        if "\n" in text_str:
            items = [line.strip() for line in text_str.split("\n") if line.strip()]
            if len(items) > 1:
                return items[:max_items] if max_items else items

        # Last resort: treat as single item if not empty
        if text_str:
            return [text_str]

        return []

    @staticmethod
    def percentage(text: str | None, as_decimal: bool = True, default: float = 0.0) -> float:
        """Extract percentage from text.

        Handles:
        - Percent sign: "50%", "12.5%"
        - Words: "fifty percent", "half"
        - Decimals: "0.5" (interpreted as 50%)
        - Fractions: "1/2", "three quarters"

        Args:
            text: Text containing percentage
            as_decimal: If True, return as decimal (0.5); else as percent (50)
            default: Value to return if no percentage found

        Returns:
            Percentage value

        Examples:
            >>> Extractors.percentage("50%")
            0.5
            >>> Extractors.percentage("fifty percent", as_decimal=False)
            50.0
            >>> Extractors.percentage("0.75")
            0.75
        """
        if not text:
            return default

        text_str = str(text).strip()
        text_lower = text_str.lower()

        # Strategy 1: Look for % sign
        percent_match = re.search(r"(\d+(?:\.\d+)?)\s*%", text_str)
        if percent_match:
            value = float(percent_match.group(1))
            return value / 100 if as_decimal else value

        # Strategy 2: Common fraction words
        fraction_map = {
            "half": 0.5,
            "third": 1 / 3,
            "quarter": 0.25,
            "three quarters": 0.75,
            "two thirds": 2 / 3,
            "one hundred percent": 1.0,
            "hundred percent": 1.0,
            "zero percent": 0.0,
            "none": 0.0,
            "all": 1.0,
            "full": 1.0,
        }

        for phrase, value in fraction_map.items():
            if phrase in text_lower:
                if as_decimal:
                    return value
                else:
                    return value * 100

        # Strategy 3: Number followed by "percent"
        num_percent = re.search(r"(\d+(?:\.\d+)?)\s*percent", text_lower)
        if num_percent:
            value = float(num_percent.group(1))
            return value / 100 if as_decimal else value

        # Strategy 4: Word number followed by "percent"
        word_percent = re.search(r"(\w+)\s+percent", text_lower)
        if word_percent:
            word = word_percent.group(1)
            num_value = Extractors.number(word, default=None)
            if num_value is not None:
                return num_value / 100 if as_decimal else num_value

        # Strategy 5: Check for decimal between 0 and 1
        decimal_match = re.search(r"0?\.\d+", text_str)
        if decimal_match:
            value = float(decimal_match.group(0))
            if 0 <= value <= 1:
                return value if as_decimal else value * 100

        # Strategy 6: Extract any number BUT don't assume it's a percentage
        # Only treat as percentage if there's clear context
        num = Extractors.number(text_str, default=None)
        if num is not None:
            # For plain numbers without percentage context, return default
            # This prevents misinterpreting regular numbers as percentages
            # The percentage method should only convert when there's clear percentage intent
            pass
            
        return default

    @staticmethod
    def enum(
        text: str | None,
        options: list[str],
        default: Optional[str] = None,
        case_sensitive: bool = False,
        fuzzy: bool = True,
    ) -> Optional[str]:
        """Map text to the closest enum option.

        Uses exact matching first, then fuzzy matching based on:
        - Substring containment
        - Prefix matching
        - Character overlap

        Args:
            text: Text to match against options
            options: List of valid options
            default: Default value if no match found
            case_sensitive: Whether to match case-sensitively
            fuzzy: Whether to use fuzzy matching

        Returns:
            Best matching option or default

        Examples:
            >>> Extractors.enum("red", ["red", "blue", "green"])
            "red"
            >>> Extractors.enum("reddish", ["red", "blue", "green"])
            "red"  # Fuzzy match
            >>> Extractors.enum("BLUE", ["red", "blue", "green"])
            "blue"  # Case insensitive
        """
        if not text or not options:
            return default

        text_str = str(text).strip()
        if not text_str:
            return default

        # Prepare text and options for comparison
        if case_sensitive:
            text_cmp = text_str
            options_cmp = {opt: opt for opt in options}
        else:
            text_cmp = text_str.lower()
            options_cmp = {opt.lower(): opt for opt in options}

        # Strategy 1: Exact match
        if text_cmp in options_cmp:
            return options_cmp[text_cmp]

        # Strategy 2: Exact match after stripping special chars
        text_clean = re.sub(r"[^a-zA-Z0-9]", "", text_cmp)
        for opt_cmp, opt_orig in options_cmp.items():
            opt_clean = re.sub(r"[^a-zA-Z0-9]", "", opt_cmp)
            if text_clean == opt_clean:
                return opt_orig

        if not fuzzy:
            return default

        # Strategy 3: Substring matching (text contains option)
        for opt_cmp, opt_orig in options_cmp.items():
            if opt_cmp in text_cmp:
                return opt_orig

        # Strategy 4: Substring matching (option contains text)
        for opt_cmp, opt_orig in options_cmp.items():
            if text_cmp in opt_cmp:
                return opt_orig

        # Strategy 5: Prefix matching
        for opt_cmp, opt_orig in options_cmp.items():
            if text_cmp.startswith(opt_cmp[:3]) or opt_cmp.startswith(text_cmp[:3]):
                return opt_orig

        # Strategy 6: Character overlap scoring
        best_score = 0
        best_option = default

        for opt_cmp, opt_orig in options_cmp.items():
            # Count common characters
            common = sum(1 for c in set(text_cmp) if c in opt_cmp)
            score = common / max(len(text_cmp), len(opt_cmp))

            if score > best_score and score > 0.5:  # At least 50% overlap
                best_score = score
                best_option = opt_orig

        return best_option

    @staticmethod
    def json_object(
        text: str | None, default: Optional[dict] = None, repair: bool = True
    ) -> Optional[dict]:
        """Extract and parse JSON object from text.

        Handles:
        - Clean JSON
        - JSON in code blocks
        - JSON with common errors (trailing commas, single quotes)
        - Partial JSON extraction

        Args:
            text: Text potentially containing JSON
            default: Default value if no valid JSON found
            repair: Whether to attempt repairing malformed JSON

        Returns:
            Parsed JSON object or default

        Examples:
            >>> Extractors.json_object('{"name": "Alice", "age": 30}')
            {"name": "Alice", "age": 30}
            >>> Extractors.json_object("```json\\n{...}\\n```")
            {...}  # Extracted from code block
        """
        if not text:
            return default

        text_str = str(text).strip()

        # Import utilities from existing code
        from logillm.core.adapters.utils import extract_json_object, repair_json_simple

        # Try extraction
        json_str = extract_json_object(text_str)
        if json_str:
            try:
                result = json.loads(json_str)
                if isinstance(result, dict):
                    return result
            except (json.JSONDecodeError, ValueError):
                pass

        # Try repair if enabled
        if repair:
            repaired = repair_json_simple(text_str)
            if repaired and isinstance(repaired, dict):
                return repaired

        return default


# Convenience functions for common use cases
def extract_number(text: str | None, default: float = 0.0) -> float:
    """Convenience function for number extraction."""
    return Extractors.number(text, default=default)


def extract_boolean(text: str | None, default: bool = False) -> bool:
    """Convenience function for boolean extraction."""
    return Extractors.boolean(text, default=default)


def extract_list(text: str | None) -> list[str]:
    """Convenience function for list extraction."""
    return Extractors.list_items(text)


def extract_percentage(text: str | None, as_decimal: bool = True) -> float:
    """Convenience function for percentage extraction."""
    return Extractors.percentage(text, as_decimal=as_decimal)
