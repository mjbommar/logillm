"""
Parsing and extraction utilities for adapters.

ZERO DEPENDENCIES - Uses only Python standard library.
These utilities help adapters extract and parse structured data from LLM responses.
"""

import json
import re
from typing import Any, Optional


def extract_json_object(text: str) -> Optional[str]:
    """
    Extract a JSON object from text that may contain other content.

    Tries multiple strategies:
    1. Direct JSON parse
    2. Find JSON between first { and last }
    3. Find JSON between first [ and last ]
    4. Clean common issues (backticks, language markers)

    Args:
        text: Text potentially containing JSON

    Returns:
        Extracted JSON string or None if not found
    """
    if not text:
        return None

    # Strategy 1: Try direct parse
    try:
        json.loads(text)
        return text
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: Remove backticks and language indicators (like ```json)
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    if cleaned != text:
        try:
            json.loads(cleaned)
            return cleaned
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 3: Extract from first { or [ to last } or ]
    # Find all possible start positions
    obj_start = text.find("{")
    arr_start = text.find("[")

    if obj_start == -1 and arr_start == -1:
        return None

    # Try object extraction
    if obj_start >= 0:
        obj_end = text.rfind("}")
        if obj_end > obj_start:
            candidate = text[obj_start : obj_end + 1]
            try:
                json.loads(candidate)
                return candidate
            except (json.JSONDecodeError, ValueError):
                pass

    # Try array extraction
    if arr_start >= 0:
        arr_end = text.rfind("]")
        if arr_end > arr_start:
            candidate = text[arr_start : arr_end + 1]
            try:
                json.loads(candidate)
                return candidate
            except (json.JSONDecodeError, ValueError):
                pass

    return None


def repair_json_simple(text: str) -> Optional[dict[str, Any]]:
    """
    Simple JSON repair for common LLM output issues.

    Handles:
    - Trailing commas
    - Single quotes instead of double quotes
    - Unquoted keys (simple cases)
    - Missing closing brackets

    Args:
        text: Potentially malformed JSON string

    Returns:
        Parsed dict or None if repair fails
    """
    if not text:
        return None

    # First try standard extraction
    json_str = extract_json_object(text)
    if json_str:
        try:
            return json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            pass

    # Try to repair common issues
    repaired = text

    # Fix trailing commas (,] or ,})
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)

    # Fix single quotes (naive approach - won't handle escaped quotes properly)
    # Only do this if we see single quotes but no double quotes
    if "'" in repaired and '"' not in repaired:
        repaired = repaired.replace("'", '"')

    # Try to parse again
    json_str = extract_json_object(repaired)
    if json_str:
        try:
            return json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            pass

    # Last resort: Try to fix unquoted keys (simple regex)
    # Match word: pattern and quote the key
    repaired = re.sub(r"(\w+):\s*", r'"\1": ', repaired)

    json_str = extract_json_object(repaired)
    if json_str:
        try:
            return json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            pass

    return None


def parse_key_value_pairs(text: str, delimiter: str = ":") -> dict[str, str]:
    """
    Parse simple key-value pairs from text.

    Handles formats like:
    - "key1: value1\\nkey2: value2"
    - "key1=value1, key2=value2"

    Args:
        text: Text containing key-value pairs
        delimiter: Delimiter between key and value

    Returns:
        Dict of parsed key-value pairs
    """
    result = {}

    # Split by newlines or commas
    lines = re.split(r"[\n,]", text)

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Look for delimiter
        if delimiter in line:
            parts = line.split(delimiter, 1)
            if len(parts) == 2:
                key = parts[0].strip().strip("\"'")
                value = parts[1].strip().strip("\"'")
                if key:
                    result[key] = value

    return result


def extract_field_markers(text: str) -> dict[str, str]:
    """
    Extract field markers from structured formats.

    Looks for patterns like:
    - [[ ## field_name ## ]] content here
    - <field_name>content</field_name>
    - **field_name**: content

    Args:
        text: Text with field markers

    Returns:
        Dict of field names to content
    """
    result = {}

    # Pattern 1: [[ ## field_name ## ]] format
    pattern = r"\[\[\s*##\s*(\w+)\s*##\s*\]\]"
    parts = re.split(pattern, text)

    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            field_name = parts[i].strip()
            content = parts[i + 1].strip()

            # Remove the next field marker if it exists
            next_marker_match = re.search(pattern, content)
            if next_marker_match:
                content = content[: next_marker_match.start()].strip()

            if field_name and field_name != "completed":
                result[field_name] = content

    # Pattern 2: XML-style <field>content</field>
    xml_pattern = r"<(\w+)>(.*?)</\1>"
    for match in re.finditer(xml_pattern, text, re.DOTALL):
        field_name = match.group(1)
        content = match.group(2).strip()
        if field_name not in result:  # Don't override if already found
            result[field_name] = content

    # Pattern 3: Markdown-style **field**: content
    md_pattern = r"\*\*(\w+)\*\*:\s*([^\n]+)"
    for match in re.finditer(md_pattern, text):
        field_name = match.group(1).lower()
        content = match.group(2).strip()
        if field_name not in result:  # Don't override if already found
            result[field_name] = content

    return result


def validate_parsed_output(
    parsed: dict[str, Any], expected_fields: set[str], strict: bool = True
) -> tuple[bool, Optional[str]]:
    """
    Validate parsed output against expected fields.

    Args:
        parsed: Parsed output dict
        expected_fields: Set of expected field names
        strict: If True, require all fields to be present

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not parsed:
        return False, "No output parsed"

    parsed_fields = set(parsed.keys())

    if strict:
        missing = expected_fields - parsed_fields
        if missing:
            return False, f"Missing required fields: {missing}"

    extra = parsed_fields - expected_fields
    if extra and strict:
        return False, f"Unexpected fields found: {extra}"

    return True, None
