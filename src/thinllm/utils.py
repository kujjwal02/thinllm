"""Internal utility functions for LLM service interactions."""

import re
from typing import TypeAlias

import jiter

# Pattern for detecting incomplete Unicode escape sequences
PARTIAL_UNICODE_PATTERN = re.compile(r"\\u[0-9a-fA-F]{0,3}$")

# Type alias for JSON values
JSONTYPE: TypeAlias = dict | list | str | int | float | bool | None


def parse_partial_json(json_str: str) -> JSONTYPE:
    """
    Parse potentially incomplete JSON strings from streaming responses.

    This function handles partial JSON that may be received during streaming,
    including incomplete strings and Unicode escape sequences.

    Args:
        json_str: The JSON string to parse (may be incomplete)

    Returns:
        Parsed JSON value (dict, list, str, int, float, bool, or None)

    Note:
        Based on: https://github.com/globalaiplatform/langdiff/blob/eb072a1829844e3d8ef5d733e31ed0011c0c4870/py/src/langdiff/parser/parser.py#L34
    """
    if not json_str.strip():
        return ""

    if (
        json_str.endswith('"') and not json_str.endswith('\\"') and not json_str.endswith(':"')
    ) or (json_str.endswith("\\") and not json_str.endswith("\\\\")):
        json_str = json_str[:-1]
    else:
        # Workaround for https://github.com/pydantic/jiter/issues/207
        m = PARTIAL_UNICODE_PATTERN.search(json_str)
        if m:
            json_str = json_str[: -len(m.group(0))]

    return jiter.from_json(
        json_str.encode("utf-8"), cache_mode="keys", partial_mode="trailing-strings"
    )
