"""Internal utility functions for LLM service interactions."""

import logging
import mimetypes
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

import jiter

if TYPE_CHECKING:
    from collections.abc import Callable

    from thinllm.messages import ToolCallContent, ToolOutputStatus, ToolResultContent
    from thinllm.tools import Tool

logger = logging.getLogger(__name__)

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


def normalize_tools(tools: "list[Tool | Callable | dict]") -> "list[Tool]":
    """
    Convert a list of mixed tool types into a list of Tool objects.

    This function normalizes various tool formats (Tool objects, callables, dicts)
    into a consistent list of Tool objects. Dict-based tools (like web_search)
    are filtered out as they're handled by LLM providers.

    Args:
        tools: List of tools that can include:
            - Tool objects
            - Callable functions (will be converted to Tool objects)
            - dict objects (filtered out, used for built-in provider tools)

    Returns:
        List of Tool objects

    Example:
        >>> from thinllm.tools import tool
        >>> @tool
        ... def add(a: int, b: int) -> int:
        ...     return a + b
        >>> def multiply(x: int, y: int) -> int:
        ...     return x * y
        >>> tool_list = normalize_tools([add, multiply])
        >>> [t.name for t in tool_list]
        ['add', 'multiply']
    """
    from thinllm.tools import Tool
    from thinllm.tools import tool as tool_decorator

    tool_list: list[Tool] = []

    for t in tools:
        match t:
            case Tool():
                tool_list.append(t)
            case dict():
                # Skip dict-based tools (like web_search) - they're handled by LLM provider
                pass
            case _ if callable(t):
                # Convert callable to Tool
                converted_tool = tool_decorator(t)
                tool_list.append(converted_tool)
            case _:
                logger.warning(f"Unknown tool type: {type(t)}, skipping")

    return tool_list


def _build_result_kwargs(
    tool_call: "ToolCallContent",
    output: str,
    status: "ToolOutputStatus",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build base kwargs dictionary for ToolResultContent."""
    result_kwargs = {
        "tool_id": tool_call.tool_id,
        "name": tool_call.name,
        "raw_input": tool_call.raw_input,
        "input": tool_call.input,
        "output": output,
        "status": status,
    }
    if tool_call.id:
        result_kwargs["id"] = tool_call.id
    if metadata:
        result_kwargs["metadata"] = metadata
    return result_kwargs


def _create_tool_error_result(
    tool_call: "ToolCallContent",
    error_msg: str,
    metadata: dict[str, Any] | None = None,
) -> "ToolResultContent":
    """Create a ToolResultContent for an error."""
    from thinllm.messages import ToolOutputStatus, ToolResultContent

    result_kwargs = _build_result_kwargs(
        tool_call,
        output=error_msg,
        status=ToolOutputStatus.FAILURE,
        metadata=metadata,
    )
    return ToolResultContent(**result_kwargs)


def _build_tool_result(tool_call: "ToolCallContent", result: Any) -> "ToolResultContent":
    """Build a ToolResultContent from the raw execution result."""
    from thinllm.messages import ToolOutput, ToolOutputStatus, ToolResultContent

    # Normalize the return value
    match result:
        case None:
            output_text = ""
            metadata = {}
            status = ToolOutputStatus.SUCCESS
        case str():
            output_text = result
            metadata = {}
            status = ToolOutputStatus.SUCCESS
        case ToolOutput():
            output_text = result.text
            metadata = result.metadata
            status = result.status
        case _:
            # Convert to string for other types
            output_text = str(result)
            metadata = {}
            status = ToolOutputStatus.SUCCESS

    result_kwargs = _build_result_kwargs(
        tool_call,
        output=output_text,
        status=status,
        metadata=metadata if metadata else None,
    )
    return ToolResultContent(**result_kwargs)


def load_image_from_file(filepath: str) -> tuple[bytes, str]:
    """
    Load an image from a file path and detect its mimetype.

    Args:
        filepath: Path to the image file

    Returns:
        Tuple of (image_bytes, mimetype)

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the mimetype cannot be determined

    Example:
        >>> image_bytes, mimetype = load_image_from_file("path/to/image.jpg")
        >>> print(mimetype)
        'image/jpeg'
    """
    file_path = Path(filepath)

    if not file_path.exists():
        raise FileNotFoundError(f"Image file not found: {filepath}")

    # Read the image file in binary mode
    with file_path.open("rb") as f:
        image_bytes = f.read()

    # Detect mimetype from file extension
    mimetype, _ = mimetypes.guess_type(str(file_path))

    # If mimetype couldn't be determined, try to infer from extension
    if mimetype is None:
        ext = file_path.suffix.lower()
        # Common image mimetypes
        mimetype_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
            ".tiff": "image/tiff",
            ".tif": "image/tiff",
            ".svg": "image/svg+xml",
        }
        mimetype = mimetype_map.get(ext)

    if mimetype is None:
        raise ValueError(
            f"Could not determine mimetype for file: {filepath}. "
            f"Ensure the file has a valid image extension."
        )

    return image_bytes, mimetype


def get_tool_result(
    tool_call: "ToolCallContent",
    tools: "list[Tool | Callable | dict]",
) -> "ToolResultContent":
    """
    Execute a tool call and return the result.

    Args:
        tool_call: The tool call content block to execute
        tools: List of tools available for execution. Can include:
            - Tool objects
            - Callable functions
            - dict objects (e.g., for built-in tools like web_search)

    Returns:
        ToolResultContent with execution result

    Example:
        >>> from thinllm.tools import tool
        >>> from thinllm.messages import ToolCallContent
        >>> @tool
        ... def add(a: int, b: int) -> int:
        ...     return a + b
        >>> tool_call = ToolCallContent(name="add", input={"a": 1, "b": 2}, tool_id="1")
        >>> result = get_tool_result(tool_call, tools=[add])
        >>> print(result.output)
        "3"
    """
    # Normalize tools and build mapping
    normalized_tools = normalize_tools(tools)
    tool_map = {t.name: t for t in normalized_tools}

    # Check if tool exists
    if tool_call.name not in tool_map:
        error_msg = f"Tool '{tool_call.name}' not found. Available tools: {list(tool_map.keys())}"
        return _create_tool_error_result(tool_call, error_msg)

    tool_func = tool_map[tool_call.name]

    # Execute the tool
    try:
        result = tool_func(**tool_call.input)
        return _build_tool_result(tool_call, result)
    except Exception as e:
        error_msg = f"Error executing tool '{tool_call.name}': {str(e)}"
        return _create_tool_error_result(
            tool_call,
            error_msg,
            metadata={"error": str(e), "error_type": type(e).__name__},
        )
