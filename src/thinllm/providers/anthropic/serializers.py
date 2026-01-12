"""Serializers for converting internal types to Anthropic API format."""

from collections.abc import Callable
from typing import Any

from thinllm.messages import (
    AIMessage,
    ContentBlock,
    InputImageBlock,
    InputTextBlock,
    MessageType,
    OutputTextBlock,
    ReasoningContent,
    SystemMessage,
    ToolCallContent,
    ToolResultContent,
    UserMessage,
)
from thinllm.tools import Tool


def _add_cache_control(result_dict: dict, block: ContentBlock) -> None:
    """Add cache_control to result dict if present and enabled in block.extra."""
    if (
        block.extra
        and block.extra.anthropic_cache_control
        and block.extra.anthropic_cache_control.enabled
    ):
        cache_control_dict = {"type": "ephemeral"}
        if block.extra.anthropic_cache_control.ttl:
            cache_control_dict["ttl"] = block.extra.anthropic_cache_control.ttl
        result_dict["cache_control"] = cache_control_dict


def _convert_content_block_to_anthropic_dict(block: ContentBlock) -> dict:  # noqa: C901
    """
    Convert a single ContentBlock to Anthropic content object format.

    Args:
        block: A ContentBlock (InputTextBlock, OutputTextBlock, etc.)

    Returns:
        Dictionary in Anthropic content object format

    Raises:
        ValueError: If an unsupported content block type is provided
    """
    match block:
        case InputTextBlock():
            result = {
                "type": "text",
                "text": block.text,
            }
            _add_cache_control(result, block)
            return result
        case OutputTextBlock():
            result = {
                "type": "text",
                "text": block.text,
            }
            _add_cache_control(result, block)
            return result
        case ReasoningContent():
            # Anthropic uses ThinkingBlock for reasoning
            if block.redacted_content:
                return {
                    "type": "redacted_thinking",
                    "data": block.redacted_content,
                }
            # Combine summaries and contents into thinking text
            thinking_text = "\n".join(block.summaries + block.contents)
            return {
                "type": "thinking",
                "signature": block.signature,
                "thinking": thinking_text,
            }
        case ToolCallContent():
            result = {
                "type": "tool_use",
                "id": block.tool_id,
                "name": block.name,
                "input": block.input,
            }
            _add_cache_control(result, block)
            return result
        case ToolResultContent():
            result = {
                "type": "tool_result",
                "tool_use_id": block.tool_id,
                "content": block.output,
                "is_error": block.status.value == "failure",
            }
            _add_cache_control(result, block)
            return result
        case InputImageBlock():
            import base64

            # Validate mimetype for Anthropic
            supported_mimetypes = {"image/jpeg", "image/png", "image/gif", "image/webp"}
            if block.mimetype and block.mimetype not in supported_mimetypes:
                raise ValueError(
                    f"Unsupported image mimetype for Anthropic: {block.mimetype}. "
                    f"Supported types: {', '.join(supported_mimetypes)}"
                )

            # Handle URL-based images
            if block.image_url:
                result = {
                    "type": "image",
                    "source": {
                        "type": "url",
                        "url": block.image_url,
                    },
                }
                _add_cache_control(result, block)
                return result

            # Handle base64-encoded images
            if block.image_bytes:
                # Default to image/jpeg if no mimetype is provided
                media_type = block.mimetype or "image/jpeg"

                # Encode bytes to base64 string
                base64_data = base64.standard_b64encode(block.image_bytes).decode("utf-8")

                result = {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64_data,
                    },
                }
                _add_cache_control(result, block)
                return result

            # This shouldn't happen due to model validation, but just in case
            raise ValueError("InputImageBlock must have either image_url or image_bytes")
        case _:
            raise ValueError(f"Unsupported content block type for Anthropic: {type(block)}")


def _get_anthropic_messages(messages: list[MessageType]) -> list[dict[str, Any]]:
    """
    Convert internal message format to Anthropic API message format.

    This function transforms our internal message types (SystemMessage, UserMessage,
    AIMessage) into the format expected by Anthropic's API.

    Args:
        messages: List of internal message objects

    Returns:
        List of message dictionaries in Anthropic format

    Note:
        - System messages are excluded (handled separately as system parameter)
        - Messages must alternate between user and assistant roles
        - Content blocks are converted to Anthropic format
    """
    result = []
    for message in messages:
        match message:
            case SystemMessage():
                # System messages are sent as system parameter
                continue
            case UserMessage():
                if isinstance(message.content, str):
                    content_list = [{"type": "text", "text": message.content}]
                else:
                    content_list = [
                        _convert_content_block_to_anthropic_dict(block) for block in message.content
                    ]

                result.append(
                    {
                        "role": "user",
                        "content": content_list,
                    }
                )
            case AIMessage():
                if isinstance(message.content, str):
                    content_list = [{"type": "text", "text": message.content}]
                else:
                    content_list = [
                        _convert_content_block_to_anthropic_dict(block) for block in message.content
                    ]

                result.append(
                    {
                        "role": "assistant",
                        "content": content_list,
                    }
                )
            case _:
                raise TypeError(f"Unsupported message type for Anthropic: {type(message)}")
    return result


def _get_anthropic_tool(tool: Tool | Callable | dict) -> dict:
    """
    Convert a Tool, callable, or dict to Anthropic tool format.

    Transforms our internal Tool representation into the format expected
    by Anthropic's function calling API.

    Args:
        tool: Tool instance, callable function, or dict for built-in tools

    Returns:
        Dictionary in Anthropic tool format with name, description, and input_schema
    """
    if isinstance(tool, dict):
        # Pass through built-in tools if needed
        return tool
    if not isinstance(tool, Tool) and isinstance(tool, Callable):
        tool = Tool.from_function(tool)
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.args_schema.model_json_schema(),
    }


def _get_system_blocks_from_messages(messages: list[MessageType]) -> list[dict[str, Any]]:
    """
    Extract system blocks from messages with cache control support.

    Transforms SystemMessage content into a list of text blocks in Anthropic format,
    preserving cache control settings on individual blocks.

    Args:
        messages: List of messages to extract system instructions from

    Returns:
        List of text block dictionaries in Anthropic format with optional cache_control

    Raises:
        TypeError: If a system message contains non-text content blocks
    """
    system_blocks: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, SystemMessage):
            content = message.content
            match content:
                case str():
                    system_blocks.append({"type": "text", "text": content})
                case list():
                    # Handle list of ContentBlocks - convert InputTextBlocks to dicts
                    for block in content:
                        if isinstance(block, InputTextBlock):
                            block_dict = {"type": "text", "text": block.text}
                            _add_cache_control(block_dict, block)
                            system_blocks.append(block_dict)
                        else:
                            raise TypeError(
                                f"Unsupported content block type in system message: {type(block)}"
                            )
    return system_blocks
