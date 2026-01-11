"""Serializers for converting internal types to Anthropic API format."""

from collections.abc import Callable
from typing import Any

from thinllm.messages import (
    AIMessage,
    ContentBlock,
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


def _convert_content_block_to_anthropic_dict(block: ContentBlock) -> dict:
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
            return {
                "type": "text",
                "text": block.text,
            }
        case OutputTextBlock():
            return {
                "type": "text",
                "text": block.text,
            }
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
            return {
                "type": "tool_use",
                "id": block.tool_id,
                "name": block.name,
                "input": block.input,
            }
        case ToolResultContent():
            return {
                "type": "tool_result",
                "tool_use_id": block.tool_id,
                "content": block.output,
                "is_error": block.status.value == "failure",
            }
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


def _get_system_from_messages(messages: list[MessageType]) -> str:
    """
    Extract system instructions from messages.

    Combines all SystemMessage content into a single system string.

    Args:
        messages: List of messages to extract system instructions from

    Returns:
        Combined system text from all system messages

    Raises:
        ValueError: If a system message contains non-text content blocks
    """
    system_parts: list[str] = []
    for message in messages:
        if isinstance(message, SystemMessage):
            content = message.content
            match content:
                case str():
                    system_parts.append(content)
                case list():
                    # Handle list of ContentBlocks - extract text from InputTextBlocks
                    text_blocks = [
                        block.text for block in content if isinstance(block, InputTextBlock)
                    ]
                    # Check if all blocks were InputTextBlocks
                    if len(text_blocks) != len(content):
                        for block in content:
                            if not isinstance(block, InputTextBlock):
                                raise TypeError(
                                    f"Unsupported content block type in system message: {type(block)}"
                                )
                    system_parts.extend(text_blocks)
    return "\n".join(system_parts)
