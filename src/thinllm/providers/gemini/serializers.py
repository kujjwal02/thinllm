"""Serializers for converting internal types to Gemini API format."""

import base64
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


def _image_bytes_to_base64(image_bytes: bytes) -> str:
    """
    Convert image bytes to base64 string.

    Args:
        image_bytes: Raw image bytes

    Returns:
        Base64 encoded string
    """
    return base64.b64encode(image_bytes).decode("utf-8")


def _serialize_image_to_gemini(image_block: InputImageBlock) -> dict:
    """
    Convert InputImageBlock to Gemini part format.
    
    Reuses the existing logic for image serialization.
    
    Args:
        image_block: InputImageBlock to serialize
        
    Returns:
        Dictionary in Gemini part format
        
    Raises:
        ValueError: If neither image_bytes nor image_url is provided
    """
    if image_block.image_bytes:
        # Use inline_data for bytes
        base64_data = _image_bytes_to_base64(image_block.image_bytes)
        return {
            "inline_data": {
                "mime_type": image_block.mimetype or "image/jpeg",
                "data": base64_data,
            }
        }
    elif image_block.image_url:
        # Use file_data for URLs
        return {
            "file_data": {
                "file_uri": image_block.image_url,
            }
        }

    raise ValueError("InputImageBlock must have either image_bytes or image_url")


def _convert_content_block_to_gemini_parts(block: ContentBlock) -> list[dict]:
    """
    Convert a single ContentBlock to Gemini part format.

    Args:
        block: A ContentBlock (InputTextBlock, OutputTextBlock, etc.)

    Returns:
        List of dictionaries in Gemini part format. Most blocks return a single-item list,
        but ReasoningContent with multiple summaries returns multiple parts.

    Raises:
        ValueError: If an unsupported content block type is provided
    """
    # Create the thought signature variable
    thought_sig = (
        block.extra.gemini_thought_signature
        if hasattr(block, "extra")
        and block.extra
        and hasattr(block.extra, "gemini_thought_signature")
        else None
    )
    match block:
        case InputTextBlock():
            # You may attach thought_signature if required; specification doesn't do it.
            return [{"text": block.text}]
        case OutputTextBlock():
            part: dict[str, Any] = {"text": block.text}
            if thought_sig:
                part["thought_signature"] = thought_sig
            return [part]
        case InputImageBlock():
            return [_serialize_image_to_gemini(block)]
        case ReasoningContent():
            # Gemini uses thought as a boolean flag with text containing the summary
            # If we have redacted content, use that
            if block.redacted_content:
                part = {
                    "text": block.redacted_content,
                    "thought": True,
                }
                if thought_sig:
                    part["thought_signature"] = thought_sig
                return [part]
            # Create separate parts for each summary using a list comprehension
            parts = []
            for summary in block.summaries:
                part = {
                    "text": summary,
                    "thought": True,
                }
                if thought_sig:
                    part["thought_signature"] = thought_sig
                parts.append(part)
            # If there are contents but no summaries, combine them into one part
            if not parts and block.contents:
                part = {
                    "text": "\n".join(block.contents),
                    "thought": True,
                }
                if thought_sig:
                    part["thought_signature"] = thought_sig
                parts.append(part)
            return parts
        case ToolCallContent():
            fc_part: dict[str, Any] = {
                "function_call": {
                    "name": block.name,
                    "args": block.input,
                }
            }
            if thought_sig:
                fc_part["thought_signature"] = thought_sig
            return [fc_part]
        case ToolResultContent():
            # Gemini supports structured response data
            # Use match/case for output handling
            match block.output:
                case str():
                    response_data = {"result": block.output}
                case list():
                    # Build structured response with multiple parts
                    result_parts = []
                    for item in block.output:
                        match item:
                            case OutputTextBlock() | InputTextBlock():
                                result_parts.append(item.text)
                            case InputImageBlock():
                                # Use the refactored helper function
                                result_parts.append(_serialize_image_to_gemini(item))
                    response_data = {"result": result_parts} if result_parts else {"result": ""}
                case None:
                    response_data = {"result": ""}
                case _:
                    response_data = {"result": ""}

            return [
                {
                    "function_response": {
                        "name": block.name,
                        "response": response_data,
                    }
                }
            ]
        case _:
            raise ValueError(f"Unsupported content block type for Gemini: {type(block)}")


def _get_gemini_contents(messages: list[MessageType]) -> list[dict[str, Any]]:
    """
    Convert internal message format to Gemini API contents format.

    This function transforms our internal message types (SystemMessage, UserMessage,
    AIMessage) into the format expected by Gemini's API.

    Args:
        messages: List of internal message objects

    Returns:
        List of content dictionaries in Gemini format

    Note:
        - System messages are excluded (handled separately as system_instruction)
        - User messages map to role="user"
        - AI messages map to role="model"
        - Content blocks are converted to Gemini parts
    """
    result = []
    for message in messages:
        match message:
            case SystemMessage():
                # System messages are sent as system_instruction parameter
                continue
            case UserMessage():
                if isinstance(message.content, str):
                    parts = [{"text": message.content}]
                else:
                    # Convert blocks to parts, flattening lists
                    parts = []
                    for block in message.content:
                        parts.extend(_convert_content_block_to_gemini_parts(block))

                result.append(
                    {
                        "role": "user",
                        "parts": parts,
                    }
                )
            case AIMessage():
                if isinstance(message.content, str):
                    parts = [{"text": message.content}]
                else:
                    # Convert blocks to parts, flattening lists
                    parts = []
                    for block in message.content:
                        parts.extend(_convert_content_block_to_gemini_parts(block))

                result.append(
                    {
                        "role": "model",
                        "parts": parts,
                    }
                )
    return result


def _get_gemini_tool(tool: Tool | Callable | dict) -> dict:
    """
    Convert a Tool, callable, or dict to Gemini tool format.

    Transforms our internal Tool representation into the format expected
    by Gemini's function calling API.

    Args:
        tool: Tool instance, callable function, or dict for built-in tools

    Returns:
        Dictionary in Gemini tool format with function_declarations or built-in tool config
    """
    if isinstance(tool, dict):
        # Pass through built-in tools like {"google_search": {}}
        return tool
    if not isinstance(tool, Tool) and isinstance(tool, Callable):
        tool = Tool.from_function(tool)

    # Gemini expects function_declarations format
    return {
        "function_declarations": [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.args_schema.model_json_schema(),
            }
        ]
    }


def _get_system_instruction(messages: list[MessageType]) -> str:
    """
    Extract system instructions from messages.

    Combines all SystemMessage content into a single system instruction string.

    Args:
        messages: List of messages to extract system instructions from

    Returns:
        Combined system instruction text from all system messages

    Raises:
        TypeError: If a system message contains non-text content blocks
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
