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
    WebSearchCallContent,
)
from thinllm.tools import Tool


def _image_bytes_to_base64_url(image_bytes: bytes) -> str:
    """
    Convert image bytes to base64 data URL.

    Args:
        image_bytes: Raw image bytes

    Returns:
        Base64 encoded data URL string
    """
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{base64_image}"


def _convert_content_block_to_oai_dict(block: ContentBlock) -> dict:
    """
    Convert a single ContentBlock to OpenAI content object format.

    Args:
        block: A ContentBlock (InputTextBlock, OutputTextBlock, InputImageBlock, etc.)

    Returns:
        Dictionary in OpenAI content object format

    Raises:
        ValueError: If InputImageBlock has neither image_bytes nor image_url
    """
    match block:
        case InputTextBlock():
            return {
                "type": "input_text",
                "text": block.text,
            }
        case OutputTextBlock():
            return {
                "type": "output_text",
                "text": block.text,
            }
        case InputImageBlock():
            # Validate that either image_bytes or image_url is present
            if block.image_bytes:
                image_url = _image_bytes_to_base64_url(block.image_bytes)
            elif block.image_url:
                image_url = block.image_url
            else:
                raise ValueError(
                    "InputImageBlock must have either image_bytes or a non-empty image_url"
                )

            return {
                "type": "input_image",
                "image_url": image_url,
                "detail": block.detail.value,
            }
        case ReasoningContent():
            return {
                "type": "reasoning",
                "id": block.id,
                "summary": [{"text": text, "type": "summary_text"} for text in block.summaries],
                "content": [{"text": text, "type": "reasoning_text"} for text in block.contents],
            }
        case WebSearchCallContent():
            action: dict[str, Any] = {
                "query": block.query,
                "type": "search",
                "sources": [{"url": source, "type": "url"} for source in block.sources]
                if block.sources
                else None,
            }
            return {
                "id": block.id,
                "action": action,
                "status": "completed",
                "type": "web_search_call",
            }
        case ToolCallContent():
            return {
                "type": "function_call",
                "id": block.id,
                "call_id": block.tool_id,
                "name": block.name,
                "arguments": block.raw_input,
            }
        case ToolResultContent():
            return {
                "type": "function_call_output",
                "call_id": block.tool_id,
                "output": block.output,
            }
        case _:
            raise ValueError(f"Unknown content block type: {type(block)}")


def _convert_content_to_oai_format(
    content: str | list[ContentBlock],
) -> str | list[dict]:
    """
    Convert content (string or list of ContentBlocks) to OpenAI API format.

    Args:
        content: Message content as string or list of ContentBlocks

    Returns:
        - For string: returns string as-is
        - For list of ContentBlocks: returns array of content objects

    Raises:
        ValueError: If InputImageBlock has neither image_bytes nor image_url,
                   or if image_url is an empty string, or if list is empty
    """
    match content:
        case str():
            return content
        case list():
            # List of ContentBlocks - convert to array of content objects
            if not content:
                raise ValueError("Content list cannot be empty")
            return [_convert_content_block_to_oai_dict(block) for block in content]
        case _:
            raise ValueError(f"Unknown content type: {type(content)}")


def _get_oai_messages(messages: list[MessageType]) -> list[dict[str, Any]]:
    """
    Convert internal message format to OpenAI API message format.

    This function transforms our internal message types (SystemMessage, UserMessage,
    AIMessage) into the format expected by OpenAI's API.

    Args:
        messages: List of internal message objects

    Returns:
        List of message dictionaries in OpenAI format

    Note:
        - System messages are excluded (handled separately as instructions)
        - User messages accept: InputTextBlock and InputImageBlock as regular content (with role)
        - Assistant messages accept: OutputTextBlock only as regular content (with role)
        - ReasoningContent, WebSearchCallContent, ToolCallContent, and ToolResultContent
          are separated into their own message entries (without role)
        - Order is preserved across all content types

    Raises:
        TypeError: If OutputTextBlock is used in UserMessage or InputTextBlock/InputImageBlock
                  is used in AIMessage
    """
    result = []
    for message in messages:
        match message:
            case SystemMessage():
                # System messages are sent as instructions to the model
                continue
            case UserMessage() | AIMessage():
                if message.content:
                    content = message.content
                    if isinstance(content, str):
                        # Simple string content - in this case, use message.id since there's no content block
                        msg_dict = {
                            "role": "user" if isinstance(message, UserMessage) else "assistant",
                            "content": content,
                        }
                        # Add id and type for assistant messages
                        if isinstance(message, AIMessage):
                            msg_dict["type"] = "message"
                            if hasattr(message, "id") and message.id:
                                msg_dict["id"] = message.id
                        result.append(msg_dict)
                    else:
                        # List of ContentBlocks - preserve order
                        # User messages: InputTextBlock and InputImageBlock are regular content (with role)
                        # Assistant messages: Only OutputTextBlock is regular content (with role)
                        # All others are separate messages (without role)
                        role = "user" if isinstance(message, UserMessage) else "assistant"
                        accumulated_content: list[ContentBlock] = []

                        for block in content:
                            # Check if this block should be accumulated based on role
                            is_regular_content = False
                            if isinstance(message, UserMessage):
                                # User messages accept InputTextBlock and InputImageBlock
                                if isinstance(block, OutputTextBlock):
                                    raise TypeError(
                                        "OutputTextBlock is not allowed in UserMessage. "
                                        "Use InputTextBlock instead."
                                    )
                                is_regular_content = isinstance(
                                    block, (InputTextBlock, InputImageBlock)
                                )
                            else:  # AIMessage
                                # Assistant messages only accept OutputTextBlock
                                if isinstance(block, (InputTextBlock, InputImageBlock)):
                                    raise TypeError(
                                        f"{type(block).__name__} is not allowed in AIMessage. "
                                        "Use OutputTextBlock instead."
                                    )
                                is_regular_content = isinstance(block, OutputTextBlock)

                            if is_regular_content:
                                # Accumulate regular content blocks
                                accumulated_content.append(block)
                            else:
                                # For all other types (reasoning, web_search, tool_call, tool_result)
                                # Flush accumulated regular content first
                                if accumulated_content:
                                    converted_content = [
                                        _convert_content_block_to_oai_dict(b)
                                        for b in accumulated_content
                                    ]
                                    msg_dict = {"role": role, "content": converted_content}
                                    # Add id and type for assistant messages
                                    if role == "assistant":
                                        msg_dict["type"] = "message"
                                        # Use the first content block's ID if available
                                        if accumulated_content and accumulated_content[0].id:
                                            msg_dict["id"] = accumulated_content[0].id
                                    result.append(msg_dict)
                                    accumulated_content = []

                                # Add as separate message without role
                                result.append(_convert_content_block_to_oai_dict(block))

                        # Flush any remaining accumulated content
                        if accumulated_content:
                            converted_content = [
                                _convert_content_block_to_oai_dict(b) for b in accumulated_content
                            ]
                            msg_dict = {"role": role, "content": converted_content}
                            # Add id and type for assistant messages
                            if role == "assistant":
                                msg_dict["type"] = "message"
                                # Use the first content block's ID if available
                                if accumulated_content and accumulated_content[0].id:
                                    msg_dict["id"] = accumulated_content[0].id
                            result.append(msg_dict)
    return result


def _get_oai_tool(tool: Tool | Callable | dict) -> dict:
    """
    Convert a Tool, callable, or dict to OpenAI tool format.

    Transforms our internal Tool representation or built-in tools (like web_search)
    into the format expected by OpenAI's function calling API.

    Args:
        tool: Tool instance, callable function, or dict for built-in tools

    Returns:
        Dictionary in OpenAI tool format with type, name, description, and parameters
    """
    if isinstance(tool, dict):
        # Pass through built-in tools like {"type": "web_search"}
        return tool
    if not isinstance(tool, Tool) and isinstance(tool, Callable):
        tool = Tool.from_function(tool)
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.args_schema.model_json_schema(),
    }


def _get_instruction_from_messages(messages: list[MessageType]) -> str:
    """
    Extract system instructions from messages.

    Combines all SystemMessage content into a single instruction string.

    Args:
        messages: List of messages to extract instructions from

    Returns:
        Combined instruction text from all system messages

    Raises:
        ValueError: If a system message contains an InputImageBlock or other non-text content (not supported by OpenAI)
    """
    instructions: list[str] = []
    for message in messages:
        if isinstance(message, SystemMessage):
            content = message.content
            match content:
                case str():
                    instructions.append(content)
                case list():
                    # Handle list of ContentBlocks
                    for block in content:
                        match block:
                            case InputTextBlock():
                                instructions.append(block.text)
                            case _:
                                raise ValueError(
                                    f"Unknown content block type: {type(block)}, only InputTextBlock is supported in system messages for OpenAI provider"
                                )
    return "\n".join(instructions)
