"""Deserializers for converting Anthropic API responses to internal types."""

from anthropic.types import Message as AnthropicMessage

from thinllm.messages import (
    AIMessage,
    ContentBlock,
    OutputTextBlock,
    ReasoningContent,
    ToolCallContent,
    ToolCallStatus,
)


def _get_ai_message_from_anthropic_response(response: AnthropicMessage) -> AIMessage:
    """
    Convert Anthropic API response to internal AIMessage format.

    Parses the Anthropic response and extracts text content, tool calls,
    and reasoning information into our internal message format.

    Args:
        response: Anthropic API response object

    Returns:
        AIMessage with parsed content, tool calls, and reasoning

    Raises:
        ValueError: If an unknown content type is encountered
    """
    ai_message = AIMessage(id=response.id, content="")
    content_blocks: list[ContentBlock] = []

    for content in response.content:
        match content.type:
            case "text":
                content_blocks.append(OutputTextBlock(text=content.text))
            case "thinking":
                # Anthropic's thinking block maps to our ReasoningContent
                # Split thinking text into contents (simplified - in production might want better parsing)
                thinking_lines = content.thinking.split("\n") if content.thinking else []
                content_blocks.append(
                    ReasoningContent(
                        signature=content.signature,
                        contents=thinking_lines,
                        summaries=[],  # Anthropic doesn't separate summaries
                    )
                )
            case "redacted_thinking":
                content_blocks.append(
                    ReasoningContent(
                        redacted_content=content.data,
                        summaries=[],
                        contents=[],
                    )
                )
            case "tool_use":
                # Map Anthropic tool_use to our ToolCallContent
                if not content.name:
                    raise ValueError("Anthropic tool_use missing required 'name' field")
                content_blocks.append(
                    ToolCallContent(
                        tool_id=content.id,
                        name=content.name,
                        input=content.input,
                        raw_input="",  # Anthropic doesn't provide raw JSON string
                        status=ToolCallStatus.COMPLETE,  # Non-streaming tool calls are complete
                    )
                )
            case _:
                raise ValueError(
                    f"Unknown content type from Anthropic: {content.type}, data: {content}"
                )

    if content_blocks:
        ai_message.content = content_blocks

    return ai_message
