"""Deserializers for converting Gemini API responses to internal types."""

from typing import Any

from google.genai.types import GenerateContentResponse

from thinllm.messages import (
    AIMessage,
    ContentBlock,
    ContentExtra,
    OutputTextBlock,
    ReasoningContent,
    ToolCallContent,
    ToolCallStatus,
)


def _get_ai_message_from_gemini_response(response: GenerateContentResponse) -> AIMessage:
    """
    Convert Gemini API response to internal AIMessage format.

    Parses the Gemini response and extracts text content, tool calls,
    and reasoning information into our internal message format.

    Args:
        response: Gemini API response object

    Returns:
        AIMessage with parsed content, tool calls, and reasoning

    Raises:
        ValueError: If an unknown content type is encountered or no candidates
    """
    # Use response_id as the message ID
    ai_message = AIMessage(id=response.response_id, content="")
    content_blocks: list[ContentBlock] = []

    # Gemini returns candidates; we use the first one
    if not response.candidates or len(response.candidates) == 0:
        raise ValueError("No candidates in Gemini response")

    candidate = response.candidates[0]

    # Check if candidate has content
    if not candidate.content or not candidate.content.parts:
        # Empty response (possibly due to safety filters)
        return ai_message

    # Process each part in the content
    for part in candidate.content.parts:
        # Extract thought_signature once at the part level
        extra = (
            ContentExtra(gemini_thought_signature=part.thought_signature)
            if part.thought_signature
            else None
        )

        # Check what type of part this is
        if part.text is not None and part.text != "":
            # Check if this text is a thought summary
            if part.thought is True:
                # Thinking/reasoning content - text is the summary
                content_blocks.append(
                    ReasoningContent(
                        signature="",
                        contents=[],  # Gemini doesn't provide raw thinking contents
                        summaries=[part.text],  # The text is the thought summary
                        extra=extra,
                    )
                )
            else:
                content_blocks.append(
                    OutputTextBlock(
                        text=part.text,
                        extra=extra,
                    )
                )
        elif part.function_call is not None:
            # Function/tool call
            function_call = part.function_call
            # Gemini function calls must have a name
            if not function_call.name:
                raise ValueError("Gemini function_call missing required 'name' field")

            content_blocks.append(
                ToolCallContent(
                    tool_id=function_call.id,
                    name=function_call.name,
                    input=function_call.args or {},
                    raw_input="",  # Gemini doesn't provide raw JSON string
                    status=ToolCallStatus.COMPLETE,  # Non-streaming tool calls are complete
                    extra=extra,
                )
            )
        # Note: function_response parts are typically in user messages, not model responses
        # so we don't need to handle them here

    if content_blocks:
        ai_message.content = content_blocks

    return ai_message
