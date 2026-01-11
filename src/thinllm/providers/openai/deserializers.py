from openai.types.responses import Response as OAIResponse

from thinllm.messages import (
    AIMessage,
    ContentBlock,
    OutputTextBlock,
    ReasoningContent,
    ToolCallContent,
    ToolCallStatus,
    WebSearchCallContent,
)
from thinllm.utils import parse_partial_json

# Map OpenAI function_call status to our ToolCallStatus
_OAI_STATUS_MAP = {
    "in_progress": ToolCallStatus.PENDING,
    "completed": ToolCallStatus.COMPLETE,
    "incomplete": ToolCallStatus.INCOMPLETE,
}


def _get_ai_message_from_oai_response(response: OAIResponse) -> AIMessage:
    """
    Convert OpenAI API response to internal AIMessage format.

    Parses the OpenAI response and extracts text content, tool calls,
    and reasoning information into our internal message format.

    Args:
        response: OpenAI API response object

    Returns:
        AIMessage with parsed content, tool calls, and reasoning

    Raises:
        ValueError: If an unknown content or output type is encountered
    """
    ai_message = AIMessage(id=response.id, content="")
    content_blocks: list[ContentBlock] = []

    for output in response.output:
        match output.type:
            case "message":
                for content in output.content:
                    match content.type:
                        case "output_text":
                            # Use output.id (from the message envelope) if available
                            if output.id:
                                content_blocks.append(
                                    OutputTextBlock(id=output.id, text=content.text)
                                )
                            else:
                                content_blocks.append(OutputTextBlock(text=content.text))
                        case _:
                            raise ValueError(
                                f"Unknown content type: {content.type} data: {content}"
                            )
            case "function_call":
                parsed_input = parse_partial_json(output.arguments)
                if not isinstance(parsed_input, dict):
                    parsed_input = {}
                # OpenAI function_call must have a name
                if not output.name:
                    raise ValueError("OpenAI function_call missing required 'name' field")
                # Map OpenAI status to our ToolCallStatus (default to PENDING if None)
                oai_status = output.status or "in_progress"
                status = _OAI_STATUS_MAP.get(oai_status, ToolCallStatus.PENDING)
                # Build ToolCallContent, including id if available
                tool_call_kwargs = {
                    "tool_id": output.call_id,
                    "name": output.name,
                    "raw_input": output.arguments,
                    "input": parsed_input,
                    "status": status,
                }
                if output.id:
                    tool_call_kwargs["id"] = output.id
                content_blocks.append(ToolCallContent(**tool_call_kwargs))
            case "reasoning":
                reasoning_kwargs: dict = {
                    "summaries": [summary.text for summary in output.summary]
                    if output.summary
                    else [],
                    "contents": [content.text for content in output.content]
                    if output.content
                    else [],
                }
                if output.id:
                    reasoning_kwargs["id"] = output.id
                content_blocks.append(ReasoningContent(**reasoning_kwargs))
            case "web_search_call":
                # Determine status based on whether action is None
                status = (
                    ToolCallStatus.PENDING if output.action is None else ToolCallStatus.COMPLETE
                )

                # If action exists, extract query and sources
                if output.action:
                    action = output.action
                    match action.type:
                        case "search":
                            query = action.query
                            sources = None
                            # Extract URLs from sources if they exist
                            if action.sources:
                                sources = [source.url for source in action.sources]
                            web_search_kwargs = {
                                "query": query,
                                "sources": sources,
                                "status": status,
                            }
                            if output.id:
                                web_search_kwargs["id"] = output.id
                            content_blocks.append(WebSearchCallContent(**web_search_kwargs))
                        case _:
                            raise NotImplementedError(
                                f"Action type: {action.type}, not implemented yet"
                            )
                else:
                    # If action is None, create a pending web search with empty query
                    web_search_kwargs = {
                        "query": "",
                        "sources": None,
                        "status": status,
                    }
                    if output.id:
                        web_search_kwargs["id"] = output.id
                    content_blocks.append(WebSearchCallContent(**web_search_kwargs))
            case _:
                raise NotImplementedError(f"Output type: {output.type} not implemented yet")

    if content_blocks:
        ai_message.content = content_blocks

    return ai_message
