"""Core LLM interface for Anthropic provider."""

import contextlib
from collections.abc import Callable, Generator
from typing import Any, Literal, overload

import anthropic
from pydantic import ValidationError

from thinllm.config import LLMConfig
from thinllm.core import OutputSchemaType
from thinllm.messages import AIMessage, MessageType, OutputTextBlock, ToolCallContent
from thinllm.tools import Tool

from .deserializers import _get_ai_message_from_anthropic_response
from .serializers import (
    _get_anthropic_messages,
    _get_anthropic_tool,
    _get_system_blocks_from_messages,
)
from .streaming import AnthropicStreamMessageBuilder


def _create_client(llm_config: LLMConfig) -> anthropic.Anthropic | Any:
    """Create either Anthropic or AnthropicBedrock client based on provider."""
    from thinllm.config import Provider

    if llm_config.provider == Provider.BEDROCK_ANTHROPIC:
        from anthropic import AnthropicBedrock

        # Only pass credentials if explicitly provided by user
        # Otherwise, let AnthropicBedrock use AWS default credential providers
        kwargs: dict[str, Any] = {}

        if llm_config.credentials:
            creds = llm_config.credentials

            # Only add credential fields that are explicitly set
            if creds.aws_access_key:
                kwargs["aws_access_key"] = creds.aws_access_key
            if creds.aws_secret_key:
                kwargs["aws_secret_key"] = creds.aws_secret_key
            if creds.aws_session_token:
                kwargs["aws_session_token"] = creds.aws_session_token
            if creds.aws_region:
                kwargs["aws_region"] = creds.aws_region

        return AnthropicBedrock(**kwargs)

    # Regular Anthropic client
    return anthropic.Anthropic()


def _get_messages_api(client: anthropic.Anthropic | Any, params: dict[str, Any]) -> Any:
    """
    Get the appropriate messages API (regular or beta) based on params.

    If the params contain a 'betas' key, use client.beta.messages, otherwise use client.messages.
    The betas parameter is only supported by beta APIs.

    Args:
        client: Anthropic or AnthropicBedrock client
        params: API parameters that may contain 'betas'

    Returns:
        The messages API object (client.messages or client.beta.messages)

    Raises:
        ValueError: If beta features are requested but client doesn't support beta API
    """
    if "betas" in params:
        if not hasattr(client, "beta"):
            raise ValueError(
                f"Beta features requested but client doesn't support beta API. "
                f"Beta headers: {params['betas']}"
            )
        return client.beta.messages
    return client.messages


def _get_last_output_text_block(ai_message: AIMessage) -> OutputTextBlock:
    """
    Extract the last OutputTextBlock from AIMessage.

    This is a generic utility that can be used across the provider wherever we need
    to extract the last text output block (e.g., for structured outputs, regular text, etc.)

    Args:
        ai_message: AIMessage containing OutputTextBlock(s)

    Returns:
        The last OutputTextBlock found in the message

    Raises:
        ValueError: If no OutputTextBlock found or content is not in expected format
    """
    if isinstance(ai_message.content, str):
        # If content is a string, wrap it in an OutputTextBlock
        return OutputTextBlock(text=ai_message.content)

    if isinstance(ai_message.content, list):
        # Find all OutputTextBlocks and return the last one
        text_blocks = [block for block in ai_message.content if isinstance(block, OutputTextBlock)]
        if text_blocks:
            return text_blocks[-1]

    raise ValueError("No OutputTextBlock found in AIMessage")


def _build_common_params(
    llm_config: LLMConfig,
    messages: list[MessageType],
    tools: list[Tool | Callable | dict] | None = None,
    output_schema: type[OutputSchemaType] | None = None,
) -> dict[str, Any]:
    """Build common parameters for Anthropic API calls."""
    # Get effective params
    effective_params = llm_config.get_effective_params()

    params: dict[str, Any] = {
        "model": llm_config.model_id,
        "messages": _get_anthropic_messages(messages),
    }

    # Add effective params
    params.update(effective_params)

    # Add system message if present
    system_blocks = _get_system_blocks_from_messages(messages)
    if system_blocks:
        params["system"] = system_blocks

    if tools:
        params["tools"] = [_get_anthropic_tool(tool) for tool in tools]

    # For structured output, use beta structured outputs API
    if output_schema:
        from thinllm.config import Provider

        # Check if using Bedrock (doesn't support structured outputs beta yet)
        is_bedrock = llm_config.provider == Provider.BEDROCK_ANTHROPIC

        if is_bedrock:
            # Fallback to tool-based approach for Bedrock
            params["tools"] = [
                {
                    "name": "respond_with_structure",
                    "description": f"Respond with structured data matching the {output_schema.__name__} schema",
                    "input_schema": output_schema.model_json_schema(),
                }
            ]
            params["tool_choice"] = {"type": "tool", "name": "respond_with_structure"}
        else:
            # Use beta structured outputs API for regular Anthropic
            from anthropic import transform_schema

            # Append to existing betas if present, otherwise create new list
            if "betas" in params:
                params["betas"].append("structured-outputs-2025-11-13")
            else:
                params["betas"] = ["structured-outputs-2025-11-13"]

            # Use Anthropic's transform_schema utility to convert Pydantic model
            params["output_format"] = {
                "type": "json_schema",
                "schema": transform_schema(output_schema),
            }

    return params


def _llm_basic(
    llm_config: LLMConfig,
    messages: list[MessageType],
    tools: list[Tool | Callable | dict] | None = None,
) -> AIMessage:
    """Basic non-streaming LLM call."""
    params = _build_common_params(llm_config, messages, tools=tools)
    client = _create_client(llm_config)

    # Get the appropriate API (messages or beta.messages)
    messages_api = _get_messages_api(client, params)
    response = messages_api.create(**params)

    return _get_ai_message_from_anthropic_response(response)


def _llm_structured(
    llm_config: LLMConfig,
    messages: list[MessageType],
    output_schema: type[OutputSchemaType],
) -> OutputSchemaType:
    """Non-streaming structured LLM call using beta API or tool-based fallback."""
    from thinllm.config import Provider
    from thinllm.utils import parse_partial_json

    params = _build_common_params(llm_config, messages, output_schema=output_schema)
    client = _create_client(llm_config)

    # Use beta.messages API (will be selected automatically based on 'betas' param)
    messages_api = _get_messages_api(client, params)
    response = messages_api.create(**params)

    # Deserialize to AIMessage
    ai_message = _get_ai_message_from_anthropic_response(response)

    # Check if using Bedrock (tool-based approach)
    is_bedrock = llm_config.provider == Provider.BEDROCK_ANTHROPIC

    if is_bedrock:
        # Tool-based approach for Bedrock - check for ToolCallContent
        if isinstance(ai_message.content, list):
            for block in ai_message.content:
                if isinstance(block, ToolCallContent) and block.name == "respond_with_structure":
                    return output_schema(**block.input)
        raise ValueError("No structured response received from Anthropic Bedrock")

    # Beta API approach for regular Anthropic
    # Extract the last OutputTextBlock (contains JSON)
    text_block = _get_last_output_text_block(ai_message)

    # Parse JSON to dict using parse_partial_json
    parsed_dict = parse_partial_json(text_block.text)

    # Validate and load into Pydantic model
    return output_schema.model_validate(parsed_dict)


def _llm_stream(
    llm_config: LLMConfig,
    messages: list[MessageType],
    tools: list[Tool | Callable | dict] | None = None,
) -> Generator[AIMessage, None, AIMessage]:
    """Streaming LLM call."""
    params = _build_common_params(llm_config, messages, tools=tools)
    builder = AnthropicStreamMessageBuilder()
    client = _create_client(llm_config)

    # Get the appropriate API (messages or beta.messages)
    messages_api = _get_messages_api(client, params)

    with messages_api.stream(**params) as stream:
        for event in stream:
            builder.add_event(event)
            yield _get_ai_message_from_anthropic_response(builder.response)

    if builder.response is None:
        raise ValueError("No response received")

    return _get_ai_message_from_anthropic_response(builder.response)


def _yield_structured_from_tool(
    ai_message: AIMessage,
    output_schema: type[OutputSchemaType],
) -> Generator[OutputSchemaType, None, None]:
    """Yield structured output from tool calls (Bedrock approach)."""
    if isinstance(ai_message.content, list):
        # Find all ToolCallContent blocks and only yield the last one
        tool_blocks = [block for block in ai_message.content if isinstance(block, ToolCallContent)]
        if tool_blocks:
            last_tool_block = tool_blocks[-1]
            with contextlib.suppress(ValidationError):
                yield output_schema(**last_tool_block.input)


def _yield_structured_from_json(
    ai_message: AIMessage,
    output_schema: type[OutputSchemaType],
) -> Generator[OutputSchemaType, None, None]:
    """Yield structured output from JSON text (Beta API approach)."""
    from thinllm.utils import parse_partial_json

    try:
        text_block = _get_last_output_text_block(ai_message)
        json_text = text_block.text
    except ValueError:
        # No text block yet
        return

    if json_text:
        parsed_dict = parse_partial_json(json_text)
        with contextlib.suppress(ValidationError):
            yield output_schema.model_validate(parsed_dict)


def _llm_structured_stream(  # type: ignore[misc]
    llm_config: LLMConfig,
    messages: list[MessageType],
    output_schema: type[OutputSchemaType],
) -> Generator[OutputSchemaType, None, OutputSchemaType]:
    """Streaming structured LLM call using beta API or tool-based fallback."""
    from thinllm.config import Provider
    from thinllm.utils import parse_partial_json

    params = _build_common_params(llm_config, messages, output_schema=output_schema)
    builder = AnthropicStreamMessageBuilder()
    client = _create_client(llm_config)

    messages_api = _get_messages_api(client, params)
    is_bedrock = llm_config.provider == Provider.BEDROCK_ANTHROPIC

    with messages_api.stream(**params) as stream:
        for event in stream:
            builder.add_event(event)
            ai_message = _get_ai_message_from_anthropic_response(builder.response)

            if is_bedrock:
                yield from _yield_structured_from_tool(ai_message, output_schema)
            else:
                yield from _yield_structured_from_json(ai_message, output_schema)

    # Final output
    if builder.response is None:
        raise ValueError("No response received")

    # Deserialize final response to AIMessage
    final_ai_message = _get_ai_message_from_anthropic_response(builder.response)

    if is_bedrock:
        # Tool-based approach for Bedrock - check for ToolCallContent
        if isinstance(final_ai_message.content, list):
            for block in final_ai_message.content:
                if isinstance(block, ToolCallContent) and block.name == "respond_with_structure":
                    return output_schema(**block.input)
        raise ValueError("No structured response received from Anthropic Bedrock")

    # Beta API approach for regular Anthropic
    final_text_block = _get_last_output_text_block(final_ai_message)

    final_dict = parse_partial_json(final_text_block.text)
    return output_schema.model_validate(final_dict)


# Overload 1: Non-streaming, no output schema (with or without tools)
@overload
def llm(
    llm_config: LLMConfig,
    messages: list[MessageType],
    *,
    output_schema: None = None,
    tools: list[Tool | Callable | dict] | None = None,
    stream: Literal[False] = False,
) -> AIMessage: ...


# Overload 2: Non-streaming, with output schema
@overload
def llm(
    llm_config: LLMConfig,
    messages: list[MessageType],
    *,
    output_schema: type[OutputSchemaType],
    tools: None = None,
    stream: Literal[False] = False,
) -> OutputSchemaType: ...


# Overload 3: Streaming, no output schema (with or without tools)
@overload
def llm(
    llm_config: LLMConfig,
    messages: list[MessageType],
    *,
    output_schema: None = None,
    tools: list[Tool | Callable | dict] | None = None,
    stream: Literal[True],
) -> Generator[AIMessage, None, AIMessage]: ...


# Overload 4: Streaming, with output schema
@overload
def llm(
    llm_config: LLMConfig,
    messages: list[MessageType],
    *,
    output_schema: type[OutputSchemaType],
    tools: None = None,
    stream: Literal[True],
) -> Generator[OutputSchemaType, None, OutputSchemaType]: ...


def llm(
    llm_config: LLMConfig,
    messages: list[MessageType],
    *,
    output_schema: type[OutputSchemaType] | None = None,
    tools: list[Tool | Callable | dict] | None = None,
    stream: bool = False,
) -> (
    AIMessage
    | OutputSchemaType
    | Generator[AIMessage, None, AIMessage]
    | Generator[OutputSchemaType, None, OutputSchemaType]
):
    """
    Unified interface for LLM interactions with Anthropic.

    This function handles all LLM interaction patterns through a single interface:
    - Regular text responses
    - Structured responses (via Pydantic models)
    - Tool/function calling
    - Streaming responses

    Args:
        llm_config: Configuration for the LLM (model, temperature, etc.)
        messages: List of conversation messages
        output_schema: Optional Pydantic model for structured output
        tools: Optional list of tools/functions the LLM can call
        stream: Whether to stream the response incrementally

    Returns:
        Depending on parameters:
        - AIMessage: Regular non-streaming response
        - OutputSchemaType: Structured non-streaming response
        - Generator[AIMessage, None, AIMessage]: Streaming response
        - Generator[OutputSchemaType, None, OutputSchemaType]: Streaming structured response

    Raises:
        NotImplementedError: If both output_schema and tools are provided (incompatible)
        ValueError: If no response is received

    Examples:
        Basic text response:
        >>> config = LLMConfig(provider=Provider.ANTHROPIC, model_id="claude-sonnet-4-5")
        >>> messages = [UserMessage(content="What is 2+2?")]
        >>> response = llm(config, messages)
        >>> print(response.content)

        Structured response:
        >>> class Recipe(BaseModel):
        ...     name: str
        ...     ingredients: list[str]
        >>> recipe = llm(config, messages, output_schema=Recipe)
        >>> print(recipe.name)

        Streaming response:
        >>> for partial_msg in llm(config, messages, stream=True):
        ...     print(partial_msg.content, end="", flush=True)

        With tools:
        >>> def get_weather(location: str) -> str:
        ...     return f"Weather in {location}"
        >>> response = llm(config, messages, tools=[get_weather])
    """
    if output_schema and tools:
        raise NotImplementedError("Cannot use both output_schema and tools simultaneously")

    # Streaming with output schema
    if stream and output_schema:
        return _llm_structured_stream(
            llm_config,
            messages,
            output_schema,  # pyrefly: ignore[bad-argument-type]
        )

    # Non-streaming with output schema
    if output_schema:
        return _llm_structured(llm_config, messages, output_schema)

    # Streaming without output schema
    if stream:
        return _llm_stream(llm_config, messages, tools)

    # Non-streaming without output schema
    return _llm_basic(llm_config, messages, tools)
