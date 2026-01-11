"""Core LLM interface for Anthropic provider."""

import contextlib
from collections.abc import Callable, Generator
from typing import Any, Literal, overload

import anthropic
from pydantic import ValidationError

from thinllm.config import LLMConfig
from thinllm.core import OutputSchemaType
from thinllm.messages import AIMessage, MessageType
from thinllm.tools import Tool

from .deserializers import _get_ai_message_from_anthropic_response
from .serializers import (
    _get_anthropic_messages,
    _get_anthropic_tool,
    _get_system_from_messages,
)
from .streaming import AnthropicStreamMessageBuilder


def _build_common_params(
    llm_config: LLMConfig,
    messages: list[MessageType],
    tools: list[Tool | Callable | dict] | None = None,
    output_schema: type[OutputSchemaType] | None = None,
) -> dict[str, Any]:
    """Build common parameters for Anthropic API calls."""
    params: dict[str, Any] = {
        "model": llm_config.model_id,
        "messages": _get_anthropic_messages(messages),
        "max_tokens": llm_config.model_args.get("max_tokens", 4096),
        **{k: v for k, v in llm_config.model_args.items() if k != "max_tokens"},
    }

    # Add system message if present
    system = _get_system_from_messages(messages)
    if system:
        params["system"] = system

    if tools:
        params["tools"] = [_get_anthropic_tool(tool) for tool in tools]

    # For structured output, we need to use tool calling with a response_format
    # Anthropic doesn't have direct structured output like OpenAI, so we'll use a tool
    if output_schema:
        # Create a tool that represents the structured output
        params["tools"] = [
            {
                "name": "respond_with_structure",
                "description": f"Respond with structured data matching the {output_schema.__name__} schema",
                "input_schema": output_schema.model_json_schema(),
            }
        ]
        params["tool_choice"] = {"type": "tool", "name": "respond_with_structure"}

    return params


def _llm_basic(
    llm_config: LLMConfig,
    messages: list[MessageType],
    tools: list[Tool | Callable | dict] | None = None,
) -> AIMessage:
    """Basic non-streaming LLM call."""
    params = _build_common_params(llm_config, messages, tools=tools)
    client = anthropic.Anthropic()
    response = client.messages.create(**params)
    return _get_ai_message_from_anthropic_response(response)


def _llm_structured(
    llm_config: LLMConfig,
    messages: list[MessageType],
    output_schema: type[OutputSchemaType],
) -> OutputSchemaType:
    """Non-streaming structured LLM call."""
    params = _build_common_params(llm_config, messages, output_schema=output_schema)
    client = anthropic.Anthropic()
    response = client.messages.create(**params)
    
    # Extract the tool call result
    for content in response.content:
        if content.type == "tool_use" and content.name == "respond_with_structure":
            return output_schema(**content.input)
    
    raise ValueError("No structured response received from Anthropic")


def _llm_stream(
    llm_config: LLMConfig,
    messages: list[MessageType],
    tools: list[Tool | Callable | dict] | None = None,
) -> Generator[AIMessage, None, AIMessage]:
    """Streaming LLM call."""
    params = _build_common_params(llm_config, messages, tools=tools)
    builder = AnthropicStreamMessageBuilder()
    client = anthropic.Anthropic()

    with client.messages.stream(**params) as stream:
        for event in stream:
            builder.add_event(event)
            yield _get_ai_message_from_anthropic_response(builder.response)

    if builder.response is None:
        raise ValueError("No response received")

    return _get_ai_message_from_anthropic_response(builder.response)


def _llm_structured_stream(  # type: ignore[misc]
    llm_config: LLMConfig,
    messages: list[MessageType],
    output_schema: type[OutputSchemaType],
) -> Generator[OutputSchemaType, None, OutputSchemaType]:
    """Streaming structured LLM call."""
    params = _build_common_params(llm_config, messages, output_schema=output_schema)
    builder = AnthropicStreamMessageBuilder()
    client = anthropic.Anthropic()

    with client.messages.stream(**params) as stream:
        for event in stream:
            builder.add_event(event)
            ai_message = _get_ai_message_from_anthropic_response(builder.response)
            
            # Try to extract partial structured output
            if isinstance(ai_message.content, list):
                for block in ai_message.content:
                    if hasattr(block, "input") and isinstance(block.input, dict):
                        with contextlib.suppress(ValidationError):
                            yield output_schema(**block.input)

    # Extract final structured output
    for content in builder.response.content:
        if content.type == "tool_use" and content.name == "respond_with_structure":
            return output_schema(**content.input)
    
    raise ValueError("No structured response received from Anthropic")


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
        return _llm_structured_stream(llm_config, messages, output_schema)  # pyrefly: ignore[bad-argument-type]

    # Non-streaming with output schema
    if output_schema:
        return _llm_structured(llm_config, messages, output_schema)

    # Streaming without output schema
    if stream:
        return _llm_stream(llm_config, messages, tools)

    # Non-streaming without output schema
    return _llm_basic(llm_config, messages, tools)

