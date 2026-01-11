"""Core LLM interface for Gemini provider."""

import contextlib
from collections.abc import Callable, Generator
from typing import Any, Literal, overload

from google import genai
from google.genai import types
from pydantic import ValidationError

from thinllm.config import LLMConfig
from thinllm.core import OutputSchemaType
from thinllm.messages import AIMessage, MessageType, OutputTextBlock
from thinllm.tools import Tool
from thinllm.utils import parse_partial_json

from .deserializers import _get_ai_message_from_gemini_response
from .serializers import (
    _get_gemini_contents,
    _get_gemini_tool,
    _get_system_instruction,
)
from .streaming import GeminiStreamMessageBuilder


def _build_common_params(  # noqa: C901
    llm_config: LLMConfig,
    messages: list[MessageType],
    tools: list[Tool | Callable | dict] | None = None,
    output_schema: type[OutputSchemaType] | None = None,
) -> dict[str, Any]:
    """Build common parameters for Gemini API calls."""
    params: dict[str, Any] = {
        "model": llm_config.model_id,
        "contents": _get_gemini_contents(messages),
    }

    # Build config dict for GenerateContentConfig
    config_dict: dict[str, Any] = {}

    # Add system instruction if present
    system_instruction = _get_system_instruction(messages)
    if system_instruction:
        config_dict["system_instruction"] = system_instruction

    # Get effective params
    effective_params = llm_config.get_effective_params()

    # Extract special parameters that need special handling
    thinking_budget = effective_params.get("thinking_budget")
    include_thoughts = effective_params.get("include_thoughts", False)

    # Add thinking config if needed
    if thinking_budget is not None:
        config_dict["thinking_config"] = types.ThinkingConfig(
            thinking_budget=thinking_budget,
            include_thoughts=include_thoughts,
        )

    # Add other effective params to config (exclude our custom ones)
    config_dict.update(
        {
            key: value
            for key, value in effective_params.items()
            if key not in ("thinking_budget", "include_thoughts")
        }
    )

    # Handle tools
    if tools:
        # Gemini expects tools as a list, where each item can be:
        # - {"function_declarations": [...]} for custom functions
        # - {"google_search": {}} for built-in tools
        gemini_tools = []
        function_declarations = []

        for tool in tools:
            tool_dict = _get_gemini_tool(tool)
            # Check if this is a built-in tool (like google_search)
            if "function_declarations" in tool_dict:
                # Custom function - accumulate
                function_declarations.extend(tool_dict["function_declarations"])
            else:
                # Built-in tool - add directly
                gemini_tools.append(tool_dict)

        # Add accumulated function declarations as a single tool entry
        if function_declarations:
            gemini_tools.append({"function_declarations": function_declarations})

        config_dict["tools"] = gemini_tools

    # Handle structured output
    if output_schema:
        config_dict["response_mime_type"] = "application/json"
        config_dict["response_schema"] = output_schema.model_json_schema()

    # Create GenerateContentConfig if we have any config
    if config_dict:
        params["config"] = types.GenerateContentConfig(**config_dict)

    return params


def _llm_basic(
    llm_config: LLMConfig,
    messages: list[MessageType],
    tools: list[Tool | Callable | dict] | None = None,
) -> AIMessage:
    """Basic non-streaming LLM call."""
    params = _build_common_params(llm_config, messages, tools=tools)
    client = genai.Client()
    response = client.models.generate_content(**params)
    return _get_ai_message_from_gemini_response(response)


def _llm_structured(
    llm_config: LLMConfig,
    messages: list[MessageType],
    output_schema: type[OutputSchemaType],
) -> OutputSchemaType:
    """Non-streaming structured LLM call."""
    params = _build_common_params(llm_config, messages, output_schema=output_schema)
    client = genai.Client()
    response = client.models.generate_content(**params)

    # Convert to AIMessage and extract text from last OutputTextBlock
    ai_message = _get_ai_message_from_gemini_response(response)

    # Extract text from the last OutputTextBlock
    text_content = ""
    if isinstance(ai_message.content, list):
        for block in ai_message.content:
            if isinstance(block, OutputTextBlock):
                text_content = block.text

    if not text_content:
        raise ValueError("No text content in Gemini structured response")

    # Parse the JSON and create the output schema
    parsed_json = parse_partial_json(text_content)
    if not isinstance(parsed_json, dict):
        raise TypeError(f"Expected JSON object, got {type(parsed_json)}")

    return output_schema(**parsed_json)


def _llm_stream(
    llm_config: LLMConfig,
    messages: list[MessageType],
    tools: list[Tool | Callable | dict] | None = None,
) -> Generator[AIMessage, None, AIMessage]:
    """Streaming LLM call."""
    params = _build_common_params(llm_config, messages, tools=tools)
    builder = GeminiStreamMessageBuilder()
    client = genai.Client()

    for chunk in client.models.generate_content_stream(**params):
        builder.add_chunk(chunk)
        yield _get_ai_message_from_gemini_response(builder.response)

    if builder.response is None:
        raise ValueError("No response received")

    return _get_ai_message_from_gemini_response(builder.response)


def _llm_structured_stream(  # type: ignore[misc]  # noqa: C901
    llm_config: LLMConfig,
    messages: list[MessageType],
    output_schema: type[OutputSchemaType],
) -> Generator[OutputSchemaType, None, OutputSchemaType]:
    """Streaming structured LLM call."""
    params = _build_common_params(llm_config, messages, output_schema=output_schema)
    builder = GeminiStreamMessageBuilder()
    client = genai.Client()

    for chunk in client.models.generate_content_stream(**params):
        builder.add_chunk(chunk)
        ai_message = _get_ai_message_from_gemini_response(builder.response)

        # Try to extract partial structured output from last OutputTextBlock
        partial_content = ""
        if isinstance(ai_message.content, list):
            for block in ai_message.content:
                if isinstance(block, OutputTextBlock):
                    partial_content = block.text
        elif isinstance(ai_message.content, str):
            partial_content = ai_message.content

        if partial_content:
            parsed_json = parse_partial_json(partial_content)
            if isinstance(parsed_json, dict):
                with contextlib.suppress(ValidationError):
                    yield output_schema(**parsed_json)

    # Extract final structured output from last OutputTextBlock
    final_ai_message = _get_ai_message_from_gemini_response(builder.response)

    text_content = ""
    if isinstance(final_ai_message.content, list):
        for block in final_ai_message.content:
            if isinstance(block, OutputTextBlock):
                text_content = block.text

    if not text_content:
        raise ValueError("No text content in Gemini structured response")

    # Parse the JSON and create the output schema
    parsed_json = parse_partial_json(text_content)
    if not isinstance(parsed_json, dict):
        raise TypeError(f"Expected JSON object, got {type(parsed_json)}")

    return output_schema(**parsed_json)


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
    Unified interface for LLM interactions with Gemini.

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
        >>> config = LLMConfig(provider=Provider.GEMINI, model_id="gemini-2.5-flash")
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
