import contextlib
from collections.abc import Callable, Generator
from typing import Any, Literal, TypeVar, overload

import openai
from pydantic import BaseModel, ValidationError

from thinllm.config import LLMConfig
from thinllm.core import OutputSchemaType
from thinllm.messages import AIMessage, MessageType, OutputTextBlock
from thinllm.tools import Tool
from thinllm.utils import parse_partial_json

from .deserializers import _get_ai_message_from_oai_response
from .serializers import (
    _get_instruction_from_messages,
    _get_oai_messages,
    _get_oai_tool,
)
from .streaming import OAIStreamMessageBuilder


def _create_client(llm_config: LLMConfig) -> openai.OpenAI:
    """Create OpenAI or Azure OpenAI client based on provider."""
    from thinllm.config import Provider

    if llm_config.provider == Provider.AZURE_OPENAI:
        # Azure OpenAI client configuration
        kwargs: dict[str, Any] = {}

        if not llm_config.credentials:
            raise ValueError("credentials with azure_endpoint required for Azure OpenAI")

        creds = llm_config.credentials

        # Required: Azure endpoint
        if not creds.azure_endpoint:
            raise ValueError("azure_endpoint is required for Azure OpenAI")

        kwargs["base_url"] = f"{creds.azure_endpoint}/openai/v1/"

        # Optional: Add custom headers for Azure-specific features
        # Only set api-version if explicitly provided by user
        if creds.azure_api_version:
            kwargs["default_headers"] = {"api-version": creds.azure_api_version}

        # Authentication: API Key or Microsoft Entra ID
        if creds.api_key:
            kwargs["api_key"] = creds.api_key
        else:
            # Microsoft Entra ID token provider
            try:
                from azure.identity import DefaultAzureCredential, get_bearer_token_provider

                token_provider = get_bearer_token_provider(
                    DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
                )
                kwargs["api_key"] = token_provider
            except ImportError as e:
                raise ImportError(
                    "azure-identity is required for Microsoft Entra ID authentication. "
                    "Install it with: pip install azure-identity"
                ) from e

        return openai.OpenAI(**kwargs)
    else:
        # Regular OpenAI client
        kwargs: dict[str, Any] = {}
        if llm_config.credentials and llm_config.credentials.api_key:
            kwargs["api_key"] = llm_config.credentials.api_key
        return openai.OpenAI(**kwargs)


def _build_common_params(
    llm_config: LLMConfig,
    messages: list[MessageType],
    tools: list[Tool | Callable | dict] | None = None,
    output_schema: type[OutputSchemaType] | None = None,
) -> dict[str, Any]:
    """Build common parameters for OpenAI API calls."""
    params: dict[str, Any] = {
        "model": llm_config.model_id,
        "input": _get_oai_messages(messages),
        "instructions": _get_instruction_from_messages(messages),
        **llm_config.get_effective_params(),
    }

    if tools:
        params["tools"] = [_get_oai_tool(tool) for tool in tools]

    if output_schema:
        params["text_format"] = output_schema

    return params


def _llm_basic(
    llm_config: LLMConfig,
    messages: list[MessageType],
    tools: list[Tool | Callable | dict] | None = None,
) -> AIMessage:
    """Basic non-streaming LLM call."""
    params = _build_common_params(llm_config, messages, tools=tools)
    client = _create_client(llm_config)
    response = client.responses.create(**params)
    return _get_ai_message_from_oai_response(response)


def _llm_structured(
    llm_config: LLMConfig,
    messages: list[MessageType],
    output_schema: type[OutputSchemaType],
) -> OutputSchemaType:
    """Non-streaming structured LLM call."""
    params = _build_common_params(llm_config, messages, output_schema=output_schema)
    client = _create_client(llm_config)
    response = client.responses.parse(**params)
    if response.output_parsed is None:
        raise ValueError("No parsed response received")
    return response.output_parsed


def _llm_stream(
    llm_config: LLMConfig,
    messages: list[MessageType],
    tools: list[Tool | Callable | dict] | None = None,
) -> Generator[AIMessage, None, AIMessage]:
    """Streaming LLM call."""
    params = _build_common_params(llm_config, messages, tools=tools)
    builder = OAIStreamMessageBuilder()
    client = _create_client(llm_config)

    with client.responses.stream(**params) as stream:
        for event in stream:
            yield _get_ai_message_from_oai_response(builder.add_event(event))

    if builder.response is None:
        raise ValueError("No response received")

    return _get_ai_message_from_oai_response(builder.response)


def _llm_structured_stream(
    llm_config: LLMConfig,
    messages: list[MessageType],
    output_schema: type[OutputSchemaType],
) -> Generator[OutputSchemaType, None, OutputSchemaType]:
    """Streaming structured LLM call."""
    params = _build_common_params(llm_config, messages, output_schema=output_schema)
    builder = OAIStreamMessageBuilder()
    client = _create_client(llm_config)

    with client.responses.stream(**params) as stream:
        for event in stream:
            ai_message = _get_ai_message_from_oai_response(builder.add_event(event))
            partial_content = ""
            if isinstance(ai_message.content, str):
                partial_content = ai_message.content
            elif isinstance(ai_message.content, list) and len(ai_message.content) > 0:
                last_text_block = ai_message.content[-1]
                if isinstance(last_text_block, OutputTextBlock):
                    partial_content = last_text_block.text
            if not partial_content:
                continue
            parsed_json = parse_partial_json(partial_content)
            if parsed_json:
                with contextlib.suppress(ValidationError):
                    yield output_schema(**parsed_json)

    content = _get_ai_message_from_oai_response(builder.response).content
    if isinstance(content, str):
        return output_schema(**parse_partial_json(content))
    elif isinstance(content, list) and len(content) > 0:
        last_text_block = content[-1]
        if isinstance(last_text_block, OutputTextBlock):
            return output_schema(**parse_partial_json(last_text_block.text))
    raise ValueError("No content received")


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
    Unified interface for LLM interactions.

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
        ValueError: If no response is received in streaming mode

    Examples:
        Basic text response:
        >>> config = LLMConfig(model_id="gpt-4")
        >>> messages = [UserMessage(content="What is 2+2?")]
        >>> response = llm(config, messages)
        >>> print(response.content)
        '2+2 equals 4'

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
            output_schema,  # pyrefly: ignore[bad-argument-type]  false positive
        )

    # Non-streaming with output schema
    if output_schema:
        return _llm_structured(llm_config, messages, output_schema)

    # Streaming without output schema
    if stream:
        return _llm_stream(llm_config, messages, tools)

    # Non-streaming without output schema
    return _llm_basic(llm_config, messages, tools)
