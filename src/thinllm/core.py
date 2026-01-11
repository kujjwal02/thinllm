from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeVar, overload

from pydantic import BaseModel

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from .config import LLMConfig
    from .messages import AIMessage, MessageType
    from .tools import Tool

OutputSchemaType = TypeVar("OutputSchemaType", bound=BaseModel)


def _get_openai_llm():
    """Lazy import for OpenAI provider."""
    from .providers.openai.core import llm as openai_llm

    return openai_llm


def _get_anthropic_llm():
    """Lazy import for Anthropic provider."""
    from .providers.anthropic.core import llm as anthropic_llm

    return anthropic_llm


def _get_gemini_llm():
    """Lazy import for Gemini provider."""
    from .providers.gemini.core import llm as gemini_llm

    return gemini_llm


# Add more provider lazy loaders here as they're implemented
# def _get_google_llm():
#     """Lazy import for Google provider."""
#     from .providers.google.core import llm as google_llm
#     return google_llm


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

    This is the main entry point for all LLM operations. It automatically
    routes to the appropriate provider implementation based on the config.

    Args:
        llm_config: Configuration for the LLM (model, temperature, provider, etc.)
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
        ValueError: If the provider is not supported

    Examples:
        Basic text response:
        >>> config = LLMConfig(model_id="gpt-4", provider="openai")
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
    """
    # Get provider name from config
    provider = llm_config.provider.lower() if hasattr(llm_config, "provider") else "openai"

    # Lazy load and call the appropriate provider's llm function
    match provider:
        case "openai":
            provider_llm = _get_openai_llm()
            return provider_llm(  # type: ignore[return-value]
                llm_config, messages, output_schema=output_schema, tools=tools, stream=stream
            )

        case "anthropic":
            provider_llm = _get_anthropic_llm()
            return provider_llm(  # type: ignore[return-value]
                llm_config, messages, output_schema=output_schema, tools=tools, stream=stream
            )

        case "gemini":
            provider_llm = _get_gemini_llm()
            return provider_llm(  # type: ignore[return-value]
                llm_config, messages, output_schema=output_schema, tools=tools, stream=stream
            )

        # Add more providers here as they're implemented
        # case "google":
        #     provider_llm = _get_google_llm()
        #     return provider_llm(llm_config, messages, output_schema=output_schema, tools=tools, stream=stream)  # type: ignore[return-value]

        case _:
            raise ValueError(f"Unsupported provider: {provider}. Supported providers: openai, anthropic, gemini")
