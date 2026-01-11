"""Configuration models for LLM service."""

from typing import Any

from pydantic import BaseModel, Field

from thinllm.compat import StrEnum


class Provider(StrEnum):
    """
    Provider of the LLM service.
    """

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    VERTEXAI = "vertexai"
    AWS_CONVERSE_API = "aws_converse_api"
    AZURE_OPENAI = "azure_openai"


class LLMConfig(BaseModel):
    """
    Configuration for LLM service interactions.

    Attributes:
        model_id: The identifier of the model to use (e.g., 'gpt-4', 'gpt-3.5-turbo')
        model_args: Model-specific arguments for generation behavior

    Example:
        >>> config = LLMConfig(model_id="gpt-4")
        >>> config.model_args.temperature
        0.7
        >>> custom_config = LLMConfig(
        ...     model_id="gpt-3.5-turbo",
        ...     model_args=ModelArgs(temperature=0.9, max_tokens=1000)
        ... )
    """

    provider: Provider = Field(description="Provider of the LLM service.")
    model_id: str = Field(description="Model identifier (e.g., 'gpt-4', 'gpt-3.5-turbo')")
    model_args: dict[str, Any] = Field(
        default_factory=dict, description="Model-specific arguments for generation behavior."
    )
