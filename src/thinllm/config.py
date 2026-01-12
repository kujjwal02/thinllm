"""Configuration models for LLM service."""

import warnings
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from thinllm.compat import StrEnum


class Provider(StrEnum):
    """
    Provider of the LLM service.
    """

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    BEDROCK_ANTHROPIC = "bedrock_anthropic"
    GEMINI = "gemini"
    VERTEXAI = "vertexai"
    AWS_CONVERSE_API = "aws_converse_api"
    AZURE_OPENAI = "azure_openai"


class ThinkingLevel(StrEnum):
    """Level of thinking/reasoning for models that support it."""

    NONE = "none"
    MINIMAL = "minimal"
    LOW = "low"
    HIGH = "high"
    XHIGH = "xhigh"


class ThinkingConfig(BaseModel):
    """Configuration for model thinking/reasoning capabilities."""

    enabled: bool = False
    thinking_level: ThinkingLevel = ThinkingLevel.NONE
    thinking_budget: int | None = None

    @field_validator("thinking_budget")
    @classmethod
    def validate_budget(cls, v: int | None) -> int | None:
        """Validate thinking budget is non-negative."""
        if v is not None and v < 0:
            raise ValueError("thinking_budget must be non-negative")
        return v


class ToolChoiceType(StrEnum):
    """Type of tool choice strategy."""

    NONE = "none"  # Don't use tools
    AUTO = "auto"  # Model decides
    FUNCTION = "function"  # Force specific function
    ALLOWLIST = "allowlist"  # Limit to subset
    ANY = "any"  # Must use at least one tool


class ToolChoiceConfig(BaseModel):
    """Configuration for tool/function calling behavior."""

    type: ToolChoiceType = ToolChoiceType.AUTO
    disable_parallel_tool_calls: bool = False
    name: str | None = None  # For type=FUNCTION
    allowed_tools: list[str] | None = None  # For type=ALLOWLIST

    @model_validator(mode="after")
    def validate_tool_choice(self) -> "ToolChoiceConfig":
        """Validate tool choice configuration."""
        if self.type == ToolChoiceType.FUNCTION and not self.name:
            raise ValueError("name is required when type=FUNCTION")
        if self.type == ToolChoiceType.ALLOWLIST and not self.allowed_tools:
            raise ValueError("allowed_tools is required when type=ALLOWLIST")
        return self


class ModelParams(BaseModel):
    """Standardized model parameters across all providers."""

    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_k: int | None = Field(default=None, ge=1)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    max_output_tokens: int | None = Field(default=None, ge=1)
    stop_sequences: list[str] | None = None

    thinking: ThinkingConfig | None = None
    tool_choice: ToolChoiceConfig | None = None


class Credentials(BaseModel):
    """Optional explicit credentials. If not provided, providers use their default auth (env vars, etc.)."""

    api_key: str | None = None
    api_base: str | None = None
    organization: str | None = None
    project_id: str | None = None
    region: str | None = None

    # AWS-specific credentials for Bedrock
    aws_access_key: str | None = None
    aws_secret_key: str | None = None
    aws_session_token: str | None = None
    aws_region: str | None = None

    # Azure OpenAI specific credentials
    azure_endpoint: str | None = None  # e.g., "https://my-resource.openai.azure.com"
    azure_deployment: str | None = None  # Optional override for model_id
    azure_api_version: str | None = None  # Optional API version (e.g., "2023-07-01-preview")

    model_config = ConfigDict(extra="allow")


class ParamTranslator:
    """Translate ModelParams to provider-specific format."""

    @staticmethod
    def to_openai(params: ModelParams) -> dict[str, Any]:
        """Translate ModelParams to OpenAI format."""
        result: dict[str, Any] = {}

        # Basic parameters
        if params.temperature is not None:
            result["temperature"] = params.temperature
        if params.top_p is not None:
            result["top_p"] = params.top_p
        if params.max_output_tokens is not None:
            result["max_output_tokens"] = params.max_output_tokens
        if params.stop_sequences:
            result["stop"] = params.stop_sequences

        # Thinking -> Reasoning (OpenAI o1/o3 style)
        if params.thinking and params.thinking.enabled:
            if not params.thinking.thinking_level:
                raise ValueError("thinking_level is required when thinking is enabled")
            if params.temperature is not None:
                raise ValueError("temperature is not supported for OpenAI when thinking is enabled")
            # Thinking budget is not supported for OpenAI
            if params.thinking.thinking_budget:
                warnings.warn(
                    "thinking_budget is not supported for OpenAI, ignoring",
                    UserWarning,
                    stacklevel=3,
                )
            result["reasoning"] = {
                "effort": params.thinking.thinking_level.value,
                "summary": "concise",
            }

        # Tool choice
        if params.tool_choice:
            if params.tool_choice.type == ToolChoiceType.NONE:
                result["tool_choice"] = "none"
            elif params.tool_choice.type == ToolChoiceType.AUTO:
                result["tool_choice"] = "auto"
            elif params.tool_choice.type == ToolChoiceType.FUNCTION:
                result["tool_choice"] = {
                    "type": "function",
                    "function": {"name": params.tool_choice.name},
                }
            elif params.tool_choice.type == ToolChoiceType.ANY:
                result["tool_choice"] = "required"

            if params.tool_choice.disable_parallel_tool_calls:
                result["parallel_tool_calls"] = False

        return result

    @staticmethod
    def to_anthropic(params: ModelParams) -> dict[str, Any]:
        """Translate ModelParams to Anthropic format."""
        result: dict[str, Any] = {}

        # Basic parameters
        if params.temperature is not None:
            result["temperature"] = params.temperature
        if params.top_k is not None:
            result["top_k"] = params.top_k
        if params.top_p is not None:
            result["top_p"] = params.top_p
        # Anthropic requires max_tokens, so we set a default if not specified
        if params.max_output_tokens is not None:
            result["max_tokens"] = params.max_output_tokens
        else:
            warnings.warn(
                "max_output_tokens not specified for Anthropic, using default value of 16000. "
                "Consider setting it explicitly via params.max_output_tokens",
                UserWarning,
                stacklevel=3,
            )
            result["max_tokens"] = 16000
        if params.stop_sequences:
            result["stop_sequences"] = params.stop_sequences

        # Thinking (Anthropic extended thinking)
        if params.thinking and params.thinking.enabled and params.thinking.thinking_budget:
            if params.temperature != 1:
                raise ValueError("Thinking is only supported with temperature=1")
            # max tokens must be greater than thinking budget
            if (
                params.max_output_tokens is not None
                and params.max_output_tokens <= params.thinking.thinking_budget
            ):
                raise ValueError(
                    f"max_output_tokens must be greater than thinking budget. max_output_tokens: {params.max_output_tokens}, thinking_budget: {params.thinking.thinking_budget}"
                )
            result["thinking"] = {
                "budget_tokens": params.thinking.thinking_budget,
                "type": "enabled",
            }

        # Tool choice
        if params.tool_choice:
            if params.tool_choice.type == ToolChoiceType.NONE:
                result["tool_choice"] = {"type": "none"}
            elif params.tool_choice.type == ToolChoiceType.AUTO:
                result["tool_choice"] = {"type": "auto"}
            elif params.tool_choice.type == ToolChoiceType.FUNCTION:
                result["tool_choice"] = {"type": "tool", "name": params.tool_choice.name}
            elif params.tool_choice.type == ToolChoiceType.ANY:
                result["tool_choice"] = {"type": "any"}

            if params.tool_choice.disable_parallel_tool_calls:
                result["disable_parallel_tool_use"] = True

        return result

    @staticmethod
    def to_gemini(params: ModelParams) -> dict[str, Any]:
        """Translate ModelParams to Gemini format."""
        result: dict[str, Any] = {}

        # Basic parameters
        if params.temperature is not None:
            result["temperature"] = params.temperature
        if params.top_k is not None:
            result["top_k"] = params.top_k
        if params.top_p is not None:
            result["top_p"] = params.top_p
        if params.max_output_tokens is not None:
            result["max_output_tokens"] = params.max_output_tokens
        if params.stop_sequences:
            result["stop_sequences"] = params.stop_sequences

        # Thinking (Gemini 2.5 thinking config)
        if params.thinking and params.thinking.enabled and params.thinking.thinking_budget:
            result["thinking_budget"] = params.thinking.thinking_budget
            result["include_thoughts"] = True

        # Note: Tool choice for Gemini would need special handling with types.ToolConfig
        # This is simplified and can be extended in provider-specific code

        return result


class LLMConfig(BaseModel):
    """
    Configuration for LLM service interactions.

    Attributes:
        provider: The LLM provider to use
        model_id: The identifier of the model to use (e.g., 'gpt-4', 'gemini-2.5-flash')
        params: Typed model parameters with validation
        provider_params: Provider-specific overrides (takes precedence over params)
        credentials: Optional explicit credentials (if None, providers use their default auth)

    Example:
        >>> config = LLMConfig(
        ...     provider=Provider.GEMINI,
        ...     model_id="gemini-2.5-flash",
        ...     params=ModelParams(temperature=0.7, max_output_tokens=4096),
        ... )
    """

    provider: Provider = Field(description="Provider of the LLM service.")
    model_id: str = Field(description="Model identifier (e.g., 'gpt-4', 'gpt-3.5-turbo')")

    # Typed parameters
    params: ModelParams = Field(default_factory=ModelParams)

    # Provider-specific overrides (takes precedence over params)
    provider_params: dict[str, Any] = Field(default_factory=dict)

    # Credentials (optional - if None, providers use their default auth)
    credentials: Credentials | None = None

    def get_effective_params(self) -> dict[str, Any]:
        """
        Get effective parameters by merging typed params with provider-specific overrides.

        Precedence:
            1. provider_params (highest - provider-specific overrides)
            2. params (translated to provider format)

        Returns:
            Merged parameters dict in provider-specific format
        """
        # Start with translated params
        match self.provider:
            case Provider.OPENAI | Provider.AZURE_OPENAI:
                result = ParamTranslator.to_openai(self.params)
            case Provider.ANTHROPIC | Provider.BEDROCK_ANTHROPIC:
                result = ParamTranslator.to_anthropic(self.params)
            case Provider.GEMINI:
                result = ParamTranslator.to_gemini(self.params)
            case _:
                result = {}

        # Provider-specific overrides
        result.update(self.provider_params)

        return result


# Type alias for preset name or config object
LLMConfigOrPreset = str | LLMConfig


# Built-in preset factory (lazy loaded - no memory overhead at startup)
def _create_builtin_preset(name: str) -> LLMConfig:
    """Create built-in preset config on-demand."""
    match name:
        case "gemini-fast":
            return LLMConfig(
                provider=Provider.GEMINI,
                model_id="gemini-2.0-flash",
                params=ModelParams(temperature=0.7, max_output_tokens=4096),
            )
        case "gemini-thinking":
            return LLMConfig(
                provider=Provider.GEMINI,
                model_id="gemini-2.5-flash",
                params=ModelParams(
                    temperature=1.0,
                    thinking=ThinkingConfig(enabled=True, thinking_budget=10000),
                ),
            )
        case "claude-creative":
            return LLMConfig(
                provider=Provider.ANTHROPIC,
                model_id="claude-sonnet-4-5",
                params=ModelParams(temperature=1.0, max_output_tokens=8192),
            )
        case "claude-precise":
            return LLMConfig(
                provider=Provider.ANTHROPIC,
                model_id="claude-sonnet-4-5",
                params=ModelParams(temperature=0.0, max_output_tokens=4096),
            )
        case "gpt4o":
            return LLMConfig(
                provider=Provider.OPENAI,
                model_id="gpt-4o",
                params=ModelParams(temperature=0.7),
            )
        case "gpt4o-mini":
            return LLMConfig(
                provider=Provider.OPENAI,
                model_id="gpt-4o-mini",
                params=ModelParams(temperature=0.7),
            )
        case "bedrock-claude-sonnet-4-5":
            return LLMConfig(
                provider=Provider.BEDROCK_ANTHROPIC,
                model_id="global.anthropic.claude-sonnet-4-5-20250929-v1:0",
                params=ModelParams(temperature=0.7, max_output_tokens=8192),
            )
        case "bedrock-claude-haiku-4-5":
            return LLMConfig(
                provider=Provider.BEDROCK_ANTHROPIC,
                model_id="global.anthropic.claude-haiku-4-5-20251001-v1:0",
                params=ModelParams(temperature=0.7, max_output_tokens=4096),
            )
        case "azure-gpt-4o":
            return LLMConfig(
                provider=Provider.AZURE_OPENAI,
                model_id="gpt-4o",  # Note: This is typically your deployment name
                params=ModelParams(temperature=0.7, max_output_tokens=4096),
            )
        case "azure-gpt-4o-mini":
            return LLMConfig(
                provider=Provider.AZURE_OPENAI,
                model_id="gpt-4o-mini",  # Note: This is typically your deployment name
                params=ModelParams(temperature=0.7, max_output_tokens=4096),
            )
        case _:
            raise KeyError(f"Unknown built-in preset: {name}")


# Built-in preset names (lightweight set)
BUILTIN_PRESET_NAMES = frozenset(
    [
        "gemini-fast",
        "gemini-thinking",
        "claude-creative",
        "claude-precise",
        "gpt4o",
        "gpt4o-mini",
        "bedrock-claude-sonnet-4-5",
        "bedrock-claude-haiku-4-5",
        "azure-gpt-4o",
        "azure-gpt-4o-mini",
    ]
)

# User presets (only user-defined configs are stored)
_user_presets: dict[str, LLMConfig] = {}


def register_preset(name: str, config: LLMConfig) -> None:
    """Register a user-defined preset."""
    _user_presets[name] = config


def get_preset(name: str) -> LLMConfig:
    """
    Get preset by name.

    User presets take precedence over built-in presets.
    Built-in presets are created on-demand (lazy).

    Args:
        name: The name of the preset to retrieve

    Returns:
        LLMConfig instance for the specified preset

    Raises:
        KeyError: If preset name is not found

    Example:
        >>> config = get_preset("gemini-thinking")
        >>> response = llm(config, messages)
    """
    if name in _user_presets:
        return _user_presets[name]
    if name in BUILTIN_PRESET_NAMES:
        return _create_builtin_preset(name)
    raise KeyError(f"Preset '{name}' not found. Available: {list_presets()}")


def list_presets() -> list[str]:
    """List all available preset names."""
    return sorted(set(_user_presets.keys()) | BUILTIN_PRESET_NAMES)
