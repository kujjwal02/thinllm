"""Unit tests for LLM configuration."""

import pytest

from thinllm.config import LLMConfig, ModelParams, Provider

# Test constants
DEFAULT_TEMPERATURE = 0.7


@pytest.mark.unit
def test_llm_config_creation():
    """Test that LLMConfig can be created with typed parameters."""
    config = LLMConfig(
        provider=Provider.OPENAI,
        model_id="gpt-4o-mini",
        params=ModelParams(temperature=DEFAULT_TEMPERATURE),
    )

    assert config.provider == Provider.OPENAI
    assert config.model_id == "gpt-4o-mini"
    assert config.params.temperature == DEFAULT_TEMPERATURE

    # Test get_effective_params translates correctly
    effective = config.get_effective_params()
    assert effective["temperature"] == DEFAULT_TEMPERATURE


@pytest.mark.unit
def test_llm_config_default_params():
    """Test that LLMConfig uses default ModelParams when not specified."""
    config = LLMConfig(provider=Provider.OPENAI, model_id="gpt-4o")

    assert config.params is not None
    assert config.params.temperature is None  # Default is None
    assert config.provider_params == {}


@pytest.mark.unit
@pytest.mark.parametrize(
    "provider",
    [
        Provider.OPENAI,
        Provider.ANTHROPIC,
        Provider.VERTEXAI,
        Provider.AWS_CONVERSE_API,
        Provider.AZURE_OPENAI,
    ],
)
def test_llm_config_all_providers(provider):
    """Test that LLMConfig accepts all provider types."""
    config = LLMConfig(provider=provider, model_id="test-model")

    assert config.provider == provider
    assert config.model_id == "test-model"


@pytest.mark.unit
def test_model_params_validation():
    """Test that ModelParams validates parameter ranges."""
    from thinllm.config import ModelParams

    # Valid params
    params = ModelParams(temperature=0.5, top_p=0.9, max_output_tokens=1000)
    assert params.temperature == 0.5
    assert params.top_p == 0.9
    assert params.max_output_tokens == 1000

    # Invalid temperature (should raise)
    with pytest.raises(Exception):  # Pydantic validation error
        ModelParams(temperature=3.0)

    # Invalid top_p (should raise)
    with pytest.raises(Exception):
        ModelParams(top_p=1.5)


@pytest.mark.unit
def test_thinking_config():
    """Test ThinkingConfig functionality."""
    from thinllm.config import ThinkingConfig, ThinkingLevel

    config = ThinkingConfig(enabled=True, thinking_budget=10000, thinking_level=ThinkingLevel.HIGH)
    assert config.enabled is True
    assert config.thinking_budget == 10000
    assert config.thinking_level == ThinkingLevel.HIGH

    # Test negative budget validation
    with pytest.raises(ValueError, match="thinking_budget must be non-negative"):
        ThinkingConfig(thinking_budget=-100)


@pytest.mark.unit
def test_tool_choice_config():
    """Test ToolChoiceConfig functionality."""
    from thinllm.config import ToolChoiceConfig, ToolChoiceType

    # Auto mode
    config = ToolChoiceConfig(type=ToolChoiceType.AUTO)
    assert config.type == ToolChoiceType.AUTO

    # Function mode requires name
    with pytest.raises(ValueError, match="name is required when type=FUNCTION"):
        ToolChoiceConfig(type=ToolChoiceType.FUNCTION)

    # Valid function mode
    config = ToolChoiceConfig(type=ToolChoiceType.FUNCTION, name="my_function")
    assert config.name == "my_function"

    # Allowlist mode requires allowed_tools
    with pytest.raises(ValueError, match="allowed_tools is required when type=ALLOWLIST"):
        ToolChoiceConfig(type=ToolChoiceType.ALLOWLIST)


@pytest.mark.unit
def test_provider_params_precedence():
    """Test that provider_params override translated params."""
    from thinllm.config import ModelParams

    config = LLMConfig(
        provider=Provider.OPENAI,
        model_id="gpt-4o",
        params=ModelParams(temperature=0.7),
        provider_params={"temperature": 0.9},  # Override
    )

    effective = config.get_effective_params()
    assert effective["temperature"] == 0.9  # provider_params takes precedence


@pytest.mark.unit
def test_nullable_thinking_and_tool_choice():
    """Test that thinking and tool_choice can be None."""
    from thinllm.config import ModelParams

    # Create ModelParams without thinking or tool_choice
    params = ModelParams(temperature=0.7)
    assert params.thinking is None
    assert params.tool_choice is None

    # Test with full config
    config = LLMConfig(
        provider=Provider.OPENAI,
        model_id="gpt-4o",
        params=ModelParams(temperature=0.5),
    )

    # get_effective_params should handle None gracefully
    effective = config.get_effective_params()
    assert effective["temperature"] == 0.5
    assert "reasoning" not in effective  # No thinking config
    assert "tool_choice" not in effective  # No tool choice config


@pytest.mark.unit
def test_preset_system():
    """Test preset registration and retrieval."""
    from thinllm.config import get_preset, list_presets, register_preset, BUILTIN_PRESET_NAMES

    # Test built-in presets exist
    assert "gemini-fast" in BUILTIN_PRESET_NAMES
    assert "claude-creative" in BUILTIN_PRESET_NAMES
    assert "gpt4o" in BUILTIN_PRESET_NAMES

    # Test listing presets
    presets = list_presets()
    assert "gemini-fast" in presets

    # Test getting built-in preset
    config = get_preset("gemini-fast")
    assert config.provider == Provider.GEMINI
    assert config.model_id == "gemini-2.0-flash"

    # Test registering custom preset
    custom_config = LLMConfig(
        provider=Provider.OPENAI,
        model_id="gpt-4o-mini",
        params=ModelParams(temperature=0.3),
    )
    register_preset("my-custom-preset", custom_config)

    # Test retrieving custom preset
    retrieved = get_preset("my-custom-preset")
    assert retrieved.provider == Provider.OPENAI
    assert retrieved.model_id == "gpt-4o-mini"

    # Test invalid preset
    with pytest.raises(KeyError):
        get_preset("non-existent-preset")
