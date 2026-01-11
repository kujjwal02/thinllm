"""Unit tests for LLM configuration."""

import pytest

from thinllm.config import LLMConfig, Provider

# Test constants
DEFAULT_TEMPERATURE = 0.7


@pytest.mark.unit
def test_llm_config_creation():
    """Test that LLMConfig can be created with basic parameters."""
    config = LLMConfig(
        provider=Provider.OPENAI,
        model_id="gpt-4o-mini",
        model_args={"temperature": DEFAULT_TEMPERATURE},
    )

    assert config.provider == Provider.OPENAI
    assert config.model_id == "gpt-4o-mini"
    assert config.model_args["temperature"] == DEFAULT_TEMPERATURE


@pytest.mark.unit
def test_llm_config_default_model_args():
    """Test that LLMConfig uses empty dict as default for model_args."""
    config = LLMConfig(provider=Provider.OPENAI, model_id="gpt-4o")

    assert config.model_args == {}


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
