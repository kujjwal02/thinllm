"""Demo of the new LLM config system with typed parameters and presets."""

from thinllm import (
    LLMConfig,
    ModelParams,
    Provider,
    ThinkingConfig,
    get_preset,
    list_presets,
    register_preset,
)

# Example 1: Using typed parameters
print("=" * 50)
print("Example 1: Typed Parameters")
print("=" * 50)

config = LLMConfig(
    provider=Provider.GEMINI,
    model_id="gemini-2.5-flash",
    params=ModelParams(
        temperature=0.7,
        max_output_tokens=4096,
        thinking=ThinkingConfig(enabled=True, thinking_budget=10000),
    ),
)

print(f"Provider: {config.provider}")
print(f"Model: {config.model_id}")
print(f"Temperature: {config.params.temperature}")
print(f"Thinking enabled: {config.params.thinking.enabled}")
print(f"Thinking budget: {config.params.thinking.thinking_budget}")
print(f"\nEffective params: {config.get_effective_params()}")

# Example 2: Using presets
print("\n" + "=" * 50)
print("Example 2: Built-in Presets")
print("=" * 50)

print(f"Available presets: {list_presets()[:6]}")  # Show first 6

gemini_config = get_preset("gemini-thinking")
print(f"\n'gemini-thinking' preset:")
print(f"  Provider: {gemini_config.provider}")
print(f"  Model: {gemini_config.model_id}")
print(f"  Thinking budget: {gemini_config.params.thinking.thinking_budget}")

claude_config = get_preset("claude-precise")
print(f"\n'claude-precise' preset:")
print(f"  Provider: {claude_config.provider}")
print(f"  Model: {claude_config.model_id}")
print(f"  Temperature: {claude_config.params.temperature}")

# Example 3: Custom preset
print("\n" + "=" * 50)
print("Example 3: Custom Preset")
print("=" * 50)

custom_config = LLMConfig(
    provider=Provider.OPENAI,
    model_id="gpt-4o-mini",
    params=ModelParams(
        temperature=0.3,
        max_output_tokens=2048,
    ),
)

register_preset("my-agent", custom_config)
print(f"Registered custom preset: 'my-agent'")

retrieved = get_preset("my-agent")
print(f"Retrieved config - Model: {retrieved.model_id}, Temp: {retrieved.params.temperature}")

# Example 4: Provider-specific overrides
print("\n" + "=" * 50)
print("Example 4: Provider-Specific Overrides")
print("=" * 50)

config_with_override = LLMConfig(
    provider=Provider.ANTHROPIC,
    model_id="claude-sonnet-4-5",
    params=ModelParams(temperature=0.7, max_output_tokens=4096),
    provider_params={"metadata": {"user_id": "demo-user"}},
)

effective = config_with_override.get_effective_params()
print(f"Effective params include Anthropic-specific metadata:")
print(f"  max_tokens (translated): {effective.get('max_tokens')}")
print(f"  metadata (provider-specific): {effective.get('metadata')}")

print("\n" + "=" * 50)
print("All examples completed successfully!")
print("=" * 50)
