"""Example demonstrating Anthropic's prompt caching feature.

Prompt caching allows you to cache frequently used content blocks to reduce
costs and improve response times. This is especially useful for:
- Long system prompts
- Large documents or context
- Few-shot examples
- Reusable instruction sets

Learn more: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
"""

import os

from thinllm import (
    AIMessage,
    AnthropicCacheControl,
    ContentExtra,
    InputTextBlock,
    LLMConfig,
    ModelParams,
    SystemMessage,
    UserMessage,
    llm,
)


def example_basic_caching():
    """Basic example with default TTL (5 minutes)."""
    print("\n=== Example 1: Basic Caching (Default TTL) ===\n")

    config = LLMConfig(
        provider="anthropic",
        model_id="claude-sonnet-4-5",
        params=ModelParams(max_output_tokens=1024),
    )

    # System message with cached instruction
    system_msg = SystemMessage(
        content=[
            InputTextBlock(text="You are a helpful assistant."),
            InputTextBlock(
                text="Here is a large document you should reference: "
                + "Lorem ipsum dolor sit amet... " * 100,  # Simulate large document
                extra=ContentExtra(
                    anthropic_cache_control=AnthropicCacheControl()  # Default TTL
                ),
            ),
        ]
    )

    user_msg = UserMessage(content="Summarize the main points.")

    response = llm(config, [system_msg, user_msg])
    print(f"Response: {response.content}")
    print(f"\nUsage: {response.metadata.get('usage', {})}")


def example_explicit_ttl():
    """Example with explicit 1-hour TTL for long-lived cache."""
    print("\n=== Example 2: Explicit TTL (1 hour) ===\n")

    config = LLMConfig(
        provider="anthropic",
        model_id="claude-sonnet-4-5",
        params=ModelParams(max_output_tokens=512),
    )

    # Cache with 1 hour TTL for stable, frequently-used content
    system_msg = SystemMessage(
        content=[
            InputTextBlock(
                text="You are an expert Python developer with deep knowledge of best practices.",
                extra=ContentExtra(
                    anthropic_cache_control=AnthropicCacheControl(ttl="1h")
                ),
            ),
        ]
    )

    user_msg = UserMessage(content="What are the key principles of writing clean Python code?")

    response = llm(config, [system_msg, user_msg])
    print(f"Response: {response.content}")


def example_multi_turn_conversation():
    """Example showing cache reuse across multiple turns."""
    print("\n=== Example 3: Multi-turn Conversation with Caching ===\n")

    config = LLMConfig(
        provider="anthropic",
        model_id="claude-sonnet-4-5",
        params=ModelParams(max_output_tokens=512),
    )

    # Cached system prompt
    system_msg = SystemMessage(
        content=[
            InputTextBlock(
                text="You are a knowledgeable assistant. Reference this context:\n"
                + "The year is 2026. AI has advanced significantly. "
                + "Large language models are widely used in production. " * 20,
                extra=ContentExtra(
                    anthropic_cache_control=AnthropicCacheControl(ttl="1h")
                ),
            )
        ]
    )

    # First turn
    print("Turn 1:")
    messages = [
        system_msg,
        UserMessage(content="What year is it?"),
    ]
    response1 = llm(config, messages)
    print(f"Assistant: {response1.content}\n")

    # Second turn - system prompt is cached, reducing costs
    print("Turn 2:")
    messages.extend(
        [
            AIMessage(content=response1.content),
            UserMessage(content="What are LLMs used for?"),
        ]
    )
    response2 = llm(config, messages)
    print(f"Assistant: {response2.content}\n")


def example_mixed_cached_and_regular():
    """Example with both cached and regular content."""
    print("\n=== Example 4: Mixed Cached and Regular Content ===\n")

    config = LLMConfig(
        provider="anthropic",
        model_id="claude-sonnet-4-5",
        params=ModelParams(max_output_tokens=512),
    )

    system_msg = SystemMessage(
        content=[
            # Regular instruction (not cached - changes frequently)
            InputTextBlock(text="You are a helpful assistant. Today's priority: Be concise."),
            # Cached examples (stable content)
            InputTextBlock(
                text="Examples of good responses:\n"
                + "Q: What is 2+2? A: 4\n"
                + "Q: Capital of France? A: Paris\n" * 10,
                extra=ContentExtra(
                    anthropic_cache_control=AnthropicCacheControl(ttl="1h")
                ),
            ),
        ]
    )

    user_msg = UserMessage(content="What is the capital of Italy?")

    response = llm(config, [system_msg, user_msg])
    print(f"Response: {response.content}")


def example_disabled_caching():
    """Example showing how to explicitly disable caching."""
    print("\n=== Example 5: Disabled Caching ===\n")

    config = LLMConfig(
        provider="anthropic",
        model_id="claude-sonnet-4-5",
        params=ModelParams(max_output_tokens=512),
    )

    system_msg = SystemMessage(
        content=[
            InputTextBlock(
                text="Dynamic content that changes frequently.",
                extra=ContentExtra(
                    # Explicitly disable caching
                    anthropic_cache_control=AnthropicCacheControl(enabled=False)
                ),
            )
        ]
    )

    user_msg = UserMessage(content="Hello!")

    response = llm(config, [system_msg, user_msg])
    print(f"Response: {response.content}")


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        print("Please set it before running this example:")
        print("  export ANTHROPIC_API_KEY='your-api-key-here'")
        exit(1)

    # Run examples
    example_basic_caching()
    example_explicit_ttl()
    example_multi_turn_conversation()
    example_mixed_cached_and_regular()
    example_disabled_caching()

    print("\n=== Cache Control Best Practices ===")
    print("1. Cache large, stable content (system prompts, documents, examples)")
    print("2. Use 1h TTL for content that rarely changes")
    print("3. Use 5m TTL (default) for moderately stable content")
    print("4. Don't cache frequently changing content")
    print("5. Monitor usage metrics to optimize cache effectiveness")
    print("6. Cache control is Anthropic-only (ignored by other providers)")
