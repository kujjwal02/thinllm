"""Example demonstrating Anthropic's interleaved thinking with tool use.

Interleaved thinking allows Claude to think between tool calls, enabling
more sophisticated reasoning in multi-step workflows. This is particularly
useful for complex problems that require planning and reflection across
multiple tool interactions.

Learn more: https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking
"""

import os

from thinllm import LLMConfig, ModelParams, ThinkingConfig, llm
from thinllm.messages import ReasoningContent, UserMessage


def get_weather(location: str) -> str:
    """Get the current weather for a location.
    
    Args:
        location: The city name
        
    Returns:
        Weather information string
    """
    # Mock weather data
    weather_data = {
        "Paris": "Sunny, 22°C (72°F), Light breeze",
        "London": "Cloudy, 15°C (59°F), Moderate rain",
        "Tokyo": "Clear, 28°C (82°F), Humid",
        "New York": "Partly cloudy, 18°C (64°F), Windy",
    }
    return weather_data.get(location, f"Weather data not available for {location}")


def get_time(timezone: str) -> str:
    """Get the current time for a timezone.
    
    Args:
        timezone: The timezone (e.g., 'Europe/Paris', 'America/New_York')
        
    Returns:
        Current time string
    """
    # Mock time data
    time_data = {
        "Europe/Paris": "14:30 CET",
        "Europe/London": "13:30 GMT",
        "Asia/Tokyo": "22:30 JST",
        "America/New_York": "08:30 EST",
    }
    return time_data.get(timezone, f"Time data not available for {timezone}")


def calculate(expression: str) -> str:
    """Calculate a mathematical expression.
    
    Args:
        expression: Mathematical expression to evaluate
        
    Returns:
        Result of the calculation
    """
    try:
        # Using eval safely for simple expressions
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error calculating: {str(e)}"


def example_basic_interleaved_thinking():
    """Basic example with interleaved thinking enabled."""
    print("\n=== Example 1: Basic Interleaved Thinking ===\n")

    config = LLMConfig(
        provider="anthropic",
        model_id="claude-sonnet-4-5",
        params=ModelParams(
            max_output_tokens=4096,
            temperature=1.0,
            thinking=ThinkingConfig(
                enabled=True,
                thinking_budget=10000,
                anthropic_interleaved_thinking=True,
            ),
        ),
    )

    messages = [
        UserMessage(
            content="What's the weather in Paris right now, and what time is it there? "
            "After you find out, tell me what activities would be good for that weather."
        )
    ]

    print("Query: What's the weather and time in Paris, and suggest activities?\n")

    response = llm(config, messages, tools=[get_weather, get_time])

    # Display the response
    if isinstance(response.content, list):
        for block in response.content:
            if isinstance(block, ReasoningContent):
                print(f"[Thinking] {' '.join(block.summaries)}\n")
            else:
                print(f"Response: {block}\n")
    else:
        print(f"Response: {response.content}\n")


def example_complex_multi_step():
    """Example demonstrating complex multi-step reasoning with calculations."""
    print("\n=== Example 2: Complex Multi-Step Reasoning ===\n")

    config = LLMConfig(
        provider="anthropic",
        model_id="claude-sonnet-4-5",
        params=ModelParams(
            max_output_tokens=6144,
            temperature=1.0,
            thinking=ThinkingConfig(
                enabled=True,
                thinking_budget=15000,
                anthropic_interleaved_thinking=True,
            ),
        ),
    )

    messages = [
        UserMessage(
            content="I'm planning a trip. First, check the weather in Paris and Tokyo. "
            "Then, calculate how much it would cost if a hotel is $150 per night "
            "for 3 nights in each city. Which city has better weather right now?"
        )
    ]

    print("Query: Compare weather and calculate trip costs for Paris and Tokyo\n")

    response = llm(config, messages, tools=[get_weather, get_time, calculate])

    # Display the response with thinking blocks highlighted
    if isinstance(response.content, list):
        for block in response.content:
            if isinstance(block, ReasoningContent):
                # Show thinking process
                if block.summaries:
                    print(f"[Thinking Summary] {' '.join(block.summaries)}")
                if block.contents:
                    print(f"[Thinking Details] {' '.join(block.contents[:100])}...\n")
            else:
                print(f"Response: {block}\n")
    else:
        print(f"Response: {response.content}\n")


def example_without_interleaved_thinking():
    """Example without interleaved thinking for comparison."""
    print("\n=== Example 3: Regular Thinking (No Interleaving) ===\n")

    config = LLMConfig(
        provider="anthropic",
        model_id="claude-sonnet-4-5",
        params=ModelParams(
            max_output_tokens=4096,
            temperature=1.0,
            thinking=ThinkingConfig(
                enabled=True,
                thinking_budget=3000,  # Must be less than max_output_tokens
                anthropic_interleaved_thinking=False,
            ),
        ),
    )

    messages = [
        UserMessage(
            content="What's the weather in Paris and what time is it there?"
        )
    ]

    print("Query: What's the weather and time in Paris? (Regular thinking)\n")

    response = llm(config, messages, tools=[get_weather, get_time])

    if isinstance(response.content, list):
        for block in response.content:
            if isinstance(block, ReasoningContent):
                print(f"[Thinking] {' '.join(block.summaries)}\n")
            else:
                print(f"Response: {block}\n")
    else:
        print(f"Response: {response.content}\n")

    print("Note: With regular thinking, Claude thinks once at the start.")
    print("With interleaved thinking, Claude can think between tool calls.\n")


def main():
    """Run all examples."""
    print("=" * 60)
    print("Anthropic Interleaved Thinking Examples")
    print("=" * 60)
    print(
        "\nInterleaved thinking allows Claude to reason between tool calls,\n"
        "enabling more sophisticated multi-step problem solving.\n"
    )

    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set.")
        print("Please set it to run these examples.")
        return

    try:
        # Run examples
        example_basic_interleaved_thinking()
        example_complex_multi_step()
        example_without_interleaved_thinking()

        print("\n" + "=" * 60)
        print("Examples completed!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError running examples: {e}")
        print("\nMake sure you have:")
        print("1. Set ANTHROPIC_API_KEY environment variable")
        print("2. Installed thinllm: pip install thinllm")
        print("3. Using a model that supports extended thinking")


if __name__ == "__main__":
    main()
