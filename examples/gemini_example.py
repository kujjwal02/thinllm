"""Example usage of the Gemini provider with thinllm.

This example demonstrates various features of the Gemini provider:
- Basic text generation
- Streaming responses
- Structured output
- Function calling
- Thinking/reasoning mode
- Built-in tools (Google Search)

To run this example, you need:
1. Install the gemini dependency: uv sync --group gemini
2. Set your GEMINI_API_KEY environment variable
3. Run: python examples/gemini_example.py
"""

import os
from pydantic import BaseModel
from dotenv import load_dotenv

from thinllm import llm, LLMConfig, Provider, UserMessage, SystemMessage
from thinllm.tools import tool

# Load environment variables
load_dotenv()

# Check for API key
if not os.getenv("GEMINI_API_KEY"):
    print("Error: GEMINI_API_KEY environment variable not set")
    print("Please set it in your .env file or as an environment variable")
    exit(1)


def example_basic_text_generation():
    """Example 1: Basic text generation with Gemini."""
    print("\n" + "=" * 80)
    print("Example 1: Basic Text Generation")
    print("=" * 80)

    config = LLMConfig(
        provider=Provider.GEMINI,
        model_id="gemini-2.5-flash",
        model_args={"temperature": 0.7}
    )

    messages = [
        UserMessage(content="What is the capital of France? Answer in one sentence.")
    ]

    response = llm(config, messages)
    print(f"\nResponse: {response.content}")


def example_streaming():
    """Example 2: Streaming text generation."""
    print("\n" + "=" * 80)
    print("Example 2: Streaming Response")
    print("=" * 80)

    config = LLMConfig(
        provider=Provider.GEMINI,
        model_id="gemini-2.5-flash",
        model_args={"temperature": 0.7}
    )

    messages = [
        UserMessage(content="Write a short haiku about programming.")
    ]

    print("\nStreaming response:")
    for chunk in llm(config, messages, stream=True):
        if isinstance(chunk.content, list):
            for block in chunk.content:
                if hasattr(block, "text"):
                    print(block.text, end="", flush=True)
        elif isinstance(chunk.content, str):
            print(chunk.content, end="", flush=True)
    print("\n")


def example_structured_output():
    """Example 3: Structured output with Pydantic models."""
    print("\n" + "=" * 80)
    print("Example 3: Structured Output")
    print("=" * 80)

    class Recipe(BaseModel):
        """Schema for a recipe."""
        name: str
        ingredients: list[str]
        prep_time_minutes: int

    config = LLMConfig(
        provider=Provider.GEMINI,
        model_id="gemini-2.5-flash",
        model_args={"temperature": 0.0}
    )

    messages = [
        SystemMessage(content="Extract the recipe information."),
        UserMessage(
            content="Please extract: Chocolate Chip Cookies need flour, sugar, chocolate chips, and butter. Takes 30 minutes to prepare."
        )
    ]

    recipe = llm(config, messages, output_schema=Recipe)
    print(f"\nExtracted Recipe:")
    print(f"  Name: {recipe.name}")
    print(f"  Ingredients: {', '.join(recipe.ingredients)}")
    print(f"  Prep Time: {recipe.prep_time_minutes} minutes")


def example_function_calling():
    """Example 4: Function calling with custom tools."""
    print("\n" + "=" * 80)
    print("Example 4: Function Calling")
    print("=" * 80)

    @tool
    def get_weather(location: str) -> str:
        """
        Get the current weather for a location.

        Args:
            location: The city name

        Returns:
            Weather information
        """
        # Simulated weather data
        return f"The weather in {location} is sunny with a temperature of 72°F."

    config = LLMConfig(
        provider=Provider.GEMINI,
        model_id="gemini-2.5-flash",
        model_args={"temperature": 0.0}
    )

    messages = [
        SystemMessage(content="You are a helpful assistant. Use the available tools when appropriate."),
        UserMessage(content="What's the weather like in San Francisco?")
    ]

    response = llm(config, messages, tools=[get_weather])

    print("\nResponse:")
    if isinstance(response.content, list):
        for block in response.content:
            if hasattr(block, "text"):
                print(f"Text: {block.text}")
            elif hasattr(block, "name"):
                print(f"Tool Call: {block.name}")
                print(f"Arguments: {block.input}")


def example_thinking_mode():
    """Example 5: Enable thinking/reasoning mode."""
    print("\n" + "=" * 80)
    print("Example 5: Thinking Mode")
    print("=" * 80)

    config = LLMConfig(
        provider=Provider.GEMINI,
        model_id="gemini-2.5-pro",
        model_args={
            "temperature": 0.0,
            "thinking_budget": 2048,  # Allocate tokens for reasoning
            "include_thoughts": True,  # Include reasoning in response
        }
    )

    messages = [
        UserMessage(content="What is the sum of the first 10 prime numbers?")
    ]

    response = llm(config, messages)

    print("\nResponse with thinking:")
    if isinstance(response.content, list):
        for block in response.content:
            if hasattr(block, "text"):
                print(f"\nAnswer: {block.text}")
            elif hasattr(block, "contents") and block.contents:
                print(f"\nReasoning:")
                for line in block.contents[:5]:  # Show first 5 lines of reasoning
                    print(f"  {line}")
                if len(block.contents) > 5:
                    print(f"  ... ({len(block.contents) - 5} more lines)")


def example_google_search():
    """Example 6: Use built-in Google Search tool."""
    print("\n" + "=" * 80)
    print("Example 6: Google Search (Built-in Tool)")
    print("=" * 80)

    config = LLMConfig(
        provider=Provider.GEMINI,
        model_id="gemini-2.5-flash",
        model_args={"temperature": 0.0}
    )

    messages = [
        UserMessage(content="What are the latest developments in quantum computing in 2024?")
    ]

    # Use the built-in google_search tool
    response = llm(config, messages, tools=[{"google_search": {}}])

    print("\nResponse with Google Search:")
    if isinstance(response.content, list):
        for block in response.content:
            if hasattr(block, "text") and block.text:
                print(f"{block.text}")


def example_multi_turn_conversation():
    """Example 7: Multi-turn conversation."""
    print("\n" + "=" * 80)
    print("Example 7: Multi-turn Conversation")
    print("=" * 80)

    config = LLMConfig(
        provider=Provider.GEMINI,
        model_id="gemini-2.5-flash",
        model_args={"temperature": 0.7}
    )

    # First turn
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="I have 3 apples."),
    ]

    response1 = llm(config, messages)
    print(f"\nUser: I have 3 apples.")
    print(f"Assistant: {response1.content}")

    # Add the assistant's response to the conversation
    messages.append(response1)

    # Second turn
    messages.append(UserMessage(content="I ate 1 apple. How many do I have now?"))

    response2 = llm(config, messages)
    print(f"\nUser: I ate 1 apple. How many do I have now?")
    print(f"Assistant: {response2.content}")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("Gemini Provider Examples for thinllm")
    print("=" * 80)

    try:
        example_basic_text_generation()
        example_streaming()
        example_structured_output()
        example_function_calling()
        # Note: Thinking mode and Google Search may require specific model access
        # Uncomment to try:
        # example_thinking_mode()
        # example_google_search()
        example_multi_turn_conversation()

        print("\n" + "=" * 80)
        print("All examples completed successfully!")
        print("=" * 80)

    except Exception as e:
        print(f"\nError running examples: {e}")
        print("Make sure you have:")
        print("1. Installed the gemini dependency: uv sync --group gemini")
        print("2. Set your GEMINI_API_KEY environment variable")
        raise


if __name__ == "__main__":
    main()
