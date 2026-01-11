"""
Comprehensive example demonstrating ThinLLM with Anthropic models via Amazon Bedrock.

This example showcases:
- Basic text generation with Bedrock
- Global vs Regional endpoints
- AWS credential configuration
- Structured output with Pydantic models
- Function calling with tools
- Streaming responses

Prerequisites:
1. AWS Account with Bedrock access
2. Model access enabled in AWS Console (Bedrock > Model Access)
3. AWS credentials configured (see setup instructions below)
4. Install dependencies: uv sync --group anthropic-bedrock

AWS Credentials Setup:
You can provide credentials in multiple ways (in order of precedence):

Option 1: Explicit credentials in code (shown in examples below)
Option 2: Environment variables:
    export AWS_ACCESS_KEY_ID="your-access-key"
    export AWS_SECRET_ACCESS_KEY="your-secret-key"
    export AWS_REGION="us-west-2"
    
Option 3: AWS credentials file (~/.aws/credentials):
    [default]
    aws_access_key_id = your-access-key
    aws_secret_access_key = your-secret-key
    
Option 4: IAM roles (for EC2, ECS, Lambda, etc.)
Option 5: AWS SSO

Model IDs:
- Global endpoint: "global.anthropic.claude-sonnet-4-5-20250929-v1:0"
- Regional endpoint (US): "anthropic.claude-sonnet-4-5-20250929-v1:0"
"""

import os
from pydantic import BaseModel

from thinllm import (
    Credentials,
    LLMConfig,
    ModelParams,
    Provider,
    UserMessage,
    llm,
)


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


# ============================================================================
# Example 1: Basic text generation with global endpoint
# ============================================================================
def example_basic_text() -> None:
    """Basic text generation using Bedrock with global endpoint."""
    print_section("Example 1: Basic Text Generation (Global Endpoint)")

    # Using global endpoint for maximum availability
    config = LLMConfig(
        provider=Provider.BEDROCK_ANTHROPIC,
        model_id="global.anthropic.claude-sonnet-4-5-20250929-v1:0",
        params=ModelParams(
            temperature=0.7,
            max_output_tokens=1024,
        ),
    )

    messages = [
        UserMessage(content="Explain quantum computing in 2-3 sentences.")
    ]

    response = llm(config, messages)
    print(f"Response: {response.content}")


# ============================================================================
# Example 2: Using regional endpoint with explicit credentials
# ============================================================================
def example_regional_endpoint() -> None:
    """Using regional endpoint with explicit AWS credentials."""
    print_section("Example 2: Regional Endpoint with Explicit Credentials")

    # Only use this if you want to provide credentials explicitly
    # Otherwise, omit the credentials parameter to use AWS default providers
    config = LLMConfig(
        provider=Provider.BEDROCK_ANTHROPIC,
        model_id="anthropic.claude-haiku-4-5-20251001-v1:0",  # Regional endpoint (no global. prefix)
        params=ModelParams(
            temperature=0.7,
            max_output_tokens=512,
        ),
        credentials=Credentials(
            aws_access_key=os.getenv("AWS_ACCESS_KEY_ID"),  # Optional
            aws_secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),  # Optional
            aws_session_token=os.getenv("AWS_SESSION_TOKEN"),  # Optional
            aws_region="us-west-2",  # Optional, defaults to AWS_REGION env var
        ),
    )

    messages = [
        UserMessage(content="What is the capital of France?")
    ]

    response = llm(config, messages)
    print(f"Response: {response.content}")
    print(f"\nNote: Using regional endpoint in {config.credentials.aws_region if config.credentials else 'default region'}")


# ============================================================================
# Example 3: Using built-in presets
# ============================================================================
def example_presets() -> None:
    """Using built-in Bedrock presets."""
    print_section("Example 3: Built-in Bedrock Presets")

    messages = [
        UserMessage(content="Write a haiku about clouds.")
    ]

    # Built-in preset for Claude Sonnet 4.5 on Bedrock (global endpoint)
    response = llm("bedrock-claude-sonnet-4-5", messages)
    print(f"Response from bedrock-claude-sonnet-4-5 preset:\n{response.content}")

    print("\n" + "-" * 70 + "\n")

    # Built-in preset for Claude Haiku 4.5 on Bedrock (global endpoint)
    response = llm("bedrock-claude-haiku-4-5", messages)
    print(f"Response from bedrock-claude-haiku-4-5 preset:\n{response.content}")


# ============================================================================
# Example 4: Structured output with Pydantic
# ============================================================================
def example_structured_output() -> None:
    """Structured output using Pydantic models."""
    print_section("Example 4: Structured Output")

    class Recipe(BaseModel):
        """A recipe with name, ingredients, and instructions."""
        name: str
        cuisine: str
        ingredients: list[str]
        instructions: list[str]
        prep_time_minutes: int
        cook_time_minutes: int

    config = LLMConfig(
        provider=Provider.BEDROCK_ANTHROPIC,
        model_id="global.anthropic.claude-sonnet-4-5-20250929-v1:0",
        params=ModelParams(temperature=0.7, max_output_tokens=2048),
    )

    messages = [
        UserMessage(content="Create a recipe for chocolate chip cookies.")
    ]

    recipe = llm(config, messages, output_schema=Recipe)
    
    print(f"Recipe: {recipe.name}")
    print(f"Cuisine: {recipe.cuisine}")
    print(f"Prep Time: {recipe.prep_time_minutes} minutes")
    print(f"Cook Time: {recipe.cook_time_minutes} minutes")
    print(f"\nIngredients:")
    for ingredient in recipe.ingredients:
        print(f"  - {ingredient}")
    print(f"\nInstructions:")
    for i, instruction in enumerate(recipe.instructions, 1):
        print(f"  {i}. {instruction}")


# ============================================================================
# Example 5: Function calling with tools
# ============================================================================
def example_function_calling() -> None:
    """Function calling with custom tools."""
    print_section("Example 5: Function Calling with Tools")

    def get_weather(location: str, unit: str = "celsius") -> str:
        """
        Get the current weather for a location.
        
        Args:
            location: The city and country, e.g. "Paris, France"
            unit: Temperature unit, either "celsius" or "fahrenheit"
        
        Returns:
            Weather information as a string
        """
        # Mock implementation
        temps = {"celsius": "22°C", "fahrenheit": "72°F"}
        return f"The weather in {location} is sunny with a temperature of {temps[unit]}"

    def calculate(operation: str, a: float, b: float) -> float:
        """
        Perform a mathematical calculation.
        
        Args:
            operation: The operation to perform (add, subtract, multiply, divide)
            a: First number
            b: Second number
        
        Returns:
            Result of the calculation
        """
        operations = {
            "add": lambda x, y: x + y,
            "subtract": lambda x, y: x - y,
            "multiply": lambda x, y: x * y,
            "divide": lambda x, y: x / y,
        }
        return operations[operation](a, b)

    config = LLMConfig(
        provider=Provider.BEDROCK_ANTHROPIC,
        model_id="global.anthropic.claude-sonnet-4-5-20250929-v1:0",
        params=ModelParams(temperature=0.7, max_output_tokens=1024),
    )

    messages = [
        UserMessage(
            content="What's the weather in Tokyo? Also, what's 15 multiplied by 7?"
        )
    ]

    response = llm(config, messages, tools=[get_weather, calculate])
    
    print(f"Response: {response.content}")
    
    # Check if tools were called
    if hasattr(response, 'tool_calls') and response.tool_calls:
        print(f"\nTools called: {len(response.tool_calls)}")
        for i, tool_call in enumerate(response.tool_calls, 1):
            print(f"  {i}. {tool_call.name}({tool_call.input})")


# ============================================================================
# Example 6: Streaming responses
# ============================================================================
def example_streaming() -> None:
    """Streaming responses for real-time output."""
    print_section("Example 6: Streaming Responses")

    config = LLMConfig(
        provider=Provider.BEDROCK_ANTHROPIC,
        model_id="global.anthropic.claude-sonnet-4-5-20250929-v1:0",
        params=ModelParams(temperature=0.7, max_output_tokens=1024),
    )

    messages = [
        UserMessage(
            content="Write a short story (3-4 sentences) about a robot learning to paint."
        )
    ]

    print("Streaming response:\n")
    for chunk in llm(config, messages, stream=True):
        print(chunk.content, end="", flush=True)
    
    print("\n")


# ============================================================================
# Example 7: Multi-turn conversation
# ============================================================================
def example_multi_turn() -> None:
    """Multi-turn conversation maintaining context."""
    print_section("Example 7: Multi-turn Conversation")

    config = LLMConfig(
        provider=Provider.BEDROCK_ANTHROPIC,
        model_id="global.anthropic.claude-haiku-4-5-20251001-v1:0",  # Using Haiku for faster responses
        params=ModelParams(temperature=0.7, max_output_tokens=512),
    )

    # First turn
    messages = [
        UserMessage(content="I'm thinking of a number between 1 and 10.")
    ]
    response1 = llm(config, messages)
    print(f"User: {messages[0].content}")
    print(f"Assistant: {response1.content}")

    # Second turn - add previous response and new user message
    from thinllm import AIMessage
    messages.append(AIMessage(content=response1.content))
    messages.append(UserMessage(content="It's 7. Can you write a fun fact about the number 7?"))
    
    response2 = llm(config, messages)
    print(f"\nUser: {messages[2].content}")
    print(f"Assistant: {response2.content}")


# ============================================================================
# Example 8: Using AWS credentials from environment
# ============================================================================
def example_env_credentials() -> None:
    """Using AWS credentials from environment variables."""
    print_section("Example 8: AWS Credentials from Environment")

    # This config doesn't specify credentials, so it will use:
    # 1. AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY env vars
    # 2. ~/.aws/credentials file
    # 3. IAM role (if running on EC2/ECS/Lambda)
    # 4. AWS SSO
    config = LLMConfig(
        provider=Provider.BEDROCK_ANTHROPIC,
        model_id="global.anthropic.claude-haiku-4-5-20251001-v1:0",
        params=ModelParams(temperature=0.7, max_output_tokens=256),
    )

    messages = [
        UserMessage(content="What are the three primary colors?")
    ]

    response = llm(config, messages)
    print(f"Response: {response.content}")
    print(f"\nNote: Used AWS default credential providers (env vars, ~/.aws/credentials, IAM roles, etc.)")


# ============================================================================
# Main execution
# ============================================================================
def main() -> None:
    """Run all examples."""
    print("\n" + "=" * 70)
    print("  ThinLLM - Amazon Bedrock Examples")
    print("=" * 70)
    print("\nThese examples demonstrate using Anthropic Claude models via AWS Bedrock.")
    print("Make sure you have AWS credentials configured and Bedrock model access enabled.")
    
    examples = [
        ("Basic Text Generation", example_basic_text),
        ("Regional Endpoint", example_regional_endpoint),
        ("Built-in Presets", example_presets),
        ("Structured Output", example_structured_output),
        ("Function Calling", example_function_calling),
        ("Streaming", example_streaming),
        ("Multi-turn Conversation", example_multi_turn),
        ("Environment Credentials", example_env_credentials),
    ]

    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    print("  0. Run all examples")
    
    choice = input("\nSelect example to run (0-8): ").strip()
    
    try:
        if choice == "0":
            for name, func in examples:
                try:
                    func()
                except Exception as e:
                    print(f"\n⚠️  Error in {name}: {e}")
                    print("Continuing to next example...\n")
        elif 1 <= int(choice) <= len(examples):
            examples[int(choice) - 1][1]()
        else:
            print("Invalid choice. Please run again and select 0-8.")
    except ValueError:
        print("Invalid input. Please enter a number.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure you have AWS credentials configured")
        print("2. Check that you have Bedrock model access enabled in AWS Console")
        print("3. Verify your AWS region supports the model you're trying to use")
        print("4. Install dependencies: uv sync --group anthropic-bedrock")


if __name__ == "__main__":
    main()
