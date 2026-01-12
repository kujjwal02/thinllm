"""
Comprehensive example demonstrating ThinLLM with Azure OpenAI.

This example showcases:
- Basic text generation with Azure OpenAI
- API Key authentication
- Microsoft Entra ID (Azure AD) authentication
- Using built-in presets
- Structured output with Pydantic models
- Function calling with tools
- Streaming responses
- Multi-turn conversations

Prerequisites:
1. Azure subscription with Azure OpenAI access
2. Azure OpenAI resource created
3. Model deployed (e.g., gpt-4o, gpt-4o-mini)
4. Install dependencies: uv sync --group azure-openai

Azure OpenAI Setup:
1. Create Azure OpenAI resource in Azure Portal
2. Deploy a model (note the deployment name)
3. Get endpoint URL (e.g., https://your-resource.openai.azure.com)
4. Get API key from "Keys and Endpoint" section OR set up Entra ID

Authentication Options:

Option 1: API Key (Simpler, good for getting started)
    export AZURE_OPENAI_API_KEY="your-api-key"
    export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"

Option 2: Microsoft Entra ID (Enterprise-grade, recommended for production)
    - Use DefaultAzureCredential (supports multiple auth methods)
    - Works with: Azure CLI, managed identities, environment variables, etc.
    - Run: az login (if using Azure CLI)
    export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"

Model Deployment Names:
- In Azure OpenAI, you create "deployments" of models
- The deployment name is what you use as model_id in thinllm
- Example: You might deploy "gpt-4o" model with deployment name "gpt-4o-deployment"
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
# Example 1: Basic text generation with API Key
# ============================================================================
def example_basic_text_api_key() -> None:
    """Basic text generation using Azure OpenAI with API Key authentication."""
    print_section("Example 1: Basic Text Generation (API Key)")

    config = LLMConfig(
        provider=Provider.AZURE_OPENAI,
        model_id="gpt-4o",  # This should be your deployment name
        params=ModelParams(
            temperature=0.7,
            max_output_tokens=1024,
        ),
        credentials=Credentials(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        ),
    )

    messages = [
        UserMessage(content="Explain quantum computing in 2-3 sentences.")
    ]

    response = llm(config, messages)
    print(f"Response: {response.content}")


# ============================================================================
# Example 2: Using Microsoft Entra ID authentication
# ============================================================================
def example_entra_id_auth() -> None:
    """Using Microsoft Entra ID (Azure AD) for authentication."""
    print_section("Example 2: Microsoft Entra ID Authentication")

    # When api_key is not provided, DefaultAzureCredential is used automatically
    config = LLMConfig(
        provider=Provider.AZURE_OPENAI,
        model_id="gpt-4o",  # Your deployment name
        params=ModelParams(
            temperature=0.7,
            max_output_tokens=512,
        ),
        credentials=Credentials(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            # No api_key - will use DefaultAzureCredential
        ),
    )

    messages = [
        UserMessage(content="What is the capital of France?")
    ]

    response = llm(config, messages)
    print(f"Response: {response.content}")
    print(f"\nNote: Used Microsoft Entra ID (DefaultAzureCredential) for authentication")


# ============================================================================
# Example 3: Using built-in presets
# ============================================================================
def example_presets() -> None:
    """Using built-in Azure OpenAI presets."""
    print_section("Example 3: Built-in Azure OpenAI Presets")

    # Note: Presets still require credentials to be set
    # You can set them via environment variables or override in code
    messages = [
        UserMessage(content="Write a haiku about clouds.")
    ]

    # Built-in preset for GPT-4o on Azure
    # Note: You still need to provide credentials
    from thinllm.config import get_preset
    
    config = get_preset("azure-gpt-4o")
    # Override credentials
    config.credentials = Credentials(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    )
    
    response = llm(config, messages)
    print(f"Response from azure-gpt-4o preset:\n{response.content}")

    print("\n" + "-" * 70 + "\n")

    # Built-in preset for GPT-4o-mini on Azure
    config_mini = get_preset("azure-gpt-4o-mini")
    config_mini.credentials = Credentials(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    )
    
    response = llm(config_mini, messages)
    print(f"Response from azure-gpt-4o-mini preset:\n{response.content}")


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
        provider=Provider.AZURE_OPENAI,
        model_id="gpt-4o",  # Your deployment name
        params=ModelParams(temperature=0.7, max_output_tokens=2048),
        credentials=Credentials(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        ),
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
        provider=Provider.AZURE_OPENAI,
        model_id="gpt-4o",  # Your deployment name
        params=ModelParams(temperature=0.7, max_output_tokens=1024),
        credentials=Credentials(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        ),
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
        provider=Provider.AZURE_OPENAI,
        model_id="gpt-4o",  # Your deployment name
        params=ModelParams(temperature=0.7, max_output_tokens=1024),
        credentials=Credentials(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        ),
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
        provider=Provider.AZURE_OPENAI,
        model_id="gpt-4o-mini",  # Using mini for faster responses
        params=ModelParams(temperature=0.7, max_output_tokens=512),
        credentials=Credentials(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        ),
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
# Example 8: Using environment variables for configuration
# ============================================================================
def example_env_config() -> None:
    """Using environment variables for Azure OpenAI configuration."""
    print_section("Example 8: Environment Variable Configuration")

    # This approach is cleaner for production code
    # Set these environment variables:
    # - AZURE_OPENAI_ENDPOINT
    # - AZURE_OPENAI_API_KEY (optional if using Entra ID)
    
    config = LLMConfig(
        provider=Provider.AZURE_OPENAI,
        model_id=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
        params=ModelParams(temperature=0.7, max_output_tokens=256),
        credentials=Credentials(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),  # Will use Entra ID if None
        ),
    )

    messages = [
        UserMessage(content="What are the three primary colors?")
    ]

    response = llm(config, messages)
    print(f"Response: {response.content}")
    print(f"\nNote: Configuration loaded from environment variables")
    print(f"  - Endpoint: {config.credentials.azure_endpoint if config.credentials else 'Not set'}")
    print(f"  - Auth: {'API Key' if config.credentials and config.credentials.api_key else 'Microsoft Entra ID'}")


# ============================================================================
# Main execution
# ============================================================================
def main() -> None:
    """Run all examples."""
    print("\n" + "=" * 70)
    print("  ThinLLM - Azure OpenAI Examples")
    print("=" * 70)
    print("\nThese examples demonstrate using Azure OpenAI with thinllm.")
    print("Make sure you have Azure OpenAI set up with a deployed model.")
    
    # Check for required environment variables
    if not os.getenv("AZURE_OPENAI_ENDPOINT"):
        print("\n⚠️  Warning: AZURE_OPENAI_ENDPOINT not set!")
        print("Please set it to your Azure OpenAI endpoint (e.g., https://your-resource.openai.azure.com)")
        print("\nYou can set it with:")
        print('  export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"')
        return
    
    if not os.getenv("AZURE_OPENAI_API_KEY"):
        print("\n⚠️  Note: AZURE_OPENAI_API_KEY not set.")
        print("Will attempt to use Microsoft Entra ID authentication.")
        print("Make sure you're logged in with: az login")
        print("\nOr set API key with:")
        print('  export AZURE_OPENAI_API_KEY="your-api-key"')
    
    examples = [
        ("Basic Text (API Key)", example_basic_text_api_key),
        ("Microsoft Entra ID Auth", example_entra_id_auth),
        ("Built-in Presets", example_presets),
        ("Structured Output", example_structured_output),
        ("Function Calling", example_function_calling),
        ("Streaming", example_streaming),
        ("Multi-turn Conversation", example_multi_turn),
        ("Environment Config", example_env_config),
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
        print("1. Ensure AZURE_OPENAI_ENDPOINT is set correctly")
        print("2. For API Key auth: Set AZURE_OPENAI_API_KEY")
        print("3. For Entra ID auth: Run 'az login' and ensure you have access")
        print("4. Verify your model deployment name matches model_id in the code")
        print("5. Install dependencies: uv sync --group azure-openai")


if __name__ == "__main__":
    main()
