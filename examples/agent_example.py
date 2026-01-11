"""Example demonstrating the Agent class with tool execution.

This example shows how to create an agent that can use tools to answer questions.
"""

from thinllm import Agent, AgentRequest, LLMConfig, Provider, SystemMessage, tool


# Define some tools for the agent
@tool
def get_weather(location: str) -> str:
    """
    Get the current weather for a location.

    Args:
        location: The city or location to get weather for

    Returns:
        Weather information as a string
    """
    # In a real application, this would call a weather API
    return f"Weather in {location}: Sunny, 72°F with light breeze"


@tool
def calculate(expression: str) -> str:
    """
    Calculate a mathematical expression.

    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 2")

    Returns:
        The result of the calculation
    """
    try:
        # WARNING: eval is dangerous in production! This is just for demonstration.
        # In production, use a safe math parser like sympy or ast.literal_eval
        result = eval(expression)
        return f"The result is: {result}"
    except Exception as e:
        return f"Error calculating expression: {str(e)}"


def main():
    """Run the agent example."""
    # Configure the LLM
    config = LLMConfig(
        provider=Provider.OPENAI,
        model_id="gpt-5.1",
        model_args={"temperature": 0.0},
    )

    # Create the agent with system message and tools
    agent = Agent(
        llm_config=config,
        messages=[
            SystemMessage(
                content="You are a helpful assistant. Use the available tools to answer user questions. "
                "Always explain what you're doing before calling a tool."
            )
        ],
        tools=[get_weather, calculate],
        max_iterations=10,
    )

    # Example 1: Weather query
    print("=" * 80)
    print("Example 1: Weather Query")
    print("=" * 80)

    request1 = AgentRequest(content="What's the weather like in San Francisco?")
    response1 = agent.ask(request1)

    print(f"\nStatus: {response1.status}")
    print(f"Iterations: {response1.iterations}")
    print(f"Steps: {len(response1.steps)}")

    # Show tool calls
    for i, step in enumerate(response1.steps, 1):
        print(f"\n--- Step {i} ---")
        if step.tool_results:
            for tool_result in step.tool_results:
                print(f"Tool: {tool_result.name}")
                print(f"Input: {tool_result.input}")
                print(f"Output: {tool_result.output}")
                print(f"Status: {tool_result.status}")

    # Show final answer
    print("\n--- Final Answer ---")
    if isinstance(response1.final_message.content, list):
        for block in response1.final_message.content:
            if hasattr(block, "text"):
                print(block.text)

    # Example 2: Math calculation
    print("\n" + "=" * 80)
    print("Example 2: Math Calculation")
    print("=" * 80)

    # Create a new agent for fresh conversation
    agent2 = Agent(
        llm_config=config,
        messages=[
            SystemMessage(
                content="You are a helpful assistant. Use the available tools to answer user questions."
            )
        ],
        tools=[calculate],
        max_iterations=10,
    )

    request2 = AgentRequest(content="What is 15 multiplied by 37?")
    response2 = agent2.ask(request2)

    print(f"\nStatus: {response2.status}")
    print(f"Iterations: {response2.iterations}")

    # Show tool calls
    for i, step in enumerate(response2.steps, 1):
        print(f"\n--- Step {i} ---")
        if step.tool_results:
            for tool_result in step.tool_results:
                print(f"Tool: {tool_result.name}")
                print(f"Output: {tool_result.output}")

    # Show final answer
    print("\n--- Final Answer ---")
    if isinstance(response2.final_message.content, list):
        for block in response2.final_message.content:
            if hasattr(block, "text"):
                print(block.text)


if __name__ == "__main__":
    main()

