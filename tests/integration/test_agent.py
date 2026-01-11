"""Integration tests for Agent with both OpenAI and Anthropic providers.

These tests make real API calls and are not mocked.
They require valid API keys to be set in the environment.
"""

import pytest

from thinllm.agent import Agent, AgentRequest, AgentResponse, StepStatus
from thinllm.config import LLMConfig, Provider
from thinllm.messages import OutputTextBlock, ReasoningContent, SystemMessage, ToolOutputStatus
from thinllm.tools import tool

# Provider configurations for parametrized tests
PROVIDER_CONFIGS = [
    pytest.param(
        Provider.OPENAI,
        "gpt-5.1",
        {"temperature": 0.0},
        "openai_api_key",
        marks=pytest.mark.openai,
        id="openai-gpt5.1",
    ),
    pytest.param(
        Provider.ANTHROPIC,
        "claude-sonnet-4-5",
        {"temperature": 0.0, "max_tokens": 4096},
        "anthropic_api_key",
        marks=pytest.mark.anthropic,
        id="anthropic-claude-sonnet-4-5",
    ),
    pytest.param(
        Provider.GEMINI,
        "gemini-2.5-flash",
        {"temperature": 0.0},
        "gemini_api_key",
        marks=pytest.mark.gemini,
        id="gemini-2.5-flash",
    ),
]


@pytest.mark.integration
@pytest.mark.parametrize(
    ("provider", "model_id", "model_args", "api_key_fixture"),
    PROVIDER_CONFIGS,
)
def test_agent_horoscope_full_loop(
    request: pytest.FixtureRequest,
    provider: Provider,
    model_id: str,
    model_args: dict,
    api_key_fixture: str,
) -> None:
    """
    Test agent with horoscope tool completing the full agentic loop.

    This integration test verifies:
    1. Agent can process a user request
    2. LLM identifies need for tool call
    3. Agent executes the tool
    4. LLM incorporates tool result into final answer
    5. Agent returns complete response with all steps

    The test demonstrates the complete flow:
    User request → LLM call → Tool execution → LLM call with result → Final answer
    """
    # Get API key from fixture
    api_key = request.getfixturevalue(api_key_fixture)

    # Define the get_horoscope tool
    @tool
    def get_horoscope(sign: str) -> str:
        """
        Get the horoscope for a given zodiac sign.

        Args:
            sign: The zodiac sign (e.g., 'Aquarius', 'Leo')

        Returns:
            A hardcoded horoscope string
        """
        return f"Today is a great day for {sign}! The stars are aligned in your favor."

    # Arrange
    config = LLMConfig(
        provider=provider,
        model_id=model_id,
        model_args=model_args,
    )

    messages = [
        SystemMessage(
            content="You are a helpful assistant. When you need information, use the available tools."
        )
    ]

    agent = Agent(llm_config=config, messages=messages, tools=[get_horoscope], max_iterations=10)

    # Act
    user_request = AgentRequest(content="What is my horoscope? I am an Aquarius.")
    response = agent.ask(user_request)

    # Assert
    assert isinstance(response, AgentResponse), f"Expected AgentResponse, got {type(response)}"
    assert response.type == "response", "Response should have discriminator type='response'"

    # Verify overall response status
    assert response.status == StepStatus.SUCCEEDED, (
        f"Expected status SUCCEEDED, got {response.status}"
    )

    # Verify we have at least 1 step (tool call step)
    # Note: steps are only created when there are tool calls
    assert len(response.steps) >= 1, (
        f"Expected at least 1 step (tool call), got {len(response.steps)}"
    )

    # Verify iterations
    assert response.iterations >= 2, f"Expected at least 2 iterations, got {response.iterations}"

    # Verify first step contains tool call and execution
    first_step = response.steps[0]
    assert first_step.status == StepStatus.SUCCEEDED, (
        f"First step should be SUCCEEDED, got {first_step.status}"
    )
    assert first_step.user_message is not None, (
        "First step should have user_message with tool results"
    )

    # Verify tool results
    tool_results = first_step.tool_results
    assert len(tool_results) > 0, "First step should have at least one tool result"

    horoscope_result = tool_results[0]
    assert horoscope_result.name == "get_horoscope", (
        f"Expected tool 'get_horoscope', got '{horoscope_result.name}'"
    )
    assert horoscope_result.status == ToolOutputStatus.SUCCESS, (
        f"Tool execution should succeed, got {horoscope_result.status}"
    )
    assert horoscope_result.output is not None, "Tool should have output"
    assert "Aquarius" in horoscope_result.output, (
        f"Tool output should mention 'Aquarius', got: {horoscope_result.output}"
    )
    assert "great day" in horoscope_result.output.lower(), (
        f"Tool output should contain horoscope text, got: {horoscope_result.output}"
    )

    # Verify final message
    assert response.final_message is not None, "Response should have final_message"
    assert isinstance(response.final_message.content, list), (
        "Final message content should be a list"
    )

    # Extract text from final message
    text_blocks = [
        block for block in response.final_message.content if isinstance(block, OutputTextBlock)
    ]
    assert text_blocks, "Final message should contain at least one OutputTextBlock"

    final_text = " ".join(block.text for block in text_blocks).lower()
    assert final_text, "Final message should have text content"

    # Verify final message incorporates the horoscope information
    # It should mention either the horoscope content or acknowledge the result
    assert any(
        keyword in final_text
        for keyword in ["horoscope", "aquarius", "great day", "stars", "favor"]
    ), f"Final message should reference horoscope information, got: {final_text}"


@pytest.mark.integration
@pytest.mark.parametrize(
    ("provider", "model_id", "model_args", "api_key_fixture"),
    PROVIDER_CONFIGS,
)
def test_agent_simple_qa_no_tools(
    request: pytest.FixtureRequest,
    provider: Provider,
    model_id: str,
    model_args: dict,
    api_key_fixture: str,
) -> None:
    """
    Test agent with simple Q&A without any tools.

    This verifies that the agent works as a simple LLM wrapper
    when no tools are provided.
    """
    # Get API key from fixture
    api_key = request.getfixturevalue(api_key_fixture)

    # Arrange
    config = LLMConfig(
        provider=provider,
        model_id=model_id,
        model_args=model_args,
    )

    messages = [SystemMessage(content="You are a helpful assistant.")]
    agent = Agent(llm_config=config, messages=messages, tools=None)

    # Act
    user_request = AgentRequest(content="What is the capital of France?")
    response = agent.ask(user_request)

    # Assert
    assert isinstance(response, AgentResponse)
    assert response.status == StepStatus.SUCCEEDED
    assert response.iterations == 1, "Should complete in one iteration without tools"
    # No steps created when there are no tool calls
    assert len(response.steps) == 0, "Should have no steps when no tools called"

    # Verify the response mentions Paris
    text_blocks = [
        block for block in response.final_message.content if isinstance(block, OutputTextBlock)
    ]
    assert text_blocks, "Response should contain text"
    final_text = " ".join(block.text for block in text_blocks).lower()
    assert "paris" in final_text, f"Response should mention Paris, got: {final_text}"


@pytest.mark.integration
@pytest.mark.parametrize(
    ("provider", "model_id", "model_args", "api_key_fixture"),
    PROVIDER_CONFIGS,
)
def test_agent_max_iterations_limit(
    request: pytest.FixtureRequest,
    provider: Provider,
    model_id: str,
    model_args: dict,
    api_key_fixture: str,
) -> None:
    """
    Test that agent respects max_iterations limit.

    This test uses a tool that returns a response suggesting more tool calls,
    potentially creating a loop. The agent should stop at max_iterations.
    """
    # Get API key from fixture
    api_key = request.getfixturevalue(api_key_fixture)

    @tool
    def get_info(query: str) -> str:
        """Get information about a query."""
        return f"Information about {query}. You might want to search for more details."

    # Arrange
    config = LLMConfig(
        provider=provider,
        model_id=model_id,
        model_args=model_args,
    )

    messages = [
        SystemMessage(
            content="You are a helpful assistant. Use the get_info tool to answer questions. "
            "After getting info, provide a final answer without calling more tools."
        )
    ]

    # Set a low max_iterations to test the limit
    agent = Agent(llm_config=config, messages=messages, tools=[get_info], max_iterations=3)

    # Act
    user_request = AgentRequest(content="Tell me about Python programming.")
    response = agent.ask(user_request)

    # Assert
    assert isinstance(response, AgentResponse)
    # The agent should either succeed or fail due to max iterations
    assert response.status in [StepStatus.SUCCEEDED, StepStatus.FAILED]
    # Should not exceed max iterations
    assert response.iterations <= 3, (
        f"Agent should not exceed max_iterations=3, got {response.iterations}"
    )
    # Should have steps
    assert len(response.steps) > 0, "Agent should have at least one step"


@pytest.mark.integration
@pytest.mark.parametrize(
    ("provider", "model_id", "model_args", "api_key_fixture"),
    PROVIDER_CONFIGS,
)
def test_agent_multiple_tool_calls_in_sequence(
    request: pytest.FixtureRequest,
    provider: Provider,
    model_id: str,
    model_args: dict,
    api_key_fixture: str,
) -> None:
    """
    Test agent handling multiple tool calls in sequence.

    This verifies that the agent can handle a scenario where the LLM
    requests multiple tool calls in a single response.
    """
    # Get API key from fixture
    api_key = request.getfixturevalue(api_key_fixture)

    @tool
    def get_weather(location: str) -> str:
        """Get weather for a location."""
        return f"Weather in {location}: Sunny, 72°F"

    @tool
    def get_time(timezone: str) -> str:
        """Get current time in a timezone."""
        return f"Current time in {timezone}: 3:45 PM"

    # Arrange
    config = LLMConfig(
        provider=provider,
        model_id=model_id,
        model_args=model_args,
    )

    messages = [
        SystemMessage(
            content="You are a helpful assistant. Use available tools to answer questions. "
            "After getting all needed information, provide a final answer."
        )
    ]

    agent = Agent(
        llm_config=config, messages=messages, tools=[get_weather, get_time], max_iterations=10
    )

    # Act
    user_request = AgentRequest(content="What's the weather and time in New York?")
    response = agent.ask(user_request)

    # Assert
    assert isinstance(response, AgentResponse)
    assert response.status == StepStatus.SUCCEEDED

    # Check that tools were called
    all_tool_results = []
    for step in response.steps:
        all_tool_results.extend(step.tool_results)

    # Should have called at least one tool (might be 1 or 2 depending on LLM behavior)
    assert len(all_tool_results) > 0, "Agent should have called at least one tool"

    # Verify final message exists and has content
    assert response.final_message is not None
    text_blocks = [
        block for block in response.final_message.content if isinstance(block, OutputTextBlock)
    ]
    assert text_blocks, "Final message should contain text"


@pytest.mark.integration
@pytest.mark.parametrize(
    ("provider", "model_id", "model_args", "api_key_fixture"),
    PROVIDER_CONFIGS,
)
def test_agent_request_discriminator(
    request: pytest.FixtureRequest,
    provider: Provider,
    model_id: str,
    model_args: dict,
    api_key_fixture: str,
) -> None:
    """Test that AgentRequest has correct discriminator type."""
    user_request = AgentRequest(content="Hello")
    assert user_request.type == "request"


@pytest.mark.integration
@pytest.mark.parametrize(
    ("provider", "model_id", "model_args", "api_key_fixture"),
    PROVIDER_CONFIGS,
)
def test_agent_response_discriminator(
    request: pytest.FixtureRequest,
    provider: Provider,
    model_id: str,
    model_args: dict,
    api_key_fixture: str,
) -> None:
    """Test that AgentResponse has correct discriminator type."""
    # Get API key from fixture
    api_key = request.getfixturevalue(api_key_fixture)

    config = LLMConfig(
        provider=provider,
        model_id=model_id,
        model_args=model_args,
    )

    messages = [SystemMessage(content="You are a helpful assistant.")]
    agent = Agent(llm_config=config, messages=messages)

    user_request = AgentRequest(content="Say hello")
    response = agent.ask(user_request)

    assert response.type == "response"


# Provider-specific tests for features that differ between providers

@pytest.mark.integration
@pytest.mark.openai
def test_agent_with_openai_reasoning(openai_api_key: str) -> None:
    """
    Test agent with OpenAI's explicit reasoning configuration (gpt-5 specific).

    This integration test verifies:
    1. Agent can use GPT-5 with explicit reasoning parameters
    2. Agent produces reasoning content blocks
    3. Agent can combine reasoning with tool execution
    4. Agent successfully completes multi-step reasoning tasks
    """

    @tool
    def get_data(country: str, data_type: str) -> str:
        """
        Get population or GDP data for a country.

        Args:
            country: Name of the country
            data_type: Type of data ('population' or 'gdp')

        Returns:
            Data as a string
        """
        data = {
            "usa": {"population": "331 million", "gdp": "$23 trillion"},
            "india": {"population": "1.38 billion", "gdp": "$3.5 trillion"},
        }
        country_lower = country.lower()
        if country_lower in data and data_type in data[country_lower]:
            return f"{data_type.title()} of {country}: {data[country_lower][data_type]}"
        return f"Data not available for {country}"

    # Configure LLM with explicit reasoning parameters (OpenAI-specific)
    config = LLMConfig(
        provider=Provider.OPENAI,
        model_id="gpt-5",
        model_args={
            "reasoning": {
                "effort": "medium",
                "summary": "concise",
            }
        },
    )

    messages = [
        SystemMessage(
            content="You are a helpful assistant. Use tools to gather data, "
            "then reason through the problem step by step."
        )
    ]

    agent = Agent(llm_config=config, messages=messages, tools=[get_data], max_iterations=10)

    # Act
    user_request = AgentRequest(
        content="Compare USA and India GDP per capita. Show your reasoning."
    )
    response = agent.ask(user_request)

    # Assert
    assert isinstance(response, AgentResponse), f"Expected AgentResponse, got {type(response)}"
    assert response.status == StepStatus.SUCCEEDED, f"Expected SUCCEEDED, got {response.status}"
    assert len(response.steps) >= 1, f"Expected at least 1 step, got {len(response.steps)}"

    # Verify tools were called successfully
    all_tool_results = [tr for step in response.steps for tr in step.tool_results]
    min_tool_calls = 2
    assert len(all_tool_results) >= min_tool_calls, (
        f"Expected at least {min_tool_calls} tool calls, got {len(all_tool_results)}"
    )

    for tool_result in all_tool_results:
        assert tool_result.status == ToolOutputStatus.SUCCESS

    # Check for reasoning content blocks in steps or final message
    def has_reasoning_blocks(content_list):
        return any(isinstance(block, ReasoningContent) for block in content_list)

    has_reasoning = has_reasoning_blocks(response.final_message.content)
    if not has_reasoning:
        has_reasoning = any(
            has_reasoning_blocks(step.ai_message.content)
            for step in response.steps
            if isinstance(step.ai_message.content, list)
        )

    assert has_reasoning, "Expected reasoning content blocks from GPT-5 with reasoning config"

    # Verify final message addresses the comparison
    text_blocks = [
        block for block in response.final_message.content if isinstance(block, OutputTextBlock)
    ]
    assert text_blocks, "Final message should contain OutputTextBlock"

    final_text = " ".join(block.text for block in text_blocks).lower()
    keywords = ["capita", "higher", "usa", "india", "gdp"]
    assert any(kw in final_text for kw in keywords), (
        f"Final message should address comparison, got: {final_text}"
    )


@pytest.mark.integration
@pytest.mark.anthropic
def test_agent_with_anthropic_thinking(anthropic_api_key: str) -> None:
    """
    Test agent with Anthropic's thinking capabilities.

    This verifies that the agent works with Anthropic's extended thinking feature.
    """

    @tool
    def calculate_compound_interest(
        principal: float, rate: float, years: int, compounds_per_year: int = 12
    ) -> str:
        """
        Calculate compound interest.

        Args:
            principal: Initial principal amount
            rate: Annual interest rate (as decimal, e.g., 0.05 for 5%)
            years: Number of years
            compounds_per_year: Number of times interest compounds per year (default: 12)

        Returns:
            Formatted string with final amount and total interest earned
        """
        amount = principal * (1 + rate / compounds_per_year) ** (compounds_per_year * years)
        interest = amount - principal
        return f"Final amount: ${amount:.2f}, Total interest earned: ${interest:.2f}"

    # Configure Anthropic with thinking enabled
    # Note: Anthropic requires temperature=1 when thinking is enabled
    config = LLMConfig(
        provider=Provider.ANTHROPIC,
        model_id="claude-sonnet-4-5",
        model_args={
            "temperature": 1.0,  # Required for thinking mode
            "max_tokens": 4096,
            "thinking": {
                "type": "enabled",
                "budget_tokens": 2000,
            },
        },
    )

    messages = [
        SystemMessage(
            content="You are a helpful financial assistant. Think through the problem step by step "
            "and use the calculate_compound_interest tool when needed for calculations."
        )
    ]

    agent = Agent(
        llm_config=config, messages=messages, tools=[calculate_compound_interest], max_iterations=10
    )

    # Act
    user_request = AgentRequest(
        content="If I invest $10,000 at 5% annual interest compounded monthly for 10 years, "
        "how much will I have? Please explain your reasoning."
    )
    response = agent.ask(user_request)

    # Assert
    assert isinstance(response, AgentResponse), f"Expected AgentResponse, got {type(response)}"
    assert response.status == StepStatus.SUCCEEDED, f"Expected SUCCEEDED, got {response.status}"

    # Verify we have at least one step with tool call
    assert len(response.steps) >= 1, (
        f"Expected at least 1 step with tool call, got {len(response.steps)}"
    )

    # Verify tool was called
    tool_results = []
    for step in response.steps:
        tool_results.extend(step.tool_results)

    assert len(tool_results) > 0, "Expected at least one tool call"

    # Verify compound interest calculation was called
    compound_interest_calls = [
        tr for tr in tool_results if tr.name == "calculate_compound_interest"
    ]
    assert len(compound_interest_calls) > 0, "Expected calculate_compound_interest to be called"

    # Verify tool execution was successful
    tool_result = compound_interest_calls[0]
    assert tool_result.status == ToolOutputStatus.SUCCESS, (
        f"Tool execution failed: {tool_result.output}"
    )
    assert tool_result.output is not None, "Tool should have output"
    assert "$" in tool_result.output, (
        f"Expected monetary values in output, got: {tool_result.output}"
    )

    # Verify final message has useful content
    assert response.final_message is not None, "Response should have final_message"

    text_blocks = [
        block for block in response.final_message.content if isinstance(block, OutputTextBlock)
    ]
    assert text_blocks, "Final message should contain at least one OutputTextBlock"

    final_text = " ".join(block.text for block in text_blocks).lower()
    assert final_text, "Final message should have text content"

    # Verify final message references the calculation result
    assert any(
        keyword in final_text
        for keyword in ["amount", "interest", "10,000", "10000", "compound", "$"]
    ), f"Final message should reference the calculation, got: {final_text}"

