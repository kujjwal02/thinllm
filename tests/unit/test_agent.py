"""Unit tests for Agent class."""

import pytest

from thinllm.agent import Agent, AgentRequest, AgentResponse, StepStatus
from thinllm.config import LLMConfig, Provider
from thinllm.messages import (
    AIMessage,
    OutputTextBlock,
    SystemMessage,
    ToolCallContent,
    ToolOutput,
    ToolOutputStatus,
    ToolResultContent,
    UserMessage,
)
from thinllm.tools import tool


@pytest.fixture(params=[
    pytest.param((Provider.OPENAI, "gpt-4", {"temperature": 0.0}), marks=pytest.mark.openai, id="openai"),
    pytest.param((Provider.ANTHROPIC, "claude-sonnet-4-5", {"temperature": 0.0, "max_tokens": 4096}), marks=pytest.mark.anthropic, id="anthropic"),
    pytest.param((Provider.GEMINI, "gemini-2.5-flash", {"temperature": 0.0}), marks=pytest.mark.gemini, id="gemini"),
])
def mock_llm_config(request):
    """Create a mock LLM config for testing with different providers."""
    provider, model_id, model_args = request.param
    return LLMConfig(provider=provider, model_id=model_id, model_args=model_args)


@pytest.fixture
def simple_tool():
    """Create a simple test tool."""

    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    return add


@pytest.fixture
def failing_tool():
    """Create a tool that always fails."""

    @tool
    def fail_always(message: str) -> str:
        """A tool that always raises an exception."""
        raise ValueError(f"Tool failed: {message}")

    return fail_always


def test_agent_initialization(mock_llm_config, simple_tool):
    """Test that agent initializes correctly with tools."""
    messages = [SystemMessage(content="You are a helpful assistant.")]
    agent = Agent(llm_config=mock_llm_config, messages=messages, tools=[simple_tool])

    assert agent.llm_config == mock_llm_config
    assert len(agent.messages) == 1
    assert agent.max_iterations == 10
    assert "add" in agent._tool_map


def test_agent_initialization_with_callable(mock_llm_config):
    """Test that agent converts callable to Tool."""

    def multiply(x: int, y: int) -> int:
        """Multiply two numbers."""
        return x * y

    messages = [SystemMessage(content="You are a helpful assistant.")]
    agent = Agent(llm_config=mock_llm_config, messages=messages, tools=[multiply])

    assert "multiply" in agent._tool_map
    assert callable(agent._tool_map["multiply"])


def test_agent_initialization_with_dict_tool(mock_llm_config):
    """Test that agent skips dict-based tools like web_search."""
    messages = [SystemMessage(content="You are a helpful assistant.")]
    web_search_tool = {"type": "web_search"}
    agent = Agent(llm_config=mock_llm_config, messages=messages, tools=[web_search_tool])

    # Dict tools should not be added to _tool_map
    assert "web_search" not in agent._tool_map
    # But should still be in raw tools for passing to LLM
    assert web_search_tool in agent._raw_tools


def test_agent_initialization_custom_max_iterations(mock_llm_config):
    """Test that agent respects custom max_iterations."""
    messages = [SystemMessage(content="You are a helpful assistant.")]
    agent = Agent(llm_config=mock_llm_config, messages=messages, max_iterations=5)

    assert agent.max_iterations == 5


def test_execute_tool_success(mock_llm_config, simple_tool):
    """Test successful tool execution."""
    messages = [SystemMessage(content="You are a helpful assistant.")]
    agent = Agent(llm_config=mock_llm_config, messages=messages, tools=[simple_tool])

    tool_call = ToolCallContent(
        tool_id="call_123", name="add", input={"a": 2, "b": 3}, raw_input='{"a": 2, "b": 3}'
    )

    result = agent._execute_tool(tool_call)

    assert isinstance(result, ToolResultContent)
    assert result.tool_id == "call_123"
    assert result.name == "add"
    assert result.output == "5"
    assert result.status == ToolOutputStatus.SUCCESS


def test_execute_tool_with_tool_output(mock_llm_config):
    """Test tool execution that returns ToolOutput."""

    @tool
    def get_data() -> ToolOutput:
        """Get some data."""
        return ToolOutput(
            text="Data retrieved", metadata={"source": "database"}, status=ToolOutputStatus.SUCCESS
        )

    messages = [SystemMessage(content="You are a helpful assistant.")]
    agent = Agent(llm_config=mock_llm_config, messages=messages, tools=[get_data])

    tool_call = ToolCallContent(tool_id="call_456", name="get_data", input={}, raw_input="{}")

    result = agent._execute_tool(tool_call)

    assert result.output == "Data retrieved"
    assert result.metadata == {"source": "database"}
    assert result.status == ToolOutputStatus.SUCCESS


def test_execute_tool_returns_none(mock_llm_config):
    """Test tool execution when tool returns None."""

    @tool
    def do_nothing() -> None:
        """Do nothing."""
        return None

    messages = [SystemMessage(content="You are a helpful assistant.")]
    agent = Agent(llm_config=mock_llm_config, messages=messages, tools=[do_nothing])

    tool_call = ToolCallContent(tool_id="call_789", name="do_nothing", input={}, raw_input="{}")

    result = agent._execute_tool(tool_call)

    assert result.output == ""
    assert result.status == ToolOutputStatus.SUCCESS


def test_execute_tool_returns_dict(mock_llm_config):
    """Test tool execution when tool returns a dict."""

    @tool
    def get_dict() -> dict:
        """Return a dictionary."""
        return {"key": "value", "number": 42}

    messages = [SystemMessage(content="You are a helpful assistant.")]
    agent = Agent(llm_config=mock_llm_config, messages=messages, tools=[get_dict])

    tool_call = ToolCallContent(tool_id="call_abc", name="get_dict", input={}, raw_input="{}")

    result = agent._execute_tool(tool_call)

    # Dict should be converted to string
    assert "key" in result.output
    assert "value" in result.output
    assert result.status == ToolOutputStatus.SUCCESS


def test_execute_tool_not_found(mock_llm_config, simple_tool):
    """Test tool execution when tool is not found."""
    messages = [SystemMessage(content="You are a helpful assistant.")]
    agent = Agent(llm_config=mock_llm_config, messages=messages, tools=[simple_tool])

    tool_call = ToolCallContent(tool_id="call_404", name="nonexistent_tool", input={}, raw_input="{}")

    result = agent._execute_tool(tool_call)

    assert result.status == ToolOutputStatus.FAILURE
    assert "not found" in result.output.lower()
    assert "nonexistent_tool" in result.output


def test_execute_tool_failure(mock_llm_config, failing_tool):
    """Test tool execution when tool raises an exception."""
    messages = [SystemMessage(content="You are a helpful assistant.")]
    agent = Agent(llm_config=mock_llm_config, messages=messages, tools=[failing_tool])

    tool_call = ToolCallContent(
        tool_id="call_fail",
        name="fail_always",
        input={"message": "test error"},
        raw_input='{"message": "test error"}',
    )

    result = agent._execute_tool(tool_call)

    assert result.status == ToolOutputStatus.FAILURE
    assert "error" in result.output.lower()
    assert "test error" in result.output
    assert "ValueError" in result.metadata.get("error_type", "")


def test_execute_tool_dict_based_tool_error(mock_llm_config):
    """Test that dict-based tools are not found in _tool_map."""
    messages = [SystemMessage(content="You are a helpful assistant.")]
    web_search_tool = {"type": "web_search"}
    agent = Agent(llm_config=mock_llm_config, messages=messages, tools=[web_search_tool])

    tool_call = ToolCallContent(
        tool_id="call_search", name="web_search", input={"query": "test"}, raw_input='{"query": "test"}'
    )

    result = agent._execute_tool(tool_call)

    assert result.status == ToolOutputStatus.FAILURE
    assert "not found" in result.output.lower()


def test_agent_tool_result_computed_field(mock_llm_config):
    """Test that AgentStep.tool_results computed field works correctly."""
    from thinllm.agent import AgentStep

    # Create an AI message with tool calls
    ai_message = AIMessage(
        content=[
            OutputTextBlock(text="Let me help you."),
            ToolCallContent(tool_id="call_1", name="tool_a", input={"x": 1}, raw_input='{"x": 1}'),
            ToolCallContent(tool_id="call_2", name="tool_b", input={"y": 2}, raw_input='{"y": 2}'),
        ]
    )

    # Create a user message with tool results
    user_message = UserMessage(
        content=[
            ToolResultContent(
                tool_id="call_1",
                name="tool_a",
                input={"x": 1},
                output="Result A",
                status=ToolOutputStatus.SUCCESS,
            ),
            ToolResultContent(
                tool_id="call_2",
                name="tool_b",
                input={"y": 2},
                output="Result B",
                status=ToolOutputStatus.SUCCESS,
            ),
        ]
    )

    step = AgentStep(ai_message=ai_message, user_message=user_message, status=StepStatus.SUCCEEDED)

    # Check computed field
    tool_results = step.tool_results
    assert len(tool_results) == 2

    assert tool_results[0].id == "call_1"
    assert tool_results[0].name == "tool_a"
    assert tool_results[0].output == "Result A"
    assert tool_results[0].status == ToolOutputStatus.SUCCESS

    assert tool_results[1].id == "call_2"
    assert tool_results[1].name == "tool_b"
    assert tool_results[1].output == "Result B"
    assert tool_results[1].status == ToolOutputStatus.SUCCESS


def test_agent_tool_result_pending_status(mock_llm_config):
    """Test that AgentStep.tool_results shows PENDING when no user_message."""
    from thinllm.agent import AgentStep

    # Create an AI message with tool calls but no user message yet
    ai_message = AIMessage(
        content=[ToolCallContent(tool_id="call_1", name="tool_a", input={"x": 1}, raw_input='{"x": 1}')]
    )

    step = AgentStep(ai_message=ai_message, user_message=None, status=StepStatus.IN_PROGRESS)

    tool_results = step.tool_results
    assert len(tool_results) == 1
    assert tool_results[0].output is None
    assert tool_results[0].status == ToolOutputStatus.PENDING


def test_agent_request_discriminator():
    """Test that AgentRequest has correct discriminator."""
    request = AgentRequest(content="Hello")
    assert request.type == "request"


def test_agent_response_discriminator():
    """Test that AgentResponse has correct discriminator."""
    response = AgentResponse(
        final_message=AIMessage(content="Done"),
        steps=[],
        status=StepStatus.SUCCEEDED,
        iterations=1,
    )
    assert response.type == "response"
