"""Unit tests for Anthropic serializer functions."""

import pytest

# Mark all tests in this module as Anthropic-specific
pytestmark = [pytest.mark.unit, pytest.mark.anthropic]

from thinllm.messages import (
    AIMessage,
    InputTextBlock,
    OutputTextBlock,
    ReasoningContent,
    SystemMessage,
    ToolCallContent,
    ToolCallStatus,
    ToolResultContent,
    UserMessage,
)
from thinllm.providers.anthropic.serializers import (
    _convert_content_block_to_anthropic_dict,
    _get_anthropic_messages,
    _get_anthropic_tool,
    _get_system_from_messages,
)
from thinllm.tools import Tool


class TestConvertContentBlockToAnthropicDict:
    """Test the _convert_content_block_to_anthropic_dict function."""

    def test_convert_input_text_block(self) -> None:
        """Test converting InputTextBlock to dict."""
        block = InputTextBlock(text="Hello!")
        result = _convert_content_block_to_anthropic_dict(block)

        assert isinstance(result, dict)
        assert result["type"] == "text"
        assert result["text"] == "Hello!"

    def test_convert_output_text_block(self) -> None:
        """Test converting OutputTextBlock to dict."""
        block = OutputTextBlock(text="Response text")
        result = _convert_content_block_to_anthropic_dict(block)

        assert isinstance(result, dict)
        assert result["type"] == "text"
        assert result["text"] == "Response text"

    def test_convert_reasoning_content_without_redaction(self) -> None:
        """Test converting ReasoningContent without redaction to thinking block."""
        block = ReasoningContent(
            signature="sig123",
            summaries=["Summary 1", "Summary 2"],
            contents=["Content 1", "Content 2"],
        )
        result = _convert_content_block_to_anthropic_dict(block)

        assert isinstance(result, dict)
        assert result["type"] == "thinking"
        assert result["signature"] == "sig123"
        assert "Summary 1" in result["thinking"]
        assert "Summary 2" in result["thinking"]
        assert "Content 1" in result["thinking"]
        assert "Content 2" in result["thinking"]

    def test_convert_reasoning_content_with_redaction(self) -> None:
        """Test converting ReasoningContent with redaction."""
        block = ReasoningContent(
            redacted_content="redacted_data_here",
            summaries=[],
            contents=[],
        )
        result = _convert_content_block_to_anthropic_dict(block)

        assert isinstance(result, dict)
        assert result["type"] == "redacted_thinking"
        assert result["data"] == "redacted_data_here"

    def test_convert_tool_call_content(self) -> None:
        """Test converting ToolCallContent to tool_use."""
        block = ToolCallContent(
            tool_id="call_123",
            name="test_tool",
            input={"param": "value"},
            raw_input='{"param": "value"}',
            status=ToolCallStatus.COMPLETE,
        )
        result = _convert_content_block_to_anthropic_dict(block)

        assert isinstance(result, dict)
        assert result["type"] == "tool_use"
        assert result["id"] == "call_123"
        assert result["name"] == "test_tool"
        assert result["input"] == {"param": "value"}

    def test_convert_tool_result_content_success(self) -> None:
        """Test converting ToolResultContent with success status."""
        block = ToolResultContent(
            tool_id="call_456",
            name="test_tool",
            output="Tool output here",
        )
        result = _convert_content_block_to_anthropic_dict(block)

        assert isinstance(result, dict)
        assert result["type"] == "tool_result"
        assert result["tool_use_id"] == "call_456"
        assert result["content"] == "Tool output here"
        assert result["is_error"] is False

    def test_convert_tool_result_content_failure(self) -> None:
        """Test converting ToolResultContent with failure status."""
        from thinllm.messages import ToolOutputStatus

        block = ToolResultContent(
            tool_id="call_789",
            name="test_tool",
            output="Error occurred",
            status=ToolOutputStatus.FAILURE,
        )
        result = _convert_content_block_to_anthropic_dict(block)

        assert isinstance(result, dict)
        assert result["type"] == "tool_result"
        assert result["tool_use_id"] == "call_789"
        assert result["content"] == "Error occurred"
        assert result["is_error"] is True


class TestGetAnthropicMessages:
    """Test the _get_anthropic_messages function."""

    def test_simple_string_user_message(self) -> None:
        """Test converting a simple user message with string content."""
        messages = [UserMessage(content="Hello!")]
        result = _get_anthropic_messages(messages)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert isinstance(result[0]["content"], list)
        assert len(result[0]["content"]) == 1
        assert result[0]["content"][0]["type"] == "text"
        assert result[0]["content"][0]["text"] == "Hello!"

    def test_user_message_with_text_block_in_list(self) -> None:
        """Test converting user message with TextBlock in list."""
        messages = [UserMessage(content=[InputTextBlock(text="Hello from TextBlock!")])]
        result = _get_anthropic_messages(messages)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert isinstance(result[0]["content"], list)
        assert len(result[0]["content"]) == 1
        assert result[0]["content"][0]["type"] == "text"
        assert result[0]["content"][0]["text"] == "Hello from TextBlock!"

    def test_system_message_excluded(self) -> None:
        """Test that system messages are excluded from Anthropic messages."""
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            UserMessage(content="Hello!"),
        ]
        result = _get_anthropic_messages(messages)

        # System message should be excluded (handled separately)
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_ai_message_with_string_content(self) -> None:
        """Test converting AI message with string content."""
        messages = [AIMessage(content="Hello, how can I help?")]
        result = _get_anthropic_messages(messages)

        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert isinstance(result[0]["content"], list)
        assert len(result[0]["content"]) == 1
        assert result[0]["content"][0]["type"] == "text"
        assert result[0]["content"][0]["text"] == "Hello, how can I help?"

    def test_ai_message_with_text_block_in_list(self) -> None:
        """Test converting AI message with OutputTextBlock in list."""
        messages = [AIMessage(content=[OutputTextBlock(text="AI response here")])]
        result = _get_anthropic_messages(messages)

        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert isinstance(result[0]["content"], list)
        assert len(result[0]["content"]) == 1
        assert result[0]["content"][0]["type"] == "text"
        assert result[0]["content"][0]["text"] == "AI response here"

    def test_user_message_with_tool_results(self) -> None:
        """Test user message with tool results."""
        messages = [
            UserMessage(
                content=[
                    InputTextBlock(text="Run the tool"),
                    ToolResultContent(tool_id="call_123", name="test_tool", output="Tool output"),
                ]
            )
        ]
        result = _get_anthropic_messages(messages)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert len(result[0]["content"]) == 2
        assert result[0]["content"][0]["type"] == "text"
        assert result[0]["content"][0]["text"] == "Run the tool"
        assert result[0]["content"][1]["type"] == "tool_result"
        assert result[0]["content"][1]["tool_use_id"] == "call_123"
        assert result[0]["content"][1]["content"] == "Tool output"

    def test_ai_message_with_tool_calls(self) -> None:
        """Test AI message with tool call requests."""
        messages = [
            AIMessage(
                content=[
                    OutputTextBlock(text="I'll call the tool"),
                    ToolCallContent(
                        tool_id="call_456",
                        name="test_tool",
                        input={"param": "value"},
                        raw_input='{"param": "value"}',
                    ),
                ]
            )
        ]
        result = _get_anthropic_messages(messages)

        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert len(result[0]["content"]) == 2
        assert result[0]["content"][0]["type"] == "text"
        assert result[0]["content"][0]["text"] == "I'll call the tool"
        assert result[0]["content"][1]["type"] == "tool_use"
        assert result[0]["content"][1]["id"] == "call_456"
        assert result[0]["content"][1]["name"] == "test_tool"
        assert result[0]["content"][1]["input"] == {"param": "value"}

    def test_mixed_message_types(self) -> None:
        """Test converting a mix of message types."""
        messages = [
            SystemMessage(content="System instruction"),
            UserMessage(content="User question"),
            AIMessage(content="AI response"),
            UserMessage(content=[InputTextBlock(text="Follow-up")]),
        ]
        result = _get_anthropic_messages(messages)

        # System message excluded, so 3 messages expected
        assert len(result) == 3
        assert result[0]["role"] == "user"
        assert result[0]["content"][0]["text"] == "User question"
        assert result[1]["role"] == "assistant"
        assert result[1]["content"][0]["text"] == "AI response"
        assert result[2]["role"] == "user"
        assert result[2]["content"][0]["text"] == "Follow-up"


class TestGetSystemFromMessages:
    """Test the _get_system_from_messages function."""

    def test_single_system_message_string(self) -> None:
        """Test extracting system from single system message with string."""
        messages = [SystemMessage(content="You are a helpful assistant.")]
        result = _get_system_from_messages(messages)
        assert result == "You are a helpful assistant."

    def test_single_system_message_text_block_in_list(self) -> None:
        """Test extracting system from system message with TextBlock in list."""
        messages = [SystemMessage(content=[InputTextBlock(text="You are helpful.")])]
        result = _get_system_from_messages(messages)
        assert result == "You are helpful."

    def test_multiple_system_messages(self) -> None:
        """Test extracting and combining multiple system messages."""
        messages = [
            SystemMessage(content="First instruction."),
            SystemMessage(content=[InputTextBlock(text="Second instruction.")]),
            UserMessage(content="User message"),
        ]
        result = _get_system_from_messages(messages)
        assert result == "First instruction.\nSecond instruction."

    def test_system_message_with_list_of_text_blocks(self) -> None:
        """Test extracting system from system message with list of TextBlocks."""
        messages = [
            SystemMessage(
                content=[InputTextBlock(text="First part."), InputTextBlock(text="Second part.")]
            )
        ]
        result = _get_system_from_messages(messages)
        assert result == "First part.\nSecond part."

    def test_no_system_messages(self) -> None:
        """Test when there are no system messages."""
        messages = [UserMessage(content="Hello"), AIMessage(content="Hi there")]
        result = _get_system_from_messages(messages)
        assert result == ""

    def test_empty_messages_list(self) -> None:
        """Test with empty messages list."""
        messages = []
        result = _get_system_from_messages(messages)
        assert result == ""


class TestGetAnthropicTool:
    """Test the _get_anthropic_tool function."""

    def test_convert_dict_tool(self) -> None:
        """Test converting dict tool returns dict as-is."""
        tool = {"type": "custom_tool", "config": {"key": "value"}}
        result = _get_anthropic_tool(tool)

        assert result == {"type": "custom_tool", "config": {"key": "value"}}

    def test_convert_callable_to_tool(self) -> None:
        """Test converting a callable function to tool format."""

        def sample_function(param1: str, param2: int = 5) -> str:
            """A sample function for testing.

            Args:
                param1: First parameter
                param2: Second parameter with default
            """
            return f"{param1}-{param2}"

        result = _get_anthropic_tool(sample_function)

        assert isinstance(result, dict)
        assert result["name"] == "sample_function"
        assert "description" in result
        assert "input_schema" in result
        assert isinstance(result["input_schema"], dict)

    def test_convert_tool_instance(self) -> None:
        """Test converting a Tool instance to Anthropic format."""

        def test_func(x: int, y: int) -> int:
            """Add two numbers."""
            return x + y

        tool = Tool.from_function(test_func)
        result = _get_anthropic_tool(tool)

        assert isinstance(result, dict)
        assert result["name"] == "test_func"
        assert "Add two numbers" in result["description"]
        assert "input_schema" in result
        assert isinstance(result["input_schema"], dict)
