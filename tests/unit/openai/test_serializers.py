"""Unit tests for OpenAI serializer functions."""

import base64

import pytest

# Mark all tests in this module as OpenAI-specific
pytestmark = [pytest.mark.unit, pytest.mark.openai]

from thinllm.messages import (
    AIMessage,
    ImageDetail,
    InputImageBlock,
    InputTextBlock,
    OutputTextBlock,
    ReasoningContent,
    SystemMessage,
    ToolCallContent,
    ToolResultContent,
    UserMessage,
    WebSearchCallContent,
)
from thinllm.providers.openai.serializers import (
    _convert_content_block_to_oai_dict,
    _convert_content_to_oai_format,
    _get_instruction_from_messages,
    _get_oai_messages,
    _get_oai_tool,
    _image_bytes_to_base64_url,
)

# Constants for test expectations (used in multiple places)
EXPECTED_SINGLE_MESSAGE_COUNT = 1
EXPECTED_MESSAGE_WITH_TOOL_RESULT_COUNT = 2
EXPECTED_MESSAGE_WITH_TOOL_CALL_COUNT = 2
EXPECTED_MIXED_MESSAGES_COUNT = 3
EXPECTED_MIXED_CONTENT_BLOCKS_COUNT = 3
EXPECTED_TEXT_AND_IMAGE_BLOCKS_COUNT = 2
EXPECTED_REASONING_CONTENTS_COUNT = 2  # Used in multiple tests


class TestImageBytesToBase64URL:
    """Test the _image_bytes_to_base64_url function."""

    def test_converts_bytes_to_data_url(self) -> None:
        """Test converting image bytes to base64 data URL."""
        image_data = b"test_image_data"
        result = _image_bytes_to_base64_url(image_data)

        expected_b64 = base64.b64encode(image_data).decode("utf-8")
        expected_url = f"data:image/jpeg;base64,{expected_b64}"
        assert result == expected_url


class TestConvertContentBlockToOAIDict:
    """Test the _convert_content_block_to_oai_dict function."""

    def test_convert_text_block(self) -> None:
        """Test converting InputTextBlock to dict."""
        block = InputTextBlock(text="Hello!")
        result = _convert_content_block_to_oai_dict(block)

        assert isinstance(result, dict)
        assert result["type"] == "input_text"
        assert result["text"] == "Hello!"

    def test_convert_image_block_with_url(self) -> None:
        """Test converting InputImageBlock with URL to dict."""
        block = InputImageBlock(image_url="https://example.com/img.jpg", detail=ImageDetail.HIGH)
        result = _convert_content_block_to_oai_dict(block)

        assert isinstance(result, dict)
        assert result["type"] == "input_image"
        assert result["image_url"] == "https://example.com/img.jpg"
        assert result["detail"] == "high"

    def test_convert_image_block_with_bytes(self) -> None:
        """Test converting InputImageBlock with bytes to dict."""
        image_data = b"test_data"
        block = InputImageBlock(image_bytes=image_data, detail=ImageDetail.LOW)
        result = _convert_content_block_to_oai_dict(block)

        assert isinstance(result, dict)
        assert result["type"] == "input_image"
        assert result["detail"] == "low"
        expected_url = _image_bytes_to_base64_url(image_data)
        assert result["image_url"] == expected_url

    def test_convert_reasoning_content_with_summaries_and_contents(self) -> None:
        """Test converting ReasoningContent with summaries and contents to dict."""
        expected_summaries_count = 2

        block = ReasoningContent(
            id="reasoning_123",
            summaries=["Summary 1", "Summary 2"],
            contents=["Content 1", "Content 2"],
        )
        result = _convert_content_block_to_oai_dict(block)

        assert isinstance(result, dict)
        assert result["id"] == "reasoning_123"
        assert "summary" in result
        assert len(result["summary"]) == expected_summaries_count
        assert result["summary"][0] == {"text": "Summary 1", "type": "summary_text"}
        assert result["summary"][1] == {"text": "Summary 2", "type": "summary_text"}
        assert "content" in result
        assert len(result["content"]) == EXPECTED_REASONING_CONTENTS_COUNT
        assert result["content"][0] == {"text": "Content 1", "type": "reasoning_text"}
        assert result["content"][1] == {"text": "Content 2", "type": "reasoning_text"}

    def test_convert_reasoning_content_empty_lists(self) -> None:
        """Test converting ReasoningContent with empty summaries and contents."""
        block = ReasoningContent(id="reasoning_456", summaries=[], contents=[])
        result = _convert_content_block_to_oai_dict(block)

        assert isinstance(result, dict)
        assert result["id"] == "reasoning_456"
        assert result["summary"] == []
        assert result["content"] == []

    def test_convert_web_search_call_with_query_only(self) -> None:
        """Test converting WebSearchCallContent with query only to dict."""
        block = WebSearchCallContent(id="ws_123", query="test query", sources=None)
        result = _convert_content_block_to_oai_dict(block)

        assert isinstance(result, dict)
        assert result["id"] == "ws_123"
        assert result["type"] == "web_search_call"
        assert result["status"] == "completed"
        assert "action" in result
        assert result["action"]["query"] == "test query"
        assert result["action"]["type"] == "search"
        assert result["action"]["sources"] is None

    def test_convert_web_search_call_with_sources(self) -> None:
        """Test converting WebSearchCallContent with query and sources to dict."""
        block = WebSearchCallContent(
            id="ws_456",
            query="python tutorial",
            sources=["https://example.com/1", "https://example.com/2"],
        )
        result = _convert_content_block_to_oai_dict(block)

        assert isinstance(result, dict)
        assert result["id"] == "ws_456"
        assert result["type"] == "web_search_call"
        assert result["status"] == "completed"
        assert "action" in result
        assert result["action"]["query"] == "python tutorial"
        assert result["action"]["type"] == "search"
        assert result["action"]["sources"] == [
            {"url": "https://example.com/1", "type": "url"},
            {"url": "https://example.com/2", "type": "url"},
        ]

    def test_convert_web_search_call_with_empty_sources(self) -> None:
        """Test converting WebSearchCallContent with empty sources list."""
        block = WebSearchCallContent(id="ws_789", query="empty sources", sources=[])
        result = _convert_content_block_to_oai_dict(block)

        assert isinstance(result, dict)
        assert result["id"] == "ws_789"
        assert result["type"] == "web_search_call"
        assert result["status"] == "completed"
        assert "action" in result
        assert result["action"]["query"] == "empty sources"
        assert result["action"]["type"] == "search"
        assert result["action"]["sources"] is None


class TestConvertContentToOAIFormat:
    """Test the _convert_content_to_oai_format function."""

    def test_convert_string_content(self) -> None:
        """Test converting plain string content."""
        content = "Hello, world!"
        result = _convert_content_to_oai_format(content)
        assert result == "Hello, world!"


    def test_convert_list_with_single_text_block(self) -> None:
        """Test converting list with single TextBlock."""
        content = [InputTextBlock(text="Hello from list!")]
        result = _convert_content_to_oai_format(content)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["type"] == "input_text"
        assert result[0]["text"] == "Hello from list!"

    def test_convert_list_with_single_image_block(self) -> None:
        """Test converting list with single InputImageBlock."""
        content = [InputImageBlock(image_url="https://example.com/img.jpg", detail=ImageDetail.AUTO)]
        result = _convert_content_to_oai_format(content)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["type"] == "input_image"
        assert result[0]["image_url"] == "https://example.com/img.jpg"
        assert result[0]["detail"] == "auto"

    def test_convert_list_with_mixed_content(self) -> None:
        """Test converting list with text and image blocks."""
        content = [
            InputTextBlock(text="Here is an image:"),
            InputImageBlock(image_url="https://example.com/img.jpg", detail=ImageDetail.HIGH),
            InputTextBlock(text="What do you see?"),
        ]
        result = _convert_content_to_oai_format(content)

        assert isinstance(result, list)
        assert len(result) == EXPECTED_MIXED_CONTENT_BLOCKS_COUNT
        assert result[0]["type"] == "input_text"
        assert result[0]["text"] == "Here is an image:"
        assert result[1]["type"] == "input_image"
        assert result[1]["image_url"] == "https://example.com/img.jpg"
        assert result[1]["detail"] == "high"
        assert result[2]["type"] == "input_text"
        assert result[2]["text"] == "What do you see?"

    def test_convert_list_with_reasoning_content(self) -> None:
        """Test converting list with ReasoningContent blocks."""
        expected_blocks_count = 2

        content = [
            ReasoningContent(
                id="reasoning_1",
                summaries=["Thinking about the problem"],
                contents=["Step 1: Analyze", "Step 2: Solve"],
            ),
            InputTextBlock(text="Here's my answer"),
        ]
        result = _convert_content_to_oai_format(content)

        assert isinstance(result, list)
        assert len(result) == expected_blocks_count
        assert result[0]["id"] == "reasoning_1"
        assert len(result[0]["summary"]) == 1
        assert result[0]["summary"][0]["text"] == "Thinking about the problem"
        assert len(result[0]["content"]) == EXPECTED_REASONING_CONTENTS_COUNT
        assert result[1]["type"] == "input_text"
        assert result[1]["text"] == "Here's my answer"

    def test_convert_list_with_web_search_content(self) -> None:
        """Test converting list with WebSearchCallContent blocks."""
        expected_blocks_count = 2
        expected_sources_count = 2

        content = [
            WebSearchCallContent(
                id="ws_001",
                query="Python best practices",
                sources=["https://docs.python.org", "https://peps.python.org"],
            ),
            InputTextBlock(text="Based on the search results..."),
        ]
        result = _convert_content_to_oai_format(content)

        assert isinstance(result, list)
        assert len(result) == expected_blocks_count
        assert result[0]["id"] == "ws_001"
        assert result[0]["type"] == "web_search_call"
        assert result[0]["status"] == "completed"
        assert result[0]["action"]["query"] == "Python best practices"
        assert result[0]["action"]["type"] == "search"
        assert len(result[0]["action"]["sources"]) == expected_sources_count
        assert result[1]["type"] == "input_text"
        assert result[1]["text"] == "Based on the search results..."

    def test_convert_list_with_all_content_types(self) -> None:
        """Test converting list with all content block types."""
        expected_all_types_count = 4

        content = [
            InputTextBlock(text="Let me search for information"),
            WebSearchCallContent(id="ws_002", query="test query", sources=None),
            ReasoningContent(id="reasoning_2", summaries=["Analyzing results"], contents=[]),
            InputTextBlock(text="Final answer"),
        ]
        result = _convert_content_to_oai_format(content)

        assert isinstance(result, list)
        assert len(result) == expected_all_types_count
        assert result[0]["type"] == "input_text"
        assert result[1]["type"] == "web_search_call"
        assert result[2]["id"] == "reasoning_2"
        assert result[3]["type"] == "input_text"

    def test_convert_empty_list_raises_error(self) -> None:
        """Test that empty list raises ValueError."""
        content = []
        with pytest.raises(ValueError, match="Content list cannot be empty"):
            _convert_content_to_oai_format(content)


class TestGetOAIMessages:
    """Test the _get_oai_messages function."""

    def test_simple_string_user_message(self) -> None:
        """Test converting a simple user message with string content."""
        messages = [UserMessage(content="Hello!")]
        result = _get_oai_messages(messages)

        assert len(result) == EXPECTED_SINGLE_MESSAGE_COUNT
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello!"

    def test_user_message_with_text_block_in_list(self) -> None:
        """Test converting user message with TextBlock in list."""
        messages = [UserMessage(content=[InputTextBlock(text="Hello from TextBlock!")])]
        result = _get_oai_messages(messages)

        assert len(result) == EXPECTED_SINGLE_MESSAGE_COUNT
        assert result[0]["role"] == "user"
        assert isinstance(result[0]["content"], list)
        assert len(result[0]["content"]) == 1
        assert result[0]["content"][0]["type"] == "input_text"
        assert result[0]["content"][0]["text"] == "Hello from TextBlock!"

    def test_user_message_with_image_block_in_list(self) -> None:
        """Test converting user message with InputImageBlock in list."""
        messages = [
            UserMessage(
                content=[
                    InputImageBlock(
                        image_url="https://placehold.co/600x400/png", detail=ImageDetail.HIGH
                    )
                ]
            )
        ]
        result = _get_oai_messages(messages)

        assert len(result) == EXPECTED_SINGLE_MESSAGE_COUNT
        assert result[0]["role"] == "user"
        assert isinstance(result[0]["content"], list)
        assert len(result[0]["content"]) == 1
        assert result[0]["content"][0]["type"] == "input_image"
        assert result[0]["content"][0]["image_url"] == "https://placehold.co/600x400/png"
        assert result[0]["content"][0]["detail"] == "high"

    def test_system_message_excluded(self) -> None:
        """Test that system messages are excluded from OAI messages."""
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            UserMessage(content="Hello!"),
        ]
        result = _get_oai_messages(messages)

        # System message should be excluded (handled separately as instructions)
        assert len(result) == EXPECTED_SINGLE_MESSAGE_COUNT
        assert result[0]["role"] == "user"

    def test_ai_message_with_string_content(self) -> None:
        """Test converting AI message with string content."""
        messages = [AIMessage(content="Hello, how can I help?")]
        result = _get_oai_messages(messages)

        assert len(result) == EXPECTED_SINGLE_MESSAGE_COUNT
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "Hello, how can I help?"

    def test_ai_message_with_text_block_in_list(self) -> None:
        """Test converting AI message with TextBlock in list."""
        messages = [AIMessage(content=[OutputTextBlock(text="AI response here")])]
        result = _get_oai_messages(messages)

        assert len(result) == EXPECTED_SINGLE_MESSAGE_COUNT
        assert result[0]["role"] == "assistant"
        assert isinstance(result[0]["content"], list)
        assert len(result[0]["content"]) == 1
        assert result[0]["content"][0]["type"] == "output_text"
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
        result = _get_oai_messages(messages)

        assert len(result) == EXPECTED_MESSAGE_WITH_TOOL_RESULT_COUNT
        assert result[0]["role"] == "user"
        assert result[0]["content"][0]["type"] == "input_text"
        assert result[0]["content"][0]["text"] == "Run the tool"
        assert result[1]["type"] == "function_call_output"
        assert result[1]["call_id"] == "call_123"
        assert result[1]["output"] == "Tool output"

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
        result = _get_oai_messages(messages)

        assert len(result) == EXPECTED_MESSAGE_WITH_TOOL_CALL_COUNT
        assert result[0]["role"] == "assistant"
        assert result[0]["content"][0]["type"] == "output_text"
        assert result[0]["content"][0]["text"] == "I'll call the tool"
        assert result[1]["type"] == "function_call"
        assert result[1]["call_id"] == "call_456"
        assert result[1]["name"] == "test_tool"
        assert result[1]["arguments"] == '{"param": "value"}'

    def test_mixed_message_types(self) -> None:
        """Test converting a mix of message types."""
        messages = [
            SystemMessage(content="System instruction"),
            UserMessage(content="User question"),
            AIMessage(content="AI response"),
            UserMessage(content=[InputTextBlock(text="Follow-up")]),
        ]
        result = _get_oai_messages(messages)

        # System message excluded, so 3 messages expected
        assert len(result) == EXPECTED_MIXED_MESSAGES_COUNT
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "User question"
        assert result[1]["role"] == "assistant"
        assert result[1]["content"] == "AI response"
        assert result[2]["role"] == "user"
        assert isinstance(result[2]["content"], list)
        assert len(result[2]["content"]) == 1
        assert result[2]["content"][0]["type"] == "input_text"
        assert result[2]["content"][0]["text"] == "Follow-up"

    def test_user_message_with_list_of_content_blocks(self) -> None:
        """Test converting user message with list of content blocks."""
        messages = [
            UserMessage(
                content=[
                    InputTextBlock(text="Describe this image:"),
                    InputImageBlock(
                        image_url="https://placehold.co/600x400/png", detail=ImageDetail.HIGH
                    ),
                ]
            )
        ]
        result = _get_oai_messages(messages)

        assert len(result) == EXPECTED_SINGLE_MESSAGE_COUNT
        assert result[0]["role"] == "user"
        assert isinstance(result[0]["content"], list)
        assert len(result[0]["content"]) == EXPECTED_TEXT_AND_IMAGE_BLOCKS_COUNT
        assert result[0]["content"][0]["type"] == "input_text"
        assert result[0]["content"][0]["text"] == "Describe this image:"
        assert result[0]["content"][1]["type"] == "input_image"
        assert result[0]["content"][1]["image_url"] == "https://placehold.co/600x400/png"
        assert result[0]["content"][1]["detail"] == "high"


class TestGetInstructionFromMessages:
    """Test the _get_instruction_from_messages function."""

    def test_single_system_message_string(self) -> None:
        """Test extracting instruction from single system message with string."""
        messages = [SystemMessage(content="You are a helpful assistant.")]
        result = _get_instruction_from_messages(messages)
        assert result == "You are a helpful assistant."

    def test_single_system_message_text_block_in_list(self) -> None:
        """Test extracting instruction from system message with TextBlock in list."""
        messages = [SystemMessage(content=[InputTextBlock(text="You are helpful.")])]
        result = _get_instruction_from_messages(messages)
        assert result == "You are helpful."

    def test_multiple_system_messages(self) -> None:
        """Test extracting and combining multiple system messages."""
        messages = [
            SystemMessage(content="First instruction."),
            SystemMessage(content=[InputTextBlock(text="Second instruction.")]),
            UserMessage(content="User message"),
        ]
        result = _get_instruction_from_messages(messages)
        assert result == "First instruction.\nSecond instruction."

    def test_system_message_with_image_block_in_list_raises_error(self) -> None:
        """Test that InputImageBlock in system message list raises ValueError."""
        messages = [
            SystemMessage(content="Text instruction."),
            SystemMessage(
                content=[
                    InputImageBlock(image_url="https://example.com/img.jpg", detail=ImageDetail.AUTO)
                ]
            ),
        ]

        with pytest.raises(ValueError, match="only InputTextBlock is supported in system messages"):
            _get_instruction_from_messages(messages)

    def test_system_message_with_list_of_text_blocks(self) -> None:
        """Test extracting instructions from system message with list of TextBlocks."""
        messages = [
            SystemMessage(content=[InputTextBlock(text="First part."), InputTextBlock(text="Second part.")])
        ]
        result = _get_instruction_from_messages(messages)
        assert result == "First part.\nSecond part."

    def test_system_message_with_list_containing_image_raises_error(self) -> None:
        """Test that list with InputImageBlock in system message raises ValueError."""
        messages = [
            SystemMessage(
                content=[
                    InputTextBlock(text="Some text"),
                    InputImageBlock(image_url="https://example.com/img.jpg", detail=ImageDetail.AUTO),
                ]
            )
        ]

        with pytest.raises(ValueError, match="only InputTextBlock is supported in system messages"):
            _get_instruction_from_messages(messages)

    def test_no_system_messages(self) -> None:
        """Test when there are no system messages."""
        messages = [UserMessage(content="Hello"), AIMessage(content="Hi there")]
        result = _get_instruction_from_messages(messages)
        assert result == ""

    def test_empty_messages_list(self) -> None:
        """Test with empty messages list."""
        messages = []
        result = _get_instruction_from_messages(messages)
        assert result == ""


class TestGetOAITool:
    """Test the _get_oai_tool function."""

    def test_convert_dict_tool_web_search(self) -> None:
        """Test converting dict tool (web_search) returns dict as-is."""
        tool = {"type": "web_search"}
        result = _get_oai_tool(tool)

        assert result == {"type": "web_search"}

    def test_convert_dict_tool_with_options(self) -> None:
        """Test converting dict tool with additional options."""
        tool = {"type": "web_search", "filters": {"region": "us"}}
        result = _get_oai_tool(tool)

        assert result == {"type": "web_search", "filters": {"region": "us"}}

    def test_convert_callable_to_tool(self) -> None:
        """Test converting a callable function to tool format."""

        def sample_function(param1: str, param2: int = 5) -> str:
            """A sample function for testing.

            Args:
                param1: First parameter
                param2: Second parameter with default
            """
            return f"{param1}-{param2}"

        result = _get_oai_tool(sample_function)

        assert isinstance(result, dict)
        assert result["type"] == "function"
        assert result["name"] == "sample_function"
        assert "description" in result
        assert "parameters" in result
