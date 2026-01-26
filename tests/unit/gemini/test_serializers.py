"""Unit tests for Gemini serializer functions."""

import base64

import pytest

# Mark all tests in this module as Gemini-specific
pytestmark = [pytest.mark.unit, pytest.mark.gemini]

from thinllm.messages import (
    InputImageBlock,
    InputTextBlock,
    OutputTextBlock,
    ToolResultContent,
)
from thinllm.providers.gemini.serializers import _convert_content_block_to_gemini_parts


class TestConvertContentBlockToGeminiParts:
    """Test the _convert_content_block_to_gemini_parts function."""

    def test_convert_tool_result_string_output(self) -> None:
        """Test converting ToolResultContent with string output."""
        block = ToolResultContent(
            tool_id="call_str",
            name="test_tool",
            output="Simple string output",
        )
        result = _convert_content_block_to_gemini_parts(block)

        assert isinstance(result, list)
        assert len(result) == 1
        assert "function_response" in result[0]
        assert result[0]["function_response"]["name"] == "test_tool"
        assert result[0]["function_response"]["response"]["result"] == "Simple string output"

    def test_convert_tool_result_with_text_blocks(self) -> None:
        """Test converting ToolResultContent with OutputTextBlock list."""
        block = ToolResultContent(
            tool_id="call_text",
            name="test_tool",
            output=[OutputTextBlock(text="First line"), OutputTextBlock(text="Second line")],
        )
        result = _convert_content_block_to_gemini_parts(block)

        assert isinstance(result, list)
        assert len(result) == 1
        assert "function_response" in result[0]
        assert result[0]["function_response"]["name"] == "test_tool"
        response_data = result[0]["function_response"]["response"]
        assert "result" in response_data
        assert isinstance(response_data["result"], list)
        assert len(response_data["result"]) == 2
        assert response_data["result"][0] == "First line"
        assert response_data["result"][1] == "Second line"

    def test_convert_tool_result_with_image_url(self) -> None:
        """Test converting ToolResultContent with InputImageBlock URL."""
        block = ToolResultContent(
            tool_id="call_image",
            name="test_tool",
            output=[InputImageBlock(image_url="https://example.com/result.jpg")],
        )
        result = _convert_content_block_to_gemini_parts(block)

        assert isinstance(result, list)
        assert len(result) == 1
        assert "function_response" in result[0]
        response_data = result[0]["function_response"]["response"]
        assert isinstance(response_data["result"], list)
        assert len(response_data["result"]) == 1
        assert "file_data" in response_data["result"][0]
        assert response_data["result"][0]["file_data"]["file_uri"] == "https://example.com/result.jpg"

    def test_convert_tool_result_with_image_bytes(self) -> None:
        """Test converting ToolResultContent with InputImageBlock containing image bytes."""
        test_bytes = b"fake_image_data"
        expected_base64 = base64.b64encode(test_bytes).decode("utf-8")

        block = ToolResultContent(
            tool_id="call_bytes",
            name="test_tool",
            output=[InputImageBlock(image_bytes=test_bytes, mimetype="image/png")],
        )
        result = _convert_content_block_to_gemini_parts(block)

        assert isinstance(result, list)
        assert len(result) == 1
        assert "function_response" in result[0]
        response_data = result[0]["function_response"]["response"]
        assert isinstance(response_data["result"], list)
        assert len(response_data["result"]) == 1
        assert "inline_data" in response_data["result"][0]
        assert response_data["result"][0]["inline_data"]["mime_type"] == "image/png"
        assert response_data["result"][0]["inline_data"]["data"] == expected_base64

    def test_convert_tool_result_with_mixed_content(self) -> None:
        """Test converting ToolResultContent with mixed text and image blocks."""
        test_bytes = b"chart_data"

        block = ToolResultContent(
            tool_id="call_mixed",
            name="test_tool",
            output=[
                OutputTextBlock(text="Here is the result:"),
                InputImageBlock(image_url="https://example.com/chart.png"),
                InputImageBlock(image_bytes=test_bytes, mimetype="image/jpeg"),
                OutputTextBlock(text="Analysis complete."),
            ],
        )
        result = _convert_content_block_to_gemini_parts(block)

        assert isinstance(result, list)
        assert len(result) == 1
        response_data = result[0]["function_response"]["response"]
        assert isinstance(response_data["result"], list)
        assert len(response_data["result"]) == 4
        # First text
        assert response_data["result"][0] == "Here is the result:"
        # First image (URL)
        assert "file_data" in response_data["result"][1]
        assert response_data["result"][1]["file_data"]["file_uri"] == "https://example.com/chart.png"
        # Second image (bytes)
        assert "inline_data" in response_data["result"][2]
        assert response_data["result"][2]["inline_data"]["mime_type"] == "image/jpeg"
        # Second text
        assert response_data["result"][3] == "Analysis complete."

    def test_convert_tool_result_with_none_output(self) -> None:
        """Test converting ToolResultContent with None output."""
        block = ToolResultContent(
            tool_id="call_none",
            name="test_tool",
            output=None,
        )
        result = _convert_content_block_to_gemini_parts(block)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["function_response"]["response"]["result"] == ""

    def test_convert_tool_result_with_empty_list(self) -> None:
        """Test converting ToolResultContent with empty list output."""
        block = ToolResultContent(
            tool_id="call_empty",
            name="test_tool",
            output=[],
        )
        result = _convert_content_block_to_gemini_parts(block)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["function_response"]["response"]["result"] == ""
