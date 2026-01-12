"""Message types and tool-related data models for LLM interactions."""

import logging
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field, model_validator

from thinllm.compat import StrEnum

if TYPE_CHECKING:
    from collections.abc import Callable

    from thinllm.tools import Tool

logger = logging.getLogger(__name__)


class AnthropicCacheControl(BaseModel):
    """
    Anthropic cache control configuration.

    Attributes:
        enabled: Whether cache control is enabled (default: True)
        ttl: Time-to-live for the cache. Can be "5m" (5 minutes) or "1h" (1 hour).
             If None, Anthropic will use the default TTL (5 minutes).
    """

    enabled: bool = True
    ttl: Literal["5m", "1h"] | None = None


class ContentExtra(BaseModel):
    """
    Extra information about content blocks.
    """

    gemini_thought_signature: bytes | None = Field(default=None, exclude=True)
    anthropic_cache_control: AnthropicCacheControl | None = None


class BaseContentBlock(BaseModel):
    """
    Base class for all content blocks.

    Attributes:
        id: Optional unique identifier for the content block
        extra: Optional extra metadata

    Note:
        Subclasses must define their own `type` field with a specific Literal type.
    """

    id: str | None = None
    extra: ContentExtra | None = None


class InputTextBlock(BaseContentBlock):
    """
    Input text block containing text content from user messages.
    """

    type: Literal["input_text"] = "input_text"
    text: str


class OutputTextBlock(BaseContentBlock):
    """
    Output text block containing text content from assistant messages.
    """

    type: Literal["output_text"] = "output_text"
    text: str


class ImageDetail(StrEnum):
    """
    Detail level for image content.
    """

    LOW = "low"
    HIGH = "high"
    AUTO = "auto"


class InputImageBlock(BaseContentBlock):
    """
    Input image block containing image content from user messages.
    Images are only used as inputs, never as outputs.
    """

    type: Literal["input_image"] = "input_image"
    image_url: str | None = None
    image_bytes: bytes | None = None
    mimetype: str | None = None
    detail: ImageDetail = ImageDetail.AUTO

    @model_validator(mode="after")
    def validate_image_url_or_bytes(self) -> "InputImageBlock":
        if self.image_url is None and self.image_bytes is None:
            raise ValueError("Either image_url or image_bytes must be provided")
        return self

    @classmethod
    def from_file(cls, filepath: str, detail: ImageDetail = ImageDetail.AUTO) -> "InputImageBlock":
        """
        Load an image from a file path.

        Args:
            filepath: Path to the image file
            detail: Detail level for image processing (default: AUTO)

        Returns:
            InputImageBlock with the loaded image data

        Example:
            >>> image_block = InputImageBlock.from_file("path/to/image.jpg")
        """
        from thinllm.utils import load_image_from_file

        image_bytes, mimetype = load_image_from_file(filepath)
        return cls(image_bytes=image_bytes, mimetype=mimetype, detail=detail)


class ToolOutputStatus(StrEnum):
    """Status codes for tool execution results."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILURE = "failure"


class ToolCallStatus(StrEnum):
    """Status codes for tool call content during streaming."""

    PENDING = "pending"
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"


class ReasoningContent(BaseContentBlock):
    """
    Reasoning content block from LLM response.

    Attributes:
        type: Content type identifier ("reasoning")
        id: Unique identifier for the reasoning block
        summaries: List of summary texts
        contents: List of reasoning content texts
        signature: Signature or identifier for the reasoning
        redacted_content: Optional redacted content
    """

    type: Literal["reasoning"] = "reasoning"
    summaries: list[str] = []
    contents: list[str] = []
    signature: str | None = None
    redacted_content: str | None = None


class WebSearchCallContent(BaseContentBlock):
    """
    Web search call content block from LLM response.

    Attributes:
        type: Content type identifier ("web_search")
        id: Unique identifier for the web search call
        query: The search query string
        sources: Optional list of source URLs
        status: Status of the web search (PENDING while in progress, COMPLETE when done)
    """

    type: Literal["web_search"] = "web_search"
    query: str
    sources: list[str] | None = None
    status: ToolCallStatus = ToolCallStatus.PENDING


class ToolCallContent(BaseContentBlock):
    """
    Tool call content block containing a request to call a tool.

    Attributes:
        type: Content type identifier ("tool_call")
        id: Unique identifier for the content block
        tool_id: Unique identifier for the tool call
        name: Name of the tool to call
        input: Parsed input arguments as a dictionary
        raw_input: Raw input string before parsing
        status: Status of the tool call (PENDING while streaming, COMPLETE when done)
    """

    type: Literal["tool_call"] = "tool_call"
    tool_id: str | None = None
    name: str
    input: dict = {}
    raw_input: str = ""
    status: ToolCallStatus = ToolCallStatus.PENDING

    def get_tool_result(
        self,
        tools: "list[Tool | Callable | dict]",
    ) -> "ToolResultContent":
        """
        Execute the tool and return the result.

        This is a convenience method that delegates to the get_tool_result function in utils.

        Args:
            tools: List of tools available for execution. Can include:
                - Tool objects
                - Callable functions
                - dict objects (e.g., for built-in tools like web_search)

        Returns:
            ToolResultContent with execution result

        Example:
            >>> from thinllm.tools import tool
            >>> @tool
            ... def add(a: int, b: int) -> int:
            ...     return a + b
            >>> tool_call = ToolCallContent(name="add", input={"a": 1, "b": 2}, tool_id="1")
            >>> result = tool_call.get_tool_result(tools=[add])
            >>> print(result.output)
            "3"
        """
        from thinllm.utils import get_tool_result

        return get_tool_result(self, tools)


class ToolResultContent(BaseContentBlock):
    """
    Tool result content block containing the result of a tool execution.

    Attributes:
        type: Content type identifier ("tool_result")
        id: Unique identifier for the content block
        tool_id: Unique identifier for the tool call this result corresponds to
        name: Name of the tool that was executed
        raw_input: Raw input string before parsing
        input: Parsed input arguments
        output: Output text from the tool
        metadata: Additional metadata
        status: Execution status (SUCCESS or FAILURE)
    """

    type: Literal["tool_result"] = "tool_result"
    tool_id: str | None = None
    name: str
    raw_input: str = ""
    input: dict | None = None
    output: str
    metadata: dict[str, Any] = {}
    status: ToolOutputStatus = ToolOutputStatus.SUCCESS


ContentBlock = (
    InputTextBlock
    | OutputTextBlock
    | InputImageBlock
    | ReasoningContent
    | WebSearchCallContent
    | ToolCallContent
    | ToolResultContent
)


class ToolOutput(BaseModel):
    """
    Output from a tool execution.

    Attributes:
        text: The text output from the tool
        metadata: Additional metadata about the execution
        status: Status of the tool execution (SUCCESS or FAILURE)
    """

    text: str
    metadata: dict[str, Any] = {}
    status: ToolOutputStatus = ToolOutputStatus.SUCCESS


class ToolResult(BaseModel):
    """
    Complete result of a tool execution including input and output.

    Attributes:
        id: Optional unique identifier for the result
        name: Name of the tool that was executed
        raw_input: Raw input string before parsing
        input: Parsed input arguments
        output: Output text from the tool
        metadata: Additional metadata
        status: Execution status (SUCCESS or FAILURE)
    """

    id: str | None = None
    name: str
    raw_input: str = ""
    input: dict | None = None
    output: str | None = None
    metadata: dict[str, Any] = {}
    status: ToolOutputStatus = ToolOutputStatus.SUCCESS


class BaseMessage(BaseModel):
    """
    Base class for all message types.

    Attributes:
        id: Unique identifier for the message
        content: Text content of the message
        role: Role of the message sender (e.g., 'user', 'assistant', 'system')
    """

    id: str | None = Field(description="The id of the message.", default=None)
    content: str | list[ContentBlock] = Field(description="The content of the message.")
    role: str = Field(description="The role of the message sender (e.g., 'user', 'assistant').")


class SystemMessage(BaseMessage):
    """
    System message containing instructions for the LLM.

    System messages are used to provide context, instructions, and guidelines
    to the language model.
    """

    role: str = "system"


class UserMessage(BaseMessage):
    """
    User message containing user input and optional tool results.

    Tool results should be included as ToolResultContent blocks in the content field.
    """

    role: str = "user"


class AIMessage(BaseMessage):
    """
    AI assistant message with optional tool calls.

    Tool call requests should be included as ToolCallContent blocks in the content field.
    """

    role: str = "ai"

    def get_tool_call_contents(self) -> list[ToolCallContent]:
        """
        Get all tool call content blocks from the message.

        Returns:
            List of ToolCallContent blocks. Empty list if content is a string or contains no tool calls.
        """
        if isinstance(self.content, list):
            return [block for block in self.content if isinstance(block, ToolCallContent)]
        return []


class AIMessageChunk(AIMessage):
    """
    A chunk of an AI message during streaming.

    This is used for incremental message building during streaming responses.
    """


# Type alias for any message type
MessageType = SystemMessage | UserMessage | AIMessage
