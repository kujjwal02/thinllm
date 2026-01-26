"""Agent system for multi-step LLM interactions with tool execution."""

import logging
from collections.abc import Callable, Generator
from typing import Any, Literal, overload

from pydantic import BaseModel, computed_field

from thinllm.compat import StrEnum
from thinllm.config import LLMConfig
from thinllm.core import llm
from thinllm.messages import (
    AIMessage,
    ContentBlock,
    MessageType,
    ToolCallContent,
    ToolOutput,
    ToolOutputStatus,
    ToolResultContent,
    UserMessage,
)
from thinllm.tools import Tool
from thinllm.utils import normalize_tools

logger = logging.getLogger(__name__)


class StepStatus(StrEnum):
    """Status codes for agent step execution."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class AgentToolResult(BaseModel):
    """
    Combined result of tool call and execution.

    Merges data from ToolCallContent and ToolResultContent to provide
    a complete view of tool invocation and result.

    Attributes:
        id: Unique identifier for the tool call
        name: Name of the tool
        raw_input: Raw input string before parsing
        input: Parsed input arguments as dictionary
        output: Tool output text (None if pending)
        metadata: Additional metadata from tool execution
        status: Execution status
    """

    id: str
    name: str
    raw_input: str = ""
    input: dict = {}
    output: str | None = None
    metadata: dict[str, Any] = {}
    status: ToolOutputStatus = ToolOutputStatus.PENDING


class AgentStep(BaseModel):
    """
    Represents one complete iteration in the agent loop.

    Each step contains an AI message (potentially with tool calls),
    an optional user message (with tool results), and the combined
    tool results data.

    Attributes:
        ai_message: LLM response with potential tool calls
        user_message: User message with tool results (None if no tools executed)
        status: Status of this step
    """

    ai_message: AIMessage
    user_message: UserMessage | None = None
    status: StepStatus = StepStatus.PENDING

    @computed_field
    @property
    def tool_results(self) -> list[AgentToolResult]:
        """
        Compute tool results by merging tool calls and tool results.

        Extracts ToolCallContent from ai_message and ToolResultContent from
        user_message, then merges them by tool_id to create AgentToolResult objects.

        Returns:
            List of AgentToolResult objects
        """
        # Extract tool calls from AI message
        tool_calls: dict[str, ToolCallContent] = {}
        if isinstance(self.ai_message.content, list):
            for block in self.ai_message.content:
                if isinstance(block, ToolCallContent):
                    tool_calls[block.tool_id] = block

        # Extract tool results from user message
        tool_results_map: dict[str, ToolResultContent] = {}
        if self.user_message and isinstance(self.user_message.content, list):
            for block in self.user_message.content:
                if isinstance(block, ToolResultContent):
                    tool_results_map[block.tool_id] = block

        # Merge tool calls and results
        merged: list[AgentToolResult] = []
        for tool_id, tool_call in tool_calls.items():
            tool_result = tool_results_map.get(tool_id)
            merged.append(
                AgentToolResult(
                    id=tool_id,
                    name=tool_call.name,
                    raw_input=tool_call.raw_input,
                    input=tool_call.input,
                    output=tool_result.output if tool_result else None,
                    metadata=tool_result.metadata if tool_result else {},
                    status=tool_result.status if tool_result else ToolOutputStatus.PENDING,
                )
            )

        return merged


class AgentRequest(BaseModel):
    """
    User request to the agent.

    Attributes:
        type: Discriminator for union type
        content: User message content (string or list of content blocks)
    """

    type: Literal["request"] = "request"
    content: str | list[ContentBlock]


class AgentResponse(BaseModel):
    """
    Complete agent execution result.

    Contains the final AI message, all execution steps, and overall status.

    Attributes:
        type: Discriminator for union type
        final_message: Final LLM response (None during streaming when tool calls are in progress)
        steps: All execution steps taken
        status: Overall execution status
        iterations: Number of iterations performed
    """

    type: Literal["response"] = "response"
    final_message: AIMessage | None = None
    steps: list[AgentStep] = []
    status: StepStatus = StepStatus.PENDING
    iterations: int = 0


# Type alias for discriminated union
AgentMessage = AgentRequest | AgentResponse


class Agent:
    """
    Agent that orchestrates multi-step LLM interactions with tool execution.

    The agent maintains a conversation history and automatically executes
    tools requested by the LLM, looping until the LLM provides a final
    answer or the maximum iteration limit is reached.

    Attributes:
        llm_config: Configuration for the LLM
        messages: Conversation message history
        tools: Available tools for the agent
        max_iterations: Maximum number of iterations before stopping

    Example:
        >>> from thinllm.config import LLMConfig, Provider
        >>> from thinllm.messages import SystemMessage
        >>> from thinllm.tools import tool
        >>>
        >>> @tool
        ... def get_weather(location: str) -> str:
        ...     return f"Weather in {location}: Sunny"
        >>>
        >>> config = LLMConfig(provider=Provider.OPENAI, model_id="gpt-4")
        >>> agent = Agent(
        ...     llm_config=config,
        ...     messages=[SystemMessage(content="You are a helpful assistant.")],
        ...     tools=[get_weather],
        ...     max_iterations=10
        ... )
        >>> request = AgentRequest(content="What's the weather in Paris?")
        >>> response = agent.ask(request)
        >>> print(response.final_message.content)
    """

    def __init__(
        self,
        llm_config: LLMConfig,
        messages: list[MessageType],
        tools: list[Tool | Callable | dict] | None = None,
        max_iterations: int = 10,
    ):
        """
        Initialize the agent.

        Args:
            llm_config: Configuration for the LLM
            messages: Initial messages (system messages and optionally preseeded user/ai messages)
            tools: Optional list of tools available to the agent
            max_iterations: Maximum number of iterations before stopping (default: 10)
        """
        self.llm_config = llm_config
        self.messages = list(messages)  # Copy to avoid modifying original
        self.max_iterations = max_iterations

        # Build tool mapping for lookup
        self._raw_tools = tools or []
        if tools:
            normalized_tools = normalize_tools(tools)
            self._tool_map: dict[str, Tool] = {t.name: t for t in normalized_tools}
        else:
            self._tool_map = {}

    def _execute_tool(self, tool_call: ToolCallContent) -> ToolResultContent:
        """
        Execute a single tool and return the result.

        Args:
            tool_call: Tool call content from AI message

        Returns:
            ToolResultContent with execution result
        """
        tool_name = tool_call.name

        # Check if tool exists
        if tool_name not in self._tool_map:
            error_msg = (
                f"Tool '{tool_name}' not found. Available tools: {list(self._tool_map.keys())}"
            )
            logger.error(error_msg)
            result_kwargs = {
                "tool_id": tool_call.tool_id,
                "name": tool_name,
                "raw_input": tool_call.raw_input,
                "input": tool_call.input,
                "output": error_msg,
                "status": ToolOutputStatus.FAILURE,
            }
            if tool_call.id:
                result_kwargs["id"] = tool_call.id
            return ToolResultContent(**result_kwargs)

        tool_obj = self._tool_map[tool_name]

        # Execute the tool
        try:
            result = tool_obj(**tool_call.input)

            # Normalize the return value
            match result:
                case None:
                    output = ""
                    metadata = {}
                    status = ToolOutputStatus.SUCCESS
                case str():
                    output = result
                    metadata = {}
                    status = ToolOutputStatus.SUCCESS
                case int() | float() | bool():
                    output = str(result)
                    metadata = {}
                    status = ToolOutputStatus.SUCCESS
                case ToolOutput():
                    output = result.output  # Now supports str | list[ToolOutputContent]
                    metadata = result.metadata
                    status = result.status
                case list():
                    # Check if it's a list of content blocks
                    from thinllm.messages import BaseContentBlock

                    if all(isinstance(item, BaseContentBlock) for item in result):
                        output = result
                        metadata = {}
                        status = ToolOutputStatus.SUCCESS
                    else:
                        output = str(result)
                        metadata = {}
                        status = ToolOutputStatus.SUCCESS
                case _:
                    # Convert to string for other types
                    output = str(result)
                    metadata = {}
                    status = ToolOutputStatus.SUCCESS

            # Build result kwargs
            result_kwargs = {
                "tool_id": tool_call.tool_id,
                "name": tool_name,
                "raw_input": tool_call.raw_input,
                "input": tool_call.input,
                "output": output,
                "status": status,
            }
            if tool_call.id:
                result_kwargs["id"] = tool_call.id
            # Only include metadata if it's not empty
            if metadata:
                result_kwargs["metadata"] = metadata

            return ToolResultContent(**result_kwargs)

        except Exception as e:
            error_msg = f"Error executing tool '{tool_name}': {str(e)}"
            logger.exception(error_msg)
            result_kwargs = {
                "tool_id": tool_call.tool_id,
                "name": tool_name,
                "raw_input": tool_call.raw_input,
                "input": tool_call.input,
                "output": error_msg,
                "metadata": {"error": str(e), "error_type": type(e).__name__},
                "status": ToolOutputStatus.FAILURE,
            }
            if tool_call.id:
                result_kwargs["id"] = tool_call.id
            return ToolResultContent(**result_kwargs)

    @overload
    def ask(
        self,
        request: AgentRequest,
        *,
        stream: Literal[False] = False,
    ) -> AgentResponse: ...

    @overload
    def ask(
        self,
        request: AgentRequest,
        *,
        stream: Literal[True],
    ) -> Generator[AgentResponse, None, AgentResponse]: ...

    def ask(
        self,
        request: AgentRequest,
        *,
        stream: bool = False,
    ) -> AgentResponse | Generator[AgentResponse, None, AgentResponse]:
        """
        Process a user request through the agent loop.

        The agent will:
        1. Add the user request to message history
        2. Loop until no tool calls or max iterations:
           - Call LLM with current messages
           - If LLM requests tool calls, execute them
           - Add tool results back to message history
           - Continue looping
        3. Return final AgentResponse with all steps

        Args:
            request: User request to process
            stream: Whether to stream the response. When True, yields AgentResponse
                objects at each meaningful event (token streaming, tool execution).

        Returns:
            AgentResponse with final message, steps, and status.
            If stream=True, returns a Generator that yields partial AgentResponse
            objects and returns the final AgentResponse.

        Example:
            >>> request = AgentRequest(content="What is 2+2?")
            >>> response = agent.ask(request)
            >>> print(response.final_message.content)

            # Streaming example:
            >>> for partial in agent.ask(request, stream=True):
            ...     print(partial.status)
        """
        if stream:
            return self._ask_stream(request)
        return self._ask_non_stream(request)

    def _ask_non_stream(self, request: AgentRequest) -> AgentResponse:
        """Non-streaming implementation of ask."""
        # Add user request to message history
        user_message = UserMessage(content=request.content)
        self.messages.append(user_message)

        # Track steps for this request (local variable)
        steps: list[AgentStep] = []
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1

            # Call LLM with current messages and tools
            ai_message = llm(
                self.llm_config,
                self.messages,
                tools=self._raw_tools if self._raw_tools else None,
            )

            # Add AI message to history
            self.messages.append(ai_message)

            # Extract tool calls from AI message
            tool_calls: list[ToolCallContent] = []
            if isinstance(ai_message.content, list):
                tool_calls = [
                    block for block in ai_message.content if isinstance(block, ToolCallContent)
                ]

            # If no tool calls, we're done
            if not tool_calls:
                return AgentResponse(
                    final_message=ai_message,
                    steps=steps,
                    status=StepStatus.SUCCEEDED,
                    iterations=iteration,
                )

            # Execute tools sequentially
            tool_results: list[ToolResultContent] = []
            for tool_call in tool_calls:
                tool_result = self._execute_tool(tool_call)
                tool_results.append(tool_result)

            # Create user message with tool results
            tool_result_user_message = UserMessage(content=tool_results)  # type: ignore[arg-type]

            # Create agent step with tool calls
            step = AgentStep(
                ai_message=ai_message,
                user_message=tool_result_user_message,
                status=StepStatus.SUCCEEDED,
            )
            steps.append(step)

            # Add tool results to message history
            self.messages.append(tool_result_user_message)

        # Max iterations reached - make one more LLM call
        logger.warning(f"Agent reached max iterations ({self.max_iterations}) without completion")

        # Add a message indicating iterations exhausted
        exhausted_message = UserMessage(
            content="Maximum iterations exhausted. Please provide a final response without calling any tools."
        )
        self.messages.append(exhausted_message)

        # Make final LLM call without tools
        final_ai_message = llm(
            self.llm_config,
            self.messages,
            tools=None,  # No tools for final call
        )

        # Add final message to history
        self.messages.append(final_ai_message)

        # Return response with failed status
        return AgentResponse(
            final_message=final_ai_message,
            steps=steps,
            status=StepStatus.FAILED,
            iterations=iteration,
        )

    def _extract_tool_calls(self, ai_message: AIMessage) -> list[ToolCallContent]:
        """Extract tool call content blocks from an AI message."""
        if isinstance(ai_message.content, list):
            return [block for block in ai_message.content if isinstance(block, ToolCallContent)]
        return []

    def _stream_llm_response(
        self,
        steps: list[AgentStep],
        iteration: int,
        use_tools: bool = True,
    ) -> Generator[AgentResponse, None, tuple[AIMessage, bool]]:
        """
        Stream LLM response and yield partial AgentResponses.

        Returns:
            Tuple of (final_ai_message, has_tool_calls)
        """
        stream = llm(
            self.llm_config,
            self.messages,
            tools=self._raw_tools if self._raw_tools and use_tools else None,
            stream=True,
        )

        has_tool_calls = False
        for partial_ai in stream:
            tool_calls = self._extract_tool_calls(partial_ai)

            if tool_calls:
                has_tool_calls = True
                yield AgentResponse(
                    final_message=None,
                    steps=[*steps, AgentStep(ai_message=partial_ai, status=StepStatus.IN_PROGRESS)],
                    status=StepStatus.IN_PROGRESS,
                    iterations=iteration,
                )
            else:
                yield AgentResponse(
                    final_message=partial_ai,
                    steps=steps,
                    status=StepStatus.IN_PROGRESS,
                    iterations=iteration,
                )

        return partial_ai, has_tool_calls  # type: ignore[return-value]

    def _execute_tools_streaming(
        self,
        ai_message: AIMessage,
        tool_calls: list[ToolCallContent],
        steps: list[AgentStep],
        iteration: int,
    ) -> Generator[AgentResponse, None, list[ToolResultContent]]:
        """
        Execute tools with streaming status updates.

        Yields AgentResponse at each status change (IN_PROGRESS -> SUCCESS/FAILURE).

        Returns:
            List of completed ToolResultContent
        """
        tool_results: list[ToolResultContent] = []

        for tool_call in tool_calls:
            # Create pending result
            pending_kwargs = {
                "tool_id": tool_call.tool_id,
                "name": tool_call.name,
                "raw_input": tool_call.raw_input,
                "input": tool_call.input,
                "output": "",
                "status": ToolOutputStatus.IN_PROGRESS,
            }
            if tool_call.id:
                pending_kwargs["id"] = tool_call.id
            pending_result = ToolResultContent(**pending_kwargs)
            tool_results.append(pending_result)

            # Yield IN_PROGRESS state
            yield self._build_tool_step_response(ai_message, tool_results, steps, iteration)

            # Execute and update
            tool_results[-1] = self._execute_tool(tool_call)

            # Yield completed state
            yield self._build_tool_step_response(ai_message, tool_results, steps, iteration)

        return tool_results

    def _build_tool_step_response(
        self,
        ai_message: AIMessage,
        tool_results: list[ToolResultContent],
        steps: list[AgentStep],
        iteration: int,
    ) -> AgentResponse:
        """Build an AgentResponse for tool execution state."""
        current_step = AgentStep(
            ai_message=ai_message,
            user_message=UserMessage(content=list(tool_results)),  # type: ignore[arg-type]
            status=StepStatus.IN_PROGRESS,
        )
        return AgentResponse(
            final_message=None,
            steps=[*steps, current_step],
            status=StepStatus.IN_PROGRESS,
            iterations=iteration,
        )

    def _ask_stream(self, request: AgentRequest) -> Generator[AgentResponse, None, AgentResponse]:
        """
        Streaming implementation of ask.

        Yields AgentResponse objects at each meaningful event:
        - Token streaming from LLM
        - Tool execution status changes (IN_PROGRESS -> SUCCESS/FAILURE)

        The last yielded value is always the complete final response.
        """
        self.messages.append(UserMessage(content=request.content))
        steps: list[AgentStep] = []
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1

            # Stream LLM and yield partial responses
            final_ai_message, has_tool_calls = yield from self._stream_llm_response(
                steps, iteration, use_tools=True
            )
            self.messages.append(final_ai_message)

            # No tool calls = done
            if not has_tool_calls:
                final_response = AgentResponse(
                    final_message=final_ai_message,
                    steps=steps,
                    status=StepStatus.SUCCEEDED,
                    iterations=iteration,
                )
                yield final_response
                return final_response

            # Execute tools with streaming
            tool_calls = self._extract_tool_calls(final_ai_message)
            tool_results = yield from self._execute_tools_streaming(
                final_ai_message, tool_calls, steps, iteration
            )

            # Finalize step
            tool_result_message = UserMessage(content=tool_results)  # type: ignore[arg-type]
            steps.append(
                AgentStep(
                    ai_message=final_ai_message,
                    user_message=tool_result_message,
                    status=StepStatus.SUCCEEDED,
                )
            )
            self.messages.append(tool_result_message)

        # Max iterations - stream final response without tools
        logger.warning(f"Agent reached max iterations ({self.max_iterations}) without completion")
        self.messages.append(
            UserMessage(
                content="Maximum iterations exhausted. Please provide a final response without calling any tools."
            )
        )

        final_ai_message, _ = yield from self._stream_llm_response(
            steps, iteration, use_tools=False
        )
        self.messages.append(final_ai_message)

        final_response = AgentResponse(
            final_message=final_ai_message,
            steps=steps,
            status=StepStatus.FAILED,
            iterations=iteration,
        )
        yield final_response
        return final_response
