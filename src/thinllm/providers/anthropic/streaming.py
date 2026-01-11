"""Streaming support for Anthropic API responses."""

from collections import defaultdict

from anthropic.types import Message as AnthropicMessage
from anthropic.types import MessageStreamEvent

from thinllm.utils import parse_partial_json


class AnthropicStreamMessageBuilder:
    """
    Builder for constructing AIMessage from streaming Anthropic responses.

    This class accumulates streaming events from Anthropic's API and builds
    a complete response object incrementally.

    Attributes:
        events: List of all received stream events

    Example:
        >>> builder = AnthropicStreamMessageBuilder()
        >>> for event in stream:
        ...     response = builder.add_event(event)
        ...     # Process incremental response
        >>> final_response = builder.response
    """

    def __init__(self):
        """Initialize an empty message builder."""
        self._response: AnthropicMessage | None = None
        self.events: list[MessageStreamEvent] = []
        self._content_idx2partial_json: dict[int, str] = defaultdict(str)

    @property
    def response(self) -> AnthropicMessage:
        """
        Get the current response object.

        Returns:
            The accumulated response

        Raises:
            ValueError: If response has not been initialized yet
        """
        if self._response is None:
            raise ValueError("Response not initialized")
        return self._response

    def add_event(self, event: MessageStreamEvent) -> AnthropicMessage:
        """
        Add a streaming event and update the response.

        Processes different event types and updates the internal response
        object accordingly. Handles text deltas, thinking deltas, tool call
        arguments, and completion events.

        Args:
            event: The stream event to process

        Returns:
            The updated response object

        Raises:
            ValueError: If an unknown event type is encountered
        """
        self.events.append(event)
        match event.type:
            case "message_start":
                self._response = event.message.model_copy()
            case "content_block_start":
                self.response.content.append(event.content_block.model_copy())
            case "content_block_delta":
                content_block = self.response.content[event.index]
                match event.delta.type:
                    case "thinking_delta":
                        content_block.thinking += event.delta.thinking
                    case "signature_delta":
                        content_block.signature = event.delta.signature
                    case "text_delta":
                        content_block.text += event.delta.text
                    case "input_json_delta":
                        self._content_idx2partial_json[event.index] += event.delta.partial_json
                        cumulated_partial_json = self._content_idx2partial_json[event.index]
                        parsed_dict = parse_partial_json(cumulated_partial_json)
                        if isinstance(parsed_dict, dict):
                            content_block.input = parsed_dict
            case "message_delta":
                self.response.usage = event.usage
                self.response.stop_reason = event.delta.stop_reason
                self.response.stop_sequence = event.delta.stop_sequence
            
            case "content_block_stop" | "message_stop" | "signature" | "text" | "input_json" | "thinking":
                # These are snapshot/completion events that don't need special handling
                # The actual data is already in the content blocks from delta events
                pass

            case _:
                raise ValueError(f"Unknown event type: {event.type} with data {event.model_dump_json(indent=2)}")
        
        return self.response

