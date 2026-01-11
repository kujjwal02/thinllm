from openai.types.responses import Response as OAIResponse
from openai.types.responses import ResponseStreamEvent


class OAIStreamMessageBuilder:
    """
    Builder for constructing AIMessage from streaming OpenAI responses.

    This class accumulates streaming events from OpenAI's API and builds
    a complete response object incrementally.

    Attributes:
        events: List of all received stream events

    Example:
        >>> builder = OAIStreamMessageBuilder()
        >>> for event in stream:
        ...     response = builder.add_event(event)
        ...     # Process incremental response
        >>> final_response = builder.response
    """

    def __init__(self):
        """Initialize an empty message builder."""
        self._response: OAIResponse | None = None
        self.events: list[ResponseStreamEvent] = []

    @property
    def response(self) -> OAIResponse:
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

    def add_event(self, event: ResponseStreamEvent) -> OAIResponse:
        """
        Add a streaming event and update the response.

        Processes different event types and updates the internal response
        object accordingly. Handles text deltas, function call arguments,
        reasoning content, and completion events.

        Args:
            event: The stream event to process

        Returns:
            The updated response object

        Raises:
            ValueError: If an unknown event type is encountered
        """
        self.events.append(event)
        match event.type:
            case "response.created" | "response.in_progress":
                self._response = event.response.model_copy()
            case "response.output_item.added":
                self.response.output.append(event.item.model_copy())
            case "response.content_part.added":
                self.response.output[  # type: ignore[attr-defined]
                    event.output_index
                ].content.append(event.part)
            case "response.reasoning_summary_part.added":
                self.response.output[  # type: ignore[attr-defined]
                    event.output_index
                ].summary.append(event.part)
            case "response.reasoning_summary_text.delta":
                self.response.output[  # type: ignore[attr-defined]
                    event.output_index
                ].summary[event.summary_index].text += event.delta
            case "response.reasoning_summary_text.done":
                self.response.output[  # type: ignore[attr-defined]
                    event.output_index
                ].summary[event.summary_index].text = event.text
            case "response.reasoning_summary_part.done":
                self.response.output[  # type: ignore[attr-defined]
                    event.output_index
                ].summary[event.summary_index] = event.part.model_copy()
            case "response.output_text.delta":
                self.response.output[  # type: ignore[attr-defined]
                    event.output_index
                ].content[event.content_index].text += event.delta
            case "response.output_text.done":
                self.response.output[  # type: ignore[attr-defined]
                    event.output_index
                ].content[event.content_index].text = event.text
            case "response.content_part.done":
                self.response.output[  # type: ignore[attr-defined]
                    event.output_index
                ].content[event.content_index] = event.part.model_copy()
            case "response.output_item.done":
                self.response.output[event.output_index] = event.item.model_copy()
            case "response.function_call_arguments.delta":
                self.response.output[  # type: ignore[attr-defined]
                    event.output_index
                ].arguments += event.delta
            case "response.function_call_arguments.done":
                self.response.output[  # type: ignore[attr-defined]
                    event.output_index
                ].arguments = event.arguments
            case "response.completed":
                self._response = event.response.model_copy()
            case "response.web_search_call.in_progress" | "response.web_search_call.searching" | "response.web_search_call.completed":
                # No new content to add to the response
                pass
            case "response.output_text.annotation.added":
                self.response.output[event.output_index].content[event.content_index].annotations.append(event.annotation)  # type: ignore[attr-defined]
            case _:
                raise ValueError(f"Unknown event type: {event.type}")
        return self.response
