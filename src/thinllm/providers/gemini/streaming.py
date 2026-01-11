"""Streaming support for Gemini API responses."""

from collections import defaultdict

from google.genai.types import GenerateContentResponse, Part

from thinllm.compat import StrEnum


class PartType(StrEnum):
    TEXT = "text"
    THOUGHT = "thought"
    FUNCTION_CALL = "function_call"
    FUNCTION_RESPONSE = "function_response"
    CODE_EXECUTION = "code_execution"
    CODE_EXECUTION_RESULT = "code_execution_result"


def get_part_type(part: Part) -> PartType:
    if part.thought:
        return PartType.THOUGHT
    if part.text:
        return PartType.TEXT
    if part.executable_code:
        return PartType.CODE_EXECUTION
    if part.code_execution_result:
        return PartType.CODE_EXECUTION_RESULT
    if part.function_call:
        return PartType.FUNCTION_CALL
    if part.function_response:
        return PartType.FUNCTION_RESPONSE

    raise NotImplementedError(f"Unable to determine part type for part: {part}")


class GeminiStreamMessageBuilder:
    """
    Builder for constructing AIMessage from streaming Gemini responses.

    This class accumulates streaming chunks from Gemini's API and builds
    a complete response object incrementally.

    Attributes:
        chunks: List of all received streaming chunks

    Example:
        >>> builder = GeminiStreamMessageBuilder()
        >>> for chunk in stream:
        ...     response = builder.add_chunk(chunk)
        ...     # Process incremental response
        >>> final_response = builder.response
    """

    def __init__(self):
        """Initialize an empty message builder."""
        self._response: GenerateContentResponse | None = None
        self.chunks: list[GenerateContentResponse] = []
        # Track accumulated text per candidate index and part index for text deltas
        self._candidate_part_to_text: dict[tuple[int, int], str] = defaultdict(str)
        # Track accumulated args JSON per candidate index and part index for function calls
        self._candidate_part_to_args_json: dict[tuple[int, int], str] = defaultdict(str)

    @property
    def response(self) -> GenerateContentResponse:
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

    def add_chunk(self, chunk: GenerateContentResponse) -> GenerateContentResponse:  # noqa: C901
        """
        Add a streaming chunk and update the response.

        Processes the chunk and updates the internal response object accordingly.
        Handles text deltas, thinking deltas, and function call arguments.

        Args:
            chunk: The streaming chunk to process

        Returns:
            The updated response object
        """
        self.chunks.append(chunk)

        # Initialize response from first chunk
        if self._response is None:
            self._response = chunk.model_copy()
            return self._response

        self._response.usage_metadata = chunk.usage_metadata

        if not chunk.candidates:
            return self._response

        if not self._response.candidates:
            self._response.candidates = []

        # Process all candidates in the chunk
        for candidate in chunk.candidates:
            if candidate.index is None:
                raise ValueError("Candidate index is required")

            if len(self._response.candidates) < candidate.index:
                self._response.candidates.append(candidate.model_copy())
                continue

            # merge the content of the candidate
            response_candidate = self._response.candidates[candidate.index]
            response_candidate.finish_reason = candidate.finish_reason
            response_candidate.finish_message = candidate.finish_message
            response_candidate.token_count = candidate.token_count

            if not response_candidate.content and candidate.content:
                response_candidate.content = candidate.content.model_copy()
                continue

            if not candidate.content:
                continue

            if not response_candidate.content:
                continue

            parts = candidate.content.parts if candidate.content and candidate.content.parts else []
            for part in parts:
                part_type = get_part_type(part)
                last_part = (
                    response_candidate.content.parts[-1]
                    if response_candidate.content and response_candidate.content.parts
                    else None
                )
                match part_type:
                    case PartType.TEXT:
                        # Check if the last part in response_candidate is also text
                        if last_part and get_part_type(last_part) == PartType.TEXT:
                            # Concatenate text to the last part
                            if last_part.text and part.text:
                                last_part.text += part.text
                            else:
                                last_part.text = part.text
                        elif response_candidate.content.parts:
                            # Create a new part
                            response_candidate.content.parts.append(part.model_copy())
                        else:
                            response_candidate.content.parts = [part.model_copy()]
                    case _:
                        # For non-text parts, always append a new part
                        if response_candidate.content.parts:
                            response_candidate.content.parts.append(part.model_copy())
                        else:
                            response_candidate.content.parts = [part.model_copy()]

        return self._response
