"""Integration tests for LLM providers (OpenAI and Anthropic).

These tests make real API calls and are not mocked.
They require valid API keys to be set in the environment.
"""

import pytest
from pydantic import BaseModel

from thinllm.config import LLMConfig, Provider
from thinllm.core import llm
from thinllm.messages import (
    AIMessage,
    OutputTextBlock,
    SystemMessage,
    ToolCallContent,
    UserMessage,
)
from thinllm.tools import tool

# Provider configurations for parametrized tests
PROVIDER_CONFIGS = [
    pytest.param(
        Provider.OPENAI,
        "gpt-5.2",
        {"temperature": 0.0},
        "openai_api_key",
        marks=pytest.mark.openai,
        id="openai-gpt5.2",
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
def test_llm_simple_question(
    request: pytest.FixtureRequest,
    provider: Provider,
    model_id: str,
    model_args: dict,
    api_key_fixture: str,
) -> None:
    """
    Test that the LLM can answer a simple factual question.

    This integration test verifies:
    1. The LLM config can be properly created with different models and parameters
    2. The LLM can receive a user message and respond
    3. The response is a valid AIMessage with text content
    4. The response contains the expected answer
    """
    # Get API key from fixture
    api_key = request.getfixturevalue(api_key_fixture)

    # Arrange
    config = LLMConfig(
        provider=provider,
        model_id=model_id,
        model_args=model_args,
    )
    messages = [UserMessage(content="What is the capital of France?")]

    # Act
    response = llm(config, messages)

    # Assert
    assert isinstance(response, AIMessage), f"Expected AIMessage, got {type(response)}"
    assert response.content, "Response content should not be empty"
    assert isinstance(response.content, list), (
        f"Expected content to be a list, got {type(response.content)}"
    )
    assert len(response.content) >= 1, f"Expected at least 1 content block, got {len(response.content)}"

    # Find text block
    text_blocks = [block for block in response.content if isinstance(block, OutputTextBlock)]
    assert text_blocks, "Expected at least one OutputTextBlock"
    assert text_blocks[0].text, "OutputTextBlock content should not be empty"
    assert "paris" in text_blocks[0].text.lower(), (
        f"Expected response to mention 'Paris', but got: {text_blocks[0].text}"
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    ("provider", "model_id", "model_args", "api_key_fixture"),
    PROVIDER_CONFIGS,
)
def test_llm_tool_call_horoscope(
    request: pytest.FixtureRequest,
    provider: Provider,
    model_id: str,
    model_args: dict,
    api_key_fixture: str,
) -> None:
    """
    Test that the LLM can call a tool to get horoscope information.

    This integration test verifies:
    1. Tools can be registered and passed to the LLM
    2. The LLM can receive a system message with instructions about tool calling
    3. The LLM correctly identifies when to call a tool
    4. The response contains a tool call request with the correct tool name
    5. The LLM informs the user before calling the tool (as per system message)
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
            content="You are a helpful assistant. Before calling a tool, inform the user that you are calling a tool."
        ),
        UserMessage(content="Hello, what is my horoscope?, I am an Aquarius."),
    ]

    # Act
    response = llm(config, messages, tools=[get_horoscope])

    # Assert
    assert isinstance(response, AIMessage), f"Expected AIMessage, got {type(response)}"

    # Check that response has content
    assert response.content, "Expected response to have content"
    assert isinstance(response.content, list), "Expected content to be a list of ContentBlocks"

    # Extract tool calls from content
    tool_calls = [block for block in response.content if isinstance(block, ToolCallContent)]
    assert tool_calls, "Expected tool call content blocks but got none"
    assert len(tool_calls) > 0, "Expected at least one tool call"

    # Verify the correct tool was called
    tool_call = tool_calls[0]
    assert tool_call.name == "get_horoscope", (
        f"Expected tool 'get_horoscope' but got '{tool_call.name}'"
    )

    # Verify the tool was called with the correct argument
    assert "sign" in tool_call.input, "Expected 'sign' parameter in tool call input"
    assert "aquarius" in tool_call.input["sign"].lower(), (
        f"Expected tool call with 'Aquarius', but got: {tool_call.input['sign']}"
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    ("provider", "model_id", "model_args", "api_key_fixture"),
    PROVIDER_CONFIGS,
)
def test_llm_structured_output_calendar_event(
    request: pytest.FixtureRequest,
    provider: Provider,
    model_id: str,
    model_args: dict,
    api_key_fixture: str,
) -> None:
    """
    Test that the LLM can extract structured data into a CalendarEvent schema.

    This integration test verifies:
    1. Structured output (output_schema parameter) works correctly
    2. The LLM can parse natural language and extract structured information
    3. The response is a properly validated Pydantic model instance
    4. All fields are correctly populated from the user message
    """
    # Get API key from fixture
    api_key = request.getfixturevalue(api_key_fixture)

    # Define the CalendarEvent schema
    class CalendarEvent(BaseModel):
        """Schema for a calendar event."""

        name: str
        date: str
        participants: list[str]

    # Arrange
    config = LLMConfig(
        provider=provider,
        model_id=model_id,
        model_args=model_args,
    )

    messages = [
        SystemMessage(content="Extract the event information."),
        UserMessage(content="Alice and Bob are going to a science fair on Friday."),
    ]

    # Act
    response = llm(config, messages, output_schema=CalendarEvent)

    # Assert
    assert isinstance(response, CalendarEvent), f"Expected CalendarEvent, got {type(response)}"

    # Verify the event name is extracted
    assert response.name, "Event name should not be empty"
    assert "science fair" in response.name.lower(), (
        f"Expected event name to contain 'science fair', but got: {response.name}"
    )

    # Verify the date is extracted
    assert response.date, "Event date should not be empty"
    assert "friday" in response.date.lower(), (
        f"Expected date to mention 'Friday', but got: {response.date}"
    )

    # Verify participants are extracted
    assert response.participants, "Participants list should not be empty"
    expected_participant_count = 2
    assert len(response.participants) == expected_participant_count, (
        f"Expected {expected_participant_count} participants, but got {len(response.participants)}: {response.participants}"
    )

    # Verify both Alice and Bob are in the participants
    participants_lower = [p.lower() for p in response.participants]
    assert "alice" in participants_lower, (
        f"Expected 'Alice' in participants, but got: {response.participants}"
    )
    assert "bob" in participants_lower, (
        f"Expected 'Bob' in participants, but got: {response.participants}"
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    ("provider", "model_id", "model_args", "api_key_fixture"),
    PROVIDER_CONFIGS,
)
def test_llm_simple_streaming(
    request: pytest.FixtureRequest,
    provider: Provider,
    model_id: str,
    model_args: dict,
    api_key_fixture: str,
) -> None:
    """
    Test that the LLM can stream a simple text response.

    This integration test verifies:
    1. Streaming returns a generator
    2. The generator yields AIMessage chunks incrementally
    3. Each chunk contains content
    4. The final message contains the complete response
    5. Content builds up over time (not all at once)
    """
    # Get API key from fixture
    api_key = request.getfixturevalue(api_key_fixture)

    # Arrange
    config = LLMConfig(
        provider=provider,
        model_id=model_id,
        model_args=model_args,
    )
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="Write a short haiku about programming."),
    ]

    # Act
    stream = llm(config, messages, stream=True)

    # Assert
    chunks = []
    for chunk in stream:
        # Verify each chunk is an AIMessage
        assert isinstance(chunk, AIMessage), f"Expected AIMessage chunk, got {type(chunk)}"

        # Store for later analysis
        chunks.append(chunk)

    # Verify we received at least one chunk (streaming can return short responses in a single chunk)
    assert len(chunks) >= 1, f"Expected at least one chunk in streaming, got {len(chunks)}"

    # Verify final message (the return value of the generator)
    final_message = chunk  # Last chunk is the final message
    assert isinstance(final_message, AIMessage), (
        f"Expected final message to be AIMessage, got {type(final_message)}"
    )
    assert final_message.content, "Final message content should not be empty"

    # Extract text from final message
    if isinstance(final_message.content, list):
        text_blocks = [b for b in final_message.content if isinstance(b, OutputTextBlock)]
        assert text_blocks, "Expected at least one OutputTextBlock in final message"
        final_text = text_blocks[0].text
    else:
        final_text = final_message.content

    assert final_text, "Final text should not be empty"
    min_text_length = 10
    assert len(final_text) > min_text_length, "Expected a substantial response"


@pytest.mark.integration
@pytest.mark.parametrize(
    ("provider", "model_id", "model_args", "api_key_fixture"),
    PROVIDER_CONFIGS,
)
def test_llm_streaming_with_structured_output(
    request: pytest.FixtureRequest,
    provider: Provider,
    model_id: str,
    model_args: dict,
    api_key_fixture: str,
) -> None:
    """
    Test that the LLM can stream a structured output (Pydantic model).

    This integration test verifies:
    1. Streaming with output_schema returns a generator
    2. The generator yields partial Pydantic model instances
    3. Each partial model contains incrementally more complete data
    4. The final model is fully validated and complete
    5. All required fields are populated in the final model
    """
    # Get API key from fixture
    api_key = request.getfixturevalue(api_key_fixture)

    # Define the Person schema
    class Person(BaseModel):
        """Schema for a person."""

        name: str
        age: int
        occupation: str
        hobbies: list[str]

    # Arrange
    config = LLMConfig(
        provider=provider,
        model_id=model_id,
        model_args=model_args,
    )

    messages = [
        SystemMessage(content="Extract person information from the text."),
        UserMessage(
            content="John Smith is a 35-year-old software engineer who enjoys hiking, photography, and reading."
        ),
    ]

    # Act
    stream = llm(config, messages, output_schema=Person, stream=True)

    # Assert
    chunks = []
    for partial_person in stream:
        # Verify each chunk is a Person instance
        assert isinstance(partial_person, Person), (
            f"Expected Person instance, got {type(partial_person)}"
        )

        # Store for later analysis
        chunks.append(partial_person)

    # Verify we received at least one chunk
    assert len(chunks) > 0, f"Expected at least one chunk in streaming, got {len(chunks)}"

    # Verify final model (the return value of the generator)
    final_person = partial_person  # Last chunk is the final model
    assert isinstance(final_person, Person), f"Expected final result to be Person, got {type(final_person)}"

    # Verify all fields are populated
    assert final_person.name, "Name should not be empty"
    assert "john" in final_person.name.lower() or "smith" in final_person.name.lower(), (
        f"Expected name to contain 'John' or 'Smith', but got: {final_person.name}"
    )

    assert final_person.age, "Age should be set"
    expected_age = 35
    assert final_person.age == expected_age, f"Expected age {expected_age}, but got: {final_person.age}"

    assert final_person.occupation, "Occupation should not be empty"
    assert "engineer" in final_person.occupation.lower() or "software" in final_person.occupation.lower(), (
        f"Expected occupation to contain 'engineer' or 'software', but got: {final_person.occupation}"
    )

    assert final_person.hobbies, "Hobbies should not be empty"
    assert len(final_person.hobbies) >= 2, (  # noqa: PLR2004
        f"Expected at least 2 hobbies, but got {len(final_person.hobbies)}: {final_person.hobbies}"
    )

