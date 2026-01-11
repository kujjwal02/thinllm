# ThinLLM

A thin, unified wrapper for LLM interactions with support for multiple providers (OpenAI, Anthropic, Gemini) and agent capabilities.

## Features

- **Unified Interface**: Single API for multiple LLM providers
- **Provider Support**: OpenAI, Anthropic (Claude), and Google Gemini
- **Streaming**: Full support for streaming responses
- **Structured Output**: Get Pydantic models directly from LLMs
- **Function Calling**: Tool/function calling with automatic serialization
- **Agent Framework**: Built-in agent with tool execution and reasoning
- **Type Safety**: Full type hints and runtime validation

## Installation

### Basic Installation

```bash
# Install core package
uv sync
```

### Provider-Specific Dependencies

Install dependencies for the providers you want to use:

```bash
# OpenAI
uv sync --group openai

# Anthropic
uv sync --group anthropic

# Gemini
uv sync --group gemini

# All providers
uv sync --group openai --group anthropic --group gemini
```

## Quick Start

### Basic Usage

```python
from thinllm import llm, LLMConfig, Provider, UserMessage

# Configure your provider
config = LLMConfig(
    provider=Provider.OPENAI,  # or Provider.ANTHROPIC, Provider.GEMINI
    model_id="gpt-4",
    model_args={"temperature": 0.7}
)

# Make a request
messages = [UserMessage(content="What is 2+2?")]
response = llm(config, messages)
print(response.content)
```

### Streaming

```python
# Stream the response
for chunk in llm(config, messages, stream=True):
    print(chunk.content, end="", flush=True)
```

### Structured Output

```python
from pydantic import BaseModel

class Recipe(BaseModel):
    name: str
    ingredients: list[str]
    prep_time_minutes: int

# Get structured output
recipe = llm(config, messages, output_schema=Recipe)
print(f"Recipe: {recipe.name}")
print(f"Ingredients: {', '.join(recipe.ingredients)}")
```

### Function Calling

```python
from thinllm.tools import tool

@tool
def get_weather(location: str) -> str:
    """Get the weather for a location."""
    return f"The weather in {location} is sunny."

response = llm(config, messages, tools=[get_weather])
```

### Agent with Tools

```python
from thinllm import Agent

agent = Agent(
    llm_config=config,
    messages=[SystemMessage(content="You are a helpful assistant.")],
    tools=[get_weather],
    max_iterations=5
)

result = agent.run("What's the weather in Paris?")
print(result.final_response)
```

## Providers

### OpenAI

```python
config = LLMConfig(
    provider=Provider.OPENAI,
    model_id="gpt-4",
    model_args={"temperature": 0.7}
)
```

**Supported Models**: `gpt-4`, `gpt-4-turbo`, `gpt-3.5-turbo`, etc.

**API Key**: Set `OPENAI_API_KEY` environment variable

### Anthropic

```python
config = LLMConfig(
    provider=Provider.ANTHROPIC,
    model_id="claude-sonnet-4-5",
    model_args={
        "temperature": 0.7,
        "max_tokens": 4096
    }
)
```

**Supported Models**: `claude-sonnet-4-5`, `claude-opus-4`, etc.

**API Key**: Set `ANTHROPIC_API_KEY` environment variable

### Gemini

```python
config = LLMConfig(
    provider=Provider.GEMINI,
    model_id="gemini-2.5-flash",
    model_args={"temperature": 0.7}
)
```

**Supported Models**: `gemini-2.5-flash`, `gemini-2.5-pro`, etc.

**API Key**: Set `GEMINI_API_KEY` environment variable

#### Gemini-Specific Features

**Thinking Mode** (Extended Reasoning):
```python
config = LLMConfig(
    provider=Provider.GEMINI,
    model_id="gemini-2.5-pro",
    model_args={
        "thinking_budget": 2048,      # Allocate tokens for reasoning
        "include_thoughts": True,     # Include reasoning in response
    }
)
```

**Built-in Tools**:
```python
# Google Search
response = llm(config, messages, tools=[{"google_search": {}}])

# Code Execution
response = llm(config, messages, tools=[{"code_execution": {}}])

# URL Context
response = llm(config, messages, tools=[{"url_context": {}}])
```

## API Reference

### Main Functions

#### `llm()`

Unified interface for LLM interactions.

```python
def llm(
    llm_config: LLMConfig,
    messages: list[MessageType],
    *,
    output_schema: type[OutputSchemaType] | None = None,
    tools: list[Tool | Callable | dict] | None = None,
    stream: bool = False,
) -> AIMessage | OutputSchemaType | Generator[...]:
    """
    Make an LLM request.
    
    Args:
        llm_config: Configuration for the LLM
        messages: List of conversation messages
        output_schema: Optional Pydantic model for structured output
        tools: Optional list of tools/functions
        stream: Whether to stream the response
        
    Returns:
        AIMessage, structured output, or generator
    """
```

### Configuration

#### `LLMConfig`

```python
class LLMConfig(BaseModel):
    provider: Provider          # OPENAI, ANTHROPIC, or GEMINI
    model_id: str              # Model identifier
    model_args: dict[str, Any] # Provider-specific arguments
```

### Messages

- `SystemMessage(content: str)` - System instructions
- `UserMessage(content: str | list[ContentBlock])` - User input
- `AIMessage(content: str | list[ContentBlock])` - AI response

### Content Blocks

- `InputTextBlock` - Text input from user
- `OutputTextBlock` - Text output from AI
- `InputImageBlock` - Image input
- `ReasoningContent` - Reasoning/thinking content
- `ToolCallContent` - Tool call request
- `ToolResultContent` - Tool execution result

## Examples

See the [`examples/`](examples/) directory for complete examples:

- **`gemini_example.py`**: Comprehensive Gemini provider examples
- **`agent_example.py`**: Agent with custom tools
- **`streamlit_agent_chat.py`**: Interactive web-based chat with debug view

## Testing

### Run All Tests

```bash
pytest
```

### Run Provider-Specific Tests

```bash
# OpenAI tests only
pytest -m openai

# Anthropic tests only
pytest -m anthropic

# Gemini tests only
pytest -m gemini

# Integration tests only (makes real API calls)
pytest -m integration

# Unit tests only (mocked)
pytest -m unit
```

### Environment Variables for Testing

Set API keys in `.env` file:

```
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
GEMINI_API_KEY=your-gemini-key
```

## Development

### Setup

```bash
# Install all dependencies including dev tools
uv sync --all-groups

# Run tests with coverage
pytest --cov

# Run linter
ruff check .

# Format code
ruff format .
```

### Project Structure

```
thinllm/
├── src/thinllm/
│   ├── __init__.py           # Main exports
│   ├── core.py               # Unified llm() function
│   ├── config.py             # Configuration models
│   ├── messages.py           # Message types
│   ├── tools.py              # Tool definitions
│   ├── agent.py              # Agent implementation
│   └── providers/
│       ├── openai/           # OpenAI provider
│       ├── anthropic/        # Anthropic provider
│       └── gemini/           # Gemini provider
├── tests/
│   ├── integration/          # Integration tests (real API calls)
│   └── unit/                 # Unit tests (mocked)
├── examples/                 # Example applications
└── docs/                     # Documentation
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

[Add your license here]

## Acknowledgments

- Built with [Pydantic](https://pydantic.dev/) for data validation
- Uses official SDKs: `openai`, `anthropic`, `google-genai`
