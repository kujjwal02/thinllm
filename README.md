# ThinLLM

A thin, unified wrapper for LLM interactions with support for multiple providers (OpenAI, Anthropic, AWS Bedrock, and Gemini).

> **⚠️ Under Development**: This project is currently under active development. APIs may change, and some features may be incomplete or subject to modification.

## Table of Contents

- [Features](#features)
- [Why ThinLLM?](#why-thinllm)
- [Key Concepts](#key-concepts)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Basic Usage](#basic-usage)
  - [Streaming Responses](#streaming-responses)
  - [Structured Output](#structured-output-with-pydantic)
  - [Function Calling](#function-calling--tools)
- [Providers](#providers)
  - [OpenAI](#openai)
  - [Anthropic](#anthropic-claude)
  - [AWS Bedrock](#aws-bedrock-anthropic-models)
  - [Google Gemini](#google-gemini)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Single Function API**: One `llm()` function for all providers - no need to learn multiple APIs
- **Provider Support**: OpenAI, Anthropic (Claude), AWS Bedrock, and Google Gemini
- **Streaming**: Full support for streaming responses
- **Structured Output**: Get Pydantic models directly from LLMs with type safety
- **Function Calling**: Tool/function calling with automatic serialization
- **Type Safety**: Full type hints and runtime validation with Pydantic

## Why ThinLLM?

**One function. All providers. Zero hassle.**

Building applications with multiple LLM providers means learning different APIs, handling different response formats, and managing provider-specific quirks. **ThinLLM** eliminates this complexity:

- **Single `llm()` Function**: One function for all your LLM needs - no need to learn provider-specific APIs
- **Write Once, Use Anywhere**: Same code works with OpenAI, Claude, Bedrock, and Gemini
- **Provider Agnostic**: Switch providers by changing just the config - no code refactoring needed
- **Type Safety**: Full Pydantic integration for validated, structured outputs
- **Minimal Overhead**: Thin wrapper that stays close to native provider APIs
- **Production Ready**: Battle-tested with comprehensive test coverage

```python
# Switch providers by updating the config
config = LLMConfig(
    provider="anthropic",      # Was "openai"
    model_id="claude-sonnet-4", # Was "gpt-4"
    params=ModelParams(temperature=0.7)
)
# Same llm() function, same messages, same code structure!
response = llm(config, messages)
```

## Key Concepts

**ThinLLM** is built around a single powerful principle: **one function for all your LLM needs**.

Instead of learning different APIs for OpenAI, Anthropic, Bedrock, and Gemini, you just use the `llm()` function. Change providers by switching one configuration parameter - everything else stays the same.

### Core Components

1. **`llm()` function**: The only function you need - handles all LLM interactions
2. **`LLMConfig`**: Configure which provider and model to use
3. **`ModelParams`**: Standard parameters (temperature, max_tokens, etc.) that work across all providers
4. **Messages**: Use `SystemMessage`, `UserMessage`, and `AIMessage` to build conversations
5. **Structured Output**: Pass a Pydantic model as `output_schema` to get validated, typed responses
6. **Tools**: Pass Python functions as `tools` parameter for function calling

## Installation

### Basic Installation

```bash
pip install thinllm
```

### Provider-Specific Dependencies

Install dependencies for the providers you want to use:

```bash
# For OpenAI
pip install thinllm[openai]

# For Anthropic (Claude)
pip install thinllm[anthropic]

# For AWS Bedrock with Anthropic models
pip install thinllm[bedrock]

# For Google Gemini
pip install thinllm[gemini]

# For all providers
pip install thinllm[all]
```

### Environment Setup

Set your API keys as environment variables or use a `.env` file:

```bash
export OPENAI_API_KEY=your-openai-key-here
export ANTHROPIC_API_KEY=your-anthropic-key-here
export GEMINI_API_KEY=your-gemini-key-here

# AWS credentials (for Bedrock)
export AWS_ACCESS_KEY_ID=your-aws-access-key
export AWS_SECRET_ACCESS_KEY=your-aws-secret-key
export AWS_REGION=us-east-1
```

Or create a `.env` file in your project:

```env
OPENAI_API_KEY=your-openai-key-here
ANTHROPIC_API_KEY=your-anthropic-key-here
GEMINI_API_KEY=your-gemini-key-here
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
AWS_REGION=us-east-1
```

## Quick Start

### Basic Usage

Here's how to make your first LLM call:

```python
from thinllm import llm, LLMConfig, ModelParams, UserMessage
from dotenv import load_dotenv

# Load your API keys
load_dotenv()

# Configure your LLM
config = LLMConfig(
    provider="openai",  # or "anthropic", "bedrock_anthropic", "gemini"
    model_id="gpt-4",
    params=ModelParams(temperature=0.7, max_output_tokens=1024)
)

# Ask a question
messages = [UserMessage(content="What is the capital of France?")]
response = llm(config, messages)

# Print the response
print(response.content[0].text)
# Output: The capital of France is Paris.
```

### Streaming Responses

Stream responses in real-time for better user experience:

```python
# Stream the response - each chunk contains the complete response up to that point
for chunk in llm(config, messages, stream=True):
    if chunk.content:
        print(chunk.content[0].text)
        # Note: chunk.content[0].text contains the full text generated so far, not just the diff
```

### Structured Output with Pydantic

Get validated, structured data directly from the LLM:

```python
from pydantic import BaseModel
from thinllm import SystemMessage

class CalendarEvent(BaseModel):
    thought: str | None = None
    name: str | None = None
    date: str | None = None
    participants: list[str] | None = None

messages = [
    SystemMessage(content="Extract the event information."),
    UserMessage(content="Alice and Bob are going to a science fair on Friday."),
]

# Get structured output that matches your Pydantic model
response = llm(config, messages, output_schema=CalendarEvent)

print(f"Event: {response.name}")
print(f"Date: {response.date}")
print(f"Participants: {', '.join(response.participants)}")
```

### Function Calling / Tools

Enable the LLM to request function calls:

```python
from thinllm import SystemMessage

# Define a simple function
def get_horoscope(sign: str):
    """Get the horoscope for a zodiac sign."""
    return f"{sign}: Next Tuesday you will befriend a baby otter."

messages = [
    SystemMessage(content="You are a helpful assistant."),
    UserMessage(content="What is my horoscope? I am an Aquarius."),
]

# The LLM will indicate which function to call
response = llm(config, messages, tools=[get_horoscope])

# Check if the LLM requested a tool call
if response.get_tool_call_contents():
    tool_call = response.get_tool_call_contents()[0]
    print(f"LLM requested function: {tool_call.name}")
    print(f"With arguments: {tool_call.input}")
    
    # Execute the tool and continue the conversation
    messages.append(response)
    messages.append(
        UserMessage(content=[tool_call.get_tool_result(tools=[get_horoscope])])
    )
    
    # Get the final response after tool execution
    final_response = llm(config, messages, tools=[get_horoscope])
    print(final_response.content[0].text)
```

## Providers

### OpenAI

```python
from thinllm import LLMConfig, ModelParams

config = LLMConfig(
    provider="openai",
    model_id="gpt-4",
    params=ModelParams(
        temperature=0.7,
        max_output_tokens=4096
    )
)
```

**Supported Models**: `gpt-4`, `gpt-4-turbo`, `gpt-3.5-turbo`, `gpt-4o`, etc.

**Setup**: Set `OPENAI_API_KEY` environment variable

### Anthropic (Claude)

```python
config = LLMConfig(
    provider="anthropic",
    model_id="claude-sonnet-4",
    params=ModelParams(
        temperature=0.7,
        max_output_tokens=4096
    )
)
```

**Supported Models**: `claude-sonnet-4`, `claude-opus-4`, `claude-3-5-sonnet-20241022`, etc.

**Setup**: Set `ANTHROPIC_API_KEY` environment variable

### AWS Bedrock (Anthropic Models)

Use Claude models through AWS Bedrock:

```python
config = LLMConfig(
    provider="bedrock_anthropic",
    model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    params=ModelParams(
        temperature=0.0,
        max_output_tokens=1024
    )
)
```

**Supported Models**: Any Anthropic model available in AWS Bedrock
- `us.anthropic.claude-sonnet-4-5-20250929-v1:0`
- `global.anthropic.claude-sonnet-4-5-20250929-v1:0`
- And other Bedrock model IDs

**Setup**: Configure AWS credentials through environment variables or AWS CLI:
```env
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_REGION=us-east-1  # or your preferred region
```

### Google Gemini

```python
config = LLMConfig(
    provider="gemini",
    model_id="gemini-2.5-flash",
    params=ModelParams(temperature=0.7)
)
```

**Supported Models**: `gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-1.5-pro`, etc.

**Setup**: Set `GEMINI_API_KEY` environment variable

### Provider Feature Comparison

| Feature | OpenAI | Anthropic | Bedrock | Gemini |
|---------|--------|-----------|---------|--------|
| Basic Chat | ✅ | ✅ | ✅ | ✅ |
| Streaming | ✅ | ✅ | ✅ | ✅ |
| Structured Output | ✅ | ✅ | ✅ | ✅ |
| Function Calling | ✅ | ✅ | ✅ | ✅ |
| Vision (Images) | ✅ | ✅ | ✅ | ✅ |
| Thinking Mode | ❌ | ❌ | ❌ | ✅ |
| Built-in Search | ❌ | ❌ | ❌ | ✅ |
| Code Execution | ❌ | ❌ | ❌ | ✅ |

#### Gemini-Specific Features

**Thinking Mode** (Extended Reasoning):
```python
config = LLMConfig(
    provider="gemini",
    model_id="gemini-2.5-pro",
    params=ModelParams(temperature=0.7),
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
    provider: str                           # "openai", "anthropic", "bedrock_anthropic", or "gemini"
    model_id: str                          # Model identifier (e.g., "gpt-4", "claude-sonnet-4")
    params: ModelParams | None = None      # Standard model parameters
    model_args: dict[str, Any] = {}        # Provider-specific arguments
```

#### `ModelParams`

Standard parameters that work across all providers:

```python
class ModelParams(BaseModel):
    temperature: float | None = None           # Controls randomness (0.0 to 2.0)
    max_output_tokens: int | None = None       # Maximum tokens to generate
    top_p: float | None = None                 # Nucleus sampling parameter
    top_k: int | None = None                   # Top-k sampling parameter
    stop_sequences: list[str] | None = None    # Sequences where the model stops
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

- **`gemini_example.py`**: Comprehensive Gemini provider examples with thinking mode
- **`bedrock_example.py`**: AWS Bedrock integration examples
- **`agent_example.py`**: Multi-turn conversations and tool usage patterns
- **`streamlit_streaming_chat.py`**: Interactive streaming chat interface
- **`streamlit_agent_chat.py`**: Web-based chat interface with debug view

### Running Examples

```bash
# Install example dependencies
pip install streamlit

# Run a specific example
python examples/bedrock_example.py

# Run the Streamlit chat interface
streamlit run examples/streamlit_streaming_chat.py
```

## Troubleshooting

### Common Issues

**Import Errors**
```python
# Ensure you've installed the provider-specific dependencies
pip install thinllm[anthropic]  # or openai, gemini, bedrock
```

**API Key Issues**
```python
# Make sure your environment variables are set
from dotenv import load_dotenv
load_dotenv()

# Or set them directly
import os
os.environ["ANTHROPIC_API_KEY"] = "your-key"
```

**AWS Bedrock Authentication**
```bash
# Configure AWS CLI (recommended)
aws configure

# Or set environment variables
export AWS_ACCESS_KEY_ID=your-key
export AWS_SECRET_ACCESS_KEY=your-secret
export AWS_REGION=us-east-1
```

**Model Not Found**
```python
# Bedrock models use region-specific IDs:
# "us.anthropic.claude-sonnet-4-5-20250929-v1:0"  # US region
# "global.anthropic.claude-sonnet-4-5-20250929-v1:0"  # Global
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines on:

- Setting up the development environment
- Running tests
- Code quality standards
- Common development patterns
- Submitting pull requests

## License

MIT License - see [LICENSE](LICENSE) file for details

## Acknowledgments

- Built with [Pydantic](https://pydantic.dev/) for data validation
- Uses official SDKs: `openai`, `anthropic`, `google-genai`
