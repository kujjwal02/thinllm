# Contributing to ThinLLM

Thank you for your interest in contributing to ThinLLM! This guide will help you get started with development, testing, and contributing to the project.

## Table of Contents

- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Common Development Patterns](#common-development-patterns)
- [Testing](#testing)
- [Code Quality](#code-quality)
- [Submitting Changes](#submitting-changes)

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Git

### Clone and Install

```bash
# Clone the repository
git clone https://github.com/yourusername/thinllm.git
cd thinllm

# Install with all dependencies
pip install -e ".[all,dev]"
```

### Using uv (Optional)

If you prefer using `uv` for faster dependency management:

```bash
# Install uv if you haven't already
pip install uv

# Sync all dependencies including dev tools
uv sync --all-groups
```

### Environment Setup

Create a `.env` file in the project root with your API keys for testing:

```env
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
GEMINI_API_KEY=your-gemini-key
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
AWS_REGION=us-east-1
```

**Important**: Never commit your `.env` file or API keys to the repository.

## Project Structure

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
│       ├── openai/           # OpenAI provider implementation
│       │   ├── core.py       # Provider core logic
│       │   ├── serializers.py    # Request serialization
│       │   ├── deserializers.py  # Response deserialization
│       │   └── streaming.py      # Streaming support
│       ├── anthropic/        # Anthropic provider implementation
│       └── gemini/           # Gemini provider implementation
├── tests/
│   ├── unit/                 # Unit tests (mocked, fast)
│   │   ├── openai/
│   │   ├── anthropic/
│   │   ├── test_agent.py
│   │   └── test_config.py
│   └── integration/          # Integration tests (real API calls)
│       ├── test_agent.py
│       └── test_llm.py
├── examples/                 # Example applications
├── docs/                     # Documentation
├── pyproject.toml           # Project metadata and dependencies
└── ruff.toml                # Linter configuration
```

## Common Development Patterns

### Multi-turn Conversations

When implementing or testing multi-turn conversations:

```python
from thinllm import SystemMessage, UserMessage, llm, LLMConfig, ModelParams

config = LLMConfig(
    provider="anthropic",
    model_id="claude-sonnet-4",
    params=ModelParams(temperature=0.7)
)

messages = [
    SystemMessage(content="You are a helpful assistant."),
    UserMessage(content="What's the capital of France?")
]

# First response
response = llm(config, messages)
messages.append(response)

# Continue the conversation
messages.append(UserMessage(content="What's the population?"))
response = llm(config, messages)
```

### Streaming with Structured Output

Testing streaming structured data:

```python
from pydantic import BaseModel

class Story(BaseModel):
    title: str
    content: str
    moral: str

messages = [UserMessage(content="Write a short fable about a clever fox.")]

# Stream structured output
for chunk in llm(config, messages, output_schema=Story, stream=True):
    if chunk.title:
        print(f"Title: {chunk.title}")
    if chunk.content:
        print(f"Content: {chunk.content}")
```

### Error Handling

Always handle errors gracefully:

```python
from thinllm import llm, LLMConfig
from anthropic import APIError

try:
    response = llm(config, messages)
except APIError as e:
    print(f"API Error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Getting Effective Parameters

Check what parameters will actually be sent to the provider:

```python
config = LLMConfig(
    provider="bedrock_anthropic",
    model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    params=ModelParams(temperature=0.0, max_output_tokens=1024)
)

# See the actual parameters that will be sent to the provider
effective_params = config.get_effective_params()
print(effective_params)
```

### Adding a New Provider

To add support for a new LLM provider:

1. Create a new directory under `src/thinllm/providers/your_provider/`
2. Implement the required modules:
   - `core.py` - Main provider logic
   - `serializers.py` - Convert ThinLLM format to provider format
   - `deserializers.py` - Convert provider format to ThinLLM format
   - `streaming.py` - Handle streaming responses
3. Add provider detection in `src/thinllm/core.py`
4. Add tests in `tests/unit/your_provider/` and `tests/integration/`
5. Update documentation

## Testing

ThinLLM has comprehensive unit and integration tests.

### Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/unit/test_config.py

# Run specific test function
pytest tests/unit/test_config.py::test_config_creation
```

### Test Categories

The test suite is organized with pytest markers:

```bash
# Unit tests only (fast, mocked, no API calls)
pytest -m unit

# Integration tests only (slower, makes real API calls)
pytest -m integration

# Provider-specific tests
pytest -m openai      # OpenAI tests only
pytest -m anthropic   # Anthropic tests only
pytest -m gemini      # Gemini tests only
```

### Test Coverage

```bash
# Run tests with coverage report
pytest --cov=src/thinllm --cov-report=html

# View the HTML coverage report
open htmlcov/index.html  # On Mac/Linux
start htmlcov/index.html  # On Windows
```

### Writing Tests

#### Unit Tests

Unit tests should be fast and not make real API calls. Use mocks:

```python
import pytest
from unittest.mock import Mock, patch
from thinllm import llm, LLMConfig, UserMessage

@pytest.mark.unit
@pytest.mark.openai
def test_openai_basic_call(mock_openai_client):
    """Test basic OpenAI call with mocked client."""
    config = LLMConfig(provider="openai", model_id="gpt-4")
    messages = [UserMessage(content="Hello")]
    
    with patch("thinllm.providers.openai.core.OpenAI") as mock:
        mock.return_value = mock_openai_client
        response = llm(config, messages)
        
    assert response is not None
```

#### Integration Tests

Integration tests make real API calls and require valid API keys:

```python
import pytest
from thinllm import llm, LLMConfig, UserMessage, ModelParams

@pytest.mark.integration
@pytest.mark.anthropic
def test_anthropic_real_call():
    """Test real Anthropic API call."""
    config = LLMConfig(
        provider="anthropic",
        model_id="claude-sonnet-4",
        params=ModelParams(temperature=0.7, max_output_tokens=100)
    )
    messages = [UserMessage(content="Say hello")]
    
    response = llm(config, messages)
    
    assert response is not None
    assert len(response.content) > 0
```

### Test Configuration

The test configuration is defined in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "unit: marks tests as unit tests (fast, mocked)",
    "integration: marks tests as integration tests (slower, real API calls)",
    "openai: marks tests specific to OpenAI provider",
    "anthropic: marks tests specific to Anthropic provider",
    "gemini: marks tests specific to Gemini provider",
]
```

**Note**: Integration tests will consume API credits. Run them sparingly during development.

## Code Quality

### Linting and Formatting

We use Ruff for both linting and formatting:

```bash
# Check for linting issues
ruff check .

# Fix auto-fixable issues
ruff check --fix .

# Format code
ruff format .
```

### Type Checking

If you're using type checkers (optional but recommended):

```bash
# Using mypy
mypy src/thinllm

# Using pyright
pyright src/thinllm
```

### Pre-commit Checks

Before committing, run:

```bash
# Run all checks
ruff check . && ruff format . && pytest -m unit
```

### Code Style Guidelines

- Follow PEP 8 style guidelines
- Use type hints for function signatures
- Write descriptive docstrings for public functions
- Keep functions focused and single-purpose
- Add comments for complex logic
- Use meaningful variable names

Example:

```python
from typing import List
from thinllm.messages import MessageType, AIMessage

def process_messages(
    messages: List[MessageType],
    max_length: int = 1000
) -> AIMessage:
    """Process a list of messages and return an AI response.
    
    Args:
        messages: List of conversation messages
        max_length: Maximum length of the response
        
    Returns:
        AIMessage with the processed response
        
    Raises:
        ValueError: If messages list is empty
    """
    if not messages:
        raise ValueError("Messages list cannot be empty")
    
    # Implementation here
    ...
```

## Submitting Changes

### Creating a Pull Request

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Make your changes** and commit them:
   ```bash
   git add .
   git commit -m "Add feature: description of your changes"
   ```
5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
6. **Create a Pull Request** on GitHub

### Pull Request Guidelines

- **Write clear commit messages** describing what and why
- **Add tests** for any new functionality
- **Update documentation** if you change APIs or add features
- **Ensure all tests pass** before submitting
- **Keep PRs focused** - one feature or fix per PR
- **Respond to feedback** from reviewers

### Commit Message Format

Use clear, descriptive commit messages:

```
Add support for streaming with Bedrock provider

- Implement streaming deserializer for Bedrock
- Add integration tests for Bedrock streaming
- Update documentation with Bedrock examples
```

### What to Include in Your PR

- Clear description of the changes
- Motivation and context
- Related issue numbers (if applicable)
- Screenshots or examples (if relevant)
- Test results

## Development Tips

### Debugging

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Use pdb for debugging
import pdb; pdb.set_trace()
```

### Testing Against Multiple Providers

When developing cross-provider features, test against all providers:

```python
import pytest
from thinllm import llm, LLMConfig, UserMessage

@pytest.mark.integration
@pytest.mark.parametrize("provider,model", [
    ("openai", "gpt-4"),
    ("anthropic", "claude-sonnet-4"),
    ("gemini", "gemini-2.5-flash"),
])
def test_basic_call_all_providers(provider, model):
    """Test basic call works across all providers."""
    config = LLMConfig(provider=provider, model_id=model)
    messages = [UserMessage(content="Hello")]
    response = llm(config, messages)
    assert response is not None
```

### Performance Testing

For performance-sensitive changes:

```python
import time

def test_performance():
    start = time.time()
    # Your code here
    elapsed = time.time() - start
    assert elapsed < 1.0  # Should complete in under 1 second
```

## Questions or Issues?

- **Bug reports**: Open an issue on GitHub with detailed reproduction steps
- **Feature requests**: Open an issue describing the feature and use case
- **Questions**: Open a discussion on GitHub Discussions

## License

By contributing to ThinLLM, you agree that your contributions will be licensed under the MIT License.
