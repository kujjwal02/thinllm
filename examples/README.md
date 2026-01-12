# ThinLLM Examples

This directory contains example applications demonstrating various features of ThinLLM.

## Examples

### 1. Gemini Example (`gemini_example.py`)

A comprehensive example demonstrating the Gemini provider features.

**Features:**
- Basic text generation
- Streaming responses
- Structured output with Pydantic models
- Function calling with custom tools
- Thinking/reasoning mode
- Built-in Google Search tool
- Multi-turn conversations

**Requirements:**
1. Install gemini dependencies:
   ```bash
   uv sync --group gemini
   ```

2. Set your Gemini API key:
   ```bash
   export GEMINI_API_KEY="your-api-key-here"
   ```
   Or add to `.env` file:
   ```
   GEMINI_API_KEY=your-api-key-here
   ```

**Run:**
```bash
python examples/gemini_example.py
```

### 2. Bedrock Example (`bedrock_example.py`)

A comprehensive example demonstrating how to use Anthropic Claude models via Amazon Bedrock.

**Features:**
- Basic text generation with Bedrock
- Global vs Regional endpoints
- AWS credential configuration (explicit and auto-detection)
- Structured output with Pydantic models
- Function calling with custom tools
- Streaming responses
- Multi-turn conversations
- Built-in Bedrock presets

**Prerequisites:**
1. AWS Account with Amazon Bedrock access
2. Enable model access in AWS Console:
   - Navigate to [AWS Console > Bedrock > Model Access](https://console.aws.amazon.com/bedrock/home?region=us-west-2#/modelaccess)
   - Request access to Anthropic models
   - Available models: Claude Sonnet 4.5, Claude Haiku 4.5, Claude Opus 4.5, etc.
   - Model availability varies by region - see [AWS documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/models-regions.html)

**Requirements:**

1. Install Bedrock dependencies:
   ```bash
   uv sync --group anthropic-bedrock
   ```

2. Configure AWS credentials (choose one method):

   **Option A: Environment variables**
   ```bash
   export AWS_ACCESS_KEY_ID="your-access-key"
   export AWS_SECRET_ACCESS_KEY="your-secret-key"
   export AWS_REGION="us-west-2"  # Optional, defaults to us-east-1
   ```
   
   **Option B: AWS credentials file** (recommended)
   
   Create or edit `~/.aws/credentials`:
   ```ini
   [default]
   aws_access_key_id = your-access-key
   aws_secret_access_key = your-secret-key
   region = us-west-2
   ```
   
   **Option C: Explicit credentials in code** (see example)
   
   **Option D: IAM roles** (if running on EC2, ECS, Lambda, etc.)
   
   **Option E: AWS SSO**

**Run:**
```bash
python examples/bedrock_example.py
```

**Model IDs:**
- **Global endpoints** (recommended for availability):
  - `global.anthropic.claude-sonnet-4-5-20250929-v1:0`
  - `global.anthropic.claude-haiku-4-5-20251001-v1:0`
  - `global.anthropic.claude-opus-4-5-20251101-v1:0`
  
- **Regional endpoints** (for data residency requirements):
  - `anthropic.claude-sonnet-4-5-20250929-v1:0` (10% pricing premium)
  - `anthropic.claude-haiku-4-5-20251001-v1:0` (10% pricing premium)
  - Set region via `aws_region` in credentials

**Built-in Presets:**
- `bedrock-claude-sonnet-4-5`: Claude Sonnet 4.5 on global endpoint
- `bedrock-claude-haiku-4-5`: Claude Haiku 4.5 on global endpoint

**Usage Example:**
```python
from thinllm import LLMConfig, Provider, ModelParams, UserMessage, llm

# Simple usage with preset
response = llm("bedrock-claude-sonnet-4-5", [UserMessage(content="Hello!")])

# Or with explicit config
config = LLMConfig(
    provider=Provider.BEDROCK_ANTHROPIC,
    model_id="global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    params=ModelParams(temperature=0.7, max_output_tokens=1024)
)
response = llm(config, [UserMessage(content="Hello!")])
```

**Important Notes:**
- Global endpoints provide better availability and no pricing premium
- Regional endpoints required for data residency compliance (10% premium)
- AWS credentials are auto-detected from environment/files if not explicitly provided
- Ensure you have model access enabled in AWS Bedrock console

### 3. Anthropic Cache Example (`anthropic_cache_example.py`)

Demonstrates Anthropic's prompt caching feature to reduce costs and improve response times.

**Features:**
- Basic caching with default TTL (5 minutes)
- Explicit TTL configuration (1 hour)
- Multi-turn conversations with cache reuse
- Mixed cached and regular content
- Disabled caching option

**Requirements:**
1. Install Anthropic dependencies:
   ```bash
   uv sync --group anthropic
   ```

2. Set your Anthropic API key:
   ```bash
   export ANTHROPIC_API_KEY="your-api-key-here"
   ```

**Run:**
```bash
python examples/anthropic_cache_example.py
```

**When to Use Caching:**
- Large system prompts or instructions
- Frequently referenced documents
- Few-shot examples
- Stable context that doesn't change between requests

**Cache Control Options:**
- `ttl`: `"5m"` (default) or `"1h"`
- `enabled`: Set to `False` to disable caching
- Omit `ttl` to use provider's default

### 4. Anthropic Vision Example (`anthropic_vision_example.py`)

Demonstrates using Claude's vision capabilities to analyze images.

**Features:**
- Image analysis from URLs
- Image analysis from local files
- Multiple images in one request
- Base64 image encoding

**Requirements:**
1. Install Anthropic dependencies:
   ```bash
   uv sync --group anthropic
   ```

2. Set your Anthropic API key:
   ```bash
   export ANTHROPIC_API_KEY="your-api-key-here"
   ```

**Run:**
```bash
python examples/anthropic_vision_example.py
```

### 5. Agent Example (`agent_example.py`)

A command-line example showing how to use the Agent class with custom tools.

**Features:**
- Custom tool definitions (weather, calculator)
- Multi-step agent execution
- Tool result inspection

**Run:**
```bash
python examples/agent_example.py
```

### 6. Streamlit Agent Chat (`streamlit_agent_chat.py`)

An interactive web-based chat application with full debug capabilities.

**Features:**
- 🤖 Interactive chat interface
- 🔍 Web search integration
- 📊 Full debug view with:
  - Agent execution status
  - Step-by-step tool execution details
  - Reasoning content display
  - Web search queries and results
  - Complete conversation history
- 💬 Real-time message streaming
- 🗑️ Conversation reset

**Requirements:**
1. Set your OpenAI API key (choose one method):
   
   **Option A: Environment variable**
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```
   
   **Option B: .env file** (recommended)
   
   Create a `.env` file in the project root:
   ```bash
   OPENAI_API_KEY=your-api-key-here
   ```

2. Install dependencies:
   ```bash
   uv sync --group examples --group openai
   ```

**Run:**
```bash
streamlit run examples/streamlit_agent_chat.py
```

The app will open in your browser at `http://localhost:8501`.

**Usage Tips:**
- Ask questions that require current information (the agent will use web search)
- Check the debug sidebar to see how the agent processes your requests
- Expand tool execution details to see inputs and outputs
- Use the "Clear Conversation" button to start fresh

**Example Queries:**
- "What's the current weather in Paris?"
- "Who won the Nobel Prize in Physics this year?"
- "What are the latest developments in AI?"
- "Compare the populations of Tokyo and New York"

## Environment Setup

Make sure you have the required dependencies installed:

```bash
# Install all dependencies including examples
uv sync --all-groups

# Or install specific groups
uv sync --group examples --group openai
```

## API Keys

The examples require API keys/credentials for the LLM providers:

- **OpenAI**: Set `OPENAI_API_KEY` environment variable
- **Anthropic**: Set `ANTHROPIC_API_KEY` environment variable
- **Gemini**: Set `GEMINI_API_KEY` environment variable
- **Bedrock**: Configure AWS credentials (see Bedrock example above for multiple options)

You can set these in your shell or in a `.env` file in the project root.

