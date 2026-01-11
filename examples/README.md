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

### 2. Agent Example (`agent_example.py`)

A command-line example showing how to use the Agent class with custom tools.

**Features:**
- Custom tool definitions (weather, calculator)
- Multi-step agent execution
- Tool result inspection

**Run:**
```bash
python examples/agent_example.py
```

### 3. Streamlit Agent Chat (`streamlit_agent_chat.py`)

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

The examples require API keys for the LLM providers:

- **OpenAI**: Set `OPENAI_API_KEY` environment variable
- **Anthropic**: Set `ANTHROPIC_API_KEY` environment variable
- **Gemini**: Set `GEMINI_API_KEY` environment variable

You can set these in your shell or in a `.env` file in the project root.

