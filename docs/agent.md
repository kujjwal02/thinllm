# Agent System

The Agent system provides multi-step LLM interactions with automatic tool execution. The agent maintains a conversation history and automatically executes tools requested by the LLM, looping until the LLM provides a final answer or the maximum iteration limit is reached.

## Overview

The Agent class orchestrates the following workflow:

1. Receives a user request
2. Calls the LLM with the current conversation history and available tools
3. If the LLM requests tool calls:
   - Executes the requested tools
   - Adds tool results to the conversation
   - Loops back to step 2
4. Returns the final response when no more tool calls are needed

## Quick Start

```python
from thinllm import Agent, AgentRequest, LLMConfig, Provider, SystemMessage, tool

# Define a tool
@tool
def get_weather(location: str) -> str:
    """Get weather for a location."""
    return f"Weather in {location}: Sunny, 72°F"

# Configure and create agent
config = LLMConfig(provider=Provider.OPENAI, model_id="gpt-4")
agent = Agent(
    llm_config=config,
    messages=[SystemMessage(content="You are a helpful assistant.")],
    tools=[get_weather],
    max_iterations=10
)

# Ask a question
request = AgentRequest(content="What's the weather in Paris?")
response = agent.ask(request)

print(response.final_message.content)
```

## Data Models

### AgentRequest

User input to the agent.

```python
class AgentRequest(BaseModel):
    type: Literal["request"] = "request"  # Discriminator
    content: str | list[ContentBlock]
```

### AgentResponse

Complete agent execution result.

```python
class AgentResponse(BaseModel):
    type: Literal["response"] = "response"  # Discriminator
    final_message: AIMessage
    steps: list[AgentStep]
    status: StepStatus
    iterations: int
```

### AgentStep

Represents one iteration in the agent loop.

```python
class AgentStep(BaseModel):
    ai_message: AIMessage
    user_message: UserMessage | None
    status: StepStatus
    
    @property
    def tool_results(self) -> list[AgentToolResult]:
        """Computed field merging tool calls and results."""
```

### AgentToolResult

Combined view of tool call and execution result.

```python
class AgentToolResult(BaseModel):
    id: str
    name: str
    raw_input: str
    input: dict
    output: str | None
    metadata: dict
    status: ToolOutputStatus
```

### StepStatus

Status codes for agent steps.

```python
class StepStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
```

## Agent Class

### Constructor

```python
Agent(
    llm_config: LLMConfig,
    messages: list[MessageType],
    tools: list[Tool | Callable | dict] | None = None,
    max_iterations: int = 10
)
```

**Parameters:**
- `llm_config`: Configuration for the LLM (model, temperature, etc.)
- `messages`: Initial messages (system messages and optionally preseeded conversation)
- `tools`: Optional list of tools available to the agent
- `max_iterations`: Maximum number of iterations before stopping (default: 10)

### ask() Method

```python
def ask(self, request: AgentRequest) -> AgentResponse:
    """Process a user request through the agent loop."""
```

**Parameters:**
- `request`: User request to process

**Returns:**
- `AgentResponse`: Complete response with final message, steps, and status

## Tool Execution

The agent supports three types of tools:

### 1. Tool Objects

```python
from thinllm import Tool

tool = Tool.from_function(my_function)
agent = Agent(config, messages, tools=[tool])
```

### 2. Decorated Functions

```python
from thinllm import tool

@tool
def my_function(arg: str) -> str:
    """Function description."""
    return f"Result: {arg}"

agent = Agent(config, messages, tools=[my_function])
```

### 3. Callables

```python
def my_function(arg: str) -> str:
    """Function description."""
    return f"Result: {arg}"

agent = Agent(config, messages, tools=[my_function])
```

### Tool Return Values

Tools can return:

1. **String**: Used directly as output
   ```python
   return "Result text"
   ```

2. **ToolOutput**: Provides text, metadata, and status
   ```python
   from thinllm.messages import ToolOutput, ToolOutputStatus
   return ToolOutput(
       text="Result",
       metadata={"source": "db"},
       status=ToolOutputStatus.SUCCESS
   )
   ```

3. **Dict/Object**: Converted to JSON string
   ```python
   return {"key": "value", "number": 42}
   ```

4. **None**: Converted to empty string
   ```python
   return None
   ```

## Error Handling

When a tool execution fails:

1. The exception is caught
2. A `ToolResultContent` is created with:
   - `status=FAILURE`
   - `output` containing the error message
   - `metadata` containing error details
3. The error is passed back to the LLM
4. The LLM can decide how to handle the error

```python
@tool
def risky_operation(value: int) -> int:
    """An operation that might fail."""
    if value < 0:
        raise ValueError("Value must be positive")
    return value * 2

# If the LLM calls this with a negative value:
# - Tool execution fails
# - Error is passed to LLM
# - LLM can retry with correct value or inform user
```

## Iteration Limit

The agent has a configurable maximum iteration limit to prevent infinite loops:

```python
agent = Agent(
    llm_config=config,
    messages=messages,
    tools=tools,
    max_iterations=5  # Stop after 5 iterations
)
```

When the limit is reached:
- The agent returns an `AgentResponse` with `status=FAILED`
- The response includes all steps taken
- A warning is logged

## Accessing Response Data

### Final Message

```python
response = agent.ask(request)

# Get final message content
if isinstance(response.final_message.content, list):
    for block in response.final_message.content:
        if isinstance(block, TextBlock):
            print(block.text)
```

### Steps and Tool Results

```python
# Iterate through all steps
for i, step in enumerate(response.steps, 1):
    print(f"Step {i}:")
    
    # Check tool results
    for tool_result in step.tool_results:
        print(f"  Tool: {tool_result.name}")
        print(f"  Input: {tool_result.input}")
        print(f"  Output: {tool_result.output}")
        print(f"  Status: {tool_result.status}")
```

### Response Status

```python
response = agent.ask(request)

if response.status == StepStatus.SUCCEEDED:
    print("Agent completed successfully")
elif response.status == StepStatus.FAILED:
    print(f"Agent failed after {response.iterations} iterations")
```

## Examples

### Example 1: Simple Q&A (No Tools)

```python
agent = Agent(
    llm_config=config,
    messages=[SystemMessage(content="You are a helpful assistant.")],
    tools=None  # No tools
)

request = AgentRequest(content="What is the capital of France?")
response = agent.ask(request)

# Agent completes in 1 iteration without tool calls
assert response.iterations == 1
```

### Example 2: Single Tool Call

```python
@tool
def get_time(timezone: str) -> str:
    """Get current time in a timezone."""
    return f"Current time in {timezone}: 3:45 PM"

agent = Agent(
    llm_config=config,
    messages=[SystemMessage(content="You are a helpful assistant.")],
    tools=[get_time]
)

request = AgentRequest(content="What time is it in Tokyo?")
response = agent.ask(request)

# Agent makes 2 iterations:
# 1. LLM requests tool call
# 2. LLM provides final answer with time
assert response.iterations == 2
assert len(response.steps[0].tool_results) == 1
```

### Example 3: Multiple Tools

```python
@tool
def search_database(query: str) -> str:
    """Search the database."""
    return f"Found 5 results for: {query}"

@tool
def get_details(item_id: int) -> str:
    """Get details for an item."""
    return f"Details for item {item_id}: ..."

agent = Agent(
    llm_config=config,
    messages=[SystemMessage(content="You are a helpful assistant.")],
    tools=[search_database, get_details]
)

request = AgentRequest(content="Find products about Python and show details")
response = agent.ask(request)

# Agent may call multiple tools across multiple iterations
```

### Example 4: Preseeded Conversation

```python
agent = Agent(
    llm_config=config,
    messages=[
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="My name is Alice."),
        AIMessage(content="Nice to meet you, Alice!"),
    ],
    tools=[get_weather]
)

# Agent remembers the conversation context
request = AgentRequest(content="What's the weather where I live?")
response = agent.ask(request)
```

## Best Practices

1. **Clear System Messages**: Provide clear instructions about when and how to use tools
   ```python
   SystemMessage(
       content="You are a helpful assistant. Use tools when you need "
               "real-time information. Always explain what you're doing."
   )
   ```

2. **Tool Descriptions**: Write clear docstrings for tools
   ```python
   @tool
   def get_stock_price(symbol: str) -> str:
       """
       Get the current stock price for a symbol.
       
       Args:
           symbol: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
       
       Returns:
           Current stock price and change
       """
   ```

3. **Error Handling**: Handle errors gracefully in tools
   ```python
   @tool
   def api_call(endpoint: str) -> str:
       """Call an external API."""
       try:
           response = requests.get(endpoint)
           response.raise_for_status()
           return response.text
       except requests.RequestException as e:
           return f"API call failed: {str(e)}"
   ```

4. **Iteration Limits**: Set appropriate limits based on task complexity
   ```python
   # Simple tasks
   agent = Agent(..., max_iterations=5)
   
   # Complex research tasks
   agent = Agent(..., max_iterations=20)
   ```

5. **Monitoring**: Check response status and iterations
   ```python
   response = agent.ask(request)
   
   if response.status == StepStatus.FAILED:
       logger.warning(f"Agent failed after {response.iterations} iterations")
       # Handle failure case
   ```

## Discriminated Unions

`AgentRequest` and `AgentResponse` use discriminated unions for type safety:

```python
# Type narrowing with discriminator
def handle_message(msg: AgentMessage):
    if msg.type == "request":
        # TypeScript/mypy knows this is AgentRequest
        process_request(msg)
    else:
        # TypeScript/mypy knows this is AgentResponse
        process_response(msg)
```

## Limitations

1. **No Streaming**: The agent currently only supports non-streaming responses
2. **Sequential Tool Execution**: Tools are executed one at a time, not in parallel
3. **No Nested Agents**: Agents cannot call other agents as tools (yet)
4. **Dict-based Tools**: Special tools like `{"type": "web_search"}` are passed to the LLM provider but cannot be executed by the agent

## See Also

- [Tools Documentation](tools.md)
- [Messages Documentation](messages.md)
- [LLM Configuration](config.md)

