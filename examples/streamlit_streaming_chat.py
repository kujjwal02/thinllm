"""Streamlit Chat Application for ThinLLM Agent with Configurable Streaming.

This application demonstrates the Agent class with configurable streaming mode,
including tool execution details, conversation history, and web search integration.

Usage:
    streamlit run examples/streamlit_streaming_chat.py

Requirements:
    - Set OPENAI_API_KEY environment variable (or add to .env file)
    - Set ANTHROPIC_API_KEY environment variable for Anthropic models
    - Install dependencies: uv sync --group examples

Features:
    - Interactive chat interface with the Agent
    - Switch between multiple LLM providers and models
    - Configurable streaming mode (toggle on/off)
    - Web search integration
    - Full debug view showing agent steps, tool execution, status
    - Conversation history viewer
    - Expandable reasoning and tool output sections
"""

import os
from typing import Any

import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
import langfuse.openai
from thinllm import (
    Agent,
    AgentRequest,
    AgentResponse,
    LLMConfig,
    Provider,
    SystemMessage,
    tool,
)

# Import rendering utilities
from streamlit_utils import render_response, render_user_message

# The location of my CA File (optional - for proxy debugging)
cert_file = os.path.expanduser("~/.mitmproxy/mitmproxy-ca-cert.pem")
if os.path.exists(cert_file):
    os.environ["REQUESTS_CA_BUNDLE"] = cert_file
    os.environ["SSL_CERT_FILE"] = cert_file
    os.environ["HTTPS_PROXY"] = "http://127.0.0.1:8080"

# Page configuration
st.set_page_config(
    page_title="ThinLLM Agent Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title
st.title("🤖 ThinLLM Agent Chat")
st.caption("Interactive chat with configurable streaming, web search, and full debug capabilities")


# Define example tools for testing
@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression.
    
    Args:
        expression: A mathematical expression to evaluate (e.g., "2 + 2", "10 * 5")
        
    Returns:
        The result of the calculation
    """
    try:
        # Use eval safely with limited builtins
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"


@tool
def get_current_time() -> str:
    """
    Get the current date and time.
    
    Returns:
        Current date and time in a readable format
    """
    from datetime import datetime
    now = datetime.now()
    return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}"


@tool
def reverse_text(text: str) -> str:
    """
    Reverse the given text string.
    
    Args:
        text: The text to reverse
        
    Returns:
        The reversed text
    """
    return text[::-1]


@tool
def count_words(text: str) -> str:
    """
    Count the number of words in the given text.
    
    Args:
        text: The text to analyze
        
    Returns:
        Word count and character count
    """
    words = text.split()
    return f"Word count: {len(words)}, Character count: {len(text)}"


def get_available_configs() -> dict[str, LLMConfig]:
    """Get available LLM configurations based on API keys."""
    configs = {}
    
    # OpenAI configs
    if os.getenv("OPENAI_API_KEY"):
        configs["OpenAI GPT-5.2"] = LLMConfig(
            provider=Provider.OPENAI,
            model_id="gpt-5.2",
            model_args={"temperature": 0.7},
        )
        configs["OpenAI GPT-5.1"] = LLMConfig(
            provider=Provider.OPENAI,
            model_id="gpt-5.1",
            model_args={"temperature": 0.7},
        )
        configs["OpenAI GPT-5 (Reasoning)"] = LLMConfig(
            provider=Provider.OPENAI,
            model_id="gpt-5",
            model_args={
                "reasoning": {
                    "effort": "medium",
                    "summary": "detailed",
                },
            },
        )
    
    # Anthropic configs
    if os.getenv("ANTHROPIC_API_KEY"):
        configs["Anthropic Claude Sonnet 4.5"] = LLMConfig(
            provider=Provider.ANTHROPIC,
            model_id="claude-sonnet-4-5",
            model_args={"temperature": 0.7, "max_tokens": 4096},
        )
        configs["Anthropic Claude Opus 4.5"] = LLMConfig(
            provider=Provider.ANTHROPIC,
            model_id="claude-opus-4-5",
            model_args={"temperature": 0.7, "max_tokens": 4096},
        )
        configs["Anthropic Claude Sonnet 4.5 (Thinking)"] = LLMConfig(
            provider=Provider.ANTHROPIC,
            model_id="claude-sonnet-4-5",
            model_args={
                "temperature": 1.0,  # Required for thinking mode
                "max_tokens": 4096,
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": 2000,
                },
            },
        )
    
    if not configs:
        st.error("⚠️ No API keys found! Please set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable.")
        st.stop()
    
    return configs


def create_agent(config: LLMConfig) -> Agent:
    """Create an agent with the given config and default tools."""
    # Adjust system message based on whether web search is available
    has_web_search = config.provider == Provider.OPENAI
    
    if has_web_search:
        system_content = (
            "You are a helpful AI assistant with access to web search and various tools. "
            "Use web search to find current information when needed. "
            "Use the calculator for mathematical operations. "
            "Always provide clear, accurate, and helpful responses."
        )
    else:
        system_content = (
            "You are a helpful AI assistant with access to various tools. "
            "Use the calculator for mathematical operations. "
            "Always provide clear, accurate, and helpful responses."
        )
    
    system_message = SystemMessage(content=system_content)
    
    # Build tools list - only add web_search for OpenAI
    tools: list[Any] = []
    
    if has_web_search:
        tools.append({"type": "web_search"})
    
    tools.extend([
        calculator,
        get_current_time,
        reverse_text,
        count_words,
    ])
    
    return Agent(
        llm_config=config,
        messages=[system_message],
        tools=tools,
        max_iterations=10,
    )


def initialize_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Get available configs
    available_configs = get_available_configs()
    
    if "selected_config_name" not in st.session_state:
        st.session_state.selected_config_name = list(available_configs.keys())[0]
    
    if "agent" not in st.session_state:
        config = available_configs[st.session_state.selected_config_name]
        st.session_state.agent = create_agent(config)
    
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    
    if "latest_response" not in st.session_state:
        st.session_state.latest_response = None
    
    # Settings with defaults
    if "enable_streaming" not in st.session_state:
        st.session_state.enable_streaming = True
    
    if "expand_reasoning" not in st.session_state:
        st.session_state.expand_reasoning = False
    
    if "expand_tools" not in st.session_state:
        st.session_state.expand_tools = False


def render_debug_sidebar() -> None:
    """Render the debug sidebar with agent execution details and settings."""
    with st.sidebar:
        st.header("🤖 Model Selection")
        
        # Get available configs
        available_configs = get_available_configs()
        
        # Model selector
        selected_config_name = st.selectbox(
            "Select LLM Configuration",
            options=list(available_configs.keys()),
            index=list(available_configs.keys()).index(st.session_state.selected_config_name),
            help="Choose which LLM provider and model to use",
        )
        
        # If config changed, recreate agent
        if selected_config_name != st.session_state.selected_config_name:
            st.session_state.selected_config_name = selected_config_name
            config = available_configs[selected_config_name]
            st.session_state.agent = create_agent(config)
            st.session_state.messages = []
            st.session_state.conversation_history = []
            st.session_state.latest_response = None
            st.rerun()
        
        # Display current config details
        current_config = available_configs[selected_config_name]
        st.caption(f"**Provider:** {current_config.provider.value}")
        st.caption(f"**Model:** {current_config.model_id}")
        
        st.divider()
        
        st.header("⚙️ Settings")
        
        # Streaming toggle
        st.toggle(
            "Enable Streaming",
            value=st.session_state.enable_streaming,
            key="enable_streaming",
            help="Stream responses in real-time vs. wait for completion"
        )
        
        # Expander settings
        st.toggle(
            "Expand Reasoning",
            value=st.session_state.expand_reasoning,
            key="expand_reasoning",
            help="Auto-expand reasoning/thought process sections"
        )
        
        st.toggle(
            "Expand Tool Outputs",
            value=st.session_state.expand_tools,
            key="expand_tools",
            help="Auto-expand tool execution details"
        )
        
        st.divider()
        
        st.header("🔍 Debug Information")
        
        if st.session_state.latest_response:
            response: AgentResponse = st.session_state.latest_response
            
            # Status and iterations
            st.subheader("📊 Execution Status")
            status_color = "🟢" if response.status == "succeeded" else "🔴"
            st.write(f"**Status:** {status_color} {response.status.upper()}")
            st.write(f"**Iterations:** {response.iterations}")
            st.write(f"**Steps:** {len(response.steps)}")
            
            st.divider()
            
            # Agent steps
            if response.steps:
                st.subheader("🔄 Agent Steps")
                
                for idx, step in enumerate(response.steps, 1):
                    with st.expander(f"Step {idx} - {step.status}", expanded=False):
                        st.write(f"**Status:** {step.status}")
                        
                        # Tool results
                        if step.tool_results:
                            st.write("**Tool Executions:**")
                            for tool_result in step.tool_results:
                                from thinllm.messages import ToolOutputStatus
                                status_icon = "✅" if tool_result.status == ToolOutputStatus.SUCCESS else "❌"
                                st.write(f"{status_icon} **{tool_result.name}**")
                                
                                with st.container():
                                    st.write("*Input:*")
                                    st.json(tool_result.input)
                                    
                                    if tool_result.output:
                                        st.write("*Output:*")
                                        st.code(tool_result.output[:500] + ("..." if len(tool_result.output) > 500 else ""))
                                    
                                    if tool_result.metadata:
                                        st.write("*Metadata:*")
                                        st.json(tool_result.metadata)
            
            st.divider()
            
            # Conversation history
            with st.expander("💬 Full Message History", expanded=False):
                st.write(f"Total messages: {len(st.session_state.agent.messages)}")
                for idx, msg in enumerate(st.session_state.agent.messages):
                    st.write(f"**{idx + 1}. {msg.role.upper()}**")
                    if isinstance(msg.content, str):
                        st.text(msg.content[:200] + ("..." if len(msg.content) > 200 else ""))
                    else:
                        st.write(f"Blocks: {len(msg.content)}")
                    st.divider()
        else:
            st.info("No conversation yet. Start chatting to see debug information!")
        
        # Clear conversation button
        st.divider()
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            # Reset agent with current config
            available_configs = get_available_configs()
            config = available_configs[st.session_state.selected_config_name]
            st.session_state.agent = create_agent(config)
            st.session_state.messages = []
            st.session_state.conversation_history = []
            st.session_state.latest_response = None
            st.rerun()


def process_request(agent: Agent, request: AgentRequest, stream: bool) -> AgentResponse:
    """Process agent request with configurable streaming.
    
    Args:
        agent: The agent instance
        request: User request
        stream: Whether to stream the response
        
    Returns:
        Final AgentResponse
    """
    if stream:
        # Streaming mode: use st.empty() container pattern
        container = st.empty()
        response = None
        for response in agent.ask(request, stream=True):
            with container.container():
                render_response(response)
        return response  # type: ignore[return-value]
    else:
        # Non-streaming mode: show spinner, then render final response
        with st.spinner("Thinking..."):
            response = agent.ask(request, stream=False)
        render_response(response)
        return response


def main() -> None:
    """Main application logic."""
    initialize_session_state()
    render_debug_sidebar()
    
    # Display chat messages
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        
        if role == "user":
            render_user_message(content)
        else:
            # Assistant message
            with st.chat_message("assistant"):
                # For stored messages, render the final response
                if isinstance(content, AgentResponse):
                    render_response(content)
                else:
                    # Fallback for any other format
                    st.markdown(str(content))
    
    # Chat input
    if prompt := st.chat_input("Ask me anything... (I can search the web!)"):
        # Display user message
        render_user_message(prompt)
        
        # Add to message history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Process with agent
        with st.chat_message("assistant"):
            try:
                # Create request
                request = AgentRequest(content=prompt)
                
                # Get response from agent (streaming or non-streaming)
                response: AgentResponse = process_request(
                    st.session_state.agent,
                    request,
                    stream=st.session_state.enable_streaming
                )
                
                # Store response for debug view
                st.session_state.latest_response = response
                st.session_state.conversation_history.append({
                    "request": prompt,
                    "response": response,
                })
                
                # Add to message history (store the full response object)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                })
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.exception(e)


if __name__ == "__main__":
    main()

