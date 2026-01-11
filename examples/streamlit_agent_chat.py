"""Streamlit Chat Application for ThinLLM Agent.

This application demonstrates the Agent class with full debug capabilities,
including tool execution details, conversation history, and web search integration.

Usage:
    streamlit run examples/streamlit_agent_chat.py

Requirements:
    - Set OPENAI_API_KEY environment variable (or add to .env file)
    - Install dependencies: uv sync --group examples

Features:
    - Interactive chat interface with the Agent
    - Web search integration
    - Full debug view showing agent steps, tool execution, status
    - Conversation history viewer
"""

import os
from typing import Any

import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from thinllm import (
    Agent,
    AgentRequest,
    AgentResponse,
    AgentStep,
    LLMConfig,
    OutputTextBlock,
    Provider,
    SystemMessage,
    ToolCallContent,
    ToolOutputStatus,
)
from thinllm.messages import ReasoningContent, WebSearchCallContent


# The location of my CA File
cert_file = os.path.expanduser("~/.mitmproxy/mitmproxy-ca-cert.pem")
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
st.caption("Interactive chat with web search and full debug capabilities")


def initialize_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "agent" not in st.session_state:
        # Check for API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("⚠️ OPENAI_API_KEY environment variable not set!")
            st.stop()
        
        # Create agent with web search tool
        config = LLMConfig(
            provider=Provider.OPENAI,
            model_id="gpt-5.2",
            model_args={
            "reasoning": {
                "effort": "medium",
                "summary": "detailed",
            },
        },
        )
        
        system_message = SystemMessage(
            content=(
                "You are a helpful AI assistant with access to web search. "
                "Use web search to find current information when needed. "
                "Always provide clear, accurate, and helpful responses."
            )
        )
        
        st.session_state.agent = Agent(
            llm_config=config,
            messages=[system_message],
            tools=[{"type": "web_search"}],
            max_iterations=10,
        )
    
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    
    if "latest_response" not in st.session_state:
        st.session_state.latest_response = None


def render_message_content(content: str | list[Any]) -> None:
    """Render message content, handling both string and list formats."""
    if isinstance(content, str):
        st.markdown(content)
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, OutputTextBlock):
                st.markdown(block.text)
            elif isinstance(block, ToolCallContent):
                with st.expander(f"🔧 Tool Call: {block.name}", expanded=False):
                    st.json(block.input)
            elif isinstance(block, WebSearchCallContent):
                with st.expander(f"🔍 Web Search: {block.query}", expanded=False):
                    st.write(f"**Query:** {block.query}")
                    if block.sources:
                        st.write("**Sources:**")
                        for source in block.sources:
                            st.write(f"- {source}")
            elif isinstance(block, ReasoningContent):
                with st.expander("💭 Reasoning", expanded=False):
                    if block.summaries:
                        st.write("**Summaries:**")
                        for summary in block.summaries:
                            st.write(f"- {summary}")
                    if block.contents:
                        st.write("**Detailed Reasoning:**")
                        for reasoning in block.contents:
                            st.write(reasoning)


def render_debug_sidebar() -> None:
    """Render the debug sidebar with agent execution details."""
    with st.sidebar:
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
            # Reset agent with new instance
            config = LLMConfig(
                provider=Provider.OPENAI,
                model_id="gpt-5.1",
                model_args={"temperature": 0.7},
            )
            
            system_message = SystemMessage(
                content=(
                    "You are a helpful AI assistant with access to web search. "
                    "Use web search to find current information when needed. "
                    "Always provide clear, accurate, and helpful responses."
                )
            )
            
            st.session_state.agent = Agent(
                llm_config=config,
                messages=[system_message],
                tools=[{"type": "web_search"}],
                max_iterations=10,
            )
            st.session_state.messages = []
            st.session_state.conversation_history = []
            st.session_state.latest_response = None
            st.rerun()


def main() -> None:
    """Main application logic."""
    initialize_session_state()
    render_debug_sidebar()
    
    # Display chat messages
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        
        with st.chat_message(role):
            render_message_content(content)
    
    # Chat input
    if prompt := st.chat_input("Ask me anything... (I can search the web!)"):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Add to message history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Process with agent
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Create request
                    request = AgentRequest(content=prompt)
                    
                    # Get response from agent
                    response: AgentResponse = st.session_state.agent.ask(request)
                    
                    # Store response for debug view
                    st.session_state.latest_response = response
                    st.session_state.conversation_history.append({
                        "request": prompt,
                        "response": response,
                    })
                    
                    # Extract and display final message
                    final_content = response.final_message.content
                    render_message_content(final_content)
                    
                    # Add to message history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": final_content,
                    })
                    
                    # Show success message with step count
                    if response.steps:
                        st.caption(f"✨ Completed in {response.iterations} iterations with {len(response.steps)} tool executions")
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.exception(e)


if __name__ == "__main__":
    main()

