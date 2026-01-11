"""Streamlit rendering utilities for ThinLLM Agent responses.

This module provides reusable rendering functions for displaying Agent responses,
steps, tool results, and various content types in Streamlit applications.
"""

from typing import Any

import streamlit as st

from thinllm import (
    AgentResponse,
    AgentStep,
    AgentToolResult,
    OutputTextBlock,
    ToolOutputStatus,
)
from thinllm.agent import StepStatus
from thinllm.messages import (
    ContentBlock,
    InputImageBlock,
    InputTextBlock,
    ReasoningContent,
    ToolCallContent,
    WebSearchCallContent,
)


def render_user_message(content: str | list[ContentBlock]) -> None:
    """Render user input in chat message.
    
    Args:
        content: User message content (string or list of content blocks)
    """
    with st.chat_message("user"):
        if isinstance(content, str):
            st.markdown(content)
        else:
            for block in content:
                if isinstance(block, InputTextBlock):
                    st.markdown(block.text)
                elif isinstance(block, InputImageBlock):
                    st.image(block.image_url or block.image_bytes)


def render_reasoning(reasoning: ReasoningContent) -> None:
    """Render reasoning in collapsible expander.
    
    Args:
        reasoning: Reasoning content block with summaries and detailed contents
    """
    if not reasoning.summaries and not reasoning.contents:
        return
    
    with st.expander(
        "Thought Process",
        icon="🧠",
        expanded=st.session_state.get("expand_reasoning", False)
    ):
        # Summaries first (more concise)
        if reasoning.summaries:
            for summary in reasoning.summaries:
                st.info(summary)
        
        # Detailed contents (if available and different from summaries)
        if reasoning.contents:
            st.divider()
            for content in reasoning.contents:
                st.caption(content)


def render_web_search(search: WebSearchCallContent) -> None:
    """Render web search call with sources and status.
    
    Args:
        search: Web search call content with query, sources, and status
    """
    from thinllm.messages import ToolCallStatus
    
    # Determine icon and status text based on status
    if search.status == ToolCallStatus.COMPLETE:
        icon = "✅"
        status_text = "Completed"
    else:
        icon = "⏳"
        status_text = "Searching"
    
    # Build title
    if search.query:
        title = f"Web Search: {search.query} ({status_text})"
    else:
        title = f"Web Search ({status_text})"
    
    with st.expander(title, icon=icon):
        if search.query:
            st.markdown(f"**Query:** {search.query}")
        
        if search.status == ToolCallStatus.COMPLETE and search.sources:
            st.markdown("**Sources:**")
            for source in search.sources:
                st.markdown(f"- [{source}]({source})")
        elif search.status == ToolCallStatus.PENDING:
            st.caption("Searching...")


def render_tool_call(tool_call: ToolCallContent) -> None:
    """Show tool call (in progress or completed).
    
    Args:
        tool_call: Tool call content with name, input, and status
    """
    from thinllm.messages import ToolCallStatus
    
    # Determine icon based on status
    if tool_call.status == ToolCallStatus.COMPLETE:
        icon = "✅"
        status_text = "Completed"
    else:
        icon = "⏳"
        status_text = "In Progress"
    
    with st.expander(f"{tool_call.name} ({status_text})", icon=icon, expanded=True):
        st.markdown("**Input**")
        if tool_call.input:
            st.json(tool_call.input)
        else:
            st.code(tool_call.raw_input or "Parsing arguments...")
        
        if tool_call.status != ToolCallStatus.COMPLETE:
            st.markdown("**Output**")
            st.caption("Executing...")


def render_tool_result(result: AgentToolResult) -> None:
    """Render tool execution result with status indicator.
    
    Args:
        result: Agent tool result with input, output, status, and metadata
    """
    # Status icon mapping
    status_icons = {
        ToolOutputStatus.SUCCESS: "✅",
        ToolOutputStatus.FAILURE: "❌",
        ToolOutputStatus.IN_PROGRESS: "⏳",
        ToolOutputStatus.PENDING: "⏳",
    }
    icon = status_icons.get(result.status, "⏳")
    
    with st.expander(
        result.name,
        icon=icon,
        expanded=st.session_state.get("expand_tools", False)
    ):
        # Input section
        if result.input:
            st.markdown("**Input**")
            st.json(result.input)
        
        # Output section
        st.markdown("**Output**")
        if result.output:
            # Truncate long outputs
            output = result.output
            if len(output) > 1000:
                st.code(output[:1000] + "\n... (truncated)")
            else:
                st.code(output)
        elif result.status == ToolOutputStatus.IN_PROGRESS:
            st.caption("Executing...")
        
        # Metadata section (if present)
        if result.metadata:
            st.markdown("**Metadata**")
            st.json(result.metadata)


def render_ai_message_content(content: str | list[ContentBlock]) -> None:
    """Render AI message content in original order.
    
    Renders content blocks in the same order they appear in the content list.
    
    Args:
        content: AI message content (string or list of content blocks)
    """
    if isinstance(content, str):
        st.markdown(content)
        return
    
    # Render content blocks in original order
    for block in content:
        match block:
            case ReasoningContent():
                render_reasoning(block)
            case OutputTextBlock():
                st.markdown(block.text)
            case WebSearchCallContent():
                render_web_search(block)
            case ToolCallContent():
                render_tool_call(block)


def render_step(step: AgentStep, step_number: int) -> None:
    """Render a single agent step.
    
    Renders content in original order, preserving the sequence from AI message.
    Then renders tool results.
    
    Args:
        step: Agent step with AI message and tool results
        step_number: Step number for display (1-indexed)
    """
    # Extract content from AI message
    content = step.ai_message.content
    
    # Render AI message content in original order
    if isinstance(content, list):
        for block in content:
            match block:
                case ReasoningContent():
                    render_reasoning(block)
                case OutputTextBlock():
                    st.markdown(block.text)
                case WebSearchCallContent():
                    render_web_search(block)
                case ToolCallContent():
                    # Skip tool calls here - they'll be rendered via tool_results
                    pass
    elif isinstance(content, str):
        st.markdown(content)
    
    # Render tool results with status indicators
    for result in step.tool_results:
        render_tool_result(result)


def render_status_badge(response: AgentResponse) -> None:
    """Show overall execution status at bottom of response.
    
    Only shows badge for completed responses (SUCCEEDED or FAILED).
    
    Args:
        response: Agent response with status and iteration info
    """
    if response.status == StepStatus.SUCCEEDED:
        step_count = len(response.steps)
        if step_count > 0:
            st.caption(f"✨ Completed in {response.iterations} iterations with {step_count} tool executions")
    elif response.status == StepStatus.FAILED:
        st.caption(f"⚠️ Failed after {response.iterations} iterations")


def render_response(response: AgentResponse) -> None:
    """Render complete AgentResponse (partial during streaming, complete otherwise).
    
    Order: Steps first (chronological), then final message.
    Each step shows reasoning -> text -> tool results.
    
    Args:
        response: Agent response with steps and final message
    """
    # 1. Render all steps (tool execution iterations)
    for i, step in enumerate(response.steps):
        render_step(step, step_number=i + 1)
    
    # 2. Render final message (if present - None during streaming with tools)
    if response.final_message:
        render_ai_message_content(response.final_message.content)
    
    # 3. Render status badge
    render_status_badge(response)

