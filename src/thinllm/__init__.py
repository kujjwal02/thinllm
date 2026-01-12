"""ThinLLM: A thin wrapper for LLM interactions with agent capabilities."""

from .agent import Agent, AgentRequest, AgentResponse, AgentStep, AgentToolResult, StepStatus
from .config import (
    BUILTIN_PRESET_NAMES,
    Credentials,
    LLMConfig,
    LLMConfigOrPreset,
    ModelParams,
    Provider,
    ThinkingConfig,
    ThinkingLevel,
    ToolChoiceConfig,
    ToolChoiceType,
    get_preset,
    list_presets,
    register_preset,
)
from .core import llm
from .messages import (
    AIMessage,
    AnthropicCacheControl,
    ContentExtra,
    ImageDetail,
    InputImageBlock,
    InputTextBlock,
    OutputTextBlock,
    SystemMessage,
    ToolCallContent,
    ToolCallStatus,
    ToolOutputStatus,
    ToolResultContent,
    UserMessage,
)
from .utils import get_tool_result, normalize_tools
from .tools import Tool, ToolKit, tool

__all__ = [
    # Core
    "llm",
    "LLMConfig",
    "LLMConfigOrPreset",
    "Provider",
    # Config - Model Parameters
    "ModelParams",
    "ThinkingConfig",
    "ThinkingLevel",
    "ToolChoiceConfig",
    "ToolChoiceType",
    "Credentials",
    # Config - Presets
    "register_preset",
    "get_preset",
    "list_presets",
    "BUILTIN_PRESET_NAMES",
    # Messages
    "SystemMessage",
    "UserMessage",
    "AIMessage",
    "InputTextBlock",
    "OutputTextBlock",
    "InputImageBlock",
    "ImageDetail",
    "ContentExtra",
    "AnthropicCacheControl",
    "ToolCallContent",
    "ToolCallStatus",
    "ToolResultContent",
    "ToolOutputStatus",
    "get_tool_result",
    # Tools
    "Tool",
    "ToolKit",
    "tool",
    # Agent
    "Agent",
    "AgentRequest",
    "AgentResponse",
    "AgentStep",
    "AgentToolResult",
    "StepStatus",
    # Utils
    "get_tool_result",
    "normalize_tools",
]
