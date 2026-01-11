# LLM Config Research: Implementation Options

## Current State Analysis

The current `LLMConfig` is minimal:

```python
class LLMConfig(BaseModel):
    provider: Provider
    model_id: str
    model_args: dict[str, Any]  # Untyped pass-through to providers
```

**Problems:**
- No type safety for common parameters
- No validation of parameter ranges
- No standardized thinking/tool_choice configuration
- Provider-specific args mixed with common args
- No credential management
- No preset system

---

## Target Parameters Schema

```
params:
    temperature: float          # 0.0-2.0 typically
    top_k: int                  # Token selection pool size
    top_p: float                # 0.0-1.0 nucleus sampling
    max_output_tokens: int      # Max response length
    stop_sequences: list[str]   # Stop generation triggers
    
    thinking:
        enabled: bool
        thinking_level: none|minimal|low|high|xhigh
        thinking_budget: int    # Token budget for thinking
    
    tool_choice:
        type: None|Auto|Function|AllowList|Any
        disable_parallel_tool_call: bool
        name: str               # For type=Function
        allowed_tools: list[str]  # For type=AllowList
```

---

## Provider Parameter Mapping Reference

| ThinLLM Param | OpenAI | Anthropic | Gemini |
|---------------|--------|-----------|--------|
| `temperature` | `temperature` | `temperature` | `temperature` |
| `top_k` | N/A | `top_k` | `top_k` |
| `top_p` | `top_p` | `top_p` | `top_p` |
| `max_output_tokens` | `max_output_tokens` | `max_tokens` | `max_output_tokens` |
| `stop_sequences` | `stop` | `stop_sequences` | `stop_sequences` |
| `thinking.budget` | `reasoning.budget_tokens` | `thinking.budget_tokens` | `thinking_config.thinking_budget` |
| `tool_choice.type=Auto` | `tool_choice="auto"` | `tool_choice={"type":"auto"}` | `tool_config.mode=AUTO` |
| `tool_choice.type=None` | `tool_choice="none"` | `tool_choice={"type":"none"}` | `tool_config.mode=NONE` |
| `tool_choice.type=Function` | `tool_choice={"type":"function","name":X}` | `tool_choice={"type":"tool","name":X}` | `tool_config.mode=ANY` + filter |

---

## Option 1: Typed Nested Pydantic Models (Recommended)

### Design

```python
from typing import Literal
from pydantic import BaseModel, Field, field_validator

class ThinkingLevel(StrEnum):
    NONE = "none"
    MINIMAL = "minimal"
    LOW = "low"
    HIGH = "high"
    XHIGH = "xhigh"

class ThinkingConfig(BaseModel):
    """Configuration for model thinking/reasoning capabilities."""
    enabled: bool = False
    thinking_level: ThinkingLevel = ThinkingLevel.NONE
    thinking_budget: int | None = None  # Provider-specific token budget
    
    @field_validator("thinking_budget")
    @classmethod
    def validate_budget(cls, v, info):
        if v is not None and v < 0:
            raise ValueError("thinking_budget must be non-negative")
        return v

class ToolChoiceType(StrEnum):
    NONE = "none"          # Don't use tools
    AUTO = "auto"          # Model decides
    FUNCTION = "function"  # Force specific function
    ALLOWLIST = "allowlist"  # Limit to subset
    ANY = "any"            # Must use at least one tool

class ToolChoiceConfig(BaseModel):
    """Configuration for tool/function calling behavior."""
    type: ToolChoiceType = ToolChoiceType.AUTO
    disable_parallel_tool_calls: bool = False
    name: str | None = None  # For type=FUNCTION
    allowed_tools: list[str] | None = None  # For type=ALLOWLIST
    
    @model_validator(mode="after")
    def validate_tool_choice(self):
        if self.type == ToolChoiceType.FUNCTION and not self.name:
            raise ValueError("name is required when type=FUNCTION")
        if self.type == ToolChoiceType.ALLOWLIST and not self.allowed_tools:
            raise ValueError("allowed_tools is required when type=ALLOWLIST")
        return self

class ModelParams(BaseModel):
    """Standardized model parameters across all providers."""
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_k: int | None = Field(default=None, ge=1)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    max_output_tokens: int | None = Field(default=None, ge=1)
    stop_sequences: list[str] | None = None
    
    thinking: ThinkingConfig = Field(default_factory=ThinkingConfig)
    tool_choice: ToolChoiceConfig = Field(default_factory=ToolChoiceConfig)

class Credentials(BaseModel):
    """Provider credentials container."""
    api_key: str | None = None
    api_base: str | None = None  # Custom endpoint
    organization: str | None = None  # OpenAI org
    project_id: str | None = None  # GCP project
    region: str | None = None  # AWS/GCP region
    
    model_config = ConfigDict(extra="allow")  # Allow provider-specific fields

class LLMConfig(BaseModel):
    """
    Configuration for LLM service interactions.
    """
    provider: Provider
    model_id: str
    
    # Typed parameters with validation
    params: ModelParams = Field(default_factory=ModelParams)
    
    # Provider-specific overrides (higher precedence)
    provider_params: dict[str, Any] = Field(default_factory=dict)
    
    # Credentials (optional - falls back to env vars)
    credentials: Credentials | None = None
    
    def get_provider_params(self) -> dict[str, Any]:
        """
        Merge params with provider_params.
        Provider-specific params take precedence.
        """
        base = self._translate_params_for_provider()
        base.update(self.provider_params)
        return base
    
    def _translate_params_for_provider(self) -> dict[str, Any]:
        """Translate ModelParams to provider-specific format."""
        # Implementation per provider
        ...
```

### Usage Example

```python
# Basic usage
config = LLMConfig(
    provider=Provider.GEMINI,
    model_id="gemini-2.5-flash",
    params=ModelParams(
        temperature=0.7,
        max_output_tokens=4096,
        thinking=ThinkingConfig(enabled=True, thinking_budget=8000),
    ),
)

# With provider-specific override
config = LLMConfig(
    provider=Provider.ANTHROPIC,
    model_id="claude-sonnet-4-5",
    params=ModelParams(temperature=0.5),
    provider_params={"metadata": {"user_id": "123"}},  # Anthropic-specific
)

# With credentials
config = LLMConfig(
    provider=Provider.OPENAI,
    model_id="gpt-4o",
    credentials=Credentials(api_key="sk-..."),
)
```

### Pros
- ✅ Full type safety and IDE autocomplete
- ✅ Validation at config creation time
- ✅ Clear separation of concerns
- ✅ Self-documenting API
- ✅ Easy to extend with new parameters
- ✅ Provider params allow escape hatch

### Cons
- ❌ More verbose for simple cases
- ❌ Breaking change from current API
- ❌ Nested models can be tedious to construct

---

## Option 2: Flat Typed Fields with Optional Dict Escape

### Design

```python
class LLMConfig(BaseModel):
    """Configuration for LLM service interactions."""
    provider: Provider
    model_id: str
    
    # Core params - flat structure
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_k: int | None = Field(default=None, ge=1)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    max_output_tokens: int | None = Field(default=None, ge=1)
    stop_sequences: list[str] | None = None
    
    # Thinking params (flat)
    thinking_enabled: bool = False
    thinking_level: ThinkingLevel | None = None
    thinking_budget: int | None = None
    
    # Tool choice params (flat)
    tool_choice_type: ToolChoiceType = ToolChoiceType.AUTO
    tool_choice_name: str | None = None
    tool_choice_allowed: list[str] | None = None
    disable_parallel_tool_calls: bool = False
    
    # Escape hatches
    provider_params: dict[str, Any] = Field(default_factory=dict)
    credentials: dict[str, str] = Field(default_factory=dict)
```

### Usage Example

```python
config = LLMConfig(
    provider=Provider.GEMINI,
    model_id="gemini-2.5-flash",
    temperature=0.7,
    thinking_enabled=True,
    thinking_budget=8000,
)
```

### Pros
- ✅ Simple, flat structure
- ✅ Less verbose for common use cases
- ✅ All params visible at top level
- ✅ Easy migration path

### Cons
- ❌ Many fields at top level (cluttered)
- ❌ Related params not grouped (thinking_*, tool_choice_*)
- ❌ Harder to validate cross-field constraints
- ❌ Harder to add new param groups without pollution

---

## Option 3: Builder Pattern with Fluent Interface

### Design

```python
class LLMConfigBuilder:
    """Fluent builder for LLMConfig."""
    
    def __init__(self, provider: Provider, model_id: str):
        self._provider = provider
        self._model_id = model_id
        self._params = {}
        self._thinking = {}
        self._tool_choice = {}
        self._credentials = {}
        self._provider_params = {}
    
    def temperature(self, value: float) -> "LLMConfigBuilder":
        self._params["temperature"] = value
        return self
    
    def max_tokens(self, value: int) -> "LLMConfigBuilder":
        self._params["max_output_tokens"] = value
        return self
    
    def with_thinking(
        self, 
        enabled: bool = True, 
        budget: int | None = None,
        level: ThinkingLevel = ThinkingLevel.HIGH
    ) -> "LLMConfigBuilder":
        self._thinking = {"enabled": enabled, "budget": budget, "level": level}
        return self
    
    def with_tool_choice(
        self, 
        type: ToolChoiceType,
        name: str | None = None,
        allowed: list[str] | None = None
    ) -> "LLMConfigBuilder":
        self._tool_choice = {"type": type, "name": name, "allowed": allowed}
        return self
    
    def with_credentials(self, api_key: str, **kwargs) -> "LLMConfigBuilder":
        self._credentials = {"api_key": api_key, **kwargs}
        return self
    
    def with_provider_params(self, **kwargs) -> "LLMConfigBuilder":
        self._provider_params.update(kwargs)
        return self
    
    def build(self) -> LLMConfig:
        return LLMConfig(
            provider=self._provider,
            model_id=self._model_id,
            params=ModelParams(**self._params, thinking=ThinkingConfig(**self._thinking)),
            provider_params=self._provider_params,
            credentials=Credentials(**self._credentials) if self._credentials else None,
        )

# Convenience factory
def config(provider: Provider, model_id: str) -> LLMConfigBuilder:
    return LLMConfigBuilder(provider, model_id)
```

### Usage Example

```python
# Fluent style
cfg = (
    config(Provider.GEMINI, "gemini-2.5-flash")
    .temperature(0.7)
    .max_tokens(4096)
    .with_thinking(enabled=True, budget=8000)
    .build()
)

# Or combine with direct construction
cfg = (
    config(Provider.ANTHROPIC, "claude-sonnet-4-5")
    .temperature(0.5)
    .with_provider_params(metadata={"user_id": "123"})
    .with_credentials(api_key="sk-...")
    .build()
)
```

### Pros
- ✅ Highly readable, self-documenting
- ✅ Great IDE autocomplete experience
- ✅ Method chaining is intuitive
- ✅ Can validate incrementally
- ✅ Easy to add optional features

### Cons
- ❌ More boilerplate code to maintain
- ❌ Two ways to create configs (builder vs direct)
- ❌ Builder state is mutable (can be confusing)
- ❌ Serialization/deserialization more complex

---

## Option 4: Hybrid - Typed Fields + model_args for Backward Compatibility

### Design

```python
class LLMConfig(BaseModel):
    """Configuration with typed params and backward-compatible model_args."""
    provider: Provider
    model_id: str
    
    # New typed interface
    params: ModelParams | None = None
    credentials: Credentials | None = None
    
    # Legacy/provider-specific escape hatch (backward compatible)
    model_args: dict[str, Any] = Field(default_factory=dict)
    
    def get_effective_params(self) -> dict[str, Any]:
        """
        Merge params, with precedence:
        1. model_args (highest - provider specific override)
        2. params (translated to provider format)
        """
        result = {}
        if self.params:
            result = self._translate_params()
        result.update(self.model_args)
        return result
```

### Usage Example

```python
# New style (recommended)
config = LLMConfig(
    provider=Provider.GEMINI,
    model_id="gemini-2.5-flash",
    params=ModelParams(temperature=0.7),
)

# Legacy style (still works)
config = LLMConfig(
    provider=Provider.GEMINI,
    model_id="gemini-2.5-flash",
    model_args={"temperature": 0.7, "thinking_budget": 8000},
)

# Mixed (provider override)
config = LLMConfig(
    provider=Provider.GEMINI,
    model_id="gemini-2.5-flash",
    params=ModelParams(temperature=0.7),
    model_args={"some_new_gemini_param": True},  # Provider-specific
)
```

### Pros
- ✅ Backward compatible
- ✅ Gradual migration path
- ✅ Best of both worlds
- ✅ model_args still works as escape hatch

### Cons
- ❌ Two ways to set same thing (confusion)
- ❌ Need to document precedence rules
- ❌ Validation gaps when using model_args

---

## Preset System Design

### Core Concept

Presets are simply a `dict[str, LLMConfig]` mapping - no special model needed.

```python
# Type alias for clarity
Presets = dict[str, LLMConfig]

# Built-in presets (module-level constant)
BUILTIN_PRESETS: Presets = {
    "gemini-fast": LLMConfig(
        provider=Provider.GEMINI,
        model_id="gemini-2.0-flash",
        params=ModelParams(temperature=0.7, max_output_tokens=4096),
    ),
    "gemini-thinking": LLMConfig(
        provider=Provider.GEMINI,
        model_id="gemini-2.5-flash",
        params=ModelParams(
            temperature=1.0,
            thinking=ThinkingConfig(enabled=True, thinking_budget=10000),
        ),
    ),
    "claude-creative": LLMConfig(
        provider=Provider.ANTHROPIC,
        model_id="claude-sonnet-4-5",
        params=ModelParams(temperature=1.0, max_output_tokens=8192),
    ),
    "claude-precise": LLMConfig(
        provider=Provider.ANTHROPIC,
        model_id="claude-sonnet-4-5",
        params=ModelParams(temperature=0.0, max_output_tokens=4096),
    ),
    "gpt4o": LLMConfig(
        provider=Provider.OPENAI,
        model_id="gpt-4o",
        params=ModelParams(temperature=0.7),
    ),
    "gpt4o-mini": LLMConfig(
        provider=Provider.OPENAI,
        model_id="gpt-4o-mini",
        params=ModelParams(temperature=0.7),
    ),
}

# User presets (mutable, user can add their own)
_user_presets: Presets = {}


def register_preset(name: str, config: LLMConfig) -> None:
    """Register a user-defined preset."""
    _user_presets[name] = config


def get_preset(name: str) -> LLMConfig:
    """
    Get preset by name.
    User presets take precedence over built-in presets.
    """
    if name in _user_presets:
        return _user_presets[name]
    if name in BUILTIN_PRESETS:
        return BUILTIN_PRESETS[name]
    raise KeyError(f"Preset '{name}' not found. Available: {list_presets()}")


def list_presets() -> list[str]:
    """List all available preset names."""
    return list(set(_user_presets.keys()) | set(BUILTIN_PRESETS.keys()))
```

### Usage

```python
# Use built-in preset directly
config = get_preset("gemini-thinking")

# Register custom preset
register_preset(
    "my-agent",
    LLMConfig(
        provider=Provider.GEMINI,
        model_id="gemini-2.5-flash",
        params=ModelParams(
            temperature=0.3,
            thinking=ThinkingConfig(enabled=True, thinking_budget=20000),
        ),
    ),
)

# Now use custom preset
config = get_preset("my-agent")
```

### String Shorthand in llm() Function

```python
from typing import Union

PresetOrConfig = Union[str, LLMConfig]

def llm(
    config: PresetOrConfig,
    messages: list[MessageType],
    **kwargs
):
    """Accept either preset name or full config."""
    if isinstance(config, str):
        config = get_preset(config)
    
    # ... rest of implementation
```

### Usage

```python
# Direct preset string usage
response = llm("gemini-fast", messages)

# Or with full config
response = llm(my_config, messages)
```

---

## Credentials Management Design

### Option A: In-Config Credentials

```python
class Credentials(BaseModel):
    """Flexible credentials container."""
    api_key: str | None = None
    api_base: str | None = None
    organization: str | None = None
    project_id: str | None = None
    region: str | None = None
    
    model_config = ConfigDict(extra="allow")

# Usage in LLMConfig
config = LLMConfig(
    provider=Provider.OPENAI,
    model_id="gpt-4o",
    credentials=Credentials(
        api_key="sk-...",
        organization="org-..."
    ),
)
```

### Option B: Credential Provider Pattern

```python
from abc import ABC, abstractmethod
import os

class CredentialProvider(ABC):
    """Abstract credential provider."""
    
    @abstractmethod
    def get_credentials(self, provider: Provider) -> dict[str, str]:
        ...

class EnvCredentialProvider(CredentialProvider):
    """Load credentials from environment variables."""
    
    ENV_MAPPING = {
        Provider.OPENAI: {"api_key": "OPENAI_API_KEY"},
        Provider.ANTHROPIC: {"api_key": "ANTHROPIC_API_KEY"},
        Provider.GEMINI: {"api_key": "GOOGLE_API_KEY"},
    }
    
    def get_credentials(self, provider: Provider) -> dict[str, str]:
        mapping = self.ENV_MAPPING.get(provider, {})
        return {
            key: os.environ.get(env_var, "")
            for key, env_var in mapping.items()
            if os.environ.get(env_var)
        }

class DictCredentialProvider(CredentialProvider):
    """Direct dict-based credentials."""
    
    def __init__(self, credentials: dict[str, str]):
        self._credentials = credentials
    
    def get_credentials(self, provider: Provider) -> dict[str, str]:
        return self._credentials

# In LLMConfig
class LLMConfig(BaseModel):
    ...
    credentials: Credentials | CredentialProvider | None = None
    
    def get_credentials(self) -> dict[str, str]:
        if self.credentials is None:
            return EnvCredentialProvider().get_credentials(self.provider)
        if isinstance(self.credentials, CredentialProvider):
            return self.credentials.get_credentials(self.provider)
        return self.credentials.model_dump(exclude_none=True)
```

### Credential Precedence (Recommended)

```
1. LLMConfig.credentials (explicit, highest priority)
2. Environment variables (fallback)
3. Provider SDK default (e.g., gcloud auth for Gemini)
```

---

## Provider Translation Implementation

### Translation Layer

```python
class ParamTranslator:
    """Translate ModelParams to provider-specific format."""
    
    @staticmethod
    def to_openai(params: ModelParams) -> dict[str, Any]:
        result = {}
        
        if params.temperature is not None:
            result["temperature"] = params.temperature
        if params.top_p is not None:
            result["top_p"] = params.top_p
        if params.max_output_tokens is not None:
            result["max_output_tokens"] = params.max_output_tokens
        if params.stop_sequences:
            result["stop"] = params.stop_sequences
        
        # Thinking -> Reasoning (OpenAI o1/o3 style)
        if params.thinking.enabled and params.thinking.thinking_budget:
            result["reasoning"] = {"budget_tokens": params.thinking.thinking_budget}
        
        # Tool choice
        if params.tool_choice.type == ToolChoiceType.NONE:
            result["tool_choice"] = "none"
        elif params.tool_choice.type == ToolChoiceType.AUTO:
            result["tool_choice"] = "auto"
        elif params.tool_choice.type == ToolChoiceType.FUNCTION:
            result["tool_choice"] = {
                "type": "function",
                "function": {"name": params.tool_choice.name}
            }
        elif params.tool_choice.type == ToolChoiceType.ANY:
            result["tool_choice"] = "required"
        
        if params.tool_choice.disable_parallel_tool_calls:
            result["parallel_tool_calls"] = False
        
        return result
    
    @staticmethod
    def to_anthropic(params: ModelParams) -> dict[str, Any]:
        result = {}
        
        if params.temperature is not None:
            result["temperature"] = params.temperature
        if params.top_k is not None:
            result["top_k"] = params.top_k
        if params.top_p is not None:
            result["top_p"] = params.top_p
        if params.max_output_tokens is not None:
            result["max_tokens"] = params.max_output_tokens  # Different name!
        if params.stop_sequences:
            result["stop_sequences"] = params.stop_sequences
        
        # Thinking (Anthropic extended thinking)
        if params.thinking.enabled and params.thinking.thinking_budget:
            result["thinking"] = {"budget_tokens": params.thinking.thinking_budget}
        
        # Tool choice
        if params.tool_choice.type == ToolChoiceType.NONE:
            result["tool_choice"] = {"type": "none"}
        elif params.tool_choice.type == ToolChoiceType.AUTO:
            result["tool_choice"] = {"type": "auto"}
        elif params.tool_choice.type == ToolChoiceType.FUNCTION:
            result["tool_choice"] = {
                "type": "tool",
                "name": params.tool_choice.name
            }
        elif params.tool_choice.type == ToolChoiceType.ANY:
            result["tool_choice"] = {"type": "any"}
        
        if params.tool_choice.disable_parallel_tool_calls:
            result["disable_parallel_tool_use"] = True
        
        return result
    
    @staticmethod
    def to_gemini(params: ModelParams) -> dict[str, Any]:
        result = {}
        
        if params.temperature is not None:
            result["temperature"] = params.temperature
        if params.top_k is not None:
            result["top_k"] = params.top_k
        if params.top_p is not None:
            result["top_p"] = params.top_p
        if params.max_output_tokens is not None:
            result["max_output_tokens"] = params.max_output_tokens
        if params.stop_sequences:
            result["stop_sequences"] = params.stop_sequences
        
        # Thinking (Gemini 2.5 thinking config)
        if params.thinking.enabled and params.thinking.thinking_budget:
            result["thinking_budget"] = params.thinking.thinking_budget
            result["include_thoughts"] = True
        
        # Tool config would need special handling with types.ToolConfig
        # Simplified here
        
        return result
```

---

## Final Recommendation

### Primary Recommendation: Option 1 (Nested Pydantic) + Option 4 Compatibility Layer

**Rationale:**
1. **Type Safety First**: Nested Pydantic models provide the best developer experience with full validation
2. **Backward Compatibility**: Keep `model_args` as escape hatch for gradual migration
3. **Provider Override**: `provider_params` allows provider-specific features without polluting core API
4. **Preset System**: Simple `dict[str, LLMConfig]` mapping with `register_preset()`/`get_preset()` functions
5. **Flexible Credentials**: Support both inline credentials and environment variable fallback

### Implementation Phases

```
Phase 1: Core Types
├── ModelParams, ThinkingConfig, ToolChoiceConfig
├── Credentials model
└── Updated LLMConfig with params + model_args compatibility

Phase 2: Translation Layer  
├── ParamTranslator with to_openai/anthropic/gemini
└── Update provider core.py files to use translator

Phase 3: Preset System
├── BUILTIN_PRESETS dict[str, LLMConfig]
├── register_preset() / get_preset() functions
└── String shorthand in llm() function

Phase 4: Credential Management
├── CredentialProvider abstraction
├── Environment variable fallback
└── Credential injection in provider clients
```

### Minimal Breaking Changes Strategy

```python
# Old code still works
config = LLMConfig(
    provider=Provider.GEMINI,
    model_id="gemini-2.5-flash",
    model_args={"temperature": 0.7, "thinking_budget": 8000},
)

# New code is cleaner and validated
config = LLMConfig(
    provider=Provider.GEMINI,
    model_id="gemini-2.5-flash",
    params=ModelParams(
        temperature=0.7,
        thinking=ThinkingConfig(enabled=True, thinking_budget=8000),
    ),
)

# Preset shorthand for common cases
response = llm("gemini-thinking", messages)
```

---

## Summary Comparison Matrix

| Criteria | Option 1 (Nested) | Option 2 (Flat) | Option 3 (Builder) | Option 4 (Hybrid) |
|----------|-------------------|-----------------|--------------------|--------------------|
| Type Safety | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Simplicity | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| Extensibility | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Backward Compat | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| IDE Experience | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Maintenance | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |

**Recommendation**: **Option 1 + Option 4 hybrid** - Nested Pydantic models for new code with `model_args` preserved for backward compatibility.
