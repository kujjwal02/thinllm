"""Tool system for LLM function calling and agent capabilities."""

import inspect
import warnings
from collections.abc import Callable
from typing import Any, Generic, ParamSpec, Protocol, TypeAlias, TypeVar, overload

from pydantic import BaseModel, ConfigDict, Field, create_model
from pydantic.fields import FieldInfo


def get_argschema_from_function(func: Callable[..., Any]) -> type[BaseModel]:
    """
    Extract Pydantic schema from a function's signature.

    This function introspects a function's parameters and creates a Pydantic model
    that represents its argument schema, suitable for tool definitions.

    Args:
        func: The function to extract schema from

    Returns:
        A Pydantic BaseModel subclass representing the function's arguments

    Example:
        >>> def add(a: int, b: int) -> int:
        ...     return a + b
        >>> schema = get_argschema_from_function(add)
        >>> schema.model_json_schema()
        {'properties': {'a': {'type': 'integer'}, 'b': {'type': 'integer'}}, ...}
    """
    from pydantic import PydanticDeprecationWarning, validate_arguments

    sig = inspect.signature(func)
    fields = list(sig.parameters.keys())
    in_class = bool(func.__qualname__ and "." in func.__qualname__)
    if in_class and fields and fields[0] in ("self", "cls"):
        fields = fields[1:]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=PydanticDeprecationWarning)
        args_pydantic_schema = validate_arguments(func).model

    final_fields = {}
    for field_name in fields:
        field_info = args_pydantic_schema.model_fields[field_name]
        final_fields[field_name] = (
            field_info.annotation,
            FieldInfo(default=field_info.default, description=field_info.description),
        )

    return create_model(  # type: ignore[call-overload]
        func.__name__, **final_fields, __config__=ConfigDict(arbitrary_types_allowed=True)
    )


# Type variables for generics
P = ParamSpec("P")  # For parameters
T = TypeVar("T")  # For return type


class ToolFunction(Protocol[P, T]):
    """
    Protocol for functions that can be converted to tools.

    This protocol defines the interface that callable objects must satisfy
    to be used as tools in the LLM system.
    """

    __name__: str
    __doc__: str | None

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T: ...


ToolCallable: TypeAlias = ToolFunction[P, T]


class Tool(BaseModel, Generic[P, T]):
    """
    A tool that can be called by an LLM agent.

    Tools wrap functions with metadata and validation schemas, making them
    suitable for use with LLM function calling capabilities.

    Attributes:
        name: The name of the tool
        description: A brief description of what the tool does
        func: The callable function that implements the tool
        args_schema: Pydantic model defining the tool's argument schema

    Example:
        >>> def get_weather(location: str) -> str:
        ...     return f"Weather in {location}: Sunny"
        >>> tool = Tool.from_function(get_weather)
        >>> tool.name
        'get_weather'
        >>> tool(location="Paris")
        'Weather in Paris: Sunny'
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(..., description="The name of the tool.")
    description: str = Field(..., description="A brief description of the tool's functionality.")
    func: Callable[P, T] = Field(
        ..., description="The function that implements the tool's behavior."
    )
    args_schema: type[BaseModel] = Field(
        ..., description="The Pydantic model defining the tool's arguments schema."
    )

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        """Call the tool's underlying function."""
        return self.func(*args, **kwargs)

    @classmethod
    def from_function(
        cls,
        func: Callable[P, T],
        name: str | None = None,
        description: str | None = None,
        args_schema: type[BaseModel] | None = None,
    ) -> "Tool[P, T]":
        """
        Create a Tool instance from a function.

        Args:
            func: The function that implements the tool's behavior
            name: The name of the tool (defaults to function name)
            description: Description of the tool (defaults to docstring)
            args_schema: The Pydantic model for arguments (auto-generated if not provided)

        Returns:
            A Tool instance wrapping the function

        Example:
            >>> def multiply(a: int, b: int) -> int:
            ...     '''Multiply two numbers'''
            ...     return a * b
            >>> tool = Tool.from_function(multiply)
            >>> tool.name
            'multiply'
            >>> tool.description
            'Multiply two numbers'
        """
        tool_name = name or func.__name__
        tool_description = description or func.__doc__ or "No description provided."
        tool_args_schema = args_schema or get_argschema_from_function(func)
        return cls(
            name=tool_name, description=tool_description, func=func, args_schema=tool_args_schema
        )


# Type overloads for better type inference
@overload
def tool(func: Callable[P, T]) -> Tool[P, T]:
    """Overload for when tool is used as a direct decorator: @tool"""


@overload
def tool(
    *,
    name: str | None = None,
    description: str | None = None,
    args_schema: type[BaseModel] | None = None,
) -> Callable[[Callable[P, T]], Tool[P, T]]:
    """Overload for when tool is used with parameters: @tool(name="custom_name")"""


@overload
def tool(
    func: None,
    *,
    name: str | None = None,
    description: str | None = None,
    args_schema: type[BaseModel] | None = None,
) -> Callable[[Callable[P, T]], Tool[P, T]]:
    """Overload for when tool is called with func=None explicitly"""


@overload
def tool(
    func: Callable[P, T],
    *,
    name: str | None = None,
    description: str | None = None,
    args_schema: type[BaseModel] | None = None,
) -> Tool[P, T]:
    """Overload for when tool is called with func as a Callable and other keyword args"""


def tool(
    func: Callable[P, T] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    args_schema: type[BaseModel] | None = None,
) -> Any:
    """
    Decorator to create a Tool instance from a function.

    This decorator can be used with or without parameters to convert
    functions into tools suitable for LLM function calling.

    Args:
        func: The function to wrap (when used as @tool without parentheses)
        name: Custom name for the tool (defaults to function name)
        description: Custom description (defaults to docstring)
        args_schema: Custom Pydantic schema for arguments

    Returns:
        A Tool instance or a decorator function

    Examples:
        >>> @tool
        ... def get_time() -> str:
        ...     '''Get the current time'''
        ...     return "12:00 PM"

        >>> @tool(name="custom_name", description="Custom description")
        ... def my_function(x: int) -> int:
        ...     return x * 2
    """
    if func is None:

        def decorator(f: Callable[P, T]) -> Tool[P, T]:
            return Tool.from_function(
                f, name=name, description=description, args_schema=args_schema
            )

        return decorator

    return Tool.from_function(func, name=name, description=description, args_schema=args_schema)


class ToolKit:
    """
    Registry for managing multiple tools.

    ToolKit provides a convenient way to register and manage multiple tools
    that can be used together by an LLM agent.

    Example:
        >>> toolkit = ToolKit()
        >>> def add(a: int, b: int) -> int:
        ...     return a + b
        >>> toolkit.register(add)
        >>> tools = toolkit.get_tools()
        >>> len(tools)
        1
    """

    def __init__(self):
        """Initialize an empty toolkit."""
        self.tools: list[Tool] = []

    def get_tools(self) -> list[Tool]:
        """
        Get all registered tools.

        Returns:
            List of all Tool instances in the toolkit
        """
        return self.tools

    def register(
        self,
        func: Callable[..., Any],
        name: str | None = None,
        description: str | None = None,
        args_schema: type[BaseModel] | None = None,
    ) -> None:
        """
        Register a function as a tool in the toolkit.

        Args:
            func: The function to register
            name: Optional custom name for the tool
            description: Optional custom description
            args_schema: Optional custom argument schema

        Example:
            >>> toolkit = ToolKit()
            >>> def greet(name: str) -> str:
            ...     return f"Hello, {name}!"
            >>> toolkit.register(greet, description="Greet a person")
        """
        result = tool(func=func, name=name, description=description, args_schema=args_schema)
        self.tools.append(result)
