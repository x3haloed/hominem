"""
Wrapper tools that implement Qwen-Agent BaseTool interface
but use OpenAI-style schemas and execution functions.
"""

from __future__ import annotations

from typing import Any, Dict, Union

from qwen_agent.tools.base import BaseTool
from hominem_agent.tools.openai_tools import TOOL_SCHEMAS, execute_tool


class OpenAIToolWrapper(BaseTool):
    """Wrapper that makes OpenAI-style tools compatible with Qwen-Agent."""

    def __init__(self, tool_schema: Dict[str, Any]):
        self.tool_schema = tool_schema
        function = tool_schema["function"]
        self.name = function["name"]
        self.description = function["description"]
        self.parameters = function["parameters"]

    def call(self, params: Union[str, dict], **kwargs) -> Dict[str, Any]:
        if isinstance(params, str):
            raise ValueError("params must be an object")

        # Execute the tool using our OpenAI tool executor
        return execute_tool(self.name, **params)


# Create wrapper instances for each tool
def get_wrapped_tools() -> list[BaseTool]:
    """Get Qwen-Agent compatible tool wrappers."""
    return [OpenAIToolWrapper(schema) for schema in TOOL_SCHEMAS]