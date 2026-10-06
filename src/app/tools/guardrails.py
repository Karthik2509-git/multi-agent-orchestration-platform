"""Structural input and output guardrails for tool execution.

Focuses on structural boundaries, types, bounds, sizes, and serialization safety.
CRITICAL NON-GOAL: Does NOT use prompt-injection keyword heuristics or blacklists.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from src.app.core.logging import get_logger
from src.app.tools.base import BaseTool, ToolResult

logger = get_logger(__name__)


class ToolGuardrails:
    """Enforces structural validation on tool inputs and outputs before and after execution."""

    def __init__(
        self,
        max_input_size_bytes: int = 65_536,
        max_string_length: int = 10_000,
        max_output_size_bytes: int = 1_000_000,
        max_collection_length: int = 1_000,
    ) -> None:
        self.max_input_size_bytes = max_input_size_bytes
        self.max_string_length = max_string_length
        self.max_output_size_bytes = max_output_size_bytes
        self.max_collection_length = max_collection_length

    def validate_input(self, tool: BaseTool, arguments: Dict[str, Any]) -> Optional[ToolResult]:
        """Validate structured tool inputs against schema, bounds, sizes, and types.

        Returns:
            Optional[ToolResult]: A validation_error ToolResult if invalid, or None if valid.
        """
        if not isinstance(arguments, dict):
            return ToolResult(
                success=False,
                error="Tool arguments must be a JSON object / dictionary.",
                error_category="validation_error",
            )

        # 1. Total serialized input size
        try:
            serialized_input = json.dumps(arguments)
            if len(serialized_input.encode("utf-8")) > self.max_input_size_bytes:
                return ToolResult(
                    success=False,
                    error=(
                        f"Input payload exceeds maximum allowed size "
                        f"({self.max_input_size_bytes} bytes)."
                    ),
                    error_category="validation_error",
                )
        except (TypeError, ValueError) as err:
            return ToolResult(
                success=False,
                error=f"Tool input is not serializable: {err}",
                error_category="validation_error",
            )

        # 2. String length and collection bounds check
        size_error = self._check_nested_bounds(arguments)
        if size_error:
            return ToolResult(
                success=False,
                error=size_error,
                error_category="validation_error",
            )

        # 3. Schema conformity
        schema = getattr(tool, "parameters_schema", None)
        if schema and isinstance(schema, dict):
            schema_error = self._validate_schema(schema, arguments)
            if schema_error:
                return ToolResult(
                    success=False,
                    error=schema_error,
                    error_category="validation_error",
                )

        # 4. URL structural safety
        for key, val in arguments.items():
            if isinstance(val, str) and ("url" in key.lower() or "uri" in key.lower()):
                url_error = self._validate_url_structure(val)
                if url_error:
                    return ToolResult(
                        success=False,
                        error=f"Invalid URL structure for parameter '{key}': {url_error}",
                        error_category="validation_error",
                    )

        return None

    def validate_output(self, tool_name: str, result: ToolResult) -> ToolResult:
        """Validate tool execution output structure, size, and serialization safety."""
        if not result.success:
            return result

        if result.data is None:
            return result

        # Verify JSON serialization safety and output size limit
        try:
            serialized_output = json.dumps(result.data)
            output_bytes = len(serialized_output.encode("utf-8"))
            if output_bytes > self.max_output_size_bytes:
                logger.warning(
                    "Tool '%s' output exceeded maximum size limit (%d > %d bytes)",
                    tool_name,
                    output_bytes,
                    self.max_output_size_bytes,
                )
                return ToolResult(
                    success=False,
                    error=(
                        f"Tool '{tool_name}' output exceeded maximum allowed size "
                        f"({self.max_output_size_bytes} bytes)."
                    ),
                    error_category="validation_error",
                )
        except (TypeError, ValueError) as err:
            logger.warning("Tool '%s' output failed JSON serialization: %s", tool_name, err)
            return ToolResult(
                success=False,
                error=f"Tool '{tool_name}' produced non-serializable output: {err}",
                error_category="validation_error",
            )

        return result

    def _check_nested_bounds(self, obj: Any, depth: int = 0) -> Optional[str]:
        """Recursively verify string lengths and collection sizes."""
        if depth > 20:
            return "Input exceeds maximum nested structure depth."

        if isinstance(obj, str):
            if len(obj) > self.max_string_length:
                return (
                    f"String argument exceeds maximum length ({self.max_string_length} characters)."
                )
        elif isinstance(obj, (list, tuple)):
            if len(obj) > self.max_collection_length:
                return (
                    f"Collection argument exceeds maximum length "
                    f"({self.max_collection_length} items)."
                )
            for item in obj:
                err = self._check_nested_bounds(item, depth + 1)
                if err:
                    return err
        elif isinstance(obj, dict):
            if len(obj) > self.max_collection_length:
                return (
                    f"Dictionary argument exceeds maximum length "
                    f"({self.max_collection_length} keys)."
                )
            for k, v in obj.items():
                if len(str(k)) > self.max_string_length:
                    return "Dictionary key exceeds maximum allowed length."
                err = self._check_nested_bounds(v, depth + 1)
                if err:
                    return err
        return None

    def _validate_schema(self, schema: Dict[str, Any], arguments: Dict[str, Any]) -> Optional[str]:
        """Validate arguments against parameters_schema properties and required fields."""
        # Check required fields
        required_fields = schema.get("required", [])
        for field in required_fields:
            if field not in arguments:
                return f"Missing required parameter '{field}'."

        properties = schema.get("properties", {})
        if not properties or not isinstance(properties, dict):
            return None

        # Check property types, enums, and bounds
        for param_name, param_val in arguments.items():
            if param_name not in properties:
                continue
            prop_def = properties[param_name]
            if not isinstance(prop_def, dict):
                continue

            expected_type = prop_def.get("type")
            if expected_type:
                type_err = self._check_type(param_name, param_val, expected_type)
                if type_err:
                    return type_err

            # Check enum allowlist if defined
            if "enum" in prop_def and isinstance(prop_def["enum"], list):
                if param_val not in prop_def["enum"]:
                    return (
                        f"Value for '{param_name}' must be one of {prop_def['enum']}, "
                        f"got '{param_val}'."
                    )

            # Check numeric minimum and maximum
            if isinstance(param_val, (int, float)) and not isinstance(param_val, bool):
                if "minimum" in prop_def and param_val < prop_def["minimum"]:
                    return f"Parameter '{param_name}' cannot be less than {prop_def['minimum']}."
                if "maximum" in prop_def and param_val > prop_def["maximum"]:
                    return f"Parameter '{param_name}' cannot be greater than {prop_def['maximum']}."

        return None

    @staticmethod
    def _check_type(param_name: str, val: Any, expected_type: str) -> Optional[str]:
        """Verify parameter value matches expected JSON schema type."""
        if val is None:
            return None

        type_mapping = {
            "string": lambda v: isinstance(v, str),
            "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
            "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
            "boolean": lambda v: isinstance(v, bool),
            "array": lambda v: isinstance(v, (list, tuple)),
            "object": lambda v: isinstance(v, dict),
        }

        checker = type_mapping.get(expected_type)
        if checker and not checker(val):
            return (
                f"Invalid type for parameter '{param_name}': "
                f"expected '{expected_type}', got '{type(val).__name__}'."
            )
        return None

    @staticmethod
    def _validate_url_structure(url_str: str) -> Optional[str]:
        """Validate structural safety of URL parameters (scheme and netloc)."""
        trimmed = url_str.strip()
        if not trimmed:
            return "URL cannot be empty."

        try:
            parsed = urlparse(trimmed)
            if parsed.scheme.lower() not in ("http", "https"):
                return (
                    f"URL scheme '{parsed.scheme}' is not allowed. Only HTTP/HTTPS are supported."
                )
            if not parsed.netloc:
                return "URL must contain a valid host / network location."
            if any(char in trimmed for char in ("\n", "\r", "\t", "\x00")):
                return "URL contains illegal control characters."
        except Exception as e:
            return f"Malformed URL syntax: {e}"

        return None
