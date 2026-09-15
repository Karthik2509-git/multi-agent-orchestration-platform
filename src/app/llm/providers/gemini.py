"""Google Gemini LLM Provider implementation using the official google-genai SDK."""

import json
from typing import Any, Dict, List, Optional, Tuple

from google import genai
from google.genai import errors, types

from src.app.core.logging import get_logger
from src.app.llm.base import LLMProvider
from src.app.models.schemas.llm import LLMResponse, ToolCall

logger = get_logger(__name__)


class GeminiLLMProvider(LLMProvider):
    """LLM provider implementation for Google Gemini models."""

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        if not api_key:
            raise ValueError("Gemini API key must be provided to initialize GeminiLLMProvider")
        self.model = model
        self.client = genai.Client(api_key=api_key)

    def _convert_tools(self, tools: Optional[List[Dict[str, Any]]]) -> Optional[List[types.Tool]]:
        """Translate tool schemas into Gemini FunctionDeclaration objects."""
        if not tools:
            return None

        func_decls: List[types.FunctionDeclaration] = []
        for tool in tools:
            name = tool.get("name", "")
            desc = tool.get("description", "")
            params = tool.get("parameters", {})
            func_decls.append(
                types.FunctionDeclaration(
                    name=name,
                    description=desc,
                    parameters=params,
                )
            )

        return [types.Tool(function_declarations=func_decls)]

    def _convert_messages(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[Optional[str], List[types.Content]]:
        """Translate internal messages into Gemini system instruction and Content objects."""
        system_instruction: Optional[str] = None
        contents: List[types.Content] = []

        for msg in messages:
            role = msg.get("role", "user")
            content_text = msg.get("content")

            if role == "system":
                # In Gemini SDK, system prompts are passed via GenerateContentConfig
                system_instruction = content_text
                continue

            if role == "user":
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=content_text or "")],
                    )
                )

            elif role == "assistant":
                tool_calls = msg.get("tool_calls", [])
                parts: List[types.Part] = []

                if content_text:
                    parts.append(types.Part.from_text(text=content_text))

                for tc in tool_calls:
                    # Support both OpenAI format dict and ToolCall object
                    if isinstance(tc, dict):
                        fn_info = tc.get("function", {})
                        fn_name = fn_info.get("name") or tc.get("name", "")
                        fn_args = fn_info.get("arguments") or tc.get("arguments", {})
                    else:
                        fn_name = getattr(tc, "name", "")
                        fn_args = getattr(tc, "arguments", {})

                    if isinstance(fn_args, str):
                        try:
                            fn_args = json.loads(fn_args)
                        except json.JSONDecodeError:
                            fn_args = {}

                    parts.append(types.Part.from_function_call(name=fn_name, args=fn_args))

                if not parts:
                    parts.append(types.Part.from_text(text=""))

                contents.append(types.Content(role="model", parts=parts))

            elif role == "tool":
                # Map tool execution result into Gemini FunctionResponse
                fn_name = msg.get("name", "")
                tool_output_raw = content_text or "{}"

                try:
                    tool_output_dict = (
                        json.loads(tool_output_raw)
                        if isinstance(tool_output_raw, str)
                        else tool_output_raw
                    )
                except Exception:
                    tool_output_dict = {"output": tool_output_raw}

                contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_function_response(
                                name=fn_name,
                                response={"response": tool_output_dict},
                            )
                        ],
                    )
                )

        return system_instruction, contents

    async def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        """Call Gemini API via official google-genai async client."""
        logger.info(
            "Invoking Gemini model %s with %d messages and %d tools",
            self.model,
            len(messages),
            len(tools) if tools else 0,
        )

        system_instruction, contents = self._convert_messages(messages)
        gemini_tools = self._convert_tools(tools)

        config = types.GenerateContentConfig()
        if system_instruction:
            config.system_instruction = system_instruction
        if gemini_tools:
            config.tools = gemini_tools

        try:
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )

            parsed_tool_calls: List[ToolCall] = []
            if response.function_calls:
                for idx, fc in enumerate(response.function_calls):
                    call_id = f"call_{fc.name}_{idx}"
                    parsed_tool_calls.append(
                        ToolCall(
                            id=call_id,
                            name=fc.name,
                            arguments=dict(fc.args) if fc.args else {},
                        )
                    )

            # Safely extract text (may be None if model returned only function calls)
            response_text = None
            try:
                response_text = response.text
            except Exception:
                # Text property may raise warning or exception when only function_call is present
                pass

            finish_reason = "tool_calls" if parsed_tool_calls else "stop"

            return LLMResponse(
                content=response_text,
                tool_calls=parsed_tool_calls,
                model=self.model,
                finish_reason=finish_reason,
            )

        except errors.APIError as api_err:
            logger.error("Gemini API invocation failed: %s", type(api_err).__name__)
            raise RuntimeError(f"Gemini service error: {type(api_err).__name__}") from api_err
        except Exception as ex:
            logger.error("Unexpected Gemini error: %s", type(ex).__name__)
            raise RuntimeError(f"Gemini provider error: {str(ex)}") from ex
