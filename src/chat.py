import os
import sys
import json
import re
from abc import ABC, abstractmethod
from typing import Optional, Union, Callable, Dict, Any, List


def _load_env_file(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    try:
        with open(path, encoding="utf-8") as env_file:
            for raw_line in env_file:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                os.environ.setdefault(key, value)
    except OSError:
        return


try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    load_dotenv = None


if load_dotenv:
    load_dotenv()
else:  # Fallback minimal loader
    _load_env_file()

def extract_all_blocks(main_content, code_format):
    """Extract all codes from code blocks"""
    sql_blocks = []
    start = 0

    while True:

        sql_query_start = main_content.find(f"```{code_format}", start)
        if sql_query_start == -1:
            break


        sql_query_end = main_content.find("```", sql_query_start + len(f"```{code_format}"))
        if sql_query_end == -1:
            break

        sql_block = main_content[sql_query_start + len(f"```{code_format}"):sql_query_end].strip()
        sql_blocks.append(sql_block)

        start = sql_query_end + len("```")

    return sql_blocks

def check_json_structure(generated, expected):
    if isinstance(expected, dict):
        if not isinstance(generated, dict):
            return False, f"Expected dictionary, got {type(generated).__name__}"
        for k in expected:
            if k not in generated:
                return False, f"Missing key '{k}'"
            is_valid, msg = check_json_structure(generated[k], expected[k])
            if not is_valid:
                return False, f"In key '{k}': {msg}"
    elif isinstance(expected, list):
        if not isinstance(generated, list):
            return False, f"Expected list, got {type(generated).__name__}"
        if expected and generated:
            # Check only the first item of generated against the first item of expected
            # This allows subsequent items to vary if the expected list only provides one example schema.
            # If generated list has items but expected is empty, we generally allow it (or user should define expected accordingly).
            # Assuming expected[0] defines the schema for items in the list.
            is_valid, msg = check_json_structure(generated[0], expected[0])
            if not is_valid:
                return False, f"In list item 0: {msg}"
    return True, ""

def extract_json_from_text(text, example_json_structure=None):
    """
    Last resort: Try to extract JSON from text even without ```json``` code blocks.
    Looks for JSON-like structures (starting with { or [) and validates them.

    Args:
        text: The text to search for JSON
        example_json_structure: Optional structure to validate against

    Returns:
        List of JSON strings found, or empty list if none found/valid
    """
    json_blocks = []
    processed_ranges = []  # Track already processed ranges to avoid nested extractions

    # Find potential JSON objects/arrays by looking for { or [
    # Process objects first (more likely to be the main response), then arrays
    start_chars = ['{', '[']
    for start_char in start_chars:
        start_idx = 0
        while True:
            # Find the next occurrence of the start character
            json_start = text.find(start_char, start_idx)
            if json_start == -1:
                break

            # Skip if this position is already inside a processed JSON block
            if any(start <= json_start < end for start, end in processed_ranges):
                start_idx = json_start + 1
                continue

            # Try to find the matching closing character
            # Start depth at 1 since we've already found the opening character
            depth = 1
            json_end = -1
            in_string = False
            escape_next = False
            closing_char = '}' if start_char == '{' else ']'

            for i in range(json_start + 1, len(text)):
                char = text[i]

                if escape_next:
                    escape_next = False
                    continue

                if char == '\\':
                    escape_next = True
                    continue

                if char == '"':
                    in_string = not in_string
                    continue

                if not in_string:
                    if char == start_char:
                        depth += 1
                    elif char == closing_char:
                        depth -= 1
                        if depth == 0:
                            json_end = i + 1
                            break

            if json_end == -1:
                start_idx = json_start + 1
                continue

            # Extract the potential JSON string
            json_candidate = text[json_start:json_end].strip()

            # Clean up common markdown formatting issues that break JSON parsing
            # Remove markdown bold/italic around key names: **"key"** -> "key"
            # This handles cases like **"reasoning"**: -> "reasoning":
            # Pattern matches **"key"** or *"key"* and replaces with "key"
            json_candidate_cleaned = re.sub(r'\*+("[\w_]+")\*+', r'\1', json_candidate)

            # Fix malformed escape sequences in JSON keys
            # Fix cases like "key\": -> "key": (incorrectly escaped closing quote in key names)
            # This handles cases where a backslash appears before a closing quote in a key name before a colon
            json_candidate_cleaned = re.sub(r'("[\w_]+)\\"(:)', r'\1"\2', json_candidate_cleaned)

            # Try to parse it as JSON
            try:
                parsed_json = json.loads(json_candidate_cleaned)

                # If example_json_structure is provided, validate against it
                if example_json_structure is not None:
                    is_valid, struct_err = check_json_structure(parsed_json, example_json_structure)
                    if is_valid:
                        json_blocks.append(json_candidate_cleaned)
                        processed_ranges.append((json_start, json_end))
                else:
                    # No structure validation needed, accept any valid JSON
                    json_blocks.append(json_candidate_cleaned)
                    processed_ranges.append((json_start, json_end))

            except json.JSONDecodeError:
                # Not valid JSON, continue searching
                pass

            start_idx = json_end

    return json_blocks

class BaseChat(ABC):
    def __init__(self, model: str):
        self.model = model
        self.messages = []

    @abstractmethod
    def get_response(self, prompt) -> str:
        pass

    def get_code_blocks(self, prompt, code_format=None, req_param_dct: dict = {}, example_json_structure=None, logger=None) -> list:
        code_blocks = []
        max_try = 3
        thoughts = None
        error_msg = None

        while max_try > 0:
            max_try -= 1
            try:
                if max_try == 2: # First turn
                    response = self.get_response(prompt, req_param_dct=req_param_dct, logger=logger)
                else:
                    retry_prompt = error_msg if error_msg else f"Can't find any ```{code_format}``` block in your previous response! Please follow the response format!"
                    response = self.get_response(retry_prompt, req_param_dct=req_param_dct, logger=logger)

                text = response["text"]
                thoughts = response["thoughts"]
                code_blocks = extract_all_blocks(text, code_format)

                if code_blocks == []:
                    # Last resort: if code_format is json, try to extract JSON from text without code blocks
                    if code_format == "json":
                        json_blocks = extract_json_from_text(text, example_json_structure)
                        if json_blocks:
                            code_blocks = json_blocks
                            if logger:
                                logger.info("Found JSON in text without ```json``` wrapper, using fallback extraction")
                        else:
                            error_msg = f"Can't find any ```{code_format}``` block in your previous response! Please follow the response format!"
                            if logger:
                                logger.warning(f"Failed to extract {code_format} block. \n Generated Text:\n{text}")
                            continue
                    else:
                        error_msg = f"Can't find any ```{code_format}``` block in your previous response! Please follow the response format!"
                        if logger:
                            logger.warning(f"Failed to extract {code_format} block. \n Generated Text:\n{text}")
                        continue

                if code_format == "json" and example_json_structure is not None:
                     try:
                         generated_json = json.loads(code_blocks[-1])
                         is_valid, struct_err = check_json_structure(generated_json, example_json_structure)
                         if not is_valid:
                             error_msg = f"The generated JSON does not match the expected structure: {struct_err}. Please correct it!"
                             if logger:
                                 logger.warning(f"JSON structure mismatch: {struct_err}\nGenerated JSON:\n{json.dumps(generated_json, indent=2)}")
                             code_blocks = [] # force retry
                             continue
                     except json.JSONDecodeError:
                         error_msg = "The generated JSON block is not valid JSON! Please ensure it is valid JSON."
                         if logger:
                             logger.warning("Invalid JSON generated.")
                         code_blocks = []
                         continue

                break

            except Exception as e:
                logger.error(f"max_try: {max_try}, exception: {e}")
                continue

        if code_blocks == []:
            logger.error(f"get_code_blocks() exit, max_try: {max_try}, code_blocks: {code_blocks}")
            logger.error(f"Generated Text:\n{text}")

        return {"code_blocks": code_blocks, "thoughts": thoughts}

    def get_model_response_txt(self, prompt, print_thoughts=False, req_param_dct: dict = {}) -> str:
        max_try = 3
        while max_try > 0:
            max_try -= 1
            # try:
            response = self.get_response(prompt, req_param_dct=req_param_dct)
            return response
            # except Exception as e:
            #     print(f"max_try: {max_try}, exception: {e}")
            #     continue
        print(f"get_model_response_txt() exit, max_try: {max_try}")
        sys.exit(0)

    def get_message_len(self):
        return {
            "prompt_len": sum(len(item["content"]) for item in self.messages if item["role"] == "user"),
            "response_len": sum(len(item["content"]) for item in self.messages if item["role"] == "assistant"),
            "num_calls": len(self.messages) // 2
        }

    def init_messages(self):
        self.messages = []


from openai import OpenAI


class Chat(BaseChat):
    def __init__(
        self,
        model,
        *,
        base_url: Optional[str] = None,
        ip: Optional[str] = None,
        port: Optional[Union[str, int]] = None,
        gen_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(model)
        self.ip = ip
        self.port = port
        self.base_url = base_url
        self.gen_config: Dict[str, Any] = gen_config or {}

        resolved_base_url = base_url or os.getenv("OPENAI_BASE_URL")
        if not resolved_base_url and ip and port:
            resolved_base_url = f"http://{ip}:{port}/v1"

        resolved_key = os.getenv("OPENAI_API_KEY") or os.getenv("VLLM_API_KEY", "EMPTY")

        client_kwargs = {"api_key": resolved_key}
        if resolved_base_url:
            client_kwargs["base_url"] = resolved_base_url.rstrip("/")

        self.client = OpenAI(**client_kwargs)

        self.usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        # Tool calling support
        self.tool_calling_enabled = False
        self.tool_functions: Dict[str, Callable] = {}

    def get_response(self, prompt, req_param_dct: dict = None, logger=None) -> str:
        current_turn = {"role": "user", "content": prompt}
        self.messages.append(current_turn)

        # Start with model-specific default parameters
        request_params = self.get_req_param_dct().copy()

        # Apply instance-level gen_config (overrides model defaults)
        if self.gen_config:
            request_params.update(self.gen_config)

        # Merge per-call parameters (highest precedence)
        if req_param_dct:
            request_params.update(req_param_dct)

        # Handle parameters that need to be in extra_body (for custom endpoints)
        # Parameters that should go in extra_body for custom endpoints (like top_k)
        extra_body_only_params = {"top_k"}

        # Extract parameters that should go in extra_body
        extra_body_params = {}
        params_to_remove = []

        for key, value in request_params.items():
            if key in extra_body_only_params:
                extra_body_params[key] = value
                params_to_remove.append(key)

        # Remove params that will go in extra_body from request_params
        for key in params_to_remove:
            del request_params[key]

        # Merge extra_body if we have custom parameters
        if extra_body_params:
            if "extra_body" not in request_params:
                request_params["extra_body"] = {}
            request_params["extra_body"].update(extra_body_params)

        # Add tools if tool calling is enabled
        if self.tool_calling_enabled and self.tool_functions:
            request_params["tools"] = self._get_tools_definition()
            request_params["tool_choice"] = "auto"

        # Log the final generation config (exclude verbose fields like tools)
        if logger:
            loggable = {k: v for k, v in request_params.items() if k not in ("tools",)}
            logger.debug(f"[gen_config] model={self.model} params={json.dumps(loggable, default=str)}")

        max_iterations = 50  # Prevent infinite loops, temporarily increased to 100 to see the model limit
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                **request_params
            )
            usage = getattr(response, "usage", None)
            if usage:
                if isinstance(usage, dict):
                    prompt_tokens = usage.get("prompt_tokens", 0) or 0
                    completion_tokens = usage.get("completion_tokens", 0) or 0
                    total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
                else:
                    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
                    total_tokens = getattr(usage, "total_tokens", prompt_tokens + completion_tokens)

                self.usage["prompt_tokens"] += prompt_tokens
                self.usage["completion_tokens"] += completion_tokens
                self.usage["total_tokens"] += total_tokens

            message = response.choices[0].message

            # Check if the model wants to call a tool
            tool_calls = getattr(message, 'tool_calls', None)

            if tool_calls and self.tool_calling_enabled:
                content = message.content or ""
                if "</think>" in content:
                    content = content.split("</think>")[-1].strip()

                if logger:
                    logger.info(f"Assistant message: {content}")
                    logger.info(f"Tool calls detected: {tool_calls}")
                # Add assistant message with tool calls
                assistant_message = {"role": "assistant", "content": content or None}
                if tool_calls:
                    assistant_message["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in tool_calls
                    ]

                self.messages.append(assistant_message)

                # Execute tool calls
                tool_results = self._execute_openai_tool_calls(tool_calls, logger=logger)
                for tool_result in tool_results:
                    self.messages.append(tool_result)

                # Continue the loop to get the model's response after tool execution
                continue

            # No tool calls, return the final response
            main_content = message.content or ""
            thoughts = getattr(message, 'reasoning_content', None)

            # Thinking-model dropout: model emitted empty content with no tool call.
            # Re-enter the loop without appending the empty message so the model retries.
            if not main_content.strip() and thoughts and self.tool_calling_enabled:
                if logger:
                    logger.warning("Empty response after tool calls (thinking-model dropout), re-prompting...")
                continue

            if "</think>" in main_content:
                thoughts = main_content.split("</think>")[0]
                main_content = main_content.split("</think>")[-1].strip()

            self.messages.append({"role": "assistant", "content": main_content})
            return {"text": main_content, "thoughts": thoughts}

        # If we exit the loop (max_iterations reached), force the model to output final response
        last_message = response.choices[0].message
        last_tool_calls = getattr(last_message, 'tool_calls', None)

        if last_tool_calls and self.tool_calling_enabled:
            if logger:
                logger.warning(f"Max iterations ({max_iterations}) reached. Forcing final response without tool calls.")

            final_request_params = {k: v for k, v in request_params.items() if k != "tools"}
            final_request_params["tool_choice"] = "none"

            final_response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                **final_request_params
            )

            usage = getattr(final_response, "usage", None)
            if usage:
                if isinstance(usage, dict):
                    prompt_tokens = usage.get("prompt_tokens", 0) or 0
                    completion_tokens = usage.get("completion_tokens", 0) or 0
                    total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
                else:
                    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
                    total_tokens = getattr(usage, "total_tokens", prompt_tokens + completion_tokens)

                self.usage["prompt_tokens"] += prompt_tokens
                self.usage["completion_tokens"] += completion_tokens
                self.usage["total_tokens"] += total_tokens

            final_message = final_response.choices[0].message
            main_content = final_message.content or ""
            thoughts = getattr(final_message, 'reasoning_content', None)
        else:
            main_content = last_message.content or ""
            thoughts = getattr(last_message, 'reasoning_content', None)

        if "</think>" in main_content:
            thoughts = main_content.split("</think>")[0]
            main_content = main_content.split("</think>")[-1].strip()

        self.messages.append({"role": "assistant", "content": main_content})
        return {"text": main_content, "thoughts": thoughts}

    def clear_chat_history(self):
        self.messages = []

    def update_turn_message(self, new_message, turn_id=-1):
        self.messages[turn_id]["content"] = new_message

    def get_model_name(self):
        return self.model.split("/")[-1]

    def get_req_param_dct(self):
        """Default request parameters for local gpt-oss."""
        return {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "reasoning_effort": "high"}

    def set_system_prompt(self, message):
        if len(self.messages) != 0:
            if self.messages[0]["role"] != "system":
                self.messages.insert(0, {"role": "system", "content": message})
            else:
                self.messages[0]["content"] = message
        else:
            self.messages = [{"role": "system", "content": message}]
        return

    def get_usage(self):
        return dict(self.usage)

    def reset_usage(self):
        self.usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def enable_tools(self, tool_names: List[str], **config):
        """
        Enable tool calling mode with specified tools.

        Args:
            tool_names: List of tool names to enable. Available tools:
                - "query_database": Execute SQL queries
                - "get_distinct_values": Get distinct values from a column
                - "search_dimension_values": Search for values in a column
                - "python_interpreter": Execute Python code in a stateful interpreter
                - "list_tables": List tables in a schema (hierarchical schema linking)
                - "list_columns": List columns in a table (hierarchical schema linking)

            **config: Configuration parameters for the tools. Common parameters:
                - db_path: Path to the database folder (required for database tools)
                - db_type: Database type "snowflake" or "sqlite" (required for database tools)
                - cursor_getter: Callable that returns a database cursor (required for database tools)
                - n_example_rows: Number of example rows to return (default: 1, for query_database)
                - database_name: Name of the database (required for hierarchical schema linking tools)
                - data_frames: List of pandas DataFrames (for python_interpreter)
                - additional_context: Dict of additional variables (for python_interpreter)

        Examples:
            # Enable database query tools
            chat.enable_tools(
                ["query_database", "get_distinct_values", "search_dimension_values"],
                db_path="/path/to/db",
                db_type="snowflake",
                cursor_getter=get_cursor,
                n_example_rows=5
            )

            # Enable Python interpreter
            chat.enable_tools(
                ["python_interpreter"],
                data_frames=[df1, df2],
                additional_context={"custom_var": 42}
            )

            # Enable hierarchical schema linking
            chat.enable_tools(
                ["list_tables", "list_columns"],
                db_type="snowflake",
                database_name="MYDB",
                db_path="/path/to/db"
            )
        """
        from tools import (
            create_query_database_tool,
            create_get_distinct_values_tool,
            create_search_dimension_values_tool,
            create_python_interpreter_tool,
            create_list_tables_tool,
            create_list_columns_tool,
            create_read_program_tool,
        )

        self.tool_calling_enabled = True
        self._tool_db_type = config.get("db_type")

        self.tool_functions = {}

        # Tool registry: maps tool names to (factory_function, required_params, optional_params)
        tool_registry = {
            "query_database": (
                create_query_database_tool,
                ["db_path", "db_type", "cursor_getter"],
                {"n_example_rows": 1}
            ),
            "get_distinct_values": (
                create_get_distinct_values_tool,
                ["db_type", "cursor_getter"],
                {}
            ),
            "search_dimension_values": (
                create_search_dimension_values_tool,
                ["db_type", "cursor_getter"],
                {}
            ),
            "python_interpreter": (
                create_python_interpreter_tool,
                [],
                {"data_frames_map": None, "additional_context": None, "cursor_getter": None, "db_type": None, "db_connection_str": None}
            ),
            "list_tables": (
                create_list_tables_tool,
                ["db_type", "database_name", "db_path"],
                {}
            ),
            "list_columns": (
                create_list_columns_tool,
                ["db_type", "database_name", "db_path"],
                {}
            ),
            "read_program": (
                create_read_program_tool,
                ["program_dir"],
                {}
            )
        }

        for tool_name in tool_names:
            if tool_name not in tool_registry:
                raise ValueError(f"Unknown tool name: {tool_name}. Available tools: {list(tool_registry.keys())}")

            factory_func, required_params, optional_params = tool_registry[tool_name]

            # Check required parameters
            missing_params = [p for p in required_params if p not in config]
            if missing_params:
                raise ValueError(f"Tool '{tool_name}' requires the following parameters: {missing_params}")

            # Build kwargs for the factory function
            kwargs = {}
            for param in required_params:
                kwargs[param] = config[param]
            for param, default_value in optional_params.items():
                kwargs[param] = config.get(param, default_value)

            # list_tables is only needed for Snowflake (multiple schemas);
            # SQLite has a single schema so skip this tool entirely
            if tool_name == "list_tables" and config.get("db_type") == "sqlite":
                continue

            # Create and register the tool
            self.tool_functions[tool_name] = factory_func(**kwargs)

    def disable_tool_calling(self):
        """Disable tool calling mode."""
        self.tool_calling_enabled = False
        self.tool_functions = {}

    def _get_tools_definition(self) -> list:
        """Get the tools definition in OpenAI format."""
        from tools import get_tools_definition
        db_type = getattr(self, "_tool_db_type", None)
        return get_tools_definition(self.tool_functions, db_type=db_type)

    def _execute_openai_tool_calls(self, tool_calls, logger=None) -> list:
        """Execute OpenAI-style tool_call objects and return role=tool result messages.

        Each item in `tool_calls` must have `.id`, `.function.name`, `.function.arguments`
        (a JSON string).
        """
        tool_results = []
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            if "<|" in function_name:
                function_name = function_name.split("<|")[0]
            try:
                function_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                _json_err = e
                function_args = None
                if "Extra data" in str(e):
                    args_text = tool_call.function.arguments.strip()
                    depth = 0
                    in_string = False
                    escape = False
                    end = 0
                    for i, c in enumerate(args_text):
                        if escape:
                            escape = False
                            continue
                        if in_string:
                            if c == "\\":
                                escape = True
                                continue
                            if c == '"':
                                in_string = False
                            continue
                        if c == '"':
                            in_string = True
                            continue
                        if c == '{':
                            depth += 1
                            continue
                        if c == '}':
                            depth -= 1
                            if depth == 0:
                                end = i + 1
                                break
                    if end > 0:
                        try:
                            function_args = json.loads(args_text[:end])
                        except json.JSONDecodeError:
                            pass
                if function_args is None:
                    if function_name == "python_interpreter":
                        args_text = tool_call.function.arguments.strip()
                        if args_text.startswith('{') and '"code"' in args_text:
                            prefix = re.match(r'\{\s*"code"\s*:\s*"', args_text)
                            suffix = re.match(r'^(.*)"\s*\}\s*$', args_text, re.DOTALL)
                            if prefix and suffix and suffix.end(1) >= prefix.end():
                                code = args_text[prefix.end():suffix.end(1)]
                                code = (code
                                    .replace('\\\\', '\x00')
                                    .replace('\\n', '\n')
                                    .replace('\\t', '\t')
                                    .replace('\\"', '"')
                                    .replace("\\'", "'")
                                    .replace('\x00', '\\'))
                                function_args = {"code": code}
                        if function_args is None:
                            function_args = {"code": args_text} if args_text else {}
                    else:
                        function_args = {}
                if logger:
                    if function_args:
                        logger.debug(f"Malformed JSON for {function_name} recovered via fallback (json error: {_json_err})")
                    else:
                        logger.error(f"Could not extract arguments for {function_name} from malformed JSON: {_json_err}\n  Raw (first 500 chars): {tool_call.function.arguments[:500]}")

            if function_name in self.tool_functions:
                try:
                    result = self.tool_functions[function_name](**function_args)
                    tool_results.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": result if isinstance(result, str) else json.dumps(result)
                    })
                except Exception as e:
                    tool_results.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": f"Error: {str(e)}"
                    })
            else:
                tool_results.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": f"Error: Tool '{function_name}' not found"
                })
        return tool_results
