"""
Per-request run logger.
Creates one log file per request under backend/logs/.
Shared between app.py (stream events) and nodes.py (LLM inputs/outputs)
via a module-level registry keyed by request_id.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

LOGS_DIR = Path(__file__).parent / "logs"

# Module-level registry: request_id -> RunLogger
_registry: dict[str, "RunLogger"] = {}


def register(request_id: str, run_logger: "RunLogger") -> None:
    _registry[request_id] = run_logger


def get(request_id: str) -> Optional["RunLogger"]:
    return _registry.get(request_id)


def remove(request_id: str) -> None:
    _registry.pop(request_id, None)


class RunLogger:
    W = 80  # line width

    def __init__(self, request_id: str, user_message: str, thread_id: str):
        LOGS_DIR.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = LOGS_DIR / f"{ts}_{request_id}.log"
        self.request_id = request_id
        self.llm_call_count = 0
        self.tool_call_count = 0
        self._f = open(self.log_path, "w", encoding="utf-8")
        register(request_id, self)
        self._header(user_message, thread_id)

    # ------------------------------------------------------------------ helpers

    def _w(self, text: str = "") -> None:
        self._f.write(text + "\n")
        self._f.flush()

    def _rule(self, char: str = "=") -> None:
        self._w(char * self.W)

    def _title(self, text: str, char: str = "=") -> None:
        self._rule(char)
        self._w(f"  {text}")
        self._rule(char)

    @staticmethod
    def _truncate(text: str, limit: int = 3000) -> str:
        if len(text) > limit:
            return text[:limit] + f"\n    ... [{len(text) - limit} more chars truncated]"
        return text

    @staticmethod
    def _indent(text: str, spaces: int = 4) -> str:
        pad = " " * spaces
        return "\n".join(pad + line for line in str(text).split("\n"))

    # ------------------------------------------------------------------ header

    def _header(self, user_message: str, thread_id: str) -> None:
        self._title("LLAMABOT — RUN LOG")
        self._w(f"  Request ID : {self.request_id}")
        self._w(f"  Thread ID  : {thread_id}")
        self._w(f"  Started    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._w(f"  Log file   : {self.log_path.name}")
        self._rule()
        self._w()
        self._w("USER MESSAGE")
        self._rule("-")
        self._w(self._indent(user_message))
        self._w()

    # ------------------------------------------------------------------ context

    def log_existing_html(self, html: str) -> None:
        self._w("EXISTING PAGE.HTML (context fed to agent)")
        self._rule("-")
        if html.strip():
            self._w(self._indent(self._truncate(html, 600)))
        else:
            self._w("    (empty — no existing page)")
        self._w()

    # ------------------------------------------------------------------ LLM input

    def log_llm_input(self, messages: list) -> None:
        self.llm_call_count += 1
        self._w()
        self._title(
            f"LLM CALL #{self.llm_call_count}  ▶  software_developer_assistant — INPUT",
            char="─",
        )
        self._w(f"  Messages in context : {len(messages)}")
        self._w()

        for i, msg in enumerate(messages):
            msg_type = type(msg).__name__
            content = msg.content if hasattr(msg, "content") else ""
            if isinstance(content, list):
                content = str(content)
            tool_calls = getattr(msg, "tool_calls", None) or []
            fc = (getattr(msg, "additional_kwargs", {}) or {}).get("function_call")

            label = f"[{i}] {msg_type}"
            if fc:
                label += f"  →  tool_call: {fc.get('name', '?')}"
            elif tool_calls:
                label += f"  →  tool_call: {tool_calls[0].get('name', '?')}"

            self._w(f"  {label}")

            if content:
                self._w(self._indent(self._truncate(str(content), 500), 6))

            if tool_calls:
                for tc in tool_calls:
                    args = tc.get("args", {})
                    args_str = json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else str(args)
                    self._w(f"      args: {self._truncate(args_str, 400)}")
            elif fc:
                try:
                    args = json.loads(fc.get("arguments", "{}"))
                    args_str = json.dumps(args, ensure_ascii=False)
                except Exception:
                    args_str = str(fc.get("arguments", ""))
                self._w(f"      args: {self._truncate(args_str, 400)}")

            self._w()

    # ------------------------------------------------------------------ LLM output

    def log_llm_output(self, message: Any) -> None:
        self._title(
            f"LLM CALL #{self.llm_call_count}  ◀  software_developer_assistant — OUTPUT",
            char="─",
        )

        tool_calls = getattr(message, "tool_calls", None) or []
        fc = (getattr(message, "additional_kwargs", {}) or {}).get("function_call")

        if tool_calls or fc:
            calls = tool_calls
            if not calls and fc:
                try:
                    args = json.loads(fc.get("arguments", "{}"))
                except Exception:
                    args = fc.get("arguments", "")
                calls = [{"name": fc.get("name", "?"), "args": args}]

            self._w(f"  Decision : CALL TOOL(S)  ({len(calls)} call(s))")
            self._w()

            for tc in calls:
                self.tool_call_count += 1
                name = tc.get("name") or tc.get("function", {}).get("name", "?")
                args = tc.get("args", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        pass

                self._w(f"  ┌── TOOL CALL #{self.tool_call_count}: {name}")
                args_json = json.dumps(args, indent=4, ensure_ascii=False) if isinstance(args, dict) else str(args)
                self._w(self._indent(self._truncate(args_json, 4000), 6))
                self._w(f"  └──")
                self._w()
        else:
            content = getattr(message, "content", "") or ""
            self._w("  Decision : RESPOND TO USER (no tool call)")
            self._w()
            self._w("  Response text:")
            self._w(self._indent(str(content), 4))
            self._w()

    # ------------------------------------------------------------------ tool result

    def log_tool_result(self, tool_name: str, result: str) -> None:
        self._w()
        self._title(f"TOOL RESULT  ◀  {tool_name}", char="·")
        self._w(f"  {result}")
        self._w()

    # ------------------------------------------------------------------ errors

    def log_gemini_error(self, error: str) -> None:
        self._w()
        self._rule("!")
        self._w("  GEMINI API ERROR")
        self._rule("!")
        self._w(self._indent(self._truncate(error, 1000)))

        if "503" in error or "UNAVAILABLE" in error:
            self._w()
            self._w("  Diagnosis : Gemini 503 — transient high-demand spike.")
            self._w("  Action    : Retry the request. Usually resolves in seconds.")
        elif "429" in error or "RESOURCE_EXHAUSTED" in error:
            self._w()
            self._w("  Diagnosis : Gemini 429 — rate limit hit.")
            self._w("  Action    : Wait a moment before retrying.")
        elif "401" in error or "403" in error or "API_KEY" in error.upper():
            self._w()
            self._w("  Diagnosis : Auth error — check GOOGLE_API_KEY in .env.")

        self._rule("!")
        self._w()

    def log_general_error(self, error: str) -> None:
        self._w()
        self._rule("!")
        self._w("  RUNTIME ERROR")
        self._rule("!")
        self._w(self._indent(self._truncate(error, 2000)))
        self._rule("!")
        self._w()

    # ------------------------------------------------------------------ completion

    def log_completion(self) -> None:
        self._w()
        self._rule()
        self._w("  RUN COMPLETE")
        self._w(f"  LLM calls  : {self.llm_call_count}")
        self._w(f"  Tool calls : {self.tool_call_count}")
        self._w(f"  Finished   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._w(f"  Log file   : {self.log_path}")
        self._rule()

    def close(self) -> None:
        self._f.close()
        remove(self.request_id)
