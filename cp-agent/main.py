"""CP Agent Multi-turn chatbot using the pull-based ``on_run`` handler model."""
# Author: yuliasky

import json
from datetime import datetime, timedelta, timezone
from typing import Any

import requests
from cwai.adk import (
    AssistantMessage,
    Context,
    Message,
    MessageFormat,
    Result,
    agent,
    traced_span,
)
from cwai.adk.tools import ToolExecutor
from opentelemetry import trace


CROSSWORK_BASE_URL = "https://172.20.163.100:30603"
CROSSWORK_USERNAME = "admin"
CROSSWORK_PASSWORD = "cRo55work!"
REQUEST_TIMEOUT_SECONDS = 30

GOODBYE_INSTRUCTION = (
    "Reply to the user's farewell with one short, warm goodbye message. "
    "Do not summarize or continue any earlier conversation."
)

PLAN_FILE_PROMPT = (
    "Please specify what **archive** and **plan file** you want to use.\n\n"
    "- If you want me to list the available archives, "
    "please enter **`list archives`**.\n\n"
    "- If you want me to list the available plan files within an archive, "
    "please enter **`list plan files for archive <archive-name>`**."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_plan_file",
            "description": "Get the archive and plan file from the user's prompt.",
            "parameters": {
                "type": "object",
                "properties": {
                    "archive": {"type": "string", "description": "Archive name"},
                    "plan_file": {"type": "string", "description": "Plan file name"},
                },
                "required": ["archive", "plan_file"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_archives",
            "description": (
                "Retrieve the available archives from the Crosswork Planning REST API."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_plan_files",
            "description": (
                "Retrieve the available plan files for a specific archive from the "
                "Crosswork Planning REST API. If the user requests a relative time "
                "range such as 'last 6 hours', 'last one day', or 'last 2 weeks', "
                "extract it into lookback_value and lookback_unit. Omit both "
                "lookback fields when the user does not request a time range."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "archive": {"type": "string", "description": "Archive name"},
                    "lookback_value": {
                        "type": "integer",
                        "minimum": 1,
                        "description": (
                            "Positive quantity in the user's requested relative "
                            "time range. Convert words such as 'one' to 1."
                        ),
                    },
                    "lookback_unit": {
                        "type": "string",
                        "enum": ["minute", "hour", "day", "week", "month", "year"],
                        "description": (
                            "Singular unit for lookback_value. A month is treated "
                            "as 30 days and a year as 365 days."
                        ),
                    },
                },
                "required": ["archive"],
                "additionalProperties": False,
            },
        },
    },
]


def _crosswork_token(session: requests.Session) -> str:
    """Authenticate with Crosswork and return a JWT."""
    response = session.post(
        f"{CROSSWORK_BASE_URL}/crosswork/sso/v1/tickets",
        data={"username": CROSSWORK_USERNAME, "password": CROSSWORK_PASSWORD},
        headers={
            "content-type": "application/x-www-form-urlencoded",
            "accept": "text/plain",
        },
        verify=False,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()

    response = session.post(
        f"{CROSSWORK_BASE_URL}/crosswork/sso/v2/tickets/jwt",
        data={
            "service": f"{CROSSWORK_BASE_URL}/app-dashboard",
            "tgt": response.text,
        },
        headers={"content-type": "application/x-www-form-urlencoded"},
        verify=False,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.text


def _crosswork_get(path: str) -> dict[str, Any]:
    """Make an authenticated GET request to the Crosswork API."""
    with requests.Session() as session:
        token = _crosswork_token(session)
        response = session.get(
            f"{CROSSWORK_BASE_URL}{path}",
            headers={
                "accept": "application/json",
                "authorization": f"Bearer {token}",
            },
            verify=False,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json()


@traced_span("tool.dispatch", include_args=["name"], include_return=True)
def execute_tool(name: str, arguments: dict[str, Any]) -> str:
    """Dispatch a local tool call and serialize its result."""
    if name == "get_plan_file":
        return json.dumps(
            {
                "archive": arguments["archive"],
                "plan_file": arguments["plan_file"],
                "found": "yes",
            }
        )

    if name == "list_archives":
        return json.dumps(_crosswork_get("/cp/apiGateway/v1/archives"))

    if name == "list_plan_files":
        archive = arguments["archive"]
        return json.dumps(
            _crosswork_get(f"/cp/apiGateway/v1/archives/{archive}/plans")
        )

    span = trace.get_current_span()
    span.set_status(trace.StatusCode.ERROR, f"Unknown tool: {name}")
    return json.dumps({"error": f"Unknown tool: {name}"})


def _message_from_reply(original: Message, reply: Any) -> Message:
    """Convert a user reply into the next message handled by the loop."""
    return Message(
        type=original.type,
        sender="user",
        user_message=reply.text,
        query=reply.data if reply.data else {"prompt": reply.text},
    )


async def _ask(ctx: Context, current: Message, text: str, *, markdown: bool = False) -> Message:
    """Ask the user a question and build the next loop message."""
    message = (
        AssistantMessage(content=text, format=MessageFormat.MARKDOWN)
        if markdown
        else AssistantMessage(content=text)
    )
    reply = await ctx.user.ask(message)
    return _message_from_reply(current, reply)


async def _ask_for_plan_file(ctx: Context, current: Message) -> Message:
    ctx.log.info("Requesting user to specify archive and plan file")
    return await _ask(ctx, current, PLAN_FILE_PROMPT, markdown=True)


def _execute_local_tool_calls(
    ctx: Context,
    tool_calls: list[dict[str, Any]],
) -> list[tuple[str, dict[str, Any], str]]:
    """Execute local tool calls and return their names, arguments, and results."""
    executed: list[tuple[str, dict[str, Any], str]] = []

    with ctx.tracer.start_as_current_span("tool.execute") as span:
        span.set_attribute("tool.count", len(tool_calls))

        for tool_call in tool_calls:
            function = tool_call["function"]
            name = function["name"]
            arguments = json.loads(function["arguments"])
            ctx.log.info(
                "Executing tool",
                extra={"tool": name, "arguments": arguments},
            )
            result = execute_tool(name, arguments)
            ctx.log.info(
                "Tool execution completed",
                extra={"tool": name, "result": result},
            )
            executed.append((name, arguments, result))

    return executed


def _get_executed_tool(
    executed: list[tuple[str, dict[str, Any], str]],
    name: str,
) -> tuple[dict[str, Any], str]:
    """Return the arguments and result for a named executed tool."""
    for tool_name, arguments, result in executed:
        if tool_name == name:
            return arguments, result

    raise ValueError(f"Expected tool was not executed: {name}")


def _api_list(response: dict[str, Any], key: str) -> list[dict[str, Any]]:
    """Validate a successful Crosswork response and return its requested list."""
    if response.get("statusCode") != 200:
        raise ValueError(
            f"Crosswork API returned status code {response.get('statusCode')!r}"
        )

    data = response.get("data")
    items = data.get(key) if isinstance(data, dict) else None
    if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
        raise ValueError(f"Crosswork API response does not contain data.{key}")

    return items


def _archive_table_rows(archives: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select and label archive fields for display."""
    return [
        {
            "Name": archive.get("name"),
            "Created": archive.get("created"),
            "Last updated": archive.get("lastUpdated"),
            "Plan file count": archive.get("planFileCount"),
        }
        for archive in archives
    ]


def _plan_file_table_rows(
    plan_files: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Select plan-file fields and remove archive directories from each path."""
    return [
        {
            "ID": plan_file.get("id"),
            "Plan file": str(plan_file.get("path", "")).rsplit("/", maxsplit=1)[-1],
            "Size (bytes)": plan_file.get("size"),
            "Timestamp": plan_file.get("timestamp"),
        }
        for plan_file in plan_files
    ]


def _filter_plan_files(
    plan_files: list[dict[str, Any]],
    arguments: dict[str, Any],
    *,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Filter plan files using the relative time range extracted by the LLM."""
    value = arguments.get("lookback_value")
    unit = arguments.get("lookback_unit")

    if value is None and unit is None:
        return plan_files
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("lookback_value must be a positive integer")

    unit_seconds = {
        "minute": 60,
        "hour": 60 * 60,
        "day": 24 * 60 * 60,
        "week": 7 * 24 * 60 * 60,
        "month": 30 * 24 * 60 * 60,
        "year": 365 * 24 * 60 * 60,
    }
    if unit not in unit_seconds:
        raise ValueError(f"Unsupported lookback_unit: {unit!r}")

    current_time = now or datetime.now(timezone.utc)
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)
    cutoff = current_time - timedelta(seconds=value * unit_seconds[unit])

    filtered = []
    for plan_file in plan_files:
        timestamp = plan_file.get("timestamp")
        if not isinstance(timestamp, str):
            continue

        try:
            plan_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError:
            continue

        if plan_time.tzinfo is None:
            plan_time = plan_time.replace(tzinfo=timezone.utc)
        if plan_time >= cutoff:
            filtered.append(plan_file)

    return filtered


def _lookback_label(arguments: dict[str, Any]) -> str | None:
    """Create a readable label for an extracted relative time range."""
    value = arguments.get("lookback_value")
    unit = arguments.get("lookback_unit")
    if value is None or unit is None:
        return None

    suffix = "" if value == 1 else "s"
    return f"last {value} {unit}{suffix}"


def _markdown_value(value: Any) -> str:
    """Convert a value to a Markdown-table-safe string."""
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).replace("|", r"\|").replace("\r\n", "<br>").replace("\n", "<br>")


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    """Render dictionaries as a deterministic GitHub-style Markdown table."""
    if not rows:
        return "_No results found._"

    columns = list(dict.fromkeys(key for row in rows for key in row))
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| "
        + " | ".join(_markdown_value(row.get(column)) for column in columns)
        + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


@agent.on_run("on-run-chatbot")
async def run(ctx: Context, msg: Message) -> Result:
    history: list[dict[str, str]] = ctx.state.setdefault("history", [])

    # Tool discovery is needed only once for this conversation run.
    discovered_tools = await ctx.tools.list_tools()
    executor = ToolExecutor(discovered_tools)
    ctx.log.info(
        "Discovered tools via MCP",
        extra={"count": len(executor.tools), "tools": discovered_tools},
    )

    while True:
        prompt = msg.query.get("prompt", "")
        normalized_prompt = prompt.lower()
        history.append({"role": "user", "content": prompt})
        plan_file = ctx.state.get("plan_file")

        if "bye" in normalized_prompt:
            goodbye_messages = [
                {"role": "system", "content": GOODBYE_INSTRUCTION},
                {"role": "user", "content": prompt},
            ]
            try:
                response = await ctx.llm.complete(messages=goodbye_messages)
                goodbye = response.content or "Goodbye! Feel free to return anytime."
            except Exception:
                ctx.log.exception("Failed to generate an LLM goodbye response")
                goodbye = "Goodbye! Feel free to return anytime."

            return Result(
                message=AssistantMessage(
                    content=goodbye,
                    format=MessageFormat.MARKDOWN,
                )
            )

        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        tool_calls: list[dict[str, Any]] = []
        content = ""

        if plan_file:
            response = await ctx.llm.complete(
                messages=messages,
                tools=executor.schemas,
                tool_choice="auto",
            )
            tool_calls = response.message.get("tool_calls") or []
            if not tool_calls:
                return Result(
                    message=AssistantMessage(content=response.content or ""),
                )
        elif any(term in normalized_prompt for term in ("plan file", "list")):
            response = await ctx.llm.complete(
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )
            tool_calls = response.message.get("tool_calls") or []
            if not tool_calls:
                return Result(
                    message=AssistantMessage(content=response.content or ""),
                )
        else:
            response = await ctx.llm.complete(messages=history)
            content = response.content or ""
            history.append({"role": "assistant", "content": content})

        if "planning" in normalized_prompt:
            msg = await _ask_for_plan_file(ctx, msg)
            continue

        if "list archives" in normalized_prompt:
            await ctx.user.notify(AssistantMessage(content="Listing archives..."))
            executed = _execute_local_tool_calls(ctx, tool_calls)
            _, result = _get_executed_tool(executed, "list_archives")
            archive_data = json.loads(result)
            rows = _archive_table_rows(_api_list(archive_data, "archives"))
            table = _markdown_table(rows)
            ctx.log.info("List of archives was successfully retrieved")
            await ctx.user.notify(
                AssistantMessage(
                    content=f"Here you go!\n\n{table}",
                    format=MessageFormat.MARKDOWN,
                )
            )
            msg = await _ask_for_plan_file(ctx, msg)
            continue

        if "list plan files" in normalized_prompt:
            await ctx.user.notify(AssistantMessage(content="Listing plan files..."))
            executed = _execute_local_tool_calls(ctx, tool_calls)
            arguments, result = _get_executed_tool(executed, "list_plan_files")
            archive = arguments["archive"]
            plan_data = json.loads(result)
            plan_files = _api_list(plan_data, "planfiles")
            filtered_plan_files = _filter_plan_files(plan_files, arguments)
            rows = _plan_file_table_rows(filtered_plan_files)
            table = _markdown_table(rows)
            lookback = _lookback_label(arguments)
            title = f"Plan files for archive `{archive}`"
            if lookback:
                title += f" from the {lookback}"
            ctx.log.info(
                "List of plan files was successfully retrieved",
                extra={
                    "archive": archive,
                    "lookback": lookback,
                    "total_count": len(plan_files),
                    "filtered_count": len(filtered_plan_files),
                },
            )
            await ctx.user.notify(
                AssistantMessage(
                    content=f"{title}:\n\n{table}",
                    format=MessageFormat.MARKDOWN,
                )
            )
            msg = await _ask_for_plan_file(ctx, msg)
            continue

        if "plan file" in normalized_prompt and "archive" in normalized_prompt:
            await ctx.user.notify(AssistantMessage(content="Fetching the plan file..."))
            executed = _execute_local_tool_calls(ctx, tool_calls)
            _, result = _get_executed_tool(executed, "get_plan_file")
            result_data = json.loads(result)
            ctx.state["plan_file"] = result_data["plan_file"]
            ctx.log.info("Plan file successfully fetched")

            msg = await _ask(
                ctx,
                msg,
                (
                    f"The plan file `{ctx.state['plan_file']}` was successfully fetched."
                    "\n\n**Please describe your simulation scenario:**"
                ),
                markdown=True,
            )
            continue

        if plan_file:
            await ctx.user.notify(
                AssistantMessage(
                    content=(
                        "Processing simulation scenario using the user's specified "
                        f"plan file: `{plan_file}`."
                    ),
                    format=MessageFormat.MARKDOWN,
                )
            )
            messages.append(response.message)
            results = await executor.execute(tool_calls)
            messages.extend(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": result,
                }
                for tool_call, result in zip(tool_calls, results)
            )

            final = await ctx.llm.complete(messages=messages)
            await ctx.user.notify(
                AssistantMessage(
                    content=final.content or "",
                    format=MessageFormat.MARKDOWN,
                )
            )
            ctx.log.info("Waiting for next simulation scenario")
            msg = await _ask(
                ctx,
                msg,
                "Please describe your **next simulation scenario**. Otherwise, enter Bye!",
                markdown=True,
            )
            continue

        msg = await _ask(ctx, msg, content, markdown=True)


if __name__ == "__main__":
    agent.start()
