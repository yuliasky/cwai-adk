# CP Agent Multi-Turn Chatbot (on_run)

Multi-turn chatbot using `@agent.on_run` with `ctx.user.ask()`.
Demonstrates interaction with Crosswork Planning for agentic-based network planning use cases, through a pull-based handler model, where the handler owns the conversation loop.

## What It Does

1. Discover MCP tools via `ctx.tools.list_tools()`
2. Interacts with CP via local tools using REST API calls
3. Receives a `{"prompt": "..."}` request
4. Sends the prompt (with conversation history) to the LLM
5. Sends the response back to the user via `ctx.user.notify()`
6. Waits for the next message with `ctx.user.ask()`
7. Repeats until the user says "bye", then returns a `Result`

Conversation history is stored in `ctx.state["history"]` and accumulated
across turns(*).

## Requirements
- Integration with external MCP server using CP SDK

(Note *) Not in all cases.
