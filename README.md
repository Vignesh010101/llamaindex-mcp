# Build your own Local MCP Client with LlamaIndex

This project demonstrates how to build a **local MCP (Model Context Protocol) client** using LlamaIndex. The client connects to a local MCP server (which exposes tools like a SQLite database) and lets you interact with it using natural language and tool-calling agents—all running locally on your machine.


### Setup

To sync dependencies, run:

```sh
uv sync
```

---

## Usage

- Start the local MCP server (for example, the included SQLite demo server):

```sh
uv run server.py --server_type=sse
```

- Run the client (choose the appropriate client script, e.g. `client.py` for OpenAI or `ollama_client.py` for Ollama):

```sh
uv run client.py
```

- Interact with the agent in your terminal. Type your message and the agent will use the available tools to answer your queries.
