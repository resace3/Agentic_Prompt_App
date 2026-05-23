# Agentic Prompt App

Prompt Flow is a local Flask app for saved AI chats with read-only Home Assistant context.

## Setup

1. Add your model provider keys to `/config/secrets.yaml`:

   ```yaml
   openai_api_key: sk-...
   claude_api_key: sk-ant-...
   ```

2. Install and start the add-on.
3. Open the add-on using Ingress or port `5000`.

## Storage

The add-on stores its editable data in `/data`:

- `/data/chat_history.json`
- `/data/sensor_map.json`

## Home Assistant Access

The add-on uses the Supervisor-provided token to call the Home Assistant REST API at:

```text
http://supervisor/core/api
```

It does not read or write the Home Assistant SQLite database.
