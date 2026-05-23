# Agentic Prompt App

Home Assistant add-on for Prompt Flow, a saved-chat prompt app with read-only Home Assistant API context.

## Add-on Files

This directory is structured as a Home Assistant add-on:

- `config.yaml` contains add-on metadata and Supervisor permissions.
- `build.yaml` defines multi-architecture base images.
- `Dockerfile` builds the Flask app container.
- `run.sh` starts the app and points persistent data to `/data`.
- `DOCS.md` appears in the Home Assistant add-on documentation tab.

## Local Development

```sh
python app.py
```

The add-on runtime expects Home Assistant to provide `SUPERVISOR_TOKEN`.
