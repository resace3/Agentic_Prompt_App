# Agentic Prompt App

Home Assistant add-on repository for Prompt Flow, a saved-chat prompt app that can use OpenAI or Claude models with read-only Home Assistant context, mapped sensors, deterministic plots, Markdown tables, LaTeX, and generated analysis visuals.

## Install In Home Assistant

Add this repository as a Home Assistant add-on repository:

```text
https://github.com/resace3/Agentic_Prompt_App
```

Then install **Agentic Prompt App** from the add-on store.

## Repository Layout

```text
Agentic_Prompt_App/
├── repository.yaml
└── agentic_prompt_app/
    ├── config.yaml
    ├── build.yaml
    ├── Dockerfile
    ├── run.sh
    ├── app.py
    ├── requirements.txt
    ├── static/
    ├── templates/
    └── tests/
```

Home Assistant expects `repository.yaml` at the repository root and the add-on files inside the `agentic_prompt_app/` folder.

## API Keys

Add provider keys to `/config/secrets.yaml` in Home Assistant:

```yaml
openai_api_key: sk-...
claude_api_key: sk-ant-...
```

Restart the add-on after changing secrets. The app reports whether OpenAI, Claude, both, or neither provider is configured without printing secret values.

## Sensor Map

The app keeps user sensor mappings in persistent add-on data under `/data`, not in the repository. Updating the add-on should not overwrite a user sensor map.

Use the Sensor Maps tab to manage mapped entities. The app can search Home Assistant recorder data and suggest sensor-map additions, but it should ask before adding a sensor.

## Diagnostics

Useful endpoints:

```text
/api/health
/api/diagnostics
/api/config-status
```

These return JSON only. They check static assets, ingress configuration, API key setup, `/data` writability, and Home Assistant API reachability.

## Development

From the add-on directory:

```sh
cd agentic_prompt_app
python -m pip install -r requirements.txt
PYTHONPATH=$PWD pytest -q
python -m flask --app app run --host 127.0.0.1 --port 5056
```

The add-on runtime is started by `run.sh` and binds Flask to `0.0.0.0:5000`, matching `config.yaml` ingress settings.

## CI

GitHub Actions runs:

- Ruff linting and format checks.
- Static asset syntax checks.
- Flask app boot and health checks.
- Unit tests with fake recorder data.
- Browser UI overflow and rendering checks.
- Docker builds for supported architectures.
- Home Assistant supervisor add-on install smoke tests.

Browser checks cover Home Assistant ingress-style layout, tab sizing, input-bar visibility, static CSS/JS loading, generated plots, DAGs, images, Markdown, code blocks, tables, and LaTeX containment.

## Updating The Add-on

In Home Assistant SSH/terminal:

```sh
ha store reload
ha addons update local_agentic_prompt_app
ha addons restart local_agentic_prompt_app
```

Depending on how the repository is installed, the slug may be `agentic_prompt_app` instead of `local_agentic_prompt_app`.

