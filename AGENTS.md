# AGENTS.md

This repository is a Home Assistant add-on repository. Keep the root/add-on structure intact:

```text
repository.yaml
agentic_prompt_app/
```

Do not move `config.yaml`, `Dockerfile`, `run.sh`, templates, static assets, or tests back to the repository root.

## Working Directory

Most code and test commands should run from:

```sh
cd agentic_prompt_app
```

## Common Commands

```sh
python -m ruff check .
python -m ruff format --check .
node --check static/app.js
PYTHONPATH=$PWD pytest -q
python -m flask --app app run --host 127.0.0.1 --port 5056
```

For browser tests in CI, GitHub Actions installs Playwright and runs:

```sh
RUN_BROWSER_TESTS=1 BROWSER_BASE_URL=http://127.0.0.1:5056 pytest tests/test_browser_ui.py -q
```

## Home Assistant Add-on Rules

- `agentic_prompt_app/config.yaml` must keep `ingress: true`.
- `ingress_port` must match the Flask port.
- Flask must bind to `0.0.0.0` in add-on runtime.
- Static assets and templates must be copied into the Docker image.
- Template asset links should use Flask `url_for`, not hardcoded `/static/...` paths.
- API routes under `/api/*` should return JSON only, including errors.

## Security Rules

- Never print API keys, secrets, tokens, or full `/config/secrets.yaml` contents.
- Do not commit real Home Assistant recorder databases.
- Do not commit user-specific persistent `/data` contents.
- Error messages may say which provider is missing, but must not expose key values.

## Persistence Rules

User state belongs in `/data`, especially:

- Sensor map data.
- Saved chats.
- Runtime history.

Add-on updates must not overwrite a user's sensor map. Tests should use temporary fake data, not repository fixtures that replace user data.

## UI And Ingress Rules

Home Assistant ingress has tight viewport constraints. Preserve these layout expectations:

- The app shell fits inside the ingress viewport.
- The message pane scrolls internally.
- The prompt input and send button stay visible.
- Generated plots, DAGs, SVGs, images, Markdown tables, code blocks, and LaTeX stay inside the assistant response card.
- The Prompts and Sensor Maps tabs remain compact and horizontal after tab switches and browser resizing.

When changing CSS, check `height`, `min-height`, `overflow`, `position`, nested flex children, and `min-height: 0`.

## Analysis And Plotting Rules

The LLM may interpret intent, but plotted data should come from deterministic Home Assistant recorder queries or simulated test data.

When adding analysis features:

- Include the selected entity IDs.
- Include the time window and units.
- Exclude invalid states such as `unknown`, `unavailable`, empty, and null.
- Frame correlations as associations, not causation.
- Frame predictive models as predictive, not causal.
- Treat DAGs as assumptions unless the design supports causal claims.

## Testing Expectations

For changes that touch analysis, plotting, ingress layout, static assets, or persistence, add or update focused tests.

Useful coverage areas:

- Static CSS/JS reachability.
- API JSON-only errors.
- Missing provider keys.
- Provider/model mismatch.
- Fake recorder DB queries.
- Sensor map persistence.
- Plot type detection and plot specs.
- Browser overflow regressions.
- Add-on install/build smoke tests.

