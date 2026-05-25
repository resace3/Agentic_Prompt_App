# JITAI Reviewer Checklist

Use this checklist for each worker branch before recommending any merge into
`JITAI_dev`. Do not approve or merge until the branch diff, tests, and risk areas
below have been inspected.

## Branch Under Review

- Branch:
- Base branch: `JITAI_dev`
- Reviewer:
- Date:
- Diff inspected:
- CI/workflows inspected:
- Local tests run:
- Verdict: Pending / Needs changes / Ready for human merge

## Required Git Safety Checks

- Confirm review work is performed only from
  `/addons/dev/Agentic_Prompt_App.worktrees/jitai-reviewer`.
- Run `git branch --show-current` before any commit, push, rebase, or merge.
- Stop if the current branch is not `agent/jitai-reviewer`.
- Confirm the worker branch does not target, modify, push, merge, or checkout
  `main` or `dev`.
- Confirm no merge into `JITAI_dev` is performed without explicit instruction.

## Required Diff Review

- Compare the worker branch against `JITAI_dev`.
- Identify files added, modified, deleted, or moved.
- Confirm the repository/add-on structure remains:
  - `repository.yaml`
  - `agentic_prompt_app/`
- Confirm add-on files remain under `agentic_prompt_app/`, including
  `config.yaml`, `Dockerfile`, `run.sh`, `templates/`, `static/`, and `tests/`.
- Confirm no real Home Assistant recorder DB, `/data` state, secrets, tokens,
  generated logs, traces, reports, or large artifacts are committed.

## Home Assistant Add-on Compatibility

- `agentic_prompt_app/config.yaml` keeps `ingress: true`.
- `ingress_port` matches the Flask runtime port.
- Add-on runtime binds Flask to `0.0.0.0`.
- Docker image copies static assets, templates, app code, config metadata, and
  an executable `run.sh`.
- Template asset links use Flask `url_for`, not hardcoded `/static/...` paths.
- API routes under `/api/*` return JSON-only success and error responses.

## Security Review

- Home Assistant API access uses `SUPERVISOR_TOKEN` only.
- No fallback to user-provided long-lived HA tokens is introduced.
- `/config/secrets.yaml` is not printed, modified, or committed.
- Error messages identify missing providers or setup steps without exposing key
  values.
- Logs, test output, workflow artifacts, and frontend text do not expose
  secrets or tokens.
- Workflow permissions remain read-only unless a human-approved need is
  documented.
- GitHub Actions do not use floating action refs such as `@main`, `@master`, or
  unpinned refs.

## Recorder DB And Persistence Review

- Home Assistant recorder DB access is read-only, preferably SQLite URI
  `mode=ro`.
- Queries use parameters for entity IDs, windows, limits, and filters.
- Recorder reads exclude invalid states: `unknown`, `unavailable`, empty, and
  null.
- Query results include entity IDs, time windows, units when available, sample
  counts, and data source.
- User state is stored in `/data` or test-specific temporary paths.
- Add-on updates do not overwrite sensor maps, saved chats, or runtime history.
- Tests use fake temporary recorder data, not real Home Assistant databases.

## JITAI Logic And Statistics Review

- JITAI recommendations distinguish observed association from causation.
- Predictive models are described as predictive, not causal.
- DAGs are framed as assumptions unless causal identification is explicitly
  justified.
- Intervention timing, eligibility, cooldowns, and decision rules are
  deterministic or auditable.
- Missing data, sparse data, outliers, and invalid states are handled without
  producing misleading recommendations.
- Statistical summaries report sample size, time window, and uncertainty or
  limitations where relevant.
- Learning behavior is bounded, testable, and does not silently persist unsafe
  state outside `/data`.

## UI And Ingress Review

- The app shell fits inside the Home Assistant ingress viewport.
- The message pane scrolls internally.
- Prompt input and send button remain visible across short, narrow, and zoomed
  viewports.
- Prompts and Sensor Maps tabs remain compact and horizontal after tab switches
  and browser resizing.
- Plots, DAGs, SVGs, images, Markdown tables, code blocks, and LaTeX stay inside
  assistant response cards.
- CSS changes preserve `height`, `min-height`, `overflow`, `position`, nested
  flex behavior, and `min-height: 0` where needed.
- Frontend JavaScript has no syntax errors and avoids unsafe HTML insertion for
  untrusted model/user content.

## Tests And Commands To Consider

Run the focused subset that matches the branch risk. For final integration
readiness, prefer the full suite where practical.

```sh
cd agentic_prompt_app
python -m ruff check .
python -m ruff format --check .
node --check static/app.js
PYTHONPATH=$PWD pytest -q
```

For UI, layout, or rendering changes:

```sh
cd agentic_prompt_app
python -m flask --app app run --host 127.0.0.1 --port 5056
RUN_BROWSER_TESTS=1 BROWSER_BASE_URL=http://127.0.0.1:5056 pytest tests/test_browser_ui.py -q
```

For recorder DB, analysis, plotting, or JITAI learning changes, require focused
tests with fake temporary data that validate:

- read-only SQLite behavior;
- invalid-state filtering;
- selected entity IDs;
- time windows and units;
- deterministic plot or analysis specs;
- association/prediction/causal wording boundaries.

## Review Report Template

### Summary

- Branch:
- Scope:
- Verdict:

### Findings

- Severity:
- File/line:
- Issue:
- Required fix:

### Tests

- Passed:
- Failed:
- Not run:

### Security And HA Notes

- Supervisor token only:
- Recorder DB read-only:
- Secrets safe:
- Ingress/add-on compatible:

### Merge Recommendation

- Ready for human-reviewed merge into `JITAI_dev`: Yes / No
- Required fixes before merge:
- Follow-up risks:
