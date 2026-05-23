import base64
import copy
import io
import json
import mimetypes
import os
import re
import secrets
import shutil
import urllib.error
import urllib.request
import urllib.parse
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from statistics import mean
from zoneinfo import ZoneInfo

import yaml
from flask import Flask, jsonify, render_template, request, session
from openai import OpenAI
from werkzeug.exceptions import HTTPException


DEFAULT_PROVIDER = "openai"
DEFAULT_MODEL = "gpt-4.1-nano"
MAX_PROMPT_CHARS = 20000
SECRET_PROVIDERS = {
    "openai": {
        "label": "OpenAI",
        "secret_name": "openai_api_key",
        "env_names": ["OPENAI_API_KEY"],
    },
    "anthropic": {
        "label": "Claude",
        "secret_name": "claude_api_key",
        "env_names": ["ANTHROPIC_API_KEY", "CLAUDE_API_KEY"],
    },
}
MODEL_PRICING_SOURCE = "https://platform.openai.com/docs/pricing"
MODEL_PRICING_NOTE = "Standard short-context API pricing per 1M tokens."
AVAILABLE_MODELS = [
    {
        "id": "gpt-4.1-nano",
        "label": "GPT-4.1 Nano",
        "input_per_1m": 0.10,
        "cached_input_per_1m": 0.025,
        "output_per_1m": 0.40,
        "note": MODEL_PRICING_NOTE,
    },
    {
        "id": "gpt-4.1-mini",
        "label": "GPT-4.1 Mini",
        "input_per_1m": 0.40,
        "cached_input_per_1m": 0.10,
        "output_per_1m": 1.60,
        "note": MODEL_PRICING_NOTE,
    },
    {
        "id": "gpt-4.1",
        "label": "GPT-4.1",
        "input_per_1m": 2.00,
        "cached_input_per_1m": 0.50,
        "output_per_1m": 8.00,
        "note": MODEL_PRICING_NOTE,
    },
    {
        "id": "gpt-5.5",
        "label": "GPT 5.5",
        "input_per_1m": 5.00,
        "cached_input_per_1m": 0.50,
        "output_per_1m": 30.00,
        "note": MODEL_PRICING_NOTE,
    },
    {
        "id": "gpt-5.5-pro",
        "label": "GPT 5.5 Pro",
        "input_per_1m": 30.00,
        "cached_input_per_1m": None,
        "output_per_1m": 180.00,
        "note": MODEL_PRICING_NOTE,
    },
    {
        "id": "gpt-5.4",
        "label": "GPT 5.4",
        "input_per_1m": 2.50,
        "cached_input_per_1m": 0.25,
        "output_per_1m": 15.00,
        "note": MODEL_PRICING_NOTE,
    },
    {
        "id": "gpt-5.4-mini",
        "label": "GPT 5.4 Mini",
        "input_per_1m": 0.75,
        "cached_input_per_1m": 0.075,
        "output_per_1m": 4.50,
        "note": MODEL_PRICING_NOTE,
    },
    {
        "id": "gpt-5.4-nano",
        "label": "GPT 5.4 Nano",
        "input_per_1m": 0.20,
        "cached_input_per_1m": 0.02,
        "output_per_1m": 1.25,
        "note": MODEL_PRICING_NOTE,
    },
    {
        "id": "gpt-5.4-pro",
        "label": "GPT 5.4 Pro",
        "input_per_1m": 30.00,
        "cached_input_per_1m": None,
        "output_per_1m": 180.00,
        "note": MODEL_PRICING_NOTE,
    },
]
DEFAULT_CLAUDE_MODEL = "claude-haiku-4-5-20251001"
CLAUDE_PRICING_SOURCE = "https://platform.claude.com/docs/en/about-claude/pricing"
CLAUDE_PRICING_NOTE = "Anthropic API pricing per 1M tokens; cached price is cache hits/refreshes."
CLAUDE_MODELS = [
    {
        "id": "claude-haiku-4-5-20251001",
        "label": "Claude Haiku 4.5",
        "input_per_1m": 1.00,
        "cached_input_per_1m": 0.10,
        "output_per_1m": 5.00,
        "note": CLAUDE_PRICING_NOTE,
    },
    {
        "id": "claude-sonnet-4-20250514",
        "label": "Claude Sonnet 4",
        "input_per_1m": 3.00,
        "cached_input_per_1m": 0.30,
        "output_per_1m": 15.00,
        "note": CLAUDE_PRICING_NOTE,
    },
    {
        "id": "claude-opus-4-1-20250805",
        "label": "Claude Opus 4.1",
        "input_per_1m": 15.00,
        "cached_input_per_1m": 1.50,
        "output_per_1m": 75.00,
        "note": CLAUDE_PRICING_NOTE,
    },
]
PROVIDERS = [
    {
        "id": "openai",
        "label": "OpenAI",
        "default_model": DEFAULT_MODEL,
        "pricing_source": MODEL_PRICING_SOURCE,
        "pricing_note": MODEL_PRICING_NOTE,
    },
    {
        "id": "anthropic",
        "label": "Claude",
        "default_model": DEFAULT_CLAUDE_MODEL,
        "pricing_source": CLAUDE_PRICING_SOURCE,
        "pricing_note": CLAUDE_PRICING_NOTE,
    },
]
DEFAULT_INSTRUCTIONS = (
    "You are a concise prompt assistant. Help the user refine ideas, ask useful "
    "follow-up questions when needed, and keep the conversation moving. When Home "
    "Assistant API context is provided, use it to give practical observations "
    "and advice. Do not claim you can change Home Assistant; this app has read-only "
    "Home Assistant API access. For sleep questions, prioritize completed_sleep.daily and "
    "completed_sleep.averages; do not average raw state-history rows or treat zero "
    "reset states as sleep sessions. When the user asks for specific metrics that are "
    "present in context, such as average time in bed, include those metrics explicitly "
    "instead of substituting a different metric. "
    "When mapped_sensor_history is provided, it is "
    "read-only Home Assistant history API data for entities from the Sensor Maps "
    "table; use it directly and do not ask the user to export that same sensor data."
)
HOME_ASSISTANT_API_URL = os.environ.get("HOME_ASSISTANT_API_URL", "http://supervisor/core/api").rstrip("/")
SENSOR_MAP_PATHS = [
    os.environ.get("SENSOR_MAP_PATH"),
    "/data/sensor_map.json",
    os.path.join(os.getcwd(), "sensor_map.json"),
]
SLEEP_ENTITY_TERMS = (
    "sleep",
    "asleep",
    "awake",
    "bed",
    "bedtime",
    "wake",
    "wakeup",
    "wake_up",
    "rem",
    "deep",
    "nap",
    "oura",
    "withings",
    "fitbit",
    "garmin",
)
SLEEP_ENTITY_EXCLUDE_TERMS = (
    "remote.",
    "remote_",
)
HA_REQUEST_TERMS = (
    "home assistant",
    "home-assistant",
    "ha db",
    "ha database",
    "database",
    "sleep",
    "bed",
    "asleep",
)
PLOT_REQUEST_TERMS = (
    "plot",
    "graph",
    "chart",
    "trend line",
    "visualize",
    "visualise",
)
MAX_CONTEXT_SLEEP_DAYS = 365
DEFAULT_SLEEP_DAYS = 30
WEEK_SLEEP_DAYS = 7
PREDICTOR_SLEEP_DAYS = 90
SENSOR_DATA_REQUEST_TERMS = (
    "show sensor data",
    "show me sensor data",
    "raw sensor data",
    "sensor data",
    "show rows",
    "list readings",
    "show readings",
    "recent readings",
    "latest readings",
    "look at",
    "give me info",
    "tell me about",
    "history for",
    "opened",
    "closed",
)
RELATION_REQUEST_TERMS = (
    "relation",
    "related",
    "correlate",
    "correlation",
    "predict",
    "predicts",
    "affect",
    "impact",
    "compare",
    "versus",
    " vs ",
)
ACTIVE_STATES = {"on", "open", "opened", "detected", "active", "home", "true"}
MAX_CONTEXT_SENSORS = 6
MAX_CONTEXT_SENSOR_ROWS = 2000

BASE_DIR = Path(__file__).resolve().parent


class IngressPrefixMiddleware:
    def __init__(self, app):
        self.app = app

    def __call__(self, environ, start_response):
        prefix = (
            environ.get("HTTP_X_INGRESS_PATH")
            or environ.get("HTTP_X_FORWARDED_PREFIX")
            or environ.get("HTTP_X_PROXY_PREFIX")
        )
        if prefix:
            environ["SCRIPT_NAME"] = f"/{prefix.strip('/')}"
        return self.app(environ, start_response)


app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
    static_url_path="/static",
)
app.wsgi_app = IngressPrefixMiddleware(app.wsgi_app)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", secrets.token_hex(32))

conversations = {}
chat_store_loaded_path = None
home_assistant_config_cache = {"expires": 0, "data": None}
home_assistant_history_cache = {}


class ApiError(Exception):
    def __init__(self, message, status=400, error_type="bad_request", fix_steps=None, details_safe=None):
        super().__init__(message)
        self.message = message
        self.status = status
        self.error_type = error_type
        self.fix_steps = fix_steps or []
        self.details_safe = details_safe or {}


class ProviderApiError(ApiError):
    pass


def api_error_payload(message, error_type="error", fix_steps=None, details_safe=None):
    return {
        "ok": False,
        "error": message,
        "error_type": error_type,
        "message": message,
        "fix_steps": fix_steps or [],
        "details_safe": details_safe or {},
    }


def api_error_response(message, status=400, error_type="bad_request", fix_steps=None, details_safe=None):
    return jsonify(api_error_payload(message, error_type, fix_steps, details_safe)), status


def api_success(payload=None):
    data = {"ok": True}
    if payload:
        data.update(payload)
    return data


@app.before_request
def log_static_requests():
    if request.path.startswith("/static/"):
        app.logger.info(
            "Static request path=%s script_root=%s ingress_path=%s forwarded_prefix=%s",
            request.path,
            request.script_root,
            request.headers.get("X-Ingress-Path"),
            request.headers.get("X-Forwarded-Prefix"),
        )


@app.after_request
def force_api_json_headers(response):
    if request.path.startswith("/api/"):
        response.headers.setdefault("Cache-Control", "no-store")
    return response


@app.errorhandler(HTTPException)
def handle_http_exception(exc):
    if request.path.startswith("/api/"):
        return api_error_response(
            exc.description,
            status=exc.code,
            error_type=f"http_{exc.code}",
            details_safe={"status": exc.code},
        )
    return exc


@app.errorhandler(ApiError)
def handle_api_error(exc):
    return api_error_response(
        exc.message,
        status=exc.status,
        error_type=exc.error_type,
        fix_steps=exc.fix_steps,
        details_safe=exc.details_safe,
    )


@app.errorhandler(Exception)
def handle_unexpected_exception(exc):
    if request.path.startswith("/api/"):
        app.logger.exception("Unhandled API error")
        return api_error_response(
            "Internal server error.",
            status=500,
            error_type="internal_error",
            fix_steps=["Check the add-on logs for the traceback."],
        )
    raise exc


SLEEP_METRIC_ENTITIES = {
    "minutes_asleep": "sensor.nick_r_sleep_minutes_asleep",
    "time_in_bed": "sensor.nick_r_sleep_time_in_bed",
    "minutes_awake": "sensor.nick_r_sleep_minutes_awake",
    "efficiency": "sensor.nick_r_sleep_efficiency",
    "start_time": "sensor.nick_r_sleep_start_time",
}


def secret_candidate_paths():
    return [
        os.environ.get("SECRETS_YAML"),
        "/config/secrets.yaml",
        os.path.join(os.getcwd(), "secrets.yaml"),
    ]


def configured_secret_source(secret_name, env_names):
    for env_name in env_names:
        if os.environ.get(env_name, "").strip():
            return "environment"

    for path in [p for p in secret_candidate_paths() if p]:
        if not os.path.exists(path):
            continue

        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
        except (OSError, yaml.YAMLError):
            continue

        api_key = str(data.get(secret_name) or "").strip()
        if api_key:
            return "secrets.yaml"
    return None


def load_secret_value(secret_name, env_names, provider_label):
    for env_name in env_names:
        value = os.environ.get(env_name, "").strip()
        if value:
            return value

    for path in [p for p in secret_candidate_paths() if p]:
        if not os.path.exists(path):
            continue

        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
        except (OSError, yaml.YAMLError):
            continue

        api_key = str(data.get(secret_name) or "").strip()
        if api_key:
            return api_key

    env_label = " or ".join(env_names)
    raise RuntimeError(
        f"{provider_label} API key not found. Add {secret_name} to /config/secrets.yaml or set {env_label}."
    )


def load_openai_key():
    config = SECRET_PROVIDERS["openai"]
    return load_secret_value(config["secret_name"], config["env_names"], config["label"])


def load_claude_key():
    config = SECRET_PROVIDERS["anthropic"]
    return load_secret_value(config["secret_name"], config["env_names"], config["label"])


def provider_key_status():
    providers = {}
    missing_labels = []
    configured_labels = []
    for provider_id, config in SECRET_PROVIDERS.items():
        source = configured_secret_source(config["secret_name"], config["env_names"])
        configured = bool(source)
        providers[provider_id] = {
            "label": config["label"],
            "configured": configured,
            "state": "configured_unverified" if configured else "missing",
            "source": source,
            "secret_name": config["secret_name"],
            "env_name": config["env_names"][0],
            "env_names": config["env_names"],
            "fix_steps": [
                "Open /config/secrets.yaml in Home Assistant.",
                f"Add {config['secret_name']}: sk-...",
                "Restart the Agentic Prompt App add-on.",
            ],
        }
        if configured:
            configured_labels.append(config["label"])
        else:
            missing_labels.append(config["label"])

    if not configured_labels:
        message = "No AI provider API keys are configured. Add OpenAI and/or Claude keys to secrets.yaml."
    elif missing_labels:
        message = f"You have {', '.join(configured_labels)} configured, but not {', '.join(missing_labels)}."
    else:
        message = "OpenAI and Claude API keys are configured."

    return {
        "providers": providers,
        "configured": configured_labels,
        "missing": missing_labels,
        "all_configured": not missing_labels,
        "any_configured": bool(configured_labels),
        "message": message,
        "help": {
            "path": "/config/secrets.yaml",
            "example": "openai_api_key: sk-...\nclaude_api_key: sk-ant-...",
            "note": "Add only the providers you plan to use, then restart the add-on.",
        },
    }


def secret_setup_fix_steps(provider_label=None):
    target = f" for {provider_label}" if provider_label else ""
    return [
        f"Add an API key{target} in /config/secrets.yaml.",
        "Use: openai_api_key: sk-...",
        "Use: claude_api_key: sk-ant-...",
        "Restart the Agentic Prompt App add-on.",
    ]


def get_client():
    return OpenAI(api_key=load_openai_key())


def provider_catalog():
    return [dict(provider) for provider in PROVIDERS]


def provider_by_id(provider_id):
    return next((provider for provider in PROVIDERS if provider["id"] == provider_id), None)


def provider_models(provider_id):
    if provider_id == "openai":
        return AVAILABLE_MODELS
    if provider_id == "anthropic":
        return CLAUDE_MODELS
    return []


def with_provider_fields(model, provider):
    enriched = dict(model)
    enriched["provider"] = provider["id"]
    enriched["provider_label"] = provider["label"]
    enriched["pricing_source"] = provider["pricing_source"]
    return enriched


def model_catalog(provider_id=None):
    providers = PROVIDERS if provider_id is None else [provider_by_id(provider_id)]
    catalog = []
    for provider in [item for item in providers if item]:
        catalog.extend(with_provider_fields(model, provider) for model in provider_models(provider["id"]))
    return catalog


def model_by_id(provider_id, model_id):
    provider = provider_by_id(provider_id)
    if not provider:
        return None
    model = next((model for model in provider_models(provider_id) if model["id"] == model_id), None)
    return with_provider_fields(model, provider) if model else None


def requested_provider_and_model(payload):
    provider_id = (payload.get("provider") or DEFAULT_PROVIDER).strip()
    provider = provider_by_id(provider_id)
    if not provider:
        raise ApiError(
            f"Unknown provider: {provider_id}",
            status=400,
            error_type="unknown_provider",
            details_safe={"supported_providers": [item["id"] for item in provider_catalog()]},
        )

    model_id = (payload.get("model") or provider["default_model"]).strip()
    model = model_by_id(provider["id"], model_id)
    if not model:
        matching_other_provider = next((item for item in model_catalog() if item["id"] == model_id), None)
        if matching_other_provider:
            message = (
                f"Model {model_id} belongs to {matching_other_provider['provider_label']}, not {provider['label']}."
            )
            error_type = "provider_model_mismatch"
        else:
            message = f"Unknown {provider['label']} model: {model_id}"
            error_type = "unknown_model"
        raise ApiError(
            message,
            status=400,
            error_type=error_type,
            fix_steps=[f"Choose a supported {provider['label']} model from the model selector."],
            details_safe={
                "provider": provider["id"],
                "requested_model": model_id,
                "supported_models": [item["id"] for item in provider_models(provider["id"])],
            },
        )
    return provider, model


def compact_model_info(model):
    return {
        "id": model.get("id"),
        "label": model.get("label"),
        "provider": model.get("provider"),
        "provider_label": model.get("provider_label"),
        "input_per_1m": model.get("input_per_1m"),
        "cached_input_per_1m": model.get("cached_input_per_1m"),
        "output_per_1m": model.get("output_per_1m"),
        "pricing_source": model.get("pricing_source"),
        "note": model.get("note"),
    }


def home_assistant_api_url():
    return os.environ.get("HOME_ASSISTANT_API_URL", HOME_ASSISTANT_API_URL).rstrip("/")


def home_assistant_token():
    return os.environ.get("SUPERVISOR_TOKEN") or os.environ.get("HOME_ASSISTANT_TOKEN")


def home_assistant_api_request(path, params=None, timeout=30):
    token = home_assistant_token()
    if not token:
        raise RuntimeError("SUPERVISOR_TOKEN is not available.")

    url = f"{home_assistant_api_url()}/{path.lstrip('/')}"
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    api_request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(api_request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Home Assistant API error {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Home Assistant API connection error: {exc.reason}") from exc


def home_assistant_config():
    now = datetime.now(timezone.utc).timestamp()
    if home_assistant_config_cache["data"] and home_assistant_config_cache["expires"] > now:
        return copy.deepcopy(home_assistant_config_cache["data"])
    data = home_assistant_api_request("config", timeout=10)
    home_assistant_config_cache["data"] = data
    home_assistant_config_cache["expires"] = now + 300
    return copy.deepcopy(data)


def home_assistant_available():
    if not home_assistant_token():
        return False
    try:
        home_assistant_config()
        return True
    except Exception:
        return False


def check_item(name, ok, status, message, fix_steps=None, safe_details=None):
    return {
        "name": name,
        "ok": bool(ok),
        "status": status,
        "message": message,
        "fix_steps": fix_steps or [],
        "safe_details": safe_details or {},
    }


def read_addon_config():
    path = BASE_DIR / "config.yaml"
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except Exception as exc:
        return {"_error": str(exc)}


def static_asset_check(filename, expected_content_type):
    path = BASE_DIR / "static" / filename
    exists = path.exists()
    readable = os.access(path, os.R_OK) if exists else False
    guessed_type = mimetypes.guess_type(str(path))[0] or ""
    ok = exists and readable and guessed_type == expected_content_type
    return check_item(
        f"static_{filename}",
        ok,
        "ok" if ok else "failed",
        f"{filename} is available." if ok else f"{filename} is missing, unreadable, or has the wrong content type.",
        fix_steps=[f"Ensure Dockerfile copies static/{filename} into /app/static/{filename}."],
        safe_details={
            "path": str(path),
            "exists": exists,
            "readable": readable,
            "content_type": guessed_type,
            "expected_content_type": expected_content_type,
        },
    )


def data_writable_check():
    path = Path(chat_store_path()).parent
    try:
        path.mkdir(parents=True, exist_ok=True)
        test_path = path / ".agentic_prompt_app_write_check"
        test_path.write_text("ok", encoding="utf-8")
        test_path.unlink(missing_ok=True)
        return check_item("data_writable", True, "ok", f"{path} is writable.", safe_details={"path": str(path)})
    except Exception as exc:
        return check_item(
            "data_writable",
            False,
            "failed",
            f"{path} is not writable.",
            fix_steps=["Verify the add-on has its /data volume and restart the add-on."],
            safe_details={"path": str(path), "error": str(exc)},
        )


def home_assistant_api_check():
    if not home_assistant_token():
        return check_item(
            "home_assistant_api",
            False,
            "missing_token",
            "SUPERVISOR_TOKEN is not available, so Home Assistant API context cannot be read.",
            fix_steps=["Set homeassistant_api: true in config.yaml.", "Rebuild and restart the add-on."],
            safe_details={"api_url": home_assistant_api_url()},
        )
    try:
        config = home_assistant_config()
        return check_item(
            "home_assistant_api",
            True,
            "ok",
            "Home Assistant API is reachable.",
            safe_details={"api_url": home_assistant_api_url(), "time_zone": config.get("time_zone")},
        )
    except Exception as exc:
        text = str(exc)
        status = "auth_failed" if "401" in text or "403" in text else "unreachable"
        return check_item(
            "home_assistant_api",
            False,
            status,
            "Home Assistant API could not be read.",
            fix_steps=[
                "Confirm homeassistant_api: true in config.yaml.",
                "Restart the add-on after updating permissions.",
                "Check Supervisor and add-on logs.",
            ],
            safe_details={"api_url": home_assistant_api_url(), "error": text[:500]},
        )


def config_status_payload():
    addon_config = read_addon_config()
    checks = [
        check_item(
            "ingress",
            addon_config.get("ingress") is True and addon_config.get("ingress_port") == 5000,
            "ok" if addon_config.get("ingress") is True and addon_config.get("ingress_port") == 5000 else "failed",
            "Ingress is configured for port 5000."
            if addon_config.get("ingress") is True and addon_config.get("ingress_port") == 5000
            else "Ingress must be enabled and ingress_port must be 5000.",
            fix_steps=["Set ingress: true and ingress_port: 5000 in config.yaml."],
            safe_details={"ingress": addon_config.get("ingress"), "ingress_port": addon_config.get("ingress_port")},
        ),
        check_item(
            "homeassistant_api_permission",
            addon_config.get("homeassistant_api") is True,
            "ok" if addon_config.get("homeassistant_api") is True else "failed",
            "homeassistant_api permission is enabled."
            if addon_config.get("homeassistant_api") is True
            else "homeassistant_api permission is missing.",
            fix_steps=["Set homeassistant_api: true in config.yaml."],
            safe_details={"homeassistant_api": addon_config.get("homeassistant_api")},
        ),
        check_item(
            "hassio_api_permission",
            True,
            "not_required",
            "hassio_api is not required for the current read-only Home Assistant API features.",
            safe_details={"hassio_api": addon_config.get("hassio_api", False)},
        ),
        data_writable_check(),
        home_assistant_api_check(),
    ]
    ok = all(item["ok"] or item["status"] == "not_required" for item in checks)
    return api_success(
        {
            "status": "ok" if ok else "needs_attention",
            "checks": checks,
            "message": "Configuration checks passed." if ok else "One or more configuration checks need attention.",
        }
    )


def diagnostics_payload():
    checks = [
        static_asset_check("styles.css", "text/css"),
        static_asset_check("app.js", "text/javascript"),
        check_item(
            "flask_bind",
            os.environ.get("PORT", "5000") == "5000",
            "ok" if os.environ.get("PORT", "5000") == "5000" else "warning",
            "Flask is configured for port 5000.",
            fix_steps=["Set PORT=5000 or match config.yaml ingress_port."],
            safe_details={"port": os.environ.get("PORT", "5000"), "host": "0.0.0.0"},
        ),
        check_item(
            "request_prefix",
            True,
            "observed",
            "Request prefix diagnostics captured.",
            safe_details={
                "script_root": request.script_root,
                "x_ingress_path": request.headers.get("X-Ingress-Path"),
                "x_forwarded_prefix": request.headers.get("X-Forwarded-Prefix"),
                "x_proxy_prefix": request.headers.get("X-Proxy-Prefix"),
            },
        ),
    ]
    config_payload = config_status_payload()
    checks.extend(config_payload["checks"])
    ok = all(item["ok"] or item["status"] in {"not_required", "observed"} for item in checks)
    return api_success(
        {
            "status": "ok" if ok else "needs_attention",
            "checks": checks,
            "message": "Diagnostics checks passed." if ok else "Diagnostics found setup issues.",
        }
    )


def home_assistant_db_path():
    return None


def home_assistant_time_zone():
    try:
        return home_assistant_config().get("time_zone") or "UTC"
    except Exception:
        path = "/config/.storage/core.config"
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                return data.get("data", {}).get("time_zone") or "UTC"
            except (OSError, json.JSONDecodeError):
                return "UTC"
    return "UTC"


def local_datetime(timestamp, time_zone=None):
    zone = ZoneInfo(time_zone or home_assistant_time_zone())
    return datetime.fromtimestamp(float(timestamp), timezone.utc).astimezone(zone)


def sensor_map_path():
    candidate_paths = [os.environ.get("SENSOR_MAP_PATH"), *SENSOR_MAP_PATHS]
    for path in [p for p in candidate_paths if p]:
        directory = os.path.dirname(path) or "."
        if os.path.exists(path):
            return path
        if os.path.isdir(directory) and os.access(directory, os.W_OK):
            return path
    return os.path.join(os.getcwd(), "sensor_map.json")


def should_include_home_assistant_context(user_text):
    lowered = user_text.lower()
    return any(term in lowered for term in HA_REQUEST_TERMS)


def should_make_plot(user_text):
    lowered = user_text.lower()
    return any(term in lowered for term in PLOT_REQUEST_TERMS)


def should_show_sensor_data(user_text):
    lowered = user_text.lower()
    mapped_history_terms = {"look at", "give me info", "tell me about", "history for", "opened", "closed"}
    explicit_terms = [term for term in SENSOR_DATA_REQUEST_TERMS if term not in mapped_history_terms]
    if any(term in lowered for term in explicit_terms):
        return True
    return any(term in lowered for term in mapped_history_terms) and prompt_mentions_mapped_sensor(user_text)


def requested_days_from_text(user_text, default_days=DEFAULT_SLEEP_DAYS):
    lowered = user_text.lower()

    if any(term in lowered for term in ("past week", "last week", "this week", "weekly")):
        return WEEK_SLEEP_DAYS
    if any(term in lowered for term in ("past month", "last month", "this month", "monthly")):
        return 30
    if any(term in lowered for term in ("past year", "last year", "this year", "yearly")):
        return 365
    if any(
        term in lowered
        for term in (
            "all data",
            "all history",
            "all sleep",
            "sleep history",
            "everything",
            "entire database",
        )
    ):
        return MAX_CONTEXT_SLEEP_DAYS
    if any(term in lowered for term in ("predict", "predicts", "correlate", "correlation", "pattern", "patterns")):
        return PREDICTOR_SLEEP_DAYS

    match = re.search(r"(\d+)\s*(day|days|night|nights)", lowered)
    if match:
        return max(1, min(int(match.group(1)), MAX_CONTEXT_SLEEP_DAYS))

    match = re.search(r"(\d+)\s*(week|weeks)", lowered)
    if match:
        return max(1, min(int(match.group(1)) * 7, MAX_CONTEXT_SLEEP_DAYS))

    match = re.search(r"(\d+)\s*(month|months)", lowered)
    if match:
        return max(1, min(int(match.group(1)) * 30, MAX_CONTEXT_SLEEP_DAYS))

    return default_days


def discover_sleep_entities():
    states = home_assistant_api_request("states", timeout=30)

    def score(entity_id):
        lowered = entity_id.lower()
        if any(term in lowered for term in SLEEP_ENTITY_EXCLUDE_TERMS):
            return -1
        points = 0
        for term in ("sleep", "asleep", "awake", "awakenings", "in_bed", "bedtime"):
            if term in lowered:
                points += 3
        for term in ("oura", "withings", "fitbit", "garmin"):
            if term in lowered:
                points += 2
        for term in ("wake", "bed", "nap"):
            if term in lowered:
                points += 1
        return points

    entities = []
    for state in states:
        entity_id = state.get("entity_id", "")
        friendly_name = state.get("attributes", {}).get("friendly_name", "")
        searchable = f"{entity_id} {friendly_name}".lower()
        if any(term in searchable for term in SLEEP_ENTITY_TERMS):
            entities.append(entity_id)

    return sorted(
        [entity for entity in entities if score(entity) > 0],
        key=lambda entity: (-score(entity), entity),
    )


def parse_number(value):
    try:
        if value in (None, "", "unknown", "unavailable"):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def summarize_entity(entity_id, rows):
    numeric_points = []
    state_counts = defaultdict(int)

    for row in rows:
        value = row["state"]
        number = parse_number(value)
        if number is None:
            state_counts[str(value)] += 1
        else:
            numeric_points.append((float(row["updated_ts"]), number))

    first_seen = rows[0]["updated_at"]
    last_seen = rows[-1]["updated_at"]

    if numeric_points:
        midpoint = max(1, len(numeric_points) // 2)
        earlier = [point[1] for point in numeric_points[:midpoint]]
        later = [point[1] for point in numeric_points[midpoint:]]
        return {
            "entity_id": entity_id,
            "kind": "numeric",
            "samples": len(numeric_points),
            "first_seen": first_seen,
            "last_seen": last_seen,
            "min": round(min(value for _, value in numeric_points), 2),
            "max": round(max(value for _, value in numeric_points), 2),
            "average": round(mean(value for _, value in numeric_points), 2),
            "earlier_average": round(mean(earlier), 2),
            "later_average": round(mean(later), 2),
            "latest": round(numeric_points[-1][1], 2),
        }

    top_states = sorted(state_counts.items(), key=lambda item: item[1], reverse=True)[:6]
    return {
        "entity_id": entity_id,
        "kind": "state",
        "samples": len(rows),
        "first_seen": first_seen,
        "last_seen": last_seen,
        "latest": rows[-1]["state"],
        "top_states": [{"state": state, "count": count} for state, count in top_states],
    }


def minutes_to_hours_label(minutes):
    rounded = int(round(minutes))
    hours, remainder = divmod(rounded, 60)
    return f"{hours}h {remainder:02d}m"


def compact_date_label(value):
    return value.strftime("%b %-d, %Y")


def date_range_label(records):
    if not records:
        return ""
    first = records[0]["date"]
    last = records[-1]["date"]
    first_date = datetime.fromisoformat(first)
    last_date = datetime.fromisoformat(last)
    if first[:7] == last[:7]:
        return f"{datetime.fromisoformat(first).strftime('%b %-d')}-{datetime.fromisoformat(last).strftime('%-d, %Y')}"
    if first[:4] == last[:4]:
        return f"{first_date.strftime('%b %-d')}-{last_date.strftime('%b %-d, %Y')}"
    return f"{compact_date_label(datetime.fromisoformat(first))}-{compact_date_label(datetime.fromisoformat(last))}"


def brief_sleep_record(record):
    return {
        "date": record["date"],
        "date_label": record["date_label"],
        "sleep_start_time": record.get("sleep_start_time"),
        "minutes_asleep": record["minutes_asleep"],
        "sleep_label": record["sleep_label"],
        "minutes_awake": record["minutes_awake"],
        "efficiency": record["efficiency"],
    }


def history_start(days):
    return datetime.now(timezone.utc) - timedelta(days=days)


def history_start_iso(days):
    return history_start(days).isoformat().replace("+00:00", "Z")


def history_end_iso():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def history_start_ts(days):
    return history_start(days).timestamp()


def history_end_ts():
    return datetime.now(timezone.utc).timestamp()


def history_iso(value):
    return value.isoformat().replace("+00:00", "Z")


def history_window(days):
    start = datetime.now(timezone.utc) - timedelta(days=days)
    end = datetime.now(timezone.utc)
    return start, end


def normalize_history_rows(entity_id, history_payload, limit=None):
    rows = []
    entity_history = []
    if isinstance(history_payload, list) and history_payload:
        entity_history = history_payload[0] if isinstance(history_payload[0], list) else history_payload

    last_entity_id = entity_id
    last_updated = None
    for item in entity_history:
        if not isinstance(item, dict):
            continue
        last_entity_id = item.get("entity_id") or last_entity_id
        timestamp_value = item.get("last_updated") or item.get("last_changed") or last_updated
        state = item.get("state")
        if state in (None, "", "unknown", "unavailable") or not timestamp_value:
            continue
        last_updated = timestamp_value
        try:
            parsed = datetime.fromisoformat(str(timestamp_value).replace("Z", "+00:00"))
        except ValueError:
            continue
        rows.append(
            {
                "entity_id": last_entity_id,
                "state": str(state),
                "updated_at": parsed.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                "updated_ts": parsed.timestamp(),
            }
        )

    rows.sort(key=lambda row: row["updated_ts"])
    if limit:
        rows = rows[-limit:]
    return rows


def query_entity_history_rows(entity_id, days, limit=None):
    cache_bucket = int(datetime.now(timezone.utc).timestamp() // 30)
    cache_key = (entity_id, int(days), limit, cache_bucket)
    if cache_key in home_assistant_history_cache:
        return copy.deepcopy(home_assistant_history_cache[cache_key])

    start, end = history_window(days)
    payload = home_assistant_api_request(
        f"history/period/{history_iso(start)}",
        params={
            "filter_entity_id": entity_id,
            "end_time": history_iso(end),
            "minimal_response": "",
            "significant_changes_only": "0",
        },
        timeout=60,
    )
    rows = normalize_history_rows(entity_id, payload)
    # The history API includes an initial boundary state at the requested start
    # even when that state actually changed before the window. Raw recorder
    # queries did not include that synthetic row, so drop it for parity.
    rows = [row for row in rows if abs(row["updated_ts"] - start.timestamp()) > 2]
    if limit:
        rows = rows[-limit:]
    home_assistant_history_cache[cache_key] = copy.deepcopy(rows)
    for key in list(home_assistant_history_cache):
        if key[-1] < cache_bucket - 2:
            home_assistant_history_cache.pop(key, None)
    return rows


def query_sleep_metric_rows(entity_id, days):
    return query_entity_history_rows(entity_id, days)


def summarize_completed_sleep(days=7):
    if not home_assistant_token():
        return {
            "available": False,
            "source": "home_assistant_api",
            "api_url": home_assistant_api_url(),
            "days_requested": days,
            "days_returned": 0,
            "message": "SUPERVISOR_TOKEN is not available.",
        }

    time_zone = home_assistant_time_zone()
    grouped = defaultdict(dict)

    with ThreadPoolExecutor(max_workers=min(5, len(SLEEP_METRIC_ENTITIES))) as executor:
        metric_rows = {
            metric: rows
            for metric, rows in zip(
                SLEEP_METRIC_ENTITIES,
                executor.map(
                    lambda item: query_sleep_metric_rows(item[1], days),
                    SLEEP_METRIC_ENTITIES.items(),
                ),
            )
        }

    for metric, entity_id in SLEEP_METRIC_ENTITIES.items():
        rows = metric_rows[metric]
        for row in rows:
            local_dt = local_datetime(row["updated_ts"], time_zone)
            day_key = local_dt.date().isoformat()

            if metric == "start_time":
                value = str(row["state"]).strip()
                if not value:
                    continue
            else:
                value = parse_number(row["state"])
                if value is None:
                    continue
                # These sensors reset to 0 around midnight; those rows are not completed nights.
                if metric in {"minutes_asleep", "time_in_bed", "minutes_awake", "efficiency"} and value == 0:
                    continue

            current = grouped[day_key].get(metric)
            if current is None or local_dt.isoformat() >= current["updated_local"]:
                grouped[day_key][metric] = {
                    "value": value,
                    "updated_local": local_dt.isoformat(),
                }

    records = []
    required_metrics = ("minutes_asleep", "time_in_bed", "minutes_awake", "efficiency")
    for day_key, values in grouped.items():
        if not all(metric in values for metric in required_metrics):
            continue

        record = {
            "date": day_key,
            "date_label": compact_date_label(datetime.fromisoformat(day_key)),
            "sleep_start_time": values.get("start_time", {}).get("value"),
            "minutes_asleep": int(round(values["minutes_asleep"]["value"])),
            "time_in_bed": int(round(values["time_in_bed"]["value"])),
            "minutes_awake": int(round(values["minutes_awake"]["value"])),
            "efficiency": int(round(values["efficiency"]["value"])),
            "updated_local": max(values[metric]["updated_local"] for metric in required_metrics),
        }
        record["sleep_label"] = minutes_to_hours_label(record["minutes_asleep"])
        record["time_in_bed_label"] = minutes_to_hours_label(record["time_in_bed"])
        records.append(record)

    records = sorted(records, key=lambda record: record["date"])[-days:]
    if not records:
        return {
            "available": False,
            "source": "home_assistant_api",
            "api_url": home_assistant_api_url(),
            "time_zone": time_zone,
            "days_requested": days,
            "days_returned": 0,
            "message": "No completed sleep records were found.",
        }

    averages = {
        "minutes_asleep": round(mean(record["minutes_asleep"] for record in records), 1),
        "time_in_bed": round(mean(record["time_in_bed"] for record in records), 1),
        "minutes_awake": round(mean(record["minutes_awake"] for record in records), 1),
        "efficiency": round(mean(record["efficiency"] for record in records), 1),
    }
    averages["sleep_label"] = minutes_to_hours_label(averages["minutes_asleep"])
    averages["time_in_bed_label"] = minutes_to_hours_label(averages["time_in_bed"])

    highest_awake = sorted(records, key=lambda record: record["minutes_awake"], reverse=True)[:2]
    shortest_sleep = sorted(records, key=lambda record: record["minutes_asleep"])[:2]
    best_sleep = max(records, key=lambda record: (record["minutes_asleep"], record["efficiency"]))

    return {
        "available": True,
        "source": "home_assistant_api",
        "api_url": home_assistant_api_url(),
        "time_zone": time_zone,
        "read_only": True,
        "days_requested": days,
        "days_returned": len(records),
        "date_range": date_range_label(records),
        "latest_sleep_local": records[-1]["updated_local"],
        "source_sensors": SLEEP_METRIC_ENTITIES,
        "averages": averages,
        "daily": records,
        "notable": {
            "highest_awake_time": [brief_sleep_record(record) for record in highest_awake],
            "shortest_sleep": [brief_sleep_record(record) for record in shortest_sleep],
            "best_sleep_duration": brief_sleep_record(best_sleep),
        },
        "method": (
            "Grouped nick_r sleep metrics by Home Assistant local wake/update date, "
            "kept the latest non-zero completed value per day, and ignored midnight reset rows."
        ),
    }


def summarize_home_assistant_sleep(days=90):
    if not home_assistant_token():
        return {
            "available": False,
            "message": "SUPERVISOR_TOKEN is not available.",
        }

    days = max(1, min(int(days), MAX_CONTEXT_SLEEP_DAYS))
    completed_sleep = summarize_completed_sleep(days=days)
    return {
        "available": True,
        "source": "home_assistant_api",
        "api_url": home_assistant_api_url(),
        "read_only": True,
        "time_zone": home_assistant_time_zone(),
        "days_requested": days,
        "completed_sleep": completed_sleep,
        "sensor_map": load_sensor_map()["sensors"],
        "guidance": (
            "For sleep questions, use completed_sleep as the authoritative source. "
            "Do not average raw history API state rows, because the sleep sensors emit temporary "
            "zero/reset states that are not completed sleep sessions."
        ),
    }


def describe_sensor(entity_id):
    name = entity_id.split(".", 1)[-1].replace("_", " ")
    lowered = entity_id.lower()

    descriptions = [
        ("awakenings", "Number of recorded awakenings during a sleep session."),
        ("minutes_awake", "Minutes recorded awake during the sleep window."),
        ("time_in_bed", "Total minutes recorded in bed."),
        ("minutes_asleep", "Total minutes recorded asleep."),
        ("fall_asleep", "Minutes recorded before falling asleep."),
        ("sleep_efficiency", "Sleep efficiency percentage from the sleep tracker."),
        ("sleep_start_time", "Reported sleep start time."),
        ("sleep_confidence", "Phone or tracker confidence that the sleep state is accurate."),
        ("sleep_duration", "Sleep duration reported by the device or integration."),
        ("sleep_segment", "Raw sleep segment duration value from the device or integration."),
        ("after_wakeup", "Minutes recorded after waking up."),
    ]

    for term, description in descriptions:
        if term in lowered:
            return description

    return f"Likely sleep-related Home Assistant entity for {name}."


def home_assistant_sensor_map():
    available = home_assistant_available()
    stored = load_persisted_sensor_map()

    if stored is not None:
        if isinstance(stored, dict) and "_sensor_map_error" in stored:
            return {
                "available": available,
                "source": "persisted_sensor_map",
                "api_url": home_assistant_api_url(),
                "read_only": True,
                "storage_path": sensor_map_path(),
                "sensors": stored.get("sensors", []),
                "message": stored["_sensor_map_error"]["message"],
                "warning": stored["_sensor_map_error"],
            }
        return {
            "available": available,
            "source": "home_assistant_api",
            "api_url": home_assistant_api_url(),
            "read_only": True,
            "storage_path": sensor_map_path(),
            "sensors": stored,
        }

    if not home_assistant_token():
        return {
            "available": False,
            "message": "SUPERVISOR_TOKEN is not available.",
            "storage_path": sensor_map_path(),
            "sensors": [],
        }

    entities = discover_sleep_entities()

    return {
        "available": True,
        "source": "home_assistant_api",
        "api_url": home_assistant_api_url(),
        "read_only": True,
        "storage_path": sensor_map_path(),
        "sensors": [
            {
                "sensor": entity_id,
                "description": describe_sensor(entity_id),
            }
            for entity_id in entities
        ],
    }


def sanitize_sensor_map(rows):
    sanitized = []
    seen = set()

    for row in rows[:250]:
        sensor = re.sub(r"\s+", "", str(row.get("sensor", "")).strip().lower())
        description = str(row.get("description", "")).strip()
        if not sensor and not description:
            continue
        if sensor and "." not in sensor:
            description = description or (
                "Sensor map row may need a full Home Assistant entity ID such as sensor.example."
            )
        if len(sensor) > 180:
            sensor = sensor[:180]
        if len(description) > 500:
            description = description[:500]
        key = sensor.lower()
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        sanitized.append({"sensor": sensor, "description": description or "No description provided."})

    return sanitized


def load_persisted_sensor_map():
    path = sensor_map_path()
    if not os.path.exists(path):
        return None

    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        backup_path = f"{path}.corrupt.{int(datetime.now(timezone.utc).timestamp())}"
        try:
            shutil.copy2(path, backup_path)
        except OSError:
            backup_path = None
        return {
            "_sensor_map_error": {
                "message": "Sensor map file is invalid JSON and was ignored.",
                "path": path,
                "backup_path": backup_path,
                "error": str(exc),
            },
            "sensors": [],
        }

    return sanitize_sensor_map(data.get("sensors", []))


def load_sensor_map():
    return home_assistant_sensor_map()


def save_sensor_map(rows):
    sanitized = sanitize_sensor_map(rows)
    path = sensor_map_path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary_path = f"{path}.tmp"

    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump({"sensors": sanitized}, handle, indent=2)
        handle.write("\n")

    if os.path.exists(path):
        backup_path = f"{path}.bak"
        shutil.copy2(path, backup_path)

    os.replace(temporary_path, path)
    return {
        "available": home_assistant_available(),
        "storage_path": path,
        "sensors": sanitized,
    }


def sensor_map_entity_ids():
    return [row["sensor"] for row in load_sensor_map()["sensors"] if row.get("sensor")]


def sleep_metric_for_entity(entity_id):
    for metric, sensor in SLEEP_METRIC_ENTITIES.items():
        if sensor == entity_id and metric != "start_time":
            return metric
    return None


def plot_axis_label(plot):
    metric = plot.get("metric") if plot else None
    if metric:
        return metric.replace("_", " ").title()
    return "Value"


def render_python_plot(plot):
    if not plot or not plot.get("available") or not plot.get("points"):
        return None

    points = plot["points"]
    x_values = [datetime.fromtimestamp(float(point["timestamp"]), timezone.utc) for point in points]
    y_values = [float(point["value"]) for point in points]
    y_label = plot_axis_label(plot)
    title = plot.get("sensor") or "Sensor plot"
    if plot.get("cleaned"):
        title = f"{title} (cleaned completed nights)"

    fig, ax = plt.subplots(figsize=(8.8, 4.4), dpi=140)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    ax.plot(x_values, y_values, color="#2563eb", linewidth=2.4, marker="o", markersize=4.5)
    ax.fill_between(x_values, y_values, min(y_values), color="#dbeafe", alpha=0.45)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color="#111827", pad=12)
    ax.set_xlabel("Date", fontsize=10, fontweight="bold", color="#334155")
    ax.set_ylabel(y_label, fontsize=10, fontweight="bold", color="#334155")
    ax.grid(True, color="#e2e8f0", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.tick_params(axis="both", colors="#64748b", labelsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %-d"))
    fig.autofmt_xdate(rotation=30, ha="right")

    stats = (
        f"samples {plot.get('samples')} | min {plot.get('min')} | "
        f"avg {plot.get('average')} | max {plot.get('max')} | latest {plot.get('latest')}"
    )
    ax.text(
        0,
        -0.28,
        stats,
        transform=ax.transAxes,
        fontsize=8.5,
        color="#64748b",
        va="top",
    )
    fig.tight_layout(rect=[0, 0.08, 1, 1])

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {
        "data_url": f"data:image/png;base64,{encoded}",
        "format": "png",
        "renderer": "matplotlib",
        "x_axis_label": "Date",
        "y_axis_label": y_label,
        "title": title,
    }


def attach_python_plot(plot):
    if plot and plot.get("available"):
        try:
            image = render_python_plot(plot)
            if image:
                plot["python_image"] = image
        except Exception as exc:
            plot["python_image_error"] = str(exc)
    return plot


def query_cleaned_sleep_points(entity_id, days=30):
    metric = sleep_metric_for_entity(entity_id)
    if not metric:
        return None

    summary = summarize_completed_sleep(days=days)
    if not summary.get("available"):
        return {
            "available": False,
            "sensor": entity_id,
            "message": summary.get("message", "No completed sleep records were found."),
            "points": [],
            "cleaned": True,
        }

    points = []
    for record in summary["daily"]:
        value = record.get(metric)
        if value is None:
            continue
        timestamp = datetime.fromisoformat(record["updated_local"]).timestamp()
        points.append(
            {
                "time": record["date_label"],
                "timestamp": timestamp,
                "value": value,
                "date": record["date"],
                "label": record.get("sleep_label") if metric == "minutes_asleep" else None,
            }
        )

    if not points:
        return {
            "available": False,
            "sensor": entity_id,
            "message": "No cleaned sleep points were found for this range.",
            "points": [],
            "cleaned": True,
        }

    values = [point["value"] for point in points]
    return {
        "available": True,
        "sensor": entity_id,
        "days": summary["days_requested"],
        "points": points,
        "min": round(min(values), 2),
        "max": round(max(values), 2),
        "average": round(mean(values), 2),
        "latest": round(values[-1], 2),
        "samples": len(points),
        "cleaned": True,
        "source": "completed_sleep",
        "metric": metric,
        "date_range": summary["date_range"],
    }


def query_sensor_points(entity_id, days=30, limit=500):
    if not home_assistant_token():
        raise RuntimeError("SUPERVISOR_TOKEN is not available.")

    days = max(1, min(int(days), 365))
    limit = max(10, min(int(limit), 2000))

    cleaned_sleep_points = query_cleaned_sleep_points(entity_id, days=days)
    if cleaned_sleep_points is not None:
        return cleaned_sleep_points

    rows = query_entity_history_rows(entity_id, days=days, limit=limit)

    points = []
    for row in rows:
        value = parse_number(row["state"])
        if value is None:
            continue
        points.append(
            {
                "time": row["updated_at"],
                "timestamp": float(row["updated_ts"]),
                "value": value,
            }
        )

    if not points:
        return {
            "available": False,
            "sensor": entity_id,
            "message": "No numeric sensor history was found for this range.",
            "points": [],
        }

    values = [point["value"] for point in points]
    return {
        "available": True,
        "sensor": entity_id,
        "days": days,
        "points": points,
        "min": round(min(values), 2),
        "max": round(max(values), 2),
        "average": round(mean(values), 2),
        "latest": round(values[-1], 2),
        "samples": len(points),
        "cleaned": False,
        "source": "home_assistant_history_api",
    }


def query_sensor_history(entity_id, days=7, limit=80):
    if not home_assistant_token():
        raise RuntimeError("SUPERVISOR_TOKEN is not available.")

    days = max(1, min(int(days), 365))
    limit = max(1, min(int(limit), MAX_CONTEXT_SENSOR_ROWS))

    rows = query_entity_history_rows(entity_id, days=days, limit=limit)

    data_rows = []
    numeric_values = []
    for row in rows:
        number = parse_number(row["state"])
        if number is not None:
            numeric_values.append(number)
        data_rows.append(
            {
                "time": local_datetime(row["updated_ts"]).strftime("%Y-%m-%d %H:%M:%S"),
                "utc_time": row["updated_at"],
                "local_time": local_datetime(row["updated_ts"]).isoformat(),
                "timestamp": float(row["updated_ts"]),
                "state": row["state"],
                "numeric_value": number,
            }
        )

    result = {
        "available": bool(data_rows),
        "sensor": entity_id,
        "days": days,
        "rows": data_rows,
        "samples": len(data_rows),
    }
    if numeric_values:
        result["numeric_summary"] = {
            "min": round(min(numeric_values), 2),
            "max": round(max(numeric_values), 2),
            "average": round(mean(numeric_values), 2),
            "latest": round(numeric_values[-1], 2),
        }
    if not data_rows:
        result["message"] = "No sensor history was found for this range."

    return result


def relevant_sensor_map_rows(user_text, max_rows=MAX_CONTEXT_SENSORS):
    rows = load_sensor_map()["sensors"]
    scored = [(score_sensor_for_prompt(row, user_text), row) for row in rows if row.get("sensor")]
    exact_matches = [row for score, row in scored if score >= 100]
    if exact_matches:
        return exact_matches[:max_rows]
    matches = [item for item in scored if item[0] > 0]
    matches.sort(key=lambda item: (-item[0], item[1].get("sensor", "")))
    return [row for _, row in matches[:max_rows]]


def prompt_mentions_mapped_sensor(user_text):
    return bool(relevant_sensor_map_rows(user_text, max_rows=1))


def should_include_sleep_context(user_text, mapped_rows=None):
    lowered = user_text.lower()
    mentions_sleep = any(term in lowered for term in SLEEP_ENTITY_TERMS)
    relation_request = any(term in lowered for term in RELATION_REQUEST_TERMS)
    mapped_rows = mapped_rows or []
    return mentions_sleep or (relation_request and any(row.get("sensor") for row in mapped_rows))


def should_include_mapped_sensor_history(user_text, mapped_rows=None, include_sleep=False):
    mapped_rows = mapped_rows or []
    if not mapped_rows:
        return False
    lowered = user_text.lower()
    relation_request = any(term in lowered for term in RELATION_REQUEST_TERMS)
    if relation_request:
        return True
    if should_show_sensor_data(user_text):
        return True
    if include_sleep:
        # Plain sleep questions are already answered from completed_sleep. Pulling
        # every matched mapped sensor can add many seconds without improving the answer.
        return False
    return prompt_mentions_mapped_sensor(user_text)


def row_local_date(row):
    value = row.get("local_time") or row.get("time") or ""
    return str(value)[:10]


def active_sensor_rows(rows):
    return [row for row in rows if str(row.get("state", "")).lower() in ACTIVE_STATES]


def daily_state_summary(rows):
    daily = {}
    for row in rows:
        date = row_local_date(row)
        if not date:
            continue
        day = daily.setdefault(
            date,
            {
                "date": date,
                "total_rows": 0,
                "active_events": 0,
                "states": Counter(),
            },
        )
        state = str(row.get("state", ""))
        day["total_rows"] += 1
        day["states"][state] += 1
        if state.lower() in ACTIVE_STATES:
            day["active_events"] += 1

    result = []
    for date in sorted(daily):
        day = daily[date]
        result.append(
            {
                "date": date,
                "total_rows": day["total_rows"],
                "active_events": day["active_events"],
                "states": dict(day["states"]),
            }
        )
    return result


def parse_sleep_start(value, record_date=None):
    if not value:
        return None

    text = str(value).strip()
    if record_date and re.match(r"^\d{1,2}:\d{2}(:\d{2})?$", text):
        text = f"{record_date} {text}"

    for parser in (
        datetime.fromisoformat,
        lambda item: datetime.strptime(item, "%Y-%m-%d %H:%M"),
        lambda item: datetime.strptime(item, "%Y-%m-%d %H:%M:%S"),
    ):
        try:
            parsed = parser(text)
            break
        except ValueError:
            parsed = None
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ZoneInfo(home_assistant_time_zone()))
    return parsed


def sleep_sensor_alignment(rows, completed_sleep):
    if not completed_sleep or not completed_sleep.get("available"):
        return []

    active_rows = active_sensor_rows(rows)
    alignment = []
    for record in completed_sleep.get("daily", []):
        sleep_start = parse_sleep_start(record.get("sleep_start_time"), record.get("date"))
        if not sleep_start:
            continue

        start_ts = sleep_start.timestamp()
        end_ts = start_ts + (record.get("time_in_bed") or 0) * 60
        before_start_ts = start_ts - 4 * 60 * 60
        before_sleep = [row for row in active_rows if before_start_ts <= row.get("timestamp", 0) < start_ts]
        during_sleep = [row for row in active_rows if start_ts <= row.get("timestamp", 0) <= end_ts]

        alignment.append(
            {
                "date": record.get("date"),
                "sleep_start_time": record.get("sleep_start_time"),
                "sleep_label": record.get("sleep_label"),
                "minutes_awake": record.get("minutes_awake"),
                "efficiency": record.get("efficiency"),
                "active_events_4h_before_sleep": len(before_sleep),
                "active_events_during_time_in_bed": len(during_sleep),
            }
        )

    return alignment[-45:]


def summarize_mapped_sensor_history(row, days, completed_sleep=None):
    sensor = row.get("sensor")
    summary = {
        "sensor": sensor,
        "description": row.get("description", ""),
        "days_requested": days,
        "read_only": True,
    }

    if not sensor:
        return {**summary, "available": False, "message": "Sensor map row has no entity ID."}

    try:
        history = query_sensor_history(sensor, days=days, limit=MAX_CONTEXT_SENSOR_ROWS)
    except Exception as exc:
        return {**summary, "available": False, "message": f"Could not read sensor history: {exc}"}

    rows = history.get("rows", [])
    summary.update(
        {
            "available": bool(rows),
            "days_returned": history.get("days"),
            "samples": history.get("samples", 0),
            "numeric_summary": history.get("numeric_summary"),
        }
    )
    if not rows:
        summary["message"] = history.get("message", "No sensor history was found.")
        return summary

    state_counts = Counter(str(item.get("state", "")) for item in rows)
    daily = daily_state_summary(rows)
    summary.update(
        {
            "first_seen_local": rows[0].get("local_time") or rows[0].get("time"),
            "last_seen_local": rows[-1].get("local_time") or rows[-1].get("time"),
            "latest_state": rows[-1].get("state"),
            "state_counts": [{"state": state, "count": count} for state, count in state_counts.most_common(12)],
            "daily_active_events": daily[-60:],
            "recent_rows": rows[-20:],
        }
    )

    alignment = sleep_sensor_alignment(rows, completed_sleep)
    if alignment:
        summary["sleep_alignment"] = alignment
        summary["sleep_alignment_method"] = (
            "Counts active states such as on/open in the 4 hours before reported "
            "sleep_start_time and during the reported time_in_bed window."
        )

    return summary


def summarize_mapped_sensor_histories(user_text, days, completed_sleep=None):
    rows = relevant_sensor_map_rows(user_text)
    if not rows:
        return []
    return [summarize_mapped_sensor_history(row, days=days, completed_sleep=completed_sleep) for row in rows]


def score_sensor_for_prompt(row, user_text):
    lowered = user_text.lower()
    sensor = row.get("sensor", "")
    description = row.get("description", "")
    sensor_lowered = sensor.lower()

    if sensor_lowered and sensor_lowered in lowered:
        return 100

    score = 0
    searchable = f"{sensor} {description}".lower()
    tokens = {
        token
        for token in re.split(r"[^a-z0-9]+", searchable)
        if len(token) > 2
        and token
        not in {
            "sensor",
            "binary",
            "count",
            "total",
            "this",
            "that",
            "with",
            "from",
            "have",
            "about",
            "what",
            "when",
            "where",
            "which",
        }
    }
    for token in tokens:
        if token in lowered:
            score += 1

    return score


def select_sensor_for_prompt(user_text):
    rows = load_sensor_map()["sensors"]
    if not rows:
        return None

    scored = sorted(
        [(score_sensor_for_prompt(row, user_text), row.get("sensor")) for row in rows],
        reverse=True,
    )
    best_score, best_sensor = scored[0]
    if best_score > 0:
        return best_sensor

    return rows[0].get("sensor")


def preferred_plot_sensor(user_text):
    lowered = user_text.lower()
    rows = load_sensor_map()["sensors"]
    mapped_sensors = {row.get("sensor") for row in rows if row.get("sensor")}

    for sensor in mapped_sensors:
        if sensor.lower() in lowered:
            return sensor

    if "sleep" in lowered and SLEEP_METRIC_ENTITIES["minutes_asleep"] in mapped_sensors:
        return SLEEP_METRIC_ENTITIES["minutes_asleep"]

    return select_sensor_for_prompt(user_text)


def build_plot_for_prompt(user_text):
    if not should_make_plot(user_text):
        return None

    sensor = preferred_plot_sensor(user_text)
    if not sensor:
        return {
            "available": False,
            "message": "No sensor map rows are available to plot.",
            "points": [],
        }

    days = requested_days_from_text(user_text)
    return attach_python_plot(query_sensor_points(sensor, days=days))


def build_sensor_data_for_prompt(user_text):
    if not should_show_sensor_data(user_text):
        return None

    sensor = select_sensor_for_prompt(user_text)
    if not sensor:
        return {
            "available": False,
            "message": "No sensor map rows are available to show.",
            "rows": [],
        }

    match = re.search(r"(\d+)\s*(day|days)", user_text.lower())
    days = int(match.group(1)) if match else 7
    return query_sensor_history(sensor, days=days)


def plot_context(plot):
    if not plot:
        return ""
    if not plot.get("available"):
        return f"\nPlot request status: {plot.get('message', 'No plot was generated.')}\n"

    summary = {
        "sensor": plot["sensor"],
        "days": plot["days"],
        "samples": plot["samples"],
        "min": plot["min"],
        "max": plot["max"],
        "average": plot["average"],
        "latest": plot["latest"],
    }
    return (
        "\nThe app also generated a read-only plot for this request. "
        "Use these plot stats in your answer:\n"
        f"{yaml.safe_dump(summary, sort_keys=False)}"
    )


def sensor_data_context(sensor_data):
    if not sensor_data:
        return ""
    if not sensor_data.get("available"):
        return f"\nSensor data request status: {sensor_data.get('message', 'No sensor data was loaded.')}\n"

    sample_rows = sensor_data["rows"][-12:]
    summary = {
        "sensor": sensor_data["sensor"],
        "days": sensor_data["days"],
        "samples": sensor_data["samples"],
        "numeric_summary": sensor_data.get("numeric_summary"),
        "recent_rows": sample_rows,
    }
    return (
        "\nThe app loaded read-only Home Assistant history API data for this request. "
        "Use these rows and mention that the table is shown in the app:\n"
        f"{yaml.safe_dump(summary, sort_keys=False)}"
    )


def summarize_home_assistant_prompt_context(user_text, days):
    mapped_rows = relevant_sensor_map_rows(user_text)
    include_sleep = should_include_sleep_context(user_text, mapped_rows)
    completed_sleep = summarize_completed_sleep(days=days) if include_sleep else None
    mapped_history_rows = mapped_rows
    if include_sleep:
        sleep_entities = set(SLEEP_METRIC_ENTITIES.values())
        mapped_history_rows = [row for row in mapped_rows if row.get("sensor") not in sleep_entities]
    include_mapped_history = should_include_mapped_sensor_history(
        user_text,
        mapped_rows=mapped_history_rows,
        include_sleep=include_sleep,
    )
    mapped_sensor_history = (
        list(
            ThreadPoolExecutor(max_workers=min(MAX_CONTEXT_SENSORS, len(mapped_history_rows))).map(
                lambda row: summarize_mapped_sensor_history(
                    row,
                    days=days,
                    completed_sleep=completed_sleep,
                ),
                mapped_history_rows,
            )
        )
        if include_mapped_history
        else []
    )

    return {
        "available": home_assistant_available(),
        "source": "home_assistant_api",
        "api_url": home_assistant_api_url(),
        "read_only": True,
        "time_zone": home_assistant_time_zone(),
        "days_requested": days,
        "sensor_map": load_sensor_map()["sensors"],
        "completed_sleep": completed_sleep,
        "mapped_sensor_history": mapped_sensor_history,
        "guidance": (
            "Use mapped_sensor_history for any Sensor Maps entity mentioned by the user, "
            "including non-sleep entities such as doors, lights, motion, plugs, or other "
            "Home Assistant sensors. For sleep questions, use completed_sleep as the "
            "authoritative sleep source and ignore zero/reset state-history rows."
        ),
    }


def build_model_input(user_text):
    mapped_sensor_requested = prompt_mentions_mapped_sensor(user_text)
    if not should_include_home_assistant_context(user_text) and not mapped_sensor_requested:
        return user_text, None

    try:
        days = requested_days_from_text(user_text)
        ha_summary = summarize_home_assistant_prompt_context(user_text, days=days)
    except Exception as exc:
        ha_summary = {
            "available": False,
            "message": f"Could not read Home Assistant API: {exc}",
        }

    context = yaml.safe_dump(ha_summary, sort_keys=False)
    prompt = (
        "The user asked this:\n"
        f"{user_text}\n\n"
        "Read-only Home Assistant API summary follows. Use only this summary; "
        "do not imply write access. The app calls the Home Assistant REST API through "
        "the Supervisor token and does not access the SQLite database. If "
        "mapped_sensor_history is present, it is the queried history API data for "
        "Sensor Maps entities, so do not say you need "
        "the user to export those same readings. If the user asks for more history, "
        "use the ranges included here.\n"
        f"{context}"
    )
    return prompt, ha_summary


def safe_build_plot_for_prompt(user_text):
    try:
        return build_plot_for_prompt(user_text)
    except Exception as exc:
        return {
            "available": False,
            "message": f"Could not build plot: {exc}",
            "points": [],
        }


def safe_build_sensor_data_for_prompt(user_text):
    try:
        return build_sensor_data_for_prompt(user_text)
    except Exception as exc:
        return {
            "available": False,
            "message": f"Could not load sensor data: {exc}",
            "rows": [],
        }


def openai_response(model, model_input, conversation):
    response_args = {
        "model": model["id"],
        "instructions": DEFAULT_INSTRUCTIONS,
        "input": model_input,
    }
    if (
        conversation["previous_response_id"]
        and conversation.get("last_provider") == "openai"
        and conversation.get("last_model") == model["id"]
    ):
        response_args["previous_response_id"] = conversation["previous_response_id"]

    response = get_client().responses.create(**response_args)
    return {
        "id": response.id,
        "output_text": response.output_text or "No text response was returned.",
    }


def anthropic_messages_for_request(conversation, model_input):
    messages = []
    for message in conversation.get("messages", [])[:-1]:
        role = message.get("role")
        if role not in {"user", "assistant"}:
            continue
        content = (message.get("content") or "").strip()
        if content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": model_input})
    return messages


def extract_anthropic_text(response_payload):
    parts = []
    for block in response_payload.get("content", []):
        if block.get("type") == "text" and block.get("text"):
            parts.append(block["text"])
    return "\n".join(parts).strip() or "No text response was returned."


def create_anthropic_message(model_id, messages):
    payload = {
        "model": model_id,
        "max_tokens": 1600,
        "system": DEFAULT_INSTRUCTIONS,
        "messages": messages,
    }
    request_body = json.dumps(payload).encode("utf-8")
    request_headers = {
        "content-type": "application/json",
        "x-api-key": load_claude_key(),
        "anthropic-version": "2023-06-01",
    }
    api_request = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=request_body,
        headers=request_headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(api_request, timeout=90) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise classify_provider_http_error("Claude", exc.code) from exc
    except urllib.error.URLError as exc:
        raise ProviderApiError(
            "Claude API could not be reached before the timeout.",
            status=502,
            error_type="provider_unreachable",
            fix_steps=["Check your network connection and try again.", "If this continues, check Anthropic status."],
            details_safe={"provider": "anthropic", "reason": str(exc.reason)[:300]},
        ) from exc


def anthropic_response(model, model_input, conversation):
    response_payload = create_anthropic_message(
        model["id"],
        anthropic_messages_for_request(conversation, model_input),
    )
    return {
        "id": response_payload.get("id"),
        "output_text": extract_anthropic_text(response_payload),
    }


def provider_response(provider, model, model_input, conversation):
    try:
        if provider["id"] == "openai":
            return openai_response(model, model_input, conversation)
        if provider["id"] == "anthropic":
            return anthropic_response(model, model_input, conversation)
    except ProviderApiError:
        raise
    except Exception as exc:
        raise classify_provider_exception(provider, exc) from exc
    raise ApiError(f"Unknown provider: {provider['id']}", status=400, error_type="unknown_provider")


def classify_provider_http_error(provider_label, status_code):
    provider_id = "anthropic" if provider_label == "Claude" else provider_label.lower()
    if status_code in {401, 403}:
        return ProviderApiError(
            f"{provider_label} rejected the configured API key.",
            status=401,
            error_type="provider_auth_failed",
            fix_steps=secret_setup_fix_steps(provider_label),
            details_safe={"provider": provider_id, "status": status_code},
        )
    if status_code == 429:
        return ProviderApiError(
            f"{provider_label} rate limit reached.",
            status=429,
            error_type="provider_rate_limited",
            fix_steps=["Wait and try again.", f"Check {provider_label} account limits or billing."],
            details_safe={"provider": provider_id, "status": status_code},
        )
    if status_code >= 500:
        return ProviderApiError(
            f"{provider_label} service returned an error.",
            status=502,
            error_type="provider_unavailable",
            fix_steps=[f"Try again later or check {provider_label} service status."],
            details_safe={"provider": provider_id, "status": status_code},
        )
    return ProviderApiError(
        f"{provider_label} request failed.",
        status=502,
        error_type="provider_request_failed",
        fix_steps=["Check the selected model and provider configuration."],
        details_safe={"provider": provider_id, "status": status_code},
    )


def classify_provider_exception(provider, exc):
    provider_label = provider["label"]
    status_code = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
    if status_code:
        return classify_provider_http_error(provider_label, int(status_code))

    text = str(exc)
    lowered = text.lower()
    if "timed out" in lowered or "timeout" in lowered:
        return ProviderApiError(
            f"{provider_label} request timed out.",
            status=504,
            error_type="provider_timeout",
            fix_steps=["Try again with a shorter prompt.", "Try again later if the provider is slow."],
            details_safe={"provider": provider["id"]},
        )
    return ProviderApiError(
        f"{provider_label} request failed.",
        status=502,
        error_type="provider_request_failed",
        fix_steps=["Check the provider key, model, billing, and provider status."],
        details_safe={"provider": provider["id"], "error_class": exc.__class__.__name__},
    )


def utc_timestamp():
    return datetime.now(timezone.utc).isoformat()


def chat_store_path():
    return os.environ.get("CHAT_STORE_PATH") or "/data/chat_history.json"


def default_conversation(conversation_id=None, title="New chat"):
    now = utc_timestamp()
    return {
        "id": conversation_id or str(uuid.uuid4()),
        "title": title,
        "pinned": False,
        "created_at": now,
        "updated_at": now,
        "previous_response_id": None,
        "last_provider": None,
        "last_model": None,
        "messages": [],
    }


def normalize_conversation(conversation_id, conversation):
    normalized = default_conversation(conversation_id)
    if isinstance(conversation, dict):
        normalized.update(conversation)
    normalized["id"] = normalized.get("id") or conversation_id
    normalized["title"] = (normalized.get("title") or "New chat").strip() or "New chat"
    normalized["pinned"] = bool(normalized.get("pinned"))
    normalized["messages"] = normalized.get("messages") if isinstance(normalized.get("messages"), list) else []
    normalized["created_at"] = normalized.get("created_at") or utc_timestamp()
    normalized["updated_at"] = normalized.get("updated_at") or normalized["created_at"]
    return normalized


def load_chat_store():
    global chat_store_loaded_path
    path = chat_store_path()
    if chat_store_loaded_path == path:
        return

    conversations.clear()
    chat_store_loaded_path = path
    if not os.path.exists(path):
        return

    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return

    stored_conversations = payload.get("conversations", {})
    if isinstance(stored_conversations, list):
        stored_conversations = {
            item.get("id", str(uuid.uuid4())): item for item in stored_conversations if isinstance(item, dict)
        }
    if not isinstance(stored_conversations, dict):
        return

    for conversation_id, conversation in stored_conversations.items():
        conversations[conversation_id] = normalize_conversation(conversation_id, conversation)


def save_chat_store():
    path = chat_store_path()
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    payload = {
        "conversations": conversations,
        "saved_at": utc_timestamp(),
    }
    temp_path = f"{path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temp_path, path)


def title_from_text(text):
    title = re.sub(r"\s+", " ", text).strip()
    if not title:
        return "New chat"
    return title[:52].rstrip() + ("..." if len(title) > 52 else "")


def touch_conversation(conversation):
    conversation["updated_at"] = utc_timestamp()


def chat_summary(conversation):
    messages = conversation.get("messages", [])
    last_message = next((message for message in reversed(messages) if message.get("content")), None)
    return {
        "id": conversation["id"],
        "title": conversation.get("title") or "New chat",
        "pinned": bool(conversation.get("pinned")),
        "created_at": conversation.get("created_at"),
        "updated_at": conversation.get("updated_at"),
        "message_count": len(messages),
        "last_message": (last_message.get("content") if last_message else "")[:120],
    }


def sorted_chat_summaries():
    load_chat_store()
    summaries = [chat_summary(conversation) for conversation in conversations.values()]
    pinned = sorted(
        (item for item in summaries if item["pinned"]),
        key=lambda item: item.get("updated_at") or "",
        reverse=True,
    )
    unpinned = sorted(
        (item for item in summaries if not item["pinned"]),
        key=lambda item: item.get("updated_at") or "",
        reverse=True,
    )
    return pinned + unpinned


def create_conversation(title="New chat"):
    load_chat_store()
    conversation = default_conversation(title=title)
    conversations[conversation["id"]] = conversation
    session["conversation_id"] = conversation["id"]
    save_chat_store()
    return conversation


def get_conversation(conversation_id=None, create=True):
    load_chat_store()
    conversation_id = conversation_id or session.get("conversation_id")
    if conversation_id and conversation_id in conversations:
        session["conversation_id"] = conversation_id
        return conversations[conversation_id]
    if not create:
        return None
    return create_conversation()


@app.get("/")
def index():
    return render_template(
        "index.html",
        default_provider=DEFAULT_PROVIDER,
        default_model=DEFAULT_MODEL,
        providers=provider_catalog(),
        models=model_catalog(),
        key_status=provider_key_status(),
        pricing_source=MODEL_PRICING_SOURCE,
    )


@app.get("/api/health")
def health():
    return jsonify(
        api_success(
            {
                "ok": True,
                "default_provider": DEFAULT_PROVIDER,
                "default_model": DEFAULT_MODEL,
                "home_assistant_api": bool(home_assistant_token()),
                "key_status": provider_key_status(),
            }
        )
    )


@app.get("/api/models")
def models():
    return jsonify(
        api_success(
            {
                "providers": provider_catalog(),
                "models": model_catalog(),
                "default_provider": DEFAULT_PROVIDER,
                "default_model": DEFAULT_MODEL,
                "key_status": provider_key_status(),
                "pricing_source": MODEL_PRICING_SOURCE,
                "pricing_note": MODEL_PRICING_NOTE,
            }
        )
    )


@app.get("/api/key-status")
def key_status():
    return jsonify(api_success(provider_key_status()))


@app.get("/api/config-status")
def config_status():
    return jsonify(config_status_payload())


@app.get("/api/diagnostics")
def diagnostics():
    return jsonify(diagnostics_payload())


@app.get("/api/messages")
def messages():
    conversation = get_conversation(request.args.get("chat_id"), create=False)
    return jsonify(
        api_success(
            {
                "chat": chat_summary(conversation) if conversation else None,
                "chats": sorted_chat_summaries(),
                "active_chat_id": conversation["id"] if conversation else None,
                "messages": conversation["messages"] if conversation else [],
                "default_provider": DEFAULT_PROVIDER,
                "default_model": DEFAULT_MODEL,
                "providers": provider_catalog(),
                "models": model_catalog(),
                "key_status": provider_key_status(),
                "pricing_source": MODEL_PRICING_SOURCE,
            }
        )
    )


@app.get("/api/chats")
def chats():
    conversation = get_conversation(create=False)
    return jsonify(
        {
            "chats": sorted_chat_summaries(),
            "active_chat_id": conversation["id"] if conversation else None,
            "messages": conversation["messages"] if conversation else [],
        }
    )


@app.post("/api/chats")
def create_chat():
    conversation = create_conversation()
    return jsonify(
        {
            "chat": chat_summary(conversation),
            "chats": sorted_chat_summaries(),
            "active_chat_id": conversation["id"],
            "messages": conversation["messages"],
        }
    ), 201


@app.patch("/api/chats/<conversation_id>")
def update_chat(conversation_id):
    conversation = get_conversation(conversation_id, create=False)
    if not conversation:
        return api_error_response("Chat not found.", status=404, error_type="chat_not_found")

    payload = request.get_json(silent=True) or {}
    if "pinned" in payload:
        conversation["pinned"] = bool(payload["pinned"])
    if "title" in payload:
        conversation["title"] = title_from_text(payload["title"])
    touch_conversation(conversation)
    save_chat_store()
    return jsonify(
        {
            "chat": chat_summary(conversation),
            "chats": sorted_chat_summaries(),
            "active_chat_id": session.get("conversation_id"),
        }
    )


@app.delete("/api/chats/<conversation_id>")
def delete_chat(conversation_id):
    load_chat_store()
    if conversation_id not in conversations:
        return api_error_response("Chat not found.", status=404, error_type="chat_not_found")

    del conversations[conversation_id]
    remaining = sorted_chat_summaries()
    if session.get("conversation_id") == conversation_id:
        session["conversation_id"] = remaining[0]["id"] if remaining else create_conversation()["id"]
    save_chat_store()
    active = get_conversation(session.get("conversation_id"))
    return jsonify(
        {
            "chats": sorted_chat_summaries(),
            "active_chat_id": active["id"],
            "messages": active["messages"],
        }
    )


@app.post("/api/message")
def message():
    payload = request.get_json(silent=True) or {}
    user_text = (payload.get("message") or "").strip()

    if not user_text:
        return api_error_response(
            "Message is required.",
            status=400,
            error_type="empty_prompt",
            fix_steps=["Type a prompt before pressing Send."],
        )
    if len(user_text) > MAX_PROMPT_CHARS:
        return api_error_response(
            f"Prompt is too large. Keep prompts under {MAX_PROMPT_CHARS:,} characters.",
            status=413,
            error_type="prompt_too_large",
            fix_steps=["Shorten the prompt or split it into smaller messages."],
            details_safe={"max_prompt_chars": MAX_PROMPT_CHARS, "received_chars": len(user_text)},
        )

    provider, model = requested_provider_and_model(payload)

    key_status_data = provider_key_status()
    provider_status = key_status_data["providers"].get(provider["id"], {})
    if not provider_status.get("configured"):
        payload = api_error_payload(
            f"{provider['label']} API key is not configured.",
            error_type="provider_key_missing",
            fix_steps=secret_setup_fix_steps(provider["label"]),
            details_safe={"provider": provider["id"]},
        )
        payload.update(
            {
                "key_status": key_status_data,
            }
        )
        return jsonify(payload), 400

    model_info = compact_model_info(model)

    conversation = get_conversation(payload.get("chat_id"))
    if not conversation["messages"] and conversation.get("title") == "New chat":
        conversation["title"] = title_from_text(user_text)
    conversation["messages"].append({"role": "user", "content": user_text})
    touch_conversation(conversation)
    plot = safe_build_plot_for_prompt(user_text)
    sensor_data = safe_build_sensor_data_for_prompt(user_text)
    model_input, ha_summary = build_model_input(user_text)
    model_input = f"{model_input}{plot_context(plot)}{sensor_data_context(sensor_data)}"

    try:
        response = provider_response(provider, model, model_input, conversation)
    except ProviderApiError as exc:
        conversation["messages"].pop()
        touch_conversation(conversation)
        save_chat_store()
        return handle_api_error(exc)

    assistant_text = response["output_text"]
    conversation["previous_response_id"] = response.get("id") if provider["id"] == "openai" else None
    conversation["last_provider"] = provider["id"]
    conversation["last_model"] = model["id"]
    assistant_message = {
        "role": "assistant",
        "content": assistant_text,
        "provider": provider["id"],
        "provider_label": provider["label"],
        "model": model["id"],
        "model_label": model["label"],
        "model_pricing": model_info,
    }
    if plot:
        assistant_message["plot"] = plot
    if sensor_data:
        assistant_message["sensor_data"] = sensor_data
    conversation["messages"].append(assistant_message)
    touch_conversation(conversation)
    save_chat_store()

    return jsonify(
        {
            "chat": chat_summary(conversation),
            "chats": sorted_chat_summaries(),
            "active_chat_id": conversation["id"],
            "assistant": assistant_text,
            "messages": conversation["messages"],
            "response_id": response.get("id"),
            "provider": provider["id"],
            "provider_info": dict(provider),
            "model": model["id"],
            "model_info": model_info,
            "home_assistant_context": ha_summary,
            "plot": plot,
            "sensor_data": sensor_data,
        }
    )


@app.get("/api/home-assistant/sleep-summary")
def home_assistant_sleep_summary():
    days = request.args.get("days", str(DEFAULT_SLEEP_DAYS))
    try:
        return jsonify(summarize_home_assistant_sleep(days=days))
    except (TypeError, ValueError):
        return api_error_response("days must be a number.", status=400, error_type="invalid_days")


@app.get("/api/sensor-map")
def sensor_map():
    return jsonify(home_assistant_sensor_map())


@app.put("/api/sensor-map")
def update_sensor_map():
    payload = request.get_json(silent=True) or {}
    sensors = payload.get("sensors", [])
    if not isinstance(sensors, list):
        return api_error_response("sensors must be a list.", status=400, error_type="invalid_sensor_map")
    return jsonify(save_sensor_map(sensors))


@app.get("/api/sensor-plot")
def sensor_plot():
    sensor = (request.args.get("sensor") or "").strip()
    if not sensor:
        return api_error_response("sensor is required.", status=400, error_type="missing_sensor")

    days = request.args.get("days", "30")
    try:
        plot = attach_python_plot(query_sensor_points(sensor, days=days))
    except Exception as exc:
        return api_error_response(str(exc), status=400, error_type="sensor_plot_failed")

    return jsonify(plot)


@app.get("/api/sensor-data")
def sensor_data():
    sensor = (request.args.get("sensor") or "").strip()
    if not sensor:
        return api_error_response("sensor is required.", status=400, error_type="missing_sensor")

    days = request.args.get("days", "7")
    limit = request.args.get("limit", "80")
    try:
        data = query_sensor_history(sensor, days=days, limit=limit)
    except Exception as exc:
        return api_error_response(str(exc), status=400, error_type="sensor_data_failed")

    return jsonify(data)


@app.post("/api/reset")
def reset():
    conversation = create_conversation()
    return jsonify(
        {
            "chat": chat_summary(conversation),
            "chats": sorted_chat_summaries(),
            "active_chat_id": conversation["id"],
            "messages": [],
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
