import base64
import copy
import io
import json
import mimetypes
import os
import re
import secrets
import shutil
import sqlite3
import urllib.error
import urllib.request
import urllib.parse
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from math import sqrt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from statistics import mean
from zoneinfo import ZoneInfo
from xml.sax.saxutils import escape as xml_escape

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
SENSOR_MAP_DISCOVERY_EXCLUDED_DOMAINS = {
    "automation",
    "button",
    "scene",
    "script",
}
HA_REQUEST_TERMS = (
    "home assistant",
    "home-assistant",
    "ha db",
    "ha database",
    "database",
    "sensor",
    "sensors",
    "entity",
    "entities",
    "find it",
    "find them",
    "search",
    "check my",
    "check if",
    "sleep",
    "bed",
    "asleep",
)
SENSOR_CATALOG_REQUEST_TERMS = (
    "sensor",
    "sensors",
    "entity",
    "entities",
    "home assistant",
    "database",
    "db",
    "data",
)
SENSOR_CATALOG_ACTION_TERMS = (
    "find",
    "search",
    "check",
    "look for",
    "look up",
    "do i have",
    "i have",
    "somewhere",
    "anything",
    "alike",
    "named",
    "name",
)
SENSOR_SEARCH_STOPWORDS = {
    "about",
    "alike",
    "also",
    "and",
    "amps",
    "anything",
    "assistant",
    "check",
    "collects",
    "could",
    "data",
    "database",
    "able",
    "for",
    "find",
    "have",
    "home",
    "search",
    "should",
    "like",
    "look",
    "name",
    "named",
    "sensor",
    "sensors",
    "somewhere",
    "that",
    "the",
    "them",
    "there",
    "think",
    "which",
    "with",
    "you",
}
SENSOR_SEARCH_SYNONYMS = {
    "amp": ("amp", "amps", "current", "amperage"),
    "amps": ("amp", "amps", "current", "amperage"),
    "amperage": ("amp", "amps", "current", "amperage"),
    "steps": ("step", "steps", "daily_steps"),
    "step": ("step", "steps", "daily_steps"),
}
PLOT_REQUEST_TERMS = (
    "plot",
    "graph",
    "chart",
    "histogram",
    "scatter",
    "box plot",
    "trend line",
    "visualize",
    "visualise",
)
PLOT_TYPE_ALIASES = {
    "histogram": ("histogram", "distribution"),
    "scatter": ("scatter", " vs ", "versus", "relationship between", "correlation"),
    "box": ("box plot", "boxplot"),
    "bar": ("bar chart", "bar graph", "bar plot", "bars"),
    "area": ("area plot", "area chart", "filled area"),
    "line": ("line plot", "line chart", "line graph", "trend line"),
}
ENTITY_ID_PATTERN = re.compile(r"\b[a-zA-Z_]+\.[a-zA-Z0-9_]+\b")
SENSOR_MAP_ADD_TERMS = ("add", "include", "put", "save")
CONFIRM_TERMS = ("yes", "yeah", "yep", "sure", "confirm", "add it", "please do", "go ahead")
DENY_TERMS = ("no", "nope", "cancel", "don't", "do not", "never mind", "nevermind")
MAX_CONTEXT_SLEEP_DAYS = 365
DEFAULT_SLEEP_DAYS = 30
WEEK_SLEEP_DAYS = 7
PREDICTOR_SLEEP_DAYS = 90
RECORDER_DB_PATHS = (
    os.environ.get("HA_RECORDER_DB_PATH"),
    "/config/home-assistant_v2.db",
    "/data/home-assistant_v2.db",
)
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
N_OF_1_REQUEST_TERMS = (
    "n of 1",
    "n=1",
    "n-of-1",
    "within-person",
    "within person",
    "causal inference",
    "g-formula",
    "g formula",
    "paper",
)
ANALYSIS_VISUAL_REQUEST_TERMS = (
    "causal dag",
    "causal dags",
    "dag",
    "dags",
    "latex",
    "equation",
    "equations",
    "analysis plot",
    "analysis plots",
    "plots describing",
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
    "awakenings": "sensor.nick_r_awakenings_count",
    "minutes_asleep": "sensor.nick_r_sleep_minutes_asleep",
    "time_in_bed": "sensor.nick_r_sleep_time_in_bed",
    "minutes_awake": "sensor.nick_r_sleep_minutes_awake",
    "efficiency": "sensor.nick_r_sleep_efficiency",
    "start_time": "sensor.nick_r_sleep_start_time",
}
SLEEP_PLOT_METRIC_LABELS = {
    "awakenings": "Awakenings",
    "minutes_asleep": "Sleep Time",
    "time_in_bed": "Time In Bed",
    "minutes_awake": "Minutes Awake",
    "efficiency": "Sleep Efficiency",
}
SLEEP_PLOT_METRIC_UNITS = {
    "awakenings": "count",
    "minutes_asleep": "minutes",
    "time_in_bed": "minutes",
    "minutes_awake": "minutes",
    "efficiency": "percent",
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


def env_value(name):
    value = os.environ.get(name)
    if value:
        return value
    for directory in ("/var/run/s6/container_environment", "/run/s6/container_environment"):
        path = os.path.join(directory, name)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    file_value = handle.read().strip()
                if file_value:
                    return file_value
            except OSError:
                continue
    return None


def home_assistant_token():
    return env_value("SUPERVISOR_TOKEN") or env_value("HOME_ASSISTANT_TOKEN")


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
    return any(term in lowered for term in HA_REQUEST_TERMS) or should_build_analysis_visuals(user_text)


def should_search_sensor_catalog(user_text):
    lowered = user_text.lower()
    return any(term in lowered for term in SENSOR_CATALOG_REQUEST_TERMS) and any(
        term in lowered for term in SENSOR_CATALOG_ACTION_TERMS
    )


def sensor_catalog_search_terms(user_text):
    lowered = user_text.lower()
    raw_tokens = [token for token in re.split(r"[^a-z0-9_]+", lowered) if len(token) > 2]
    terms = []
    for token in raw_tokens:
        if token in SENSOR_SEARCH_STOPWORDS:
            continue
        terms.append(token)
        terms.extend(SENSOR_SEARCH_SYNONYMS.get(token, ()))

    if re.search(r"\bamp(s|erage)?\b", lowered):
        terms.extend(("amp", "amps", "current", "amperage"))
    if re.search(r"\bstep(s)?\b", lowered):
        terms.extend(("step", "steps", "daily_steps"))

    deduped = []
    seen = set()
    for term in terms:
        normalized = term.strip().lower()
        if normalized and normalized not in seen:
            deduped.append(normalized)
            seen.add(normalized)
    return deduped[:12]


def should_make_plot(user_text):
    lowered = user_text.lower()
    return any(term in lowered for term in PLOT_REQUEST_TERMS)


def should_build_analysis_visuals(user_text):
    lowered = user_text.lower()
    return any(term in lowered for term in ANALYSIS_VISUAL_REQUEST_TERMS) or (
        "plot" in lowered and "analysis" in lowered
    )


def detect_plot_type(user_text):
    lowered = f" {user_text.lower()} "
    for plot_type, aliases in PLOT_TYPE_ALIASES.items():
        if any(alias in lowered for alias in aliases):
            return plot_type
    return "line"


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
        domain = lowered.split(".", 1)[0]
        if domain in SENSOR_MAP_DISCOVERY_EXCLUDED_DOMAINS:
            return -1
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


def recorder_db_path():
    candidate_paths = (
        os.environ.get("HA_RECORDER_DB_PATH"),
        *RECORDER_DB_PATHS,
    )
    for path in candidate_paths:
        if path and os.path.exists(path):
            return path
    return None


def recorder_query_info(entity_ids, days, limit=None):
    db_path = recorder_db_path()
    if not db_path:
        return {
            "source": "home_assistant_history_api",
            "path": None,
            "sql": None,
            "params_safe": {"entity_ids": entity_ids, "days": days, "limit": limit},
        }

    return {
        "source": "home_assistant_recorder_db",
        "path": db_path,
        "sql": (
            "SELECT states.state, states.last_updated_ts "
            "FROM states JOIN states_meta ON states.metadata_id = states_meta.metadata_id "
            "WHERE states_meta.entity_id = ? AND states.last_updated_ts >= ? "
            "AND states.last_updated_ts <= ? ORDER BY states.last_updated_ts"
        ),
        "params_safe": {"entity_ids": entity_ids, "days": days, "limit": limit},
    }


def query_entity_history_rows_from_db(entity_id, days, limit=None):
    path = recorder_db_path()
    if not path:
        return None

    start, end = history_window(days)
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        metadata = connection.execute(
            "SELECT metadata_id FROM states_meta WHERE entity_id = ?",
            (entity_id,),
        ).fetchone()
        if not metadata:
            return []

        rows = connection.execute(
            (
                "SELECT state, last_updated_ts FROM states "
                "WHERE metadata_id = ? AND last_updated_ts >= ? AND last_updated_ts <= ? "
                "ORDER BY last_updated_ts"
            ),
            (metadata["metadata_id"], start.timestamp(), end.timestamp()),
        ).fetchall()

    normalized = []
    for row in rows:
        if row["last_updated_ts"] is None:
            continue
        updated_ts = float(row["last_updated_ts"])
        normalized.append(
            {
                "state": row["state"],
                "updated_at": datetime.fromtimestamp(updated_ts, timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                "updated_ts": updated_ts,
            }
        )
    if limit:
        normalized = normalized[-limit:]
    return normalized


def score_entity_candidate(entity_id, friendly_name, terms):
    searchable = f"{entity_id} {friendly_name or ''}".lower()
    domain = entity_id.split(".", 1)[0].lower() if "." in entity_id else ""
    score = 0
    for term in terms:
        if not term:
            continue
        if term in searchable:
            score += 4
        if f"_{term}" in searchable or f"{term}_" in searchable:
            score += 2
    if score <= 0:
        return -1
    if domain == "sensor":
        score += 3
    elif domain in SENSOR_MAP_DISCOVERY_EXCLUDED_DOMAINS:
        score -= 6
    return score


def recorder_sensor_catalog_matches(terms, limit=12):
    path = recorder_db_path()
    if not path or not terms:
        return []

    patterns = [f"%{term.lower()}%" for term in terms if term]
    where = " OR ".join(["lower(states_meta.entity_id) LIKE ?"] * len(patterns))
    if not where:
        return []

    sql = (
        "SELECT states_meta.entity_id, COUNT(states.state_id) AS state_rows, "
        "MAX(states.last_updated_ts) AS latest_updated_ts "
        "FROM states_meta LEFT JOIN states ON states.metadata_id = states_meta.metadata_id "
        f"WHERE {where} "
        "GROUP BY states_meta.entity_id"
    )
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(sql, patterns).fetchall()

    candidates = []
    for row in rows:
        entity_id = row["entity_id"]
        score = score_entity_candidate(entity_id, "", terms)
        if score <= 0:
            continue
        latest_ts = row["latest_updated_ts"]
        candidates.append(
            {
                "entity_id": entity_id,
                "source": "home_assistant_recorder_db",
                "score": score,
                "state_rows": int(row["state_rows"] or 0),
                "latest_updated_local": (
                    local_datetime(float(latest_ts)).strftime("%Y-%m-%d %H:%M:%S") if latest_ts is not None else None
                ),
            }
        )

    candidates.sort(key=lambda item: (-item["score"], item["entity_id"]))
    return candidates[:limit]


def home_assistant_state_catalog_matches(terms, limit=12):
    if not terms or not home_assistant_token():
        return []
    try:
        states = home_assistant_api_request("states", timeout=30)
    except Exception:
        return []

    candidates = []
    for state in states:
        entity_id = state.get("entity_id") or ""
        attributes = state.get("attributes") or {}
        friendly_name = attributes.get("friendly_name") or ""
        score = score_entity_candidate(entity_id, friendly_name, terms)
        if score <= 0:
            continue
        candidates.append(
            {
                "entity_id": entity_id,
                "friendly_name": friendly_name,
                "unit_of_measurement": attributes.get("unit_of_measurement"),
                "device_class": attributes.get("device_class"),
                "source": "home_assistant_states_api",
                "score": score,
                "latest_state": state.get("state"),
                "last_updated": state.get("last_updated"),
            }
        )

    candidates.sort(key=lambda item: (-item["score"], item["entity_id"]))
    return candidates[:limit]


def discover_sensor_catalog(user_text, limit=12):
    if not should_search_sensor_catalog(user_text):
        return {"searched": False, "terms": [], "matches": []}

    terms = sensor_catalog_search_terms(user_text)
    if not terms:
        return {
            "searched": True,
            "terms": [],
            "matches": [],
            "message": "No specific sensor search terms were found in the request.",
        }

    merged = {}
    for candidate in [
        *recorder_sensor_catalog_matches(terms, limit=limit),
        *home_assistant_state_catalog_matches(terms),
    ]:
        entity_id = candidate.get("entity_id")
        if not entity_id:
            continue
        existing = merged.get(entity_id)
        if not existing or candidate.get("score", 0) > existing.get("score", 0):
            merged[entity_id] = candidate
        elif existing:
            existing["source"] = "home_assistant_recorder_db_and_states_api"
            for key in ("friendly_name", "unit_of_measurement", "device_class", "latest_state", "last_updated"):
                if candidate.get(key) is not None:
                    existing[key] = candidate[key]

    matches = sorted(merged.values(), key=lambda item: (-item.get("score", 0), item.get("entity_id", "")))[:limit]
    return {
        "searched": True,
        "terms": terms,
        "matches": matches,
        "query": {
            "source": "home_assistant_recorder_db_and_states_api",
            "recorder_db_path": recorder_db_path(),
            "params_safe": {"terms": terms, "limit": limit},
        },
        "message": (
            f"Found {len(matches)} possible Home Assistant entities matching the request."
            if matches
            else "No Home Assistant entities matched the requested sensor terms."
        ),
    }


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
    db_rows = query_entity_history_rows_from_db(entity_id, days, limit=limit)
    if db_rows is not None:
        return db_rows

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
    if not home_assistant_token() and not recorder_db_path():
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
        if "awakenings" in values:
            record["awakenings"] = int(round(values["awakenings"]["value"]))
        record["sleep_label"] = minutes_to_hours_label(record["minutes_asleep"])
        record["time_in_bed_label"] = minutes_to_hours_label(record["time_in_bed"])
        records.append(record)

    records = sorted(records, key=lambda record: record["date"])[-days:]
    if not records:
        return {
            "available": False,
            "source": recorder_query_info(list(SLEEP_METRIC_ENTITIES.values()), days)["source"],
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
        "source": recorder_query_info(list(SLEEP_METRIC_ENTITIES.values()), days)["source"],
        "query": recorder_query_info(list(SLEEP_METRIC_ENTITIES.values()), days),
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


def describe_sensor_map_candidate(entity_id):
    if "step" in entity_id.lower():
        return "Step count sensor."
    return describe_sensor(entity_id)


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

    try:
        entities = discover_sleep_entities()
    except Exception as exc:
        return {
            "available": False,
            "source": "home_assistant_api",
            "api_url": home_assistant_api_url(),
            "read_only": True,
            "storage_path": sensor_map_path(),
            "sensors": [],
            "message": f"Could not discover Home Assistant sensors: {exc}",
        }

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


def normalize_entity_id(entity_id):
    return re.sub(r"\s+", "", str(entity_id or "").strip().lower())


def extract_entity_ids(user_text):
    return [normalize_entity_id(match.group(0)) for match in ENTITY_ID_PATTERN.finditer(user_text)]


def is_sensor_map_entity_candidate(entity_id):
    domain = normalize_entity_id(entity_id).split(".", 1)[0]
    return domain in {
        "sensor",
        "binary_sensor",
        "light",
        "switch",
        "device_tracker",
        "person",
        "input_boolean",
        "input_number",
        "number",
        "counter",
        "button",
    }


def text_confirms(text):
    lowered = text.lower()
    return any(term in lowered for term in CONFIRM_TERMS)


def text_denies(text):
    lowered = text.lower()
    return any(term in lowered for term in DENY_TERMS)


def detect_sensor_map_add_request(user_text):
    lowered = user_text.lower()
    if "sensor map" not in lowered:
        return None
    if not any(term in lowered for term in SENSOR_MAP_ADD_TERMS):
        return None

    entity_ids = [entity_id for entity_id in extract_entity_ids(user_text) if is_sensor_map_entity_candidate(entity_id)]
    if not entity_ids:
        return None

    entity_id = entity_ids[0]
    return {
        "type": "add_sensor_map",
        "sensor": entity_id,
        "description": describe_sensor_map_candidate(entity_id),
    }


def detect_app_delete_request(user_text):
    lowered = user_text.lower()
    delete_requested = any(term in lowered for term in ("delete", "remove", "uninstall"))
    app_requested = any(
        term in lowered
        for term in (
            "add-on",
            "addon",
            "app",
            "agentic prompt",
            "prompt flow",
            "this tool",
        )
    )
    return delete_requested and app_requested and "chat" not in lowered and "sensor map" not in lowered


def sleep_metric_for_entity(entity_id):
    for metric, sensor in SLEEP_METRIC_ENTITIES.items():
        if sensor == entity_id and metric != "start_time":
            return metric
    return None


def plot_axis_label(plot):
    metric = plot.get("metric") if plot else None
    if metric:
        label = SLEEP_PLOT_METRIC_LABELS.get(metric, metric.replace("_", " ").title())
        unit = SLEEP_PLOT_METRIC_UNITS.get(metric)
        return f"{label} ({unit})" if unit else label
    return "Value"


def plot_time_window_label(days):
    if int(days) == 7:
        return "past_week"
    if int(days) == 30:
        return "past_month"
    if int(days) == 365:
        return "past_year"
    return f"past_{int(days)}_days"


def plot_title_for_metric(metric, days, plot_type):
    label = SLEEP_PLOT_METRIC_LABELS.get(metric, metric.replace("_", " ").title())
    window = "Past Week" if int(days) == 7 else "Past Month" if int(days) == 30 else f"Past {int(days)} Days"
    if plot_type == "histogram":
        return f"Distribution of {label}"
    if plot_type == "box":
        return f"{label} Distribution"
    return f"{label} Over the {window}"


def build_plot_spec(
    *,
    plot_type,
    entity_ids,
    title,
    x_label,
    y_label,
    days,
    aggregation="daily",
    x="date",
    y="value",
):
    return {
        "plot_type": plot_type,
        "entity_ids": entity_ids,
        "x": x,
        "y": y,
        "title": title,
        "x_label": x_label,
        "y_label": y_label,
        "time_window": plot_time_window_label(days),
        "aggregation": aggregation,
    }


def encoded_matplotlib_figure(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def point_datetime(point):
    if point.get("date"):
        return datetime.fromisoformat(point["date"]).replace(tzinfo=timezone.utc)
    return datetime.fromtimestamp(float(point["timestamp"]), timezone.utc)


def style_plot_axes(ax, title, x_label, y_label, *, show_grid=True):
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold", color="#111827", pad=16)
    ax.set_xlabel(x_label, fontsize=11, fontweight="bold", color="#334155", labelpad=10)
    ax.set_ylabel(y_label, fontsize=11, fontweight="bold", color="#334155", labelpad=10)
    if show_grid:
        ax.grid(True, axis="y", color="#e2e8f0", linewidth=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.tick_params(axis="both", colors="#475569", labelsize=9)


def render_multi_series_python_plot(plot):
    series = plot.get("series") or []
    series = [item for item in series if item.get("points")]
    if not series:
        return None

    spec = plot.get("plot_spec") or {}
    plot_type = spec.get("plot_type") or plot.get("plot_type") or "line"
    title = spec.get("title") or plot.get("title") or "Sleep Metrics"
    x_label = spec.get("x_label") or "Date"
    y_label = spec.get("y_label") or "; ".join(
        f"{item.get('label')} ({item.get('unit')})" for item in series if item.get("label")
    )

    fig, ax_minutes = plt.subplots(figsize=(10.6, 5.6), dpi=145)
    fig.patch.set_facecolor("#ffffff")
    ax_minutes.set_facecolor("#ffffff")

    if plot_type == "scatter" and len(series) >= 2:
        y_series = series[0]
        x_series = series[1]
        x_by_date = {point.get("date"): point for point in x_series.get("points", [])}
        y_by_date = {point.get("date"): point for point in y_series.get("points", [])}
        common_dates = sorted(set(x_by_date) & set(y_by_date))
        x_values = [float(x_by_date[date]["value"]) for date in common_dates]
        y_values = [float(y_by_date[date]["value"]) for date in common_dates]
        ax_minutes.scatter(x_values, y_values, s=70, color="#2563eb", edgecolor="#1e3a8a", linewidth=0.8)
        style_plot_axes(ax_minutes, title, x_label, y_label)
        ax_minutes.grid(True, color="#e2e8f0", linewidth=0.9)
        stats = f"{len(common_dates)} paired samples | source: {plot.get('source')}"
        ax_minutes.text(0, -0.22, stats, transform=ax_minutes.transAxes, fontsize=9, color="#64748b", va="top")
        fig.tight_layout(rect=[0, 0.08, 1, 1])
        return {
            "data_url": encoded_matplotlib_figure(fig),
            "format": "png",
            "renderer": "matplotlib",
            "plot_type": plot_type,
            "x_axis_label": x_label,
            "y_axis_label": y_label,
            "title": title,
            "series_count": len(series),
        }

    axes_by_unit = {"minutes": ax_minutes}
    colors = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#ea580c"]
    same_unit = len({item.get("unit") or "value" for item in series}) == 1

    for index, item in enumerate(series):
        unit = item.get("unit") or "value"
        axis = ax_minutes if same_unit else axes_by_unit.get(unit)
        if axis is None:
            axis = ax_minutes.twinx()
            axes_by_unit[unit] = axis
        points = item["points"]
        x_values = [point_datetime(point) for point in points]
        y_values = [float(point["value"]) for point in points]
        color = colors[index % len(colors)]
        label = item.get("label") or item.get("metric") or item.get("sensor")
        if plot_type == "bar":
            width = max(0.25, 0.8 / max(1, len(series)))
            offsets = [(index - (len(series) - 1) / 2) * width for index in range(len(series))]
            shifted = [value + timedelta(days=offsets[index]) for value in x_values]
            axis.bar(shifted, y_values, width=width, color=color, alpha=0.82, label=label)
        elif plot_type == "area":
            axis.plot(x_values, y_values, linewidth=2.4, color=color, label=label)
            axis.fill_between(x_values, y_values, min(y_values), color=color, alpha=0.18)
        else:
            axis.plot(x_values, y_values, linewidth=2.6, marker="o", markersize=5, color=color, label=label)
        axis.tick_params(axis="y", colors=color, labelsize=8)
        if axis is not ax_minutes:
            axis.spines["right"].set_color(color)

    style_plot_axes(ax_minutes, title, "Date", y_label)
    for unit, axis in axes_by_unit.items():
        if axis is not ax_minutes:
            axis.set_ylabel(unit.title(), fontsize=11, fontweight="bold")
    ax_minutes.grid(True, color="#e2e8f0", linewidth=0.8)
    ax_minutes.xaxis.set_major_formatter(mdates.DateFormatter("%b %-d"))
    fig.autofmt_xdate(rotation=28, ha="right")

    lines = []
    labels = []
    for axis in dict.fromkeys(axes_by_unit.values()):
        axis_lines, axis_labels = axis.get_legend_handles_labels()
        lines.extend(axis_lines)
        labels.extend(axis_labels)
    ax_minutes.legend(lines, labels, loc="upper left", fontsize=8, frameon=False)

    stats = "; ".join(
        (
            f"{item.get('label')}: min {item.get('min')}, "
            f"avg {item.get('average')}, max {item.get('max')}, "
            f"latest {item.get('latest')}"
        )
        for item in series
    )
    ax_minutes.text(
        0,
        -0.28,
        stats,
        transform=ax_minutes.transAxes,
        fontsize=8.2,
        color="#64748b",
        va="top",
    )
    fig.tight_layout(rect=[0, 0.1, 1, 1])
    y_axis_label = "; ".join(f"{item.get('label')} ({item.get('unit')})" for item in series if item.get("label"))

    return {
        "data_url": encoded_matplotlib_figure(fig),
        "format": "png",
        "renderer": "matplotlib",
        "plot_type": plot_type,
        "x_axis_label": "Date",
        "y_axis_label": y_axis_label,
        "title": title,
        "series_count": len(series),
    }


def render_python_plot(plot):
    if plot and plot.get("series"):
        return render_multi_series_python_plot(plot)
    if not plot or not plot.get("available") or not plot.get("points"):
        return None

    points = plot["points"]
    spec = plot.get("plot_spec") or {}
    plot_type = spec.get("plot_type") or plot.get("plot_type") or "line"
    x_values = [point_datetime(point) for point in points]
    y_values = [float(point["value"]) for point in points]
    y_label = spec.get("y_label") or plot_axis_label(plot)
    x_label = spec.get("x_label") or "Date"
    title = spec.get("title") or plot.get("title") or plot.get("sensor") or "Sensor plot"

    fig, ax = plt.subplots(figsize=(10.2, 5.4), dpi=145)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    if plot_type == "histogram":
        bins = min(12, max(5, int(len(y_values) ** 0.5) + 1))
        ax.hist(y_values, bins=bins, color="#2563eb", edgecolor="#1e3a8a", alpha=0.82)
        x_label = y_label
        y_label = "Number of Nights"
    elif plot_type == "bar":
        ax.bar(x_values, y_values, width=0.72, color="#2563eb", alpha=0.86)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %-d"))
        fig.autofmt_xdate(rotation=28, ha="right")
    elif plot_type == "box":
        ax.boxplot(
            y_values,
            vert=True,
            patch_artist=True,
            labels=[plot.get("label") or plot.get("sensor") or "Values"],
            boxprops={"facecolor": "#dbeafe", "color": "#2563eb"},
            medianprops={"color": "#dc2626", "linewidth": 2},
        )
        x_label = ""
    elif plot_type == "area":
        ax.plot(x_values, y_values, color="#2563eb", linewidth=2.6)
        ax.fill_between(x_values, y_values, min(y_values), color="#93c5fd", alpha=0.35)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %-d"))
        fig.autofmt_xdate(rotation=28, ha="right")
    else:
        ax.plot(x_values, y_values, color="#2563eb", linewidth=2.6, marker="o", markersize=5.2)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %-d"))
        fig.autofmt_xdate(rotation=28, ha="right")

    style_plot_axes(ax, title, x_label, y_label)

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

    return {
        "data_url": encoded_matplotlib_figure(fig),
        "format": "png",
        "renderer": "matplotlib",
        "plot_type": plot_type,
        "x_axis_label": x_label,
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


def encoded_svg_image(svg):
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def render_n_of_1_association_plot(analysis):
    ranked = analysis.get("ranked_associations") or []
    rows = []
    for predictor in ranked[:6]:
        sensor = predictor.get("sensor") or "unknown"
        label = sensor.replace("binary_sensor.", "").replace("sensor.", "").replace("_", " ")
        for association in predictor.get("lagged_associations", []):
            rows.append(
                {
                    "label": f"{label} (lag {association.get('lag_days')})",
                    "r": float(association.get("pearson_r") or 0),
                    "n": association.get("n"),
                    "slope": association.get("slope_minutes_asleep_per_feature_unit"),
                }
            )
    rows = sorted(rows, key=lambda item: abs(item["r"]), reverse=True)[:8]
    if not rows:
        return None

    fig, ax = plt.subplots(figsize=(10.8, 5.8), dpi=150)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    labels = [row["label"] for row in rows][::-1]
    values = [row["r"] for row in rows][::-1]
    colors = ["#2563eb" if value >= 0 else "#dc2626" for value in values]
    ax.barh(labels, values, color=colors, alpha=0.86)
    ax.axvline(0, color="#334155", linewidth=1.2)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Pearson r", fontsize=11, fontweight="bold", color="#334155")
    ax.set_title("N-of-1 Associations With Minutes Asleep", loc="left", fontsize=16, fontweight="bold", pad=16)
    ax.grid(True, axis="x", color="#e2e8f0", linewidth=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.tick_params(axis="x", colors="#475569", labelsize=9)
    ax.tick_params(axis="y", colors="#334155", labelsize=9)
    for index, row in enumerate(rows[::-1]):
        value = row["r"]
        x_offset = 0.025 if value >= 0 else -0.025
        ha = "left" if value >= 0 else "right"
        ax.text(
            value + x_offset,
            index,
            f"r={value:.3f}, N={row['n']}, slope={row['slope']}",
            va="center",
            ha=ha,
            fontsize=8.2,
            color="#111827",
        )
    outcome = analysis.get("outcome") or {}
    footer = (
        f"Outcome: {outcome.get('metric', 'minutes_asleep')} | "
        f"{outcome.get('days_returned')} nights | {outcome.get('date_range')} | observational screening"
    )
    ax.text(0, -0.18, footer, transform=ax.transAxes, fontsize=9, color="#64748b", va="top")
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    return {
        "type": "plot",
        "title": "N-of-1 Association Plot",
        "description": "Ranked same-day and prior-day associations with completed minutes asleep.",
        "data_url": encoded_matplotlib_figure(fig),
        "format": "png",
        "renderer": "matplotlib",
    }


def causal_dag_svg(title, nodes, edges):
    width = 920
    height = 430
    node_width = 210
    node_height = 62
    node_lookup = {node["id"]: node for node in nodes}

    defs = """
    <defs>
      <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path d="M 0 0 L 10 5 L 0 10 z" fill="#334155"/>
      </marker>
      <filter id="shadow" x="-10%" y="-10%" width="120%" height="130%">
        <feDropShadow dx="0" dy="4" stdDeviation="5" flood-color="#0f172a" flood-opacity="0.12"/>
      </filter>
    </defs>
    """
    edge_parts = []
    for edge in edges:
        start = node_lookup[edge["from"]]
        end = node_lookup[edge["to"]]
        x1 = start["x"] + node_width
        y1 = start["y"] + node_height / 2
        x2 = end["x"]
        y2 = end["y"] + node_height / 2
        if start["x"] > end["x"]:
            x1 = start["x"]
            x2 = end["x"] + node_width
        mid = (x1 + x2) / 2
        path = f"M{x1},{y1} C{mid},{y1} {mid},{y2} {x2},{y2}"
        label = edge.get("label")
        edge_parts.append(
            f'<path d="{path}" fill="none" stroke="#334155" stroke-width="2.2" marker-end="url(#arrow)"/>'
        )
        if label:
            edge_parts.append(
                f'<text x="{mid}" y="{(y1 + y2) / 2 - 8}" text-anchor="middle" font-size="13" '
                f'font-weight="700" fill="#475569">{xml_escape(label)}</text>'
            )

    node_parts = []
    for node in nodes:
        fill = node.get("fill", "#eff6ff")
        stroke = node.get("stroke", "#2563eb")
        node_parts.append(
            f'<rect x="{node["x"]}" y="{node["y"]}" width="{node_width}" height="{node_height}" '
            f'rx="12" fill="{fill}" stroke="{stroke}" stroke-width="2" filter="url(#shadow)"/>'
        )
        label = xml_escape(node["label"])
        words = label.split()
        lines = []
        current = ""
        for word in words:
            next_line = f"{current} {word}".strip()
            if len(next_line) > 24 and current:
                lines.append(current)
                current = word
            else:
                current = next_line
        if current:
            lines.append(current)
        start_y = node["y"] + 27 - (len(lines) - 1) * 8
        for line_index, line in enumerate(lines[:3]):
            node_parts.append(
                f'<text x="{node["x"] + node_width / 2}" y="{start_y + line_index * 17}" '
                f'text-anchor="middle" font-size="14" font-weight="700" fill="#0f172a">{xml_escape(line)}</text>'
            )

    svg = f"""
    <svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{xml_escape(title)}">
      {defs}
      <rect width="{width}" height="{height}" fill="#ffffff"/>
      <text x="28" y="36" font-size="22" font-weight="800" fill="#111827">{xml_escape(title)}</text>
      <text x="28" y="62" font-size="13" fill="#64748b">Conceptual DAG for observational N-of-1 screening; unmeasured confounding is shown explicitly.</text>
      {"".join(edge_parts)}
      {"".join(node_parts)}
    </svg>
    """
    return {
        "type": "dag",
        "title": title,
        "description": "Causal assumptions diagram, not proof of causal effect.",
        "data_url": encoded_svg_image(svg),
        "format": "svg",
        "renderer": "svg",
    }


def build_n_of_1_dags(analysis):
    ranked = analysis.get("ranked_associations") or []
    top_sensor = ranked[0].get("sensor") if ranked else "Selected Sensor Map Predictor"
    top_label = top_sensor.replace("binary_sensor.", "").replace("sensor.", "").replace("_", " ").title()
    outcome = "Minutes Asleep (day t)"
    return [
        causal_dag_svg(
            "Same-Day Association DAG",
            [
                {
                    "id": "conf",
                    "label": "Unmeasured context: stress, illness, schedule",
                    "x": 355,
                    "y": 92,
                    "fill": "#fff7ed",
                    "stroke": "#ea580c",
                },
                {"id": "x", "label": f"{top_label} (day t)", "x": 92, "y": 228, "fill": "#eff6ff", "stroke": "#2563eb"},
                {"id": "y", "label": outcome, "x": 618, "y": 228, "fill": "#ecfdf5", "stroke": "#059669"},
            ],
            [
                {"from": "x", "to": "y", "label": "estimated r/slope"},
                {"from": "conf", "to": "x", "label": "may affect"},
                {"from": "conf", "to": "y", "label": "may affect"},
            ],
        ),
        causal_dag_svg(
            "Lagged N-of-1 DAG",
            [
                {
                    "id": "conf",
                    "label": "Prior context and routines",
                    "x": 355,
                    "y": 92,
                    "fill": "#fff7ed",
                    "stroke": "#ea580c",
                },
                {
                    "id": "x",
                    "label": f"{top_label} (day t-1)",
                    "x": 92,
                    "y": 228,
                    "fill": "#eff6ff",
                    "stroke": "#2563eb",
                },
                {"id": "y", "label": outcome, "x": 618, "y": 228, "fill": "#ecfdf5", "stroke": "#059669"},
            ],
            [
                {"from": "x", "to": "y", "label": "prior-day association"},
                {"from": "conf", "to": "x", "label": "history"},
                {"from": "conf", "to": "y", "label": "history"},
            ],
        ),
    ]


def n_of_1_latex_equations():
    return [
        {
            "title": "Pearson correlation",
            "latex": (
                r"r = \frac{\sum_{i=1}^{N}(X_i-\bar{X})(Y_i-\bar{Y})}"
                r"{\sqrt{\sum_{i=1}^{N}(X_i-\bar{X})^2}\sqrt{\sum_{i=1}^{N}(Y_i-\bar{Y})^2}}"
            ),
            "description": "Within-person association between a daily predictor and completed minutes asleep.",
        },
        {
            "title": "Linear slope",
            "latex": (
                r"\hat{\beta} = \frac{\sum_{i=1}^{N}(X_i-\bar{X})(Y_i-\bar{Y})}"
                r"{\sum_{i=1}^{N}(X_i-\bar{X})^2}"
            ),
            "description": "Estimated minutes asleep per one-unit increase in the predictor.",
        },
        {
            "title": "Lagged exposure alignment",
            "latex": r"Y_t = \alpha + \beta X_{t-k} + \epsilon_t,\quad k \in \{0,1\}",
            "description": "Same-day uses k=0; prior-day uses k=1.",
        },
    ]


def build_analysis_visuals(user_text, ha_summary):
    analysis = (ha_summary or {}).get("n_of_1_analysis")
    if not analysis or not analysis.get("available") or not should_build_analysis_visuals(user_text):
        return None

    artifacts = []
    plot = render_n_of_1_association_plot(analysis)
    if plot:
        artifacts.append(plot)
    artifacts.extend(build_n_of_1_dags(analysis))
    artifacts.extend({"type": "latex", **equation} for equation in n_of_1_latex_equations())
    return {
        "available": True,
        "title": "N-of-1 Analysis Visuals",
        "artifacts": artifacts,
        "summary": "Rendered deterministic association plot, causal DAGs, and LaTeX equations for the N-of-1 analysis.",
    }


def query_cleaned_sleep_points(entity_id, days=30, plot_type="line"):
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
    label = SLEEP_PLOT_METRIC_LABELS.get(metric, metric.replace("_", " ").title())
    unit = SLEEP_PLOT_METRIC_UNITS.get(metric, "value")
    y_label = f"{label} ({unit})"
    plot_spec = build_plot_spec(
        plot_type=plot_type,
        entity_ids=[entity_id],
        title=plot_title_for_metric(metric, days, plot_type),
        x_label="Date",
        y_label=y_label,
        days=summary["days_requested"],
    )
    return {
        "available": True,
        "sensor": entity_id,
        "label": label,
        "unit": unit,
        "plot_type": plot_type,
        "plot_spec": plot_spec,
        "days": summary["days_requested"],
        "points": points,
        "min": round(min(values), 2),
        "max": round(max(values), 2),
        "average": round(mean(values), 2),
        "latest": round(values[-1], 2),
        "samples": len(points),
        "cleaned": True,
        "source": "completed_sleep",
        "query": summary.get("query") or recorder_query_info([entity_id], days),
        "metric": metric,
        "date_range": summary["date_range"],
    }


def requested_sleep_plot_metrics(user_text):
    lowered = user_text.lower()
    if "sleep" not in lowered and "asleep" not in lowered:
        return []

    metrics = []
    if any(term in lowered for term in ("awakening", "awakenings", "awake count", "wakeup count")):
        metrics.append("awakenings")
    if "time in bed" in lowered:
        metrics.append("time_in_bed")
    if any(term in lowered for term in ("sleep time", "sleep duration", "minutes asleep", "time asleep", "asleep")):
        if "time_in_bed" not in metrics:
            metrics.append("minutes_asleep")
    elif "sleep" in lowered and "time_in_bed" not in metrics:
        metrics.append("minutes_asleep")
    if "efficiency" in lowered:
        metrics.append("efficiency")
    if "minutes awake" in lowered or "awake time" in lowered:
        metrics.append("minutes_awake")

    deduped = []
    for metric in metrics:
        if metric not in deduped:
            deduped.append(metric)
    return deduped


def build_multi_sleep_plot(metrics, days, plot_type="line"):
    summary = summarize_completed_sleep(days=days)
    if not summary.get("available"):
        return {
            "available": False,
            "message": summary.get("message", "No completed sleep records were found."),
            "points": [],
            "series": [],
            "cleaned": True,
        }

    series = []
    for metric in metrics:
        points = []
        for record in summary["daily"]:
            value = record.get(metric)
            if value is None:
                continue
            points.append(
                {
                    "time": record["date_label"],
                    "timestamp": datetime.fromisoformat(record["updated_local"]).timestamp(),
                    "value": value,
                    "date": record["date"],
                    "label": record.get("sleep_label") if metric == "minutes_asleep" else None,
                }
            )
        if not points:
            continue
        values = [point["value"] for point in points]
        series.append(
            {
                "metric": metric,
                "sensor": SLEEP_METRIC_ENTITIES[metric],
                "label": SLEEP_PLOT_METRIC_LABELS.get(metric, metric.replace("_", " ").title()),
                "unit": SLEEP_PLOT_METRIC_UNITS.get(metric, "value"),
                "points": points,
                "min": round(min(values), 2),
                "max": round(max(values), 2),
                "average": round(mean(values), 2),
                "latest": round(values[-1], 2),
                "samples": len(points),
            }
        )

    if not series:
        return {
            "available": False,
            "message": "No cleaned sleep points were found for the requested metrics.",
            "points": [],
            "series": [],
            "cleaned": True,
        }

    primary = next((item for item in series if item["metric"] == "minutes_asleep"), series[0])
    title = " and ".join(item["label"] for item in series)
    entity_ids = [item["sensor"] for item in series]
    if plot_type == "scatter" and len(series) >= 2:
        y_series = series[0]
        x_series = series[1]
        spec_title = f"{y_series['label']} vs {x_series['label']}"
        x_label = f"{x_series['label']} ({x_series['unit']})"
        y_label = f"{y_series['label']} ({y_series['unit']})"
    else:
        spec_title = title
        x_label = "Date"
        y_label = "; ".join(f"{item['label']} ({item['unit']})" for item in series)
    plot_spec = build_plot_spec(
        plot_type=plot_type,
        entity_ids=entity_ids,
        title=spec_title,
        x_label=x_label,
        y_label=y_label,
        days=summary["days_requested"],
    )
    return {
        "available": True,
        "sensor": "sleep_metrics",
        "title": spec_title,
        "plot_type": plot_type,
        "plot_spec": plot_spec,
        "days": summary["days_requested"],
        "points": primary["points"],
        "series": series,
        "min": primary["min"],
        "max": primary["max"],
        "average": primary["average"],
        "latest": primary["latest"],
        "samples": max(item["samples"] for item in series),
        "cleaned": True,
        "source": "completed_sleep",
        "query": summary.get("query") or recorder_query_info(entity_ids, days),
        "metric": "multi_sleep_metrics",
        "date_range": summary["date_range"],
    }


def query_sensor_points(entity_id, days=30, limit=500, plot_type="line"):
    if not home_assistant_token() and not recorder_db_path():
        raise RuntimeError("SUPERVISOR_TOKEN is not available.")

    days = max(1, min(int(days), 365))
    limit = max(10, min(int(limit), 2000))

    cleaned_sleep_points = query_cleaned_sleep_points(entity_id, days=days, plot_type=plot_type)
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
    y_label = "Value"
    plot_spec = build_plot_spec(
        plot_type=plot_type,
        entity_ids=[entity_id],
        title=f"{entity_id} Over {plot_time_window_label(days).replace('_', ' ').title()}",
        x_label="Date",
        y_label=y_label,
        days=days,
        aggregation="raw_history",
    )
    return {
        "available": True,
        "sensor": entity_id,
        "plot_type": plot_type,
        "plot_spec": plot_spec,
        "days": days,
        "points": points,
        "min": round(min(values), 2),
        "max": round(max(values), 2),
        "average": round(mean(values), 2),
        "latest": round(values[-1], 2),
        "samples": len(points),
        "cleaned": False,
        "source": recorder_query_info([entity_id], days, limit=limit)["source"],
        "query": recorder_query_info([entity_id], days, limit=limit),
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


def prompt_mentions_entity_id(user_text):
    lowered = user_text.lower()
    if re.search(r"\b[a-z_]+\.[a-z0-9_]+\b", lowered):
        return True
    return any(sensor.lower() in lowered for sensor in sensor_map_entity_ids())


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


def should_build_n_of_1_analysis(user_text):
    lowered = user_text.lower()
    relation_requested = any(term in lowered for term in RELATION_REQUEST_TERMS)
    n_of_1_requested = any(term in lowered for term in N_OF_1_REQUEST_TERMS)
    visual_requested = should_build_analysis_visuals(user_text)
    mentions_sleep = any(term in lowered for term in ("sleep", "asleep", "time asleep", "minutes asleep"))
    return (mentions_sleep and (relation_requested or n_of_1_requested)) or visual_requested


def is_sleep_related_sensor(entity_id):
    lowered = str(entity_id or "").lower()
    return lowered in SLEEP_METRIC_ENTITIES.values() or any(term in lowered for term in SLEEP_ENTITY_TERMS)


def pearson_correlation(x_values, y_values):
    pairs = [(float(x), float(y)) for x, y in zip(x_values, y_values) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    x_mean = mean(x for x, _ in pairs)
    y_mean = mean(y for _, y in pairs)
    x_diffs = [x - x_mean for x, _ in pairs]
    y_diffs = [y - y_mean for _, y in pairs]
    x_var = sum(value * value for value in x_diffs)
    y_var = sum(value * value for value in y_diffs)
    if x_var <= 0 or y_var <= 0:
        return None
    return sum(x_diff * y_diff for x_diff, y_diff in zip(x_diffs, y_diffs)) / sqrt(x_var * y_var)


def linear_slope(x_values, y_values):
    pairs = [(float(x), float(y)) for x, y in zip(x_values, y_values) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    x_mean = mean(x for x, _ in pairs)
    y_mean = mean(y for _, y in pairs)
    denominator = sum((x - x_mean) ** 2 for x, _ in pairs)
    if denominator <= 0:
        return None
    return sum((x - x_mean) * (y - y_mean) for x, y in pairs) / denominator


def sensor_daily_feature(row, days):
    sensor = row.get("sensor")
    result = {
        "sensor": sensor,
        "description": row.get("description", ""),
        "available": False,
        "feature": None,
        "daily": {},
        "samples": 0,
    }
    if not sensor:
        result["message"] = "Sensor map row has no entity ID."
        return result
    if is_sleep_related_sensor(sensor):
        result["message"] = "Sleep outcome sensors are excluded as predictors."
        return result

    try:
        rows = query_entity_history_rows(sensor, days=days, limit=MAX_CONTEXT_SENSOR_ROWS)
    except Exception as exc:
        result["message"] = f"Could not read sensor history: {exc}"
        return result

    grouped = defaultdict(lambda: {"numeric": [], "active_events": 0, "states": Counter()})
    for history_row in rows:
        date = local_datetime(history_row["updated_ts"]).date().isoformat()
        state = str(history_row.get("state", ""))
        number = parse_number(state)
        if number is not None:
            grouped[date]["numeric"].append(number)
        else:
            grouped[date]["states"][state] += 1
        if state.lower() in ACTIVE_STATES:
            grouped[date]["active_events"] += 1

    daily = {}
    numeric_day_count = 0
    for date, values in grouped.items():
        if values["numeric"]:
            numeric_day_count += 1
            daily[date] = round(mean(values["numeric"]), 4)
        else:
            daily[date] = values["active_events"]

    if not daily:
        result["message"] = "No usable numeric or active-state history was found."
        return result

    result.update(
        {
            "available": True,
            "feature": "daily_mean" if numeric_day_count else "daily_active_event_count",
            "daily": daily,
            "samples": len(daily),
            "date_range": f"{min(daily)} to {max(daily)}",
        }
    )
    return result


def paired_sleep_feature_values(sleep_by_date, feature_by_date, lag_days=0):
    pairs = []
    for date, sleep_value in sleep_by_date.items():
        feature_date = (datetime.fromisoformat(date).date() - timedelta(days=lag_days)).isoformat()
        feature_value = feature_by_date.get(feature_date)
        if feature_value is None:
            continue
        pairs.append(
            {
                "date": date,
                "feature_date": feature_date,
                "x": float(feature_value),
                "y": float(sleep_value),
            }
        )
    return pairs


def correlation_summary_for_pairs(pairs):
    if len(pairs) < 3:
        return None
    x_values = [pair["x"] for pair in pairs]
    y_values = [pair["y"] for pair in pairs]
    correlation = pearson_correlation(x_values, y_values)
    slope = linear_slope(x_values, y_values)
    if correlation is None or slope is None:
        return None
    return {
        "n": len(pairs),
        "pearson_r": round(correlation, 3),
        "slope_minutes_asleep_per_feature_unit": round(slope, 3),
        "x_min": round(min(x_values), 3),
        "x_max": round(max(x_values), 3),
        "y_min": round(min(y_values), 3),
        "y_max": round(max(y_values), 3),
        "paired_dates": [pair["date"] for pair in pairs],
    }


def build_n_of_1_sleep_analysis(user_text, days, completed_sleep=None):
    if not should_build_n_of_1_analysis(user_text):
        return None

    completed_sleep = completed_sleep or summarize_completed_sleep(days=days)
    if not completed_sleep or not completed_sleep.get("available"):
        return {
            "available": False,
            "message": "No completed sleep records were available for N-of-1 analysis.",
            "method": "Requires completed daily sleep records and at least one non-sleep Sensor Maps predictor.",
        }

    mapped_rows = [row for row in load_sensor_map()["sensors"] if row.get("sensor")]
    predictor_rows = [row for row in mapped_rows if not is_sleep_related_sensor(row.get("sensor"))]
    sleep_by_date = {
        record["date"]: float(record["minutes_asleep"])
        for record in completed_sleep.get("daily", [])
        if record.get("minutes_asleep") is not None
    }

    predictors = []
    for row in predictor_rows[:MAX_CONTEXT_SENSORS]:
        feature = sensor_daily_feature(row, days=days)
        predictor = {
            "sensor": row.get("sensor"),
            "description": row.get("description", ""),
            "feature": feature.get("feature"),
            "available": feature.get("available"),
            "samples": feature.get("samples", 0),
            "message": feature.get("message"),
        }
        if feature.get("available"):
            lag_summaries = []
            for lag_days in (0, 1):
                pairs = paired_sleep_feature_values(sleep_by_date, feature["daily"], lag_days=lag_days)
                summary = correlation_summary_for_pairs(pairs)
                if summary:
                    summary["lag_days"] = lag_days
                    summary["interpretation"] = (
                        "same-day association"
                        if lag_days == 0
                        else "prior-day exposure association with next sleep outcome"
                    )
                    lag_summaries.append(summary)
            predictor["lagged_associations"] = lag_summaries
            predictor["daily_feature_values"] = [
                {"date": date, "value": feature["daily"][date]} for date in sorted(feature["daily"])[-14:]
            ]
        predictors.append(predictor)

    usable = [item for item in predictors if item.get("lagged_associations")]
    for item in usable:
        item["max_abs_r"] = max(abs(summary["pearson_r"]) for summary in item["lagged_associations"])
    usable.sort(key=lambda item: (-item["max_abs_r"], item["sensor"]))

    outcome_values = list(sleep_by_date.values())
    outcome_lag_pairs = [(outcome_values[index - 1], outcome_values[index]) for index in range(1, len(outcome_values))]
    autocorrelation = (
        pearson_correlation(
            [pair[0] for pair in outcome_lag_pairs],
            [pair[1] for pair in outcome_lag_pairs],
        )
        if len(outcome_lag_pairs) >= 3
        else None
    )

    return {
        "available": bool(usable),
        "paper_reference": {
            "url": "https://arxiv.org/abs/2407.17666",
            "title": "Causal estimands and identification of time-varying effects in non-stationary time series from N-of-1 mobile device data",
        },
        "method": (
            "Deterministic N-of-1 screening: align one person's daily completed minutes_asleep "
            "with Sensor Maps predictors, compute same-day and prior-day Pearson correlations, "
            "linear slopes, sample sizes, outcome autocorrelation, and positivity/range diagnostics. "
            "This is an observational approximation inspired by the paper's time-varying exposure, "
            "lagged-history, and positivity concepts; it is not a full state-space g-formula causal estimate."
        ),
        "outcome": {
            "sensor": SLEEP_METRIC_ENTITIES["minutes_asleep"],
            "metric": "minutes_asleep",
            "days_requested": days,
            "days_returned": len(sleep_by_date),
            "date_range": completed_sleep.get("date_range"),
            "mean": round(mean(outcome_values), 2) if outcome_values else None,
            "lag1_autocorrelation": round(autocorrelation, 3) if autocorrelation is not None else None,
        },
        "predictors_tested": predictors,
        "ranked_associations": usable[:8],
        "positivity_diagnostics": [
            {
                "sensor": item["sensor"],
                "feature": item.get("feature"),
                "n_feature_days": item.get("samples"),
                "status": "ok" if item.get("samples", 0) >= 5 else "limited",
            }
            for item in predictors
        ],
        "query": {
            "source": recorder_query_info(
                [SLEEP_METRIC_ENTITIES["minutes_asleep"], *[row.get("sensor") for row in predictor_rows]],
                days,
                limit=MAX_CONTEXT_SENSOR_ROWS,
            ),
            "sleep_source": completed_sleep.get("query"),
        },
        "message": (
            "Use ranked_associations and report N, r, lag, slope, and caution about observational N-of-1 inference."
            if usable
            else "No Sensor Maps predictors had enough overlapping daily data for correlation."
        ),
    }


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

    days = requested_days_from_text(user_text)
    plot_type = detect_plot_type(user_text)
    sleep_metrics = requested_sleep_plot_metrics(user_text)
    if plot_type == "scatter" and len(sleep_metrics) < 2:
        if "awakening" in user_text.lower() or "awakenings" in user_text.lower():
            sleep_metrics = ["awakenings", "minutes_asleep"]
    if len(sleep_metrics) > 1:
        return attach_python_plot(build_multi_sleep_plot(sleep_metrics, days=days, plot_type=plot_type))
    if len(sleep_metrics) == 1:
        sensor = SLEEP_METRIC_ENTITIES[sleep_metrics[0]]
        return attach_python_plot(query_sensor_points(sensor, days=days, plot_type=plot_type))

    if plot_type in {"bar", "histogram", "box"} and not prompt_mentions_entity_id(user_text):
        sensor = SLEEP_METRIC_ENTITIES["minutes_asleep"]
    else:
        sensor = preferred_plot_sensor(user_text)
    if not sensor:
        return {
            "available": False,
            "message": "No sensor map rows are available to plot.",
            "points": [],
        }

    return attach_python_plot(query_sensor_points(sensor, days=days, plot_type=plot_type))


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

    if plot.get("series"):
        summary = {
            "title": plot.get("title"),
            "plot_spec": plot.get("plot_spec"),
            "query": plot.get("query"),
            "days": plot["days"],
            "date_range": plot.get("date_range"),
            "series": [
                {
                    "metric": item.get("metric"),
                    "sensor": item.get("sensor"),
                    "unit": item.get("unit"),
                    "samples": item.get("samples"),
                    "min": item.get("min"),
                    "max": item.get("max"),
                    "average": item.get("average"),
                    "latest": item.get("latest"),
                }
                for item in plot["series"]
            ],
        }
    else:
        summary = {
            "sensor": plot["sensor"],
            "plot_spec": plot.get("plot_spec"),
            "query": plot.get("query"),
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


def analysis_visuals_context(analysis_visuals):
    if not analysis_visuals:
        return ""
    return (
        "\nThe app rendered visual artifacts for this response: an N-of-1 association plot, "
        "causal DAG SVG diagrams, and LaTeX equation cards. Do not describe non-existent "
        "attachments or ask the user to use a notebook. Refer to the rendered cards below "
        "the message and summarize what each visual shows.\n"
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
    sensor_catalog = discover_sensor_catalog(user_text)
    include_sleep = should_include_sleep_context(user_text, mapped_rows)
    include_n_of_1 = should_build_n_of_1_analysis(user_text)
    include_sleep = include_sleep or include_n_of_1
    completed_sleep = summarize_completed_sleep(days=days) if include_sleep else None
    mapped_history_rows = mapped_rows
    if include_sleep:
        sleep_entities = set(SLEEP_METRIC_ENTITIES.values())
        mapped_history_rows = [row for row in mapped_rows if row.get("sensor") not in sleep_entities]
    if include_n_of_1:
        mapped_history_rows = []
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
    n_of_1_analysis = build_n_of_1_sleep_analysis(user_text, days=days, completed_sleep=completed_sleep)

    return {
        "available": home_assistant_available(),
        "source": "home_assistant_api",
        "api_url": home_assistant_api_url(),
        "read_only": True,
        "time_zone": home_assistant_time_zone(),
        "days_requested": days,
        "sensor_map": load_sensor_map()["sensors"],
        "sensor_catalog_search": sensor_catalog,
        "completed_sleep": completed_sleep,
        "mapped_sensor_history": mapped_sensor_history,
        "n_of_1_analysis": n_of_1_analysis,
        "guidance": (
            "If sensor_catalog_search.searched is true, use its matches to answer "
            "questions about whether the user has a sensor, even when that sensor is "
            "not in Sensor Maps yet. Mention exact entity_id values from the matches. "
            "If n_of_1_analysis is present, use ranked_associations for any correlation "
            "or N-of-1 inference answer; report N, lag, Pearson r, slope, date range, "
            "and the observational limitation. Do not invent causal claims beyond this "
            "deterministic analysis. Do not compute correlations from sensor_map or "
            "mapped_sensor_history when n_of_1_analysis is available. "
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
        "the Supervisor token and may read the mapped recorder SQLite database when available. If "
        "sensor_catalog_search is present and searched=true, it is a read-only search of "
        "Home Assistant entity metadata/current states; use the exact matching entity_id values. If "
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


def append_assistant_message(conversation, content, extra=None):
    assistant_message = {"role": "assistant", "content": content}
    if extra:
        assistant_message.update(extra)
    conversation["messages"].append(assistant_message)
    touch_conversation(conversation)
    save_chat_store()
    return assistant_message


def message_response_payload(conversation, assistant_text, provider=None, model=None, extra=None):
    payload = {
        "chat": chat_summary(conversation),
        "chats": sorted_chat_summaries(),
        "active_chat_id": conversation["id"],
        "assistant": assistant_text,
        "messages": conversation["messages"],
    }
    if provider:
        payload["provider"] = provider["id"]
        payload["provider_info"] = dict(provider)
    if model:
        payload["model"] = model["id"]
        payload["model_info"] = compact_model_info(model)
    if extra:
        payload.update(extra)
    return payload


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


def handle_sensor_map_confirmation(conversation, user_text):
    pending = conversation.get("pending_action")
    if not isinstance(pending, dict) or pending.get("type") != "add_sensor_map":
        return None
    if text_denies(user_text):
        conversation.pop("pending_action", None)
        assistant_text = f"Okay. I did not add `{pending.get('sensor')}` to the sensor map."
        append_assistant_message(
            conversation,
            assistant_text,
            {"sensor_map_action": {"type": "add_sensor_map_cancelled", "sensor": pending.get("sensor")}},
        )
        return assistant_text
    if not text_confirms(user_text):
        return None

    sensor = normalize_entity_id(pending.get("sensor"))
    description = str(pending.get("description") or describe_sensor_map_candidate(sensor)).strip()
    current_rows = load_sensor_map().get("sensors", [])
    existing = {row.get("sensor") for row in current_rows}
    if sensor in existing:
        assistant_text = f"`{sensor}` is already in the sensor map, so I did not add a duplicate."
        action_type = "add_sensor_map_already_exists"
        saved = load_sensor_map()
    else:
        saved = save_sensor_map([*current_rows, {"sensor": sensor, "description": description}])
        assistant_text = f"Added `{sensor}` to the sensor map with description: {description}"
        action_type = "add_sensor_map_completed"

    conversation.pop("pending_action", None)
    append_assistant_message(
        conversation,
        assistant_text,
        {
            "sensor_map_action": {
                "type": action_type,
                "sensor": sensor,
                "description": description,
                "sensors_count": len(saved.get("sensors", [])),
            }
        },
    )
    return assistant_text


def handle_deterministic_message_action(conversation, user_text):
    confirmation = handle_sensor_map_confirmation(conversation, user_text)
    if confirmation is not None:
        return confirmation

    if detect_app_delete_request(user_text):
        conversation.pop("pending_action", None)
        assistant_text = (
            "I cannot delete or uninstall a Home Assistant add-on for you. "
            "You need to do that manually in Home Assistant: Settings > Add-ons > "
            "Agentic Prompt App > Uninstall or Remove."
        )
        append_assistant_message(
            conversation,
            assistant_text,
            {"manual_action_required": {"type": "delete_addon", "target": "agentic_prompt_app"}},
        )
        return assistant_text

    add_request = detect_sensor_map_add_request(user_text)
    if add_request:
        conversation["pending_action"] = add_request
        assistant_text = (
            f"I found `{add_request['sensor']}`. Do you want me to add it to the sensor map "
            f"with this description: {add_request['description']} Reply yes to add it, or no to cancel."
        )
        append_assistant_message(
            conversation,
            assistant_text,
            {
                "sensor_map_action": {
                    "type": "add_sensor_map_confirmation_required",
                    "sensor": add_request["sensor"],
                    "description": add_request["description"],
                }
            },
        )
        return assistant_text

    return None


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
    model_info = compact_model_info(model)

    conversation = None
    deterministic_possible = bool(detect_app_delete_request(user_text) or detect_sensor_map_add_request(user_text))
    if not deterministic_possible and payload.get("chat_id"):
        existing_conversation = get_conversation(payload.get("chat_id"), create=False)
        pending = existing_conversation.get("pending_action") if existing_conversation else None
        deterministic_possible = isinstance(pending, dict) and pending.get("type") == "add_sensor_map"

    if deterministic_possible:
        conversation = get_conversation(payload.get("chat_id"))
        if not conversation["messages"] and conversation.get("title") == "New chat":
            conversation["title"] = title_from_text(user_text)
        conversation["messages"].append({"role": "user", "content": user_text})
        touch_conversation(conversation)

        deterministic_assistant = handle_deterministic_message_action(conversation, user_text)
        if deterministic_assistant is not None:
            return jsonify(
                message_response_payload(
                    conversation,
                    deterministic_assistant,
                    provider=provider,
                    model=model,
                    extra={
                        "response_id": None,
                        "home_assistant_context": None,
                        "plot": None,
                        "sensor_data": None,
                        "analysis_visuals": None,
                    },
                )
            )

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

    if conversation is None:
        conversation = get_conversation(payload.get("chat_id"))
        if not conversation["messages"] and conversation.get("title") == "New chat":
            conversation["title"] = title_from_text(user_text)
        conversation["messages"].append({"role": "user", "content": user_text})
        touch_conversation(conversation)

    plot = safe_build_plot_for_prompt(user_text)
    sensor_data = safe_build_sensor_data_for_prompt(user_text)
    model_input, ha_summary = build_model_input(user_text)
    analysis_visuals = build_analysis_visuals(user_text, ha_summary)
    model_input = (
        f"{model_input}{plot_context(plot)}"
        f"{analysis_visuals_context(analysis_visuals)}"
        f"{sensor_data_context(sensor_data)}"
    )

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
    if analysis_visuals:
        assistant_message["analysis_visuals"] = analysis_visuals
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
            "analysis_visuals": analysis_visuals,
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
