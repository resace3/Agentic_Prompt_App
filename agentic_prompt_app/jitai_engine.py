import json
import os
import shutil
import urllib.error
import urllib.request
from datetime import datetime, time, timezone
from uuid import uuid4


INVALID_SENSOR_STATES = {"", "unknown", "unavailable", "none", "null"}
SUPPORTED_OPERATORS = {">", ">=", "<", "<=", "==", "!="}
DEFAULT_STORE = {"jitais": [], "runtime": {}, "events": []}
MAX_EVENTS = 1000


class JitaiError(Exception):
    def __init__(self, message, status=400, error_type="jitai_error", details_safe=None):
        super().__init__(message)
        self.message = message
        self.status = status
        self.error_type = error_type
        self.details_safe = details_safe or {}


def utc_now():
    return datetime.now(timezone.utc)


def isoformat(value):
    if not value:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def parse_datetime(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def jitai_store_path():
    return os.environ.get("JITAI_STORE_PATH") or "/data/jitai_store.json"


def fallback_store_path(path):
    directory = os.path.dirname(path) or "."
    if os.path.isdir(directory) and os.access(directory, os.W_OK):
        return path
    return os.path.join(os.getcwd(), "jitai_store.json")


def load_store(path=None):
    path = fallback_store_path(path or jitai_store_path())
    if not os.path.exists(path):
        return {**DEFAULT_STORE, "storage_path": path}

    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        backup_path = f"{path}.corrupt.{int(utc_now().timestamp())}"
        try:
            shutil.copy2(path, backup_path)
        except OSError:
            backup_path = None
        return {
            **DEFAULT_STORE,
            "storage_path": path,
            "storage_error": {
                "message": "JITAI storage file is invalid JSON and was ignored.",
                "backup_path": backup_path,
                "error": str(exc),
            },
        }

    store = {
        "jitais": data.get("jitais") if isinstance(data.get("jitais"), list) else [],
        "runtime": data.get("runtime") if isinstance(data.get("runtime"), dict) else {},
        "events": data.get("events") if isinstance(data.get("events"), list) else [],
        "storage_path": path,
    }
    return store


def save_store(store, path=None):
    path = fallback_store_path(path or store.get("storage_path") or jitai_store_path())
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    serializable = {
        "jitais": store.get("jitais", []),
        "runtime": store.get("runtime", {}),
        "events": store.get("events", [])[-MAX_EVENTS:],
    }
    temporary_path = f"{path}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary_path, path)
    return {**serializable, "storage_path": path}


def normalize_jitai(raw):
    if not isinstance(raw, dict):
        raise JitaiError("JITAI must be an object.", error_type="invalid_jitai")
    now = isoformat(utc_now())
    jitai_id = str(raw.get("id") or uuid4()).strip()
    if not jitai_id:
        raise JitaiError("JITAI id is required.", error_type="invalid_jitai")

    triggers = raw.get("triggers") or []
    if not isinstance(triggers, list) or not triggers:
        raise JitaiError("JITAI requires at least one trigger.", error_type="invalid_jitai")

    return {
        "id": jitai_id,
        "name": str(raw.get("name") or "Untitled JITAI").strip()[:120],
        "enabled": bool(raw.get("enabled", True)),
        "triggers": [normalize_trigger(trigger) for trigger in triggers],
        "time_windows": normalize_time_windows(raw.get("time_windows") or raw.get("windows") or []),
        "cooldown_seconds": clamp_int(raw.get("cooldown_seconds", 0), 0, 30 * 24 * 60 * 60),
        "max_retries": clamp_int(raw.get("max_retries", 0), 0, 20),
        "retry_delay_seconds": clamp_int(raw.get("retry_delay_seconds", 300), 0, 24 * 60 * 60),
        "action": normalize_action(raw.get("action") or {}),
        "learning": raw.get("learning") if isinstance(raw.get("learning"), dict) else {},
        "created_at": raw.get("created_at") or now,
        "updated_at": now,
    }


def normalize_trigger(raw):
    if not isinstance(raw, dict):
        raise JitaiError("Trigger must be an object.", error_type="invalid_trigger")
    trigger_type = str(raw.get("type") or "sensor_threshold").strip()
    if trigger_type != "sensor_threshold":
        raise JitaiError("Only sensor_threshold triggers are supported.", error_type="unsupported_trigger")
    entity_id = str(raw.get("entity_id") or "").strip().lower()
    operator = str(raw.get("operator") or "").strip()
    if not entity_id:
        raise JitaiError("Trigger entity_id is required.", error_type="invalid_trigger")
    if operator not in SUPPORTED_OPERATORS:
        raise JitaiError(
            "Unsupported threshold operator.", error_type="invalid_trigger", details_safe={"operator": operator}
        )
    try:
        threshold = float(raw.get("threshold"))
    except (TypeError, ValueError) as exc:
        raise JitaiError("Trigger threshold must be numeric.", error_type="invalid_trigger") from exc
    return {
        "id": str(raw.get("id") or uuid4()),
        "type": trigger_type,
        "entity_id": entity_id,
        "operator": operator,
        "threshold": threshold,
    }


def normalize_time_windows(windows):
    if not windows:
        return []
    if not isinstance(windows, list):
        raise JitaiError("time_windows must be a list.", error_type="invalid_time_window")
    return [normalize_time_window(window) for window in windows]


def normalize_time_window(raw):
    if not isinstance(raw, dict):
        raise JitaiError("Time window must be an object.", error_type="invalid_time_window")
    start = parse_time(raw.get("start") or "00:00")
    end = parse_time(raw.get("end") or "23:59")
    days = raw.get("days") or []
    normalized_days = [str(day).strip().lower()[:3] for day in days] if isinstance(days, list) else []
    return {
        "days": [day for day in normalized_days if day],
        "start": start.strftime("%H:%M"),
        "end": end.strftime("%H:%M"),
    }


def normalize_action(raw):
    if not isinstance(raw, dict):
        raw = {}
    action_type = str(raw.get("type") or "log").strip()
    return {
        "type": action_type,
        "message": str(raw.get("message") or "").strip()[:500],
    }


def clamp_int(value, minimum, maximum):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = minimum
    return max(minimum, min(parsed, maximum))


def parse_time(value):
    try:
        hour, minute = str(value).split(":", 1)
        return time(hour=int(hour), minute=int(minute[:2]))
    except (TypeError, ValueError) as exc:
        raise JitaiError("Time windows must use HH:MM times.", error_type="invalid_time_window") from exc


def upsert_jitai(raw, path=None):
    store = load_store(path)
    jitai = normalize_jitai(raw)
    existing = [item for item in store["jitais"] if item.get("id") != jitai["id"]]
    prior = next((item for item in store["jitais"] if item.get("id") == jitai["id"]), {})
    jitai["created_at"] = prior.get("created_at") or jitai["created_at"]
    store["jitais"] = [*existing, jitai]
    return save_store(store, path), jitai


def list_jitais(path=None):
    store = load_store(path)
    return {"storage_path": store["storage_path"], "jitais": store["jitais"], "runtime": store["runtime"]}


def delete_jitai(jitai_id, path=None):
    store = load_store(path)
    before = len(store["jitais"])
    store["jitais"] = [item for item in store["jitais"] if item.get("id") != jitai_id]
    store["runtime"].pop(jitai_id, None)
    if len(store["jitais"]) == before:
        raise JitaiError("JITAI not found.", status=404, error_type="jitai_not_found")
    return save_store(store, path)


def find_jitai(store, jitai_id):
    for jitai in store.get("jitais", []):
        if jitai.get("id") == jitai_id:
            return jitai
    raise JitaiError("JITAI not found.", status=404, error_type="jitai_not_found")


def is_within_time_windows(jitai, now=None):
    windows = jitai.get("time_windows") or []
    if not windows:
        return True, None
    now = now or utc_now()
    day = now.strftime("%a").lower()[:3]
    now_time = now.time().replace(second=0, microsecond=0)
    for window in windows:
        days = window.get("days") or []
        if days and day not in days:
            continue
        start = parse_time(window.get("start"))
        end = parse_time(window.get("end"))
        if start <= end:
            in_window = start <= now_time <= end
        else:
            in_window = now_time >= start or now_time <= end
        if in_window:
            return True, window
    return False, None


def parse_sensor_value(value):
    text = str(value).strip().lower()
    if text in INVALID_SENSOR_STATES:
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def compare_value(value, operator, threshold):
    if operator == ">":
        return value > threshold
    if operator == ">=":
        return value >= threshold
    if operator == "<":
        return value < threshold
    if operator == "<=":
        return value <= threshold
    if operator == "==":
        return value == threshold
    if operator == "!=":
        return value != threshold
    return False


def evaluate_trigger(trigger, sensor_states):
    entity_id = trigger["entity_id"]
    raw_state = sensor_states.get(entity_id)
    numeric_value = parse_sensor_value(raw_state)
    if numeric_value is None:
        return {
            "trigger_id": trigger["id"],
            "entity_id": entity_id,
            "matched": False,
            "reason": "invalid_or_missing_state",
            "state": raw_state,
            "threshold": trigger["threshold"],
            "operator": trigger["operator"],
        }
    matched = compare_value(numeric_value, trigger["operator"], trigger["threshold"])
    return {
        "trigger_id": trigger["id"],
        "entity_id": entity_id,
        "matched": matched,
        "reason": "matched" if matched else "threshold_not_met",
        "value": numeric_value,
        "threshold": trigger["threshold"],
        "operator": trigger["operator"],
    }


def evaluate_jitai(jitai, sensor_states, runtime=None, now=None):
    now = now or utc_now()
    runtime = runtime or {}
    if not jitai.get("enabled", True):
        return decision("skipped", "disabled", now)

    in_window, matched_window = is_within_time_windows(jitai, now=now)
    if not in_window:
        return decision("skipped", "outside_time_window", now, time_window=None)

    cooldown_until = parse_datetime(runtime.get("cooldown_until"))
    if cooldown_until and now < cooldown_until:
        return decision("skipped", "cooldown_active", now, cooldown_until=isoformat(cooldown_until))

    next_retry_at = parse_datetime(runtime.get("next_retry_at"))
    if next_retry_at and now < next_retry_at:
        return decision("skipped", "retry_delay_active", now, next_retry_at=isoformat(next_retry_at))

    trigger_results = [evaluate_trigger(trigger, sensor_states) for trigger in jitai.get("triggers", [])]
    if not trigger_results or not all(result["matched"] for result in trigger_results):
        return decision(
            "skipped", "trigger_not_matched", now, trigger_results=trigger_results, time_window=matched_window
        )

    return decision("ready", "trigger_matched", now, trigger_results=trigger_results, time_window=matched_window)


def decision(status, reason, now, **extra):
    return {"status": status, "reason": reason, "evaluated_at": isoformat(now), **extra}


def execute_action(jitai):
    action = jitai.get("action") or {}
    action_type = action.get("type") or "log"
    if action_type in {"log", "noop"}:
        return {"ok": True, "type": action_type, "message": action.get("message")}
    if action_type == "test_fail":
        return {"ok": False, "type": action_type, "message": "Configured test failure."}
    return {"ok": False, "type": action_type, "message": "Unsupported JITAI action type."}


def log_event(store, jitai, event_type, reason, decision_result, action_result=None, now=None):
    now = now or utc_now()
    event = {
        "id": str(uuid4()),
        "jitai_id": jitai.get("id"),
        "jitai_name": jitai.get("name"),
        "event_type": event_type,
        "reason": reason,
        "occurred_at": isoformat(now),
        "decision": decision_result,
        "action": action_result,
        "learning": {
            "outcome": None,
            "reward": None,
            "context_features": decision_result.get("trigger_results") or [],
        },
    }
    store.setdefault("events", []).append(event)
    store["events"] = store["events"][-MAX_EVENTS:]
    return event


def evaluate_and_maybe_execute(jitai_id, sensor_states, execute=False, path=None, now=None):
    store = load_store(path)
    jitai = find_jitai(store, jitai_id)
    runtime = store.setdefault("runtime", {}).setdefault(jitai_id, {})
    now = now or utc_now()
    decision_result = evaluate_jitai(jitai, sensor_states, runtime=runtime, now=now)

    if decision_result["status"] != "ready" or not execute:
        event = log_event(
            store,
            jitai,
            "skipped" if decision_result["status"] == "skipped" else "evaluated",
            decision_result["reason"],
            decision_result,
            now=now,
        )
        save_store(store, path)
        return {"jitai": jitai, "decision": decision_result, "event": event, "runtime": runtime}

    action_result = execute_action(jitai)
    if action_result["ok"]:
        runtime["last_success_at"] = isoformat(now)
        runtime["retry_count"] = 0
        runtime.pop("next_retry_at", None)
        cooldown_seconds = int(jitai.get("cooldown_seconds") or 0)
        if cooldown_seconds:
            runtime["cooldown_until"] = isoformat(
                datetime.fromtimestamp(now.timestamp() + cooldown_seconds, timezone.utc)
            )
        event = log_event(store, jitai, "executed", "action_succeeded", decision_result, action_result, now=now)
    else:
        retry_count = int(runtime.get("retry_count") or 0) + 1
        runtime["retry_count"] = retry_count
        if retry_count <= int(jitai.get("max_retries") or 0):
            delay = int(jitai.get("retry_delay_seconds") or 0)
            runtime["next_retry_at"] = isoformat(datetime.fromtimestamp(now.timestamp() + delay, timezone.utc))
            reason = "action_failed_retry_scheduled"
        else:
            runtime.pop("next_retry_at", None)
            reason = "action_failed_retries_exhausted"
        event = log_event(store, jitai, "failed", reason, decision_result, action_result, now=now)

    save_store(store, path)
    return {"jitai": jitai, "decision": decision_result, "event": event, "runtime": runtime}


def events(path=None, jitai_id=None, limit=100):
    store = load_store(path)
    rows = store.get("events", [])
    if jitai_id:
        rows = [event for event in rows if event.get("jitai_id") == jitai_id]
    try:
        limit = max(1, min(int(limit), MAX_EVENTS))
    except (TypeError, ValueError):
        limit = 100
    return {"events": rows[-limit:], "storage_path": store["storage_path"]}


def supervisor_entity_state(entity_id, timeout=10):
    token = os.environ.get("SUPERVISOR_TOKEN")
    if not token:
        raise JitaiError("SUPERVISOR_TOKEN is not available.", status=400, error_type="home_assistant_token_missing")
    url = f"http://supervisor/core/api/states/{entity_id}"
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise JitaiError(
            "Home Assistant API returned an error.",
            status=502,
            error_type="home_assistant_api_error",
            details_safe={"status": exc.code, "entity_id": entity_id},
        ) from exc
    except urllib.error.URLError as exc:
        raise JitaiError(
            "Home Assistant API connection failed.",
            status=502,
            error_type="home_assistant_api_error",
            details_safe={"entity_id": entity_id, "error_class": exc.__class__.__name__},
        ) from exc
    return data.get("state")


def sensor_states_for_jitai(jitai, supplied_states=None, state_reader=None):
    supplied_states = supplied_states or {}
    normalized_supplied = {str(key).strip().lower(): value for key, value in supplied_states.items()}
    states = {}
    reader = state_reader or supervisor_entity_state
    for trigger in jitai.get("triggers", []):
        entity_id = trigger.get("entity_id")
        if entity_id in normalized_supplied:
            states[entity_id] = normalized_supplied[entity_id]
        else:
            states[entity_id] = reader(entity_id)
    return states
