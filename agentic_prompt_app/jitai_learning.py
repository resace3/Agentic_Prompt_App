import copy
import json
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from statistics import mean


VALID_RESPONSES = {"accept", "dismiss", "ignore"}
RESPONSE_REWARDS = {
    "accept": 1.0,
    "dismiss": -0.35,
    "ignore": 0.0,
}
DEFAULT_LEARNING_RATE = 0.35
MAX_EVENTS = 1000
MAX_SAFE_MESSAGE_CHARS = 240
SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"\bsk-ant-[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"\b[A-Za-z0-9_]*(?:token|secret|api[_-]?key)[A-Za-z0-9_]*\s*[:=]\s*\S+", re.IGNORECASE),
)


def jitai_store_path():
    return os.environ.get("JITAI_LEARNING_PATH") or "/data/jitai_learning.json"


def utc_timestamp():
    return datetime.now(timezone.utc).isoformat()


def default_store():
    return {
        "version": 1,
        "events": [],
        "timing_buckets": {},
        "message_variants": {},
        "interventions": {},
        "updated_at": None,
    }


def load_store(path=None):
    path = path or jitai_store_path()
    if not os.path.exists(path):
        return default_store()
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return default_store()
    if not isinstance(payload, dict):
        return default_store()
    store = default_store()
    store.update(payload)
    for key in ("events",):
        if not isinstance(store.get(key), list):
            store[key] = []
    for key in ("timing_buckets", "message_variants", "interventions"):
        if not isinstance(store.get(key), dict):
            store[key] = {}
    return store


def save_store(store, path=None):
    path = path or jitai_store_path()
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    payload = copy.deepcopy(store)
    payload["updated_at"] = utc_timestamp()
    temp_path = f"{path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temp_path, path)
    return payload


def redact_safe_text(value):
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    for pattern in SECRET_PATTERNS:
        text = pattern.sub("[redacted]", text)
    if len(text) > MAX_SAFE_MESSAGE_CHARS:
        text = f"{text[:MAX_SAFE_MESSAGE_CHARS].rstrip()}..."
    return text


def parse_timestamp(value):
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    text = str(value or "").strip()
    if not text:
        return datetime.now(timezone.utc)
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return datetime.now(timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def timing_bucket(value):
    parsed = parse_timestamp(value)
    minute = 0 if parsed.minute < 30 else 30
    return f"{parsed.hour:02d}:{minute:02d}"


def bounded_number(value, default=None, minimum=None, maximum=None):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if minimum is not None:
        number = max(minimum, number)
    if maximum is not None:
        number = min(maximum, number)
    return number


def context_key(context):
    if not isinstance(context, dict):
        return "general"
    parts = []
    goal = redact_safe_text(context.get("goal")).lower()
    if goal:
        parts.append(goal[:48])
    entity_ids = context.get("entity_ids")
    if isinstance(entity_ids, list) and entity_ids:
        safe_entities = sorted(redact_safe_text(item).lower() for item in entity_ids if item)[:4]
        parts.extend(safe_entities)
    return "|".join(parts) if parts else "general"


def metric_record():
    return {
        "shown": 0,
        "accepted": 0,
        "dismissed": 0,
        "ignored": 0,
        "score": 0.0,
        "last_reward": 0.0,
        "last_seen_at": None,
    }


def update_record(record, response, reward, occurred_at, learning_rate=DEFAULT_LEARNING_RATE):
    record["shown"] = int(record.get("shown", 0)) + 1
    if response == "accept":
        record["accepted"] = int(record.get("accepted", 0)) + 1
    elif response == "dismiss":
        record["dismissed"] = int(record.get("dismissed", 0)) + 1
    elif response == "ignore":
        record["ignored"] = int(record.get("ignored", 0)) + 1

    previous_score = float(record.get("score", 0.0) or 0.0)
    record["score"] = round(previous_score + learning_rate * (reward - previous_score), 4)
    record["last_reward"] = reward
    record["last_seen_at"] = occurred_at
    return record


def response_rates(record):
    shown = int(record.get("shown", 0) or 0)
    accepted = int(record.get("accepted", 0) or 0)
    dismissed = int(record.get("dismissed", 0) or 0)
    ignored = int(record.get("ignored", 0) or 0)
    denominator = shown or 1
    return {
        "shown": shown,
        "accepted": accepted,
        "dismissed": dismissed,
        "ignored": ignored,
        "accept_rate": round(accepted / denominator, 4),
        "dismiss_rate": round(dismissed / denominator, 4),
        "ignore_rate": round(ignored / denominator, 4),
        "score": round(float(record.get("score", 0.0) or 0.0), 4),
        "last_seen_at": record.get("last_seen_at"),
    }


def normalize_response_payload(payload):
    if not isinstance(payload, dict):
        raise ValueError("JSON object is required.")
    response = str(payload.get("response") or "").strip().lower()
    if response not in VALID_RESPONSES:
        raise ValueError("response must be accept, dismiss, or ignore.")

    occurred_at = parse_timestamp(payload.get("responded_at") or payload.get("suggested_at")).isoformat()
    bucket = timing_bucket(payload.get("suggested_at") or occurred_at)
    message_key = redact_safe_text(payload.get("message_key") or payload.get("intervention_type") or "default")
    context = payload.get("context") if isinstance(payload.get("context"), dict) else {}
    duration_seconds = bounded_number(payload.get("response_latency_seconds"), default=None, minimum=0, maximum=86400)

    return {
        "intervention_id": redact_safe_text(payload.get("intervention_id") or ""),
        "response": response,
        "reward": RESPONSE_REWARDS[response],
        "suggested_at": parse_timestamp(payload.get("suggested_at") or occurred_at).isoformat(),
        "responded_at": occurred_at,
        "timing_bucket": bucket,
        "message_key": message_key or "default",
        "message": redact_safe_text(payload.get("message") or payload.get("message_template") or ""),
        "context_key": context_key(context),
        "context": {
            "entity_ids": [
                redact_safe_text(item)
                for item in context.get("entity_ids", [])
                if isinstance(context.get("entity_ids"), list) and item
            ][:8],
            "goal": redact_safe_text(context.get("goal")),
            "time_window": redact_safe_text(context.get("time_window")),
            "units": redact_safe_text(context.get("units")),
        },
        "response_latency_seconds": duration_seconds,
    }


def record_response(payload, path=None):
    event = normalize_response_payload(payload)
    store = load_store(path)
    learning_rate = bounded_number(
        payload.get("learning_rate"), default=DEFAULT_LEARNING_RATE, minimum=0.01, maximum=1.0
    )

    timing = store["timing_buckets"].setdefault(event["timing_bucket"], metric_record())
    update_record(timing, event["response"], event["reward"], event["responded_at"], learning_rate=learning_rate)

    message = store["message_variants"].setdefault(event["message_key"], metric_record())
    update_record(message, event["response"], event["reward"], event["responded_at"], learning_rate=learning_rate)
    if event["message"]:
        message["message"] = event["message"]

    intervention_id = event.get("intervention_id")
    if intervention_id:
        store["interventions"][intervention_id] = {
            "response": event["response"],
            "reward": event["reward"],
            "timing_bucket": event["timing_bucket"],
            "message_key": event["message_key"],
            "responded_at": event["responded_at"],
            "context_key": event["context_key"],
        }

    store["events"].append(event)
    store["events"] = store["events"][-MAX_EVENTS:]
    saved = save_store(store, path)
    return {
        "event": event,
        "metrics": effectiveness_metrics(saved),
        "suggestions": suggest_improvements(saved, context=event["context"]),
    }


def ranked_records(records, minimum_shown=1):
    ranked = []
    for key, record in records.items():
        if int(record.get("shown", 0) or 0) < minimum_shown:
            continue
        item = response_rates(record)
        item["key"] = key
        if record.get("message"):
            item["message"] = redact_safe_text(record.get("message"))
        ranked.append(item)
    ranked.sort(key=lambda item: (-item["score"], -item["accept_rate"], item["dismiss_rate"], item["key"]))
    return ranked


def effectiveness_metrics(store=None, path=None):
    store = load_store(path) if store is None else store
    events = store.get("events", [])
    aggregate = metric_record()
    latencies = []
    for event in events:
        response = event.get("response")
        if response in VALID_RESPONSES:
            update_record(aggregate, response, RESPONSE_REWARDS[response], event.get("responded_at"))
        if isinstance(event.get("response_latency_seconds"), (int, float)):
            latencies.append(float(event["response_latency_seconds"]))

    metrics = {
        "total_events": len(events),
        "overall": response_rates(aggregate),
        "timing_buckets": ranked_records(store.get("timing_buckets", {})),
        "message_variants": ranked_records(store.get("message_variants", {})),
        "updated_at": store.get("updated_at"),
    }
    if latencies:
        metrics["response_latency_seconds"] = {
            "average": round(mean(latencies), 2),
            "min": round(min(latencies), 2),
            "max": round(max(latencies), 2),
        }
    return metrics


def suggest_improvements(store=None, path=None, context=None):
    store = load_store(path) if store is None else store
    timing_rank = ranked_records(store.get("timing_buckets", {}))
    message_rank = ranked_records(store.get("message_variants", {}))
    best_timing = timing_rank[0] if timing_rank else None
    best_message = message_rank[0] if message_rank else None

    avoid_times = [
        item["key"]
        for item in sorted(timing_rank, key=lambda row: (-row["dismiss_rate"], row["score"], row["key"]))
        if item["dismiss_rate"] >= 0.5 and item["shown"] >= 2
    ][:3]
    suggestions = {
        "recommended_timing_bucket": best_timing["key"] if best_timing else None,
        "recommended_message_key": best_message["key"] if best_message else None,
        "recommended_message": best_message.get("message") if best_message else None,
        "avoid_timing_buckets": avoid_times,
        "basis": {
            "total_events": len(store.get("events", [])),
            "minimum_events_for_confidence": 5,
            "confidence": "limited" if len(store.get("events", [])) < 5 else "learned",
            "context_key": context_key(context) if context else None,
        },
    }
    if best_timing:
        suggestions["timing_score"] = best_timing["score"]
    if best_message:
        suggestions["message_score"] = best_message["score"]
    return suggestions


def context_breakdown(path=None):
    store = load_store(path)
    grouped = defaultdict(metric_record)
    for event in store.get("events", []):
        response = event.get("response")
        if response in VALID_RESPONSES:
            update_record(
                grouped[event.get("context_key") or "general"],
                response,
                RESPONSE_REWARDS[response],
                event.get("responded_at"),
            )
    return ranked_records(grouped)
