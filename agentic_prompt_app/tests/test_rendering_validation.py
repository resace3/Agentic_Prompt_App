import base64
import io
import json
import math
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from xml.etree import ElementTree

import matplotlib.image as mpimg
import pytest

import app as app_module


SLEEP_VALUES = {
    "sensor.nick_r_sleep_minutes_asleep": [390, 405, 420, 435, 450, 465, 480, 495, 470, 460, 455, 445, 430, 410],
    "sensor.nick_r_sleep_time_in_bed": [430, 445, 460, 475, 490, 505, 520, 535, 510, 500, 495, 485, 470, 450],
    "sensor.nick_r_sleep_minutes_awake": [40, 40, 40, 40, 40, 40, 40, 40, 38, 40, 40, 40, 40, 40],
    "sensor.nick_r_sleep_efficiency": [91, 91, 91, 92, 92, 92, 92, 93, 92, 92, 92, 92, 91, 91],
    "sensor.nick_r_awakenings_count": [3, 3, 2, 2, 2, 1, 1, 1, 2, 3, 2, 3, 4, 2],
    "sensor.nick_r_sleep_start_time": [
        "23:15",
        "23:10",
        "23:05",
        "23:00",
        "22:55",
        "22:50",
        "22:45",
        "22:40",
        "22:35",
        "22:30",
        "22:25",
        "22:20",
        "22:15",
        "22:10",
    ],
}


def configure_isolated_app(monkeypatch, tmp_path, db_path):
    monkeypatch.setenv("HA_RECORDER_DB_PATH", str(db_path))
    monkeypatch.setenv("SENSOR_MAP_PATH", str(tmp_path / "sensor_map.json"))
    monkeypatch.setenv("CHAT_STORE_PATH", str(tmp_path / "chat_history.json"))
    monkeypatch.setenv("SECRETS_YAML", str(tmp_path / "missing-secrets.yaml"))
    monkeypatch.delenv("SUPERVISOR_TOKEN", raising=False)
    monkeypatch.delenv("HOME_ASSISTANT_TOKEN", raising=False)
    monkeypatch.delenv("HOME_ASSISTANT_API_URL", raising=False)
    app_module.home_assistant_history_cache.clear()
    app_module.home_assistant_config_cache["data"] = None
    app_module.home_assistant_config_cache["expires"] = 0
    Path(os.environ["SENSOR_MAP_PATH"]).write_text(
        json.dumps(
            {
                "sensors": [
                    {"sensor": "sensor.nick_r_steps", "description": "Daily steps"},
                    {"sensor": "binary_sensor.pantry_door_window", "description": "Pantry door open/closed state"},
                    {"sensor": "sensor.nutribullet_plug_current", "description": "NutriBullet current in amps"},
                ]
            }
        ),
        encoding="utf-8",
    )


def create_fake_recorder_db(path):
    now = datetime.now(timezone.utc).replace(microsecond=0)
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE states_meta (metadata_id INTEGER PRIMARY KEY, entity_id VARCHAR(255))")
        connection.execute(
            """
            CREATE TABLE states (
                state_id INTEGER PRIMARY KEY,
                state VARCHAR(255),
                last_updated_ts FLOAT,
                metadata_id INTEGER
            )
            """
        )
        state_id = 1
        entities = {
            **SLEEP_VALUES,
            "sensor.nick_r_steps": [3000, 3500, 4200, 4900, 5600, 6300, 7000, 7700, 6900, 6100, 5300, 4500, 3700, 3200],
            "binary_sensor.pantry_door_window": [
                "off",
                "off",
                "on",
                "off",
                "on",
                "off",
                "off",
                "on",
                "off",
                "on",
                "off",
                "off",
                "on",
                "off",
            ],
            "sensor.nutribullet_plug_current": [0, 0, 1.2, 0, 1.5, 0, 0.8, 0, 0, 1.1, 0, 0, 1.4, 0],
        }
        for metadata_id, (entity_id, values) in enumerate(entities.items(), start=1):
            connection.execute(
                "INSERT INTO states_meta (metadata_id, entity_id) VALUES (?, ?)",
                (metadata_id, entity_id),
            )
            for index, value in enumerate(values):
                updated = now - timedelta(days=len(values) - index - 1, hours=6)
                connection.execute(
                    "INSERT INTO states (state_id, state, last_updated_ts, metadata_id) VALUES (?, ?, ?, ?)",
                    (state_id, str(value), updated.timestamp(), metadata_id),
                )
                state_id += 1


def read_only_connection(path):
    return sqlite3.connect(f"{Path(path).absolute().as_uri()}?mode=ro", uri=True)


def direct_completed_sleep_truth(db_path, days):
    start, end = app_module.history_window(days)
    zone_name = app_module.home_assistant_time_zone()
    grouped = {}
    with read_only_connection(db_path) as connection:
        rows_by_metric = {}
        for metric, entity_id in app_module.SLEEP_METRIC_ENTITIES.items():
            rows = connection.execute(
                """
                SELECT states.state, states.last_updated_ts
                FROM states
                JOIN states_meta ON states.metadata_id = states_meta.metadata_id
                WHERE states_meta.entity_id = ?
                  AND states.last_updated_ts >= ?
                  AND states.last_updated_ts <= ?
                ORDER BY states.last_updated_ts
                """,
                (entity_id, start.timestamp(), end.timestamp()),
            ).fetchall()
            rows_by_metric[metric] = rows

    for metric, rows in rows_by_metric.items():
        for state, updated_ts in rows:
            updated_at = app_module.parse_recorder_datetime(updated_ts)
            if updated_at is None:
                continue
            local_dt = updated_at.astimezone(app_module.ZoneInfo(zone_name))
            day = local_dt.date().isoformat()
            if metric == "start_time":
                value = str(state).strip()
                if not value:
                    continue
            else:
                value = app_module.parse_number(state)
                if value is None:
                    continue
                if metric in {"minutes_asleep", "time_in_bed", "minutes_awake", "efficiency"} and value == 0:
                    continue
            current = grouped.setdefault(day, {}).get(metric)
            if current is None or local_dt.isoformat() >= current["updated_local"]:
                grouped[day][metric] = {"value": value, "updated_local": local_dt.isoformat()}

    records = []
    required = ("minutes_asleep", "time_in_bed", "minutes_awake", "efficiency")
    for day, values in grouped.items():
        if all(metric in values for metric in required):
            records.append(
                {
                    "date": day,
                    "time_in_bed": int(round(values["time_in_bed"]["value"])),
                    "updated_local": max(values[metric]["updated_local"] for metric in required),
                }
            )
    records = sorted(records, key=lambda item: item["date"])[-days:]
    values = [record["time_in_bed"] for record in records]
    rolling = [round(sum(values[index - 6 : index + 1]) / 7, 3) for index in range(6, len(values))]
    return {
        "row_count": len(values),
        "date_range": f"{records[0]['date']} to {records[-1]['date']}" if records else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "mean": round(sum(values) / len(values), 2) if values else None,
        "missingness": 1 - (len(values) / days),
        "rolling_7": rolling,
        "values": values,
    }


def decode_data_url(data_url):
    header, encoded = data_url.split(",", 1)
    return header, base64.b64decode(encoded)


def assert_png_is_readable(data_url, *, min_width=900, min_height=450):
    header, payload = decode_data_url(data_url)
    assert header == "data:image/png;base64"
    assert payload.startswith(b"\x89PNG\r\n\x1a\n")
    width = int.from_bytes(payload[16:20], "big")
    height = int.from_bytes(payload[20:24], "big")
    assert width >= min_width, {"width": width, "height": height}
    assert height >= min_height, {"width": width, "height": height}
    image = mpimg.imread(io.BytesIO(payload), format="png")
    assert image.size > 0
    assert float(image.std()) > 0.001, "plot image appears blank"


def assert_svg_is_bounded(data_url):
    header, payload = decode_data_url(data_url)
    assert header == "data:image/svg+xml;base64"
    svg = payload.decode("utf-8")
    root = ElementTree.fromstring(svg)
    width = int(root.attrib["width"])
    height = int(root.attrib["height"])
    assert width <= 1000
    assert height <= 520
    assert root.attrib.get("viewBox") == f"0 0 {width} {height}"
    assert root.attrib.get("role") == "img"
    assert "<script" not in svg.lower()


@pytest.fixture()
def fake_recorder(monkeypatch, tmp_path):
    db_path = tmp_path / "home-assistant_v2.db"
    create_fake_recorder_db(db_path)
    configure_isolated_app(monkeypatch, tmp_path, db_path)
    return db_path


def test_sleep_time_in_bed_plot_matches_readonly_db_truth(fake_recorder):
    truth = direct_completed_sleep_truth(fake_recorder, days=14)
    plot = app_module.attach_python_plot(
        app_module.query_sensor_points("sensor.nick_r_sleep_time_in_bed", days=14, plot_type="bar")
    )
    plotted_values = [point["value"] for point in plot["points"]]

    assert plot["available"]
    assert plot["source"] == "completed_sleep"
    assert plot["query"]["source"] == "home_assistant_recorder_db"
    assert plot["plot_spec"]["plot_type"] == "bar"
    assert plot["plot_spec"]["entity_ids"] == ["sensor.nick_r_sleep_time_in_bed"]
    assert plotted_values == truth["values"]
    assert plot["samples"] == truth["row_count"]
    assert plot["min"] == truth["min"]
    assert plot["max"] == truth["max"]
    assert math.isclose(plot["average"], truth["mean"], abs_tol=0.01)
    assert truth["missingness"] == 0
    assert len(truth["rolling_7"]) == 8
    assert_png_is_readable(plot["python_image"]["data_url"])


def test_supported_plot_types_render_nonblank_images(fake_recorder):
    cases = [
        ("line", app_module.query_sensor_points("sensor.nick_r_sleep_time_in_bed", days=14, plot_type="line")),
        ("bar", app_module.query_sensor_points("sensor.nick_r_sleep_time_in_bed", days=14, plot_type="bar")),
        (
            "histogram",
            app_module.query_sensor_points("sensor.nick_r_sleep_time_in_bed", days=14, plot_type="histogram"),
        ),
        ("scatter", app_module.build_multi_sleep_plot(["awakenings", "minutes_asleep"], days=14, plot_type="scatter")),
        ("heatmap", app_module.build_sleep_correlation_heatmap(days=14)),
    ]
    for expected_type, plot in cases:
        rendered = app_module.attach_python_plot(plot)
        assert rendered["available"], rendered
        assert rendered["plot_spec"]["plot_type"] == expected_type
        assert rendered["python_image"]["plot_type"] == expected_type
        assert rendered["python_image"]["title"]
        assert rendered["python_image"]["x_axis_label"] is not None
        assert rendered["python_image"]["y_axis_label"]
        assert_png_is_readable(rendered["python_image"]["data_url"])


def test_n_of_1_visual_artifacts_are_bounded_and_deterministic(fake_recorder):
    summary = app_module.summarize_home_assistant_prompt_context(
        (
            "Make plots describing the N-of-1 analysis, show causal DAGs, and show the LaTeX equations "
            "for my time asleep sleep with non-sleep variables in the Sensor Map using "
            "https://arxiv.org/abs/2407.17666"
        ),
        days=14,
    )
    visuals = app_module.build_analysis_visuals("make plots describing the analysis and show causal dags", summary)
    artifacts = visuals["artifacts"]

    assert visuals["available"]
    assert visuals["computed_results"]["outcome_sensor"] == "sensor.nick_r_sleep_minutes_asleep"
    assert {artifact["type"] for artifact in artifacts} >= {"plot", "dag", "latex"}
    for artifact in artifacts:
        if artifact["type"] == "plot":
            assert_png_is_readable(artifact["data_url"])
        elif artifact["type"] == "dag":
            assert_svg_is_bounded(artifact["data_url"])
        elif artifact["type"] == "latex":
            assert "\\frac" in artifact["latex"] or "\\beta" in artifact["latex"]
            assert "<script" not in artifact["latex"].lower()


def test_actual_recorder_db_consistency_when_available(monkeypatch, tmp_path):
    real_db = Path("/config/home-assistant_v2.db")
    if not real_db.exists():
        pytest.skip("/config/home-assistant_v2.db is not available in this environment")

    configure_isolated_app(monkeypatch, tmp_path, real_db)
    truth = direct_completed_sleep_truth(real_db, days=30)
    assert truth["row_count"] > 0, "sensor.nick_r_sleep_time_in_bed has no completed rows in the last 30 days"

    plot = app_module.query_sensor_points("sensor.nick_r_sleep_time_in_bed", days=30, plot_type="line")
    assert plot["available"], plot
    assert [point["value"] for point in plot["points"]] == truth["values"]
    assert plot["samples"] == truth["row_count"]
    assert plot["min"] == truth["min"]
    assert plot["max"] == truth["max"]
    assert math.isclose(plot["average"], truth["mean"], abs_tol=0.01)
