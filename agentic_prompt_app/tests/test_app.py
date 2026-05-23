import base64
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import app as app_module


ROOT = Path(__file__).resolve().parents[1]


class PromptAppTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.previous_sensor_map_path = os.environ.get("SENSOR_MAP_PATH")
        self.previous_chat_store_path = os.environ.get("CHAT_STORE_PATH")
        os.environ["SENSOR_MAP_PATH"] = os.path.join(self.temp_dir.name, "sensor_map.json")
        os.environ["CHAT_STORE_PATH"] = os.path.join(self.temp_dir.name, "chat_history.json")
        app_module.conversations.clear()
        app_module.chat_store_loaded_path = None
        self.client = app_module.app.test_client()

    def tearDown(self):
        if self.previous_sensor_map_path is None:
            os.environ.pop("SENSOR_MAP_PATH", None)
        else:
            os.environ["SENSOR_MAP_PATH"] = self.previous_sensor_map_path
        if self.previous_chat_store_path is None:
            os.environ.pop("CHAT_STORE_PATH", None)
        else:
            os.environ["CHAT_STORE_PATH"] = self.previous_chat_store_path
        app_module.chat_store_loaded_path = None
        self.temp_dir.cleanup()

    def test_sensor_map_persists_twenty_rows(self):
        rows = [{"sensor": f"sensor.test_{index}", "description": f"Description {index}"} for index in range(20)]

        save_response = self.client.put("/api/sensor-map", json={"sensors": rows})
        self.assertEqual(save_response.status_code, 200)
        self.assertEqual(len(save_response.get_json()["sensors"]), 20)

        load_response = self.client.get("/api/sensor-map")
        loaded_rows = load_response.get_json()["sensors"]
        self.assertEqual(len(loaded_rows), 20)
        self.assertEqual(loaded_rows[0]["sensor"], "sensor.test_0")
        self.assertEqual(loaded_rows[-1]["sensor"], "sensor.test_19")

    def test_home_assistant_api_uses_supervisor_token(self):
        original_token = os.environ.get("SUPERVISOR_TOKEN")
        original_home_assistant_token = os.environ.get("HOME_ASSISTANT_TOKEN")
        os.environ["SUPERVISOR_TOKEN"] = "supervisor-test-token"
        os.environ.pop("HOME_ASSISTANT_TOKEN", None)
        try:
            self.assertEqual(app_module.home_assistant_token(), "supervisor-test-token")
            self.assertIsNone(app_module.home_assistant_db_path())
        finally:
            if original_token is None:
                os.environ.pop("SUPERVISOR_TOKEN", None)
            else:
                os.environ["SUPERVISOR_TOKEN"] = original_token
            if original_home_assistant_token is None:
                os.environ.pop("HOME_ASSISTANT_TOKEN", None)
            else:
                os.environ["HOME_ASSISTANT_TOKEN"] = original_home_assistant_token

    def test_models_endpoint_exposes_pricing(self):
        response = self.client.get("/api/models")
        data = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(data["default_provider"], app_module.DEFAULT_PROVIDER)
        self.assertEqual(data["default_model"], app_module.DEFAULT_MODEL)
        self.assertIn("pricing", data["pricing_source"])
        self.assertIn("openai", {provider["id"] for provider in data["providers"]})
        self.assertIn("anthropic", {provider["id"] for provider in data["providers"]})
        gpt_41_nano = next(model for model in data["models"] if model["id"] == "gpt-4.1-nano")
        claude_haiku = next(model for model in data["models"] if model["id"] == app_module.DEFAULT_CLAUDE_MODEL)
        self.assertEqual(data["default_model"], "gpt-4.1-nano")
        self.assertEqual(gpt_41_nano["provider"], "openai")
        self.assertEqual(gpt_41_nano["input_per_1m"], 0.1)
        self.assertEqual(gpt_41_nano["output_per_1m"], 0.4)
        self.assertEqual(claude_haiku["provider"], "anthropic")
        self.assertEqual(claude_haiku["input_per_1m"], 1.0)
        self.assertEqual(claude_haiku["output_per_1m"], 5.0)

    def test_index_has_model_selector_and_price_text(self):
        response = self.client.get("/")
        html = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn('href="static/styles.css"', html)
        self.assertIn('id="providerSelect"', html)
        self.assertIn('id="modelSelect"', html)
        self.assertIn('id="modelPrice"', html)
        self.assertNotIn('id="balanceBadge"', html)
        self.assertNotIn("/api/openai-balance", html)
        self.assertIn("OpenAI", html)
        self.assertIn("Claude", html)
        self.assertIn("gpt-4.1-nano", html)
        self.assertIn(app_module.DEFAULT_CLAUDE_MODEL, html)
        self.assertIn("promptFlowSelectedModel.v5", html)
        self.assertIn('className = "pinned-marker"', html)

    def test_index_uses_ingress_prefix_for_static_assets(self):
        response = self.client.get("/", headers={"X-Ingress-Path": "/api/hassio_ingress/test-token"})
        html = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn('href="/api/hassio_ingress/test-token/static/styles.css"', html)
        self.assertIn('const REQUEST_SCRIPT_ROOT = "/api/hassio_ingress/test-token";', html)

    def test_message_uses_selected_model_and_stores_response_model_stamp(self):
        class FakeResponses:
            def create(self, **kwargs):
                assert kwargs["model"] == "gpt-5.4-mini"
                return SimpleNamespace(
                    id="resp_test",
                    output_text="Selected model response.",
                )

        class FakeClient:
            responses = FakeResponses()

        original_get_client = app_module.get_client
        app_module.get_client = lambda: FakeClient()
        try:
            response = self.client.post(
                "/api/message",
                json={"message": "hello", "provider": "openai", "model": "gpt-5.4-mini"},
            )
        finally:
            app_module.get_client = original_get_client

        data = response.get_json()
        assistant = data["messages"][-1]
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data["provider"], "openai")
        self.assertEqual(data["model"], "gpt-5.4-mini")
        self.assertEqual(data["model_info"]["output_per_1m"], 4.5)
        self.assertEqual(assistant["provider"], "openai")
        self.assertEqual(assistant["provider_label"], "OpenAI")
        self.assertEqual(assistant["model"], "gpt-5.4-mini")
        self.assertEqual(assistant["model_label"], "GPT 5.4 Mini")
        self.assertEqual(assistant["model_pricing"]["input_per_1m"], 0.75)

    def test_chat_is_saved_and_loaded_from_store(self):
        class FakeResponses:
            def create(self, **kwargs):
                return SimpleNamespace(
                    id="resp_test",
                    output_text="Saved response.",
                )

        class FakeClient:
            responses = FakeResponses()

        original_get_client = app_module.get_client
        app_module.get_client = lambda: FakeClient()
        try:
            response = self.client.post(
                "/api/message",
                json={"message": "Name this saved conversation"},
            )
        finally:
            app_module.get_client = original_get_client

        data = response.get_json()
        chat_id = data["active_chat_id"]
        self.assertEqual(response.status_code, 200)
        self.assertTrue(os.path.exists(os.environ["CHAT_STORE_PATH"]))
        self.assertEqual(data["chat"]["title"], "Name this saved conversation")

        app_module.conversations.clear()
        app_module.chat_store_loaded_path = None
        load_response = self.client.get("/api/chats")
        load_data = load_response.get_json()

        self.assertEqual(load_response.status_code, 200)
        self.assertEqual(load_data["active_chat_id"], chat_id)
        self.assertEqual(load_data["chats"][0]["title"], "Name this saved conversation")
        self.assertEqual(load_data["messages"][-1]["content"], "Saved response.")

    def test_chats_can_be_pinned_and_deleted(self):
        first = self.client.post("/api/chats").get_json()["active_chat_id"]
        second = self.client.post("/api/chats").get_json()["active_chat_id"]

        pin_response = self.client.patch(f"/api/chats/{first}", json={"pinned": True})
        pin_data = pin_response.get_json()

        self.assertEqual(pin_response.status_code, 200)
        self.assertEqual(pin_data["chats"][0]["id"], first)
        self.assertTrue(pin_data["chats"][0]["pinned"])

        delete_response = self.client.delete(f"/api/chats/{first}")
        delete_data = delete_response.get_json()

        self.assertEqual(delete_response.status_code, 200)
        self.assertEqual(delete_data["active_chat_id"], second)
        self.assertNotIn(first, {chat["id"] for chat in delete_data["chats"]})

    def test_reading_chats_does_not_create_blank_saved_chat(self):
        first_client = app_module.app.test_client()
        response = first_client.get("/api/chats")
        data = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertIsNone(data["active_chat_id"])
        self.assertEqual(data["messages"], [])
        self.assertEqual(data["chats"], [])
        self.assertFalse(os.path.exists(os.environ["CHAT_STORE_PATH"]))

        second_client = app_module.app.test_client()
        response = second_client.get("/api/messages")
        data = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertIsNone(data["active_chat_id"])
        self.assertEqual(data["messages"], [])
        self.assertEqual(data["chats"], [])
        self.assertFalse(os.path.exists(os.environ["CHAT_STORE_PATH"]))

    def test_pinned_chats_stay_at_top_after_reload(self):
        first = self.client.post("/api/chats").get_json()["active_chat_id"]
        second = self.client.post("/api/chats").get_json()["active_chat_id"]
        third = self.client.post("/api/chats").get_json()["active_chat_id"]

        self.client.patch(f"/api/chats/{second}", json={"pinned": True})
        self.client.patch(f"/api/chats/{first}", json={"pinned": True})

        before_reload = self.client.get("/api/chats").get_json()["chats"]
        self.assertEqual([chat["id"] for chat in before_reload[:2]], [first, second])
        self.assertTrue(all(chat["pinned"] for chat in before_reload[:2]))

        app_module.conversations.clear()
        app_module.chat_store_loaded_path = None
        after_reload = self.client.get("/api/chats").get_json()["chats"]

        self.assertEqual([chat["id"] for chat in after_reload[:2]], [first, second])
        self.assertNotIn(third, [chat["id"] for chat in after_reload[:2]])
        self.assertTrue(all(chat["pinned"] for chat in after_reload[:2]))

    def test_message_uses_selected_claude_model_and_stores_provider_stamp(self):
        calls = {}

        def fake_create_anthropic_message(model_id, messages):
            calls["model_id"] = model_id
            calls["messages"] = messages
            return {
                "id": "msg_test",
                "content": [{"type": "text", "text": "Claude response."}],
            }

        original_create = app_module.create_anthropic_message
        app_module.create_anthropic_message = fake_create_anthropic_message
        try:
            response = self.client.post(
                "/api/message",
                json={
                    "message": "hello claude",
                    "provider": "anthropic",
                    "model": app_module.DEFAULT_CLAUDE_MODEL,
                },
            )
        finally:
            app_module.create_anthropic_message = original_create

        data = response.get_json()
        assistant = data["messages"][-1]
        self.assertEqual(response.status_code, 200)
        self.assertEqual(calls["model_id"], app_module.DEFAULT_CLAUDE_MODEL)
        self.assertEqual(calls["messages"][-1]["role"], "user")
        self.assertIn("hello claude", calls["messages"][-1]["content"])
        self.assertEqual(data["provider"], "anthropic")
        self.assertEqual(data["model"], app_module.DEFAULT_CLAUDE_MODEL)
        self.assertEqual(data["model_info"]["input_per_1m"], 1.0)
        self.assertEqual(assistant["provider"], "anthropic")
        self.assertEqual(assistant["provider_label"], "Claude")
        self.assertEqual(assistant["model"], app_module.DEFAULT_CLAUDE_MODEL)
        self.assertEqual(assistant["content"], "Claude response.")

    def test_message_rejects_unknown_provider(self):
        response = self.client.post(
            "/api/message",
            json={"message": "hello", "provider": "not-a-provider"},
        )
        data = response.get_json()

        self.assertEqual(response.status_code, 400)
        self.assertIn("Unknown provider", data["error"])
        self.assertTrue(data["providers"])
        self.assertTrue(data["models"])

    def test_message_rejects_unknown_model(self):
        response = self.client.post(
            "/api/message",
            json={"message": "hello", "model": "not-a-model"},
        )
        data = response.get_json()

        self.assertEqual(response.status_code, 400)
        self.assertIn("Unknown OpenAI model", data["error"])
        self.assertTrue(data["models"])

    def test_completed_sleep_summary_matches_command_line_metrics(self):
        if not app_module.home_assistant_available():
            self.skipTest("Home Assistant API is not available.")

        summary = app_module.summarize_completed_sleep(days=7)

        self.assertTrue(summary["available"])
        self.assertGreater(summary["days_returned"], 0)
        self.assertLessEqual(summary["days_returned"], 7)
        self.assertEqual(summary["date_range"], app_module.date_range_label(summary["daily"]))
        self.assertEqual(summary["latest_sleep_local"], summary["daily"][-1]["updated_local"])
        self.assertNotIn(0, [record["minutes_asleep"] for record in summary["daily"]])
        self.assertNotIn(0, [record["time_in_bed"] for record in summary["daily"]])
        expected_sleep_average = sum(record["minutes_asleep"] for record in summary["daily"]) / summary["days_returned"]
        expected_bed_average = sum(record["time_in_bed"] for record in summary["daily"]) / summary["days_returned"]
        expected_awake_average = sum(record["minutes_awake"] for record in summary["daily"]) / summary["days_returned"]
        expected_efficiency_average = (
            sum(record["efficiency"] for record in summary["daily"]) / summary["days_returned"]
        )
        self.assertEqual(summary["averages"]["sleep_label"], app_module.minutes_to_hours_label(expected_sleep_average))
        self.assertEqual(
            summary["averages"]["time_in_bed_label"],
            app_module.minutes_to_hours_label(expected_bed_average),
        )
        self.assertEqual(summary["averages"]["minutes_awake"], round(expected_awake_average, 1))
        self.assertEqual(summary["averages"]["efficiency"], round(expected_efficiency_average, 1))

    def test_sleep_prompt_context_uses_completed_nights(self):
        if not app_module.home_assistant_available():
            self.skipTest("Home Assistant API is not available.")

        expected = app_module.summarize_completed_sleep(days=7)

        class FakeResponses:
            def create(self, **kwargs):
                prompt = kwargs["input"]
                assert "completed_sleep:" in prompt
                assert f"date_range: {expected['date_range']}" in prompt
                assert f"sleep_label: {expected['averages']['sleep_label']}" in prompt
                assert "240.74" not in prompt
                return SimpleNamespace(
                    id="resp_test",
                    output_text=(
                        f"Average sleep was {expected['averages']['sleep_label']} over {expected['date_range']}."
                    ),
                )

        class FakeClient:
            responses = FakeResponses()

        original_get_client = app_module.get_client
        app_module.get_client = lambda: FakeClient()
        try:
            response = self.client.post(
                "/api/message",
                json={"message": "How has my sleep been the past week?"},
            )
        finally:
            app_module.get_client = original_get_client

        data = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertIn(expected["averages"]["sleep_label"], data["assistant"])

    def test_sleep_prompt_context_can_use_more_than_one_week(self):
        if not app_module.home_assistant_available():
            self.skipTest("Home Assistant API is not available.")

        class FakeResponses:
            def create(self, **kwargs):
                prompt = kwargs["input"]
                assert "days_requested: 90" in prompt
                assert "days_returned:" in prompt
                assert "date_range: May 15-21, 2026" not in prompt
                return SimpleNamespace(
                    id="resp_test",
                    output_text="I used a larger read-only sleep history.",
                )

        class FakeClient:
            responses = FakeResponses()

        original_get_client = app_module.get_client
        app_module.get_client = lambda: FakeClient()
        try:
            response = self.client.post(
                "/api/message",
                json={"message": "What predicts good sleep over the last 90 days?"},
            )
        finally:
            app_module.get_client = original_get_client

        self.assertEqual(response.status_code, 200)

    def test_prompt_context_queries_mapped_non_sleep_sensor_history(self):
        if not app_module.home_assistant_available():
            self.skipTest("Home Assistant API is not available.")

        self.client.put(
            "/api/sensor-map",
            json={
                "sensors": [
                    {
                        "sensor": "binary_sensor.pantry_door_window",
                        "description": "This is my pantry door that I get food out of.",
                    }
                ]
            },
        )

        class FakeResponses:
            def create(self, **kwargs):
                prompt = kwargs["input"]
                assert "mapped_sensor_history:" in prompt
                assert "binary_sensor.pantry_door_window" in prompt
                assert "state_counts:" in prompt
                assert "daily_active_events:" in prompt
                assert "recent_rows:" in prompt
                return SimpleNamespace(
                    id="resp_test",
                    output_text="The pantry door history was queried from Home Assistant.",
                )

        class FakeClient:
            responses = FakeResponses()

        original_get_client = app_module.get_client
        app_module.get_client = lambda: FakeClient()
        try:
            response = self.client.post(
                "/api/message",
                json={
                    "message": (
                        "look at binary_sensor.pantry_door_window in the home assistant db and give me info on it"
                    )
                },
            )
        finally:
            app_module.get_client = original_get_client

        data = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data["sensor_data"]["sensor"], "binary_sensor.pantry_door_window")
        self.assertGreater(len(data["sensor_data"]["rows"]), 0)

    def test_relation_prompt_includes_mapped_sensor_sleep_alignment(self):
        if not app_module.home_assistant_available():
            self.skipTest("Home Assistant API is not available.")

        self.client.put(
            "/api/sensor-map",
            json={
                "sensors": [
                    {
                        "sensor": "binary_sensor.pantry_door_window",
                        "description": "This is my pantry door that I get food out of.",
                    },
                    {
                        "sensor": "sensor.nick_r_sleep_minutes_asleep",
                        "description": "Total minutes asleep.",
                    },
                ]
            },
        )

        class FakeResponses:
            def create(self, **kwargs):
                prompt = kwargs["input"]
                assert "completed_sleep:" in prompt
                assert "mapped_sensor_history:" in prompt
                assert "binary_sensor.pantry_door_window" in prompt
                assert "sleep_alignment:" in prompt
                assert "active_events_4h_before_sleep" in prompt
                assert "active_events_during_time_in_bed" in prompt
                return SimpleNamespace(
                    id="resp_test",
                    output_text="I compared pantry door activity with completed sleep nights.",
                )

        class FakeClient:
            responses = FakeResponses()

        original_get_client = app_module.get_client
        app_module.get_client = lambda: FakeClient()
        try:
            response = self.client.post(
                "/api/message",
                json={"message": "Is there any relation to when I open the pantry door and my sleep?"},
            )
        finally:
            app_module.get_client = original_get_client

        data = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(data["home_assistant_context"]["mapped_sensor_history"])
        pantry = data["home_assistant_context"]["mapped_sensor_history"][0]
        self.assertEqual(pantry["sensor"], "binary_sensor.pantry_door_window")
        self.assertIn("sleep_alignment", pantry)

    def test_requested_days_from_prompt_text(self):
        self.assertEqual(app_module.requested_days_from_text("How was last week?"), 7)
        self.assertEqual(app_module.requested_days_from_text("show me 60 days of sleep"), 60)
        self.assertEqual(app_module.requested_days_from_text("last 3 months sleep patterns"), 90)
        self.assertEqual(app_module.requested_days_from_text("what predicts good sleep?"), 90)
        self.assertEqual(app_module.requested_days_from_text("analyze all sleep history"), 365)

    def test_prompt_plot_request_returns_plot(self):
        if not app_module.home_assistant_available():
            self.skipTest("Home Assistant API is not available.")

        self.client.put(
            "/api/sensor-map",
            json={
                "sensors": [
                    {
                        "sensor": "sensor.nick_r_sleep_minutes_asleep",
                        "description": "Total minutes asleep",
                    },
                    {
                        "sensor": "sensor.nick_r_sleep_efficiency",
                        "description": "Sleep efficiency percentage",
                    },
                ]
            },
        )

        class FakeResponses:
            def create(self, **kwargs):
                assert kwargs["model"] == app_module.DEFAULT_MODEL
                assert "plot" in kwargs["input"].lower()
                return SimpleNamespace(
                    id="resp_test",
                    output_text="I made the requested sensor plot.",
                )

        class FakeClient:
            responses = FakeResponses()

        original_get_client = app_module.get_client
        app_module.get_client = lambda: FakeClient()
        try:
            response = self.client.post(
                "/api/message",
                json={"message": "make a plot of sensor.nick_r_sleep_minutes_asleep for 30 days"},
            )
        finally:
            app_module.get_client = original_get_client

        data = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(data["plot"]["available"])
        self.assertTrue(data["plot"]["cleaned"])
        self.assertEqual(data["plot"]["source"], "completed_sleep")
        self.assertEqual(data["plot"]["sensor"], "sensor.nick_r_sleep_minutes_asleep")
        self.assertGreater(data["plot"]["samples"], 0)
        self.assertNotIn(0, [point["value"] for point in data["plot"]["points"]])
        self.assertEqual(data["plot"]["python_image"]["renderer"], "matplotlib")
        self.assertEqual(data["plot"]["python_image"]["x_axis_label"], "Date")
        self.assertTrue(data["plot"]["python_image"]["data_url"].startswith("data:image/png;base64,"))

    def test_sleep_week_prompt_plot_uses_seven_cleaned_completed_nights(self):
        if not app_module.home_assistant_available():
            self.skipTest("Home Assistant API is not available.")

        self.client.put(
            "/api/sensor-map",
            json={
                "sensors": [
                    {
                        "sensor": "sensor.nick_r_sleep_time_in_bed",
                        "description": "Total minutes recorded in bed.",
                    },
                    {
                        "sensor": "sensor.nick_r_sleep_minutes_asleep",
                        "description": "Total minutes asleep.",
                    },
                ]
            },
        )

        class FakeResponses:
            def create(self, **kwargs):
                assert kwargs["model"] == "gpt-4.1-nano"
                assert "days_requested: 7" in kwargs["input"]
                return SimpleNamespace(
                    id="resp_test",
                    output_text="I plotted the past week of cleaned sleep values.",
                )

        class FakeClient:
            responses = FakeResponses()

        original_get_client = app_module.get_client
        app_module.get_client = lambda: FakeClient()
        try:
            response = self.client.post(
                "/api/message",
                json={
                    "message": "What has my sleep been over the past week and plot it?",
                    "model": "gpt-4.1-nano",
                },
            )
        finally:
            app_module.get_client = original_get_client

        data = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data["plot"]["sensor"], "sensor.nick_r_sleep_minutes_asleep")
        self.assertEqual(data["plot"]["days"], 7)
        self.assertGreater(data["plot"]["samples"], 0)
        self.assertLessEqual(data["plot"]["samples"], 7)
        self.assertTrue(data["plot"]["cleaned"])
        self.assertEqual(data["plot"]["python_image"]["renderer"], "matplotlib")
        self.assertEqual(data["plot"]["python_image"]["y_axis_label"], "Minutes Asleep")

    def test_prompt_sensor_data_request_returns_rows(self):
        if not app_module.home_assistant_available():
            self.skipTest("Home Assistant API is not available.")

        self.client.put(
            "/api/sensor-map",
            json={
                "sensors": [
                    {
                        "sensor": "sensor.nick_r_sleep_minutes_asleep",
                        "description": "Total minutes asleep",
                    }
                ]
            },
        )

        class FakeResponses:
            def create(self, **kwargs):
                assert kwargs["model"] == app_module.DEFAULT_MODEL
                assert "raw sensor data" in kwargs["input"].lower()
                return SimpleNamespace(
                    id="resp_test",
                    output_text="I loaded the recent sensor rows.",
                )

        class FakeClient:
            responses = FakeResponses()

        original_get_client = app_module.get_client
        app_module.get_client = lambda: FakeClient()
        try:
            response = self.client.post(
                "/api/message",
                json={"message": "show raw sensor data for sleep minutes asleep for 7 days"},
            )
        finally:
            app_module.get_client = original_get_client

        data = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(data["sensor_data"]["available"])
        self.assertEqual(data["sensor_data"]["sensor"], "sensor.nick_r_sleep_minutes_asleep")
        self.assertGreater(len(data["sensor_data"]["rows"]), 0)

    def test_plot_endpoint_returns_numeric_points(self):
        if not app_module.home_assistant_available():
            self.skipTest("Home Assistant API is not available.")

        response = self.client.get("/api/sensor-plot?sensor=sensor.nick_r_sleep_minutes_asleep&days=30")
        data = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertTrue(data["available"])
        self.assertTrue(data["cleaned"])
        self.assertEqual(data["source"], "completed_sleep")
        self.assertGreater(len(data["points"]), 0)
        self.assertNotIn(0, [point["value"] for point in data["points"]])
        self.assertEqual(data["python_image"]["renderer"], "matplotlib")
        self.assertEqual(data["python_image"]["x_axis_label"], "Date")
        self.assertEqual(data["python_image"]["y_axis_label"], "Minutes Asleep")
        encoded = data["python_image"]["data_url"].split(",", 1)[1]
        self.assertEqual(base64.b64decode(encoded)[:8], b"\x89PNG\r\n\x1a\n")

    def test_sleep_efficiency_plot_uses_cleaned_completed_values(self):
        if not app_module.home_assistant_available():
            self.skipTest("Home Assistant API is not available.")

        response = self.client.get("/api/sensor-plot?sensor=sensor.nick_r_sleep_efficiency&days=30")
        data = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertTrue(data["available"])
        self.assertTrue(data["cleaned"])
        self.assertEqual(data["metric"], "efficiency")
        self.assertGreater(len(data["points"]), 0)
        self.assertNotIn(0, [point["value"] for point in data["points"]])
        self.assertEqual(data["python_image"]["y_axis_label"], "Efficiency")

    def test_sensor_data_endpoint_returns_rows(self):
        if not app_module.home_assistant_available():
            self.skipTest("Home Assistant API is not available.")

        response = self.client.get("/api/sensor-data?sensor=sensor.nick_r_sleep_minutes_asleep&days=7&limit=20")
        data = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertTrue(data["available"])
        self.assertGreater(len(data["rows"]), 0)

    def test_sleep_summary_endpoint_accepts_days(self):
        if not app_module.home_assistant_available():
            self.skipTest("Home Assistant API is not available.")

        response = self.client.get("/api/home-assistant/sleep-summary?days=90")
        data = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(data["completed_sleep"]["days_requested"], 90)
        if data["completed_sleep"]["available"]:
            self.assertGreater(data["completed_sleep"]["days_returned"], 0)

    def test_markdown_table_renderer_outputs_html_table(self):
        template = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")
        start = template.index("function escapeHtml")
        end = template.index("function messageElement")
        renderer_source = template[start:end]
        script = (
            renderer_source + "\nconst input = `| Domain | WHO-aligned target / concern |\\n"
            "|---|---|\\n"
            "| BMI | **18.5-24.9 kg/m²** |\\n"
            "| Sleep | 7-9 hours/night |`;\n"
            "process.stdout.write(markdownToHtml(input));"
        )

        result = subprocess.run(
            ["node", "-e", script],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn('<table class="markdown-table">', result.stdout)
        self.assertIn("<th>Domain</th>", result.stdout)
        self.assertIn("<td>BMI</td>", result.stdout)
        self.assertIn("<strong>18.5-24.9 kg/m²</strong>", result.stdout)
        self.assertNotIn("|---|---|", result.stdout)

    def test_plot_renderer_has_axis_labels(self):
        template = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")
        start = template.index("function buildPlotCard")
        end = template.index("function buildSensorDataCard")
        renderer_source = template[start:end]
        script = (
            "class Node {"
            "constructor(name){this.name=name;this.children=[];this.attrs={};this.textContent='';this.className='';}"
            "appendChild(node){this.children.push(node);return node;}"
            "append(...nodes){this.children.push(...nodes);}"
            "setAttribute(key,value){this.attrs[key]=String(value);}"
            "}"
            "const document={createElement:(name)=>new Node(name),createElementNS:(_ns,name)=>new Node(name)};"
            + renderer_source
            + (
                "\nconst plot={available:true,sensor:'sensor.nick_r_sleep_minutes_asleep',cleaned:true,"
                "metric:'minutes_asleep',"
            )
            + "samples:2,min:315,max:472,average:393.5,latest:472,date_range:'May 20-21, 2026',"
            + "points:[{timestamp:1,value:315,time:'May 20, 2026'},{timestamp:2,value:472,time:'May 21, 2026'}]};"
            + "const card=buildPlotCard(plot);"
            + "function collect(node, klass, out=[]){"
            + "if(node.attrs && node.attrs.class===klass) out.push(node.textContent);"
            + "for (const child of node.children || []) collect(child, klass, out); return out;}"
            + "process.stdout.write(JSON.stringify({"
            + "axis:collect(card,'plot-axis-label'),ticks:collect(card,'plot-tick-label')"
            + "}));"
        )

        result = subprocess.run(
            ["node", "-e", script],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn('"Date"', result.stdout)
        self.assertIn('"minutes asleep"', result.stdout)
        self.assertIn("May 20, 2026", result.stdout)
        self.assertIn("May 21, 2026", result.stdout)

    def test_plot_renderer_prefers_python_plot_image(self):
        template = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")
        start = template.index("function buildPlotCard")
        end = template.index("function buildSensorDataCard")
        renderer_source = template[start:end]
        script = (
            "class Node {"
            "constructor(name){"
            "this.name=name;this.children=[];this.attrs={};this.textContent='';"
            "this.className='';this.src='';this.alt='';"
            "}"
            "appendChild(node){this.children.push(node);return node;}"
            "append(...nodes){this.children.push(...nodes);}"
            "setAttribute(key,value){this.attrs[key]=String(value);}"
            "}"
            "const document={createElement:(name)=>new Node(name),createElementNS:(_ns,name)=>new Node(name)};"
            + renderer_source
            + (
                "\nconst plot={available:true,sensor:'sensor.nick_r_sleep_minutes_asleep',cleaned:true,"
                "metric:'minutes_asleep',"
            )
            + "samples:2,min:315,max:472,average:393.5,latest:472,date_range:'May 20-21, 2026',"
            + "python_image:{data_url:'data:image/png;base64,abc123',renderer:'matplotlib',"
            + "x_axis_label:'Date',y_axis_label:'Minutes Asleep',title:'Sleep'},"
            + "points:[{timestamp:1,value:315,time:'May 20, 2026'},{timestamp:2,value:472,time:'May 21, 2026'}]};"
            + "const card=buildPlotCard(plot);"
            + "function collect(node, out=[]){"
            + "if(node.name==='img') out.push({src:node.src, alt:node.alt, klass:node.className});"
            + "for (const child of node.children || []) collect(child, out); return out;}"
            + "function texts(node, out=[]){"
            + "if(node.textContent) out.push(node.textContent); "
            + "for (const child of node.children || []) texts(child, out); return out;"
            + "}"
            + "process.stdout.write(JSON.stringify({images:collect(card), text:texts(card)}));"
        )

        result = subprocess.run(
            ["node", "-e", script],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn('"src":"data:image/png;base64,abc123"', result.stdout)
        self.assertIn('"klass":"python-plot-image"', result.stdout)
        self.assertIn("Rendered with Python (matplotlib)", result.stdout)
        self.assertIn("Y: Minutes Asleep", result.stdout)


if __name__ == "__main__":
    unittest.main()
