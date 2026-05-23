import os

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_BROWSER_TESTS") != "1",
    reason="browser tests require RUN_BROWSER_TESTS=1",
)


def test_prompt_flow_shell_loads(page):
    base_url = os.environ.get("BROWSER_BASE_URL", "http://127.0.0.1:5056")
    console_errors = []
    failed_requests = []
    page.on("console", lambda message: console_errors.append(message.text) if message.type == "error" else None)
    page.on("requestfailed", lambda request: failed_requests.append(request.url))

    page.goto(base_url, wait_until="networkidle")

    assert page.locator("h1").inner_text() == "Prompt Flow"
    assert page.locator("#chatList").is_visible()
    assert page.locator("#newChatButton").is_visible()
    assert page.locator("#modelSelect").input_value() == "gpt-4.1-nano"
    assert page.locator("#setupPanel").count() == 0
    assert page.get_by_text("Setup diagnostics").count() == 0
    assert page.locator("#keyStatus").is_visible()
    assert page.evaluate("window.PROMPT_FLOW_STATIC_JS_LOADED") is True
    css_loaded = page.evaluate(
        "getComputedStyle(document.documentElement).getPropertyValue('--prompt-flow-css-loaded').trim()"
    )
    assert css_loaded == "yes"
    assert console_errors == []
    assert failed_requests == []


def test_key_help_is_visible_without_setup_diagnostics_panel(page):
    base_url = os.environ.get("BROWSER_BASE_URL", "http://127.0.0.1:5056")

    page.goto(base_url, wait_until="networkidle")
    page.locator("#keyHelpButton").click()

    assert page.locator("#keyHelpPanel").is_visible()
    assert page.locator("#keyHelpPanel").inner_text().find("/config/secrets.yaml") >= 0
    assert page.locator("#keyHelpPanel").inner_text().find("openai_api_key") >= 0
    assert page.locator("#keyHelpPanel").inner_text().find("claude_api_key") >= 0
    assert page.locator("#setupStatusText").count() == 0
    assert page.get_by_text("Setup diagnostics").count() == 0


def tab_layout_snapshot(page):
    return page.evaluate(
        """
        () => {
          const wanted = [
            ["promptsTab", ".tab[data-tab='prompts']"],
            ["sensorMapsTab", ".tab[data-tab='sensorMaps']"],
            ["tabsContainer", ".tabs"],
            ["mainPane", ".main-pane"],
            ["promptsPanel", "#promptsPanel"],
            ["sensorMapsPanel", "#sensorMapsPanel"],
          ];
          const props = [
            "display",
            "flex",
            "flexGrow",
            "alignSelf",
            "alignItems",
            "height",
            "minHeight",
            "maxHeight",
            "gridRow",
            "writingMode",
          ];
          const out = {};
          for (const [name, selector] of wanted) {
            const el = document.querySelector(selector);
            const style = window.getComputedStyle(el);
            const rect = el.getBoundingClientRect();
            out[name] = {
              className: el.className,
              offsetHeight: el.offsetHeight,
              clientHeight: el.clientHeight,
              rectHeight: rect.height,
            };
            for (const prop of props) {
              out[name][prop] = style[prop];
            }
          }
          return out;
        }
        """
    )


def assert_compact_tabs(snapshot):
    for name in ("promptsTab", "sensorMapsTab"):
        height = snapshot[name]["rectHeight"]
        assert 25 <= height <= 45, f"{name} height {height}; snapshot={snapshot}"


def test_tabs_stay_compact_when_switching_and_resizing(page):
    base_url = os.environ.get("BROWSER_BASE_URL", "http://127.0.0.1:5056")
    page.set_viewport_size({"width": 950, "height": 620})
    page.goto(base_url, wait_until="networkidle")

    initial = tab_layout_snapshot(page)
    print("tab layout initial", initial)
    assert_compact_tabs(initial)

    page.locator(".tab[data-tab='prompts']").click()
    prompts = tab_layout_snapshot(page)
    print("tab layout after prompts click", prompts)
    assert_compact_tabs(prompts)

    page.locator(".tab[data-tab='sensorMaps']").click()
    sensor_maps = tab_layout_snapshot(page)
    print("tab layout after sensor maps click", sensor_maps)
    assert_compact_tabs(sensor_maps)

    page.set_viewport_size({"width": 640, "height": 620})
    resized = tab_layout_snapshot(page)
    print("tab layout after resize", resized)
    assert_compact_tabs(resized)


def test_sidebar_new_chat_button_creates_empty_chat(page):
    base_url = os.environ.get("BROWSER_BASE_URL", "http://127.0.0.1:5056")

    page.goto(base_url, wait_until="networkidle")
    initial_count = page.locator(".chat-row").count()
    with page.expect_response(lambda response: response.url.endswith("/api/chats") and response.status == 201):
        page.locator("#newChatButton").click()

    assert page.locator(".chat-row").count() == initial_count + 1
    assert page.locator(".chat-row.active .chat-title").inner_text()
