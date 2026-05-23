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


def composer_layout_snapshot(page):
    return page.evaluate(
        """
        () => {
          const selectors = {
            app: ".app-shell",
            sidebar: ".chat-sidebar",
            main: ".main-pane",
            topbar: ".topbar",
            tabs: ".tabs",
            panel: "#promptsPanel",
            messages: "#messages",
            composer: "#promptForm",
            input: "#messageInput",
            send: "#sendButton",
          };
          const out = { viewport: { width: innerWidth, height: innerHeight } };
          for (const [name, selector] of Object.entries(selectors)) {
            const el = document.querySelector(selector);
            const rect = el.getBoundingClientRect();
            const style = getComputedStyle(el);
            out[name] = {
              top: rect.top,
              bottom: rect.bottom,
              height: rect.height,
              width: rect.width,
              clientHeight: el.clientHeight,
              scrollHeight: el.scrollHeight,
              overflow: style.overflow,
              minHeight: style.minHeight,
            };
          }
          out.inputVisible = out.input.top >= 0 && out.input.bottom <= innerHeight;
          out.sendVisible = out.send.top >= 0 && out.send.bottom <= innerHeight;
          out.composerBottomOverflow = out.composer.bottom - innerHeight;
          return out;
        }
        """
    )


def assert_composer_visible(snapshot):
    assert snapshot["inputVisible"], snapshot
    assert snapshot["sendVisible"], snapshot
    assert snapshot["composerBottomOverflow"] <= 16, snapshot


def test_prompt_composer_stays_visible_in_short_and_zoom_like_viewports(page):
    base_url = os.environ.get("BROWSER_BASE_URL", "http://127.0.0.1:5056")

    for size in (
        {"width": 950, "height": 620},
        {"width": 950, "height": 413},
        {"width": 950, "height": 320},
        {"width": 633, "height": 413},
        {"width": 633, "height": 320},
        {"width": 390, "height": 320},
        {"width": 390, "height": 460},
    ):
        page.set_viewport_size(size)
        page.goto(base_url, wait_until="networkidle")
        snapshot = composer_layout_snapshot(page)
        print("composer layout", size, snapshot)
        assert_composer_visible(snapshot)


def test_prompt_composer_stays_visible_with_browser_zoom(page):
    base_url = os.environ.get("BROWSER_BASE_URL", "http://127.0.0.1:5056")

    for size, zoom in (
        ({"width": 950, "height": 620}, 1.25),
        ({"width": 950, "height": 620}, 1.5),
        ({"width": 950, "height": 500}, 2),
        ({"width": 633, "height": 413}, 2),
        ({"width": 390, "height": 320}, 2.5),
    ):
        page.set_viewport_size(size)
        page.goto(base_url, wait_until="networkidle")
        page.evaluate("(zoom) => { document.documentElement.style.zoom = String(zoom); }", zoom)
        snapshot = composer_layout_snapshot(page)
        print("composer zoom layout", size, zoom, snapshot)
        assert_composer_visible(snapshot)
        assert snapshot["composer"]["overflow"] != "hidden", snapshot


def test_sidebar_new_chat_button_creates_empty_chat(page):
    base_url = os.environ.get("BROWSER_BASE_URL", "http://127.0.0.1:5056")

    page.goto(base_url, wait_until="networkidle")
    initial_count = page.locator(".chat-row").count()
    with page.expect_response(lambda response: response.url.endswith("/api/chats") and response.status == 201):
        page.locator("#newChatButton").click()

    assert page.locator(".chat-row").count() == initial_count + 1
    assert page.locator(".chat-row.active .chat-title").inner_text()


def test_analysis_visual_artifacts_render_user_friendly_cards(page):
    base_url = os.environ.get("BROWSER_BASE_URL", "http://127.0.0.1:5056")

    page.goto(base_url, wait_until="networkidle")
    result = page.evaluate(
        """
        () => {
          const svg = btoa('<svg xmlns="http://www.w3.org/2000/svg" width="400" height="120"><rect width="400" height="120" fill="white"/><text x="20" y="60">DAG</text></svg>');
          const visuals = {
            title: 'Generated visuals',
            artifacts: [
              {type: 'plot', title: 'Association Plot', description: 'Readable plot', data_url: 'data:image/png;base64,iVBORw0KGgo='},
              {type: 'dag', title: 'Same-Day DAG', description: 'Readable DAG', data_url: `data:image/svg+xml;base64,${svg}`},
              {type: 'latex', title: 'Pearson correlation', description: 'Rendered equation', latex: 'r = \\\\frac{\\\\sum (X_i-\\\\bar X)(Y_i-\\\\bar Y)}{...}'}
            ]
          };
          const card = buildAnalysisVisualsCard(visuals);
          document.querySelector('#messages').appendChild(card);
          const plot = document.querySelector('.analysis-plot-image');
          const dag = document.querySelector('.dag-image');
          const latex = document.querySelector('.latex-equation');
          return {
            title: document.querySelector('.analysis-visuals-title').textContent,
            plotVisible: !!plot && plot.getBoundingClientRect().width > 100,
            dagVisible: !!dag && dag.getBoundingClientRect().width > 100,
            latexText: latex ? latex.textContent : '',
            latexHtml: latex ? latex.innerHTML : '',
            artifactCount: document.querySelectorAll('.analysis-artifact').length,
          };
        }
        """
    )

    assert result["title"] == "Generated visuals"
    assert result["artifactCount"] == 3
    assert result["plotVisible"] is True
    assert result["dagVisible"] is True
    assert "∑" in result["latexText"]
    assert "latex-frac" in result["latexHtml"]


def test_n_of_1_visual_response_stays_inside_assistant_card(page):
    base_url = os.environ.get("BROWSER_BASE_URL", "http://127.0.0.1:5056")
    prompt = (
        "Make plots describing the N-of-1 analysis, show causal DAGs, and show the LaTeX equations "
        "for my time asleep sleep with non-sleep variables in the Sensor Map using "
        "https://arxiv.org/abs/2407.17666"
    )

    page.set_viewport_size({"width": 820, "height": 720})
    page.goto(base_url, wait_until="networkidle")
    result = page.evaluate(
        """
        (prompt) => {
          const wideSvg = btoa('<svg xmlns="http://www.w3.org/2000/svg" width="2400" height="700"><rect width="2400" height="700" fill="white"/><text x="80" y="180" font-size="80">Wide visual should fit response card</text></svg>');
          const message = {
            role: 'assistant',
            content: `For this request: ${prompt}\\n\\nThe deterministic N-of-1 analysis uses sensor.nick_r_sleep_minutes_asleep as the outcome and compares sensor.nick_r_steps plus binary_sensor.pantry_door_window. Inline math: \\\\(r = \\\\frac{a}{b}\\\\).\\n\\n$$\\\\hat{\\\\beta} = \\\\frac{\\\\sum_i X_iY_i}{\\\\sum_i X_i^2}$$`,
            provider_label: 'OpenAI',
            model_label: 'GPT-4.1 Nano',
            model: 'gpt-4.1-nano',
            analysis_visuals: {
              title: 'Generated visuals',
              artifacts: [
                {type: 'plot', title: 'Association Plot', description: 'Plot caption', data_url: `data:image/svg+xml;base64,${wideSvg}`},
                {type: 'dag', title: 'Same-Day DAG', description: 'DAG caption', data_url: `data:image/svg+xml;base64,${wideSvg}`},
                {type: 'latex', title: 'Pearson correlation', description: 'Equation caption', latex: 'r = \\\\frac{\\\\sum_i (X_i-\\\\bar{X})(Y_i-\\\\bar{Y})}{\\\\sqrt{\\\\sum_i (X_i-\\\\bar{X})^2}}'}
              ]
            }
          };
          document.querySelector('#messages').innerHTML = '';
          document.querySelector('#messages').appendChild(messageElement(message));

          const bubble = document.querySelector('.message.assistant');
          const content = bubble.querySelector('.content');
          const visuals = bubble.querySelector('.analysis-visuals');
          const plot = bubble.querySelector('.analysis-plot-image');
          const dag = bubble.querySelector('.dag-image');
          const latex = bubble.querySelector('.latex-equation');
          const stamp = bubble.querySelector('.model-stamp');
          const bubbleRect = bubble.getBoundingClientRect();
          const plotRect = plot.getBoundingClientRect();
          const dagRect = dag.getBoundingClientRect();
          return {
            prompt,
            bubbleWidth: bubbleRect.width,
            contentBeforeVisuals: !!(content.compareDocumentPosition(visuals) & Node.DOCUMENT_POSITION_FOLLOWING),
            visualsBeforeStamp: !!(visuals.compareDocumentPosition(stamp) & Node.DOCUMENT_POSITION_FOLLOWING),
            plotInside: plotRect.left >= bubbleRect.left - 1 && plotRect.right <= bubbleRect.right + 1,
            dagInside: dagRect.left >= bubbleRect.left - 1 && dagRect.right <= bubbleRect.right + 1,
            messagesOverflow: document.querySelector('#messages').scrollWidth <= document.querySelector('#messages').clientWidth + 1,
            plotMaxWidth: getComputedStyle(plot).maxWidth,
            dagMaxWidth: getComputedStyle(dag).maxWidth,
            responseText: bubble.textContent,
            latexHtml: latex.innerHTML,
            inlineMathCount: bubble.querySelectorAll('.math-inline .latex-frac').length,
            artifactTitles: Array.from(bubble.querySelectorAll('.analysis-artifact-title')).map((node) => node.textContent),
          };
        }
        """,
        prompt,
    )

    assert "sensor.nick_r_sleep_minutes_asleep" in result["prompt"] or "time asleep" in result["prompt"]
    assert "sensor.nick_r_sleep_minutes_asleep" in result["responseText"]
    assert "sensor.nick_r_steps" in result["responseText"]
    assert "binary_sensor.pantry_door_window" in result["responseText"]
    assert result["bubbleWidth"] <= 780
    assert result["contentBeforeVisuals"] is True
    assert result["visualsBeforeStamp"] is True
    assert result["plotInside"] is True
    assert result["dagInside"] is True
    assert result["messagesOverflow"] is True
    assert result["plotMaxWidth"] == "100%"
    assert result["dagMaxWidth"] == "100%"
    assert "latex-frac" in result["latexHtml"]
    assert result["inlineMathCount"] >= 1
    assert result["artifactTitles"] == ["Association Plot", "Same-Day DAG", "Pearson correlation"]


def test_heavy_analysis_response_has_no_page_or_card_overflow(page):
    base_url = os.environ.get("BROWSER_BASE_URL", "http://127.0.0.1:5056")
    markdown = """
### Descriptive, Predictive, and Causal Output

- Descriptive: sleep minutes and steps are summarized as observed values.
- Predictive: regression is framed as prediction, not proof of cause.
- Causal: DAG edges are assumptions for N-of-1 screening, not proven effects.

| Analysis | Variables | Interpretation |
|---|---|---|
| Time-series | sleep minutes, steps | trends over time |
| Histogram | sleep efficiency | distribution only |
| Correlation | sleep, steps, awakenings, minutes awake, time in bed, efficiency | association, not causation |

```python
very_long_variable_name = "this line is intentionally long to verify that code scrolls inside the response card instead of expanding the Home Assistant ingress page width"
```

Inline math: \\(r = \\frac{\\sum_i (X_i-\\bar{X})(Y_i-\\bar{Y})}{\\sqrt{\\sum_i (X_i-\\bar{X})^2}}\\).

$$Y_t = \\alpha + \\beta X_{t-1} + \\epsilon_t$$
""".strip()

    viewports = (
        {"width": 1280, "height": 720},
        {"width": 1440, "height": 900},
        {"width": 1920, "height": 1080},
        {"width": 430, "height": 740},
    )

    for size in viewports:
        page.set_viewport_size(size)
        page.goto(base_url, wait_until="networkidle")
        result = page.evaluate(
            """
            (markdown) => {
              const wideSvg = btoa('<svg xmlns="http://www.w3.org/2000/svg" width="2600" height="900" viewBox="0 0 2600 900"><rect width="2600" height="900" fill="white"/><text x="80" y="150" font-size="72">Responsive plot or DAG content</text><rect x="80" y="220" width="520" height="190" rx="24" fill="#eff6ff" stroke="#2563eb" stroke-width="8"/><text x="340" y="330" text-anchor="middle" font-size="42">Steps</text><rect x="1960" y="220" width="520" height="190" rx="24" fill="#ecfdf5" stroke="#059669" stroke-width="8"/><text x="2220" y="330" text-anchor="middle" font-size="42">Sleep</text><path d="M 600 315 C 1050 315 1510 315 1960 315" fill="none" stroke="#334155" stroke-width="10"/></svg>');
              const message = {
                role: 'assistant',
                content: markdown,
                provider_label: 'OpenAI',
                model_label: 'GPT-4.1 Nano',
                model: 'gpt-4.1-nano',
                analysis_visuals: {
                  title: 'Generated visuals',
                  artifacts: [
                    {type: 'plot', title: 'Sleep Time-Series Plot', description: 'Line plot caption', data_url: `data:image/svg+xml;base64,${wideSvg}`},
                    {type: 'plot', title: 'Sleep Histogram', description: 'Histogram caption', data_url: `data:image/svg+xml;base64,${wideSvg}`},
                    {type: 'plot', title: 'Steps vs Sleep Scatterplot', description: 'Scatterplot caption', data_url: `data:image/svg+xml;base64,${wideSvg}`},
                    {type: 'plot', title: 'Correlation Heatmap', description: 'Heatmap caption', data_url: `data:image/svg+xml;base64,${wideSvg}`},
                    {type: 'dag', title: 'Causal DAG', description: 'Assumptions, not proof', data_url: `data:image/svg+xml;base64,${wideSvg}`},
                    {type: 'latex', title: 'N-of-1 Equation', description: 'Equation caption', latex: 'r = \\\\frac{\\\\sum_i (X_i-\\\\bar{X})(Y_i-\\\\bar{Y})}{\\\\sqrt{\\\\sum_i (X_i-\\\\bar{X})^2}}'}
                  ]
                }
              };
              const messages = document.querySelector('#messages');
              messages.innerHTML = '';
              messages.appendChild(messageElement(message));

              const rectFor = (selector) => {
                const node = document.querySelector(selector);
                const rect = node.getBoundingClientRect();
                const style = getComputedStyle(node);
                return {
                  left: rect.left,
                  right: rect.right,
                  top: rect.top,
                  bottom: rect.bottom,
                  width: rect.width,
                  height: rect.height,
                  overflowX: style.overflowX,
                  overflowY: style.overflowY,
                  maxWidth: style.maxWidth,
                  position: style.position,
                };
              };
              const bubble = rectFor('.message.assistant');
              const app = rectFor('.app-shell');
              const messagesRect = rectFor('#messages');
              const input = rectFor('#messageInput');
              const send = rectFor('#sendButton');
              const wrappers = Array.from(document.querySelectorAll('.markdown-table-wrap, .code-block pre, .latex-equation, .analysis-artifact, .plot-card')).map((node) => {
                const rect = node.getBoundingClientRect();
                const style = getComputedStyle(node);
                return {className: node.className, left: rect.left, right: rect.right, width: rect.width, overflowX: style.overflowX, scrollWidth: node.scrollWidth, clientWidth: node.clientWidth};
              });
              const media = Array.from(document.querySelectorAll('.analysis-plot-image, .dag-image, .python-plot-image, .content img, .content svg, .content canvas')).map((node) => {
                const rect = node.getBoundingClientRect();
                const parentRect = node.parentElement.getBoundingClientRect();
                return {className: node.className, left: rect.left, right: rect.right, width: rect.width, parentLeft: parentRect.left, parentRight: parentRect.right};
              });
              const tabs = Array.from(document.querySelectorAll('.main-pane > .tabs > .tab')).map((node) => node.getBoundingClientRect().height);
              return {
                viewport: {width: innerWidth, height: innerHeight},
                pageScrollWidth: document.documentElement.scrollWidth,
                pageClientWidth: document.documentElement.clientWidth,
                bodyScrollWidth: document.body.scrollWidth,
                bodyClientWidth: document.body.clientWidth,
                app,
                messagesRect,
                bubble,
                input,
                send,
                wrappers,
                media,
                tabs,
              };
            }
            """,
            markdown,
        )
        print("heavy analysis overflow snapshot", size, result)

        assert result["pageScrollWidth"] <= result["pageClientWidth"] + 1, result
        assert result["bodyScrollWidth"] <= result["bodyClientWidth"] + 1, result
        assert result["app"]["top"] >= -1 and result["app"]["bottom"] <= result["viewport"]["height"] + 1, result
        assert result["input"]["bottom"] <= result["viewport"]["height"] + 1, result
        assert result["send"]["bottom"] <= result["viewport"]["height"] + 1, result
        assert result["bubble"]["right"] <= result["messagesRect"]["right"] + 1, result
        assert all(wrapper["right"] <= result["bubble"]["right"] + 1 for wrapper in result["wrappers"]), result
        assert all(item["right"] <= item["parentRight"] + 1 for item in result["media"]), result
        assert all(height <= 45 for height in result["tabs"]), result

        page.locator(".tab[data-tab='sensorMaps']").click()
        sensor_height = page.locator(".app-shell").bounding_box()["height"]
        page.locator(".tab[data-tab='prompts']").click()
        prompts_height = page.locator(".app-shell").bounding_box()["height"]
        assert abs(sensor_height - prompts_height) <= 1
