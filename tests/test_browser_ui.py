import os

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_BROWSER_TESTS") != "1",
    reason="browser tests require RUN_BROWSER_TESTS=1",
)


def test_prompt_flow_shell_loads(page):
    base_url = os.environ.get("BROWSER_BASE_URL", "http://127.0.0.1:5056")

    page.goto(base_url, wait_until="networkidle")

    assert page.locator("h1").inner_text() == "Prompt Flow"
    assert page.locator("#chatList").is_visible()
    assert page.locator("#newChatButton").is_visible()
    assert page.locator("#modelSelect").input_value() == "gpt-4.1-nano"


def test_sidebar_new_chat_button_creates_empty_chat(page):
    base_url = os.environ.get("BROWSER_BASE_URL", "http://127.0.0.1:5056")

    page.goto(base_url, wait_until="networkidle")
    initial_count = page.locator(".chat-row").count()
    with page.expect_response(lambda response: response.url.endswith("/api/chats") and response.status == 201):
        page.locator("#newChatButton").click()

    assert page.locator(".chat-row").count() == initial_count + 1
    assert page.locator(".chat-row.active .chat-title").inner_text()
