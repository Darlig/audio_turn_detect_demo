from __future__ import annotations

from turn_detector_module_demo.token import browser_signaling_url


def test_browser_signaling_url_uses_same_origin_proxy_for_lan_browser() -> None:
    url = browser_signaling_url(
        livekit_url="ws://127.0.0.1:8890",
        request_host="192.168.1.20:8090",
        request_scheme="http",
    )

    assert url == "ws://192.168.1.20:8090"


def test_browser_signaling_url_uses_same_origin_for_https_proxy() -> None:
    url = browser_signaling_url(
        livekit_url="ws://127.0.0.1:8890",
        request_host="demo.local:8443",
        request_scheme="https",
    )

    assert url == "wss://demo.local:8443"
