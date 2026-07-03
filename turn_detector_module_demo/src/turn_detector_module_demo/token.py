from __future__ import annotations

import datetime as dt
from urllib.parse import SplitResult, urlsplit, urlunsplit

from livekit import api


def build_token(
    *,
    api_key: str,
    api_secret: str,
    room: str,
    identity: str,
    name: str | None = None,
    can_publish: bool,
    can_subscribe: bool,
) -> str:
    return (
        api.AccessToken(api_key, api_secret)
        .with_identity(identity)
        .with_name(name or identity)
        .with_ttl(dt.timedelta(hours=2))
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room,
                can_publish=can_publish,
                can_subscribe=can_subscribe,
                can_publish_data=True,
            )
        )
        .to_jwt()
    )


def browser_signaling_url(
    *,
    livekit_url: str,
    request_host: str,
    request_scheme: str,
    public_livekit_url: str = "",
) -> str:
    if public_livekit_url:
        return public_livekit_url

    parsed = urlsplit(livekit_url)
    if parsed.scheme not in {"ws", "wss"}:
        return livekit_url
    if parsed.hostname not in {"127.0.0.1", "localhost", "0.0.0.0", "::1"}:
        return livekit_url

    host = _host_without_port(request_host)
    if not host:
        return livekit_url

    if host not in {"127.0.0.1", "localhost", "0.0.0.0", "::1"}:
        return urlunsplit(
            SplitResult(
                scheme="wss" if request_scheme == "https" else "ws",
                netloc=request_host,
                path="",
                query="",
                fragment="",
            )
        )

    if request_scheme == "https":
        return urlunsplit(
            SplitResult(
                scheme="wss",
                netloc=request_host,
                path=parsed.path,
                query=parsed.query,
                fragment=parsed.fragment,
            )
        )

    port = parsed.port or (443 if parsed.scheme == "wss" else 80)
    return urlunsplit(
        SplitResult(
            scheme=parsed.scheme,
            netloc=_format_host_port(host, port),
            path=parsed.path,
            query=parsed.query,
            fragment=parsed.fragment,
        )
    )


def _host_without_port(request_host: str) -> str:
    request_host = request_host.strip()
    if not request_host:
        return ""
    if request_host.startswith("["):
        end = request_host.find("]")
        return request_host[1:end] if end > 0 else request_host
    if request_host.count(":") == 1:
        return request_host.rsplit(":", 1)[0]
    return request_host


def _format_host_port(host: str, port: int) -> str:
    if ":" in host and not host.startswith("["):
        return f"[{host}]:{port}"
    return f"{host}:{port}"
