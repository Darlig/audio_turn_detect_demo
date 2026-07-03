from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
WEB_DIR = PROJECT_DIR / "web"


def load_dotenv(path: Path | None = None) -> None:
    env_path = path or PROJECT_DIR / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass(frozen=True)
class DemoConfig:
    livekit_url: str
    livekit_api_key: str
    livekit_api_secret: str
    public_livekit_url: str
    agent_name: str
    auto_dispatch_agent: bool
    room: str
    language: str
    host: str
    port: int
    ssl_cert_file: str
    ssl_key_file: str


def load_config() -> DemoConfig:
    load_dotenv()
    return DemoConfig(
        livekit_url=os.getenv("LIVEKIT_URL", "ws://127.0.0.1:8890"),
        livekit_api_key=os.getenv("LIVEKIT_API_KEY", "devkey"),
        livekit_api_secret=os.getenv("LIVEKIT_API_SECRET", "secret"),
        public_livekit_url=os.getenv("LIVEKIT_PUBLIC_URL", "").strip(),
        agent_name=os.getenv("LIVEKIT_AGENT_NAME", "cascade-voice-agent").strip(),
        auto_dispatch_agent=_env_bool("DEMO_AUTO_DISPATCH_AGENT", True),
        room=os.getenv("DEMO_ROOM", "audio-turn-demo"),
        language=os.getenv("DEMO_LANGUAGE", "zh"),
        host=os.getenv("DEMO_HOST", "127.0.0.1"),
        port=int(os.getenv("DEMO_PORT", "8090")),
        ssl_cert_file=os.getenv("DEMO_SSL_CERT_FILE", "").strip(),
        ssl_key_file=os.getenv("DEMO_SSL_KEY_FILE", "").strip(),
    )


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
