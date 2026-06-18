import os
from pathlib import Path

_ENV_FILE = Path(__file__).resolve().parent / ".env"


def _load_dotenv(path: Path) -> dict[str, str]:
    """Parse a flat ``KEY=VALUE`` .env file. Blank lines, ``#`` comments and
    inline ``# ...`` trailers are ignored; surrounding quotes are stripped."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        val = val.split("#", 1)[0].strip().strip('"').strip("'")
        out[key.strip()] = val
    return out


_DOTENV = _load_dotenv(_ENV_FILE)


def _get(key: str, default):
    """OS env overrides .env, which overrides the built-in default."""
    if key in os.environ:
        return os.environ[key]
    if key in _DOTENV:
        return _DOTENV[key]
    return default


def get_int(key: str, default: int) -> int:
    return int(_get(key, default))


def get_float(key: str, default: float) -> float:
    return float(_get(key, default))


def get_str(key: str, default: str) -> str:
    return str(_get(key, default))


def get_bool(key: str, default: bool) -> bool:
    v = _get(key, default)
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "on")


# ── Shared run parameters ────────────────────────────────────────────────────
SAMPLES   = get_int("SAMPLES", 64)         # FPS sample count (M), all demos
DIMS      = get_int("DIMS", 3)             # 2 = top-down view, 3 = full 3-D
MAX_RANGE = get_float("MAX_RANGE", 40.0)   # horizontal crop radius, metres
MIN_Z     = get_float("MIN_Z", -1.5)       # ground-removal height, metres
BUDGET    = get_int("BUDGET", 40)          # max edits/frame before full FPS rebuild
