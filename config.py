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


def get_float_opt(key: str, default):
    """Like :func:`get_float`, but a ``None`` default or blank value yields ``None``
    (used for optional crops that are off unless set)."""
    v = _get(key, default)
    if v is None or (isinstance(v, str) and not v.strip()):
        return None
    return float(v)


def get_str(key: str, default: str) -> str:
    return str(_get(key, default))


def get_bool(key: str, default: bool) -> bool:
    v = _get(key, default)
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def get_tuple(key: str, default):
    """Parse a ``x,y,z`` triple from env/.env, e.g. ``ROI_MAX=50,50,3``.

    Returns ``default`` (which may be ``None``) when unset or blank.
    """
    v = _get(key, default)
    if v is default or v is None or (isinstance(v, str) and not v.strip()):
        return default
    if isinstance(v, (tuple, list)):
        return tuple(float(x) for x in v)
    return tuple(float(x) for x in str(v).split(","))


# ── Shared run parameters ────────────────────────────────────────────────────
SAMPLES   = get_int("SAMPLES", 64)         # FPS sample count (M), all demos
DIMS      = get_int("DIMS", 3)             # 2 = top-down view, 3 = full 3-D
BUDGET    = get_int("BUDGET", 40)          # max edits/frame before full FPS rebuild

# ── Cropping ─────────────────────────────────────────────────────────────────
# The region-of-interest box + ego-vehicle removal is the DEFAULT preprocessing
# (it stabilises FPS). The older cylinder/floor crop is off by default but stays
# available as a manual override (set MAX_RANGE / MIN_Z to re-enable it).
MAX_RANGE  = get_float_opt("MAX_RANGE", None)        # horizontal crop radius, m (off)
MIN_Z      = get_float_opt("MIN_Z", None)            # ground-removal height, m (off)
ROI_MIN    = get_tuple("ROI_MIN", (-50.0, -50.0, -3.0))  # keep inside box, (x,y,z) min
ROI_MAX    = get_tuple("ROI_MAX", (50.0, 50.0, 3.0))     # keep inside box, (x,y,z) max
EGO_EXTENT = get_tuple("EGO_EXTENT", (0.75, 1.5, 1.0))   # drop ego box, (x,y,z) half-extents


def triple_arg(s: str):
    """argparse ``type`` that parses ``'x,y,z'`` into a float triple."""
    parts = s.split(",")
    if len(parts) != 3:
        raise ValueError(f"expected 'x,y,z' (3 comma-separated numbers), got {s!r}")
    return tuple(float(x) for x in parts)


def add_crop_args(parser):
    """Add the shared ROI-box / ego-vehicle crop flags to an argparse parser."""
    parser.add_argument("--roi-min", type=triple_arg, default=ROI_MIN,
                        help="Keep points inside box; 'x,y,z' min, e.g. -50,-50,-3.")
    parser.add_argument("--roi-max", type=triple_arg, default=ROI_MAX,
                        help="Keep points inside box; 'x,y,z' max, e.g. 50,50,3.")
    parser.add_argument("--ego-extent", type=triple_arg, default=EGO_EXTENT,
                        help="Drop ego-vehicle box; 'x,y,z' half-extents, e.g. 0.75,1.5,1.")


def crop_kwargs(args) -> dict:
    """Collect the ROI/ego crop kwargs from parsed args for ``load_lidar_bin``."""
    return dict(roi_min=args.roi_min, roi_max=args.roi_max, ego_extent=args.ego_extent)
