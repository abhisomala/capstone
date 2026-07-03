"""Load config/paths.yaml and resolve relative paths against the repo root.

No machine-specific absolute paths live in source. Everything is derived from the
repository location at run time, so the code is portable across machines and CI.
"""

from pathlib import Path
import yaml

# repo root = two levels up from this file (src/utils/config.py -> repo/)
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config" / "paths.yaml"


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    """Return the parsed configuration dictionary."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def resolve_path(relative_or_absolute: str) -> Path:
    """Resolve a config path string against the repo root (absolute paths pass through)."""
    p = Path(relative_or_absolute)
    return p if p.is_absolute() else (REPO_ROOT / p)
