"""Where the engine's data lives — one root, set by ``SWE_DATA_ROOT``.

Operator ruling 2026-09-18 (``DECISIONS.md`` D31): the data is not stored in
git; it lives on the operator's desktop, outside the repository folder, with
Google Drive as the verified backup. The repository holds code, docs, tests
and a small fixture data root only.

Mechanics
---------
Three conventional relative prefixes name data in this codebase::

    data/            data/bloomberg/*.csv, data/features/, data/schemas.py (code!)
    data_raw/        raw pulls (yfinance, constituents, day-bot ticks)
    data_processed/  regenerable local stores (theta, option_premium, ibkr, sim)

:func:`resolve` re-roots such a *relative* path under ``SWE_DATA_ROOT`` when
the variable is set, and returns it unchanged otherwise — so with the variable
unset every caller behaves exactly as before (CWD-relative, repo root when run
from the repo). Absolute paths and paths outside the three prefixes are never
touched, which keeps every explicit ``tmp_path`` in the test-suite intact.

The narrower overrides that already existed keep winning inside their scope:
``SWE_DATA_PROCESSED_DIR``, ``SWE_IBKR_DATA_DIR``, ``SWE_OPTION_PREMIUM_DIR``,
``SWE_SIM_DATA_DIR``. An absolute override is used as given; a *relative*
override — any relative value, not only the three prefixes — lands under the
root when one is set (``data_processed_b/`` → ``<root>/data_processed_b``) and
stays CWD-relative when none is, so a checkout never receives output meant
for the desktop root.

The decision-layer trio is untouched: ``WheelRunner`` keeps passing its
``"data/bloomberg"`` default to the connector, and the connector re-roots it.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "DATA_ROOT_ENV",
    "REROOTED_PREFIXES",
    "repo_root",
    "data_root",
    "resolve",
    "bloomberg_dir",
    "broad_pull_dir",
    "deep_dir",
    "features_dir",
    "raw_dir",
    "processed_dir",
    "theta_dir",
    "option_premium_dir",
    "ibkr_dir",
    "sim_dir",
]

DATA_ROOT_ENV = "SWE_DATA_ROOT"
REROOTED_PREFIXES: tuple[str, ...] = ("data", "data_raw", "data_processed")


def repo_root() -> Path:
    """The repository root (the parent of ``engine/``)."""
    return Path(__file__).resolve().parent.parent


def data_root() -> Path | None:
    """``SWE_DATA_ROOT`` as an absolute path, or ``None`` when unset/blank."""
    raw = os.environ.get(DATA_ROOT_ENV, "").strip()
    return Path(raw).expanduser().resolve() if raw else None


def resolve(path: str | Path) -> Path:
    """Re-root a conventional relative data path under ``SWE_DATA_ROOT``.

    ``resolve("data/bloomberg")`` → ``<root>/data/bloomberg`` when the root is
    set, ``Path("data/bloomberg")`` when it is not. Absolute paths, and relative
    paths whose first component is not one of :data:`REROOTED_PREFIXES`, are
    returned as ``Path(path)`` unchanged.
    """
    p = Path(path)
    root = data_root()
    if root is None or p.is_absolute() or not p.parts:
        return p
    if p.parts[0] not in REROOTED_PREFIXES:
        return p
    return root.joinpath(*p.parts)


def _override(name: str) -> Path | None:
    """A narrow override's path, or ``None`` when the variable is unset/blank.

    Absolute values are returned as given. Relative values land under the data
    root when one is set (whatever their first component), else stay relative.
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p
    root = data_root()
    return root.joinpath(*p.parts) if root is not None else p


def bloomberg_dir() -> Path:
    """The Bloomberg CSV panels the connector serves (``data/bloomberg``)."""
    return resolve("data/bloomberg")


def broad_pull_dir() -> Path:
    return resolve("data/bloomberg/broad_pull")


def deep_dir() -> Path:
    """The deep-history gz slices (opt-in via ``SWE_DEEP_HISTORY``)."""
    return resolve("data/bloomberg/deep")


def features_dir() -> Path:
    return resolve("data/features")


def raw_dir() -> Path:
    return resolve("data_raw")


def processed_dir() -> Path:
    """``SWE_DATA_PROCESSED_DIR`` if set, else ``data_processed`` under the root.

    Without a root the default stays the repository's own ``data_processed/``
    (absolute, from the module path) — the behaviour every existing caller had.
    """
    override = _override("SWE_DATA_PROCESSED_DIR")
    if override is not None:
        return override
    root = data_root()
    return root / "data_processed" if root is not None else repo_root() / "data_processed"


def theta_dir() -> Path:
    return processed_dir() / "theta"


def option_premium_dir() -> Path:
    """``SWE_OPTION_PREMIUM_DIR`` wins (the test-suite pins it); else the rail's
    conventional home under the processed tree."""
    override = _override("SWE_OPTION_PREMIUM_DIR")
    return override if override is not None else processed_dir() / "option_premium"


def ibkr_dir() -> Path:
    """``SWE_IBKR_DATA_DIR`` wins (deployments and the fixture demo point it);
    else ``data_processed/ibkr`` under the root."""
    override = _override("SWE_IBKR_DATA_DIR")
    return override if override is not None else processed_dir() / "ibkr"


def sim_dir() -> Path:
    """``SWE_SIM_DATA_DIR`` wins; else ``data_processed/sim`` under the root."""
    override = _override("SWE_SIM_DATA_DIR")
    return override if override is not None else processed_dir() / "sim"
