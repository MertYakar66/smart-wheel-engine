"""Tests for the headless IB Gateway puller (scripts/ibkr_gateway_pull.py).

The socket I/O needs a live Gateway, so these cover the pure, regression-prone
piece: the structured-contract → contract_description synthesis, and crucially
that it ROUND-TRIPS losslessly back through the shared transform's parser (so the
gateway path and the cloud path produce identical option fields). ib_insync is
imported lazily inside the I/O functions, so importing the module here needs no
ib_insync install.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "ibkr_gateway_pull", _REPO / "scripts" / "ibkr_gateway_pull.py"
)
gw = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gw)
live = gw._live  # the shared transform module the puller loaded


def test_syn_description_option_put():
    assert (
        gw._syn_description("MU", "OPT", "P", 970.0, "20260612", "AMEX")
        == "MU Jun12'26 970 PUT @AMEX"
    )


def test_syn_description_option_call_fractional_default_exchange():
    assert (
        gw._syn_description("AMD", "OPT", "C", 532.5, "20260612", "")
        == "AMD Jun12'26 532.5 CALL @SMART"
    )


def test_syn_description_stock_is_bare_symbol():
    assert gw._syn_description("CLS", "STK", "", 0.0, "", "") == "CLS"


def test_synthesized_option_roundtrips_through_the_shared_parser():
    # gateway structured fields -> description -> shared parser must recover them
    for sym, right, strike, exp in [
        ("MU", "P", 970.0, "20260612"),
        ("AMD", "C", 532.5, "20260612"),
        ("ENB", "C", 86.0, "20261016"),
        ("MRVL", "P", 297.5, "20260710"),
    ]:
        desc = gw._syn_description(sym, "OPT", right, strike, exp, "AMEX")
        got = live.parse_option_description(desc)
        want_expiry = f"{exp[0:4]}-{exp[4:6]}-{exp[6:8]}"
        assert got == (sym, right, strike, want_expiry)


def test_summary_tag_map_covers_the_account_card_fields():
    # the puller must map exactly the scalars the snapshot account block needs
    assert set(gw._SUMMARY_TAGS.values()) == {
        "net_liquidation",
        "total_cash_value",
        "available_funds",
        "excess_liquidity",
        "maintenance_margin",
        "buying_power",
    }


def test_module_imports_nothing_from_the_trio():
    src = (_REPO / "scripts" / "ibkr_gateway_pull.py").read_text(encoding="utf-8")
    import_lines = "\n".join(
        ln for ln in src.splitlines() if ln.strip().startswith(("import ", "from "))
    )
    for forbidden in ("ev_engine", "wheel_runner", "candidate_dossier"):
        assert forbidden not in import_lines, f"gateway puller must not import {forbidden} (sec 2)"


def test_no_order_methods_referenced():
    # §3 hard line: read-only. The puller must never CALL an order method. Check
    # the call form ("method(") so the docstring may still describe the contract.
    src = (_REPO / "scripts" / "ibkr_gateway_pull.py").read_text(encoding="utf-8")
    for forbidden in ("placeOrder(", "cancelOrder(", "reqGlobalCancel(", "bracketOrder("):
        assert forbidden not in src, f"read-only puller must not call {forbidden} (sec 3)"
