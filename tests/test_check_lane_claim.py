"""Tests for ``scripts/check_lane_claim.py`` — the decision-layer lane gate.

The gate fails a PR that edits one of the three decision-layer files
(``engine/ev_engine.py`` / ``engine/wheel_runner.py`` /
``engine/candidate_dossier.py``) without naming that file in a
``lane-claim`` block in the PR description. These tests pin the
behaviour matrix the gate promises in ``docs/PARALLEL_SESSIONS.md``:
no-decision-layer-touch passes, an unclaimed decision-layer edit fails,
a claimed one passes, partial claims fail on the unclaimed remainder,
and the absence of any claim source (not a PR context) skips rather
than fails.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from check_lane_claim import (  # noqa: E402  — sys.path injection above
    DECISION_LAYER_FILES,
    _claimed_files,
    main,
)

_CLAIM = (
    "## Summary\nRefactor.\n\n"
    "<!-- lane-claim\n"
    "files: engine/wheel_runner.py\n"
    "board: https://github.com/MertYakar66/smart-wheel-engine/issues/113#issuecomment-1\n"
    "-->\n"
)
_NO_BLOCK = "## Summary\nNo lane-claim block at all.\n"


class TestClaimedFiles:
    def test_path_in_block_is_claimed(self) -> None:
        touched = ["engine/wheel_runner.py"]
        assert _claimed_files(_CLAIM, touched) == {"engine/wheel_runner.py"}

    def test_path_only_in_prose_outside_block_not_claimed(self) -> None:
        # The path is mentioned, but not inside a lane-claim block.
        body = "I edited engine/wheel_runner.py for a refactor."
        assert _claimed_files(body, ["engine/wheel_runner.py"]) == set()

    def test_no_block_claims_nothing(self) -> None:
        assert _claimed_files(_NO_BLOCK, list(DECISION_LAYER_FILES)) == set()

    def test_only_named_paths_claimed(self) -> None:
        touched = ["engine/wheel_runner.py", "engine/ev_engine.py"]
        assert _claimed_files(_CLAIM, touched) == {"engine/wheel_runner.py"}


class TestMainExitCodes:
    def test_no_decision_layer_touch_passes(self) -> None:
        rc = main(["--changed-files", "engine/tail_risk.py", "tests/test_x.py"])
        assert rc == 0

    def test_unclaimed_edit_fails(self, tmp_path: Path) -> None:
        claim = tmp_path / "body.md"
        claim.write_text(_NO_BLOCK, encoding="utf-8")
        rc = main(["--changed-files", "engine/wheel_runner.py", "--claim-file", str(claim)])
        assert rc == 1

    def test_claimed_edit_passes(self, tmp_path: Path) -> None:
        claim = tmp_path / "body.md"
        claim.write_text(_CLAIM, encoding="utf-8")
        rc = main(
            [
                "--changed-files",
                "engine/wheel_runner.py",
                "engine/regime_hmm.py",
                "--claim-file",
                str(claim),
            ]
        )
        assert rc == 0

    def test_partial_claim_fails_on_remainder(self, tmp_path: Path) -> None:
        claim = tmp_path / "body.md"
        claim.write_text(_CLAIM, encoding="utf-8")
        rc = main(
            [
                "--changed-files",
                "engine/wheel_runner.py",
                "engine/ev_engine.py",
                "--claim-file",
                str(claim),
            ]
        )
        assert rc == 1

    def test_no_claim_source_skips(self, monkeypatch) -> None:
        # No --claim-file and no PR_BODY => not a PR context => skip (exit 0)
        # even though a decision-layer file is touched.
        monkeypatch.delenv("PR_BODY", raising=False)
        rc = main(["--changed-files", "engine/candidate_dossier.py"])
        assert rc == 0

    def test_pr_body_env_is_honored(self, monkeypatch) -> None:
        monkeypatch.setenv("PR_BODY", _CLAIM)
        rc = main(["--changed-files", "engine/wheel_runner.py"])
        assert rc == 0

    def test_empty_pr_body_in_context_fails(self, monkeypatch) -> None:
        # Empty string body is still a PR context (enforced), not a skip.
        monkeypatch.setenv("PR_BODY", "")
        rc = main(["--changed-files", "engine/ev_engine.py"])
        assert rc == 1
