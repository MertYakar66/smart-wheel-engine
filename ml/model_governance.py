"""ML model lifecycle governance: model cards, drift detection,
champion/challenger promotion, and audit-ready registry."""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Literal


# ---------------------------------------------------------------------------
# 1. ModelCard
# ---------------------------------------------------------------------------

@dataclass
class ModelCard:
    """Immutable-ish descriptor for a trained model artifact."""

    model_id: str
    model_name: str
    version: str
    description: str
    training_date: date
    training_data_range: tuple[date, date]
    features: list[str]
    target: str
    metrics: dict[str, float] = field(default_factory=dict)
    deployment_threshold: dict[str, float] = field(default_factory=dict)
    status: Literal["development", "shadow", "champion", "retired"] = "development"
    last_validation_date: date | None = None
    drift_metrics: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 2. DriftDetector
# ---------------------------------------------------------------------------

class DriftDetector:
    """Statistical drift checks between baseline and current distributions."""

    # -- feature drift (PSI) ------------------------------------------------

    @staticmethod
    def check_feature_drift(
        baseline_stats: dict,
        current_stats: dict,
        threshold: float = 0.05,
    ) -> dict:
        """Compute Population Stability Index per feature.

        ``baseline_stats`` and ``current_stats`` map feature names to lists of
        bucket proportions (the same binning must be applied to both).  Returns
        a dict with per-feature PSI values and an overall ``drifted`` flag.
        """
        results: dict[str, object] = {}
        any_drifted = False

        common_features = set(baseline_stats.keys()) & set(current_stats.keys())
        for feat in sorted(common_features):
            baseline_buckets = baseline_stats[feat]
            current_buckets = current_stats[feat]
            psi = DriftDetector._psi(baseline_buckets, current_buckets)
            feat_drifted = psi > threshold
            if feat_drifted:
                any_drifted = True
            results[feat] = {"psi": round(psi, 6), "drifted": feat_drifted}

        results["drifted"] = any_drifted
        results["threshold"] = threshold
        return results

    # -- prediction drift ---------------------------------------------------

    @staticmethod
    def check_prediction_drift(
        baseline_preds: list[float],
        current_preds: list[float],
        threshold: float = 0.10,
    ) -> dict:
        """Compare prediction distributions via PSI.

        Both inputs are flat lists of prediction values.  They are binned into
        deciles internally so the caller does not need to pre-bucket.
        """
        baseline_buckets = DriftDetector._to_decile_proportions(baseline_preds)
        current_buckets = DriftDetector._to_decile_proportions(current_preds)
        psi = DriftDetector._psi(baseline_buckets, current_buckets)
        return {
            "psi": round(psi, 6),
            "drifted": psi > threshold,
            "threshold": threshold,
        }

    # -- performance drift --------------------------------------------------

    @staticmethod
    def check_performance_drift(
        baseline_metrics: dict[str, float],
        current_metrics: dict[str, float],
        tolerance: float = 0.15,
    ) -> dict:
        """Flag metrics that degraded beyond *tolerance* (relative drop).

        A metric is considered drifted when:
            (baseline - current) / |baseline| > tolerance
        """
        results: dict[str, object] = {}
        any_drifted = False

        for name in sorted(baseline_metrics.keys()):
            if name not in current_metrics:
                continue
            base_val = baseline_metrics[name]
            curr_val = current_metrics[name]
            if base_val == 0:
                rel_change = 0.0 if curr_val == 0 else float("inf")
            else:
                rel_change = (base_val - curr_val) / abs(base_val)
            metric_drifted = rel_change > tolerance
            if metric_drifted:
                any_drifted = True
            results[name] = {
                "baseline": base_val,
                "current": curr_val,
                "relative_change": round(rel_change, 6),
                "drifted": metric_drifted,
            }

        results["drifted"] = any_drifted
        results["tolerance"] = tolerance
        return results

    # -- internal helpers ---------------------------------------------------

    @staticmethod
    def _psi(baseline_proportions: list[float], current_proportions: list[float]) -> float:
        """Population Stability Index between two proportion vectors."""
        eps = 1e-6
        total = 0.0
        for b, c in zip(baseline_proportions, current_proportions):
            b = max(b, eps)
            c = max(c, eps)
            total += (c - b) * math.log(c / b)
        return total

    @staticmethod
    def _to_decile_proportions(values: list[float]) -> list[float]:
        """Bin *values* into 10 equal-width buckets and return proportions."""
        if not values:
            return [0.1] * 10  # uniform fallback
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        lo = sorted_vals[0]
        hi = sorted_vals[-1]
        if lo == hi:
            # all identical – put everything in one bucket
            props = [0.0] * 10
            props[0] = 1.0
            return props
        width = (hi - lo) / 10
        counts = [0] * 10
        for v in sorted_vals:
            idx = int((v - lo) / width)
            idx = min(idx, 9)
            counts[idx] += 1
        return [c / n for c in counts]


# ---------------------------------------------------------------------------
# 3. ChampionChallenger
# ---------------------------------------------------------------------------

class ChampionChallenger:
    """Manage the champion / shadow lifecycle for a set of models."""

    def __init__(self) -> None:
        self._models: dict[str, ModelCard] = {}

    # -- registration / queries ---------------------------------------------

    def register_model(self, card: ModelCard) -> None:
        """Register a new model card (must be in *development* status)."""
        if card.model_id in self._models:
            raise ValueError(f"Model {card.model_id!r} is already registered.")
        if card.status != "development":
            raise ValueError("Only models in 'development' status can be registered.")
        self._models[card.model_id] = card

    def get_champion(self) -> ModelCard | None:
        for card in self._models.values():
            if card.status == "champion":
                return card
        return None

    def get_shadow(self) -> ModelCard | None:
        for card in self._models.values():
            if card.status == "shadow":
                return card
        return None

    # -- promotions ---------------------------------------------------------

    def promote_to_shadow(self, model_id: str) -> None:
        """Move a *development* model into *shadow* mode.

        Only one shadow model is allowed at a time; any existing shadow is
        returned to *development*.
        """
        card = self._get_card(model_id)
        if card.status != "development":
            raise ValueError(
                f"Cannot promote model {model_id!r} to shadow: "
                f"current status is {card.status!r} (must be 'development')."
            )
        # Demote existing shadow back to development
        current_shadow = self.get_shadow()
        if current_shadow is not None:
            current_shadow.status = "development"
        card.status = "shadow"

    def promote_to_champion(self, model_id: str) -> None:
        """Promote a *shadow* model to *champion*, retiring the old champion."""
        card = self._get_card(model_id)
        if card.status != "shadow":
            raise ValueError(
                f"Cannot promote model {model_id!r} to champion: "
                f"current status is {card.status!r} (must be 'shadow')."
            )
        current_champion = self.get_champion()
        if current_champion is not None:
            current_champion.status = "retired"
        card.status = "champion"

    # -- evaluation ---------------------------------------------------------

    @staticmethod
    def evaluate_switch(
        champion_metrics: dict[str, float],
        challenger_metrics: dict[str, float],
        min_improvement: float = 0.05,
    ) -> bool:
        """Return ``True`` if the challenger beats the champion on every shared
        metric by at least *min_improvement* (relative)."""
        common = set(champion_metrics.keys()) & set(challenger_metrics.keys())
        if not common:
            return False
        for metric in common:
            champ_val = champion_metrics[metric]
            chall_val = challenger_metrics[metric]
            if champ_val == 0:
                if chall_val <= 0:
                    return False
                continue
            improvement = (chall_val - champ_val) / abs(champ_val)
            if improvement < min_improvement:
                return False
        return True

    # -- helpers ------------------------------------------------------------

    def _get_card(self, model_id: str) -> ModelCard:
        try:
            return self._models[model_id]
        except KeyError:
            raise KeyError(f"No model registered with id {model_id!r}.")


# ---------------------------------------------------------------------------
# 4. GovernanceRegistry
# ---------------------------------------------------------------------------

class GovernanceRegistry:
    """Central registry for model cards with approval workflow and audit trail."""

    def __init__(self) -> None:
        self._cards: dict[str, ModelCard] = {}
        self._audit: dict[str, list[dict]] = {}  # model_id -> list of events

    # -- card management ----------------------------------------------------

    def register(self, card: ModelCard) -> None:
        self._cards[card.model_id] = card
        self._record_event(
            card.model_id,
            action="registered",
            detail=f"Model '{card.model_name}' v{card.version} registered.",
        )

    def get_card(self, model_id: str) -> ModelCard:
        try:
            return self._cards[model_id]
        except KeyError:
            raise KeyError(f"No model card found for {model_id!r}.")

    def list_cards(self) -> list[ModelCard]:
        return list(self._cards.values())

    # -- deployment approval ------------------------------------------------

    def approve_deployment(self, model_id: str, approver: str) -> bool:
        """Approve *model_id* for deployment if all threshold metrics are met.

        Returns ``True`` when approved, ``False`` when metrics are below the
        deployment thresholds defined on the model card.
        """
        card = self.get_card(model_id)

        # Check every deployment threshold is satisfied
        for metric_name, min_value in card.deployment_threshold.items():
            actual = card.metrics.get(metric_name)
            if actual is None or actual < min_value:
                self._record_event(
                    model_id,
                    action="deployment_rejected",
                    detail=(
                        f"Metric '{metric_name}' below threshold: "
                        f"{actual} < {min_value}."
                    ),
                    approver=approver,
                )
                return False

        self._record_event(
            model_id,
            action="deployment_approved",
            detail=f"All thresholds met. Approved by {approver}.",
            approver=approver,
        )
        return True

    # -- audit trail --------------------------------------------------------

    def get_audit_trail(self, model_id: str) -> list[dict]:
        """Return the full ordered audit trail for *model_id*."""
        return list(self._audit.get(model_id, []))

    # -- reporting ----------------------------------------------------------

    def export_report(self) -> dict:
        """Export a governance summary suitable for serialisation."""
        cards_summary = []
        for card in self._cards.values():
            cards_summary.append({
                "model_id": card.model_id,
                "model_name": card.model_name,
                "version": card.version,
                "status": card.status,
                "training_date": card.training_date.isoformat(),
                "metrics": card.metrics,
                "drift_metrics": card.drift_metrics,
                "last_validation_date": (
                    card.last_validation_date.isoformat()
                    if card.last_validation_date
                    else None
                ),
            })

        return {
            "generated_at": datetime.utcnow().isoformat(),
            "total_models": len(self._cards),
            "models": cards_summary,
            "audit_events": {
                mid: list(events) for mid, events in self._audit.items()
            },
        }

    # -- internal -----------------------------------------------------------

    def _record_event(
        self,
        model_id: str,
        *,
        action: str,
        detail: str,
        approver: str | None = None,
    ) -> None:
        event: dict = {
            "event_id": uuid.uuid4().hex,
            "timestamp": datetime.utcnow().isoformat(),
            "model_id": model_id,
            "action": action,
            "detail": detail,
        }
        if approver is not None:
            event["approver"] = approver
        self._audit.setdefault(model_id, []).append(event)
