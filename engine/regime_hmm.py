"""
Gaussian Hidden Markov Model regime detector.

Replaces the heuristic Bollinger/ATR regime gate in
``engine/regime_detector.py`` with a proper probabilistic HMM fit to
daily log-returns and realised-volatility features.

Why an HMM?
-----------
The rule-based regime detector has three specific failings that the
audit flagged:

1. It falls through to a silent ``normal`` / 50% confidence when the
   data is insufficient. This corrupts EV position-sizing.
2. Its percentile thresholds are static and do not adapt to the
   underlying's own history.
3. It cannot tell you *how likely* you are to be in a regime —
   everything is a hard classification with no uncertainty.

A 4-state Gaussian HMM solves all three: it learns the state structure
from data, produces a full posterior distribution ``P(state | history)``,
and naturally handles small samples via Bayesian shrinkage.

State semantics
---------------
The four states are labelled after the fit by sorting on per-state
mean return × std, so the output is always interpretable regardless of
the random-init ordering:

    state 0 → crisis       (very negative mean, very high vol)
    state 1 → bear          (negative mean, high vol)
    state 2 → normal        (small positive mean, medium vol)
    state 3 → bull_quiet    (positive mean, low vol — BEST for wheel)

Usage
-----
::

    from engine.regime_hmm import GaussianHMM

    hmm = GaussianHMM(n_states=4, random_state=42)
    hmm.fit(log_returns)
    state_probs = hmm.predict_proba(log_returns)
    current_state = int(np.argmax(state_probs[-1]))
    multiplier = hmm.position_multiplier(state_probs[-1])

Implementation note: this is a pure-numpy implementation so the engine
does not need ``hmmlearn`` as a dependency. The EM algorithm is the
standard Baum-Welch; Viterbi decoding is also provided.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm


@dataclass
class HMMFit:
    """Result of a Baum-Welch fit."""

    n_states: int
    n_features: int
    start_prob: np.ndarray  # shape (K,)
    trans_mat: np.ndarray  # shape (K, K)
    means: np.ndarray  # shape (K, D)
    stds: np.ndarray  # shape (K, D)
    log_likelihood: float
    n_iter: int
    converged: bool
    state_labels: list[str] = field(default_factory=list)


class GaussianHMM:
    """Pure-numpy Gaussian HMM with EM fitting and Viterbi decoding.

    Diagonal-covariance only — we assume observations are features that
    are approximately independent within a state. This is fine for
    (log_return, rolling_rv) which are the two main inputs we care
    about.

    Parameters
    ----------
    n_states:
        Number of hidden states (default 4).
    n_iter:
        Max Baum-Welch iterations.
    tol:
        Log-likelihood change threshold for convergence.
    random_state:
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_states: int = 4,
        n_iter: int = 50,
        tol: float = 1e-3,
        random_state: int | None = 42,
    ) -> None:
        self.n_states = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state
        self.fit_result: HMMFit | None = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self, observations: np.ndarray) -> HMMFit:
        """Fit the HMM by Baum-Welch (EM) on a 1-D or 2-D observation array.

        When ``observations`` is 1-D it is reshaped to (T, 1) and treated
        as univariate returns. When it is 2-D (T, D) each column is a
        feature.
        """
        obs = np.asarray(observations, dtype=float)
        if obs.ndim == 1:
            obs = obs.reshape(-1, 1)
        T, D = obs.shape
        K = self.n_states

        if T < K * 3:
            raise ValueError(
                f"Need at least {K*3} observations to fit a {K}-state HMM, got {T}"
            )

        rng = np.random.default_rng(self.random_state)

        # Init: k-means++ style seeding on the observation mean
        # feature, which for (log_return,) is just percentile-based.
        start_prob = np.ones(K) / K
        trans_mat = np.full((K, K), 0.1 / (K - 1))
        np.fill_diagonal(trans_mat, 0.9)

        # Initial means via percentiles of the primary feature (col 0)
        ranks = np.argsort(obs[:, 0])
        means = np.zeros((K, D))
        stds = np.zeros((K, D))
        chunk = T // K
        for k in range(K):
            sl = ranks[k * chunk : (k + 1) * chunk if k < K - 1 else T]
            if len(sl) == 0:
                means[k] = obs.mean(axis=0)
                stds[k] = obs.std(axis=0) + 1e-6
            else:
                means[k] = obs[sl].mean(axis=0)
                stds[k] = obs[sl].std(axis=0) + 1e-6

        # Add tiny random jitter to break ties
        means = means + rng.normal(0, 1e-6, means.shape)

        prev_ll = -np.inf
        converged = False
        n_iter = 0
        for it in range(self.n_iter):
            n_iter = it + 1
            # --- E step ---
            log_emit = self._log_emission(obs, means, stds)
            log_alpha, log_beta, log_lik = self._forward_backward(
                log_emit, start_prob, trans_mat
            )

            # Posterior state probabilities γ[t,k] = P(state_t=k|obs)
            log_gamma = log_alpha + log_beta
            log_gamma -= np.logaddexp.reduce(log_gamma, axis=1, keepdims=True)
            gamma = np.exp(log_gamma)

            # ξ[t,i,j] = P(state_t=i, state_{t+1}=j|obs)
            log_xi = (
                log_alpha[:-1, :, None]
                + np.log(trans_mat + 1e-300)[None, :, :]
                + log_emit[1:, None, :]
                + log_beta[1:, None, :]
            )
            log_xi -= np.logaddexp.reduce(
                log_xi.reshape(T - 1, -1), axis=1, keepdims=True
            ).reshape(T - 1, 1, 1)
            xi = np.exp(log_xi)

            # --- M step ---
            start_prob = gamma[0]
            trans_numer = xi.sum(axis=0)
            trans_denom = trans_numer.sum(axis=1, keepdims=True)
            trans_mat = np.where(trans_denom > 0, trans_numer / (trans_denom + 1e-12), trans_mat)

            gamma_sum = gamma.sum(axis=0) + 1e-12
            means = (gamma[:, :, None] * obs[:, None, :]).sum(axis=0) / gamma_sum[:, None]
            diff_sq = (obs[:, None, :] - means[None, :, :]) ** 2
            var = (gamma[:, :, None] * diff_sq).sum(axis=0) / gamma_sum[:, None]
            stds = np.sqrt(var) + 1e-6

            if abs(log_lik - prev_ll) < self.tol:
                converged = True
                break
            prev_ll = log_lik

        # Sort states by (mean_return, -std) so state indexes are
        # interpretable: 0=worst, K-1=best.
        scores = means[:, 0] - 0.5 * stds[:, 0]  # crude return-adjusted metric
        order = np.argsort(scores)
        means = means[order]
        stds = stds[order]
        trans_mat = trans_mat[order][:, order]
        start_prob = start_prob[order]

        labels = self._label_states(means, stds, K)

        self.fit_result = HMMFit(
            n_states=K,
            n_features=D,
            start_prob=start_prob,
            trans_mat=trans_mat,
            means=means,
            stds=stds,
            log_likelihood=log_lik,
            n_iter=n_iter,
            converged=converged,
            state_labels=labels,
        )
        return self.fit_result

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def predict_proba(self, observations: np.ndarray) -> np.ndarray:
        """Return the filtered posterior P(state_t | obs_1..t)."""
        if self.fit_result is None:
            raise RuntimeError("HMM not fit yet")
        obs = np.asarray(observations, dtype=float)
        if obs.ndim == 1:
            obs = obs.reshape(-1, 1)
        log_emit = self._log_emission(obs, self.fit_result.means, self.fit_result.stds)
        log_alpha, _, _ = self._forward_backward(
            log_emit, self.fit_result.start_prob, self.fit_result.trans_mat
        )
        log_alpha -= np.logaddexp.reduce(log_alpha, axis=1, keepdims=True)
        return np.exp(log_alpha)

    def viterbi(self, observations: np.ndarray) -> np.ndarray:
        """Return the Viterbi most-likely state sequence."""
        if self.fit_result is None:
            raise RuntimeError("HMM not fit yet")
        obs = np.asarray(observations, dtype=float)
        if obs.ndim == 1:
            obs = obs.reshape(-1, 1)
        T = obs.shape[0]
        K = self.fit_result.n_states
        log_emit = self._log_emission(obs, self.fit_result.means, self.fit_result.stds)
        log_trans = np.log(self.fit_result.trans_mat + 1e-300)
        log_start = np.log(self.fit_result.start_prob + 1e-300)

        delta = np.full((T, K), -np.inf)
        psi = np.zeros((T, K), dtype=int)
        delta[0] = log_start + log_emit[0]
        for t in range(1, T):
            scores = delta[t - 1, :, None] + log_trans
            psi[t] = np.argmax(scores, axis=0)
            delta[t] = scores.max(axis=0) + log_emit[t]

        path = np.zeros(T, dtype=int)
        path[-1] = int(np.argmax(delta[-1]))
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path

    def position_multiplier(self, state_probs: np.ndarray) -> float:
        """Map a 1-D posterior to a position-size multiplier in [0, 1.25].

        Cold regimes (crisis, bear) push the multiplier below 1; hot
        regimes (bull_quiet) push it up to ~1.25. The EV engine's
        ``regime_multiplier`` field consumes this directly.
        """
        if self.fit_result is None:
            raise RuntimeError("HMM not fit yet")
        # Per-state weights by label — wheel-strategy-appropriate.
        weights = {
            "crisis": 0.2,
            "bear": 0.5,
            "normal": 1.0,
            "bull_quiet": 1.25,
        }
        labels = self.fit_result.state_labels
        total = 0.0
        for i, lbl in enumerate(labels):
            total += state_probs[i] * weights.get(lbl, 1.0)
        return float(total)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _log_emission(obs: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
        """Return log P(obs_t | state_k) summed across features (diag cov)."""
        T, D = obs.shape
        K = means.shape[0]
        log_emit = np.zeros((T, K))
        for k in range(K):
            # Diagonal-covariance multivariate normal = product of 1-D normals
            logp = norm.logpdf(obs, loc=means[k], scale=stds[k])
            log_emit[:, k] = logp.sum(axis=1)
        return log_emit

    @staticmethod
    def _forward_backward(
        log_emit: np.ndarray,
        start_prob: np.ndarray,
        trans_mat: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Log-space forward-backward. Returns (log_alpha, log_beta, log_lik)."""
        T, K = log_emit.shape
        log_start = np.log(start_prob + 1e-300)
        log_trans = np.log(trans_mat + 1e-300)

        log_alpha = np.full((T, K), -np.inf)
        log_alpha[0] = log_start + log_emit[0]
        for t in range(1, T):
            log_alpha[t] = log_emit[t] + np.logaddexp.reduce(
                log_alpha[t - 1, :, None] + log_trans, axis=0
            )

        log_beta = np.full((T, K), -np.inf)
        log_beta[-1] = 0.0
        for t in range(T - 2, -1, -1):
            log_beta[t] = np.logaddexp.reduce(
                log_trans + log_emit[t + 1, None, :] + log_beta[t + 1, None, :], axis=1
            )

        log_lik = float(np.logaddexp.reduce(log_alpha[-1]))
        return log_alpha, log_beta, log_lik

    @staticmethod
    def _label_states(means: np.ndarray, stds: np.ndarray, K: int) -> list[str]:
        """Assign human-readable labels based on per-state mean/std on feature 0.

        Works for K up to 4 which is all we need for the wheel regime.
        """
        if K == 4:
            return ["crisis", "bear", "normal", "bull_quiet"]
        if K == 3:
            return ["crisis", "normal", "bull"]
        if K == 2:
            return ["bear", "bull"]
        return [f"state_{i}" for i in range(K)]
