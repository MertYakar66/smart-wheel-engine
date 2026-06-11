"""Observe-only premium-correction pilot.

Measures how far the *real* Theta EOD option mid sits from the engine's
synthetic ``BSM(iv)`` premium on short cash-secured puts — i.e. the
``edge_vs_fair`` the engine would see if its synthetic-premium path were
swapped for a real market mid, with ``fair`` held at ``BSM(iv)``.

NOT the volatility risk premium. See ``docs/PREMIUM_CORRECTION_PILOT.md``
for the labeling discipline (Refinement 1) and what this pilot can / cannot
settle (Refinement 2).
"""
