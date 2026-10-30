"""
hybrid.py  --  the learned seed feeding a P&O fine-tracker. LIBRARY ONLY.
REVISION 3 -- region taken from the classifier, not derived backwards (D16).

PLACE THIS FILE AT:   C:\\Users\\user\\gmppt\\gmppt\\hybrid.py
                      OVERWRITE revision 2.

WHAT CHANGED IN REVISION 3  (defect D16)

    Revision 2 recovered the region by INVERTING the model's coefficient:

        k = model.predict_k(x)
        region = min(int(k * N_SUB), N_SUB - 1)

    while gmppt/fallback.py read the region straight from the classifier. Two
    derivations of one quantity, in two files.

    They disagree on a reachable edge case. model.predict_k clips the offset to
    [0, 1] INCLUSIVE:

        off = np.clip(reg.predict(...), 0.0, 1.0)
        k   = (r + off) / n_sub

    so an offset regressor that overshoots returns exactly 1.0, giving
    k * N_SUB = r + 1 and int(k * N_SUB) = r + 1. The derived region is one
    higher than the classifier's. For region 2 the outer min() masks it; for
    regions 0 and 1 it does not. A gradient-boosting regressor on a target
    bounded in [0, 1) has no constraint preventing overshoot, so this is not a
    theoretical corner.

    NO REPORTED FIGURE IS AFFECTED. The region clamp was measured never to bind
    (0 of 200 trajectories left the predicted region), and fallback.py's
    confident branch assigns its region variable without using it. predict_k's
    own k is correct in every case; only the reverse-derived region was wrong.

    Revision 3 takes the region from model.classifier in both files. One
    quantity, one authority. Same family as the duplicated make_scan_refine
    recorded in the Phase 3 log: two implementations of one thing diverge
    silently, and the divergence is found later than it should be.

    The private alias _seed_voltage is also removed. It existed only so the
    runner would not need touching during the file split, and keeping a
    private-named duplicate of a public function is the same one-thing-two-names
    pattern. Callers import seed_voltage.

WHAT CHANGED IN REVISION 2
    The main() that ran the first comparison was removed. Its verdict logic
    carried defect D10 -- condition (iii) required the hybrid to beat the
    seed-only control on STEADY EFFICIENCY, and it passed at 99.92% against
    99.92%, a margin below displayed precision. Steady efficiency is saturated
    at this level. The restated criterion lives in
    phase2/p7_tracker_comparison.py and names arrival rate and worst-case watts,
    where the difference is visible. Two verdict paths for one question also
    invites reading the wrong one: one place defines the hybrid, one place judges
    it.

    The MEASUREMENTS stand and are unaffected by either revision: on multi-peak
    scenarios the hybrid reaches the global peak on 99.4% of cases against P&O's
    42.1%, and cuts the worst single-scenario loss from 109.25 W to 3.13 W.

WHAT THIS IS
    The funded proposal, section 4.3, specifies a hybrid structure: a model
    predicts the voltage region near the GMPP to narrow the search range, after
    which a conventional method (P&O) performs fine tracking.

    Until P2.7 only the first half existed, and it was measured single-shot --
    choose a voltage, score the power there -- which is a coefficient result, not
    a tracking result. This file supplies the second half.

COST ACCOUNTING -- declared, and not free
    The seed spends five probes building its feature vector. On hardware those
    are real measurements taken by the same converter at the same control rate as
    any other sample, so they are charged as CONTROL STEPS: the seed consumes
    five steps of the window before fine tracking begins. Every method in the
    comparison runs for the same window. Giving the seed free measurements would
    be the uncost-matched comparison recorded as defect D2.

TWO VARIANTS, BECAUSE THEY ANSWER DIFFERENT QUESTIONS
    bounded   P&O is clamped inside the predicted substring region. The region
              prediction is ENFORCED and cannot be left.
    free      P&O starts at the seed's landing and is otherwise unconstrained.
              The region prediction is only a STARTING POINT.

    Identical scores would mean the seed's value is entirely in where it starts
    the search. The runner settles this by measurement rather than inference: it
    reads the free variant's recorded v_hist and asks whether the operating point
    ever left the predicted region. It never does, so the clamp could not have
    bound and the simpler unbounded form is the method.

THE CONTROL THAT DECIDES WHETHER THE HYBRID IS NEEDED AT ALL
    seed only -- park at the seed's prediction for the whole window, no fine
    tracking. If the hybrid cannot beat this, the P&O stage adds nothing and the
    proposal's hybrid structure is more machinery than the problem requires.
    Same class of control as P3.5's best-of-5 probes: the cheapest thing that
    might already work, run before claiming the elaborate thing is needed.

SCOPE
    Everything here is simulated. The multi-peak magnitude at full module scale
    is unvalidated against measurement and awaits partner data.
"""

from __future__ import annotations

from . import config
from .features import PROBE_FRACTIONS, extract_features
from .model import TwoStageModel
from .tracking import DEFAULT_STEP_FRAC, N_STEPS, Trajectory

N_SUB = int(config.N_SUBSTRINGS)
SEED_PROBE_COST = len(PROBE_FRACTIONS)      # charged as control steps

__all__ = ["SEED_PROBE_COST", "seed_voltage", "make_seed_only", "make_hybrid",
           "_po_from"]


def seed_voltage(traj, model, temp_c, v_oc=None) -> tuple[float, int]:
    """Run the seed on the trajectory. Returns (landing voltage, region).

    Features are drawn through traj.step, so the five probes are recorded as
    control steps and appear in the trajectory's cost. extract_features takes a
    sample(v) -> power callable and does not know its source -- the train-serve
    parity property asserted in P3.0. ONE module temperature, never
    per-substring, per the same contract.

    The region comes from model.classifier directly (D16). It is NOT recovered
    from the predicted coefficient: predict_k clips the offset to [0, 1]
    inclusive, so an overshooting offset regressor yields exactly 1.0 and the
    inverted region lands one interval too high.

    v_oc overrides the open-circuit voltage used BOTH to place the probes and to
    scale the landing. None keeps traj.v_oc, which is the value read from the
    simulated curve and the behaviour of every result before 26 Oct 2026.
    """
    voc = float(traj.v_oc) if v_oc is None else float(v_oc)
    x = extract_features(traj.step, voc, traj.module, float(temp_c)).reshape(1, -1)
    region = int(model.classifier.predict(x)[0])
    region = min(max(region, 0), N_SUB - 1)
    k = float(model.predict_k(x)[0])
    k = min(max(k, 0.0), 1.0)
    return k * voc, region


def _po_from(traj: Trajectory, v_start: float, n_steps: int,
             step_frac: float, lo: float | None = None,
             hi: float | None = None) -> None:
    """P&O from a given start, optionally clamped to [lo, hi].

    Logic is identical to tracking.perturb_and_observe; it differs only in taking
    a start point and optional bounds, so the fine-tracking stage is the same
    algorithm the baseline uses and no advantage is smuggled in.
    """
    dv = step_frac * traj.v_oc
    lo = traj.v_floor if lo is None else max(lo, traj.v_floor)
    hi = traj.v_oc if hi is None else min(hi, traj.v_oc)

    v = min(max(v_start, lo), hi)
    p_prev = traj.step(v)
    direction = -1.0
    for _ in range(n_steps - 1):
        v_next = v + direction * dv
        if v_next < lo or v_next > hi:
            direction = -direction
            v_next = v + direction * dv
        v = min(max(v_next, lo), hi)
        p = traj.step(v)
        if p < p_prev:
            direction = -direction
        p_prev = p


def make_seed_only(model: TwoStageModel):
    """Park at the seed's prediction. The control for condition (iii)."""
    def fn(traj: Trajectory, temp_c: float, n_steps: int = N_STEPS,
           **_) -> None:
        v_seed, _ = seed_voltage(traj, model, temp_c)
        for _ in range(n_steps - SEED_PROBE_COST):
            traj.step(v_seed)
    fn.__name__ = "seed_only"
    return fn


def make_hybrid(model: TwoStageModel, bounded: bool = True):
    """The proposal's section 4.3 design: seed predicts region, P&O refines."""
    def fn(traj: Trajectory, temp_c: float, n_steps: int = N_STEPS,
           step_frac: float = DEFAULT_STEP_FRAC) -> None:
        v_seed, region = seed_voltage(traj, model, temp_c)
        lo = hi = None
        if bounded:
            width = traj.v_oc / N_SUB
            lo, hi = region * width, (region + 1) * width
        _po_from(traj, v_seed, n_steps - SEED_PROBE_COST, step_frac, lo, hi)
    fn.__name__ = f"hybrid_{'bounded' if bounded else 'free'}"
    return fn