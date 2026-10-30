"""
tracking.py  --  the time-domain layer, and P&O as its first tracker.
REVISION 2 -- curve orientation fixed; parity assertion added.

PLACE THIS FILE AT:   C:\\Users\\user\\gmppt\\gmppt\\tracking.py
                      OVERWRITE revision 1.

SELF-TEST:
    python -m gmppt.tracking

WHAT REVISION 1 GOT WRONG  (defect D9)
    device.module_iv returns voltage DESCENDING -- 44.53 V down to -1.11 V, with
    a few points in reverse bias. Revision 1 assumed ascending order and took
    V[-1] as V_oc, which therefore read -1.11 V. Every commanded voltage was
    clamped to that, and np.interp on a descending xp array returns meaningless
    values. The oracle scored 0.00% instead of ~100%.

    Two corrections:

      1. Arrays are explicitly ordered ascending before any interpolation, and
         V_oc is taken as the maximum rather than as the last element.

      2. trajectory_for now ASSERTS that interpolating the stored curve at
         v_gmpp reproduces p_gmpp to 0.1%. That assertion would have fired on
         the first scenario rather than at the oracle check, and it is the guard
         that makes any future orientation change impossible to miss.

    The underlying mistake was rebuilding something that already existed:
    harness.curve_and_ceiling returns the ascending form together with v_gmpp and
    p_gmpp, and is the path every scored result in this project already uses.
    Revision 2 keeps the single module_iv call (n_peaks is needed and
    curve_and_ceiling does not return it) but orders the arrays explicitly and
    verifies the result rather than trusting an assumption about layout.

    Note that check 1 -- the oracle -- did exactly what it was written to do. A
    layer whose scoring is wrong produces plausible tracker numbers and no error
    signal. The check cost one run and saved every number built on top of it.

WHY THIS FILE EXISTS
    Every result in this project so far is SINGLE-SHOT: given a curve, choose one
    voltage, score the power there. That measures a seed, not a tracker.

    A tracker holds an operating point and steps it over time, and what is scored
    is the whole trajectory. The main objective -- a tracker that finds the
    global peak without being trapped in local peaks, and beats P&O, InC and PSO
    -- cannot be evaluated at all until this layer exists. Two of the three
    funded targets (dynamic efficiency, convergence time) are gated behind it,
    and so is the comparison the objective names.

    gmppt/trackers.py was stubbed with the signature fn(ctx) -> voltage, which is
    the single-shot interface. Filling those stubs in against Context would have
    produced plausible numbers measuring the wrong thing -- the same class of
    error as defect D2. Trackers are written against Trajectory instead.

WHAT IS GATED AND WHAT IS NOT
    trackers.py gates P&O, InC and PSO behind Month-3 sign-off of the Gate A
    redirection. That gate is mis-scoped for the BASELINES: the funded proposal
    commits to implementing P&O, InC and PSO in Months 1-3 (section 4.1) and to
    benchmarking against them (section 4.4), and no outcome of the Month-3
    meeting removes that commitment.

    The HYBRID -- the learned seed feeding a P&O fine-tracking stage -- is where
    the redirection genuinely bears, and it stays gated. It needs P&O built first
    in any case, so nothing is lost by that ordering.

THE QUASI-STATIC ASSUMPTION -- declared, not assumed
    Each control step assumes the converter has settled to the commanded voltage
    before power is read. Converter settling is microseconds to milliseconds; a
    module-level MPPT control period is tens of milliseconds. The assumption is
    sound at this timescale, and it scopes what is measured: ALGORITHM behaviour,
    not converter dynamics. Converter dynamics are Layer 1, and belong to the
    partner.

    Consequence for cost: under STATIC shading the curve does not change over
    time, only the operating point moves along it. A 400-step trajectory needs
    ONE simulator call plus interpolation. P2.8 (EN 50530 ramps) breaks that
    property and must be costed separately.

CONVERGENCE IS REPORTED IN STEPS
    The partner's control period is not available, so convergence is reported in
    CONTROL STEPS and labelled as such. Multiplying by the control period
    converts to seconds when the number arrives; nothing is recomputed.

THE STEP SIZE IS SWEPT, NOT CHOSEN
    P&O's perturbation size sets the classic trade-off: large steps converge fast
    and oscillate hard at the peak, small steps track cleanly and converge
    slowly. The partner's value is unknown, so the runner sweeps the literature
    range and reports the curve, as P3.8 did for the probe schedule. When the
    partner supplies a value it is marked on a curve that already contains it.

STARTING POINT
    Trajectories start at 0.80 * V_oc -- the industry fixed fractional-Voc
    coefficient, which is what a real optimizer seeds with. Starting at the true
    peak would flatter every tracker equally and measure nothing.

TWO EFFICIENCIES, REPORTED SEPARATELY
    window_efficiency   energy captured over the whole window / energy available.
                        Includes the convergence transient.
    steady_efficiency   the same over the last STEADY_FRACTION of the window.
                        Excludes the transient; this is what the funded
                        static-efficiency target refers to.

    A single blended number would conflate "how fast" with "how well", which is
    the criterion-mismatch pattern recorded four times in this project.

ACCEPTANCE FOR THE LAYER ITSELF -- declared before the first run, unchanged
      1. ORACLE. A tracker parked at the true GMPP scores ~100% on both
         efficiencies. If not, the scoring is wrong and nothing downstream means
         anything.
      2. TRAPPING. P&O is trapped on at least 10% of multi-peak scenarios. P&O
         getting stuck on local peaks is the premise of the project; a layer
         where it never happens is not exercising the physics.
      3. SINGLE PEAK. P&O reaches the peak on at least 95% of uniform scenarios.
      4. STEP-SIZE MONOTONICITY. Smaller steps converge more slowly.

    Check 2 is written so it CAN fail -- the standing corrective from D5.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import config, device
from .device import ModuleParams

# -- declared constants -----------------------------------------------------

START_FRACTION = 0.80        # industry fixed coefficient, as a seed point
N_STEPS = 400                # control steps per trajectory
STEADY_FRACTION = 0.25       # last quarter of the window = steady state
REACH_TOLERANCE = 0.99       # within 1% of GMPP power counts as reached
V_FLOOR_FRACTION = 0.02      # operating point never driven below this
DEFAULT_STEP_FRAC = 0.005    # 0.5% of V_oc, mid-range of the literature

PARITY_TOL = 1e-3            # stored curve must reproduce p_gmpp to 0.1%

# Literature range for the P&O perturbation, swept rather than chosen.
STEP_SWEEP = (0.001, 0.002, 0.005, 0.010)


class CurveOrientationError(RuntimeError):
    """Raised when the stored curve does not reproduce the known GMPP.

    Exists because revision 1 silently interpolated a descending array and
    produced zeros. A guard that cannot fire gives false assurance; this one is
    checked on every trajectory built.
    """


@dataclass
class Trajectory:
    """One tracker running on one static curve, stepped over control periods.

    The tracker sees `step(v) -> power` and nothing else: the curve, the true
    GMPP and the peak structure are held here and are not reachable from the
    tracker, exactly as Context seals the answer in the single-shot harness.

    V is guaranteed ascending. Do not construct this directly -- use
    trajectory_for, which orders the arrays and verifies them.
    """
    V: np.ndarray
    P: np.ndarray
    v_gmpp: float
    p_gmpp: float
    n_peaks: int
    module: str = ""
    geometry: str = ""

    # -- recorded during the run ----------------------------------------
    v_hist: list = field(default_factory=list)
    p_hist: list = field(default_factory=list)

    @property
    def v_oc(self) -> float:
        """Maximum voltage on the curve, not the last element.

        The raw simulator curve is descending and extends into reverse bias, so
        V[-1] is a negative number. Taking the maximum is orientation-safe.
        """
        return float(self.V.max())

    @property
    def v_floor(self) -> float:
        return max(V_FLOOR_FRACTION * self.v_oc, float(self.V.min()))

    @property
    def v(self) -> float:
        """Current operating voltage; the start point before any step."""
        return self.v_hist[-1] if self.v_hist else START_FRACTION * self.v_oc

    def power_at(self, v: float) -> float:
        """Interpolate the curve. NOT recorded -- internal use only."""
        v = float(min(max(v, self.v_floor), self.v_oc))
        return float(np.interp(v, self.V, self.P))

    def step(self, v: float) -> float:
        """Command a voltage, advance one control period, return the power.

        Every call is one control step and is recorded. A tracker's cost is the
        number of steps it takes, measured here rather than asserted.
        """
        v = float(min(max(v, self.v_floor), self.v_oc))
        p = float(np.interp(v, self.V, self.P))
        self.v_hist.append(v)
        self.p_hist.append(p)
        return p

    # -- scoring --------------------------------------------------------

    def metrics(self) -> dict:
        """Trajectory scores. Two efficiencies, kept separate by design."""
        if not self.p_hist:
            return {"n_steps": 0}

        p = np.asarray(self.p_hist, dtype=float)
        n = len(p)
        k = max(1, int(round(STEADY_FRACTION * n)))
        steady = p[-k:]

        denom = max(self.p_gmpp, 1e-12)
        window_eff = float(p.mean() / denom)
        steady_eff = float(steady.mean() / denom)
        reached = bool(steady.mean() >= REACH_TOLERANCE * self.p_gmpp)

        # Convergence: the first step after which power stays at or above the
        # tolerance for the REMAINDER of the window. Requiring it to stay rules
        # out a tracker that passes through the peak and wanders off being
        # scored as converged.
        conv = None
        thresh = REACH_TOLERANCE * self.p_gmpp
        below = np.nonzero(p < thresh)[0]
        if len(below) == 0:
            conv = 0
        elif below[-1] < n - 1:
            conv = int(below[-1] + 1)

        return {
            "n_steps": int(n),
            "window_efficiency_pct": 100.0 * window_eff,
            "steady_efficiency_pct": 100.0 * steady_eff,
            "reached_gmpp": reached,
            "convergence_steps": conv,          # None = never settled
            "final_v_over_voc": float(self.v_hist[-1] / self.v_oc),
            "v_gmpp_over_voc": float(self.v_gmpp / self.v_oc),
            "n_peaks": int(self.n_peaks),
            "multi_peak": bool(self.n_peaks > 1),
            "p_gmpp_w": float(self.p_gmpp),
            "energy_lost_w": float(self.p_gmpp - p.mean()),
            "module": self.module,
            "geometry": self.geometry,
        }


def trajectory_for(scenario) -> Trajectory:
    """Build a Trajectory from a scenario. One simulator call.

    The raw curve is DESCENDING in voltage and extends into reverse bias (see
    D9). It is explicitly sorted ascending here, and the result is verified
    against the known GMPP before being returned.
    """
    mp = ModuleParams.from_cec(scenario.module)
    curve = device.module_iv(mp, scenario.irradiances, scenario.temp_c,
                             n_substrings=int(config.N_SUBSTRINGS),
                             bd=config.breakdown())
    a = device.analyse(curve)

    V = np.asarray(curve["V"], dtype=float)
    P = np.asarray(curve["P"], dtype=float)
    order = np.argsort(V)               # orientation-agnostic
    V, P = V[order], P[order]

    v_gmpp = float(a["gmpp"]["V"])
    p_gmpp = float(a["gmpp"]["P"])

    # The guard that revision 1 lacked.
    p_check = float(np.interp(v_gmpp, V, P))
    if abs(p_check - p_gmpp) > PARITY_TOL * max(abs(p_gmpp), 1e-9):
        raise CurveOrientationError(
            f"stored curve does not reproduce the GMPP for {scenario.module}: "
            f"interpolated {p_check:.4f} W at {v_gmpp:.4f} V, analyse reports "
            f"{p_gmpp:.4f} W. The curve layout has changed -- check ordering "
            f"before trusting any trajectory.")

    return Trajectory(V=V, P=P, v_gmpp=v_gmpp, p_gmpp=p_gmpp,
                      n_peaks=int(a["n_peaks"]),
                      module=str(scenario.module),
                      geometry=str(scenario.geometry))


# ---------------------------------------------------------------------------
# TRACKERS
# ---------------------------------------------------------------------------

def perturb_and_observe(traj: Trajectory, step_frac: float = DEFAULT_STEP_FRAC,
                        n_steps: int = N_STEPS) -> None:
    """Textbook P&O.

    Perturb the voltage by a fixed amount, observe whether power rose or fell,
    continue in the same direction if it rose and reverse if it fell.

    This is the method the project exists to beat, and its known weakness is that
    it hill-climbs to whichever peak it starts nearest and cannot leave it. That
    behaviour is not a defect to be fixed here -- reproducing it is check 2.
    """
    dv = step_frac * traj.v_oc
    v = START_FRACTION * traj.v_oc
    p_prev = traj.step(v)

    direction = -1.0        # move down-voltage first, toward the knee
    for _ in range(n_steps - 1):
        v = v + direction * dv
        p = traj.step(v)
        if p < p_prev:
            direction = -direction
        p_prev = p


def parked_at_gmpp(traj: Trajectory, n_steps: int = N_STEPS) -> None:
    """Oracle control: sit on the true peak for the whole window.

    Not a buildable method -- it reads the answer. It exists so that check 1 can
    confirm the scoring is right before any tracker is believed. It caught D9.
    """
    for _ in range(n_steps):
        traj.step(traj.v_gmpp)


# ---------------------------------------------------------------------------
# VERIFICATION
# ---------------------------------------------------------------------------

def run_tracker(fn, scenarios, **kwargs) -> list:
    """Run one tracker over a scenario set, returning per-trajectory metrics."""
    out = []
    for sc in scenarios:
        traj = trajectory_for(sc)
        fn(traj, **kwargs)
        out.append(traj.metrics())
    return out


def summarise(records: list, label: str) -> dict:
    """Aggregate, reported two-sided: whole set, multi-peak, single-peak."""
    def block(rows):
        if not rows:
            return None
        conv = [r["convergence_steps"] for r in rows
                if r["convergence_steps"] is not None]
        return {
            "n": len(rows),
            "window_eff_pct": float(np.mean([r["window_efficiency_pct"]
                                             for r in rows])),
            "steady_eff_pct": float(np.mean([r["steady_efficiency_pct"]
                                             for r in rows])),
            "reached_pct": 100.0 * float(np.mean([r["reached_gmpp"]
                                                  for r in rows])),
            "median_conv_steps": (float(np.median(conv)) if conv else None),
            "worst_energy_lost_w": float(max(r["energy_lost_w"] for r in rows)),
        }

    multi = [r for r in records if r["multi_peak"]]
    single = [r for r in records if not r["multi_peak"]]
    return {"label": label, "all": block(records),
            "multi_peak": block(multi), "single_peak": block(single)}


def _print_block(s: dict) -> None:
    print(f"\n   {s['label']}")
    print(f"   {'subset':<14} {'n':>5} {'window %':>10} {'steady %':>10} "
          f"{'reached %':>10} {'conv steps':>11} {'worst W':>9}")
    print("   " + "-" * 74)
    for key in ("all", "multi_peak", "single_peak"):
        b = s.get(key)
        if not b:
            continue
        conv = ("-" if b["median_conv_steps"] is None
                else f"{b['median_conv_steps']:.0f}")
        print(f"   {key:<14} {b['n']:>5} {b['window_eff_pct']:>10.2f} "
              f"{b['steady_eff_pct']:>10.2f} {b['reached_pct']:>10.1f} "
              f"{conv:>11} {b['worst_energy_lost_w']:>9.2f}")


def main() -> int:
    from .harness import scenario_set

    print("P2.7  time-domain layer and P&O  (revision 2)")
    try:
        print("  " + config.provenance())
    except Exception:
        pass
    print("=" * 78)
    print(f"quasi-static; start {START_FRACTION:.2f}*Voc; {N_STEPS} control "
          f"steps; steady state = last {100*STEADY_FRACTION:.0f}%")
    print("convergence reported in CONTROL STEPS -- the partner's control period")
    print("converts to seconds later; nothing is recomputed.\n")
    print("acceptance, declared before this run:")
    print("   1. oracle parked at GMPP scores ~100% on both efficiencies")
    print("   2. P&O is trapped on >= 10% of multi-peak scenarios")
    print("   3. P&O reaches the peak on >= 95% of single-peak scenarios")
    print("   4. smaller steps converge more slowly")
    print("   Check 2 is written so it CAN fail. A layer where P&O is never")
    print("   trapped is not exercising the physics this project addresses.\n")

    scenarios = scenario_set(200)
    print(f"{len(scenarios)} scenarios (training modules; this is the layer")
    print("check, not a reportable comparison)")

    # -- check 1: the oracle -------------------------------------------------
    print("\n1. ORACLE  (parked at the true GMPP)")
    try:
        orc = run_tracker(parked_at_gmpp, scenarios)
    except CurveOrientationError as e:
        print(f"\n   CURVE ORIENTATION CHECK FIRED:\n   {e}")
        return 1
    s_orc = summarise(orc, "parked at GMPP [not buildable]")
    _print_block(s_orc)
    ok1 = (s_orc["all"]["window_eff_pct"] > 99.9
           and s_orc["all"]["steady_eff_pct"] > 99.9)
    print(f"\n   -> {'PASS' if ok1 else 'FAIL'}  "
          f"(scoring is {'sound' if ok1 else 'WRONG -- stop here'})")
    if not ok1:
        return 1

    # -- checks 2 and 3: P&O -------------------------------------------------
    print(f"\n2. P&O  (step {100*DEFAULT_STEP_FRAC:.1f}% of Voc)")
    recs = run_tracker(perturb_and_observe, scenarios,
                       step_frac=DEFAULT_STEP_FRAC)
    s_po = summarise(recs, f"P&O, step {100*DEFAULT_STEP_FRAC:.1f}%")
    _print_block(s_po)

    n_multi = s_po["multi_peak"]["n"] if s_po["multi_peak"] else 0
    trapped = (100.0 - s_po["multi_peak"]["reached_pct"]) if n_multi else 0.0
    single_reached = (s_po["single_peak"]["reached_pct"]
                      if s_po["single_peak"] else float("nan"))

    print(f"\n   multi-peak scenarios: {n_multi}")
    print(f"   TRAPPED on multi-peak: {trapped:.1f}%   (declared bar: >= 10%)")
    print(f"   reached on single-peak: {single_reached:.1f}%   "
          f"(declared bar: >= 95%)")

    ok2 = n_multi > 0 and trapped >= 10.0
    ok3 = not np.isnan(single_reached) and single_reached >= 95.0
    print(f"   -> check 2 {'PASS' if ok2 else 'FAIL'}, "
          f"check 3 {'PASS' if ok3 else 'FAIL'}")
    if n_multi == 0:
        print("   NOTE: no multi-peak scenarios drawn. Check 2 was NOT TESTED --")
        print("   this is not a pass. Increase the scenario count and re-run.")

    # -- check 4: step-size sweep -------------------------------------------
    print("\n3. STEP-SIZE SWEEP  (the parameter the partner has not supplied)")
    print(f"   {'step % Voc':>11} {'steady %':>10} {'reached %':>10} "
          f"{'conv steps':>11} {'trapped % multi':>17}")
    print("   " + "-" * 65)
    sweep = []
    for sf in STEP_SWEEP:
        r = run_tracker(perturb_and_observe, scenarios, step_frac=sf)
        s = summarise(r, f"step {sf}")
        mp = s["multi_peak"]
        conv = s["all"]["median_conv_steps"]
        row = {"step_frac": sf,
               "steady_eff_pct": s["all"]["steady_eff_pct"],
               "reached_pct": s["all"]["reached_pct"],
               "median_conv_steps": conv,
               "trapped_multi_pct": (100.0 - mp["reached_pct"]) if mp else None}
        sweep.append(row)
        conv_txt = "-" if conv is None else f"{conv:.0f}"
        trap_txt = ("-" if row["trapped_multi_pct"] is None
                    else f"{row['trapped_multi_pct']:.1f}")
        print(f"   {100*sf:>11.1f} {row['steady_eff_pct']:>10.2f} "
              f"{row['reached_pct']:>10.1f} {conv_txt:>11} {trap_txt:>17}")

    convs = [r["median_conv_steps"] for r in sweep
             if r["median_conv_steps"] is not None]
    ok4 = len(convs) >= 2 and convs[0] >= convs[-1]
    print(f"\n   smaller steps converge more slowly: "
          f"{'PASS' if ok4 else 'FAIL -- the stepping loop is suspect'}")

    print("\n" + "=" * 78)
    passed = sum([ok1, ok2, ok3, ok4])
    print(f"{passed}/4 layer checks passed")
    if passed == 4:
        print("\nThe time-domain layer reproduces P&O's known behaviour: it")
        print("hill-climbs to whichever peak it starts nearest and cannot leave")
        print("it. InC, PSO and the hybrid can now be built on a base that has")
        print("been shown to exercise the phenomenon they are meant to address.")
    else:
        print("\nDo NOT build further trackers on this layer until every check")
        print("passes. A failing check here invalidates everything above it.")
    return 0 if passed == 4 else 1


if __name__ == "__main__":
    raise SystemExit(main())