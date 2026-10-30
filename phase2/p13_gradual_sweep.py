"""P2.9c  GRADUAL relocation: the same events, walked instead of spliced.

WHY THIS EXISTS
---------------
Every window in p12 is a SPLICED jump: the tracker settles on the pre-transition
curve, the curve is swapped for the post-transition one in a single control
step, and the detector is handed a step change. That is the most favourable
case a change detector can be given, and p12 says so in its own output.

The real event is not a step. The shadow band advances 0.000744 of a module
width per 30-second sample, so the global peak takes HOURS to migrate from one
substring to the next. A detector that compares instantaneous power against an
exponentially-weighted baseline with alpha = 0.10 -- roughly a one-second memory
at a 0.1 s control period -- has a baseline that sinks WITH a change that slow.

The standing prediction, recorded before this run:

    the power-drop detector fires on 0% of gradual relocations,
    at every threshold in the published range 0.05 - 0.20.

That is structural, not a tuning problem, and if it holds it is a finding rather
than a defeat: where there is no detectable event, a CLOCK is the correct
answer, and Eltamaly's predefined-time re-initialisation is doing the right
thing in that regime. The result then stops being "we beat the clock" and
becomes a boundary -- detection wins where the event outruns the detector's
memory, a clock wins where it does not -- which EXPLAINS the published method
instead of merely outscoring it.

THE DESIGN, AND WHY IT IS A FAIR PAIR
-------------------------------------
Identical endpoints to p12. The window still runs from the mid-plateau of one
developed GMPP-region run to the mid-plateau of the next adjacent one, selected
by the same `find_adjacent_transitions`, drawn from the same seeded-shuffled
module pool, de-duplicated on the same physics fingerprint and balanced across
the same two directions.

The ONLY difference is that p12 splices the two endpoint curves and this walks
every curve between them. Same scenarios, same drop depths, same directions --
one presented as a step and one as a ramp. Anything that differs between the two
runs is the event SHAPE and nothing else.

DWELL
-----
Each intermediate curve is held for `--dwell` control steps. The physical value
is 300: the generator samples the sun every 30 s and the control period is
0.1 s. Lowering it makes the run finish sooner by making the relocation FASTER,
which is the independent variable -- so it is a different experiment, not a
cheaper one. The script says so on every run that uses a non-physical dwell.

Cost at the physical dwell: a window spans 134 - 286 curves, so 40k - 86k
control steps per arm. Use `--n` to control the total, never `--dwell`.

WHAT IS MEASURED
----------------
The target MOVES, so p12's scalar metrics do not apply. Per scored step the
oracle is the true global peak of the curve in force at that step:

  energy loss %     1 - (captured / oracle) summed over the scored span
  off-peak %        share of scored steps below 99% of the current global peak
  arrived           captured >= 99% of the FINAL peak for HOLD consecutive
                    steps at the end of the span
  fires             how many times the re-seed policy triggered -- the headline

USAGE
    python phase2/p13_gradual_sweep.py --n 12
    python phase2/p13_gradual_sweep.py --n 12 --sweep-eps
    python phase2/p13_gradual_sweep.py --n 24 --sweep-schedule
    python phase2/p13_gradual_sweep.py --n 24 --with-competitors \
           --jamaa-model results/phase3/jamaa_net.joblib
    python phase2/p13_gradual_sweep.py --n 6 --dwell 60     # NOT the same experiment
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phase2 import p12_relocation_comparison as p12          # noqa: E402
from phase2.transition import (TRANSITION_FAMILIES,          # noqa: E402
                               pattern_sequence, validate_sequence)
from phase2.detectors import (ReseedPolicy,                  # noqa: E402
                              PowerDropDetector, ScheduledReseed)
from gmppt import config                                     # noqa: E402
from gmppt.model import TwoStageModel                        # noqa: E402
from gmppt.pso import DEFAULT_ITERATIONS, DEFAULT_POPULATION  # noqa: E402
from gmppt.tracking import DEFAULT_STEP_FRAC                 # noqa: E402

OUT = config.RESULTS_DIR / "phase2"

PHYSICAL_DWELL = 300      # 30 s per generator sample / 0.1 s control period

# Scheduled re-seed periods to sweep, in CONTROL STEPS at 0.1 s. 240 is
# Eltamaly's 24 s and is included as the inherited value, not as a candidate.
# The ladder runs to half an hour because the events in this set take 1.1-2.4
# hours: a 30-minute clock still samples the slowest of them four or five
# times, which is the regime where the cost term should start to win.
SCHEDULE_LADDER = [240, 600, 3000, 6000, 18000]    # 24 s, 1 m, 5 m, 10 m, 30 m
SECONDS_PER_DAY = 86_400.0
SETTLE_STEPS = p12.SETTLE_STEPS
EPS = p12.EPS             # "on peak" = captured >= (1-EPS) * current GMPP
HOLD = p12.HOLD


# ---------------------------------------------------------------------------
# WINDOW CONSTRUCTION
# ---------------------------------------------------------------------------

def make_gradual_window(traj_full, per_step, pre_idx, post_idx, win_seed,
                        dwell):
    """Walk every curve from pre_idx to post_idx inclusive.

    Block 0 is the settle on the starting curve and is NOT scored, exactly as in
    p12, so both runs give the tracker the same chance to converge before the
    event begins. Every later block is one generator sample held for `dwell`
    control steps, and all of them are scored.
    """
    lo, hi = int(pre_idx), int(post_idx)
    curves = [traj_full.curves[i + 1] for i in range(lo, hi + 1)]
    n_mid = len(curves) - 1
    if n_mid < 1:
        raise ValueError("gradual window needs at least two curves")

    block_steps = np.array([SETTLE_STEPS] + [dwell] * n_mid, int)
    scored = np.array([False] + [True] * n_mid, bool)

    # The oracle, per scored step: the true global peak of the curve in force.
    # Built here rather than in the metric so it cannot drift out of step with
    # the block layout.
    p_oracle = np.repeat([float(c[3]) for c in curves[1:]], dwell)
    regions = np.repeat([int(per_step["regions"][i])
                         for i in range(lo + 1, hi + 1)], dwell)

    module, geometry = traj_full.module, traj_full.geometry

    def build():
        t = p12.NoisyTrajectory(
            curves=curves, block_steps=block_steps, scored=scored,
            profile_name=traj_full.profile_name, module=module,
            geometry=geometry)
        return t.configure_noise(p12.NOISE_I_RMS, p12.NOISE_V_RMS,
                                 np.random.default_rng(win_seed))

    # Same difficulty measure as p12, on the same endpoints, so the two runs are
    # directly comparable: how much of the FINAL peak a tracker still sitting at
    # the STARTING peak's voltage would be missing.
    first, last = curves[0], curves[-1]
    V, P = np.asarray(last[0], float), np.asarray(last[1], float)
    v_start = float(first[2])
    order = np.argsort(V)
    p_held = float(np.interp(v_start, V[order], P[order]))
    p_final = float(last[3])

    meta = {"module": module, "pre_idx": lo, "post_idx": hi,
            "n_curves": len(curves), "dwell": int(dwell),
            "scored_steps": int(n_mid * dwell),
            "span_seconds": float(n_mid * 30.0),
            "p_gmpp_start": float(first[3]), "p_gmpp_final": p_final,
            "v_gmpp_start": v_start, "p_final_at_start_v": p_held,
            "drop_depth": float(1.0 - p_held / max(p_final, 1e-9)),
            "region_pre": int(per_step["regions"][lo]),
            "region_post": int(per_step["regions"][hi])}
    return build, meta, p_oracle, regions


# ---------------------------------------------------------------------------
# METRICS  (the target moves, so p12's scalar versions do not apply)
# ---------------------------------------------------------------------------

def gradual_energy_loss_pct(traj, p_oracle):
    """Energy missed against the per-step true global peak."""
    p = np.asarray(traj.p_hist, float)[SETTLE_STEPS:]
    n = min(len(p), len(p_oracle))
    avail = float(p_oracle[:n].sum())
    return float(100.0 * (avail - float(p[:n].sum())) / max(avail, 1e-9))


def off_peak_fraction(traj, p_oracle):
    """Share of the scored span spent below (1-EPS) of the CURRENT peak."""
    p = np.asarray(traj.p_hist, float)[SETTLE_STEPS:]
    n = min(len(p), len(p_oracle))
    if n == 0:
        return float("nan")
    return float(np.mean(p[:n] < (1.0 - EPS) * p_oracle[:n]))


def arrived_at_end(traj, p_oracle):
    """Is it on the FINAL peak for HOLD consecutive steps at the end?"""
    p = np.asarray(traj.p_hist, float)[SETTLE_STEPS:]
    n = min(len(p), len(p_oracle))
    if n < HOLD:
        return False
    tail = p[n - HOLD:n]
    return bool(np.all(tail >= (1.0 - EPS) * p_oracle[n - 1]))


def first_on_peak_step(traj, p_oracle):
    """First scored step from which it holds the CURRENT peak to the end.

    A gradual run has no single 'jump', so 'how long did it take' is read as
    'from where was it never wrong again'. Returns None if that never happens.
    """
    p = np.asarray(traj.p_hist, float)[SETTLE_STEPS:]
    n = min(len(p), len(p_oracle))
    ok = p[:n] >= (1.0 - EPS) * p_oracle[:n]
    if not ok.any():
        return None
    # last index that is NOT ok; everything after it is ok
    bad = np.flatnonzero(~ok)
    if len(bad) == 0:
        return 0
    last_bad = int(bad[-1])
    return (last_bad + 1) if last_bad + 1 < n else None


# ---------------------------------------------------------------------------
# THE COMPOSITE POLICY  (what the two-regime result dictates)
# ---------------------------------------------------------------------------

class ClockFloorPolicy(ReseedPolicy):
    """Detection, with a slow clock as a floor underneath it.

    Fires when EITHER the detector confirms a drop or the clock comes round.
    The two failure modes are complementary and both are measured:

      the detector  fires within a few steps of a fast relocation and never
                    on a quiet window (0 of 60 false alarms on the uniform
                    control), but is structurally blind to a relocation
                    slower than its own EMA memory -- 0 fires on 12 of 12
                    gradual windows, at every published threshold.

      the clock     cannot miss anything, because it does not look; it pays
                    the re-seed cost every period whether or not the operating
                    point moved, and cannot respond faster than one period.

    Implementation notes that matter for fairness:

    * Both terms are evaluated on EVERY step and never short-circuited. The
      detector's `_step` has to run even when the clock has already fired, or
      its frozen-baseline state would drift and the composite would quietly
      be a worse detector than the detector alone.

    * The parent's re-arm dead time is SHARED. A clock tick and a detector
      confirmation landing within the same dead window cost one re-seed, not
      two, which is the physically honest accounting -- there is only one
      re-seed to pay for.

    * n_clock and n_detector attribute ACTUAL fires, not candidates: the
      attribution happens in `update`, after the parent has decided whether
      the step fired at all. Both counters can increment on one fire, when
      the clock and the detector agree, so their sum can exceed n_fires. That
      overlap is itself worth reporting -- it is the share of the clock's
      work the detector would have done anyway.

      The counters are a floor, not a census. A clock tick landing a few
      steps BEFORE a confirmation the detector was about to make absorbs it
      into the shared dead time, so the detector is credited with nothing
      even though it had found the event. Read a detector count of zero as
      "the clock got there first or there was nothing to find" -- which of
      the two it was comes from the detector-alone arm, not from here.

    * The policy is NOT re-reset after a fire, matching p12's bare triggered
      arm exactly: the detector's baseline carries across a re-seed in both.
      That staleness is a property of the published method, not of this
      composite, and it is left alone so the arms stay comparable.
    """

    name = "clock_floor"

    def __init__(self, detector, period, phase=0):
        super().__init__()
        self.detector = detector
        self.period = int(period)
        self.phase = int(phase) % max(int(period), 1)
        self.n_clock = 0
        self.n_detector = 0
        self._last_clk = False
        self._last_det = False

    def reset(self, p0=0.0, v0=0.0):
        super().reset(p0, v0)
        self.detector.reset(p0=p0, v0=v0)
        self.n_clock = 0
        self.n_detector = 0
        self._last_clk = False
        self._last_det = False

    def update(self, v, i, p):
        self._last_clk = False
        self._last_det = False
        fired = super().update(v, i, p)
        if fired:
            if self._last_clk:
                self.n_clock += 1
            if self._last_det:
                self.n_detector += 1
        return fired

    def _step(self, v, i, p):
        det = bool(self.detector._step(v, i, p))
        clk = self._k > 0 and (self._k - self.phase) % self.period == 0
        self._last_det = det
        self._last_clk = clk
        return det or clk

    def describe(self):
        return (f"clock_floor(period={self.period}, "
                f"det={self.detector.name})")


def rescale_phase(phase_base, period):
    """Rescale the main run's integer phase onto another period, EXACTLY.

    `phase_base` is the phase the main loop drew in [0, SCHEDULE_PERIOD_STEPS)
    -- Eltamaly's 240. Rescaling that single draw across the whole ladder is
    what stops a period from scoring well merely because its clock happened to
    land near the event: every period is scored at the same relative alignment
    on the same window.

    Done in INTEGER arithmetic, deliberately. The obvious form -- carry the
    phase as a fraction and multiply back up -- truncates downward whenever
    the division is inexact:

        int((123 / 240.0) * 240) == 122

    At the base period that is wrong for exactly one phase of 240, and across
    this ladder for 28 of 1,200 phase-period pairs. Rare is the problem: it
    would have surfaced only as a MISMATCH on the one window in 240 that drew
    phase 123, with nothing in the output to say why. Integer floor division
    is exact at the base period by construction and cannot drift at any other.
    """
    period = int(period)
    base = int(p12.SCHEDULE_PERIOD_STEPS)
    return (int(phase_base) * period // max(base, 1)) % max(period, 1)


def make_clock(period, phase_base):
    return ScheduledReseed(period=int(period),
                           phase=rescale_phase(phase_base, period))


def make_clock_floor(period, phase_base, eps):
    return ClockFloorPolicy(
        PowerDropDetector(eps=eps, persist=p12.TRIGGER_PERSIST,
                          alpha=p12.TRIGGER_EMA_ALPHA),
        period=int(period),
        phase=rescale_phase(phase_base, period))


def probes_per_day(n_probes, span_seconds):
    """Extrapolate a window's probe cost to a day of operation.

    This is the number the cost claim has to be made on. Inside a 1-2 hour
    window every curve really is moving, so a fast clock's probes are mostly
    useful and the cost looks mild. A real day is not a relocation window: it
    is mostly quiet, and a clock does not know that. Scaling the window rate
    to 24 h is a generous-to-the-clock estimate, since it assumes the whole
    day is as eventful as the window we selected FOR being eventful.
    """
    if span_seconds <= 0:
        return float("nan")
    return float(n_probes) * SECONDS_PER_DAY / float(span_seconds)


# ---------------------------------------------------------------------------
# ARMS  (a focused set -- this run answers one question)
# ---------------------------------------------------------------------------

def build_arms(model, eps=None, step_frac=DEFAULT_STEP_FRAC,
               competitors=()):
    eps = p12.TRIGGER_DROP_FRAC if eps is None else eps

    def po(traj, **_):
        p12.t_po(traj, step_frac=step_frac)

    def hyb_never(traj, **_):
        p12.t_hybrid_never(traj, model, 25.0, step_frac)

    def hyb_sched(traj, phase=0, **_):
        p12._t_hybrid_repolicy(traj, model, 25.0, step_frac,
                               p12._make_policy("scheduled", phase=phase))

    def hyb_trig(traj, **_):
        p12._t_hybrid_repolicy(traj, model, 25.0, step_frac,
                               p12._make_policy("power_drop", eps=eps))

    def hyb_ratchet(traj, **_):
        p12._t_hybrid_repolicy(traj, model, 25.0, step_frac,
                               p12._make_policy("ratchet", eps=eps))

    def pso_trig(traj, seed=0, **_):
        p12._t_pso_repolicy(traj, p12._make_policy("power_drop", eps=eps),
                            seed, DEFAULT_POPULATION, DEFAULT_ITERATIONS)

    def hyb_sched_nofine(traj, phase=0, **_):
        p12._t_estimator_nofine(traj, p12.est_learned_seed(model, 25.0),
                                p12._make_policy("scheduled", phase=phase),
                                step_frac)

    def hyb_trig_nofine(traj, **_):
        p12._t_estimator_nofine(traj, p12.est_learned_seed(model, 25.0),
                                p12._make_policy("power_drop", eps=eps),
                                step_frac)

    arms = [("P&O", po),
            ("hybrid, never reseed", hyb_never),
            ("hybrid, scheduled reseed", hyb_sched),
            ("hybrid, triggered reseed", hyb_trig),
            ("hybrid, triggered [ratchet]", hyb_ratchet),
            ("PSO, triggered reseed", pso_trig),
            # seed re-fired by a policy, with NO P&O between fires --
            # isolates the seed from the hill-climber
            ("hybrid, scheduled [no P&O]", hyb_sched_nofine),
            ("hybrid, triggered [no P&O]", hyb_trig_nofine)]

    # Published competitors, as ESTIMATORS under our policies -- which is what
    # isolates the estimator axis from the policy axis. Same construction as
    # p12's build_methods, so an arm means the same thing in both runs.
    for label, est in competitors:
        def mk_sched(e=est):
            def f(traj, phase=0, **_):
                p12._t_estimator_repolicy(
                    traj, e, p12._make_policy("scheduled", phase=phase),
                    step_frac)
            return f

        def mk_trig(e=est):
            def f(traj, **_):
                p12._t_estimator_repolicy(
                    traj, e, p12._make_policy("power_drop", eps=eps),
                    step_frac)
            return f
        arms.append((f"{label}, scheduled", mk_sched()))
        arms.append((f"{label}, triggered", mk_trig()))
    return arms


# ---------------------------------------------------------------------------
# SELECTION  (same scenarios as p12, by construction)
# ---------------------------------------------------------------------------

def select(args):
    """Reproduce p12's window selection so the scenario set is identical.

    Same seeded shuffle of the validation pool, same adjacent-transition
    selector, same physics fingerprint for de-duplication, same direction
    balancing. The fingerprints are recorded so the two runs can be joined
    scenario by scenario rather than compared in aggregate.
    """
    all_val, split_label = p12.load_val_modules(args.split, args.split_file)
    pool_rng = np.random.default_rng(int(config.seed_for("p12_pool")) % 2**31)
    all_val = list(all_val)
    pool_rng.shuffle(all_val)
    val_modules = all_val[:max(args.n * 12, args.n)]

    buckets, seen, flat = {}, set(), []
    discarded = duplicate = 0
    for mod in val_modules:
        if sum(len(v) for v in buckets.values()) >= args.n * 3:
            break
        seq = pattern_sequence(str(mod), 25.0, family=args.family)
        try:
            ok, traj_full, per_step, _ = validate_sequence(seq, verbose=False)
        except RuntimeError:
            discarded += 1
            continue
        if not ok:
            discarded += 1
            continue
        eps_list = p12.find_adjacent_transitions(per_step)
        if not eps_list:
            discarded += 1
            continue
        for ti, (pre_idx, post_idx, _ra, _rb) in enumerate(eps_list):
            widx = (int(config.seed_for("p12_noise")) % 2**31
                    + 1000 * len(flat) + ti)
            try:
                build, meta, p_or, regs = make_gradual_window(
                    traj_full, per_step, pre_idx, post_idx, widx, args.dwell)
            except ValueError:
                continue
            sig = (round(meta["p_gmpp_final"], 6),
                   round(meta["p_gmpp_start"], 6),
                   round(meta["v_gmpp_start"], 6),
                   round(meta["p_final_at_start_v"], 6),
                   meta["region_pre"], meta["region_post"])
            if sig in seen:
                duplicate += 1
                continue
            seen.add(sig)
            meta["scenario_sig"] = str(hash(sig) % 10**12)
            meta["transition_index"] = ti
            buckets.setdefault((meta["region_pre"], meta["region_post"]),
                               []).append((build, meta, p_or, regs))
            flat.append(1)

    if args.balance_directions and len(buckets) > 1:
        per = max(1, args.n // len(buckets))
        picked = []
        for d in sorted(buckets):
            picked.extend(buckets[d][:per])
        for d in sorted(buckets):
            if len(picked) >= args.n:
                break
            picked.extend(buckets[d][per:per + args.n - len(picked)])
        picked = picked[:args.n]
    else:
        picked = [w for d in sorted(buckets) for w in buckets[d]][:args.n]
    return picked, split_label, discarded, duplicate


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12,
                    help="gradual windows to draw. Each is 40k-86k control "
                         "steps per arm at the physical dwell, so this is the "
                         "knob to turn for run time -- never --dwell.")
    ap.add_argument("--family", default=p12.FAMILY_DEFAULT,
                    choices=list(TRANSITION_FAMILIES))
    ap.add_argument("--dwell", type=int, default=PHYSICAL_DWELL,
                    help=f"control steps per generator sample. The PHYSICAL "
                         f"value is {PHYSICAL_DWELL} (30 s sample / 0.1 s "
                         f"control). Lowering it speeds the relocation up, "
                         f"which is the independent variable -- a different "
                         f"experiment, not a cheaper one.")
    ap.add_argument("--sweep-eps", action="store_true",
                    help="re-run the triggered arm at every published dP/P. "
                         "On the spliced windows this was flat because the "
                         "events are 6-13x the largest threshold; here is "
                         "where it can actually bind.")
    ap.add_argument("--sweep-schedule", action="store_true",
                    help="sweep the scheduled re-seed PERIOD, with a "
                         "clock-only arm and a clock-floor-plus-detector arm "
                         "at each period. Reports the probe cost per day as "
                         "well as per window, and finds the knee -- the "
                         "longest period that gives up no measurable energy. "
                         "This is how the clock period becomes a DERIVED "
                         "parameter instead of Eltamaly's inherited 24 s.")
    ap.add_argument("--schedule-periods", type=int, nargs="+", default=None,
                    metavar="STEPS",
                    help=f"periods for --sweep-schedule, in control steps at "
                         f"0.1 s. Default {SCHEDULE_LADDER} "
                         f"(24 s, 1 m, 5 m, 10 m, 30 m).")
    ap.add_argument("--knee-tol", type=float, default=0.50,
                    help="percentage POINTS of extra energy loss tolerated "
                         "when choosing the knee. The knee is the longest "
                         "period within this much of the best period's loss. "
                         "Declared before the run, not chosen after it.")
    ap.add_argument("--sweep-competitors", action="store_true",
                    help="also sweep the published competitors across "
                         "--schedule-periods, scheduled policy only. Requires "
                         "--with-competitors and --sweep-schedule. Without it "
                         "the competitors are scored at the base 24 s period "
                         "only, which is not a fair ladder comparison.")
    ap.add_argument("--with-competitors", action="store_true",
                    help="add the published competitors as estimator arms, "
                         "each under BOTH a scheduled and a triggered policy. "
                         "The scheduled arm is the faithful one for a method "
                         "that re-probes continuously; the triggered arm "
                         "lends them our detector and is generous. Report "
                         "both, always.")
    ap.add_argument("--jamaa-model", default=None, metavar="PATH",
                    help="saved Jamaa net (results/phase3/jamaa_net.joblib). "
                         "Without it the jamaa arm is SKIPPED WITH A MESSAGE "
                         "rather than silently dropped -- see "
                         "save_jamaa_net.md for the three lines that write "
                         "it.")
    ap.add_argument("--split", default="design", choices=["design", "name"])
    ap.add_argument("--split-file", default=None)
    ap.add_argument("--model-tag", default="c3_two_stage_design")
    ap.add_argument("--no-balance-directions", dest="balance_directions",
                    action="store_false")
    ap.add_argument("--voc-source", choices=["curve", "measured", "datasheet"],
                    default="curve",
                    help="where the seed's V_oc comes from. Threaded through "
                         "p12.VOC_SOURCE, which every arm here inherits "
                         "because they all build on p12.est_learned_seed. "
                         "curve: the simulated characteristic, which every "
                         "result before 26 Oct 2026 used. measured: the same "
                         "true value, charged as one open-circuit step. "
                         "datasheet: the nameplate value corrected to the "
                         "module temperature.")
    args = ap.parse_args()

    # p13 has no seed call sites of its own: every arm reaches the seed
    # through p12.est_learned_seed, which reads p12.VOC_SOURCE. Setting it
    # here is the whole of the wiring.
    p12.VOC_SOURCE = args.voc_source

    rng = np.random.default_rng(int(config.seed_for("p13_stats")) % 2**31)

    print("P2.9c  GRADUAL relocation -- the same events, walked not spliced")
    try:
        print("  " + config.provenance())
    except Exception:
        pass
    print("=" * 78)
    print(f"family {args.family}; settle {SETTLE_STEPS} steps, then every "
          f"curve held {args.dwell} steps at 100 ms")
    if args.dwell != PHYSICAL_DWELL:
        print(f"   *** NON-PHYSICAL DWELL ({args.dwell} vs {PHYSICAL_DWELL}) "
              f"-- this speeds the relocation up by "
              f"{PHYSICAL_DWELL/max(args.dwell,1):.1f}x.")
        print("   *** It is a DIFFERENT EXPERIMENT, not a faster one. Label "
              "any figure accordingly.")
    print(f"trigger: power_drop, dP/P = {p12.TRIGGER_DROP_FRAC:.2f}, "
          f"EMA alpha {p12.TRIGGER_EMA_ALPHA:.2f}, persist "
          f"{p12.TRIGGER_PERSIST}")
    print(f"schedule: every {p12.SCHEDULE_PERIOD_S:.0f} s = "
          f"{p12.SCHEDULE_PERIOD_STEPS} steps (Eltamaly T_PD), phase "
          f"randomised")
    print("DECLARED SYNTHETIC site (Jeju reference), pending Nanoom Energy "
          "geometry.")
    print(f"MEASUREMENT NOISE paired: current "
          f"{100*p12.NOISE_I_RMS:.1f}% RMS, voltage "
          f"{100*p12.NOISE_V_RMS:.1f}% RMS.\n")
    print("PREDICTION, on record before this run:")
    print("   the power-drop detector fires on 0% of gradual relocations, at")
    print("   every threshold in the published range 0.05-0.20. If it holds, a")
    print("   CLOCK is the correct answer in this regime and the result is a")
    print("   REGIME BOUNDARY, not a defeat.\n")

    try:
        model = TwoStageModel.load(tag=args.model_tag)
    except FileNotFoundError:
        print(f"model {args.model_tag!r} not found.")
        return 1

    picked, split_label, discarded, duplicate = select(args)
    if len(picked) < 2:
        print(f"only {len(picked)} gradual windows ({discarded} discarded). "
              f"Widen --n.")
        return 1

    print(f"split: {split_label}\n   model tag {args.model_tag}")
    spans = [m["span_seconds"] / 3600.0 for _b, m, _o, _r in picked]
    steps = [m["scored_steps"] for _b, m, _o, _r in picked]
    from collections import Counter
    cnt = Counter((m["region_pre"], m["region_post"]) for _b, m, _o, _r in picked)
    print(f"\n{len(picked)} GRADUAL windows, {discarded} discarded by gates, "
          f"{duplicate} duplicates rejected")
    print(f"   directions: "
          f"{', '.join(f'{a}->{b} (n={cnt[(a,b)]})' for a, b in sorted(cnt))}")
    print(f"   span {min(spans):.2f}-{max(spans):.2f} h of real sun movement "
          f"per window")
    print(f"   scored steps {min(steps):,}-{max(steps):,} per arm per window")
    dd = [m["drop_depth"] for _b, m, _o, _r in picked]
    print(f"   drop depth (same measure as p12): median {np.median(dd):.3f}, "
          f"range {min(dd):.3f}-{max(dd):.3f}\n")

    competitors = []
    if args.with_competitors:
        print("\ncompetitors (each run under BOTH policies):")
        competitors = p12.load_competitors(args.jamaa_model)
        for lab, _ in competitors:
            print(f"   [OK] {lab}")
        if not competitors:
            print("   none loaded -- the gradual table will have NO published")
            print("   learned estimator in it. That is the comparison a "
                  "reviewer")
            print("   asks for first; do not write the section without it.")
        if args.with_competitors and not args.jamaa_model:
            print("   note: --jamaa-model not given, so the jamaa arm cannot "
                  "load.")

    arms = build_arms(model, competitors=competitors)
    names = [n for n, _ in arms]
    per_arm = {n: {"energy": [], "off_peak": [], "arrived": [],
                   "fires": [], "n_fires": [], "settle_step": [],
                   "probes": []}
               for n in names}
    phase_bases = []

    for wi, (build, meta, p_or, _regs) in enumerate(picked):
        phase = int(rng.integers(0, p12.SCHEDULE_PERIOD_STEPS))
        # This SAME integer draw is what the period sweep rescales -- so the
        # main arm below is bit-identical to the pre-patch run and the sweep
        # is phase-matched to it rather than re-randomised. Kept as an int,
        # never a fraction; see rescale_phase for why that matters.
        phase_bases.append(phase)
        print(f"   window {wi+1}/{len(picked)}  {meta['module'][:38]:<38} "
              f"{meta['region_pre']}->{meta['region_post']}  "
              f"{meta['scored_steps']:,} steps", flush=True)
        for name, fn in arms:
            traj = build()
            fn(traj, phase=phase, seed=wi)
            fires = getattr(traj, "trigger_fires", None) or []
            per_arm[name]["energy"].append(gradual_energy_loss_pct(traj, p_or))
            per_arm[name]["off_peak"].append(off_peak_fraction(traj, p_or))
            per_arm[name]["arrived"].append(arrived_at_end(traj, p_or))
            per_arm[name]["fires"].append([int(x) for x in fires])
            per_arm[name]["n_fires"].append(len(fires))
            per_arm[name]["settle_step"].append(first_on_peak_step(traj, p_or))
            per_arm[name]["probes"].append(
                int(getattr(traj, "estimator_probes", 0) or 0))

    # -- table ---------------------------------------------------------------
    print("\n" + "=" * 78)
    print("GRADUAL RELOCATION  (per-step oracle; the target moves every block)")
    print(f"   {'arm':<30} {'fires':>7} {'energy %':>9} {'off-peak %':>11} "
          f"{'arrived':>8}")
    print("   " + "-" * 70)
    stats = {}
    for n in names:
        a = per_arm[n]
        nf = float(np.mean(a["n_fires"]))
        en = float(np.mean(a["energy"]))
        op = 100.0 * float(np.mean(a["off_peak"]))
        ar = 100.0 * float(np.mean(a["arrived"]))
        stats[n] = {"mean_fires": nf, "energy_loss_pct": en,
                    "off_peak_pct": op, "arrived_pct": ar,
                    "energy": a["energy"], "off_peak": a["off_peak"],
                    "arrived": a["arrived"], "n_fires": a["n_fires"]}
        print(f"   {n:<30} {nf:>7.2f} {en:>9.2f} {op:>11.1f} {ar:>7.0f}%")

    # -- the verdict ---------------------------------------------------------
    trig = stats["hybrid, triggered reseed"]
    sched = stats["hybrid, scheduled reseed"]
    ratch = stats["hybrid, triggered [ratchet]"]
    total_fires = int(sum(per_arm["hybrid, triggered reseed"]["n_fires"]))

    print("\n" + "=" * 78)
    print("VERDICT")
    print(f"\n   power-drop detector fired {total_fires} times across "
          f"{len(picked)} gradual windows")
    if total_fires == 0:
        print("   -> PREDICTION HELD, exactly. The detector is blind to a "
              "relocation")
        print("      slower than its own baseline memory. This is structural: "
              "the EMA")
        print("      sinks with the signal, so there is no step for a "
              "difference test")
        print("      to find.")
    else:
        print(f"   -> PREDICTION DID NOT HOLD. It fired "
              f"{total_fires/len(picked):.2f} times per window on average.")
        print("      Find out WHY before changing anything: check whether the "
              "fires sit")
        print("      at the bypass-diode handover, where the curve can move "
              "faster than")
        print("      the band does.")

    print(f"\n   clock vs detection in this regime:")
    print(f"      scheduled re-seed   {sched['energy_loss_pct']:6.2f}% lost, "
          f"{sched['arrived_pct']:.0f}% arrived, "
          f"{sched['mean_fires']:.1f} fires")
    print(f"      triggered re-seed   {trig['energy_loss_pct']:6.2f}% lost, "
          f"{trig['arrived_pct']:.0f}% arrived, "
          f"{trig['mean_fires']:.1f} fires")
    if sched["energy_loss_pct"] < trig["energy_loss_pct"]:
        print("      -> the CLOCK wins here, which is the honest two-regime "
              "result and")
        print("         the design argument for a clock FLOOR underneath the "
              "detector.")
    else:
        print("      -> the clock does NOT win here. Report it; the "
              "two-regime story")
        print("         does not survive as stated.")

    print(f"\n   ratchet detector (bench predicted it catches the gradual "
          f"case):")
    print(f"      {ratch['mean_fires']:.1f} fires, "
          f"{ratch['energy_loss_pct']:.2f}% lost, "
          f"{ratch['arrived_pct']:.0f}% arrived")
    print("      On the SPLICED windows the ratchet tied us on performance "
          "and failed")
    print("      the uniform control 23/60. If it rescues the gradual case "
          "here, the")
    print("      trade is explicit: a detector that sees slow change is a "
          "detector")
    print("      that fires when nothing moved.")

    # -- probe cost, per window and extrapolated to a day --------------------
    print("\n" + "=" * 78)
    print("PROBE COST  (the same fires, priced over a window and over a day)")
    print("   A window is eventful by construction, so a fast clock's probes")
    print("   look cheap here. A day is mostly quiet and the clock cannot "
          "tell.")
    print(f"\n   {'arm':<30} {'probes/win':>11} {'probes/day':>11} "
          f"{'energy %':>9}")
    print("   " + "-" * 70)
    for n in names:
        a = per_arm[n]
        pw = float(np.mean(a["probes"])) if a["probes"] else float("nan")
        pd_ = float(np.mean([
            probes_per_day(pr, m["span_seconds"])
            for pr, (_b, m, _o, _r) in zip(a["probes"], picked)])) \
            if a["probes"] else float("nan")
        stats[n]["probes_per_window"] = pw
        stats[n]["probes_per_day"] = pd_
        print(f"   {n:<30} {pw:>11,.0f} {pd_:>11,.0f} "
              f"{stats[n]['energy_loss_pct']:>9.2f}")
    print("\n   Read the last two columns together with the energy column: the")
    print("   clock buys its 1-point energy advantage with a probe bill that")
    print("   is paid all day, which is exactly the trade the period sweep")
    print("   below is for.")

    # -- epsilon sweep -------------------------------------------------------
    sweeps = {}
    if args.sweep_eps:
        print("\n" + "=" * 78)
        print("dP/P SWEEP on the GRADUAL windows")
        print("   On the spliced windows this was flat, because the events are")
        print("   6-13x the largest published threshold. Here the event is a "
              "ramp,")
        print("   so this is where a threshold could actually bind.")
        print(f"\n   {'eps':>6} {'source':<44} {'fires':>7} {'energy %':>9}")
        print("   " + "-" * 70)
        for e, src in p12.EPS_PUBLISHED:
            arms_e = dict(build_arms(model, eps=e))
            fn = arms_e["hybrid, triggered reseed"]
            fires, en = [], []
            for build, meta, p_or, _r in picked:
                traj = build()
                fn(traj)
                fires.append(len(getattr(traj, "trigger_fires", None) or []))
                en.append(gradual_energy_loss_pct(traj, p_or))
            sweeps[f"eps_{e}"] = {"fires": fires, "energy": en}
            print(f"   {e:>6.2f} {src[:44]:<44} "
                  f"{float(np.mean(fires)):>7.2f} {float(np.mean(en)):>9.2f}")

    # -- scheduled-period sweep ----------------------------------------------
    sched_sweep = {}
    # Knee summaries live in their OWN dict. They used to be written back into
    # sched_sweep, which is also what the knee loop iterates over to collect
    # each variant's rows -- so the first variant's summary became the second
    # variant's KeyError. Never mutate the dict you are reading.
    knee_rows = {}
    if args.sweep_schedule:
        periods = args.schedule_periods or list(SCHEDULE_LADDER)
        periods = sorted({int(x) for x in periods if int(x) > 0})
        print("\n" + "=" * 78)
        print("SCHEDULED RE-SEED PERIOD SWEEP")
        if getattr(args, "sweep_competitors", False) and competitors:
            print("\n   PREDICTION, on record before this run:")
            print("      a longer interval helps the PUBLISHED competitors")
            print("      too, by a similar margin. The interval is a property")
            print("      of the re-seeding POLICY, not of our predictor, so")
            print("      the ordering of periods should be the same for every")
            print("      estimator. If instead it helps them markedly more or")
            print("      less, there is a predictor-interval interaction and")
            print("      the \"works with any predictor\" claim comes out.")
            print("      Competitors are swept under the SCHEDULED policy only:")
            print("      the detector fires 0 times here for every arm, so a")
            print("      triggered competitor returns its never-reseed value")
            print("      at every period.")
        print("   The clock wins this regime, so the period is a real design")
        print("   parameter. 240 steps is Eltamaly's 24 s -- inherited, not")
        print("   derived. The events here take 1.1-2.4 h, so a period of")
        print("   MINUTES should still sample them many times over.")
        print("   Phase is rescaled from the main run's own draw, so every")
        print("   period is scored at the same relative alignment.")
        print(f"\n   {'period':>8} {'':<6} {'arm':<22} {'fires':>7} "
              f"{'probes/day':>11} {'energy %':>9} {'arrived':>8}")
        print("   " + "-" * 76)

        for period in periods:
            secs = period * 0.1
            label = (f"{secs:.0f} s" if secs < 90
                     else f"{secs / 60:.0f} m" if secs < 5400
                     else f"{secs / 3600:.1f} h")
            # (name, policy factory, run function). The run function is new:
            # the sweep used to call _t_hybrid_repolicy directly with our
            # model hardcoded, which is why no competitor could be swept.
            def _run_ours(traj, pol):
                p12._t_hybrid_repolicy(traj, model, 25.0,
                                       DEFAULT_STEP_FRAC, pol)

            variants = [
                ("clock only",
                 lambda pb, _p=period: make_clock(_p, pb),
                 _run_ours),
                ("clock floor + detector",
                 lambda pb, _p=period: make_clock_floor(
                     _p, pb, p12.TRIGGER_DROP_FRAC),
                 _run_ours),
            ]
            if getattr(args, "sweep_competitors", False) and competitors:
                # Scheduled policy only. The detector fires 0 times on gradual
                # windows for every arm, so a triggered competitor returns its
                # never-reseed value at every period -- no information, hours
                # of compute. Same estimator path build_arms uses, so an arm
                # means the same thing here as in the main table.
                for _label, _est in competitors:
                    def _run_comp(traj, pol, _e=_est):
                        p12._t_estimator_repolicy(traj, _e, pol,
                                                  DEFAULT_STEP_FRAC)
                    variants.append(
                        (f"{_label}, clock only",
                         lambda pb, _p=period: make_clock(_p, pb),
                         _run_comp))
            for vname, mk, _run in variants:
                en, op, ar, nf, pr, prd = [], [], [], [], [], []
                n_clk, n_det = [], []
                lastgap = []
                for wi, (build, meta, p_or, _r) in enumerate(picked):
                    traj = build()
                    pol = mk(phase_bases[wi])
                    _run(traj, pol)
                    fires = getattr(traj, "trigger_fires", None) or []
                    probes = int(getattr(traj, "estimator_probes", 0) or 0)
                    en.append(gradual_energy_loss_pct(traj, p_or))
                    op.append(off_peak_fraction(traj, p_or))
                    ar.append(arrived_at_end(traj, p_or))
                    nf.append(len(fires))
                    pr.append(probes)
                    prd.append(probes_per_day(probes, meta["span_seconds"]))
                    # Steps between the LAST re-seed and the end of the span.
                    # Arrival is an endpoint test, so if it is an artifact of
                    # where the window was cut rather than a property of the
                    # period, this is the quantity that will say so.
                    n_total = int(getattr(traj, "n_steps",
                                          SETTLE_STEPS + meta["scored_steps"]))
                    lastgap.append(int(n_total - fires[-1]) if fires
                                   else int(n_total))
                    if isinstance(pol, ClockFloorPolicy):
                        n_clk.append(pol.n_clock)
                        n_det.append(pol.n_detector)
                rec = {"period_steps": period, "period_seconds": secs,
                       "variant": vname,
                       "estimator": ("ours" if vname.startswith("clock")
                                     else vname.rsplit(",", 1)[0]),
                       "mean_fires": float(np.mean(nf)),
                       "mean_probes": float(np.mean(pr)),
                       "mean_probes_per_day": float(np.mean(prd)),
                       "energy_loss_pct": float(np.mean(en)),
                       "off_peak_pct": 100.0 * float(np.mean(op)),
                       "arrived_pct": 100.0 * float(np.mean(ar)),
                       "energy": en, "n_fires": nf, "probes": pr,
                       "last_gap": lastgap,
                       "mean_last_gap_steps": float(np.mean(lastgap)),
                       "arrived_flags": [bool(x) for x in ar]}
                if n_clk:
                    rec["mean_fires_from_clock"] = float(np.mean(n_clk))
                    rec["mean_fires_from_detector"] = float(np.mean(n_det))
                key = f"period_{period}_{vname.replace(' ', '_')}"
                sched_sweep[key] = rec
                print(f"   {period:>8} {label:<6} {vname:<22} "
                      f"{rec['mean_fires']:>7.2f} "
                      f"{rec['mean_probes_per_day']:>11,.0f} "
                      f"{rec['energy_loss_pct']:>9.2f} "
                      f"{rec['arrived_pct']:>7.0f}%")

        # -- self-check: at 240 the clock-only arm IS the main scheduled arm
        base = sched_sweep.get("period_240_clock_only")
        if base is not None:
            d_en = abs(base["energy_loss_pct"] - sched["energy_loss_pct"])
            d_nf = abs(base["mean_fires"] - sched["mean_fires"])
            ok = d_en < 1e-6 and d_nf < 1e-6
            print(f"\n   self-check at 240 steps vs the main scheduled arm: "
                  f"{'MATCH' if ok else 'MISMATCH'} "
                  f"(d energy {d_en:.2e}, d fires {d_nf:.2e})")
            if not ok:
                print("   -> the sweep is NOT scoring the same clock as the "
                      "main table.")
                print("      Do not read the ladder until that is explained; "
                      "a phase or")
                print("      step-size mismatch would make every row "
                      "incomparable.")

        # -- the knee ---------------------------------------------------------
        for vname in ("clock only", "clock floor + detector"):
            rows = [r for r in sched_sweep.values()
                    if r.get("variant") == vname]
            if not rows:
                continue
            rows.sort(key=lambda r: r["period_steps"])
            best = min(rows, key=lambda r: r["energy_loss_pct"])
            within = [r for r in rows
                      if r["energy_loss_pct"]
                      <= best["energy_loss_pct"] + args.knee_tol]
            knee = max(within, key=lambda r: r["period_steps"])
            saving = (best["mean_probes_per_day"]
                      / max(knee["mean_probes_per_day"], 1e-9))
            # Monotone, or an interior optimum? The distinction decides
            # whether "longest period within tolerance" is a knee rule at all.
            e_seq = [r["energy_loss_pct"] for r in rows]
            interior = (best["period_steps"] != rows[0]["period_steps"]
                        and best["period_steps"] != rows[-1]["period_steps"])
            print(f"\n   ENERGY vs PERIOD, {vname}  --  "
                  f"{'INTERIOR OPTIMUM' if interior else 'monotone over the ladder'}")
            print("      " + "   ".join(
                f"{r['period_seconds']:.0f}s {r['energy_loss_pct']:.2f}%"
                for r in rows))
            if interior:
                print("      Both ends cost something: too fast spends energy "
                      "on re-seed")
                print("      probes, too slow leaves the tracker stale. So "
                      "this is NOT a")
                print("      knee -- report the curve and name the optimum, "
                      "and state the")
                print("      deployment choice separately as the trade it is.")
            print(f"   operating point (tolerance {args.knee_tol:.2f} pt, "
                  f"declared before the run)")
            print(f"      best energy   {best['energy_loss_pct']:.2f}% at "
                  f"{best['period_steps']} steps "
                  f"({best['period_seconds']:.0f} s), "
                  f"{best['mean_probes_per_day']:,.0f} probes/day")
            print(f"      chosen        {knee['energy_loss_pct']:.2f}% at "
                  f"{knee['period_steps']} steps "
                  f"({knee['period_seconds']:.0f} s), "
                  f"{knee['mean_probes_per_day']:,.0f} probes/day")
            print(f"      -> the chosen point costs "
                  f"{knee['energy_loss_pct'] - best['energy_loss_pct']:+.2f} pt "
                  f"of energy and saves {saving:.1f}x the probes.")
            if knee["period_steps"] > 240:
                print("      It is SLOWER than Eltamaly's 24 s on our events, "
                      "which is the")
                print("      expected direction: their clock was set for a "
                      "step change, ours")
                print("      for a relocation that takes hours. Whether that "
                      "is a DERIVED")
                print("      period or only an energy-optimal one depends on "
                      "the arrival")
                print("      lines below -- do not promote it until those "
                      "agree.")
            elif knee["period_steps"] == 240:
                print("      It LANDS on Eltamaly's 24 s. Report that -- an "
                      "inherited")
                print("      parameter that survives being re-derived is a "
                      "result, and a")
                print("      better one than replacing it.")
            else:
                print("      It is FASTER than 24 s, which the 1.1-2.4 h "
                      "event timescale")
                print("      does not predict. Treat it as a bug until "
                      "explained.")
            # Energy and arrival can pick different periods. If they do,
            # neither is "the" derived period: the choice is an objective
            # question, not a measurement, and quoting whichever column
            # flatters the result is how an overclaim gets written.
            a_best = max(rows, key=lambda r: r["arrived_pct"])
            grain = 100.0 / max(len(picked), 1)
            print(f"      arrival at the optimum {best['arrived_pct']:.0f}%, "
                  f"best anywhere {a_best['arrived_pct']:.0f}% at "
                  f"{a_best['period_steps']} steps; one window = "
                  f"{grain:.1f} pt of that column")
            if a_best["period_steps"] != best["period_steps"]:
                print("      DISAGREEMENT: energy and arrival do not pick the "
                      "same period.")
                print("      They measure different things -- energy "
                      "integrates over the")
                print("      whole span, arrival is an endpoint test -- so "
                      "read the last-gap")
                print("      diagnostic below before treating arrival as a "
                      "property of the")
                print("      period at all. Do not name a derived period off "
                      "one column.")
            a_seq = [r["arrived_pct"] for r in rows]
            if not all(b <= a + 1e-9 for a, b in zip(a_seq, a_seq[1:])):
                swing = (max(a_seq) - min(a_seq)) / max(grain, 1e-9)
                print(f"      Arrival is NON-monotone across the ladder, and "
                      f"its whole range")
                print(f"      is {swing:.0f} window(s). At this sample size "
                      f"that is noise until")
                print(f"      shown otherwise -- raise --n before reading a "
                      f"trend into it.")
            knee_rows[f"knee_{vname.replace(' ', '_')}"] = {
                "arrived_pct_at_optimum": best["arrived_pct"],
                "best_arrival_period_steps": a_best["period_steps"],
                "best_arrival_pct": a_best["arrived_pct"],
                "energy_curve_interior_optimum": bool(interior),
                "energy_curve": e_seq,
                "arrival_curve": a_seq,
                "arrival_points_per_window": grain,
                "period_steps": knee["period_steps"],
                "period_seconds": knee["period_seconds"],
                "energy_loss_pct": knee["energy_loss_pct"],
                "best_period_steps": best["period_steps"],
                "best_energy_loss_pct": best["energy_loss_pct"],
                "knee_tol_points": float(args.knee_tol),
                "probe_saving_vs_best": float(saving)}

        comp = [r for r in sched_sweep.values()
                if r.get("variant") == "clock floor + detector"
                and "mean_fires_from_detector" in r]
        if comp:
            det_share = sum(r["mean_fires_from_detector"] for r in comp)
            print(f"\n   detector contribution inside the composite: "
                  f"{det_share:.2f} fires summed over the whole ladder")
            if det_share == 0:
                print("      ZERO. On gradual windows the composite is the "
                      "clock and nothing")
                print("      else -- consistent with the 0-fire result, and "
                      "the reason the")
                print("      composite is worth having is the SPLICED regime, "
                      "where the")
                print("      detector fires in 9 steps and the clock would "
                      "wait a period.")
                print("      Say both halves or neither.")
                print("      Caveat on this zero: a clock tick that lands "
                      "just before a")
                print("      confirmation absorbs it into the shared dead "
                      "time, so the")
                print("      detector is credited with nothing either way. "
                      "The 0-fire")
                print("      detector-alone arm above is what rules out the "
                      "second reading.")

        # -- is the arrival column an artifact of where the window ends? --
        rows_c = [r for r in sched_sweep.values()
                  if r.get("variant") == "clock only"
                  and "last_gap" in r]
        if rows_c:
            print("\n   LAST-RESEED-TO-END GAP  (is arrival a property of the "
                  "period,")
            print("   or of where the window was cut?)")
            print("      Arrival asks whether the tracker is on the FINAL peak "
                  "at the end")
            print("      of the span. A re-seed long before that end leaves "
                  "time to drift")
            print("      off again, so if the missed windows are the "
                  "long-trailing-gap")
            print("      ones, the column is about the cut and not about the "
                  "clock.")
            print(f"\n      {'period':>8}  {'mean gap':>9}  "
                  f"{'gap|arrived':>12}  {'gap|missed':>11}  {'n missed':>8}")
            print("      " + "-" * 60)
            hit_all, miss_all = [], []
            for r in sorted(rows_c, key=lambda r: r["period_steps"]):
                g = np.asarray(r["last_gap"], float)
                a = np.asarray(r["arrived_flags"], bool)
                hit = g[a]
                miss = g[~a]
                hit_all.extend(hit.tolist())
                miss_all.extend(miss.tolist())
                r["mean_gap_arrived"] = (float(np.mean(hit)) if hit.size
                                         else float("nan"))
                r["mean_gap_missed"] = (float(np.mean(miss)) if miss.size
                                        else float("nan"))
                print(f"      {r['period_steps']:>8}  "
                      f"{float(np.mean(g)):>9,.0f}  "
                      f"{(float(np.mean(hit)) if hit.size else float('nan')):>12,.0f}  "
                      f"{(float(np.mean(miss)) if miss.size else float('nan')):>11,.0f}  "
                      f"{int(miss.size):>8}")
            if hit_all and miss_all:
                mh, mm = float(np.mean(hit_all)), float(np.mean(miss_all))
                ratio = mm / max(mh, 1e-9)
                print(f"\n      pooled over the ladder: arrived {mh:,.0f} "
                      f"steps, missed {mm:,.0f} steps "
                      f"({ratio:.2f}x)")
                if ratio >= 1.5:
                    print("      -> the missed windows are the ones whose last "
                          "re-seed was long")
                    print("         before the span ended. Arrival is "
                          "substantially an artifact")
                    print("         of the window cut; ENERGY LOSS is the "
                          "metric for this")
                    print("         regime, and the arrival column should be "
                          "reported as a")
                    print("         diagnostic with that caveat attached, not "
                          "as a result.")
                elif ratio <= 1.0 / 1.5:
                    print("      -> missed windows had SHORTER trailing gaps, "
                          "which the artifact")
                    print("         story does not predict. Something else is "
                          "going on; find")
                    print("         out what before using either column.")
                else:
                    print("      -> the trailing gap does not separate arrived "
                          "from missed, so")
                    print("         arrival is NOT merely an artifact of the "
                          "cut. A slow clock")
                    print("         really does leave the tracker off-peak at "
                          "the end, and the")
                    print("         period recommendation has to carry that "
                          "cost explicitly.")
            else:
                print("\n      every window on one side; no comparison "
                      "possible at this --n.")

    payload = {
        "runner": "p13_gradual_sweep", "family": args.family,
        "n_windows": len(picked), "dwell": args.dwell,
        "physical_dwell": PHYSICAL_DWELL,
        "dwell_is_physical": args.dwell == PHYSICAL_DWELL,
        "split": args.split, "split_description": split_label,
        "model_tag": args.model_tag, "settle_steps": SETTLE_STEPS,
        "voc_source": args.voc_source,
        "trigger_drop_frac": p12.TRIGGER_DROP_FRAC,
        "trigger_persist": p12.TRIGGER_PERSIST,
        "trigger_ema_alpha": p12.TRIGGER_EMA_ALPHA,
        "schedule_period_steps": p12.SCHEDULE_PERIOD_STEPS,
        "noise": {"i_rms": p12.NOISE_I_RMS, "v_rms": p12.NOISE_V_RMS},
        "window_meta": [m for _b, m, _o, _r in picked],
        "per_arm": {n: per_arm[n] for n in names},
        "stats": stats, "sweeps": sweeps,
        "competitors": [lab for lab, _ in competitors],
        "jamaa_model": args.jamaa_model,
        "schedule_sweep": sched_sweep,
        "schedule_knees": knee_rows,
        "schedule_ladder_default": list(SCHEDULE_LADDER),
        "knee_tol_points": float(args.knee_tol),
        "window_phase_base": phase_bases,
        "prediction_held": total_fires == 0}
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"gradual_sweep_{args.family}.json"
    # Archive copy. This runner writes ONE filename whatever its flags, so a
    # run with different flags replaces the previous one's data with no notice;
    # three sets of measurements were lost that way. The canonical name stays
    # (fig5 and the report read it); the tagged copy is never overwritten.
    import time as _time
    _tag = []
    if getattr(args, "sweep_schedule", False):
        _tag.append("sweep")
    if getattr(args, "sweep_competitors", False):
        _tag.append("compladder")
    elif getattr(args, "with_competitors", False):
        _tag.append("comp")
    if getattr(args, "jamaa_model", None):
        _tag.append("plain" if "plain" in str(args.jamaa_model)
                    else "keepbest")
    if getattr(args, "voc_source", "curve") != "curve":
        _tag.append(f"voc-{args.voc_source}")
    _archive = OUT / (f"gradual_sweep_{args.family}"
                      + ("_" + "_".join(_tag) if _tag else "_base")
                      + f"_{_time.strftime('%Y%m%d-%H%M%S')}.json")
    _archive.write_text(
        json.dumps(p12._json_safe(payload), indent=2, default=float),
        encoding="utf-8")
    print(f"   archive copy -> {_archive}")
    out.write_text(json.dumps(p12._json_safe(payload), indent=2, default=float),
                   encoding="utf-8")
    print(f"\nresults -> {out}")
    print("   scenario_sig in window_meta matches p12's, so the spliced and "
          "gradual")
    print("   runs join scenario by scenario rather than only in aggregate.")
    print("Validation modules; declared synthetic site (Jeju reference). "
          "All simulated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())