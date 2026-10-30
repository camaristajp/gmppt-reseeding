"""
p12_relocation_comparison.py  --  P2.9b: trackers through a peak RELOCATION.
REVISION 5 -- four changes, each closing something a reviewer would raise.

  (a) InC REPAIRED. Rev 4's InC was algebraically identical to P&O (63.81 vs
      63.80), and that was not a coincidence: dP/dV = I + V dI/dV, so
      dI/dV > -I/V is exactly the sign of dP/dV from the same two samples P&O
      uses. The ONE place the two methods genuinely differ is the "at the MPP,
      hold" branch -- InC stops perturbing where P&O keeps dithering -- and that
      branch was gated on an ABSOLUTE tolerance of 1e-6 on a quantity in A/V,
      against 1% current noise. It never fired. It is now a RELATIVE tolerance
      scaled to the noise floor, and the dV~0 branch no longer steps on the sign
      of a noise-dominated dI. Both changes make the baseline STRONGER, which is
      the only honest direction to move a comparator. Nguyen (2025) reports InC
      as the best of four methods on a 60-cell three-substring bypass-diode
      module -- our geometry -- so a degenerate InC arm was a liability.

  (b) SCHEDULED (predefined-time) RE-SEED added, for both PSO and the hybrid.
      Rev 4 compared detection against methods that do not detect, so the result
      read "detection beats nothing". Eltamaly, Farh & Al Saud (2019) and
      Eltamaly & Farh (2019, Solar Energy 177, 306-316) re-initialise on a CLOCK
      -- every T_PD seconds, 24 s in their work -- with no detection at all.
      That is the published comparator, and "detection beats a clock" is the
      claim we actually want to make. The clock's PHASE is randomised per window
      and it is scored over all alignments, because a fixed phase would make the
      arm's score depend on whether its clock happened to land near the jump.

  (c) THE DETECTOR IS NOW SWAPPABLE, from phase2/detectors.py, so the trigger
      design itself can be ablated rather than asserted: power-drop (rev 4's,
      the default), ratchet baseline, sign-run (Al Abri et al. 2021) and
      threshold-then-timer (Zaki et al. 2023). Run with --ablate-detectors.
      The dP/P criterion's provenance is Miyatake et al. (2011), canonically
      Ahmed & Salam (2014) eq. (15) with dP = 0.1; published thresholds span
      0.05-0.20 and --sweep-eps walks that range.

  (d) REPORTING AND STATISTICS.
      * Per-window re-convergence values are now WRITTEN to the JSON. Rev 4
        dropped them on write, so the 53-step tail against a 9-step median could
        not be diagnosed from the results file.
      * The windows are PAIRED (same module, same noise draw across methods), so
        the comparison is now a paired Wilcoxon signed-rank with a bootstrap CI
        on the median difference, instead of comparing a difference of MEDIANS
        against a standard error of the MEAN.
      * The uniform control runs EVERY triggered arm. Rev 4 ran only the hybrid,
        so the PSO arm shared the detector but never its false-alarm gate.
      * Censored comparisons print "censored", not "nan".

REVISION 4 -- MEASUREMENT NOISE, applied equally and paired. The module delivers
             TRUE power at the commanded voltage; the tracker READS it through
             noisy sensing (~1%/0.5% RMS, declared). Energy is scored on true
             power; trackers decide on noisy reads.
REVISION 3 -- change-detecting PSO: the FAIR baseline isolating cheap
             re-PREDICT from expensive re-SEARCH.
REVISION 2 -- DEVELOPED-relocation window (not crossover-adjacent); InC added.

PLACE THIS FILE AT:   phase2/p12_relocation_comparison.py   (replaces rev 4)
REQUIRES:             phase2/detectors.py                   (new, ships with this)

RUN:
    python phase2/p12_relocation_comparison.py --n 12 --family pole
    python phase2/p12_relocation_comparison.py --n 100 --ablate-detectors
    python phase2/p12_relocation_comparison.py --n 100 --sweep-eps
    python phase2/p12_relocation_comparison.py --n 12 --sweep-inc-tol

WHAT THIS MEASURES
    EN 50530 (p10) cannot move the global peak between substrings, so every
    figure there is dynamic POWER tracking, not RELOCATION. phase2/transition.py
    builds the case EN 50530 cannot: a sun-driven shading pattern whose GMPP
    jumps from one substring to another. This runner steps the trackers through
    ONE such jump and measures how fast each RE-LOCATES the new peak.

    The trigger fires on MEASURED power only -- never on traj.block_changed.
    Reading the block change would be reading the answer and would void the
    result.

SCOPE
    Validation modules only (reportable). Declared synthetic site (Jeju
    reference), pending Nanoom Energy geometry -- travels with every figure.
    All simulated.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path 
import os as _os
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gmppt import config, dataset, datasheet  # noqa: E402
from gmppt.dynamic import CONTROL_PERIOD_S, DynamicTrajectory  # noqa: E402
from gmppt.hybrid import SEED_PROBE_COST, seed_voltage  # noqa: E402
from gmppt.model import TwoStageModel  # noqa: E402
from gmppt.pso import DEFAULT_ITERATIONS, DEFAULT_POPULATION  # noqa: E402
from gmppt.tracking import DEFAULT_STEP_FRAC, START_FRACTION  # noqa: E402
from phase2.transition import (N_SUB, TRANSITION_FAMILIES,  # noqa: E402
                               gate_continuity, gate_relocation,
                               pattern_sequence, sequence_trajectory_for,
                               validate_sequence)

try:
    from phase2.detectors import (EPS_PUBLISHED, PowerDropDetector,
                                  RatchetDropDetector, SignRunDetector,
                                  TimerLatchDetector)
except ImportError as _e:                                    # pragma: no cover
    raise SystemExit(
        "rev 5 needs phase2/detectors.py (ships alongside this file).\n"
        f"  import failed: {_e}")

OUT = config.RESULTS_DIR / "phase2"
REPO = Path(__file__).resolve().parent.parent

# The design-level split lives under paper/, not results/. Same search order as
# phase3/p6_build_full_dataset.py, so the two cannot silently disagree about
# which partition they mean. --split-file overrides it.
SPLIT_SEARCH = ("paper/splits/design_split.json",
                "results/design_split.json",
                "design_split.json")


def resolve_split_file(explicit=None):
    if explicit:
        q = Path(explicit)
        if not q.is_absolute():
            q = REPO / q
        if not q.exists():
            raise SystemExit(f"--split-file {q} does not exist")
        return q
    for rel in SPLIT_SEARCH:
        q = REPO / rel
        if q.exists():
            return q
    raise SystemExit(
        "could not find design_split.json. Searched:\n  "
        + "\n  ".join(str(REPO / r) for r in SPLIT_SEARCH)
        + "\nPass --split-file PATH, or run --split name and label the result "
          "preliminary.")

# -- window shape (declared, unchanged from rev 2) ---------------------------
SETTLE_STEPS = 200        # control steps on the pre-jump curve; > PSO budget so
#                           PSO converges and then holds stale across the jump
WINDOW_STEPS = 300        # control steps on the post-jump curve
EPS = 0.01                # "re-converged" = captured >= (1-EPS)*p_gmpp_new
HOLD = 10                 # ... and stays there for HOLD steps

# -- trigger (declared, unchanged defaults) ----------------------------------
TRIGGER_DROP_FRAC = 0.05  # dP/P threshold. Miyatake et al. (2011); canonical
#                           form Ahmed & Salam (2014) eq. 15 at 0.1. Published
#                           range 0.05-0.20; 0.05 is the low end, not an
#                           invention. Swept by --sweep-eps.
TRIGGER_COOLDOWN = 15
TRIGGER_PERSIST = 4       # consecutive below-baseline reads required to fire
TRIGGER_EMA_ALPHA = 0.10

# -- scheduled re-seed (declared; Eltamaly et al. 2019 use T_PD = 24 s) ------
SCHEDULE_PERIOD_S = 24.0
SCHEDULE_PERIOD_STEPS = max(1, int(round(SCHEDULE_PERIOD_S / CONTROL_PERIOD_S)))

# -- InC repair (declared) ---------------------------------------------------
INC_MPP_TOL = 0.15        # RELATIVE tolerance on |dI/dV - (-I/V)|, as a fraction
#                           of |I/V|. Rev 4 used an absolute 1e-6, which against
#                           1% current noise never fired, collapsing InC to P&O.
#                           Swept by --sweep-inc-tol: InC's advantage exists only
#                           in a band, and reporting the band is the honest form
#                           of the result.
INC_DI_NOISE_FACTOR = 3.0  # in the dV~0 branch, require |dI| to exceed this many
#                           times the current-noise RMS before acting on its
#                           sign. Textbook InC reads that branch as "irradiance
#                           changed"; under noise it otherwise steps on the ADC.
INC_DV_EPS = 1e-9


def inc_mpp_tol_noise_floor(step_frac=DEFAULT_STEP_FRAC):
    """The smallest at-MPP tolerance that noise does not swamp, a priori.

    Near the MPP dI/dV ~ -I/V, so the at-MPP test is measuring a difference of
    two nearly equal quantities and its error is set by the noise in dI/dV. With
    independent current noise on both samples, sd(dI) ~ sqrt(2)*noise_i*I, and
    dV = step_frac*V_oc, so

        sd(dI/dV) ~ sqrt(2) * noise_i * I / (step_frac * V_oc)

    which, relative to |I/V| and evaluated near V = START_FRACTION*V_oc, is

        tol_floor ~ sqrt(2) * noise_i * START_FRACTION / step_frac

    A tolerance below this is reading noise. Printed with the header because the
    number is itself a result: at our declared 1% current noise and a 1% step it
    comes out ABOVE 1, which says fixed-step InC cannot reliably detect the MPP
    at this step size at all -- and that is the honest explanation for why InC
    collapsed onto P&O, as well as why the published InC implementations that
    beat P&O (Wang 2025, and the variable-step InC in Li et al. 2021) use a
    variable step or averaging rather than a fixed one.
    """
    return float(np.sqrt(2.0) * NOISE_I_RMS * START_FRACTION / max(step_frac, 1e-12))

# -- acceptance (declared) ---------------------------------------------------
MIN_RECONV_FRACTION = 0.80
MAX_FALSE_TRIGGER_FRAC = 0.05
ALPHA = 0.05                 # paired Wilcoxon significance level
N_BOOT = 10_000              # bootstrap resamples for the median-difference CI

# -- measurement noise (declared) --------------------------------------------
NOISE_I_RMS = 0.01
NOISE_V_RMS = 0.005
MAX_CONTROL_FALSE_ALARM_FRAC = 0.05
VOC_SOURCE = "curve"   # set once in main() from --voc-source; see est_learned_seed

FAMILY_DEFAULT = "pole"
CENSORED = None


class NoisyTrajectory(DynamicTrajectory):
    """DynamicTrajectory with MEASUREMENT noise on what the tracker READS.

    step() records the TRUE power in p_hist (used for SCORING) but RETURNS a
    noisy reading (used by the tracker to decide). Noise is paired across
    methods within a window, so it handicaps no one.
    """
    def configure_noise(self, noise_i, noise_v, rng):
        self._ni = float(noise_i)
        self._nv = float(noise_v)
        self._nrng = rng
        return self

    def step(self, v):
        # Every tracker reaches the simulator through this method, so it is the
        # only place a commanded voltage can be recorded without editing eleven
        # tracker implementations -- and an instrument that needs eleven edits
        # is an instrument that will disagree with itself.
        #
        # v_hist[SETTLE_STEPS-1] is where the tracker was resting at the
        # instant the relocation happened. The start-fraction sweep showed that
        # is the quantity deciding whether it ever re-converges, so it is worth
        # one append per step.
        try:
            self.v_hist.append(float(v))
        except AttributeError:
            self.v_hist = [float(v)]
        p_true = DynamicTrajectory.step(self, v)
        e = ((1.0 + self._nrng.normal(0.0, self._ni))
             * (1.0 + self._nrng.normal(0.0, self._nv)))
        return p_true * e


# ---------------------------------------------------------------------------
# RE-SEED POLICIES
# ---------------------------------------------------------------------------

def _make_policy(kind, eps=TRIGGER_DROP_FRAC, phase=0):
    """Build a re-seed policy. `kind` names the trigger design under test."""
    if kind == "power_drop":
        return PowerDropDetector(eps=eps, persist=TRIGGER_PERSIST,
                                 alpha=TRIGGER_EMA_ALPHA)
    if kind == "ratchet":
        return RatchetDropDetector(eps=eps, persist=TRIGGER_PERSIST)
    if kind == "sign_run":
        return SignRunDetector()
    if kind == "timer_latch":
        return TimerLatchDetector(eps=eps)
    if kind == "scheduled":
        from phase2.detectors import ScheduledReseed
        # The period may be overridden for a whole run, so that every
        # scheduled arm -- with P&O, without P&O, and the scheduled
        # competitors -- moves together and stays matched. Imported locally
        # so nothing is added at module scope.
        import os as _os
        _override = _os.environ.get("GMPPT_SCHEDULE_PERIOD_STEPS")
        _period = int(_override) if _override else SCHEDULE_PERIOD_STEPS
        if _override and not getattr(_make_policy, "_announced", False):
            print(f"   [policy] SCHEDULED PERIOD OVERRIDDEN: {_period} steps "
                  f"= {_period / 10:.0f} s   (default "
                  f"{SCHEDULE_PERIOD_STEPS} steps "
                  f"= {SCHEDULE_PERIOD_STEPS / 10:.0f} s)")
            print("   [policy] arm NAMES do not record this. Snapshot the "
                  "output under a\n   [policy] period-named label as soon as "
                  "the run finishes.")
            _make_policy._announced = True
        return ScheduledReseed(period=_period, phase=phase)
    raise ValueError(f"unknown policy {kind!r}")


def _fire(policy, v, p):
    """Feed one reading to a policy. Current is reconstructed as p/v, which is
    what the optimizer has. Voltage passed is the COMMANDED value -- the only
    detector that uses it is sign-run, and giving it a clean v is generous to
    it, not to us."""
    return policy.update(v, p / max(v, 1e-9), p)


# ---------------------------------------------------------------------------
# TRACKERS
# ---------------------------------------------------------------------------

def _clip(traj, v):
    return float(min(max(v, traj.v_floor), traj.v_oc))


class TrajContext:
    """A harness Context backed by a LIVE trajectory.

    The seed-level methods in baselines.py and the Jamaa reimplementation are
    written against the harness contract -- fn(ctx) -> landing voltage, with
    ctx.probe(v) returning power and counting itself. That contract is exactly
    what a tracker loop can supply, so those methods drop into the dynamic
    comparison without being rewritten: every probe becomes a control step and
    is charged to the window's energy, which is what makes the cost comparison
    honest rather than nominal.

    The power returned is the trajectory's reading, so under NoisyTrajectory a
    competitor decides on noisy measurements exactly as our seed does.
    """

    def __init__(self, traj, seed=0):
        self.traj = traj
        self.v_oc = float(traj.v_oc)
        self.i_sc = None
        self.module = str(getattr(traj, "module", ""))
        self.geometry = str(getattr(traj, "geometry", ""))
        self.rng = np.random.default_rng(seed)
        self.n_probes = 0
        # The relocation sequence does not carry per-substring irradiance into
        # the window, so conditional methods get an EMPTY conditions dict and
        # must infer from probing. ahmed_salam_scan is the fully scan-based
        # variant and is specified to do exactly that; the oracle variants that
        # read conditions are deliberately not offered here.
        self.conditions = {}

    def probe(self, v):
        self.n_probes += 1
        if self.traj._k >= self.traj.n_steps:
            return 0.0
        return float(self.traj.step(_clip(self.traj, v)))


def est_learned_seed(model, temp_c, voc_source=None):
    src = VOC_SOURCE if voc_source is None else voc_source
    def f(traj):
        extra, voc = 0, None
        if src == "measured":
            traj.step(traj.v_oc)          # open circuit: one step, ~zero power
            extra = 1
        elif src == "datasheet":
            voc = datasheet.v_oc_at(traj.module, temp_c)
        v, _ = seed_voltage(traj, model, temp_c, v_oc=voc)
        return _clip(traj, v), SEED_PROBE_COST + extra
    f.label = "learned seed"
    return f


def est_from_context_method(method, label):
    """Wrap any harness-contract method as a dynamic estimator.

    Returns (voltage, probes_actually_spent) -- the probe count is read from the
    context rather than declared, so a method that scans more than its nominal
    budget is charged for it.
    """
    def f(traj):
        ctx = TrajContext(traj, seed=int(config.seed_for("p12_ctx")) % 2**31)
        v = float(method(ctx))
        return _clip(traj, v), ctx.n_probes
    f.label = label
    return f


def t_po(traj, model=None, temp_c=25.0, step_frac=DEFAULT_STEP_FRAC,
         start_frac=START_FRACTION, **_):
    """Fixed-step perturb-and-observe from a FIXED start point.

    start_frac is a parameter rather than the module constant because it is the
    quantity --sweep-start-fraction varies. It is the only thing that decides
    which basin this tracker begins in, and the n=60 run shows that decides
    whether it arrives at all.
    """
    dv = step_frac * traj.v_oc
    v = start_frac * traj.v_oc
    p_prev = traj.step(v)
    direction = -1.0
    for _ in range(traj.n_steps - 1):
        v = _clip(traj, v + direction * dv)
        p = traj.step(v)
        if p < p_prev:
            direction = -direction
        p_prev = p


def t_inc(traj, model=None, temp_c=25.0, step_frac=DEFAULT_STEP_FRAC,
          mpp_tol=INC_MPP_TOL, start_frac=START_FRACTION, **_):
    """Fixed-step incremental conductance, REPAIRED (see rev-5 note (a)).

    The decision rule dI/dV vs -I/V is algebraically the sign of dP/dV, so it
    agrees with P&O everywhere except at the MPP, where InC HOLDS and P&O keeps
    dithering. That hold is InC's entire advantage, and it only exists if the
    "am I at the MPP" test can actually succeed on noisy readings -- hence a
    relative tolerance, scaled to |I/V|, rather than an absolute 1e-6.
    """
    dv = step_frac * traj.v_oc
    v = start_frac * traj.v_oc
    p = traj.step(v)
    i_prev, v_prev = p / max(v, 1e-9), v
    v = max(v - dv, traj.v_floor)
    for _ in range(traj.n_steps - 1):
        p = traj.step(v)
        i = p / max(v, 1e-9)
        d_v, d_i = v - v_prev, i - i_prev
        i_prev, v_prev = i, v
        rhs = -i / max(v, 1e-9)
        if abs(d_v) < INC_DV_EPS:
            # Textbook: dV = 0 and dI != 0 means the irradiance moved. Under
            # noise dI is noise, so require it to clear the declared noise floor
            # before acting on its sign; otherwise hold.
            if abs(d_i) < INC_DI_NOISE_FACTOR * NOISE_I_RMS * abs(i):
                continue
            v = _clip(traj, v + (dv if d_i > 0 else -dv))
            continue
        lhs = d_i / d_v
        if abs(lhs - rhs) <= mpp_tol * max(abs(rhs), 1e-12):
            continue                      # at the MPP: hold. The one real
            #                               difference from P&O.
        v = _clip(traj, v + (dv if lhs > rhs else -dv))


def t_pso(traj, model=None, temp_c=25.0, seed=0,
          population=DEFAULT_POPULATION, iterations=DEFAULT_ITERATIONS, **_):
    """Searches once, then HOLDS the global best -- no re-init."""
    rng = np.random.default_rng(
        (int(config.seed_for("pso_relocation")) + seed * 7919) % 2**31)
    gbx, used = _pso_search(traj, rng, traj.n_steps, population, iterations)
    while used < traj.n_steps:
        traj.step(gbx)
        used += 1


def _pso_search(traj, rng, budget, population, iterations):
    """One PSO search consuming up to `budget` steps; returns (g_best_x, used).

    The probes are scored as captured energy, because they are: a swarm that
    re-searches pays for every point it visits. That is the cost this comparison
    exists to measure, and the hybrid's re-seed probes are charged the same way.
    """
    from gmppt.pso import (C_COGNITIVE, C_SOCIAL, INERTIA, V_MAX_FRAC,
                           V_MIN_FRAC, VEL_MAX_FRAC)
    v_lo = max(V_MIN_FRAC * traj.v_oc, traj.v_floor)
    v_hi = V_MAX_FRAC * traj.v_oc
    vel_max = VEL_MAX_FRAC * traj.v_oc
    base = np.linspace(v_lo, v_hi, population)
    x = np.clip(base + (rng.random(population) - 0.5) * (v_hi - v_lo)
                / (2 * population), v_lo, v_hi)
    vel = np.zeros(population)
    pbx, pbp = x.copy(), np.full(population, -np.inf)
    gbx, gbp = float(x[0]), -np.inf
    used = 0
    for _ in range(iterations):
        for i in range(population):
            if used >= budget:
                break
            pw = traj.step(float(x[i]))
            used += 1
            if pw > pbp[i]:
                pbp[i], pbx[i] = pw, float(x[i])
            if pw > gbp:
                gbp, gbx = pw, float(x[i])
        if used >= budget:
            break
        r1, r2 = rng.random(population), rng.random(population)
        vel = np.clip(INERTIA * vel + C_COGNITIVE * r1 * (pbx - x)
                      + C_SOCIAL * r2 * (gbx - x), -vel_max, vel_max)
        x = np.clip(x + vel, v_lo, v_hi)
    return gbx, used


def _t_pso_repolicy(traj, policy, seed, population, iterations):
    """PSO that RE-SEARCHES from scratch whenever `policy` fires.

    With policy='scheduled' this is the predefined-time dispersal of Eltamaly
    et al. (2019); with a detector it is the change-detecting PSO of rev 3.
    Either way a swarm cannot re-predict -- it must pay for a new search.
    """
    rng = np.random.default_rng(
        (int(config.seed_for("pso_trig")) + seed * 7919) % 2**31)
    gbx, _ = _pso_search(traj, rng, traj.n_steps, population, iterations)
    p0 = traj.step(gbx) if traj._k < traj.n_steps else 0.0
    policy.reset(p0=p0, v0=gbx)
    fires = []
    while traj._k < traj.n_steps:
        p = traj.step(gbx)
        if _fire(policy, gbx, p):
            if traj._k >= traj.n_steps:
                break
            gbx, _ = _pso_search(traj, rng, traj.n_steps - traj._k,
                                 population, iterations)
            fires.append(int(traj._k))
    setattr(traj, "trigger_fires", fires)


def t_seed_only(traj, model, temp_c=25.0, **_):
    v_seed, _ = seed_voltage(traj, model, temp_c)
    for _ in range(traj.n_steps - SEED_PROBE_COST):
        traj.step(v_seed)


def t_hybrid_never(traj, model, temp_c=25.0, step_frac=DEFAULT_STEP_FRAC, **_):
    v_seed, _ = seed_voltage(traj, model, temp_c)
    n = traj.n_steps - SEED_PROBE_COST
    if n <= 0:
        return
    dv = step_frac * traj.v_oc
    v = _clip(traj, v_seed)
    p_prev = traj.step(v)
    direction = -1.0
    for _ in range(n - 1):
        v = _clip(traj, v + direction * dv)
        p = traj.step(v)
        if p < p_prev:
            direction = -direction
        p_prev = p


def _t_estimator_repolicy(traj, estimator, policy, step_frac):
    """An ESTIMATOR + P&O fine-tracking, re-fired by `policy`.

    This is the generic form of the hybrid arm. Holding the policy and the
    fine-tracker fixed and varying only the estimator is what isolates the cost
    axis: our learned seed, the analytic Ahmed-Salam scan and the Jamaa ANN all
    answer the same question -- where is the global peak now -- at ~6, ~33 and
    ~6 probes respectively, and every one of those probes is charged to the same
    energy score.

    Probes spent on estimation are accumulated on the trajectory so the table
    can report cost per relocation, which is the quantity the A1 claim is about.
    """
    dv = step_frac * traj.v_oc
    v_est, probes = estimator(traj)
    spent = probes
    v = _clip(traj, v_est)
    p_prev = traj.step(v)
    policy.reset(p0=p_prev, v0=v)
    direction = -1.0
    fires = []
    while traj._k < traj.n_steps:
        v = _clip(traj, v + direction * dv)
        p = traj.step(v)
        if p < p_prev:
            direction = -direction
        p_prev = p
        if _fire(policy, v, p):
            if traj._k >= traj.n_steps:
                break
            v_est, probes = estimator(traj)
            spent += probes
            v = _clip(traj, v_est)
            if traj._k >= traj.n_steps:
                break
            p = traj.step(v)
            p_prev = p
            direction = -1.0
            fires.append(int(traj._k))
    setattr(traj, "trigger_fires", fires)
    setattr(traj, "estimator_probes", spent)


def _t_hybrid_repolicy(traj, model, temp_c, step_frac, policy):
    """Back-compatible wrapper: our learned seed as the estimator."""
    _t_estimator_repolicy(traj, est_learned_seed(model, temp_c), policy,
                          step_frac)


# ---------------------------------------------------------------------------
# FIXED RESTART  (added for the estimator ablation)
# ---------------------------------------------------------------------------

FIXED_RESTART_FRACTION = START_FRACTION


def est_fixed_restart(fraction=None):
    """The cheapest possible re-seed: jump to a fixed fraction of V_oc.

    Costs ZERO probes. It reads nothing and predicts nothing, which is exactly
    the point: every estimator this study compares is sophisticated (PSO
    re-search, the ~33-probe analytic scan, the published ANN), and none is
    compared against doing nothing clever. Without this arm a reviewer can ask
    whether the gradient boosting does real work or whether ANY restart would
    do, and the answer is not in the data.

    0.80*V_oc is the same point P&O starts from, so region = floor(3 * 0.80)
    = 2 by construction -- which holds 54.5% of GMPPs over the whole dataset
    but only 44.5% on the multi-peak subset. That 44.5% is the prior this arm
    should be read against.
    """
    frac = FIXED_RESTART_FRACTION if fraction is None else fraction

    def est(traj):
        return frac * traj.v_oc, 0

    return est


def _t_estimator_nofine(traj, estimator, policy, step_frac):
    """An ESTIMATOR re-fired by `policy`, with NO P&O fine-tracking.

    Identical to `_t_estimator_repolicy` except that between fires the
    operating point is HELD at the estimate rather than dithered. Isolates how
    much of the tracking is the seed and how much is the hill-climber -- the
    split the energy decomposition rests on. The existing `seed only` arm is
    never re-seeded, so it conflates removing the triggers with removing P&O.

    `step_frac` is accepted and unused so this arm stays interchangeable with
    the fine-tracked form in an arms table.
    """
    v_est, probes = estimator(traj)
    spent = probes
    v = _clip(traj, v_est)
    p_prev = traj.step(v)
    policy.reset(p0=p_prev, v0=v)
    fires = []
    while traj._k < traj.n_steps:
        p = traj.step(v)                      # held, not perturbed
        if _fire(policy, v, p):
            if traj._k >= traj.n_steps:
                break
            v_est, probes = estimator(traj)
            spent += probes
            v = _clip(traj, v_est)
            if traj._k >= traj.n_steps:
                break
            traj.step(v)
            fires.append(int(traj._k))
    setattr(traj, "trigger_fires", fires)
    setattr(traj, "estimator_probes", spent)


def load_competitors(jamaa_model=None):
    """Seed-level competitors, lifted into the dynamic loop as estimators.

    Returns a list of (label, estimator). A competitor that cannot be loaded is
    SKIPPED WITH A MESSAGE rather than silently dropped -- a missing arm that
    nobody notices is how a comparison quietly becomes unfair.

    Only methods relevant to the dynamic axis are offered:

      ahmed_salam_scan  the published analytic estimator, fully scan-based.
                        It is the expensive-per-cycle competitor the cost claim
                        names: ~33 probes to re-estimate against our ~6. Running
                        it under the SAME re-seed policy isolates the estimator
                        cost from the policy, which is the A1 measurement.

      jamaa             the closest published learned method (5 probes,
                        sensorless, ANN -> beta). Loaded from a saved net; see
                        --jamaa-model. Its published form re-probes continuously
                        with no trigger, so the scheduled arm is the faithful
                        one and the triggered arm is the generous one.

    Not offered, deliberately: the Eltamaly GWO-FLC (its contribution is the
    re-initialisation trichotomy, already carried by the scheduled and triggered
    PSO arms -- a different optimiser is not a different idea), the Kalogerakis
    Q-learning tracker (a stateful learned controller worth comparing, but one
    to build rather than adapt), and Wang 2025 (not retrievable to a standard
    that allows verification against a worked example).
    """
    out = []
    # The fixed-restart ablation arm. Opt-in via the environment so no
    # existing run gains arms silently:
    #     GMPPT_FIXED_RESTART=1 python phase2/p12_... --with-competitors
    # Imported locally so this patch adds nothing at module scope.
    import os as _os
    if _os.environ.get("GMPPT_FIXED_RESTART"):
        out.append((f"fixed restart {FIXED_RESTART_FRACTION:.2f}Voc",
                    est_fixed_restart()))
        print(f"   [competitors] fixed-restart arm ENABLED at "
              f"{FIXED_RESTART_FRACTION:.2f}*Voc, 0 probes")
    try:
        from gmppt.baselines import METHODS as BASE_METHODS  # noqa
    except Exception:
        BASE_METHODS = None
    if BASE_METHODS is None:
        try:
            from gmppt.harness import METHODS as BASE_METHODS  # noqa
        except Exception as e:
            print(f"   [competitors] no method registry found ({e}); "
                  f"ahmed_salam arm skipped")
            BASE_METHODS = {}
    m = BASE_METHODS.get("ahmed_salam_scan")
    if m is None:
        print("   [competitors] 'ahmed_salam_scan' not in the registry; "
              "arm skipped")
    else:
        out.append(("ahmed-salam scan",
                    est_from_context_method(m, "ahmed_salam_scan")))

    if jamaa_model:
        try:
            import joblib
            blob = joblib.load(jamaa_model)
            import importlib.util
            spec = None
            for cand in ("paper/experiments/e3_jamaa_reimplementation.py",
                         "e3_jamaa_reimplementation.py",
                         "phase3/e3_jamaa_reimplementation.py"):
                f = REPO / cand
                if f.exists():
                    spec = importlib.util.spec_from_file_location("e3j", f)
                    break
            if spec is None:
                raise FileNotFoundError("e3_jamaa_reimplementation.py not found")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            fn = mod.make_jamaa_method(blob["net"], blob["p_ref"],
                                       blob.get("keep_best", True))
            out.append(("jamaa ANN", est_from_context_method(fn, "jamaa")))
        except Exception as e:
            print(f"   [competitors] jamaa arm skipped: {e}")
    return out


def build_methods(detector="power_drop", eps=TRIGGER_DROP_FRAC,
                  inc_tol=INC_MPP_TOL, extra_detectors=(), competitors=()):
    """The arm list. `phase` is supplied per window for the scheduled arms."""
    def pso_sched(traj, model=None, temp_c=25.0, seed=0, phase=0, **_):
        _t_pso_repolicy(traj, _make_policy("scheduled", phase=phase), seed,
                        DEFAULT_POPULATION, DEFAULT_ITERATIONS)

    def pso_trig(traj, model=None, temp_c=25.0, seed=0, **_):
        _t_pso_repolicy(traj, _make_policy(detector, eps=eps), seed,
                        DEFAULT_POPULATION, DEFAULT_ITERATIONS)

    def hyb_sched(traj, model, temp_c=25.0, step_frac=DEFAULT_STEP_FRAC,
                  phase=0, **_):
        _t_hybrid_repolicy(traj, model, temp_c, step_frac,
                           _make_policy("scheduled", phase=phase))

    def hyb_trig(traj, model, temp_c=25.0, step_frac=DEFAULT_STEP_FRAC, **_):
        _t_hybrid_repolicy(traj, model, temp_c, step_frac,
                           _make_policy(detector, eps=eps))

    def inc(traj, model=None, temp_c=25.0, step_frac=DEFAULT_STEP_FRAC, **_):
        t_inc(traj, step_frac=step_frac, mpp_tol=inc_tol)

    def hyb_sched_nofine(traj, model, temp_c=25.0,
                         step_frac=DEFAULT_STEP_FRAC, phase=0, **_):
        """Scheduled re-seed, operating point HELD between fires.

        Same seed, same policy, same windows as `hyb_sched`; the only
        difference is that P&O does not perturb between re-seeds. Built from
        `est_learned_seed` because `_t_estimator_nofine` takes an estimator
        rather than (model, temp_c) -- the same construction p13 used, and the
        probe bill must come out EQUAL to `hyb_sched`'s or the pair is not
        matched.
        """
        _t_estimator_nofine(traj, est_learned_seed(model, temp_c),
                            _make_policy("scheduled", phase=phase), step_frac)

    def hyb_trig_nofine(traj, model, temp_c=25.0,
                        step_frac=DEFAULT_STEP_FRAC, **_):
        """Detected re-seed, operating point HELD between fires."""
        _t_estimator_nofine(traj, est_learned_seed(model, temp_c),
                            _make_policy(detector, eps=eps), step_frac)

    methods = [
        ("P&O", t_po),
        ("InC", inc),
        ("PSO", t_pso),
        ("PSO, scheduled reseed", pso_sched),
        ("PSO, triggered reseed", pso_trig),
        ("seed only", t_seed_only),
        ("hybrid, never reseed", t_hybrid_never),
        ("hybrid, scheduled reseed", hyb_sched),
        ("hybrid, triggered reseed", hyb_trig),
    ]
    # The no-P&O ablation arms. OPT-IN, because `methods` is the comparison
    # family: two more arms make the Holm correction more conservative and
    # move p-values that are already reported. Imported locally so nothing is
    # added at module scope.
    import os as _os
    if _os.environ.get("GMPPT_NOPO"):
        methods.append(("hybrid, scheduled [no P&O]", hyb_sched_nofine))
        methods.append(("hybrid, triggered [no P&O]", hyb_trig_nofine))
        print("   [arms] no-P&O ablation arms ENABLED")
        print("   [arms] the comparison family is LARGER in this run, so its "
              "Holm-corrected")
        print("   [arms] p-values are NOT comparable with the reported ones. "
              "Read the")
        print("   [arms] no-P&O result from test_nopo.py on this JSON "
              "instead.")
    for label, est in competitors:
        def mk_sched(e=est):
            def f(traj, model=None, temp_c=25.0, step_frac=DEFAULT_STEP_FRAC,
                  phase=0, **_):
                _t_estimator_repolicy(traj, e,
                                      _make_policy("scheduled", phase=phase),
                                      step_frac)
            return f

        def mk_trig(e=est):
            def f(traj, model=None, temp_c=25.0, step_frac=DEFAULT_STEP_FRAC,
                  **_):
                _t_estimator_repolicy(traj, e, _make_policy(detector, eps=eps),
                                      step_frac)
            return f
        methods.append((f"{label}, scheduled", mk_sched()))
        methods.append((f"{label}, triggered", mk_trig()))

    for d in extra_detectors:
        if d == detector:
            continue

        def mk(dd=d):
            def f(traj, model, temp_c=25.0, step_frac=DEFAULT_STEP_FRAC, **_):
                _t_hybrid_repolicy(traj, model, temp_c, step_frac,
                                   _make_policy(dd, eps=eps))
            return f
        methods.append((f"hybrid, triggered [{d}]", mk()))
    return methods


TRIGGERED_ARMS = ("PSO, scheduled reseed", "PSO, triggered reseed",
                  "hybrid, scheduled reseed", "hybrid, triggered reseed")


# ---------------------------------------------------------------------------
# WINDOW EXTRACTION  (unchanged from rev 2)
# ---------------------------------------------------------------------------

def strided(seq, stride):
    """A copy of `seq` sampled every `stride`-th step.

    Finding the two developed endpoints costs one simulated curve per sample,
    and the pole family has ~1,345 samples per module above 5 deg elevation --
    of which exactly TWO are used. Since the GMPP-region runs are hours long
    (measured: 4.5 h / 2.2 h / 4.5 h on the equinox), a region change cannot
    hide between samples taken ten times further apart, so the search can run
    on a coarse grid at a tenth of the cost.

    A coarse index is still a REAL pattern and a real curve -- nothing is
    interpolated. All that is lost is temporal precision in locating a plateau
    midpoint, and a plateau that lasts hours does not need it.

    G3 continuity must still be checked on the FULL-resolution sequence: on a
    strided one the per-step irradiance change is `stride` times larger and
    would fail the gate for a reason that is an artefact of the search, not of
    the physics.
    """
    import copy
    s2 = copy.copy(seq)
    s2.irradiances_list = list(seq.irradiances_list)[::stride]
    s2.times = seq.times[::stride]
    s2.xc = np.asarray(seq.xc)[::stride]
    s2.sample_seconds = float(seq.sample_seconds) * stride
    return s2


def find_adjacent_transitions(per_step, min_run=3):
    """Every DEVELOPED, ADJACENT pair of GMPP-region runs, with its direction.

    Replaces find_developed_endpoints, which took the two LONGEST runs of
    different regions. That rule has two defects the region timeline makes
    plain:

      TIE INSTABILITY. On a measured sequence the runs are
      1:504  2:67  1:202  2:66  1:506 -- the first and last region-1 plateaus
      differ by two samples out of 1,345, because the sun's path is roughly
      symmetric about solar noon and the two end plateaus are naturally the same
      length. Which of them wins decides the transition DIRECTION, and a
      coarsened search flips it. An experiment whose direction turns on a
      two-sample margin is not measuring what it claims to.

      NON-ADJACENT SPLICING. The two longest runs need not be consecutive. The
      pair above splices run 2 (ending at 570) to run 5 (starting at 839),
      skipping two intervening relocations. That is not a transition the day
      contains; it is two plateaus glued together across 4.6 hours.

    Taking ADJACENT runs fixes both. Each pair is a relocation that actually
    occurs, in the order it occurs, and both endpoints are still mid-plateau and
    therefore fully developed. The measured sequence yields FOUR transitions per
    module, two in each direction, so direction balance becomes possible instead
    of being an artefact of a tie-break.

    Returns [(pre_idx, post_idx, region_pre, region_post), ...] in time order.
    """
    regions = per_step["regions"]
    runs, s = [], 0
    for i in range(1, len(regions) + 1):
        if i == len(regions) or regions[i] != regions[s]:
            runs.append((int(regions[s]), s, i - 1))
            s = i
    out = []
    for x, y in zip(runs, runs[1:]):
        if x[0] == y[0]:
            continue
        if (x[2] - x[1] + 1) < min_run or (y[2] - y[1] + 1) < min_run:
            continue
        out.append(((x[1] + x[2]) // 2, (y[1] + y[2]) // 2, int(x[0]), int(y[0])))
    return out


def find_developed_endpoints(per_step, min_run=3):
    """Mid-plateau of the two longest runs of DIFFERENT GMPP regions, so the
    jump carries the peak-height gap of a COMPLETED relocation."""
    regions = per_step["regions"]
    runs, s = [], 0
    for i in range(1, len(regions) + 1):
        if i == len(regions) or regions[i] != regions[s]:
            runs.append((int(regions[s]), s, i - 1))
            s = i
    runs = [r for r in runs if (r[2] - r[1] + 1) >= min_run]
    if len(runs) < 2:
        return None
    runs.sort(key=lambda r: (r[2] - r[1]), reverse=True)
    a = runs[0]
    b = next((r for r in runs[1:] if r[0] != a[0]), None)
    if b is None:
        return None
    if a[1] > b[1]:
        a, b = b, a
    return (a[1] + a[2]) // 2, (b[1] + b[2]) // 2, int(a[0]), int(b[0])


def make_window(traj_full, per_step, pre_idx, post_idx, win_seed):
    pre_curve = traj_full.curves[pre_idx + 1]
    post_curve = traj_full.curves[post_idx + 1]
    p_gmpp_new = float(post_curve[3])
    p_gmpp_old = float(pre_curve[3])
    module, geometry = traj_full.module, traj_full.geometry

    def build():
        t = NoisyTrajectory(
            curves=[pre_curve, post_curve],
            block_steps=np.array([SETTLE_STEPS, WINDOW_STEPS], int),
            scored=np.array([False, True], bool),
            profile_name=traj_full.profile_name, module=module,
            geometry=geometry)
        return t.configure_noise(NOISE_I_RMS, NOISE_V_RMS,
                                 np.random.default_rng(win_seed))
    build.curves = (pre_curve, post_curve)   # read by --export-fig1

    # How hard THIS window is: the fraction of the new peak that a tracker still
    # sitting at the OLD peak's voltage is missing. Comparing the two GMPP
    # VALUES would not measure that -- what matters is the new curve's height at
    # the old voltage, which is where a trapped tracker actually is. Saved per
    # window so a long re-convergence can be read against its own difficulty
    # rather than guessed at.
    V_new, P_new = np.asarray(post_curve[0], float), np.asarray(post_curve[1], float)
    v_gmpp_old = float(pre_curve[2])
    order = np.argsort(V_new)
    p_held = float(np.interp(v_gmpp_old, V_new[order], P_new[order]))

    meta = {"module": module, "pre_idx": int(pre_idx), "post_idx": int(post_idx),
            "p_gmpp_new": p_gmpp_new, "p_gmpp_old": p_gmpp_old,
            "v_gmpp_old": v_gmpp_old, "p_new_at_old_v": p_held,
            "drop_depth": float(1.0 - p_held / max(p_gmpp_new, 1e-9)),
            "region_pre": int(per_step["regions"][pre_idx]),
            "region_post": int(per_step["regions"][post_idx])}
    return build, meta


def make_uniform_control(traj_full, idx, win_seed):
    """G5 control: same curve before and after -- no relocation."""
    post_curve = traj_full.curves[idx + 1]

    def build():
        t = NoisyTrajectory(
            curves=[post_curve, post_curve],
            block_steps=np.array([SETTLE_STEPS, WINDOW_STEPS], int),
            scored=np.array([False, True], bool),
            profile_name="uniform_control", module=traj_full.module,
            geometry="uniform_control")
        return t.configure_noise(NOISE_I_RMS, NOISE_V_RMS,
                                 np.random.default_rng(win_seed + 1_000_000))
    return build


# ---------------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------------

def reconvergence_steps(traj, p_gmpp_new):
    p = np.asarray(traj.p_hist, float)
    post = p[SETTLE_STEPS:]
    thresh = (1.0 - EPS) * p_gmpp_new
    ok = post >= thresh
    for i in range(len(ok) - HOLD):
        if ok[i:i + HOLD].all():
            return int(i)
    return CENSORED


def transition_energy_loss_pct(traj, p_gmpp_new):
    p = np.asarray(traj.p_hist, float)[SETTLE_STEPS:]
    avail = p_gmpp_new * len(p)
    return float(100.0 * (avail - p.sum()) / max(avail, 1e-9))


def resting_voltage(traj):
    """The commanded voltage on the LAST pre-jump step.

    Not the start fraction: a tracker spends SETTLE_STEPS on the pre-jump curve
    before the relocation, and where it comes to rest there is its own
    business. Returns None if the trajectory did not record enough steps, which
    happens for an arm that exhausted its budget during the settle -- reported
    as unknown rather than guessed.
    """
    vh = getattr(traj, "v_hist", None)
    if not vh or len(vh) < SETTLE_STEPS:
        return None
    return float(vh[SETTLE_STEPS - 1])


def voltage_region(v, v_oc):
    """Which substring region a voltage falls in. Same rule as the GMPP labels,
    so the two are directly comparable -- that comparability is the whole point.
    """
    if v is None or not np.isfinite(v) or v_oc <= 0:
        return None
    return int(min(int((v / v_oc) * N_SUB), N_SUB - 1))


# ---------------------------------------------------------------------------
# STATISTICS  (paired -- the windows are the same modules and the same noise)
# ---------------------------------------------------------------------------

def paired_compare(a, b, censor_value, rng):
    """Compare arm A against arm B on PAIRED windows.

    Censored re-convergences are replaced by `censor_value` (the window length),
    which is the conservative treatment: it credits a censored method with the
    best score it could possibly have had. A win that survives that is real.

    Returns the median paired difference (B - A, so positive favours A), a
    bootstrap 95% CI on it, and a paired Wilcoxon signed-rank p-value.
    """
    av = np.array([censor_value if x is None else x for x in a], float)
    bv = np.array([censor_value if x is None else x for x in b], float)
    d = bv - av
    med = float(np.median(d))
    idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    boots = np.median(d[idx], axis=1)
    lo, hi = (float(np.percentile(boots, 2.5)),
              float(np.percentile(boots, 97.5)))
    try:
        from scipy.stats import wilcoxon
        if np.allclose(d, 0):
            p = 1.0
        else:
            p = float(wilcoxon(av, bv, zero_method="zsplit").pvalue)
    except Exception:
        p = float("nan")
    return {"median_diff": med, "ci_lo": lo, "ci_hi": hi, "p": p,
            "n": int(len(d)),
            "a_censored": int(sum(x is None for x in a)),
            "b_censored": int(sum(x is None for x in b))}


def load_val_modules(which, split_file=None):
    """Validation modules from the chosen partition, plus a label for the file.

    `dataset.module_split()` is the NAME-level split, under which 78.2% of
    validation modules have their design already present in training and the
    median distance to the nearest training module is zero. A relocation result
    drawn from it measures recall of duplicate nameplates alongside whatever
    else it measures. design_split.json is the design-level partition built at
    z <= 0.05 and is the one to use.

    The split is RECORDED in the results file. Without that, two runs months
    apart are indistinguishable on disk, and the one thing that most changes how
    the numbers should be read is the one thing not written down.
    """
    if which == "name":
        s = dataset.module_split()
        return list(s.val), "name-level (module name) -- CARRIES THE LEAKAGE CAVEAT"

    p = resolve_split_file(split_file)
    d = json.loads(p.read_text(encoding="utf-8"))
    node = d.get("val", d.get("validation"))
    if node is None:
        raise SystemExit(f"{p.name} has no 'val' or 'validation' key; "
                         f"top-level keys are {sorted(d)}")
    if isinstance(node, dict):
        for k in ("modules", "module_names", "names", "members"):
            if k in node:
                node = node[k]
                break
        else:
            raise SystemExit(f"{p.name}: 'val' is a dict with keys "
                             f"{sorted(node)}; none of them is a module list")
    mods = [str(x) for x in node]
    if not mods:
        raise SystemExit(f"{p.name}: the validation partition is empty")
    return mods, f"design-level (z <= 0.05, {p}, {len(mods)} modules)"


def _json_safe(o):
    """nan/inf -> null, recursively.

    A censored median is nan, and Python's json writes a bare NaN token that
    strict parsers reject -- pandas, jq and every JS reader among them. The
    results file has to be readable by whatever draws the figures.
    """
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, (np.floating, np.integer)):
        f = float(o)
        return f if np.isfinite(f) else None
    return o


def _fmt_cmp(label, c):
    sig = "" if np.isnan(c["p"]) else (
        "  significant" if c["p"] < ALPHA else "  NOT significant")
    cens = ""
    if c["b_censored"]:
        cens = f"  [{c['b_censored']}/{c['n']} of the comparator censored]"
    return (f"   {label:<42}{c['median_diff']:+7.0f} steps   "
            f"95% CI [{c['ci_lo']:+.0f}, {c['ci_hi']:+.0f}]   "
            f"p={c['p']:.4f}{sig}{cens}")


# ---------------------------------------------------------------------------
# RUNNER
# ---------------------------------------------------------------------------

def run_one_window(build, meta, methods, model, temp_c, phase):
    out = {}
    for name, fn in methods:
        traj = build()
        fn(traj, model=model, temp_c=temp_c, phase=phase)
        fires = getattr(traj, "trigger_fires", None)
        probes = getattr(traj, "estimator_probes", None)
        v_rest = resting_voltage(traj)
        out[name] = {
            "reconv_steps": reconvergence_steps(traj, meta["p_gmpp_new"]),
            "energy_loss_pct": transition_energy_loss_pct(traj,
                                                          meta["p_gmpp_new"]),
            "trigger_fires": fires,
            "estimator_probes": probes,
            "resting_v": v_rest,
            "resting_frac": (None if v_rest is None
                             else float(v_rest / max(traj.v_oc, 1e-9))),
            "resting_region": voltage_region(v_rest, traj.v_oc),
            "n_fires": (len(fires) if fires else 0)}
    return out


def export_fig1_windows(windows, methods, model, out_dir):
    """Fig. 1 of the paper (make_fig1.py): write each selected window's two
    curves and the P&O resting voltage, one .npz per window.

    Same windows and the same noise draw as a normal run, because selection is
    untouched; only the P&O arm is simulated, for its resting voltage (the
    commanded voltage on the last pre-jump step). Nothing else is run or saved.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    po = dict(methods)["P&O"]
    for wi, (build, meta) in enumerate(windows):
        traj = build()
        po(traj, model=model, temp_c=25.0, phase=0)
        v_rest = resting_voltage(traj)
        pre, post = build.curves
        wid = f"w{wi:02d}_{meta['region_pre']}to{meta['region_post']}"
        np.savez(out_dir / f"{wid}.npz",
                 v_before=np.asarray(pre[0], float),
                 p_before=np.asarray(pre[1], float),
                 v_after=np.asarray(post[0], float),
                 p_after=np.asarray(post[1], float),
                 v_op=np.nan if v_rest is None else float(v_rest),
                 module=str(meta["module"]), temp_c=25.0, window_id=wid)
    print(f"exported {len(windows)} windows to {out_dir} "
          f"(both curves + P&O resting voltage) for make_fig1.py")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12,
                    help="validation modules to draw a relocation window from")
    ap.add_argument("--family", default=FAMILY_DEFAULT,
                    choices=list(TRANSITION_FAMILIES))
    ap.add_argument("--detector", default="power_drop",
                    choices=["power_drop", "ratchet", "sign_run", "timer_latch"],
                    help="trigger design for the triggered arms")
    ap.add_argument("--ablate-detectors", action="store_true",
                    help="add one hybrid arm per alternative detector")
    ap.add_argument("--sweep-eps", action="store_true",
                    help="re-run the triggered hybrid at each published dP/P")
    ap.add_argument("--sweep-inc-tol", action="store_true",
                    help="sweep InC's at-MPP tolerance and report the band")
    ap.add_argument("--sweep-start-fraction", action="store_true",
                    help="re-run the fixed-start hill-climbers from several "
                         "start points, reported per transition direction. "
                         "Separates 'fails outside its starting basin' from "
                         "'fails on deeper relocations', which the balanced "
                         "sample confounds. Does not touch the seeded arms.")
    ap.add_argument("--start-fractions", default="0.40,0.60,0.80",
                    help="comma-separated fractions of Voc to start the "
                         "hill-climbers from. Region boundaries are at 1/3 and "
                         "2/3, so the default set spans two regions; a set "
                         "inside one region cannot answer the question and the "
                         "sweep says so rather than reporting a null result.")
    ap.add_argument("--split", default="design", choices=["design", "name"],
                    help="module partition to draw validation windows from. "
                         "'design' uses design_split.json (z <= 0.05); 'name' "
                         "uses dataset.module_split(), which carries the "
                         "duplicate-nameplate leakage.")
    ap.add_argument("--no-balance-directions", dest="balance_directions",
                    action="store_false",
                    help="do NOT balance the sample across transition "
                         "directions. Unbalanced samples measure one problem "
                         "and read as if they measured two.")
    ap.add_argument("--with-competitors", action="store_true",
                    help="add the published seed-level competitors as dynamic "
                         "arms: ahmed_salam_scan always, jamaa if "
                         "--jamaa-model is given")
    ap.add_argument("--jamaa-model", default=None,
                    help="joblib file holding {'net':..., 'p_ref':...} saved "
                         "from the E3 reimplementation")
    ap.add_argument("--stride", type=int, default=1,
                    help="sample every Nth pattern when SEARCHING for the two "
                         "developed endpoints. The region runs are hours long, "
                         "so a coarse search finds the same plateaus at 1/N the "
                         "cost; G3 continuity is still checked at full "
                         "resolution. DEFAULT 1 (exact): each module now yields "
                         "every adjacent transition, so far fewer modules are "
                         "needed and the exact search is affordable. Coarsening "
                         "can shift run-length ties -- use it for exploration "
                         "only.")
    ap.add_argument("--split-file", default=None,
                    help="path to design_split.json; searched automatically "
                         "under paper/splits/ then results/ if omitted")
    ap.add_argument("--model-tag", default="c3_two_stage_design",
                    help="trained seed to load. Defaults to the design-split "
                         "model; pass c3_two_stage_full to reproduce the older "
                         "numbers.")
    ap.add_argument("--noise-seed", type=int, default=0,
                    help="offset the per-window measurement-noise RNG. "
                         "Scenario selection is deterministic from the module "
                         "list, so this re-draws the NOISE and nothing else: "
                         "same windows, same curves, same depths. Use it to "
                         "ask whether a result is a property of a method or of "
                         "one noise draw. Noise stays paired across arms "
                         "within a window at every seed.")
    ap.add_argument("--export-fig1", default=None, metavar="DIR",
                    help="write each selected window's two curves and the P&O "
                         "resting voltage to DIR (.npz) for make_fig1.py, then "
                         "exit without running the comparison. Window selection "
                         "is the same as a normal run with the same --n.")
    ap.add_argument("--voc-source", choices=["curve", "measured", "datasheet"],
                    default="curve",
                    help="where the seed's V_oc comes from. curve: the "
                         "simulated characteristic, which every result "
                         "before 26 Oct 2026 used. measured: the same true "
                         "value, but charged -- one open-circuit step, "
                         "which produces no power, so both the step and its "
                         "energy are paid for (6 probes, 7 in the "
                         "manuscript's landing-inclusive convention). "
                         "datasheet: the nameplate value corrected to the "
                         "module temperature, costing no extra read "
                         "(5 probes, 6 in that convention).")
    args = ap.parse_args()

    global VOC_SOURCE
    VOC_SOURCE = args.voc_source

    rng = np.random.default_rng(int(config.seed_for("p12_stats")) % 2**31)

    print("P2.9b  relocation comparison: trackers through a single peak jump")
    print("       REVISION 5 -- scheduled arm, swappable detector, repaired InC,")
    print("       paired statistics, per-window data saved")
    try:
        print("  " + config.provenance())
    except Exception:
        pass
    print("=" * 78)
    print(f"family {args.family}; window settle {SETTLE_STEPS} + hold "
          f"{WINDOW_STEPS} steps at {1000*CONTROL_PERIOD_S:.0f} ms")
    print(f"trigger: {args.detector}, dP/P = {TRIGGER_DROP_FRAC:.2f} "
          f"(Miyatake 2011; Ahmed & Salam 2014 eq. 15; published range "
          f"0.05-0.20)")
    print(f"schedule: every {SCHEDULE_PERIOD_S:.0f} s = {SCHEDULE_PERIOD_STEPS} "
          f"steps (Eltamaly et al. 2019 T_PD), phase randomised per window")
    floor = inc_mpp_tol_noise_floor()
    print(f"InC at-MPP tolerance {INC_MPP_TOL:.2f} relative "
          f"(rev 4 used an absolute 1e-6, which never fired);")
    print(f"   a-priori noise floor on that test is {floor:.2f} -- "
          f"{'ABOVE 1, so fixed-step InC cannot' if floor >= 1.0 else 'so InC can'}")
    print("   reliably detect the MPP at this step size. Report the sweep, and "
          "declare the")
    print("   tolerance from this floor -- never from whichever value scores best.")
    print("DECLARED SYNTHETIC site (Jeju reference), pending Nanoom Energy "
          "geometry -- travels with every figure.")
    print(f"MEASUREMENT NOISE applied equally (paired): current "
          f"{100*NOISE_I_RMS:.1f}% RMS, voltage {100*NOISE_V_RMS:.1f}% RMS.\n")
    print("acceptance, declared before this run:")
    print(f"   (1) triggered hybrid re-converges on >= "
          f"{100*MIN_RECONV_FRACTION:.0f}% of events")
    print("   (2) [FAIR] it beats the CHANGE-DETECTING PSO on a paired Wilcoxon")
    print(f"       signed-rank at alpha = {ALPHA}, CI excluding zero")
    print("   (2b) [NEW] it beats the SCHEDULED (predefined-time) re-seed, same test")
    print(f"   (3) triggers fire near the jump (false-trigger < "
          f"{100*MAX_FALSE_TRIGGER_FRAC:.0f}%)")
    print("   (4) uniform control: no trigger fires, on EVERY triggered arm\n")

    try:
        model = TwoStageModel.load(tag=args.model_tag)
    except FileNotFoundError:
        print(f"model {args.model_tag!r} not found. Train it with\n"
              "   python phase3/p6_build_full_dataset.py --reuse-dataset "
              "--retrain\n"
              "or pass --model-tag c3_two_stage_full for the older model.")
        return 1

    extra = (("ratchet", "sign_run", "timer_latch")
             if args.ablate_detectors else ())
    comp = ()
    if args.with_competitors:
        print("loading published competitors as dynamic arms:")
        comp = load_competitors(args.jamaa_model)
        for label, est in comp:
            print(f"   {label}")
        if not comp:
            print("   none loaded -- continuing without them")
        print()
    methods = build_methods(detector=args.detector, inc_tol=INC_MPP_TOL,
                            extra_detectors=extra, competitors=comp)
    names = [n for n, _ in methods]

    all_val, split_label = load_val_modules(args.split, args.split_file)
    # Walk the pool in a SEEDED SHUFFLED order, not alphabetically. The CEC
    # names are ordered so that rebadges of one panel sit together
    # (PM060P00/P01/P02/PA0/PA1/... are thirteen names for one design), so
    # taking the first N alphabetically collects a handful of designs replicated
    # many times and calls it N samples.
    pool_rng = np.random.default_rng(int(config.seed_for("p12_pool")) % 2**31)
    all_val = list(all_val)
    pool_rng.shuffle(all_val)
    val_modules = all_val[:max(args.n * 12, args.n)]
    print(f"split: {split_label}")
    if args.split == "name":
        print("   WARNING: under this split 78.2% of validation modules have "
              "their design")
        print("   already in training, median distance zero. The seed's "
              "re-prediction is")
        print("   then partly recall. Label any figure from this run preliminary.")
    print(f"   model tag {args.model_tag}")
    if (args.split == "design") != (args.model_tag == "c3_two_stage_design"):
        print("   MISMATCH: the evaluation split and the model's training split")
        print("   do not look like a matched pair. Evaluating on a clean split")
        print("   buys nothing if the model was fitted under the other one.")
    print(f"drawing relocation windows on validation modules "
          f"(pool {len(val_modules)} of {len(all_val)}, shuffled)")
    if args.stride > 1:
        print(f"   endpoint search on every {args.stride}th pattern "
              f"(~{args.stride}x faster; G3 still at full resolution)")
    print()

    windows, uniform_builds, discarded, duplicate = [], [], 0, 0
    coarse_misses = 0
    seen = set()
    buckets = {}
    buckets_flat = []
    # Collect into per-direction buckets and balance at the end. A relocation
    # INTO the region P&O starts in (0.80*Voc, i.e. region 2) is a different
    # problem from one out of it: the hill-climber begins in the destination
    # basin and needs no help. A sample drawn from one direction only measures
    # one of those two problems and reads as if it measured both.
    def _collected():
        return sum(len(v) for v in buckets.values())
    for mod in val_modules:
        if _collected() >= args.n * (3 if args.balance_directions else 1):
            break
        seq = pattern_sequence(str(mod), 25.0, family=args.family)
        try:
            # Coarsening may only ever cost TIME, never results: if the coarse
            # grid is too sparse for find_developed_endpoints (which needs runs
            # of at least min_run samples), this module is retried at full
            # resolution rather than discarded. A silent "0 windows" caused by
            # a search parameter would be the worst possible failure here.
            if args.stride > 1:
                # G3 on the full-resolution geometry (pure arithmetic, no
                # curves, so it is free); curves only on the coarse grid.
                g3, _ = gate_continuity(seq, verbose=False)
                traj_full, per_step = sequence_trajectory_for(
                    strided(seq, args.stride))
                g1, _ = gate_relocation(per_step, verbose=False)
                ok = bool(g1 and g3)
            else:
                ok, traj_full, per_step, _rep = validate_sequence(
                    seq, verbose=False)
        except RuntimeError:
            discarded += 1
            continue
        if not ok:
            discarded += 1
            continue
        eps = find_adjacent_transitions(per_step)
        if not eps and args.stride > 1:
            try:
                ok2, traj_full, per_step, _r = validate_sequence(seq, verbose=False)
                eps = find_adjacent_transitions(per_step) if ok2 else []
                if eps:
                    coarse_misses += 1
            except RuntimeError:
                eps = []
        if not eps:
            discarded += 1
            continue
        for ti, (pre_idx, post_idx, _ra, _rb) in enumerate(eps):
            widx = (int(config.seed_for("p12_noise")) % 2**31
                    + 1000 * len(buckets_flat) + ti
                    + 7_919_311 * int(args.noise_seed))
            build, meta = make_window(traj_full, per_step, pre_idx, post_idx,
                                      widx)
        # A window is a SCENARIO, not a module name. Rebadged modules share one
        # nameplate, so they produce numerically identical curves and identical
        # windows; counting them separately is pseudo-replication and inflates
        # every paired test. Fingerprint the physics and keep one of each.
            sig = (round(meta["p_gmpp_new"], 6), round(meta["p_gmpp_old"], 6),
                   round(meta["v_gmpp_old"], 6),
                   round(meta["p_new_at_old_v"], 6),
                   meta["region_pre"], meta["region_post"])
            if sig in seen:
                duplicate += 1
                continue
            seen.add(sig)
            meta["scenario_sig"] = str(hash(sig) % 10**12)
            meta["transition_index"] = ti
            buckets.setdefault((meta["region_pre"], meta["region_post"]),
                               []).append(
                (build, meta, make_uniform_control(traj_full, post_idx, widx)))
            buckets_flat.append(1)

    if args.balance_directions and len(buckets) > 1:
        per = max(1, args.n // len(buckets))
        picked = []
        for d in sorted(buckets):
            picked.extend(buckets[d][:per])
        for d in sorted(buckets):            # top up from whatever is left
            if len(picked) >= args.n:
                break
            picked.extend(buckets[d][per:per + args.n - len(picked)])
        picked = picked[:args.n]
    else:
        picked = [w for d in sorted(buckets) for w in buckets[d]][:args.n]
    windows = [(b, m) for b, m, _ in picked]
    uniform_builds = [u for _, _, u in picked]

    if len(windows) < 3:
        print(f"only {len(windows)} clean relocation windows "
              f"({discarded} discarded). Need >= 3; widen --n.")
        return 1
    print(f"{len(windows)} DISTINCT relocation scenarios, {discarded} discarded "
          f"by gates,\n   {duplicate} rejected as numerically identical to a "
          f"scenario already held")
    if coarse_misses:
        print(f"   {coarse_misses} module(s) needed the full-resolution search "
              f"after the coarse grid\n   found no developed pair -- consider a "
              f"smaller --stride")
    if duplicate:
        print(f"   ({duplicate} rebadged duplicates rejected -- the CEC database "
              f"collapses 20,946\n    names to 8,234 designs, so module count is "
              f"not scenario count)")
    from collections import Counter
    cnt = Counter((m["region_pre"], m["region_post"]) for _, m in windows)
    trans = sorted(cnt)
    print(f"   transition directions: "
          f"{', '.join(f'{a}->{b} (n={cnt[(a, b)]})' for a, b in trans)}")
    start_region = min(int(START_FRACTION * N_SUB), N_SUB - 1)
    into_start = sum(c for (a, b), c in cnt.items() if b == start_region)
    print(f"   {into_start}/{len(windows)} relocate INTO region {start_region}, "
          f"where P&O starts (0.80*Voc)")
    print("   -- a hill-climber that begins in the destination basin needs no")
    print("      help, so this split largely decides how P&O scores.")
    if len(trans) == 1:
        print("   WARNING: every scenario is the SAME transition direction. The")
        print("   sample is broad in modules and narrow in event shape; say so")
        print("   rather than reporting n as if the events were varied.")
    dd = [m["drop_depth"] for _, m in windows]
    print(f"   drop depth: median {np.median(dd):.3f}, "
          f"range {min(dd):.3f}-{max(dd):.3f}\n")

    if args.export_fig1:
        export_fig1_windows(windows, methods, model, args.export_fig1)
        return 0

    # -- the comparison ------------------------------------------------------
    per_method = {n: {"reconv": [], "energy": [], "fires": [], "probes": [],
                      "n_fires": [], "resting_frac": [], "resting_region": []}
                  for n in names}
    for wi, (build, meta) in enumerate(windows):
        phase = int(rng.integers(0, SCHEDULE_PERIOD_STEPS))
        res = run_one_window(build, meta, methods, model, 25.0, phase)
        for n in names:
            per_method[n]["reconv"].append(res[n]["reconv_steps"])
            per_method[n]["energy"].append(res[n]["energy_loss_pct"])
            per_method[n]["fires"].append(res[n]["trigger_fires"])
            per_method[n]["probes"].append(res[n]["estimator_probes"])
            per_method[n]["n_fires"].append(res[n]["n_fires"])
            per_method[n]["resting_frac"].append(res[n]["resting_frac"])
            per_method[n]["resting_region"].append(res[n]["resting_region"])

    # -- uniform control, EVERY triggered arm (rev 4 ran only the hybrid) -----
    ctrl = {n: 0 for n in names if n in TRIGGERED_ARMS or "triggered [" in n}
    for ui, ub in enumerate(uniform_builds):
        phase = int(rng.integers(0, SCHEDULE_PERIOD_STEPS))
        for n, fn in methods:
            if n not in ctrl:
                continue
            traj = ub()
            fn(traj, model=model, temp_c=25.0, phase=phase)
            if getattr(traj, "trigger_fires", []):
                ctrl[n] += 1

    # -- table ---------------------------------------------------------------
    print("RE-CONVERGENCE AFTER THE JUMP  (control steps; 'cens' = never)")
    print(f"   {'method':<30} {'median':>8} {'worst':>8} {'re-conv %':>10} "
          f"{'energy loss %':>14} {'fires':>7} {'probes':>8}")
    print("   " + "-" * 88)
    stats = {}
    for n in names:
        rc = per_method[n]["reconv"]
        got = [x for x in rc if x is not None]
        frac = len(got) / len(rc)
        med = float(np.median(got)) if got else float("nan")
        wor = float(np.max(got)) if got else float("nan")
        el = float(np.mean(per_method[n]["energy"]))
        stats[n] = {"reconv": rc, "median": med, "worst": wor,
                    "reconv_frac": frac, "energy_loss_pct": el}
        pr = [x for x in per_method[n]["probes"] if x is not None]
        nf = [x for x in per_method[n]["n_fires"] if x is not None]
        stats[n]["mean_probes"] = float(np.mean(pr)) if pr else None
        stats[n]["mean_fires"] = float(np.mean(nf)) if nf else None
        print(f"   {n:<30} {(f'{med:.0f}' if got else 'cens'):>8} "
              f"{(f'{wor:.0f}' if got else 'cens'):>8} {100*frac:>9.0f}% "
              f"{el:>14.2f} {(f'{np.mean(nf):.1f}' if nf else '--'):>7} "
              f"{(f'{np.mean(pr):.0f}' if pr else '--'):>8}")

    # -- verdict -------------------------------------------------------------
    print("\n" + "=" * 78 + "\nVERDICT")
    ours = "hybrid, triggered reseed"
    trig = stats[ours]
    censor_value = WINDOW_STEPS

    c1 = trig["reconv_frac"] >= MIN_RECONV_FRACTION
    print(f"\n   (1) triggered re-converges on {100*trig['reconv_frac']:.0f}% "
          f"of events (bar {100*MIN_RECONV_FRACTION:.0f}%) -> "
          f"{'PASS' if c1 else 'FAIL'}")

    print("\n   PAIRED COMPARISONS (positive favours the triggered hybrid;")
    print("   censored re-convergences scored at the window length, which is the")
    print("   most generous value the comparator could have had)")
    comparisons = {}
    for other in names:
        if other == ours:
            continue
        c = paired_compare(per_method[ours]["reconv"],
                           per_method[other]["reconv"], censor_value, rng)
        comparisons[other] = c
        print(_fmt_cmp(f"vs {other}", c))

    def _gate(c):
        """Both the median CI and the signed-rank test must agree.

        They answer different questions and can disagree: the bootstrap CI is on
        the MEDIAN difference and shrugs off one bad window, while Wilcoxon ranks
        every pair and one large discordant difference can dominate it. Requiring
        both means the advantage has to be consistent AND robust, which is the
        standard a reviewer will apply.
        """
        return c["ci_lo"] > 0 and (np.isnan(c["p"]) or c["p"] < ALPHA)

    def _explain(c):
        if c["ci_lo"] > 0 and not np.isnan(c["p"]) and c["p"] >= ALPHA:
            print("       NOTE: the median CI is clear of zero but the signed-rank")
            print("       test is not. That means most windows favour us and at")
            print("       least one strongly does not. Find it in the results file")
            print("       (per_window) and read it against its own drop_depth --")
            print("       do not raise alpha.")

    cf = comparisons["PSO, triggered reseed"]
    c2 = _gate(cf)
    print(f"\n   (2) [FAIR] vs CHANGE-DETECTING PSO -> {'PASS' if c2 else 'FAIL'}")
    print("       both detect the jump; this isolates re-PREDICT from re-SEARCH.")
    _explain(cf)

    cs = comparisons["hybrid, scheduled reseed"]
    c2b = _gate(cs)
    print(f"   (2b) vs SCHEDULED re-seed (Eltamaly T_PD) -> "
          f"{'PASS' if c2b else 'FAIL'}")
    print("       both re-seed with the same cheap predictor; this isolates")
    print("       DETECTION from a clock, which is the published comparator.")
    _explain(cs)

    if len(windows) < 20:
        pmin = 2.0 / (2 ** len(windows))
        print(f"\n   POWER WARNING: with {len(windows)} paired windows the smallest")
        print(f"   two-sided signed-rank p attainable is {pmin:.4f}, and a single")
        print("   discordant window can push the test above alpha however large")
        print("   the effect is. Rev 4's acceptance test was weaker (a difference")
        print("   of medians against a standard error of the mean), so a rev-4")
        print("   PASS can legitimately become a rev-5 FAIL here. That is the")
        print("   test working. Run n >= 50 before reading this as a result.")

    late_lo, late_hi = SETTLE_STEPS, SETTLE_STEPS + TRIGGER_COOLDOWN + 40
    clean_fires = total_fires = 0
    for fires in per_method[ours]["fires"]:
        for f in (fires or []):
            total_fires += 1
            if late_lo <= f <= late_hi:
                clean_fires += 1
    ft = 1.0 - (clean_fires / total_fires) if total_fires else 0.0
    c3 = ft <= MAX_FALSE_TRIGGER_FRAC
    print(f"\n   (3) {total_fires} triggers, {clean_fires} near the jump; "
          f"false-trigger {100*ft:.0f}% (bar {100*MAX_FALSE_TRIGGER_FRAC:.0f}%) "
          f"-> {'PASS' if c3 else 'FAIL'}")

    bar = int(np.ceil(MAX_CONTROL_FALSE_ALARM_FRAC * max(len(uniform_builds), 1)))
    print(f"   (4) uniform control UNDER NOISE (bar {bar}/{len(uniform_builds)}):")
    c4 = True
    for n, k in ctrl.items():
        scheduled = "scheduled" in n
        ok = k <= bar
        if n == ours:
            c4 = ok
        verdict = ("fires by design (a clock cannot false-alarm; shown for the "
                   "cost it pays)" if scheduled
                   else ("ok" if ok else "FALSE-ALARMING -- detector unsound"))
        print(f"       {n:<36} {k:>3}/{len(uniform_builds)}  {verdict}")

    # -- energy decomposition, the way the paper should report it ------------
    print("\n   ENERGY DECOMPOSITION (mean % of post-jump energy lost)")
    # The "no detection" baseline must be the BEST arm that does not detect,
    # not our own never-reseed variant. Using hybrid-never flatters the result
    # whenever the seed lands worse than a naive start -- which happens exactly
    # when the relocation moves the peak INTO the region P&O starts in, and
    # then plain P&O is the honest comparator.
    nondet = ["P&O", "InC", "PSO", "seed only", "hybrid, never reseed"]
    best_nd = min((n for n in nondet if n in stats),
                  key=lambda n: stats[n]["energy_loss_pct"])
    nev = stats[best_nd]["energy_loss_pct"]
    pst = stats["PSO, triggered reseed"]["energy_loss_pct"]
    hsc = stats["hybrid, scheduled reseed"]["energy_loss_pct"]
    hyb = trig["energy_loss_pct"]
    print(f"       best method with NO detection          {nev:>6.2f}%   "
          f"({best_nd})")
    print(f"       + a clock (scheduled re-seed)          {hsc:>6.2f}%   "
          f"(buys {nev-hsc:+.2f} pt)")
    print(f"       + detection, re-SEARCH (PSO)           {pst:>6.2f}%   "
          f"(buys {nev-pst:+.2f} pt vs no detection)")
    print(f"       + detection, re-PREDICT (ours)         {hyb:>6.2f}%   "
          f"(buys {pst-hyb:+.2f} pt over re-search)")
    if hyb > nev:
        print(f"\n       NOTE: {best_nd} BEATS the triggered hybrid on energy here")
        print(f"       ({nev:.2f}% vs {hyb:.2f}%). Re-seeding costs probes, and on a")
        print("       relocation shallow enough for a hill-climber to walk across,")
        print("       those probes are not repaid. Report this -- it is the")
        print("       condition that bounds the claim, and a reviewer who runs a")
        print("       shallow case will find it whether or not it is reported.")
    else:
        print("       Detection is the large lever; re-prediction is a real but")
        print("       second-order refinement on top of it. That framing is what a")
        print("       reviewer who knows the Eltamaly papers will accept.")

    print("\n   INTERPRETATION")
    if stats["hybrid, never reseed"]["reconv_frac"] < trig["reconv_frac"]:
        print("   The never-reseed hybrid lags (like P&O); only a re-seeding")
        print("   variant re-locates -- measured, not assumed.")
    if c1 and c2 and c2b and c3 and c4:
        print("   Cheap re-prediction on a DETECTED change re-locates the moving")
        print("   peak faster than both a swarm that must re-search and a clock")
        print("   that re-seeds blind.")
    else:
        print("   Claim NOT supported as declared. Report what held and what did")
        print("   not; do not adjust thresholds to force a pass.")

    # -- optional sweeps -----------------------------------------------------
    sweeps = {}
    if args.sweep_eps:
        print("\n" + "=" * 78)
        print("dP/P SWEEP over the published range")
        print(f"   {'eps':>6} {'source':<52} {'median':>8} {'loss %':>8} "
              f"{'ctrl FA':>8}")
        print("   " + "-" * 86)
        for e, src in EPS_PUBLISHED:
            ms = build_methods(detector=args.detector, eps=e,
                               inc_tol=INC_MPP_TOL)
            fn = dict(ms)[ours]
            rc, el = [], []
            for build, meta in windows:
                traj = build()
                fn(traj, model=model, temp_c=25.0, phase=0)
                rc.append(reconvergence_steps(traj, meta["p_gmpp_new"]))
                el.append(transition_energy_loss_pct(traj, meta["p_gmpp_new"]))
            fa = 0
            for ub in uniform_builds:
                traj = ub()
                fn(traj, model=model, temp_c=25.0, phase=0)
                if getattr(traj, "trigger_fires", []):
                    fa += 1
            got = [x for x in rc if x is not None]
            sweeps[f"eps_{e}"] = {"reconv": rc, "energy": el, "ctrl_fa": fa}
            print(f"   {e:>6.2f} {src[:52]:<52} "
                  f"{(f'{np.median(got):.0f}' if got else 'cens'):>8} "
                  f"{np.mean(el):>8.2f} {fa:>5}/{len(uniform_builds)}")

    if args.sweep_inc_tol:
        print("\n" + "=" * 78)
        print("InC at-MPP TOLERANCE SWEEP -- where does InC stop being P&O?")
        print(f"   {'tol':>8} {'InC loss %':>12} {'P&O loss %':>12} "
              f"{'difference':>12}")
        print("   " + "-" * 50)
        po_loss = stats["P&O"]["energy_loss_pct"]
        for tol in (0.0, 0.01, 0.05, 0.10, 0.15, 0.25, 0.40):
            el = []
            for build, meta in windows:
                traj = build()
                t_inc(traj, mpp_tol=tol)
                el.append(transition_energy_loss_pct(traj, meta["p_gmpp_new"]))
            m = float(np.mean(el))
            sweeps[f"inc_tol_{tol}"] = {"energy": el}
            print(f"   {tol:>8.2f} {m:>12.2f} {po_loss:>12.2f} "
                  f"{po_loss - m:>+12.2f}")
        print(f"   tol = 0 reproduces rev 4: InC collapses to P&O exactly.")
        print(f"   a-priori noise floor on this test: {inc_mpp_tol_noise_floor():.2f}")
        print("   READ THIS AS SENSITIVITY, NOT SELECTION. A larger tolerance")
        print("   suppresses more dither and will usually score better here,")
        print("   because this window does not punish holding still too long.")
        print("   Choosing the best-scoring value would be fitting a BASELINE to")
        print("   the test data -- the opposite of what a baseline is for. Declare")
        print("   the tolerance from the noise floor and report the curve.")

    if args.sweep_start_fraction:
        fracs = []
        for tok in args.start_fractions.split(","):
            tok = tok.strip()
            if tok:
                fracs.append(float(tok))
        if not fracs:
            raise SystemExit("--start-fractions parsed to nothing")

        def start_region_of(sf):
            return min(int(sf * N_SUB), N_SUB - 1)

        print("\n" + "=" * 78)
        print("START-FRACTION SWEEP -- BASIN or DEPTH?")
        print("   The balanced run confounds transition direction with drop")
        print("   depth. Moving the hill-climber's start moves the BASIN and")
        print("   leaves every window's DEPTH untouched, so the two separate:")
        print("     basin  -> the trapped direction FLIPS with the start region")
        print("     depth  -> the same direction traps at every start")
        print("   Our hybrid arms start from the learned seed and never read")
        print("   START_FRACTION, so they are identical at every row here by")
        print("   construction -- nothing of ours moves with the baseline.")

        regions = sorted({start_region_of(sf) for sf in fracs})
        if len(regions) < 2:
            print(f"\n   WARNING: every requested start lands in region "
                  f"{regions[0]} (boundaries at {1/N_SUB:.3f} and "
                  f"{2/N_SUB:.3f} of Voc).")
            print("   A sweep inside one region cannot separate basin from "
                  "depth. Widen")
            print("   --start-fractions across a boundary, or read the rows "
                  "as a null result.")

        # the per-direction depth means, printed once: this IS the confound
        depth_by_dir = {}
        for _b, m in windows:
            d = (int(m["region_pre"]), int(m["region_post"]))
            depth_by_dir.setdefault(d, []).append(float(m["drop_depth"]))
        print("\n   the confound, for reference (unchanged by this sweep):")
        for d in sorted(depth_by_dir):
            dd = depth_by_dir[d]
            print(f"      {d[0]}->{d[1]}  n={len(dd):<3d}  mean drop depth "
                  f"{float(np.mean(dd)):.3f}")

        climbers = (("P&O", t_po, {}),
                    ("InC", t_inc, {"mpp_tol": INC_MPP_TOL}))
        dirs = sorted(depth_by_dir)
        sf_rows = {}

        print(f"\n   {'start':>7} {'region':>7} {'method':<6} {'dir':>6} "
              f"{'re-conv':>8} {'median':>8} {'energy %':>9} {'rest':>6} "
              f"{'r.reg':>6}")
        print("   " + "-" * 76)
        for sf in fracs:
            reg = start_region_of(sf)
            for label, fn, kw in climbers:
                for d in dirs:
                    rc, el = [], []
                    rest_f, rest_r = [], []
                    for build, meta in windows:
                        if (int(meta["region_pre"]),
                                int(meta["region_post"])) != d:
                            continue
                        traj = build()
                        fn(traj, start_frac=sf, **kw)
                        rv = resting_voltage(traj)
                        rest_f.append(None if rv is None
                                      else float(rv / max(traj.v_oc, 1e-9)))
                        rest_r.append(voltage_region(rv, traj.v_oc))
                        rc.append(reconvergence_steps(traj,
                                                      meta["p_gmpp_new"]))
                        el.append(transition_energy_loss_pct(
                            traj, meta["p_gmpp_new"]))
                    got = [x for x in rc if x is not None]
                    frac = len(got) / max(len(rc), 1)
                    rr = [x for x in rest_r if x is not None]
                    rest_mode = (max(set(rr), key=rr.count) if rr else None)
                    rf = [x for x in rest_f if x is not None]
                    sf_rows[(sf, label, d)] = {
                        "n": len(rc), "reconv_frac": frac,
                        "median": (float(np.median(got)) if got else None),
                        "energy_mean": float(np.mean(el)) if el else None,
                        "start_region": reg,
                        "resting_region_mode": rest_mode,
                        "resting_frac_mean": (float(np.mean(rf)) if rf
                                              else None),
                        "resting_region": rest_r, "resting_frac": rest_f,
                        "reconv": rc, "energy": el}
                    print(f"   {sf:>7.2f} {reg:>7d} {label:<6} "
                          f"{f'{d[0]}->{d[1]}':>6} {100*frac:>7.0f}% "
                          f"{(f'{np.median(got):.0f}' if got else 'cens'):>8} "
                          f"{(f'{np.mean(el):.2f}' if el else '--'):>9} "
                          f"{(f'{np.mean(rf):.2f}' if rf else '--'):>6} "
                          f"{('--' if rest_mode is None else rest_mode):>6}")

        # -- the verdict ---------------------------------------------------
        # basin hypothesis: a window SUCCEEDS iff its destination region is the
        # one the tracker starts in. depth hypothesis: success does not depend
        # on the start at all. Score both against what was measured, treating
        # "succeeded" as re-converging on most of the stratum.
        SUCCESS_BAR = 0.5
        basin_hits = basin_n = 0
        for (sf, label, d), r in sf_rows.items():
            observed = r["reconv_frac"] >= SUCCESS_BAR
            predicted = (d[1] == r["start_region"])
            basin_hits += int(observed == predicted)
            basin_n += 1
        basin_agree = basin_hits / max(basin_n, 1)

        invariant = 0
        invariant_n = 0
        for label, _fn, _kw in climbers:
            for d in dirs:
                seen_obs = {sf_rows[(sf, label, d)]["reconv_frac"]
                            >= SUCCESS_BAR for sf in fracs}
                invariant += int(len(seen_obs) == 1)
                invariant_n += 1
        depth_agree = invariant / max(invariant_n, 1)

        print("\n   VERDICT")
        print(f"      basin hypothesis agreement  {100*basin_agree:5.0f}%  "
              f"({basin_hits}/{basin_n} strata match "
              f"'succeeds iff destination == start region')")
        print(f"      depth hypothesis agreement  {100*depth_agree:5.0f}%  "
              f"({invariant}/{invariant_n} direction/method pairs unchanged "
              f"across starts)")
        if len(regions) < 2:
            print("      INCONCLUSIVE by construction -- the starts do not "
                  "cross a region")
            print("      boundary, so both hypotheses predict the same thing "
                  "here.")
        elif basin_agree >= 0.9 and depth_agree <= 0.5:
            print("      -> BASIN. The trapped direction follows the start "
                  "point while")
            print("         drop depth is unchanged, so the failure is "
                  "geometric: a")
            print("         fixed-start hill-climber arrives only when its "
                  "start already")
            print("         lies in the destination basin. Report A3 in its "
                  "unconfounded")
            print("         form and cite this sweep.")
        elif depth_agree >= 0.9 and basin_agree <= 0.6:
            print("      -> DEPTH. Moving the start does not move which "
                  "direction traps,")
            print("         so the n=60 bifurcation is about how deep the "
                  "relocation is,")
            print("         NOT about the starting basin. The mechanism "
                  "paragraph in the")
            print("         findings doc is wrong and must be rewritten "
                  "around depth.")
        else:
            print("      -> MIXED. Neither hypothesis explains the strata on "
                  "its own.")
            print("         Report the table as measured and do not assert "
                  "either")
            print("         mechanism; the honest form is that the start "
                  "point matters")
            print("         and so does depth.")
        print("      Scored at a 50% re-convergence bar. The n=60 strata were "
              "100% or")
        print("      0%, so this verdict does not rest on where that bar sits "
              "-- but")
        print("      check the table above rather than taking the label.")

        sweeps["start_fraction"] = {
            "fractions": fracs,
            "n_sub": int(N_SUB),
            "boundaries": [1.0 / N_SUB, 2.0 / N_SUB],
            "success_bar": SUCCESS_BAR,
            "basin_agreement": basin_agree,
            "depth_agreement": depth_agree,
            "regions_spanned": regions,
            "depth_by_direction": {f"{d[0]}->{d[1]}": depth_by_dir[d]
                                   for d in dirs},
            "rows": {f"{sf}|{label}|{d[0]}->{d[1]}": v
                     for (sf, label, d), v in sf_rows.items()}}

    # -- the mechanism, MEASURED rather than inferred ------------------------
    # The hypothesis after the start-fraction sweep: a tracker re-converges iff
    # it was ALREADY resting in the post-jump GMPP's region when the jump
    # happened. Everything needed to test that is now recorded, so test it
    # directly instead of reasoning from start fractions.
    resting = {}
    rest_arms = [n for n in names
                 if per_method[n]["resting_region"]
                 and any(x is not None for x in per_method[n]["resting_region"])]
    if rest_arms:
        print("\n" + "=" * 78)
        print("RESTING POSITION AT THE JUMP -- the mechanism, measured")
        print("   Where each tracker was sitting on the LAST pre-jump step.")
        print("   The start fraction only influences this; the pre-jump curve")
        print("   decides the rest, which is why starts 0.40 and 0.60 agreed")
        print("   and 0.80 inverted.")
        print(f"\n   {'arm':<30} {'rest frac':>10} {'rest reg':>9} "
              f"{'in dest basin':>14} {'re-conv':>8}")
        print("   " + "-" * 76)
        hit = tot = 0
        for n in rest_arms:
            rf = [x for x in per_method[n]["resting_frac"] if x is not None]
            rr = per_method[n]["resting_region"]
            rc = per_method[n]["reconv"]
            in_basin = agree = cnt = 0
            for i, (_b, m) in enumerate(windows):
                if i >= len(rr) or rr[i] is None:
                    continue
                cnt += 1
                was_in = (rr[i] == int(m["region_post"]))
                got = rc[i] is not None
                in_basin += int(was_in)
                agree += int(was_in == got)
            if not cnt:
                continue
            hit += agree
            tot += cnt
            mode = max(set(x for x in rr if x is not None),
                       key=[x for x in rr if x is not None].count)
            print(f"   {n:<30} {np.mean(rf):>10.3f} {mode:>9d} "
                  f"{100*in_basin/cnt:>13.0f}% "
                  f"{100*sum(1 for x in rc if x is not None)/len(rc):>7.0f}%")
            resting[n] = {"resting_frac": per_method[n]["resting_frac"],
                          "resting_region": rr,
                          "frac_in_destination_basin": in_basin / cnt,
                          "agreement_with_reconvergence": agree / cnt,
                          "n_scored": cnt}
        if tot:
            print(f"\n   POOLED: 're-converges iff it was already resting in "
                  f"the destination")
            print(f"   region' holds on {100*hit/tot:.0f}% of "
                  f"{tot} arm-window pairs.")
            if hit / tot >= 0.9:
                print("   -> the mechanism is the RESTING POSITION, measured "
                      "directly rather")
                print("      than inferred from the start fraction. State A3 "
                      "in those terms:")
                print("      a fixed-start hill-climber re-converges only when "
                      "the jump finds")
                print("      it already inside the new peak's basin. Our "
                      "triggered arms are")
                print("      exempt because they do not rely on where they "
                      "happened to be.")
            elif hit / tot >= 0.7:
                print("   -> mostly, with real exceptions. Report the "
                      "percentage, not the")
                print("      rule, and look at the arms that disagree before "
                      "claiming a")
                print("      mechanism.")
            else:
                print("   -> NOT the mechanism, or not the whole of it. Do not "
                      "state the")
                print("      basin claim; report the cross-tab and say what is "
                      "unexplained.")
        resting["_pooled_agreement"] = (hit / tot) if tot else None

    # -- InC tolerance sweep, now crossed with the start fraction -------------
    # The explanation on offer for InC diverging from P&O is that its at-MPP
    # hold fires on noise: the a-priori floor is 2.26 at start 0.80 and 1.13 at
    # 0.40, so the branch misfires about twice as often down there, and random
    # freezing near a peak moves where it comes to rest. That predicts a
    # MONOTONE interpolation in tol -- P&O-like at 0, frozen at large -- and a
    # STEEPER one at low start fractions. Both halves are falsifiable here.
    if args.sweep_inc_tol and getattr(args, "start_fractions", None):
        sfs = [float(x) for x in args.start_fractions.split(",") if x.strip()]
        if len(sfs) > 1:
            print("\n" + "=" * 78)
            print("InC TOLERANCE x START FRACTION -- is InC's divergence from "
                  "P&O noise?")
            print("   prediction: at tol -> 0 the hold never fires and InC "
                  "collapses onto")
            print("   P&O at every start; as tol grows InC freezes more, and "
                  "the drift is")
            print("   steeper where the noise floor is smaller (low start).")
            # PER DIRECTION, never pooled. The existing --sweep-inc-tol reports
            # one mean over all 60 windows, and on this sample half of them are
            # saturated at ~60% because both arms are trapped. A real sixfold
            # change on the solvable half shows up there as a 1.6-point pooled
            # delta, which reads as noise and is not. Re-convergence is carried
            # alongside because energy saturates for a trapped arm and the
            # fraction that ever arrives does not.
            dirs_x = sorted({(int(m["region_pre"]), int(m["region_post"]))
                             for _b, m in windows})
            print(f"\n   {'start':>7} {'floor':>7} {'tol':>7} {'dir':>6} "
                  f"{'InC loss %':>11} {'P&O loss %':>11} {'gap':>9} "
                  f"{'InC re-conv':>12}")
            print("   " + "-" * 80)
            inc_x = {}
            for sf in sfs:
                floor = float(np.sqrt(2.0) * NOISE_I_RMS * sf
                              / max(DEFAULT_STEP_FRAC, 1e-12))
                po_by = {d: [] for d in dirs_x}
                for build, meta in windows:
                    d = (int(meta["region_pre"]), int(meta["region_post"]))
                    traj = build()
                    t_po(traj, start_frac=sf)
                    po_by[d].append(transition_energy_loss_pct(
                        traj, meta["p_gmpp_new"]))
                for tol in (0.0, 0.01, 0.05, 0.15, 0.40, 1.00):
                    by = {d: {"el": [], "rc": []} for d in dirs_x}
                    for build, meta in windows:
                        d = (int(meta["region_pre"]), int(meta["region_post"]))
                        traj = build()
                        t_inc(traj, mpp_tol=tol, start_frac=sf)
                        by[d]["el"].append(transition_energy_loss_pct(
                            traj, meta["p_gmpp_new"]))
                        by[d]["rc"].append(reconvergence_steps(
                            traj, meta["p_gmpp_new"]))
                    for d in dirs_x:
                        el, rc = by[d]["el"], by[d]["rc"]
                        if not el:
                            continue
                        m = float(np.mean(el))
                        pom = float(np.mean(po_by[d])) if po_by[d] else float("nan")
                        rcf = sum(1 for x in rc if x is not None) / len(rc)
                        key = f"{sf}|{tol}|{d[0]}->{d[1]}"
                        inc_x[key] = {"inc": m, "po": pom, "floor": floor,
                                      "inc_reconv_frac": rcf,
                                      "n": len(el), "energy": el, "reconv": rc}
                        print(f"   {sf:>7.2f} {floor:>7.2f} {tol:>7.2f} "
                              f"{f'{d[0]}->{d[1]}':>6} "
                              f"{m:>11.2f} {pom:>11.2f} {pom - m:>+9.2f} "
                              f"{100*rcf:>11.0f}%")
            sweeps["inc_tol_x_start"] = inc_x
            print("\n   Read the tol = 0 rows first: that is InC with the hold")
            print("   disabled, which IS P&O's decision rule. A gap there is "
                  "NOT the")
            print("   hold branch and points at the other structural "
                  "difference --")
            print("   P&O carries a direction across steps and InC recomputes "
                  "it, so")
            print("   near a peak P&O dithers deterministically while InC "
                  "random-walks.")
            print("   A gap that opens only as tol grows is the noise story, "
                  "and then")
            print("   InC's advantage is a defect rather than a result and "
                  "must not be")
            print("   reported as a method property.")
            print("   Compare rows WITHIN a direction. A direction where both "
                  "arms are")
            print("   trapped saturates near the drop depth and will look "
                  "flat whatever")
            print("   the tolerance does -- that flatness is the sample, not "
                  "the method.")

    # -- payload: per-window values INCLUDED (rev 4 dropped them) ------------

    payload = {
        "revision": 5, "family": args.family, "n_windows": len(windows),
        "split": args.split, "split_description": split_label,
        "n_val_pool": len(all_val), "model_tag": args.model_tag,
        "voc_source": args.voc_source,
        "jamaa_model": args.jamaa_model,
        "arms_optin": {"fixed_restart": bool(_os.environ.get("GMPPT_FIXED_RESTART")),
                       "no_po": bool(_os.environ.get("GMPPT_NOPO"))},
        "discarded": discarded, "duplicates_rejected": duplicate,
        "n_distinct_scenarios": len(windows),
        "transition_directions": [list(t) for t in trans],
        "search_stride": args.stride,
        "coarse_search_misses": coarse_misses,
        "settle_steps": SETTLE_STEPS,
        "window_steps": WINDOW_STEPS, "detector": args.detector,
        "trigger_drop_frac": TRIGGER_DROP_FRAC,
        "trigger_persist": TRIGGER_PERSIST,
        "schedule_period_steps": SCHEDULE_PERIOD_STEPS,
        "inc_mpp_tol": INC_MPP_TOL,
        "noise": {"i_rms": NOISE_I_RMS, "v_rms": NOISE_V_RMS},
        "window_meta": [m for _, m in windows],
        "noise_seed": int(args.noise_seed),
        "resting_analysis": resting,
        "per_window": {n: {"reconv": per_method[n]["reconv"],
                           "energy": per_method[n]["energy"],
                           "resting_frac": per_method[n]["resting_frac"],
                           "resting_region": per_method[n]["resting_region"],
                           "fires": per_method[n]["fires"]} for n in names},
        "stats": stats,
        "comparisons": comparisons,
        "control_false_alarms": ctrl,
        "sweeps": sweeps,
        "criteria": {"c1": bool(c1), "c2": bool(c2), "c2b": bool(c2b),
                     "c3": bool(c3), "c4": bool(c4)}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"relocation_comparison_{args.family}.json").write_text(
        json.dumps(_json_safe(payload), indent=2, default=float),
        encoding="utf-8")
    print(f"\nresults -> {OUT / f'relocation_comparison_{args.family}.json'}")
    print("   per-window re-convergence, energy and trigger times are now in the")
    print("   file, so a long tail can be traced to its own window's drop depth.")
    print("Validation modules; declared synthetic site (Jeju reference). "
          "All simulated.")
    return 0 if (c1 and c2 and c2b and c3 and c4) else 1


if __name__ == "__main__":
    raise SystemExit(main())