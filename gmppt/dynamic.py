"""
dynamic.py  --  the EN 50530 dynamic harness, extended to partial shading.
REVISION 3 -- slope now determines the control-step count (defect D17).

PLACE THIS FILE AT:   C:\\Users\\user\\gmppt\\gmppt\\dynamic.py
                      OVERWRITE revision 2. gmppt/tracking.py is NOT modified.

SELF-TEST:
    python -m gmppt.dynamic

WHAT REVISION 2 GOT WRONG -- defect D17

    Revision 2 built every profile with a FIXED steps_per_block. The slope was
    stored in the dataclass and never used to determine how many control steps
    the ramp occupied, so every slope in a sequence produced the identical block
    sequence and the identical result:

        30-100 @ 10 W/m2/s   ->  69.634%
        30-100 @ 100 W/m2/s  ->  69.634%

    Byte for byte, across all six slopes. A tenfold change in ramp speed changed
    nothing, which is impossible for a real dynamic test. The p10 comparison run
    on that profile therefore re-measured the STATIC result under a different
    name: P&O's 69.6% on shaded scenarios against 99.96% on uniform is
    local-peak trapping, already measured in P2.7 as a 42.1% arrival rate. No
    dynamic conclusion could be drawn from it.

    Slope only becomes meaningful once control steps are tied to TIME:

        steps in a ramp = t1 / T_control = dG / (slope * T_control)

    At a 100 ms control period, slope 10 gives 70 s -> 700 steps while slope 100
    gives 7 s -> 70 steps. A tracker that keeps up with the first may not keep up
    with the second, and that difference is the entire point of the dynamic test.

    Revision 3 derives per-block step counts from the standard's phase times and
    a declared control period. test_slope_changes_step_count is added as the
    check that would have caught D17, written so it CAN fail.

WHAT CHANGED IN REVISION 2 (retained)

    Revision 1 reconstructed the ramp profile from figure captions and
    arithmetic consistency. Alonso & Chenlo reproduce EN 50530's Annex B
    sequences in full as their Tables IV, V and VI. Three parameters were wrong:
    Sequence A has ELEVEN slopes not six; cycle counts VARY from 2 to 10 with
    slope; t0 = 300 s was absent; and the third sequence (1% => 10%) was missing
    entirely -- the case every measured inverter scored worst on.

    Two ambiguities were resolved. The standard's Figure 6 labels its y-axis
    "equivalent normalised irradiance G/G_STC", so ramps are linear in
    IRRADIANCE and the "% P_DCn" in the Annex B titles is a naming convention.
    And every tabulated ramp time satisfies t1 = t3 = dG / slope, which is how
    the tables were cross-checked.

    A later correction: t0 = 300 s applies per SLOPE ROW, not once per sequence.
    Confirmed from the published per-row durations -- Sequence B row 1 is
    10*(70+10+70+10) = 1600 s against a printed 1900 s.

SOURCES

    EN 50530:2010, consulted as SIST EN 50530:2011 (stated identical). Available
    portion: front matter through clause 4.3.2. Annexes B and C NOT available.

    M. Alonso and F. Chenlo, "Testing microinverters according to EN 50530",
    CIEMAT, Madrid. Reproduces Annex B as Tables IV, V and VI, and publishes
    measured dynamic efficiencies for three commercial inverters.

    EVERY TIMING BELOW COMES FROM ALONSO & CHENLO, not from the standard
    directly. That provenance travels with any figure produced here.

VERIFIED IN THE STANDARD ITSELF -- clause 3.4.1, equation (5)

        eta_dyn = [ sum_i U_DC,i * I_DC,i * dT_i ]
                  / [ sum_j P_MPP,PVS,j * dT_j ]

    Two consequences, both load-bearing:

      IT IS AN ENERGY RATIO, not a mean of instantaneous efficiencies. The two
      differ whenever available power varies, and the mean-of-ratios form would
      overweight the dim portions of a ramp where absolute errors are small.

      THE DENOMINATOR IS BLOCK-WISE. The sum over j runs across intervals of
      constant available power. Block discretisation is therefore the standard's
      OWN structure, not an approximation introduced here to save compute.

    Aggregation across slopes is an UNWEIGHTED mean (a_i = 1). Verified
    arithmetically: Alonso & Chenlo's eleven Sequence-A values average to
    99.116%, matching their printed total of 99.12%.

DECLARED DIVERGENCES -- five, all deliberate, all stated with any figure

  1. EQUIPMENT CLASS. Clause 1 scopes the standard to grid-connected inverters
     energising a low-voltage grid. This project measures a module-level DC
     optimizer: no AC side, no grid, no conversion efficiency. Only the
     MPPT-efficiency half is borrowed.

  2. SINGLE VERSUS MULTIPLE MPP. The standard assumes one MPP. Alonso & Chenlo
     state plainly that operation with shadowed I-V curves having more than one
     local MPP is not considered in EN 50530 and could be included for future
     developments. This project is entirely about that case, so the extension
     fills a gap the standard's own testers identified -- but it must be NAMED.

  3. HOW SHADING MEETS THE RAMP. The standard ramps irradiance uniformly. Here a
     FIXED shading pattern is held and every substring is scaled by the same
     factor, so the ratio between substrings -- which creates the peaks -- is
     preserved while the level follows the standard's ramp.

     MEASURED CONSEQUENCE: the GMPP voltage moves only about 2% of V_oc across a
     full 300-1000 W/m2 excursion, because current scales with irradiance while
     the peak voltage shifts only logarithmically. This profile moves the
     available POWER a great deal and the peak LOCATION barely at all. It is
     therefore a test of dynamic power tracking, NOT of dynamic global-peak
     relocation. Shading-PATTERN transitions, where the peak jumps between
     substrings, are the experiment that stresses the latter, and EN 50530
     cannot produce one by construction.

  4. PV GENERATOR MODEL. Annex C is normative and specifies a 1-diode model with
     technology-dependent parameters. It was not available. This project uses
     pvlib's single-diode model with CEC parameters, validated in Phase 1 to
     about 1% against 616 measured curves. Swapping simulators for the dynamic
     profile alone would make dynamic figures incomparable with every static
     result in the project, which is a worse problem than the model mismatch.

  5. CONTROL PERIOD AND TIME COMPRESSION. The partner's control period is
     unknown and is DECLARED at 100 ms below. A uniform time compression is
     applied per sequence so profiles fit a simulation budget; it is applied
     EQUALLY to every slope in a sequence, so the ratio between slopes -- which
     is what the test measures -- is preserved exactly.

THE VALIDATION TARGET -- the reason this harness can be trusted

    Alonso & Chenlo measured three commercial inverters under these sequences:

        Inverter PH (micro)   99.12%  99.21%  96.55%
        Inverter SM (string)  93.48%  98.07%  94.02%
        Inverter CY (micro)   85.91%  85.13%  86.33%
                              10-50%  30-100%  1-10%

    Running P&O on UNIFORM scenarios -- single peak, the case the standard was
    written for -- should land in roughly the same band. DECLARED: outside
    85-100%, the instrument is suspect and no shaded figure should be trusted.

    CAVEAT: those are inverters with their own converters and sampling rates.
    Agreement to a BAND is meaningful; agreement to a decimal place would be
    coincidence. Scoring ABOVE them is expected, because this simulation is
    noiseless and quasi-static -- it measures the algorithm without the converter
    losses real hardware carries.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import config, device
from .device import ModuleParams

N_SUB = int(config.N_SUBSTRINGS)

# ---------------------------------------------------------------------------
# EN 50530 ANNEX B, as reproduced by Alonso & Chenlo Tables IV, V and VI.
# Each row: (n_cycles, slope W/m2/s, t1 s, t2 s, t3 s, t4 s).
# ---------------------------------------------------------------------------

T0_WAIT_S = 300.0          # waiting time at the low level, per SLOPE ROW

# Sequence A -- 10% => 50%, 100 to 500 W/m2, dG = 400. Total 15 939 s.
SEQ_A_RANGE = (100.0, 500.0)
SEQ_A = (
    (2,   0.5, 800.0, 10.0, 800.0, 10.0),
    (2,   1.0, 400.0, 10.0, 400.0, 10.0),
    (3,   2.0, 200.0, 10.0, 200.0, 10.0),
    (4,   3.0, 133.0, 10.0, 133.0, 10.0),
    (6,   5.0,  80.0, 10.0,  80.0, 10.0),
    (8,   7.0,  57.0, 10.0,  57.0, 10.0),
    (10, 10.0,  40.0, 10.0,  40.0, 10.0),
    (10, 14.0,  29.0, 10.0,  29.0, 10.0),
    (10, 20.0,  20.0, 10.0,  20.0, 10.0),
    (10, 30.0,  13.0, 10.0,  13.0, 10.0),
    (10, 50.0,   8.0, 10.0,   8.0, 10.0),
)

# Sequence B -- 30% => 100%, 300 to 1000 W/m2, dG = 700. Total 6 987 s.
SEQ_B_RANGE = (300.0, 1000.0)
SEQ_B = (
    (10,  10.0, 70.0, 10.0, 70.0, 10.0),
    (10,  14.0, 50.0, 10.0, 50.0, 10.0),
    (10,  20.0, 35.0, 10.0, 35.0, 10.0),
    (10,  30.0, 23.0, 10.0, 23.0, 10.0),
    (10,  50.0, 14.0, 10.0, 14.0, 10.0),
    (10, 100.0,  7.0, 10.0,  7.0, 10.0),
)

# Sequence C -- 1% => 10%, 11 to 100 W/m2, dG = 90. Total 2 320 s.
# The low-light case. Every inverter Alonso & Chenlo measured scored WORST here.
SEQ_C_RANGE = (11.0, 100.0)
SEQ_C = (
    (1, 0.1, 980.0, 30.0, 980.0, 30.0),
)

SEQUENCES = {
    "10-50":  (SEQ_A_RANGE, SEQ_A),
    "30-100": (SEQ_B_RANGE, SEQ_B),
    "1-10":   (SEQ_C_RANGE, SEQ_C),
}

# Published measured dynamic efficiencies, for the validation bound.
PUBLISHED_DYN_EFF = {
    "10-50":  {"PH": 99.12, "SM": 93.48, "CY": 85.91},
    "30-100": {"PH": 99.21, "SM": 98.07, "CY": 85.13},
    "1-10":   {"PH": 96.55, "SM": 94.02, "CY": 86.33},
}
VALIDATION_BAND_PCT = (85.0, 100.0)

# ---------------------------------------------------------------------------
# TIMING -- the fix for D17
# ---------------------------------------------------------------------------

# DECLARED, NOT MEASURED. The partner's control period is outstanding.
# Published EN 50530 work finds at least 10 Hz is needed to reach 99% across all
# slopes, so 100 ms is the SLOWEST rate at which the funded target is reachable
# at all. Treat as a swept parameter.
CONTROL_PERIOD_S = 0.1

# Uniform time compression per sequence. SET TO 1.0 EVERYWHERE -- i.e. OFF.
# History (recorded, not silently corrected -- defect D20): compression was
# introduced to fit profiles into a simulation budget, under the mistaken belief
# that control-step count drives simulator cost. It does not -- simulator calls
# scale with BLOCK count, and the trajectory loop is interpolation, so
# compression bought nothing. When applied it was applied EQUALLY to every slope
# in a sequence, preserving the slope ratio the test measures; but at its former
# 10x on Sequence A it made that sequence's ramps 10x steeper than the standard
# specifies (5-500 W/m2/s against a specified 0.5-50), the probable cause of
# Sequence A's block-convergence divergence. Now 1.0 for every sequence.
# Sequence B was always 1.0, so its 99.833% figure is unaffected either way.
TIME_COMPRESSION = {"10-50": 1.0, "30-100": 1.0, "1-10": 1.0}

# t0 is a WAITING period before the measured cycles begin. Simulated so the
# tracker starts from a settled state, but EXCLUDED from the efficiency integral
# -- scoring initial acquisition would measure convergence, not dynamic
# tracking, and the two are reported separately everywhere else in this project.
T0_STEPS = 20

DEFAULT_BLOCKS_PER_PHASE = 6      # simulator calls per ramp; the cost knob
BLOCK_SWEEP = (3, 6, 12, 24)
BLOCK_CONVERGENCE_PT = 0.1


@dataclass
class RampProfile:
    """One EN 50530 trapezoid, discretised into blocks of constant irradiance.

    Each block carries its OWN step count, derived from the standard's phase
    times and the declared control period. That is what makes slope meaningful:
    a fast ramp occupies fewer control steps than a slow one over the same
    irradiance excursion (D17).

    `scored` marks which blocks enter the efficiency integral.
    """
    sequence: str
    slope_w_m2_s: float
    g_lo: float
    g_hi: float
    n_cycles: int
    levels: np.ndarray            # irradiance per block
    block_steps: np.ndarray       # control steps per block
    scored: np.ndarray            # bool per block
    t1: float
    t2: float
    t3: float
    t4: float
    compression: float
    control_period_s: float

    @property
    def n_blocks(self) -> int:
        return len(self.levels)

    @property
    def n_steps(self) -> int:
        return int(self.block_steps.sum())

    @property
    def scored_steps(self) -> int:
        return int(self.block_steps[self.scored].sum())

    def duration_s(self) -> float:
        """Profile duration by the standard's own arithmetic, uncompressed."""
        return T0_WAIT_S + self.n_cycles * (self.t1 + self.t2 + self.t3 + self.t4)

    def simulated_duration_s(self) -> float:
        """What the simulated trajectory represents, after compression."""
        return self.n_steps * self.control_period_s * self.compression

    @property
    def name(self) -> str:
        return f"{self.sequence} @ {self.slope_w_m2_s:g} W/m2/s"


def make_profile(sequence: str, slope: float,
                 blocks_per_phase: int = DEFAULT_BLOCKS_PER_PHASE,
                 control_period_s: float = CONTROL_PERIOD_S,
                 max_cycles: int | None = 2) -> RampProfile:
    """Build one Annex B row as a block sequence with slope-dependent timing.

    Block COUNT is fixed by blocks_per_phase and sets the simulator cost. Block
    DURATION comes from the standard's phase times divided by the control
    period, so a 7 s ramp occupies a tenth of the control steps of a 70 s ramp.

    `max_cycles` caps repetitions for tractability. The standard runs up to 10;
    each cycle is identical and the trackers are deterministic apart from PSO,
    so the cap loses no information. It is a DIVERGENCE and must be stated.
    """
    g_range, rows = SEQUENCES[sequence]
    row = next((r for r in rows if abs(r[1] - slope) < 1e-9), None)
    if row is None:
        raise ValueError(f"slope {slope} not in Annex B sequence {sequence}")

    n_cycles, slp, t1, t2, t3, t4 = row
    if max_cycles is not None:
        n_cycles = min(n_cycles, max_cycles)

    g_lo, g_hi = g_range
    comp = TIME_COMPRESSION.get(sequence, 1.0)

    # Exact ramp time from the standard's own relation, not the rounded print.
    t_ramp = (g_hi - g_lo) / slp

    def steps_for(seconds: float, n_blocks: int) -> int:
        """Control steps in ONE block of a phase lasting `seconds`."""
        total = seconds / (control_period_s * comp)
        return max(1, int(round(total / max(n_blocks, 1))))

    n_dwell_blocks = max(1, blocks_per_phase // 3)
    ramp_block_steps = steps_for(t_ramp, blocks_per_phase)
    dwell_hi_steps = steps_for(t2, n_dwell_blocks)
    dwell_lo_steps = steps_for(t4, n_dwell_blocks)

    levels, steps, scored = [], [], []

    # t0 waiting period: simulated so the tracker settles, NOT scored.
    levels.append(g_lo)
    steps.append(T0_STEPS)
    scored.append(False)

    up = np.linspace(g_lo, g_hi, blocks_per_phase + 2)[1:-1]
    for _ in range(n_cycles):
        for g in up:                                   # t1, ramp up
            levels.append(float(g))
            steps.append(ramp_block_steps)
            scored.append(True)
        for _d in range(n_dwell_blocks):               # t2, dwell high
            levels.append(g_hi)
            steps.append(dwell_hi_steps)
            scored.append(True)
        for g in up[::-1]:                             # t3, ramp down
            levels.append(float(g))
            steps.append(ramp_block_steps)
            scored.append(True)
        for _d in range(n_dwell_blocks):               # t4, dwell low
            levels.append(g_lo)
            steps.append(dwell_lo_steps)
            scored.append(True)

    return RampProfile(sequence=sequence, slope_w_m2_s=float(slp),
                       g_lo=float(g_lo), g_hi=float(g_hi),
                       n_cycles=int(n_cycles),
                       levels=np.asarray(levels, float),
                       block_steps=np.asarray(steps, int),
                       scored=np.asarray(scored, bool),
                       t1=t1, t2=t2, t3=t3, t4=t4,
                       compression=comp, control_period_s=control_period_s)


def sequence_profiles(sequence: str, **kw) -> list[RampProfile]:
    """Every slope in one Annex B sequence."""
    _, rows = SEQUENCES[sequence]
    return [make_profile(sequence, r[1], **kw) for r in rows]


def aggregate_efficiency(per_slope: list[float]) -> float:
    """The standard's eq. (7): unweighted mean across slopes, a_i = 1."""
    vals = [v for v in per_slope if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


# ---------------------------------------------------------------------------
# THE DYNAMIC TRAJECTORY
# ---------------------------------------------------------------------------

@dataclass
class DynamicTrajectory:
    """A tracker stepping through a sequence of curves, one per block.

    Blocks have DIFFERENT step counts (D17): a fast ramp occupies fewer control
    steps than a slow one. The block index therefore comes from cumulative step
    boundaries, not from division by a constant.
    """
    curves: list                  # [(V, P, v_gmpp, p_gmpp), ...] one per block
    block_steps: np.ndarray
    scored: np.ndarray
    profile_name: str = ""
    module: str = ""
    geometry: str = ""

    v_hist: list = field(default_factory=list)
    p_hist: list = field(default_factory=list)
    avail_hist: list = field(default_factory=list)
    block_hist: list = field(default_factory=list)

    _k: int = 0
    _bounds: np.ndarray | None = None

    def __post_init__(self):
        self._bounds = np.cumsum(np.asarray(self.block_steps, int))

    @property
    def n_steps(self) -> int:
        return int(self._bounds[-1])

    @property
    def block_index(self) -> int:
        i = int(np.searchsorted(self._bounds, self._k, side="right"))
        return min(i, len(self.curves) - 1)

    @property
    def v_oc(self) -> float:
        """V_oc of the CURRENT block's curve. It moves with irradiance."""
        return float(self.curves[self.block_index][0].max())

    @property
    def v_floor(self) -> float:
        V = self.curves[self.block_index][0]
        return max(0.02 * self.v_oc, float(V.min()))

    def step(self, v: float) -> float:
        """Command a voltage, advance one control period, return the power.

        Crossing a block boundary switches curves silently -- which is the
        point. A real tracker is not told that conditions changed.
        """
        V, P, _, p_gmpp = self.curves[self.block_index]
        v = float(min(max(v, self.v_floor), float(V.max())))
        p = float(np.interp(v, V, P))

        self.v_hist.append(v)
        self.p_hist.append(p)
        self.avail_hist.append(p_gmpp)
        self.block_hist.append(self.block_index)
        self._k += 1
        return p

    def block_changed(self) -> bool:
        """True if the previous step crossed a block boundary.

        Exposed for the SCHEDULED re-seed policy only. A detection-based trigger
        must not use this -- it would be reading the answer.
        """
        if len(self.block_hist) < 2:
            return False
        return self.block_hist[-1] != self.block_hist[-2]

    def metrics(self) -> dict:
        """EN 50530 eq. (5), over SCORED blocks only.

        The t0 waiting period is simulated so the tracker starts settled, but
        excluded from the integral: scoring initial acquisition would measure
        convergence rather than dynamic tracking.
        """
        if not self.p_hist:
            return {"n_steps": 0}

        p = np.asarray(self.p_hist, float)
        a = np.asarray(self.avail_hist, float)
        blocks = np.asarray(self.block_hist, int)
        keep = np.asarray(self.scored, bool)[blocks]

        if not keep.any():
            return {"n_steps": int(len(p)), "n_scored_steps": 0,
                    "dynamic_efficiency_pct": float("nan")}

        ps, as_ = p[keep], a[keep]
        captured = float(ps.sum())
        available = float(as_.sum())
        eta = captured / available if available > 0 else float("nan")

        per_block = []
        for b in np.unique(blocks[keep]):
            sel = blocks == b
            av = float(a[sel].sum())
            per_block.append(float(p[sel].sum() / av) if av > 0 else np.nan)

        inst = np.divide(ps, as_, out=np.zeros_like(ps), where=as_ > 0)
        return {
            "n_steps": int(len(p)),
            "n_scored_steps": int(keep.sum()),
            "n_blocks": int(len(np.unique(blocks))),
            "dynamic_efficiency_pct": 100.0 * eta,
            "energy_captured": captured,
            "energy_available": available,
            "energy_lost": available - captured,
            "worst_block_efficiency_pct": (100.0 * float(np.nanmin(per_block))
                                           if per_block else float("nan")),
            "worst_instant_loss_w": float((as_ - ps).max()),
            "mean_instant_efficiency_pct": 100.0 * float(inst.mean()),
            "profile": self.profile_name,
            "module": self.module,
            "geometry": self.geometry,
        }


def dynamic_trajectory_for(scenario, profile: RampProfile) -> DynamicTrajectory:
    """Build a DynamicTrajectory: one simulator call per block.

    THE SHADING PATTERN IS PRESERVED. Every substring is scaled by the same
    factor, so the ratio between substrings -- which creates the multiple peaks
    -- is unchanged while the overall level follows the standard's ramp.
    """
    mp = ModuleParams.from_cec(scenario.module)
    base = list(np.atleast_1d(np.asarray(scenario.irradiances, dtype=object)))
    ref = 1000.0

    def scaled(entry, factor):
        arr = np.atleast_1d(np.asarray(entry, float))
        out = arr * factor
        return float(out[0]) if out.size == 1 else out.tolist()

    curves = []
    for g in profile.levels:
        factor = float(g) / ref
        irr = [scaled(e, factor) for e in base]
        c = device.module_iv(mp, irr, scenario.temp_c,
                             n_substrings=N_SUB, bd=config.breakdown())
        a = device.analyse(c)

        V = np.asarray(c["V"], float)
        P = np.asarray(c["P"], float)
        order = np.argsort(V)          # raw output is DESCENDING -- see D9
        V, P = V[order], P[order]

        v_gmpp = float(a["gmpp"]["V"])
        p_gmpp = float(a["gmpp"]["P"])

        p_check = float(np.interp(v_gmpp, V, P))
        if abs(p_check - p_gmpp) > 1e-3 * max(abs(p_gmpp), 1e-9):
            raise RuntimeError(
                f"block curve does not reproduce its GMPP for "
                f"{scenario.module} at {g:.0f} W/m2: interpolated "
                f"{p_check:.4f} W, analyse reports {p_gmpp:.4f} W")

        curves.append((V, P, v_gmpp, p_gmpp))

    return DynamicTrajectory(curves=curves,
                             block_steps=profile.block_steps,
                             scored=profile.scored,
                             profile_name=profile.name,
                             module=str(scenario.module),
                             geometry=str(scenario.geometry))


# ---------------------------------------------------------------------------
# VERIFICATION
# ---------------------------------------------------------------------------

def test_tables_self_consistent(verbose: bool = True) -> bool:
    """Every Annex B row must satisfy t1 = t3 = dG / slope, and each sequence's
    total must equal sum over rows of [t0 + n*(t1+t2+t3+t4)].

    NOTE ON t0: the 300 s waiting period applies to EVERY SLOPE ROW, not once
    per sequence. Confirmed from the published per-row durations.

    NOTE ON ROUNDING: the paper prints t1 rounded to whole seconds (23 where
    700/30 = 23.33). Exact values are used for the arithmetic; printed values
    are checked to 5%.

    KNOWN ANOMALY, Sequence C: the paper gives t1 = 980 s at slope 0.1 W/m2/s
    over a stated range of 11-100 W/m2. Neither 89/0.1 nor "Delta 90"/0.1 gives
    980. Its printed duration of 2320 s DOES follow from the stated timings, so
    they are used as given and the relation check is skipped for that sequence.
    Recorded rather than silently corrected.
    """
    totals = {"10-50": 15939.0, "30-100": 6987.0, "1-10": 2320.0}
    skip_relation = {"1-10"}
    ok = True

    for seq, (g_range, rows) in SEQUENCES.items():
        dg = g_range[1] - g_range[0]
        total = 0.0
        for n, slope, t1, t2, t3, t4 in rows:
            exact = dg / slope
            if seq not in skip_relation:
                if abs(t1 - exact) > max(1.0, 0.05 * exact):
                    ok = False
                    if verbose:
                        print(f"   {seq} slope {slope}: t1={t1}, "
                              f"expected {exact:.1f}")
                t1_use = t3_use = exact
            else:
                t1_use, t3_use = t1, t3
            total += T0_WAIT_S + n * (t1_use + t2 + t3_use + t4)

        stated = totals[seq]
        close = abs(total - stated) <= 0.01 * stated
        ok &= close
        if verbose:
            print(f"   {seq:<8} computed {total:>8.0f} s, "
                  f"published {stated:>8.0f} s  "
                  f"{'match' if close else 'MISMATCH'}")

    if verbose:
        print("   t0 = 300 s applies per SLOPE ROW, not per sequence.")
        print("   Sequence C's t1 does not follow dG/slope; its published")
        print("   duration confirms the printed timings, so they stand.")
        print(f"   -> {'PASS' if ok else 'FAIL'}")
    return ok


def test_slope_changes_step_count(verbose: bool = True) -> bool:
    """Different slopes must produce different control-step counts.

    THE CHECK THAT WOULD HAVE CAUGHT D17. Revision 2 stored the slope and never
    used it, so all six slopes in a sequence produced identical profiles and
    identical results -- 69.634% at both 10 and 100 W/m2/s. Written so it can
    fail: it requires the fastest slope to occupy at MOST half the steps of the
    slowest, which a tenfold slope range must produce.
    """
    _, rows = SEQUENCES["30-100"]
    profs = {r[1]: make_profile("30-100", r[1], max_cycles=1) for r in rows}
    slow = profs[10.0].n_steps
    fast = profs[100.0].n_steps
    ok = fast <= 0.5 * slow
    if verbose:
        print(f"   {'slope':>8} {'steps':>8} {'scored':>8} {'sim seconds':>13}")
        for s, p in profs.items():
            print(f"   {s:>8.0f} {p.n_steps:>8} {p.scored_steps:>8} "
                  f"{p.simulated_duration_s():>13.1f}")
        print(f"   fastest/slowest step ratio: {fast/slow:.3f}")
        print(f"   -> {'PASS' if ok else 'FAIL -- slope is not affecting the profile (D17)'}")
    return ok


def test_aggregation_reproduces_published(verbose: bool = True) -> bool:
    """eq. (7) must reproduce Alonso & Chenlo's printed sequence totals.

    Their Table VII lists eleven Sequence-A values for Inverter PH; the
    unweighted mean must give their printed 99.12%. If it does not, the
    aggregation is weighted differently from the standard's.
    """
    seq_a_ph = [99.36, 99.35, 99.34, 99.31, 99.30, 99.22,
                99.23, 99.16, 98.71, 98.70, 98.60]
    seq_b_ph = [99.38, 99.06, 99.35, 99.29, 99.21, 98.99]

    got_a = aggregate_efficiency(seq_a_ph)
    got_b = aggregate_efficiency(seq_b_ph)
    ok = abs(got_a - 99.12) < 0.02 and abs(got_b - 99.21) < 0.02
    if verbose:
        print(f"   Sequence A: computed {got_a:.3f}%, published 99.12%")
        print(f"   Sequence B: computed {got_b:.3f}%, published 99.21%")
        print(f"   -> {'PASS' if ok else 'FAIL'}")
    return ok


def test_ceiling_moves(verbose: bool = True) -> bool:
    """Available power must CHANGE across the profile.

    If it does not, the harness is running a static test wearing a dynamic name.
    Written so it can fail: a 300-1000 W/m2 excursion must vary the ceiling by
    at least a factor of two.
    """
    from .harness import scenario_set

    sc = scenario_set(4)[0]
    prof = make_profile("30-100", 20.0)
    traj = dynamic_trajectory_for(sc, prof)

    ceilings = np.array([c[3] for c in traj.curves])
    ratio = float(ceilings.max() / max(ceilings.min(), 1e-9))
    ok = ratio >= 2.0
    if verbose:
        print(f"   available power {ceilings.min():.1f} to {ceilings.max():.1f} W "
              f"(ratio {ratio:.2f}x)")
        print(f"   -> {'PASS' if ok else 'FAIL'} (must vary by >= 2x)")
    return ok


def test_peak_moves(verbose: bool = True) -> bool:
    """The GMPP VOLTAGE must move.

    A moving ceiling with a stationary peak would let a one-shot predictor score
    perfectly without tracking anything. Note the measured movement is small --
    about 2% of V_oc -- which is declared as divergence 3: this profile tests
    dynamic POWER tracking, not dynamic peak RELOCATION.
    """
    from .harness import scenario_set

    sc = scenario_set(4)[0]
    prof = make_profile("30-100", 20.0)
    traj = dynamic_trajectory_for(sc, prof)

    v_peaks = np.array([c[2] for c in traj.curves])
    v_oc = float(traj.curves[0][0].max())
    spread = float(v_peaks.max() - v_peaks.min())
    ok = spread > 0.01 * v_oc
    if verbose:
        print(f"   GMPP voltage {v_peaks.min():.2f} to {v_peaks.max():.2f} V "
              f"(spread {spread:.2f} V, {100*spread/v_oc:.1f}% of Voc)")
        print(f"   -> {'PASS' if ok else 'FAIL'} (must move > 1% of Voc)")
        print("   NOTE: this is small. Under proportional scaling the current")
        print("   scales with irradiance while the peak VOLTAGE shifts only")
        print("   logarithmically. See declared divergence 3.")
    return ok


def test_against_published_inverters(verbose: bool = True) -> bool:
    """P&O on UNIFORM scenarios must land in the published measured band.

    Alonso & Chenlo measured three commercial inverters under these sequences.
    Uniform irradiance is a single-peak case -- what the standard was written
    for -- so a competent tracker should land in roughly the same range.

    DECLARED: outside 85-100%, the instrument is suspect and no shaded figure
    should be trusted until the discrepancy is understood.
    """
    from .harness import scenario_set
    from .tracking import DEFAULT_STEP_FRAC, START_FRACTION

    uniform = [s for s in scenario_set(60) if s.geometry == "uniform"][:5]
    if not uniform:
        if verbose:
            print("   no uniform scenarios drawn -- check NOT TESTED, "
                  "which is not a pass")
        return False

    results = []
    for seq in ("10-50", "30-100"):
        _, rows = SEQUENCES[seq]
        per_slope = []
        for row in rows:
            effs = []
            for sc in uniform:
                prof = make_profile(seq, row[1], max_cycles=1)
                traj = dynamic_trajectory_for(sc, prof)

                dv = DEFAULT_STEP_FRAC * traj.v_oc
                v = START_FRACTION * traj.v_oc
                p_prev = traj.step(v)
                direction = -1.0
                for _ in range(traj.n_steps - 1):
                    v = v + direction * dv
                    p = traj.step(v)
                    if p < p_prev:
                        direction = -direction
                    p_prev = p
                effs.append(traj.metrics()["dynamic_efficiency_pct"])
            per_slope.append(float(np.mean(effs)))
        results.append((seq, aggregate_efficiency(per_slope), per_slope))

    lo, hi = VALIDATION_BAND_PCT
    ok = True
    if verbose:
        print(f"   {'sequence':<10} {'P&O uniform':>13} {'published range':>28}")
    for seq, total, per_slope in results:
        pub = PUBLISHED_DYN_EFF[seq]
        band = f"{min(pub.values()):.2f} - {max(pub.values()):.2f}%"
        inside = lo <= total <= hi
        ok &= inside
        if verbose:
            print(f"   {seq:<10} {total:>12.2f}% {band:>28}")
    if verbose:
        _, rows_b = SEQUENCES["30-100"]
        per_b = results[1][2]
        print(f"\n   per-slope, sequence 30-100 (does speed matter now?):")
        for (n, slope, *_), eff in zip(rows_b, per_b):
            print(f"     {slope:>6.0f} W/m2/s  {eff:>7.3f}%")
        print(f"\n   declared acceptable band: {lo:.0f}-{hi:.0f}%")
        print(f"   -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            print("   The instrument does not reproduce published behaviour on")
            print("   the single-peak case the standard was written for. Do NOT")
            print("   trust any shaded figure until this is understood.")
    return ok


def test_shading_pattern_preserved(verbose: bool = True) -> bool:
    """Diagnostic: does the peak COUNT stay fixed across the ramp?

    Proportional scaling should preserve it. If the count varies, the curve
    changes SHAPE as well as scale, and efficiency loss would be partly
    attributable to that rather than to tracking lag. Not a gate; a fact to
    report.
    """
    from .harness import scenario_set

    scenarios = [s for s in scenario_set(60) if s.geometry != "uniform"][:5]
    prof = make_profile("30-100", 20.0, max_cycles=1)
    varied = 0
    if verbose:
        print(f"   {'module':<30} {'peak counts':>18}")
    for sc in scenarios:
        traj = dynamic_trajectory_for(sc, prof)
        counts = []
        for V, P, _, _ in traj.curves:
            pk = [i for i in range(1, len(P) - 1)
                  if P[i] > P[i - 1] and P[i] >= P[i + 1]]
            counts.append(len(pk))
        if len(set(counts)) > 1:
            varied += 1
        if verbose:
            print(f"   {sc.module[:30]:<30} {str(sorted(set(counts))):>18}")
    if verbose:
        print(f"   {varied}/{len(scenarios)} scenarios change peak count "
              f"across the ramp")
        if varied:
            print("   Report this: efficiency loss on those scenarios is partly")
            print("   attributable to the curve changing shape.")
    return True


def main() -> int:
    print("P2.8  EN 50530 dynamic harness, extended to partial shading  (rev 3)")
    try:
        print("  " + config.provenance())
    except Exception:
        pass
    print("=" * 78)
    print("Annex B sequences from Alonso & Chenlo (CIEMAT), Tables IV-VI.")
    print("Efficiency definition from EN 50530:2010 clause 3.4.1 eq. (5).")
    print(f"\ncontrol period {1000*CONTROL_PERIOD_S:.0f} ms (DECLARED, not "
          f"measured -- the partner's")
    print("value is outstanding). Time compression per sequence: "
          f"{TIME_COMPRESSION}")
    print("applied EQUALLY across slopes, so the ratio between them is exact.")
    print("\nDECLARED DIVERGENCES: module-level optimizer not a grid inverter;")
    print("multi-peak curves outside the standard's scope by its own statement;")
    print("shading applied by proportional scaling, which moves the available")
    print("POWER but the peak LOCATION by only ~2% of Voc; pvlib CEC generator")
    print("model rather than the standard's Annex C model; declared control")
    print("period and time compression.\n")

    print("1. ANNEX B TABLES ARE SELF-CONSISTENT  (transcription check)")
    a = test_tables_self_consistent()

    print("\n2. SLOPE CHANGES THE CONTROL-STEP COUNT  (D17)")
    b = test_slope_changes_step_count()

    print("\n3. AGGREGATION REPRODUCES PUBLISHED TOTALS  (eq. 7)")
    c = test_aggregation_reproduces_published()

    print("\n4. THE AVAILABLE POWER MOVES")
    d = test_ceiling_moves()

    print("\n5. THE PEAK VOLTAGE MOVES")
    e = test_peak_moves()

    print("\n6. P&O ON UNIFORM SCENARIOS vs PUBLISHED INVERTERS")
    f = test_against_published_inverters()

    print("\n7. DOES THE SHADING STRUCTURE SURVIVE SCALING?  (diagnostic)")
    test_shading_pattern_preserved()

    print("\n" + "=" * 78)
    passed = sum([a, b, c, d, e, f])
    print(f"{passed}/6 checks passed")
    if passed == 6:
        print("\nThe profile is transcribed correctly, RESPONDS TO SLOPE,")
        print("aggregates as the standard does, moves both the ceiling and the")
        print("peak, and reproduces published measured behaviour on the")
        print("single-peak case. Shaded figures measured on it can be trusted.")
    else:
        print("\nDo NOT score anything on this profile until every check passes.")
    return 0 if passed == 6 else 1


if __name__ == "__main__":
    raise SystemExit(main())