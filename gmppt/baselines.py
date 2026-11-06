from __future__ import annotations

import numpy as np

from . import config
from .harness import Context, constant_method, register  # noqa: F401


# ---------------------------------------------------------------------------
# The alpha curve, paper Fig. 8.
# EXPLICIT in the text: 1000 -> 0.80, 900 -> 0.83, 800 -> 0.85, 700 -> 0.87,
#                       300 -> 0.945, 100 -> 0.97
# DIGITISED from Fig. 8 (approximate, roughly +/- 0.01): 200, 400, 500, 600
# ---------------------------------------------------------------------------
ALPHA_G = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], float)
ALPHA_V = np.array([0.970, 0.960, 0.945, 0.920, 0.905, 0.890,
                    0.870, 0.850, 0.830, 0.800], float)
ALPHA_DIGITISED = {200, 400, 500, 600}     # not stated numerically in the paper

LEVEL_TOL = 1.0          # W/m2; irradiances closer than this are one subassembly

# --- detector parameters. BOTH ARE DIMENSIONLESS FRACTIONS, so neither is
# --- coupled to how densely the curve is sampled. That coupling was the rev-4 bug.
SCAN_PROBES = 60         # default probe budget; p3c shows 30 performs identically
PLATEAU_TOL = 0.04       # a plateau holds current within 4% of Isc of its median
MIN_RUN_FRAC = 0.06      # a plateau must span >= 6% of the probe budget
ALPHA_MID = 0.90         # mid-range alpha, used only for group-size estimation


def alpha_of(g: float) -> float:
    """alpha at irradiance g, linearly interpolated on the Fig. 8 curve.

    Clips at 100 and 1000 W/m2. See ALPHA RANGE LIMIT -- 12.2% of sub_substring
    scenarios land on the lower clip.
    """
    return float(np.interp(float(g), ALPHA_G, ALPHA_V))


# ---------------------------------------------------------------------------
# Irradiance handling (true-G path, used by the oracle variants)
# ---------------------------------------------------------------------------

def _reduce_entry(entry) -> float:
    """One substring's effective irradiance: the minimum across its cell groups,
    because series elements share a current set by the weakest."""
    arr = np.atleast_1d(np.asarray(entry, dtype=float))
    return float(arr.min())


def substring_irradiances(irr) -> list[float]:
    """Reduce a scenario's irradiance description to one value per substring.

    Handles RAGGED lists that mix scalars and sequences across substrings:
      [1000, 1000, 400]              -> [1000, 1000, 400]
      [1000, [600, 200], 1000]       -> [1000, 200, 1000]
      [[1000, 900], [400, 400], 800] -> [900, 400, 800]
    """
    n_sub = int(config.N_SUBSTRINGS)
    if irr is None:
        raise ValueError(
            "no irradiances on the Context -- gmppt/harness.py must be revision 3 "
            "or later (Context needs the `conditions` field).")
    entries = list(irr)
    if len(entries) == n_sub:
        return [_reduce_entry(e) for e in entries]
    flat = np.asarray(entries, dtype=float).ravel()
    if flat.size % n_sub == 0:
        return [float(x) for x in flat.reshape(n_sub, -1).min(axis=1)]
    raise ValueError(
        f"cannot map an irradiance list of length {len(entries)} onto {n_sub} "
        f"substrings")


def subassemblies(g_sub: list[float]) -> list[tuple[float, int]]:
    """Group substrings by irradiance level, brightest first: [(G, N), ...]."""
    groups: list[list[float]] = []
    for g in sorted(g_sub, reverse=True):
        if groups and abs(groups[-1][0] - g) <= LEVEL_TOL:
            groups[-1].append(g)
        else:
            groups.append([g])
    return [(float(np.mean(grp)), len(grp)) for grp in groups]


# ---------------------------------------------------------------------------
# Scan-based inference
# ---------------------------------------------------------------------------

def scan_steps(ctx: Context, n_probes: int = SCAN_PROBES,
               plateau_tol: float = PLATEAU_TOL,
               min_run_frac: float = MIN_RUN_FRAC,
               return_samples: bool = False):
    """Locate the I-V staircase by probing.

    Returns (levels, step_voltages), or (levels, step_voltages, vs, ps) when
    `return_samples` is set.

    levels          plateau currents, brightest (highest) first
    step_voltages   voltage midway between consecutive plateaus, len(levels)-1
    vs, ps          the probed voltages and their MEASURED POWERS

    Density-independent by construction: a plateau is a RUN of samples holding
    current within `plateau_tol` of the run median (a fraction of Isc, not of a
    neighbour gap), spanning at least `min_run_frac` of the budget. Both are
    dimensionless, so the same staircase is seen at 30 and at 240 probes.

    Current is recovered as I = P / V, so the power-only probe interface
    suffices. Every probe is counted by the harness -- that count is the
    measurement cost of not knowing the irradiance.

    REVISION 7: `return_samples` exposes (vs, ps) so a caller can use the
    measurements it has already been charged for. The staircase logic is
    untouched, and the default return shape is unchanged, so every existing
    caller and every prior number is unaffected.
    """
    v_oc = float(ctx.v_oc)
    n_probes = max(8, int(n_probes))
    vs = np.linspace(0.03 * v_oc, 0.97 * v_oc, n_probes)
    ps = np.array([ctx.probe(float(v)) for v in vs], dtype=float)

    def _ret(levels, steps):
        return (levels, steps, vs, ps) if return_samples else (levels, steps)

    with np.errstate(divide="ignore", invalid="ignore"):
        cur = np.where(vs > 0, ps / vs, 0.0)
    cur = np.nan_to_num(cur, nan=0.0, posinf=0.0, neginf=0.0)

    i_max = float(cur.max())
    if not np.isfinite(i_max) or i_max <= 0:
        return _ret([1.0], [])

    tol = plateau_tol * i_max
    min_run = max(2, int(round(min_run_frac * n_probes)))

    runs, start = [], 0
    for j in range(1, n_probes):
        seg = cur[start:j + 1]
        if (seg.max() - seg.min()) > 2 * tol:
            runs.append((start, j - 1))
            start = j
    runs.append((start, n_probes - 1))

    plateaus = [(a, b) for (a, b) in runs if (b - a + 1) >= min_run]
    if not plateaus:
        return _ret([float(i_max)], [])

    levels = [float(np.median(cur[a:b + 1])) for a, b in plateaus]
    mids = [float(0.5 * (vs[plateaus[k][1]] + vs[plateaus[k + 1][0]]))
            for k in range(len(plateaus) - 1)]

    keep_levels, keep_steps = [levels[0]], []
    for lv, st in zip(levels[1:], mids):
        if lv < keep_levels[-1] * (1 - 1e-3):
            keep_levels.append(lv)
            keep_steps.append(st)

    # An optimizer knows its own module has N_SUBSTRINGS substrings, so more
    # levels than substrings is a detection artefact -- merge the shallowest gaps.
    n_sub = int(config.N_SUBSTRINGS)
    while len(keep_levels) > n_sub:
        gaps = [keep_levels[k] - keep_levels[k + 1]
                for k in range(len(keep_levels) - 1)]
        k = int(np.argmin(gaps))
        keep_levels.pop(k + 1)
        keep_steps.pop(k)

    return _ret(keep_levels, keep_steps)


def infer_group_sizes(step_voltages, n_groups: int, v_oc_elem: float) -> list[int]:
    """Estimate N_k from the step voltages. OUR EXTENSION -- the paper is silent
    on recovering group sizes from a scan; its examples simply state them.

    Region k ends near (N_1+...+N_k) * alpha * Voc_element, so the cumulative
    element count is read off each step voltage and differenced. Clipped so every
    group holds at least one element and the total is exactly N_SUBSTRINGS.

    p3b showed this is not load-bearing.
    """
    n_sub = int(config.N_SUBSTRINGS)
    if n_groups <= 1:
        return [n_sub]
    if n_groups >= n_sub:
        return [1] * n_sub

    cums, prev = [], 0
    for k, v_step in enumerate(step_voltages[:n_groups - 1]):
        raw = v_step / (ALPHA_MID * v_oc_elem) if v_oc_elem > 0 else k + 1
        lo = prev + 1
        hi = n_sub - (n_groups - 1 - k)
        cum = int(min(max(round(raw), lo), hi))
        cums.append(cum)
        prev = cum

    sizes, prev = [], 0
    for cum in cums:
        sizes.append(cum - prev)
        prev = cum
    sizes.append(n_sub - prev)
    return sizes


# ---------------------------------------------------------------------------
# Equation (27)
# ---------------------------------------------------------------------------

def predicted_peaks(v_oc_module: float,
                    groups: list[tuple[float, int]],
                    anchor: str = "absolute") -> list[float]:
    """Equation (27) per subassembly. groups = [(level, N_k), ...] brightest first.

    anchor="absolute"  alpha at the group's own irradiance (Fig. 8 as printed);
                       `level` must be an irradiance.
    anchor="ratio"     alpha at 1000 * level_k / level_{k-1}; works with any
                       proportional quantity, so scanned current levels are used
                       directly with no absolute calibration -- eq. (14).
    """
    n_sub = int(config.N_SUBSTRINGS)
    v_oc_elem = v_oc_module / n_sub

    peaks, cumulative = [], 0
    for idx, (level, n_k) in enumerate(groups):
        if idx == 0:
            offset = 0.0
        elif anchor == "ratio":
            prev = groups[idx - 1][0]
            ratio = (level / prev) if prev > 0 else 1.0
            offset = alpha_of(1000.0 * min(max(ratio, 0.0), 1.0)) * cumulative
        else:
            offset = alpha_of(level) * cumulative
        peaks.append((offset + 0.80 * n_k) * v_oc_elem)
        cumulative += n_k
    return peaks


def _land_best(ctx: Context, peaks: list[float]) -> tuple[float, float]:
    """Probe each predicted peak; return (voltage, power) of the best.

    REVISION 7. Same probe count as the old _land -- one probe per candidate.
    Handing the power back means a caller can compare the landing against other
    measurements WITHOUT spending another probe, which would otherwise corrupt
    the probe-cost comparison the whole claim rests on.
    """
    cands = [min(max(v, 0.0), ctx.v_oc) for v in peaks]
    pairs = [(v, float(ctx.probe(v))) for v in cands]
    return max(pairs, key=lambda vp: vp[1])


def _land(ctx: Context, peaks: list[float]) -> float:
    """Probe each predicted peak, keep the highest-power one."""
    return _land_best(ctx, peaks)[0]


# ---------------------------------------------------------------------------
# The registered variants
# ---------------------------------------------------------------------------

@register("ahmed_salam")
def ahmed_salam(ctx: Context) -> float:
    """True G, true group sizes, alpha at absolute irradiance.

    UPPER BOUND: the paper states per-module irradiance cannot feasibly be
    measured, so this variant cannot be built as tested. Reported as a ceiling.

    Bounds the INPUT QUALITY to eq. (27), not eq. (27) itself -- see the
    revised write-up note in the module docstring.
    """
    groups = subassemblies(substring_irradiances(ctx.conditions.get("irradiances")))
    return _land(ctx, predicted_peaks(ctx.v_oc, groups, anchor="absolute"))


@register("ahmed_salam_ratio")
def ahmed_salam_ratio(ctx: Context) -> float:
    """True G, true group sizes, alpha at the irradiance RATIO. UPPER BOUND."""
    groups = subassemblies(substring_irradiances(ctx.conditions.get("irradiances")))
    return _land(ctx, predicted_peaks(ctx.v_oc, groups, anchor="ratio"))


@register("ahmed_salam_scan")
def ahmed_salam_scan(ctx: Context) -> float:
    """Fully scan-based: G and group sizes both inferred. THE PUBLISHED METHOD.

    Uses only what an optimizer can observe -- a sweep of its own module.
    Returns the formula's landing, as the paper specifies, without consulting
    the scan's own power samples. UNCHANGED from revision 6, so every prior
    number reproduces exactly.

    For the deployable form, see ahmed_salam_scan_best below; report both.
    """
    n_sub = int(config.N_SUBSTRINGS)
    levels, steps = scan_steps(ctx)
    sizes = infer_group_sizes(steps, len(levels), ctx.v_oc / n_sub)
    return _land(ctx, predicted_peaks(ctx.v_oc, list(zip(levels, sizes)),
                                      anchor="ratio"))


@register("ahmed_salam_scan_best")
def ahmed_salam_scan_best(ctx: Context) -> float:
    """Ahmed-Salam as it would actually be deployed. NEW IN REVISION 7.

    Identical inference and identical probe budget to ahmed_salam_scan. The one
    difference: having already measured SCAN_PROBES operating points while
    detecting the staircase, it keeps the best of them when the formula's
    landing is worse.

    Measured on validation, the formula's landing is further from the true GMPP
    than the scan's own best sample in 75% of scenarios, so the published form
    discards measurements it has been charged for. No deployed tracker would do
    that, and a reviewer familiar with the method will ask why it did.

    THIS IS THE ROW A COST OR ACCURACY CLAIM MUST BE MADE AGAINST. By
    construction it cannot be worse than ahmed_salam_scan at equal probe count
    -- it returns the better of two points it has already measured. If p3b
    reports it as worse, or as costing more probes, the implementation is wrong.
    """
    n_sub = int(config.N_SUBSTRINGS)
    levels, steps, vs, ps = scan_steps(ctx, return_samples=True)
    sizes = infer_group_sizes(steps, len(levels), ctx.v_oc / n_sub)
    peaks = predicted_peaks(ctx.v_oc, list(zip(levels, sizes)), anchor="ratio")

    v_land, p_land = _land_best(ctx, peaks)
    if ps.size and float(ps.max()) > p_land:
        return float(vs[int(np.argmax(ps))])
    return v_land


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

class _FakeContext:
    """A synthetic staircase, for testing the detector without the simulator."""

    def __init__(self, levels, break_fracs, v_oc=36.0):
        self.v_oc = float(v_oc)
        self._levels = list(levels)
        self._breaks = list(break_fracs)
        self.n_probes = 0
        self.conditions = {}

    def probe(self, v):
        self.n_probes += 1
        frac = float(v) / self.v_oc
        cur = self._levels[-1]
        for lv, br in zip(self._levels, self._breaks + [1.1]):
            if frac < br:
                cur = lv
                break
        return cur * float(v)


def test_ahmed_salam_paper_example(verbose: bool = True) -> bool:
    """Equation (27) against the paper's Case 2. Pure arithmetic, no simulator."""
    v_oc_elem = 21.1
    groups = [(1000.0, 4), (700.0, 4), (300.0, 2)]
    expected = [67.52, 140.94, 193.27]

    got, cumulative = [], 0
    for idx, (g_k, n_k) in enumerate(groups):
        offset = 0.0 if idx == 0 else alpha_of(g_k) * cumulative
        got.append((offset + 0.80 * n_k) * v_oc_elem)
        cumulative += n_k

    ok = all(abs(a - b) < 0.5 for a, b in zip(got, expected))
    if verbose:
        print("Ahmed-Salam equation (27) vs paper Case 2 (MSX60, Voc 21.1 V):")
        for i, (a, b) in enumerate(zip(got, expected), start=1):
            print(f"   peak {i}: computed {a:7.2f} V   paper {b:7.2f} V   "
                  f"diff {abs(a-b):.2f} V")
        print(f"   -> {'PASS' if ok else 'FAIL'} (tolerance 0.5 V)")
    return ok


def test_ragged_irradiances(verbose: bool = True) -> bool:
    """Ragged irradiance reduction. Assumes N_SUBSTRINGS == 3."""
    if int(config.N_SUBSTRINGS) != 3:
        if verbose:
            print(f"   skipped: N_SUBSTRINGS is {config.N_SUBSTRINGS}, not 3")
        return True
    cases = [
        ([1000, 1000, 400], [1000.0, 1000.0, 400.0]),
        ([1000, [600, 200], 1000], [1000.0, 200.0, 1000.0]),
        ([[1000, 900], [400, 400], 800], [900.0, 400.0, 800.0]),
        ([[1000, 1000], [500, 300], [200, 900]], [1000.0, 300.0, 200.0]),
    ]
    ok = True
    for raw, want in cases:
        got = substring_irradiances(raw)
        hit = all(abs(a - b) < 1e-9 for a, b in zip(got, want))
        ok &= hit
        if verbose:
            print(f"   {str(raw):<34} -> {got}  {'ok' if hit else 'MISMATCH'}")
    if verbose:
        print(f"   -> {'PASS' if ok else 'FAIL'}")
    return ok


def test_detector_invariance(verbose: bool = True) -> bool:
    """The detector must see the same staircase regardless of sampling density.

    This is what revision 4 failed, and no loss number would have revealed it --
    only invariance does. Criterion: across 30/60/120/240 probes the detected
    level COUNT must be identical and each level value agree within 2%.
    """
    cases = [
        ("three levels", [5.0, 3.2, 1.4], [0.34, 0.67]),
        ("two levels",   [4.5, 2.0],      [0.5]),
        ("one level",    [4.0],           []),
    ]
    budgets = (30, 60, 120, 240)
    ok = True

    for name, levels, breaks in cases:
        seen = []
        for n in budgets:
            ctx = _FakeContext(levels, breaks)
            got, _ = scan_steps(ctx, n_probes=n)
            seen.append(got)
        counts = {len(g) for g in seen}
        same_count = len(counts) == 1
        stable = True
        if same_count:
            for k in range(len(seen[0])):
                vals = [g[k] for g in seen]
                if max(vals) - min(vals) > 0.02 * max(vals):
                    stable = False
        hit = same_count and stable and len(seen[0]) == len(levels)
        ok &= hit
        if verbose:
            print(f"   {name:<14} true {len(levels)} level(s); detected "
                  f"{[len(g) for g in seen]} at probes {list(budgets)}  "
                  f"{'ok' if hit else 'UNSTABLE'}")
            if not hit:
                for n, g in zip(budgets, seen):
                    print(f"      {n:>4} probes -> {[round(x, 3) for x in g]}")
    if verbose:
        print(f"   -> {'PASS' if ok else 'FAIL'}")
    return ok


def test_return_samples(verbose: bool = True) -> bool:
    """REVISION 7: return_samples must not change the staircase it detects.

    The new keyword exists only to expose measurements already taken. If the
    detected levels or steps differ with the flag set, the refactor broke the
    detector and every ahmed_salam row is suspect.
    """
    cases = [
        ("three levels", [5.0, 3.2, 1.4], [0.34, 0.67]),
        ("two levels",   [4.5, 2.0],      [0.5]),
        ("one level",    [4.0],           []),
    ]
    ok = True
    for name, levels, breaks in cases:
        a = _FakeContext(levels, breaks)
        lv_a, st_a = scan_steps(a, n_probes=60)
        b = _FakeContext(levels, breaks)
        lv_b, st_b, vs, ps = scan_steps(b, n_probes=60, return_samples=True)

        same = (len(lv_a) == len(lv_b)
                and all(abs(x - y) < 1e-12 for x, y in zip(lv_a, lv_b))
                and len(st_a) == len(st_b)
                and all(abs(x - y) < 1e-12 for x, y in zip(st_a, st_b)))
        shaped = (vs.shape == ps.shape == (60,))
        costed = (a.n_probes == b.n_probes)
        hit = same and shaped and costed
        ok &= hit
        if verbose:
            print(f"   {name:<14} levels/steps identical: {same};  "
                  f"samples 60x2: {shaped};  probes {a.n_probes}=={b.n_probes}: "
                  f"{costed}  {'ok' if hit else 'BROKEN'}")
    if verbose:
        print(f"   -> {'PASS' if ok else 'FAIL'}")
    return ok


def test_scan_best_dominates(verbose: bool = True) -> bool:
    """REVISION 7: scan_best must never return a worse point than scan, and must
    never cost more probes.

    On the synthetic staircase the true peak is at the top of a plateau, so the
    dense scan usually finds a better point than eq. (27)'s landing. The test
    asserts only the two invariants -- domination and equal cost -- because
    those are what the claim depends on.
    """
    cases = [
        ("three levels", [5.0, 3.2, 1.4], [0.34, 0.67]),
        ("two levels",   [4.5, 2.0],      [0.5]),
    ]
    ok = True
    for name, levels, breaks in cases:
        a = _FakeContext(levels, breaks)
        v_scan = ahmed_salam_scan(a)
        b = _FakeContext(levels, breaks)
        v_best = ahmed_salam_scan_best(b)

        meter = _FakeContext(levels, breaks)
        p_scan = meter.probe(v_scan)
        p_best = meter.probe(v_best)

        dominates = p_best >= p_scan - 1e-9
        costed = (b.n_probes == a.n_probes)
        hit = dominates and costed
        ok &= hit
        if verbose:
            print(f"   {name:<14} power {p_scan:8.3f} -> {p_best:8.3f}  "
                  f"probes {a.n_probes} vs {b.n_probes}  "
                  f"{'ok' if hit else 'BROKEN'}")
    if verbose:
        print(f"   -> {'PASS' if ok else 'FAIL'}")
    return ok


def test_group_size_estimator(verbose: bool = True) -> bool:
    """The group-size estimator on synthetic step voltages. OUR extension."""
    if int(config.N_SUBSTRINGS) != 3:
        if verbose:
            print(f"   skipped: N_SUBSTRINGS is {config.N_SUBSTRINGS}, not 3")
        return True
    v_elem = 12.0
    cases = [
        ([], 1, [3]),
        ([1 * ALPHA_MID * v_elem], 2, [1, 2]),
        ([2 * ALPHA_MID * v_elem], 2, [2, 1]),
        ([1 * ALPHA_MID * v_elem, 2 * ALPHA_MID * v_elem], 3, [1, 1, 1]),
    ]
    ok = True
    for steps, n_groups, want in cases:
        got = infer_group_sizes(steps, n_groups, v_elem)
        hit = (got == want) and (sum(got) == int(config.N_SUBSTRINGS))
        ok &= hit
        if verbose:
            print(f"   steps={[round(s,1) for s in steps]} n={n_groups} -> "
                  f"{got}  {'ok' if hit else f'MISMATCH (want {want})'}")
    if verbose:
        print(f"   -> {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    a = test_ahmed_salam_paper_example()
    print("\nragged irradiance reduction:")
    b = test_ragged_irradiances()
    print("\ndetector invariance:")
    c = test_detector_invariance()
    print("\nreturn_samples leaves the detector alone (rev 7):")
    e = test_return_samples()
    print("\nscan_best dominates scan at equal cost (rev 7):")
    f = test_scan_best_dominates()
    print("\ngroup-size estimator (our extension):")
    d = test_group_size_estimator()
    raise SystemExit(0 if (a and b and c and d and e and f) else 1)
