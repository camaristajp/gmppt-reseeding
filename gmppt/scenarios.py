"""
Seeded shading-scenario generator (S6).

Produces reproducible partial-shading scenarios that the pipeline (S7 labelling,
S8 characterisation) runs through the verified simulator. Everything is derived
from config.MASTER_SEED via config.seed_for, so one number reproduces the whole
scenario set on any machine — this is the community-released generator (plan
Section 9.11, gap 6).

Parameter ranges are taken from the plan, not invented:
  * irradiance 100–1000 W/m^2            (Section 4)
  * temperature {25, 45, 60} deg C       (Section 4)
  * three shading geometries             (Section 8.2 / Gate A(i)):
      - uniform          : all substrings equal (baseline)
      - whole_substring  : each substring uniform, substrings differ
      - sub_substring    : one cell group within one substring shaded
  * near-threshold irradiance ratios     (Section 10, Task 1)
  * held-out module split                (Task 3): a fraction of c-Si modules is
    reserved for testing generalisation and NEVER appears in the training set.

A scenario is a dict: module name, temperature, geometry, and the per-substring
irradiance argument in the exact form module_iv expects (scalars or, for
sub_substring, a list for the shaded substring).
"""
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

from . import config


# geometry mix (fraction of scenarios of each kind). Sub-substring is the Gate
# A(i) case; whole-substring is the classic multi-peak; uniform anchors the
# no-shading baseline. This mix is the one genuine free choice (recorded here).
GEOMETRY_MIX = {"uniform": 0.15, "whole_substring": 0.45, "sub_substring": 0.40}

TEMPERATURES = (25.0, 45.0, 60.0)
IRR_MIN, IRR_MAX = 100.0, 1000.0
N_CELL_GROUPS = 3          # cell groups per substring for sub-substring shading
HELDOUT_FRACTION = 0.20    # 20% of c-Si modules reserved for generalisation tests


@dataclass
class Scenario:
    module: str
    temp_c: float
    geometry: str
    irradiances: list        # length n_substrings; entries scalar or list
    near_threshold: bool


def split_modules(pool_df, heldout_fraction=HELDOUT_FRACTION):
    """Deterministic train/held-out split of the c-Si module pool.

    Stratified by coefficient so both sets span the V_mp/V_oc range. Seeded, so
    the same modules are always held out. Returns (train_names, heldout_names).
    """
    csi = pool_df[pool_df["Technology"].isin(config.CSI_TECHNOLOGIES)].copy()
    csi = csi.sort_values("vmp_voc")                    # stratify by coefficient
    rng = np.random.default_rng(config.seed_for("module_split"))
    # take every k-th module into held-out, offset by a seeded start, so the
    # held-out set is spread across the whole coefficient range
    k = int(round(1 / heldout_fraction))
    offset = int(rng.integers(0, k))
    names = list(csi.index)
    heldout = set(names[offset::k])
    train = [n for n in names if n not in heldout]
    return train, sorted(heldout)


def _draw_irradiances(rng, geometry, n_substrings):
    """Per-substring irradiance argument for module_iv, plus a near-threshold flag.

    near-threshold = at least one substring/group ratio within 0.15 of another,
    the regime where peaks merge and the coefficient is hardest (Section 10).
    """
    def lit(): return float(rng.uniform(IRR_MIN, IRR_MAX))

    if geometry == "uniform":
        g = lit()
        return [g] * n_substrings, False

    if geometry == "whole_substring":
        vals = [lit() for _ in range(n_substrings)]
        # occasionally force a near-threshold pair
        near = bool(rng.random() < 0.35)
        if near:
            vals[1] = vals[0] * float(rng.uniform(0.85, 0.98))
        return vals, near

    # sub_substring: substrings uniform except one, which has one shaded cell group
    vals = [lit() for _ in range(n_substrings)]
    s = int(rng.integers(0, n_substrings))            # which substring
    base = vals[s]
    groups = [base] * N_CELL_GROUPS
    gidx = int(rng.integers(0, N_CELL_GROUPS))        # which cell group
    # shade it: sometimes deep, sometimes near-threshold
    near = bool(rng.random() < 0.45)
    factor = float(rng.uniform(0.85, 0.98)) if near else float(rng.uniform(0.1, 0.7))
    groups[gidx] = base * factor
    vals[s] = groups
    return vals, near


def generate(pool_df, n_scenarios, train_only=True,
             n_substrings=config.N_SUBSTRINGS):
    """Generate `n_scenarios` seeded scenarios over the training modules.

    Reproducible: same pool + same n_scenarios + same MASTER_SEED -> identical
    scenarios. Returns a list[Scenario].
    """
    train, heldout = split_modules(pool_df)
    modules = train if train_only else (train + heldout)
    rng = np.random.default_rng(config.seed_for("scenarios"))

    geoms = list(GEOMETRY_MIX)
    probs = np.array([GEOMETRY_MIX[g] for g in geoms])
    probs = probs / probs.sum()

    out = []
    for _ in range(n_scenarios):
        module = str(modules[int(rng.integers(0, len(modules)))])
        temp = float(TEMPERATURES[int(rng.integers(0, len(TEMPERATURES)))])
        geometry = geoms[int(rng.choice(len(geoms), p=probs))]
        irr, near = _draw_irradiances(rng, geometry, n_substrings)
        out.append(Scenario(module, temp, geometry, irr, near))
    return out


def to_frame(scenarios):
    """Flatten scenarios to a DataFrame for logging/inspection (irradiances as str)."""
    rows = []
    for sc in scenarios:
        d = asdict(sc)
        d["irradiances"] = str(d["irradiances"])
        rows.append(d)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Region-error and coefficient helpers (S6 convergence, S8 characterisation)
# --------------------------------------------------------------------------
def evaluate_scenario(scenario, device_mod, ModuleParams, cost_margin=0.01):
    """Run one scenario and return its coefficient, true GMPP, and both a raw and
    a COSTLY region-error flag for the fixed-0.8 model.

    The fixed-0.8 model places candidates at n*0.8*Voc/n_sub (n=1..n_sub) and
    lands at the candidate whose actual power is highest. Two error notions:
      * region_error (raw): the peak nearest that landing is not the true GMPP.
      * region_error_costly: the model's landing power is below the true GMPP by
        more than `cost_margin` (default 1%, matching the fallback margin in the
        plan's protocol). This is the error that actually loses watts and is what
        Gate A(i) cares about — a "wrong region" that costs nothing is a near-tie,
        not a failure.
    The power lost (fraction of true GMPP) is returned so S8 can report the
    distribution, not just the rate.
    """
    import numpy as np
    mp = ModuleParams.from_cec(scenario.module)
    c = device_mod.module_iv(mp, scenario.irradiances, scenario.temp_c,
                             bd=config.breakdown())
    a = device_mod.analyse(c)
    V, P = c["V"], c["P"]
    voc = float(V.max())
    gmpp_V = a["gmpp"]["V"]
    gmpp_P = a["gmpp"]["P"]

    # implied coefficient at the true global peak
    coeff = gmpp_V / voc if voc > 0 else float("nan")

    # fixed-0.8 model: candidate voltages, land at the highest-power one.
    # interp needs V ascending; the module curve has V descending (I ascending),
    # so sort first. Clip candidates to the curve's V range and floor power at 0
    # (a candidate landing in reverse bias delivers no useful power, not negative).
    order = np.argsort(V)
    Vs, Ps = V[order], P[order]
    n_sub = config.N_SUBSTRINGS
    cands = [n * 0.8 * voc / n_sub for n in range(1, n_sub + 1)]
    cands = [min(max(cv, float(Vs.min())), float(Vs.max())) for cv in cands]
    cand_P = [max(0.0, float(np.interp(cv, Vs, Ps))) for cv in cands]
    best = int(np.argmax(cand_P))
    chosen_V = cands[best]
    landed_P = cand_P[best]

    # raw region error: nearest peak to the landing is not the true GMPP
    peak_Vs = np.array([pv for pv, _, _ in a["peaks"]])
    nearest_peak_V = peak_Vs[int(np.argmin(np.abs(peak_Vs - chosen_V)))]
    region_error = abs(nearest_peak_V - gmpp_V) > 1e-6

    # costly error: the landing loses more than cost_margin of true GMPP power
    power_lost_frac = (gmpp_P - landed_P) / gmpp_P if gmpp_P > 0 else 0.0
    region_error_costly = power_lost_frac > cost_margin

    return dict(coeff=coeff, gmpp_V=gmpp_V, gmpp_P=gmpp_P, voc=voc,
                landed_P=landed_P, power_lost_frac=float(power_lost_frac),
                region_error=bool(region_error),
                region_error_costly=bool(region_error_costly),
                n_peaks=a["n_peaks"], geometry=scenario.geometry,
                near_threshold=scenario.near_threshold)


def convergence_curve(results, checkpoints=(200, 400, 800, 1200, 1600, 2000)):
    """Report how the coefficient mean and the sub-substring region-error rates
    (raw and costly) evolve as scenario count grows. Stable tails mean the count
    is sufficient (plan Section 9.11). Checkpoints beyond len(results) are
    dropped rather than duplicated."""
    import numpy as np
    coeffs = np.array([r["coeff"] for r in results])
    sub_mask = np.array([r["geometry"] == "sub_substring" for r in results])
    sub_err = np.array([r["region_error"] for r in results])
    sub_err_costly = np.array([r["region_error_costly"] for r in results])
    # unique, capped checkpoints (no duplicate final rows)
    pts = sorted(set(min(n, len(results)) for n in checkpoints))
    rows = []
    for n in pts:
        cm = float(np.nanmean(coeffs[:n]))
        cs = float(np.nanstd(coeffs[:n]))
        sm = sub_mask[:n]
        raw = float(np.mean(sub_err[:n][sm])) if sm.any() else float("nan")
        cost = float(np.mean(sub_err_costly[:n][sm])) if sm.any() else float("nan")
        rows.append(dict(n=n, coeff_mean=cm, coeff_sd=cs, n_sub=int(sm.sum()),
                         sub_region_error_raw=raw, sub_region_error_costly=cost))
    return rows