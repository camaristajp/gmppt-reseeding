"""
datasheet.py  --  the naive datasheet baseline (limitation 8 in the Phase 2 log).
REVISION 2 -- beta_oc sourced from pvlib; direction verified, not asserted.

PLACE THIS FILE AT:   C:\\Users\\user\\gmppt\\gmppt\\datasheet.py
                      OVERWRITE revision 1.

WHAT CHANGED IN REVISION 2

  1. beta_oc SOURCE. The S1 pool keeps only seven columns
     (V_mp_ref, V_oc_ref, I_mp_ref, I_sc_ref, N_s, Technology, vmp_voc) and does
     not carry a temperature coefficient. Revision 1 silently fell back to the
     STC coefficient, so k was flat across 25/45/60 C. beta_oc is now read from
     pvlib's full CEC table -- the same source S1 built the pool from -- and
     cached.

  2. DIRECTION TEST CORRECTED. Revision 1 asserted that k RISES with temperature.
     That was wrong arithmetic. Applying the same absolute shift d to both
     voltages gives (V_mp - d)/(V_oc - d), and since V_mp < V_oc that ratio
     FALLS. The test would have failed on correct data and sent us hunting a sign
     error that did not exist.

     Rather than assert a direction a second time, revision 2 VERIFIES it against
     the simulator: device.module_iv under uniform illumination gives the true
     V_mp/V_oc at each temperature, and the datasheet correction must move the
     same way. This uses the simulator to check the rule's direction, not to
     build the rule -- the same separation L4 used for the breakdown parameters.

WHAT THIS IS AND WHY IT MATTERS
    What an engineer reaches for BEFORE 0.80: the module's own V_mp/V_oc from its
    datasheet, optionally corrected for temperature. Both inputs are already
    available to a real optimizer -- nameplate data and a back-of-module sensor.

    It is a different KIND of baseline from the two constants. Fixed 0.80 is one
    number for every module; recalibrated 0.82 is one number tuned on this study's
    data. The datasheet coefficient VARIES BY MODULE, and its temperature-corrected
    form varies by operating point too. It is already conditional, using only
    information a deployed optimizer has.

TEMPERATURE CORRECTION -- AN APPROXIMATION, DECLARED
    The CEC table supplies beta_oc (the V_oc temperature coefficient) but no V_mp
    temperature coefficient. The correction therefore applies beta_oc to both:
        V_oc(T) = V_oc,ref + beta_oc * (T - 25)
        V_mp(T) = V_mp,ref + beta_oc * (T - 25)
    A datasheet quoting a separate V_mp coefficient would be preferable; the CEC
    entry does not carry one. Documented approximation, not a derived result.

    Expected magnitude is small: roughly 0.02 in k across 25-60 C, against a
    shading-induced spread of 0.21 s.d. under sub-substring geometry (Phase 1).
    Temperature is a tenth-order effect next to shading, and the run should be
    read with that in mind.

EXPECTATIONS -- RECORDED BEFORE THE RUN
    1. On UNIFORM scenarios the datasheet coefficient should clearly beat fixed
       0.80 (0.84% mean). The CEC coefficient is derived at STC, exactly the
       unshaded case.
    2. On SUB-SUBSTRING scenarios it should help little -- under 0.5 pt against
       fixed 0.80. Phase 1 found the true coefficient collapses to mean 0.64 with
       s.d. 0.21 there; a datasheet value describes the UNSHADED module and
       carries no information about shading.

    If both hold: the coefficient's MODULE-TO-MODULE variation is not the
    problem, its SHADING-TO-SHADING variation is. That sharpens C1 and forecloses
    "just use the datasheet" permanently.

VERIFICATION ANCHOR
    config.py pins CANONICAL_DEMO_MODULE at coefficient 0.8107. If the pool
    lookup does not reproduce that to three decimals it is reading the wrong
    columns. (Revision 1 passed this check; the lookup is sound.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .harness import Context, register


# ---------------------------------------------------------------------------
# Data access: the trimmed S1 pool, plus beta_oc from the full CEC table
# ---------------------------------------------------------------------------

_POOL = None
_BETA: pd.Series | None = None

_BETA_NAMES = ("beta_oc", "BetaOc", "beta_voc", "Bvoco", "B_voc")


def pool() -> pd.DataFrame:
    global _POOL
    if _POOL is None:
        _POOL = pd.read_parquet(config.CEC_POOL)
    return _POOL


def beta_table() -> pd.Series:
    """V_oc temperature coefficient (V/degC) per module, from pvlib's CEC table.

    The S1 pool does not carry it. pvlib is the same source S1 used, so this
    introduces no new dependency and no new provenance question.
    """
    global _BETA
    if _BETA is not None:
        return _BETA

    import pvlib
    cec = pvlib.pvsystem.retrieve_sam("CECMod")     # columns are module names
    cec = cec.T                                      # -> rows are module names

    col = next((c for c in _BETA_NAMES if c in cec.columns), None)
    if col is None:
        lower = {c.lower(): c for c in cec.columns}
        col = next((lower[n.lower()] for n in _BETA_NAMES
                    if n.lower() in lower), None)
    if col is None:
        raise KeyError(
            "no V_oc temperature coefficient in the pvlib CEC table.\n"
            f"   available columns: {list(cec.columns)}\n"
            "   Send me this list and I will map the correct name.")

    _BETA = pd.to_numeric(cec[col], errors="coerce")
    return _BETA


def beta_oc(module_name: str) -> float:
    """beta_oc for one module, or NaN when unavailable."""
    try:
        return float(beta_table().get(module_name, np.nan))
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# The coefficient
# ---------------------------------------------------------------------------
def v_oc_at(module_name: str, temp_c: float) -> float:
    """Temperature-corrected open-circuit voltage from nameplate data alone.

    V_oc(T) = V_oc_ref + beta_oc * (T - 25), the correction declared in this
    module's header. Uses only information a deployed optimizer has: the
    datasheet and a back-of-module temperature.
    """
    v_oc_ref = float(pool().loc[module_name, "V_oc_ref"])
    return v_oc_ref + beta_oc(module_name) * (float(temp_c) - 25.0)

def coefficient_stc(module_name: str) -> float:
    """k = V_mp,ref / V_oc,ref for one module, straight from the catalogue."""
    row = pool().loc[module_name]
    k = float(row["vmp_voc"])
    if np.isfinite(k) and 0.3 < k < 1.0:
        return k
    v_mp, v_oc = float(row["V_mp_ref"]), float(row["V_oc_ref"])
    return v_mp / v_oc if v_oc > 0 else float("nan")


def coefficient_at(module_name: str, temp_c: float) -> float:
    """Temperature-corrected k(T). See the approximation note in the docstring.

    Falls back to the STC coefficient when beta_oc is unavailable, so a missing
    value degrades to the simpler variant rather than producing a wrong number.
    """
    beta = beta_oc(module_name)
    if not np.isfinite(beta):
        return coefficient_stc(module_name)

    row = pool().loc[module_name]
    v_mp, v_oc = float(row["V_mp_ref"]), float(row["V_oc_ref"])
    if not (np.isfinite(v_mp) and np.isfinite(v_oc)):
        return coefficient_stc(module_name)

    d_t = float(temp_c) - config.STC_TEMPERATURE
    v_oc_t, v_mp_t = v_oc + beta * d_t, v_mp + beta * d_t
    if v_oc_t <= 0 or v_mp_t <= 0:
        return coefficient_stc(module_name)
    return v_mp_t / v_oc_t


# ---------------------------------------------------------------------------
# The registered methods
# ---------------------------------------------------------------------------

def _land_at(ctx: Context, k: float) -> float:
    """Same three-candidate structure as the fixed and recalibrated baselines,
    so the only difference between those rows is the value of k."""
    n_sub = int(config.N_SUBSTRINGS)
    cands = [n * k * ctx.v_oc / n_sub for n in range(1, n_sub + 1)]
    return max(cands, key=ctx.probe)


@register("datasheet")
def datasheet(ctx: Context) -> float:
    """Per-module V_mp/V_oc at STC. Nameplate data only."""
    k = coefficient_stc(ctx.module)
    return _land_at(ctx, k if np.isfinite(k) else config.ALPHA_FIXED)


@register("datasheet_temp")
def datasheet_temp(ctx: Context) -> float:
    """Per-module V_mp/V_oc, temperature-corrected. Nameplate plus a sensor."""
    temp = ctx.conditions.get("temp_c")
    k = (coefficient_stc(ctx.module) if temp is None
         else coefficient_at(ctx.module, float(temp)))
    return _land_at(ctx, k if np.isfinite(k) else config.ALPHA_FIXED)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def test_canonical_coefficient(verbose: bool = True) -> bool:
    """config.py pins the canonical module at 0.8107. If the pool lookup does not
    reproduce that, every datasheet figure below is meaningless."""
    target, tol = 0.8107, 0.001
    k = coefficient_stc(config.CANONICAL_DEMO_MODULE)
    ok = abs(k - target) <= tol
    if verbose:
        print(f"   {config.CANONICAL_DEMO_MODULE}")
        print(f"   coefficient {k:.4f}   pinned {target}   tolerance +/-{tol}")
        print(f"   -> {'PASS' if ok else 'FAIL'}")
    return ok


def test_beta_available(verbose: bool = True) -> bool:
    """beta_oc must actually be found, and must be negative (V_oc falls with
    temperature). A positive or missing value silently disables the correction."""
    m = config.CANONICAL_DEMO_MODULE
    b = beta_oc(m)
    ok = np.isfinite(b) and b < 0
    if verbose:
        print(f"   beta_oc({m}) = {b:.5f} V/degC")
        if np.isfinite(b):
            row = pool().loc[m]
            pct = 100 * b / float(row["V_oc_ref"])
            print(f"   = {pct:.4f} %/degC of V_oc "
                  f"(L3 measured the simulator slope at -0.2852 %/degC)")
        print(f"   -> {'PASS' if ok else 'FAIL'} (must be finite and negative)")
    return ok


def test_direction_against_simulator(verbose: bool = True) -> bool:
    """Does the datasheet correction move k the way the physics does?

    Revision 1 ASSERTED that k rises with temperature. That was wrong arithmetic.
    Rather than assert again, this compares the correction against the simulator's
    true V_mp/V_oc under UNIFORM illumination at the same temperatures -- the
    simulator checks the rule's direction, it does not supply the rule.

    Criterion: the datasheet correction and the simulated truth must move in the
    same direction across 25 -> 60 C. Magnitudes will differ, because the CEC
    entry carries no separate V_mp coefficient.
    """
    from . import device
    from .device import ModuleParams

    m = config.CANONICAL_DEMO_MODULE
    temps = (25.0, 45.0, 60.0)
    mp = ModuleParams.from_cec(m)

    ds, sim = [], []
    for t in temps:
        ds.append(coefficient_at(m, t))
        c = device.module_iv(mp, [config.STC_IRRADIANCE] * config.N_SUBSTRINGS, t)
        a = device.analyse(c)
        V = np.asarray(c["V"], float)
        v_oc = float(V.max())
        v_mp = float(a["gmpp"]["V"])
        sim.append(v_mp / v_oc if v_oc > 0 else float("nan"))

    d_ds = ds[-1] - ds[0]
    d_sim = sim[-1] - sim[0]
    ok = (d_ds * d_sim) > 0 or (abs(d_ds) < 1e-4 and abs(d_sim) < 1e-4)

    if verbose:
        print(f"   {'T':>5} {'datasheet k':>13} {'simulated k':>13}")
        for t, a_, b_ in zip(temps, ds, sim):
            print(f"   {t:>5.0f} {a_:>13.4f} {b_:>13.4f}")
        print(f"   change 25->60 C:  datasheet {d_ds:+.4f}   "
              f"simulated {d_sim:+.4f}")
        print(f"   same direction: {ok}")
        print(f"   -> {'PASS' if ok else 'FAIL'}")
        print("   (magnitudes differ by design: the CEC entry has no separate")
        print("    V_mp temperature coefficient, so beta_oc is applied to both)")
    return ok


def describe_population(verbose: bool = True) -> dict:
    """Spread of the datasheet coefficient across the c-Si pool -- context for how
    much per-module tailoring could possibly buy."""
    df = pool()
    csi = df[df["Technology"].isin(config.CSI_TECHNOLOGIES)]
    k = csi["vmp_voc"].astype(float)
    k = k[np.isfinite(k)]
    out = {"n": int(len(k)), "mean": float(k.mean()), "sd": float(k.std()),
           "min": float(k.min()), "max": float(k.max())}
    if verbose:
        print(f"   c-Si pool n={out['n']}   mean {out['mean']:.4f}   "
              f"s.d. {out['sd']:.4f}   range {out['min']:.4f}-{out['max']:.4f}")
        print(f"   (config pins mean {config.CSI_COEFF_MEAN}, "
              f"s.d. {config.CSI_COEFF_SD})")
        print(f"   NOTE: this s.d. of {out['sd']:.3f} is the MODULE-TO-MODULE")
        print(f"   spread. Phase 1 measured the SHADING-TO-SHADING spread at")
        print(f"   0.21 s.d. under sub-substring geometry -- an order larger.")
    return out


if __name__ == "__main__":
    print("datasheet coefficient, canonical module:")
    a = test_canonical_coefficient()
    print("\nbeta_oc availability (from the pvlib CEC table):")
    b = test_beta_available()
    print("\ntemperature direction, against the simulator:")
    c = test_direction_against_simulator()
    print("\nc-Si population spread:")
    describe_population()
    raise SystemExit(0 if (a and b and c) else 1)