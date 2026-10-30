"""
transitions.py  --  P2.9: shading-PATTERN-transition sequences (dynamic peak
                    relocation), driven by real sun geometry.

PLACE THIS FILE AT:   C:\\Users\\user\\gmppt\\phase2\\transitions.py
                      NEW FILE. gmppt/ is not modified.

SELF-TEST:
    python phase2\\transitions.py

WHAT THIS IS, AND WHY IT EXISTS
    EN 50530 (gmppt/dynamic.py) cannot move the global peak between substrings:
    it holds ONE shading pattern and scales every substring by the same factor,
    so the peak voltage shifts only ~2% of V_oc. This module produces the case
    it cannot -- the shading PATTERN changing over time so the darkest substring
    shifts and the GMPP RELOCATES from one substring region to another. That
    relocation is the event the tracker comparison (p12) measures.

    The sequences it emits are the dataset for BOTH the journal's dynamic-
    relocation result AND the dashboard playback. Generated once, to journal
    standard, so the dashboard gets it for free.

TIME-OF-DAY IS PHYSICAL, NOT COSMETIC
    The shadow sweeps because the SUN MOVES. Sun azimuth/elevation over the day
    come from pvlib.solarposition for a DECLARED site and date; the shadow band's
    position on the module is driven by that azimuth. So a relocation "at 13:50"
    is a consequence of solar geometry, not an arbitrary label -- and the sweep
    rate is non-uniform through the day exactly as the real sun's is.

DECLARED SYNTHETIC PARAMETERS -- labelled, cited, and UPGRADEABLE
    Nanum's array geometry is not yet in hand, so the site and shading object are
    DECLARED SYNTHETIC. They are not invented conveniences:
      * SITE = Jeju, KR (33.45 N, 126.57 E) -- the collaboration's actual 
        location. Real place; only the array geometry on it is synthetic.
      * REFERENCE_DATE = an equinox -- the representative mid-range sun path
        (a solstice would be an edge case, not a default).
      * MODULE tilt/azimuth = fixed-tilt, tilt ~= latitude, due south -- the
        textbook default for the site, pending Nanum's real mounting.
      * SHADOW BAND width / depth and its azimuth window = a representative
        structural/pole shadow, declared with physical ranges.
    Every one of these is a module constant below; replace with Nanum's measured
    values when they arrive and the sequences upgrade with no code change. Any
    figure derived here TRAVELS WITH the label "declared synthetic site (Jeju
    reference), pending partner geometry."

VALIDATION IS THE SPINE (the standing rule: validate before any figure is read)
    A relocation dataset is the easiest to accidentally rig, so every generated
    sequence must pass the gates below before it may enter the tracker
    comparison. A sequence that fails a gate is DISCARDED and counted, never
    reported. Gates, not diagnostics.

      G1  RELOCATION ACTUALLY OCCURS.  The TRUE GMPP substring (from the
          simulator's analyse(), not from the shading proxy) must change at least
          once and cross a boundary. Prevents a "relocation test" that never
          relocates -- silently becoming the EN 50530 non-case. If G1 fails, the
          shading is too shallow to make a bypass diode conduct; deepen
          SHADE_DEPTH. This is the gate that makes the goal real.
      G2  CURVE INTEGRITY.  Every step's curve must reproduce its own GMPP under
          interpolation to 1e-3 (mirrors dynamic.py's p_check). Prevents
          orientation/label bugs (D9-class) contaminating a curve. Enforced
          inside the builder -- it RAISES rather than returning a bad curve.
      G3  PHYSICAL CONTINUITY.  Per-substring irradiance may change between
          consecutive steps by at most MAX_STEP_IRR_FRAC. Prevents a shadow
          "teleporting" -- an unphysical jump that would make relocation
          artificially easy or hard. Ties the data to the sun-driven rate.
      G4  PEAK-COUNT TIMELINE (diagnostic, not a gate).  n_peaks at each step is
          recorded and reported, so shape changes during a transition are visible
          rather than hidden inside an efficiency number. This is the "peaks on
          the curve" quantity.
      G5  UNIFORM CONTROL (run-level, in p12).  A no-relocation control sequence
          must score ~100% for every tracker, or the instrument is unsound.
      G6  REPRODUCIBILITY & SPLIT (run-level).  All randomness from
          config.seed_for; sequences drawn on VALIDATION modules only; held-out
          never touched here.

    This file owns G1-G4 and the data they guard. G5/G6 are the p12 runner's.

PROVENANCE
    Sun positions from pvlib.solarposition (astronomical, no network, no fit).
    Curves from the same device.module_iv / analyse used everywhere else, so a
    relocation curve is scored by the identical instrument as every static and
    EN 50530 figure.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gmppt import config, device  # noqa: E402
from gmppt.device import ModuleParams  # noqa: E402
from gmppt.dynamic import CONTROL_PERIOD_S, DynamicTrajectory  # noqa: E402

N_SUB = int(config.N_SUBSTRINGS)

# ---------------------------------------------------------------------------
# DECLARED SYNTHETIC SITE & SHADING GEOMETRY  (upgrade when Nanum data arrives)
# ---------------------------------------------------------------------------

# Site: Jeju, KR -- the collaboration's real location. Array geometry synthetic.
SITE_LAT = 33.45
SITE_LON = 126.57
SITE_TZ = "Asia/Seoul"

# Equinox reference day: representative mid-range sun path (declared).
REFERENCE_DATE = "2026-03-20"

# Fixed-tilt mounting, tilt ~= latitude, due south -- textbook default, declared.
MODULE_TILT_DEG = 33.0
MODULE_AZIMUTH_DEG = 180.0

# Representative structural/pole shadow, declared with physical ranges:
#   BAND_WIDTH  -- shadow band width in units of module width (a pole/gap casts a
#                  finite band; ~0.5 module-widths is a mid-range structural edge)
#   SHADE_DEPTH -- fraction by which a fully-covered substring's irradiance drops
#                  (0.80 -> shaded substring at 20%, deep enough to conduct the
#                  bypass diode; G1 fails loudly if it is set too shallow)
#   AZ_ON/AZ_OFF-- azimuth window over which the shadow lies on the module; the
#                  band centre sweeps 0->1 across the module as azimuth crosses it
BAND_WIDTH = 0.5
SHADE_DEPTH = 0.80
AZ_ON_DEG = 95.0
AZ_OFF_DEG = 265.0

# Base (unshaded) plane-of-array irradiance for the reference day, declared.
BASE_IRRADIANCE = 900.0

# Transition families -- the sampling cadence sets how fast the pattern moves.
#   "pole"  : structural shadow, edge crosses over MINUTES (coarse cadence)
#   "cloud" : soft front, crosses over SECONDS (fine cadence, stresses re-seed)
# Both are the SAME sun-driven sweep; only the wall-clock cadence differs, which
# is the declared, physically-ranged rate (cloud ~ seconds, pole ~ minutes).
TRANSITION_FAMILIES = {
    "pole":  {"sample_seconds": 30.0},
    "cloud": {"sample_seconds": 2.0},
}

# Continuity gate (G3): the largest per-substring fractional irradiance change
# allowed between consecutive samples. Set above the sun-driven step so honest
# sweeps pass and only teleports fail; verified in the self-test.
MAX_STEP_IRR_FRAC = 0.20

# t0 settle steps before the scored window (the tracker starts settled), not
# scored -- same rationale as dynamic.py.
T0_STEPS = 20


# ---------------------------------------------------------------------------
# SUN GEOMETRY  (pure; depends only on pvlib + numpy)
# ---------------------------------------------------------------------------

def solar_over_day(date: str = REFERENCE_DATE, lat: float = SITE_LAT,
                   lon: float = SITE_LON, tz: str = SITE_TZ,
                   sample_seconds: float = 30.0,
                   min_elevation_deg: float = 5.0):
    """Apparent sun elevation/azimuth over the daylit part of one day.

    Astronomical (pvlib.solarposition) -- no network, no fit. Returns
    (times, elevation_deg, azimuth_deg) restricted to sun above min_elevation.
    """
    import pvlib
    freq = f"{int(round(sample_seconds))}s"
    times = pd.date_range(f"{date} 05:30", f"{date} 18:30", freq=freq, tz=tz)
    sp = pvlib.solarposition.get_solarposition(times, lat, lon)
    elev = sp["apparent_elevation"].to_numpy(float)
    az = sp["azimuth"].to_numpy(float)
    up = elev > float(min_elevation_deg)
    return times[up], elev[up], az[up]


def band_centre_from_azimuth(az: np.ndarray,
                             az_on: float = AZ_ON_DEG,
                             az_off: float = AZ_OFF_DEG) -> np.ndarray:
    """Shadow-band centre position across the module width, in [0, 1].

    Driven by real azimuth: morning (east) -> 0, evening (west) -> 1. The sweep
    inherits the sun's non-uniform angular rate, so the relocation timing is
    physical rather than linear-in-clock.
    """
    return np.clip((np.asarray(az, float) - az_on) / (az_off - az_on), 0.0, 1.0)


def substring_shade_fractions(xc: float, n_sub: int = N_SUB,
                              band_width: float = BAND_WIDTH) -> np.ndarray:
    """Fraction of each substring covered by the shadow band centred at xc."""
    edges = np.linspace(0.0, 1.0, n_sub + 1)
    seg = 1.0 / n_sub
    lo, hi = xc - band_width / 2.0, xc + band_width / 2.0
    out = np.empty(n_sub, float)
    for s in range(n_sub):
        overlap = max(0.0, min(hi, edges[s + 1]) - max(lo, edges[s]))
        out[s] = overlap / seg
    return out


def irradiances_for_step(base_g: float, shade_frac: np.ndarray,
                         shade_depth: float = SHADE_DEPTH) -> list:
    """Per-substring irradiance argument for module_iv (whole-substring form).

    A substring's irradiance = base * (1 - shade_depth * covered_fraction). This
    is the whole_substring geometry (each substring a scalar), and the transition
    moves WHICH substring is darkest -- a module-level, between-substring
    relocation, which is the case the thesis is about.
    """
    return [float(base_g * (1.0 - shade_depth * f)) for f in shade_frac]


# ---------------------------------------------------------------------------
# PATTERN SEQUENCE
# ---------------------------------------------------------------------------

@dataclass
class TransitionSequence:
    """A time-ordered set of shading patterns for one module, sun-driven."""
    module: str
    temp_c: float
    family: str
    times: object                      # tz-aware DatetimeIndex
    irradiances_list: list             # one per step, module_iv form
    xc: np.ndarray                     # band centre per step (diagnostic)
    sample_seconds: float

    @property
    def n_steps(self) -> int:
        return len(self.irradiances_list)


def pattern_sequence(module: str, temp_c: float, family: str = "pole",
                     base_g: float = BASE_IRRADIANCE,
                     date: str = REFERENCE_DATE) -> TransitionSequence:
    """Build one sun-driven shading-pattern-transition sequence.

    The shadow band sweeps across the module as the sun moves; each time sample
    is one shading pattern. NOT scaled from a single pattern (that is EN 50530);
    each step is a genuinely distinct pattern, which is what lets the peak
    relocate.
    """
    if family not in TRANSITION_FAMILIES:
        raise ValueError(f"unknown transition family {family!r}; "
                         f"choose from {sorted(TRANSITION_FAMILIES)}")
    ss = float(TRANSITION_FAMILIES[family]["sample_seconds"])
    times, _elev, az = solar_over_day(date=date, sample_seconds=ss)
    xc = band_centre_from_azimuth(az)
    irr_list = [irradiances_for_step(base_g, substring_shade_fractions(c))
                for c in xc]
    return TransitionSequence(module=str(module), temp_c=float(temp_c),
                              family=str(family), times=times,
                              irradiances_list=irr_list, xc=xc,
                              sample_seconds=ss)


# ---------------------------------------------------------------------------
# TRAJECTORY BUILDER  (mirrors dynamic_trajectory_for; enforces G2)
# ---------------------------------------------------------------------------

def gmpp_region(v_gmpp: float, v_oc: float, n_sub: int = N_SUB) -> int:
    """Which substring interval the GMPP voltage sits in (0 .. n_sub-1)."""
    if v_oc <= 0:
        return 0
    k = v_gmpp / v_oc
    return int(min(max(int(k * n_sub), 0), n_sub - 1))


def sequence_trajectory_for(seq: TransitionSequence,
                            control_period_s: float = CONTROL_PERIOD_S,
                            geometry: str = "relocation"):
    """Build a DynamicTrajectory from a pattern sequence: one curve per pattern.

    Each pattern is simulated by the SAME device.module_iv / analyse used
    everywhere else, so a relocation curve is scored by the identical instrument
    as every other figure. G2 (curve integrity) is enforced here: a curve that
    does not reproduce its own GMPP RAISES rather than being returned.

    Returns (trajectory, per_step) where per_step carries the GMPP region and
    peak count at each step, for the gates.
    """
    mp = ModuleParams.from_cec(seq.module)
    curves, v_gmpps, p_gmpps, regions, n_peaks = [], [], [], [], []

    for irr in seq.irradiances_list:
        c = device.module_iv(mp, irr, seq.temp_c,
                             n_substrings=N_SUB, bd=config.breakdown())
        a = device.analyse(c)
        V = np.asarray(c["V"], float)
        P = np.asarray(c["P"], float)
        order = np.argsort(V)                      # raw is DESCENDING (D9)
        V, P = V[order], P[order]

        v_gmpp = float(a["gmpp"]["V"])
        p_gmpp = float(a["gmpp"]["P"])
        p_check = float(np.interp(v_gmpp, V, P))   # G2
        if abs(p_check - p_gmpp) > 1e-3 * max(abs(p_gmpp), 1e-9):
            raise RuntimeError(
                f"G2 curve integrity FAILED for {seq.module}: interpolated "
                f"{p_check:.4f} W at v_gmpp, analyse reports {p_gmpp:.4f} W")

        v_oc = float(V.max())
        curves.append((V, P, v_gmpp, p_gmpp))
        v_gmpps.append(v_gmpp)
        p_gmpps.append(p_gmpp)
        regions.append(gmpp_region(v_gmpp, v_oc))
        n_peaks.append(int(a.get("n_peaks", 0)))

    # control steps per block from the real sample interval; t0 settle unscored
    per_block = max(1, int(round(seq.sample_seconds / control_period_s)))
    block_steps = np.array([T0_STEPS] + [per_block] * len(curves), int)
    scored = np.array([False] + [True] * len(curves), bool)
    # a settling copy of the first curve so the tracker starts settled (unscored)
    curves = [curves[0]] + curves

    traj = DynamicTrajectory(curves=curves, block_steps=block_steps,
                             scored=scored, profile_name=f"relocation/{seq.family}",
                             module=seq.module, geometry=geometry)
    per_step = {"regions": np.asarray(regions, int),
                "n_peaks": np.asarray(n_peaks, int),
                "v_gmpp": np.asarray(v_gmpps, float),
                "p_gmpp": np.asarray(p_gmpps, float),
                "xc": seq.xc}
    return traj, per_step


# ---------------------------------------------------------------------------
# GATES  (G1, G3, G4)  -- G2 lives in the builder, G5/G6 in the p12 runner
# ---------------------------------------------------------------------------

def gate_relocation(per_step: dict, verbose: bool = True) -> tuple[bool, dict]:
    """G1: the TRUE GMPP substring must change and cross a boundary."""
    regions = per_step["regions"]
    changes = int(np.sum(regions[1:] != regions[:-1]))
    visited = sorted(set(int(r) for r in regions))
    events = [i + 1 for i in range(len(regions) - 1)
              if regions[i + 1] != regions[i]]
    ok = changes >= 1 and len(visited) >= 2
    info = {"changes": changes, "regions_visited": visited,
            "event_indices": events}
    if verbose:
        print(f"   G1 relocation: GMPP substring changed {changes} time(s), "
              f"visited {visited}")
        if not ok:
            print("      -> FAIL: the GMPP never crosses a substring boundary. "
                  "Shading too shallow to")
            print("         conduct a bypass diode -- raise SHADE_DEPTH (this is "
                  "a real discard, not a bug).")
    return ok, info


def gate_continuity(seq: TransitionSequence,
                    max_step_frac: float = MAX_STEP_IRR_FRAC,
                    verbose: bool = True) -> tuple[bool, dict]:
    """G3: no per-substring irradiance may teleport between consecutive steps."""
    arr = np.asarray(seq.irradiances_list, float)          # steps x n_sub
    denom = np.maximum(arr[:-1], 1e-9)
    step_frac = np.abs(arr[1:] - arr[:-1]) / denom
    worst = float(step_frac.max()) if step_frac.size else 0.0
    ok = worst <= max_step_frac
    if verbose:
        print(f"   G3 continuity: worst single-step irradiance change "
              f"{100*worst:.1f}% (bar {100*max_step_frac:.0f}%) -> "
              f"{'PASS' if ok else 'FAIL (teleport -- sample finer)'}")
    return ok, {"worst_step_frac": worst}


def peak_count_timeline(per_step: dict, verbose: bool = True) -> dict:
    """G4 (diagnostic): report the peak-count timeline, never scored."""
    npk = per_step["n_peaks"]
    uniq = sorted(set(int(x) for x in npk))
    if verbose:
        print(f"   G4 peak-count timeline: values {uniq}, "
              f"changes {int(np.sum(npk[1:] != npk[:-1]))} time(s) "
              f"(reported, not scored)")
    return {"n_peaks_values": uniq}


def validate_sequence(seq: TransitionSequence, verbose: bool = True):
    """Build the trajectory (G2 inside) and run G1, G3, G4. Returns (ok, report).

    A sequence is USABLE only if G1 and G3 pass. G2 raises during the build if a
    curve is malformed. G4 is reported alongside.
    """
    traj, per_step = sequence_trajectory_for(seq)        # raises on G2
    g1, i1 = gate_relocation(per_step, verbose)
    g3, i3 = gate_continuity(seq, verbose=verbose)
    g4 = peak_count_timeline(per_step, verbose)
    ok = g1 and g3
    report = {"module": seq.module, "family": seq.family,
              "n_steps": seq.n_steps, "usable": ok, **i1, **i3, **g4}
    return ok, traj, per_step, report


# ---------------------------------------------------------------------------
# SELF-TEST
# ---------------------------------------------------------------------------

def main() -> int:
    from gmppt.harness import scenario_set  # noqa: E402

    print("P2.9  shading-pattern-transition generator (sun-driven relocation)")
    try:
        print("  " + config.provenance())
    except Exception:
        pass
    print("=" * 78)
    print(f"DECLARED SYNTHETIC site: Jeju, KR ({SITE_LAT} N, {SITE_LON} E), "
          f"{REFERENCE_DATE}")
    print(f"shadow band width {BAND_WIDTH} module-widths, depth {SHADE_DEPTH}, "
          f"azimuth window {AZ_ON_DEG}-{AZ_OFF_DEG} deg")
    print("Site is the collaboration's real location; array geometry is declared")
    print("synthetic, upgradeable to Nanum's measured values. Any figure travels")
    print("with that label.\n")
    print("VALIDATION GATES run before any sequence is usable: G1 relocation, "
          "G2 curve")
    print("integrity (raises), G3 continuity, G4 peak-count (diagnostic). A "
          "sequence that")
    print("fails a gate is DISCARDED and counted, never reported.\n")

    # a few VALIDATION-module scenarios, used only to borrow real module names
    pool = [s for s in scenario_set(200) if s.geometry != "uniform"]
    if not pool:
        print("no shaded scenarios drawn -- check scenario_set.")
        return 1
    modules = []
    for s in pool:
        if s.module not in modules:
            modules.append(s.module)
        if len(modules) >= 4:
            break

    usable, discarded = 0, 0
    for fam in ("pole", "cloud"):
        for mod in modules:
            temp = 25.0
            print(f"\n-- module {mod[:34]:<34} family {fam}")
            seq = pattern_sequence(mod, temp, family=fam)
            try:
                ok, traj, per_step, rep = validate_sequence(seq)
            except RuntimeError as e:
                print(f"   G2 RAISED -> sequence discarded: {e}")
                discarded += 1
                continue
            print(f"   steps {seq.n_steps}, trajectory steps {traj.n_steps}, "
                  f"events at {rep['event_indices'][:6]}"
                  f"{' ...' if len(rep['event_indices'])>6 else ''}")
            if ok:
                usable += 1
            else:
                discarded += 1

    print("\n" + "=" * 78)
    print(f"{usable} usable, {discarded} discarded "
          f"(discards are the gates working, not failures).")
    if usable == 0:
        print("No usable sequence. If G1 failed everywhere, raise SHADE_DEPTH so")
        print("the shaded substring conducts its bypass diode; if G3 failed,")
        print("sample finer. Do NOT run p12 until at least the pole family yields")
        print("usable relocation sequences.")
        return 1
    print("Usable relocation sequences exist. Next: p12 tracker comparison over")
    print("gated sequences (validation modules), with the re-convergence metric")
    print("and the measurement-triggered re-seed. Reportable = validation only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())