"""
Central configuration for the GMPPT Phase 1 codebase.

Two kinds of constant live here:

1. FIXED PROTOCOL CONSTANTS (Section 9.10). These are "set here and not changed
   once Phase 1 begins." Changing them mid-study invalidates cross-phase
   comparison, so they live in exactly one place and are imported everywhere.

2. Reproducibility settings: the master seed and derived per-purpose seeds, and
   the project paths. Everything random in this project derives its seed from
   MASTER_SEED so a single number reproduces the entire study.
"""
from __future__ import annotations
import hashlib
import subprocess
from pathlib import Path

# --------------------------------------------------------------------------
# Paths (resolved relative to the project root, independent of CWD)
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

CEC_POOL = RESULTS_DIR / "cec_pool.parquet"
VERIFICATION_LOG = RESULTS_DIR / "phase1_verification_log.md"

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
MASTER_SEED = 20260901  # study start month; arbitrary but recorded

def seed_for(purpose: str) -> int:
    """Deterministic per-purpose seed derived from the master seed.

    Using a named sub-seed (e.g. 'scenario_gen', 'module_split') keeps streams
    independent and reproducible without threading one global RNG through the
    whole codebase. Same MASTER_SEED + same purpose string -> same seed, always.
    """
    h = hashlib.sha256(f"{MASTER_SEED}:{purpose}".encode()).hexdigest()
    return int(h[:8], 16)

# --------------------------------------------------------------------------
# FIXED PROTOCOL CONSTANTS (Section 9.10) -- do not change after Phase 1 begins
# --------------------------------------------------------------------------
CONVERGENCE_BAND_FRAC = 0.01   # +/-1% of global-peak power (convergence band)
RESIDENCE_PERIODS = 20         # consecutive switching periods inside the band
FALLBACK_MARGIN_FRAC = 0.01    # fallback margin: 1% of current operating power

# --------------------------------------------------------------------------
# Standard test conditions and the study's operating grid (Section 4)
# --------------------------------------------------------------------------
STC_IRRADIANCE = 1000.0        # W/m^2
STC_TEMPERATURE = 25.0         # deg C
TEMPERATURES_C = (25.0, 45.0, 60.0)          # Section 4 temperature set
IRRADIANCE_RANGE = (100.0, 1000.0)           # W/m^2, per-substring range

# Fractional-Voc coefficients
ALPHA_FIXED = 0.80             # the conventional constant (all baselines)
ALPHA_RECAL_PRELIM = 0.87      # Section 4 recalibrated constant (indicative)

# CEC single-diode bandgap defaults (pvlib calcparams_cec)
EG_REF = 1.121                 # eV
DEG_DT = -0.0002677            # eV/K

# Module architecture in scope (Section 8.1)
N_SUBSTRINGS = 3               # conventional full-cell c-Si, three bypass diodes

# Reverse-bias breakdown setting for scenario evaluation (S6/S8) and validation.
# Physically, a cell driven into reverse bias under sub-substring shading DOES
# break down (avalanche); modelling it as never breaking down is less realistic.
# So the primary characterisation uses breakdown ON. Breakdown OFF is retained as
# a reproducible documented bound (it gives a higher, more conservative-looking
# region-error rate). L4 showed the rate is insensitive to the exact ON parameters.
#   "on"  -> Breakdown(factor=1e-2, voltage=-15.0, exp=3.28)  [realistic, primary]
#   "off" -> Breakdown(factor=0.0)                            [bound, was S8 default]
BREAKDOWN_MODE = "on"          # "on" (primary) or "off" (documented bound)
BREAKDOWN_ON_PARAMS = dict(factor=1e-2, voltage=-15.0, exp=3.28)


def breakdown(mode=None):
    """Return a device.Breakdown for the given mode ("on"/"off"), defaulting to
    BREAKDOWN_MODE. Imported lazily to avoid a config<->device import cycle."""
    from gmppt.device import Breakdown
    mode = mode or BREAKDOWN_MODE
    if mode == "on":
        return Breakdown(**BREAKDOWN_ON_PARAMS)
    return Breakdown()  # off: factor=0.0
CSI_TECHNOLOGIES = ("Mono-c-Si", "Multi-c-Si")

# --------------------------------------------------------------------------
# CEC population reference (c-Si only; the study's in-scope population)
# --------------------------------------------------------------------------
# These are frozen from S1 on the pinned pvlib/CEC toolchain (see PVLIB_PIN,
# CEC_ROWS_EXPECTED below). The full-catalogue figures (all technologies) are
# 21,535 modules, 0.633-0.874, mean 0.810, s.d. 0.017; the in-scope c-Si subset
# is the one every downstream step draws from. Recompute and re-freeze these two
# constants only if the toolchain pins change.
CSI_POOL_N = 20946             # c-Si modules in the S1 pool (Mono + Multi)
CSI_COEFF_MEAN = 0.8107        # c-Si population mean V_mp/V_oc at STC (S1)
CSI_COEFF_SD = 0.0152          # c-Si population s.d.
CSI_COEFF_RANGE = (0.6977, 0.8719)   # c-Si population min/max

# Canonical demonstration module (S3 figures, S3 tests).
#
# Selected ONCE and pinned here so the S3 figure and every S3 number are
# identical on every machine. Selection criterion, in order:
#   (1) c-Si, cell count divisible by 3 (a clean three-substring part);
#   (2) STC power within 280-320 W (echoes the Section 4 300 W module);
#   (3) among those, the module whose STC coefficient V_mp/V_oc is closest to
#       the c-Si population mean CSI_COEFF_MEAN -- i.e. a *median* design, not a
#       tail, so the demo is representative rather than anecdotal.
# The name is pinned as a string (not recomputed at run time) so the S3 tests do
# not depend on the S1 pool artefact existing on disk, and so database growth
# cannot silently move the canonical module underneath the recorded figures.
# `phase1/s3_reverse_bias_bypass.py: pick_demo_module()` reproduces the criterion
# and asserts it still selects this module; if the assertion ever fails, the pin
# is re-chosen deliberately and the S3 artefacts regenerated.
CANONICAL_DEMO_MODULE = "Canadian_Solar_Inc__CS6U_310P"
# ^ coefficient 0.8107, equal to the c-Si population mean to four decimals (the
#   closest-to-mean candidate in the 280-320 W, N_s%3==0 band). 33 candidates
#   tie at this minimum coefficient distance, all at coeff 0.8107; the
#   alphabetical index tie-break settles them deterministically here. A power-
#   first selection would instead have picked a 300 W part at coeff 0.813, off
#   the mean, purely by naming accident -- hence selecting on the coefficient.

# Verification tolerance
STC_SIG_FIGS = 3               # datasheet reproduction requirement (Section 9.2)

# --------------------------------------------------------------------------
# Toolchain provenance (stamped onto every S-step output)
# --------------------------------------------------------------------------
PVLIB_PIN = "0.15.2"           # the CEC database ships with pvlib; pin it
CEC_ROWS_EXPECTED = 21535      # full CEC catalogue row count on the pinned pvlib


def git_hash() -> str:
    """Short git commit of the working tree, or 'nogit' if unavailable."""
    try:
        out = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "nogit"
    except Exception:
        return "nogit"


def provenance() -> str:
    """One-line provenance stamp for output headers and the verification log.

    Records the toolchain that produced a result so stale artefacts are visible
    at a glance rather than silently diverging from the code.
    """
    import pvlib
    tag = "OK" if pvlib.__version__ == PVLIB_PIN else \
        f"MISMATCH(pinned {PVLIB_PIN})"
    return (f"pvlib {pvlib.__version__} [{tag}] | "
            f"CEC rows expected {CEC_ROWS_EXPECTED} | "
            f"seed {MASTER_SEED} | git {git_hash()}")