"""
dataset.py  --  training dataset construction, splitting, and power-loss scoring.

PLACE THIS FILE AT:   C:\\Users\\user\\gmppt\\gmppt\\dataset.py
                      NEW FILE in the package, beside features.py.

SELF-TEST (fast):
    python -m gmppt.dataset

WHAT THIS IS
    The "training dataset" box of the pipeline. It turns scenarios into a table
    the model can learn from, splits it so the reported number is honest, and
    scores a predicted coefficient in the units the thesis reports.

THE SPLIT -- BY MODULE, NOT BY ROW
    The same module appears in hundreds of scenarios. A random row split would
    put one module in both train and test, letting the model memorise that
    module's coefficient rather than learn to read the curve. It would score well
    and generalise badly.

    Splitting by MODULE is the analogue of splitting by time in a forecasting
    problem: cut along the axis where leakage would occur.

        test        20% of c-Si modules -- scenarios.split_modules, unchanged.
                    Touched ONCE, at the end. Every look costs a little of its
                    independence.
        validation  a further slice of the training modules. Used freely: model
                    size, stopping point, probe schedule.
        train       the rest.

    split_modules() is reused verbatim so the held-out set is identical to the
    one Phase 1 declared. The validation slice is carved out with its own seed.

CURVE STORAGE -- why the extra file
    Each curve is stored downsampled to CURVE_POINTS on a normalised voltage
    grid, so power loss can be evaluated for any predicted coefficient WITHOUT
    re-simulating. Roughly 15 MB per 4,000 scenarios. Two reasons this matters:

      1. Validation reports power loss (the thesis metric) as cheaply as it
         reports coefficient error. The two are not interchangeable -- a k error
         of 0.02 costs little on a flat peak and a lot on a steep one -- so both
         must be visible from the first run.
      2. A power-based training objective needs the curve. Storing it now keeps
         that option open without a rebuild, which matters because the objective
         is still a Month-3 decision.

HOW A PREDICTED COEFFICIENT IS SCORED
    C3 predicts where the peak actually is, so it lands ONCE: v = k_pred * V_oc.
    The constants need three candidates because a fixed 0.80 cannot know which
    peak is global; C3 does not have that problem. Serving cost is therefore
    len(PROBE_FRACTIONS) + 1 = 6 probes, against Ahmed-Salam's ~33 (P2.4c).

    power_loss_for_k() reads the landed power off the stored curve and returns
    (p_gmpp - p_landed) / p_gmpp -- the same definition the harness uses, so
    validation numbers are on the same scale as the baseline table. The FINAL
    number still comes through the harness itself; this is the fast proxy.

WHAT THIS MODULE DOES NOT DO
    No model, no training. It prepares and scores. The learning curve that sizes
    the dataset lives in phase3/p1_dataset_sizing.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from . import config
from . import device
from . import scenarios as scen
from .device import ModuleParams
from .features import (FEATURE_NAMES, build_row, curve_sampler,  # noqa: F401
                       extract_features, extract_targets, pool)

DATA_DIR = config.RESULTS_DIR / "phase3"
CURVE_POINTS = 256                 # downsampled grid per scenario
VAL_FRACTION_OF_TRAIN = 0.20       # so overall roughly 64 / 16 / 20


# ---------------------------------------------------------------------------
# SPLIT
# ---------------------------------------------------------------------------

@dataclass
class Split:
    train: list[str]
    val: list[str]
    test: list[str]

    def summary(self) -> str:
        n = len(self.train) + len(self.val) + len(self.test)
        return (f"train {len(self.train)} ({100*len(self.train)/n:.0f}%), "
                f"val {len(self.val)} ({100*len(self.val)/n:.0f}%), "
                f"test {len(self.test)} ({100*len(self.test)/n:.0f}%) modules")


def module_split(val_fraction: float = VAL_FRACTION_OF_TRAIN) -> Split:
    """Three-way split by module.

    The test set is scenarios.split_modules' held-out list, unchanged, so it is
    the same 20% Phase 1 declared. Validation is carved from the training
    modules under its own seed.
    """
    train_all, heldout = scen.split_modules(pool())
    rng = np.random.default_rng(config.seed_for("c3_val_split"))
    idx = rng.permutation(len(train_all))
    n_val = int(round(val_fraction * len(train_all)))
    val = [str(train_all[i]) for i in idx[:n_val]]
    train = [str(train_all[i]) for i in idx[n_val:]]
    return Split(train=train, val=val, test=[str(m) for m in heldout])


def assign_split(df: pd.DataFrame, split: Split) -> pd.DataFrame:
    """Add a `split` column. Any module in none of the lists is dropped."""
    lookup = {}
    for name, tag in ((split.train, "train"), (split.val, "val"),
                      (split.test, "test")):
        for m in name:
            lookup[m] = tag
    out = df.copy()
    out["split"] = out["module"].map(lookup)
    return out[out["split"].notna()].reset_index(drop=True)


# ---------------------------------------------------------------------------
# BUILD
# ---------------------------------------------------------------------------

def _normalised_curve(V: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Power on a fixed grid of V/V_oc in [0, 1]. One row per scenario."""
    v_oc = float(V[-1])
    grid = np.linspace(0.0, 1.0, CURVE_POINTS) * v_oc
    return np.interp(grid, V, P).astype(np.float32)


def build(scenarios: Sequence, verbose: bool = True
          ) -> tuple[pd.DataFrame, np.ndarray]:
    """Features, targets, and downsampled curves for a scenario list.

    The simulation is the expensive part and happens once per scenario here.
    Everything downstream reads the saved arrays.
    """
    rows, curves = [], []
    for i, sc in enumerate(scenarios):
        mp = ModuleParams.from_cec(sc.module)
        c = device.module_iv(mp, sc.irradiances, sc.temp_c,
                             bd=config.breakdown())
        a = device.analyse(c)

        V = np.asarray(c["V"], float)
        P = np.asarray(c["P"], float)
        order = np.argsort(V)
        V, P = V[order], P[order]
        v_oc = float(V[-1])
        if v_oc <= 0 or float(a["gmpp"]["P"]) <= 0:
            continue

        x = extract_features(curve_sampler(V, P), v_oc, sc.module, sc.temp_c)
        y = extract_targets(V, P, a)

        row = {name: float(val) for name, val in zip(FEATURE_NAMES, x)}
        row.update(k_true=y.k_true, region=y.region, offset=y.offset,
                   v_gmpp=y.v_gmpp, p_gmpp=y.p_gmpp, n_peaks=y.n_peaks,
                   module=str(sc.module), geometry=str(sc.geometry),
                   near_threshold=bool(getattr(sc, "near_threshold", False)))
        rows.append(row)
        curves.append(_normalised_curve(V, P))

        if verbose and (i + 1) % 500 == 0:
            print(f"   ...{i + 1}/{len(scenarios)}")

    return pd.DataFrame(rows), np.vstack(curves)


def save(df: pd.DataFrame, curves: np.ndarray, tag: str = "c3") -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATA_DIR / f"{tag}_features.parquet", index=False)
    np.save(DATA_DIR / f"{tag}_curves.npy", curves)
    print(f"   dataset -> {DATA_DIR / (tag + '_features.parquet')}  "
          f"({len(df)} rows)")
    print(f"   curves   -> {DATA_DIR / (tag + '_curves.npy')}  "
          f"({curves.nbytes / 1e6:.1f} MB)")
    return DATA_DIR


def load(tag: str = "c3") -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_parquet(DATA_DIR / f"{tag}_features.parquet")
    curves = np.load(DATA_DIR / f"{tag}_curves.npy")
    return df, curves


# ---------------------------------------------------------------------------
# SCORING -- the thesis metric, computed fast
# ---------------------------------------------------------------------------

def power_at_k(curves: np.ndarray, rows: np.ndarray | Sequence[int],
               k: np.ndarray) -> np.ndarray:
    """Power delivered when landing at k * V_oc, read off the stored curves.

    The curve grid is already normalised to V/V_oc, so k indexes it directly --
    no per-module rescaling, which is the point of storing it that way.
    """
    rows = np.asarray(rows, dtype=int)
    k = np.clip(np.asarray(k, dtype=float), 0.0, 1.0)
    grid = np.linspace(0.0, 1.0, CURVE_POINTS)
    return np.array([float(np.interp(kk, grid, curves[r]))
                     for r, kk in zip(rows, k)])


def power_loss_for_k(curves: np.ndarray, rows: np.ndarray | Sequence[int],
                     k_pred: np.ndarray, p_gmpp: np.ndarray) -> np.ndarray:
    """Fractional power loss vs the true GMPP -- the harness definition.

    Fast proxy for validation. The number that goes in the thesis still comes
    from run_method() through the harness, on the same scenarios the baselines
    used.
    """
    landed = power_at_k(curves, rows, k_pred)
    p_gmpp = np.asarray(p_gmpp, dtype=float)
    return np.maximum(0.0, (p_gmpp - landed) / np.where(p_gmpp > 0, p_gmpp, 1.0))


def report_metrics(k_true: np.ndarray, k_pred: np.ndarray,
                   loss: np.ndarray, label: str = "") -> dict:
    """Both metrics side by side. They are not interchangeable."""
    k_true = np.asarray(k_true, float)
    k_pred = np.asarray(k_pred, float)
    loss = np.asarray(loss, float)
    m = {
        "n": int(len(loss)),
        "k_mae": float(np.mean(np.abs(k_pred - k_true))),
        "mean_loss_pct": 100 * float(loss.mean()),
        "median_loss_pct": 100 * float(np.median(loss)),
        "p95_loss_pct": 100 * float(np.percentile(loss, 95)),
        "worst_loss_pct": 100 * float(loss.max()),
        "costly_frac_pct": 100 * float((loss > 0.01).mean()),
    }
    if label:
        print(f"   {label:<22} k MAE {m['k_mae']:.4f}   "
              f"mean loss {m['mean_loss_pct']:.2f}%   "
              f"costly {m['costly_frac_pct']:.1f}%   "
              f"worst {m['worst_loss_pct']:.1f}%")
    return m


# ---------------------------------------------------------------------------
# VERIFICATION
# ---------------------------------------------------------------------------

def test_split_disjoint(verbose: bool = True) -> bool:
    """No module may appear in more than one set. This is the leakage guard."""
    s = module_split()
    a, b, c = set(s.train), set(s.val), set(s.test)
    ok = not (a & b) and not (a & c) and not (b & c)
    if verbose:
        print(f"   {s.summary()}")
        print(f"   train∩val {len(a & b)}, train∩test {len(a & c)}, "
              f"val∩test {len(b & c)}")
        print(f"   -> {'PASS' if ok else 'FAIL'}")
    return ok


def test_split_matches_phase1(verbose: bool = True) -> bool:
    """The test set must be exactly the held-out list Phase 1 declared."""
    s = module_split()
    _, heldout = scen.split_modules(pool())
    ok = set(s.test) == {str(m) for m in heldout}
    if verbose:
        print(f"   test set is scenarios.split_modules' held-out list: {ok}")
        print(f"   -> {'PASS' if ok else 'FAIL'}")
    return ok


def test_oracle_scores_zero(verbose: bool = True) -> bool:
    """Feeding the TRUE coefficient back must score near-zero loss.

    If it does not, the stored curve or the scoring path disagrees with the
    simulator, and every validation number would be wrong by that amount.
    """
    from .harness import scenario_set
    df, curves = build(scenario_set(60), verbose=False)
    rows = np.arange(len(df))
    loss = power_loss_for_k(curves, rows, df["k_true"].to_numpy(),
                            df["p_gmpp"].to_numpy())
    worst = 100 * float(loss.max())
    mean = 100 * float(loss.mean())
    ok = worst < 0.5                      # sub-half-percent from downsampling
    if verbose:
        print(f"   scoring the true coefficient: mean {mean:.3f}%, "
              f"worst {worst:.3f}%")
        print(f"   (residual is the {CURVE_POINTS}-point curve grid, not error)")
        print(f"   -> {'PASS' if ok else 'FAIL'} (worst must be < 0.5%)")
    return ok


def test_geometry_balance(verbose: bool = True) -> bool:
    """Each split should carry all three geometries -- a set missing one would
    make its numbers incomparable with the baseline table."""
    from .harness import scenario_set
    df, _ = build(scenario_set(400), verbose=False)
    df = assign_split(df, module_split())
    ok = True
    if verbose:
        print(f"   {'split':<8} {'n':>6}  geometry mix")
    for tag in ("train", "val", "test"):
        sub = df[df["split"] == tag]
        if len(sub) == 0:
            ok = False
            continue
        mix = sub["geometry"].value_counts(normalize=True)
        if len(mix) < 3:
            ok = False
        if verbose:
            parts = "  ".join(f"{g} {100*v:.0f}%" for g, v in mix.items())
            print(f"   {tag:<8} {len(sub):>6}  {parts}")
    if verbose:
        print(f"   -> {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    print("C3 dataset: split, build, and scoring")
    try:
        print("  " + config.provenance())
    except Exception:
        pass
    print("=" * 70)

    print("\n1. SPLIT IS DISJOINT  (the leakage guard)")
    a = test_split_disjoint()

    print("\n2. TEST SET MATCHES PHASE 1")
    b = test_split_matches_phase1()

    print("\n3. ORACLE SCORES ZERO  (stored curve agrees with the simulator)")
    c = test_oracle_scores_zero()

    print("\n4. GEOMETRY BALANCE ACROSS SPLITS")
    d = test_geometry_balance()

    print("\n" + "=" * 70)
    passed = sum([a, b, c, d])
    print(f"{passed}/4 checks passed")
    raise SystemExit(0 if passed == 4 else 1)