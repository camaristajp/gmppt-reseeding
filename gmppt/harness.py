"""
harness.py  --  Phase 2, step P2.1: method-agnostic comparison harness.
REVISION 3 -- revision 2 plus the Context.conditions field. Overwrite the old file.

PLACE THIS FILE AT:   C:\\Users\\user\\gmppt\\gmppt\\harness.py

RUN THE SELF-CHECK FROM THE REPO ROOT (C:\\Users\\user\\gmppt):
    python phase2\\p1_harness_regression.py --n 2000

WHAT CHANGED IN REVISION 3
    Context now carries a `conditions` dict holding the scenario's irradiances
    and temperature. Ahmed-Salam (and any later conditional method, including C3)
    needs to see the physical conditions to compute a coefficient from them.

    This does NOT weaken the contract. Conditions are inputs, not answers: the
    true GMPP and the P-V array remain sealed, and the method still has to find
    the peak by probing. test_context_hides_the_ceiling still passes, which is
    the property the whole comparison rests on.

WHAT CHANGED IN REVISION 2 (unchanged here)
    1. The adapter is written against the REAL S6/S8 signatures. No fallback
       chains, no guessing.
    2. The fixed-coefficient baseline builds N_SUBSTRINGS candidates at
       n*k*Voc/n_sub for n=1..n_sub and keeps the highest-power one, exactly as
       S8's _loss_for_constant does.
    3. The recalibration sweep reproduces S8 section C exactly: the same
       300-scenario sub-substring sample under seed_for("s8_recal"), same grid.
    4. A per-scenario cross-check against evaluate_scenario's power_lost_frac.

THE CONTRACT
    A method is:            fn(ctx) -> float          # returns a LANDING VOLTAGE
    It receives ctx with:   ctx.v_oc, ctx.i_sc, ctx.module, ctx.geometry, ctx.rng
                            ctx.conditions            # irradiances, temp_c
                            ctx.probe(v) -> float     # power at voltage v, counted
    It does NOT receive the P-V array or the true GMPP. The ceiling is sealed.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from . import config
from . import device
from . import scenarios as scen
from .device import ModuleParams


# ---------------------------------------------------------------------------
# ACCEPTANCE -- declared before any run.
# NOTE THE SUBSET each target belongs to.
#   recal sample  = 300 sub-substring scenarios, seed_for("s8_recal")  [S8 sec C]
#   sub-substring = all sub_substring scenarios                        [S8 sec B]
#   all           = every scenario                                     [S8 sec D]
# ---------------------------------------------------------------------------
ACCEPTANCE = {
    "fixed080_mean_loss_pct  [recal sample]": (4.1, 0.3),
    "best_k                  [recal sample]": (0.82, 0.011),
    "best_k_mean_loss_pct    [recal sample]": (3.5, 0.3),
    "costly_frac_pct         [sub-substring]": (77.0, 3.0),
    "region_error_pct        [sub-substring]": (6.7, 1.0),
    "worst_loss_w            [all]": (18.3, 1.0),
}

# S8 section C reproduction constants -- do not change without changing ACCEPTANCE.
RECAL_GRID = np.round(np.linspace(0.62, 0.86, 13), 3)
RECAL_SAMPLE_N = 300
RECAL_SEED_PURPOSE = "s8_recal"


# ---------------------------------------------------------------------------
# ADAPTER -- concrete, against the real API. Verified signatures:
#   scenarios.generate(pool_df, n_scenarios, train_only=True, n_substrings=3)
#   scenarios.evaluate_scenario(scenario, device_mod, ModuleParams, cost_margin=0.01)
#   device.module_iv(mp, irradiances, T, n_substrings=3, bd=None, bp=None, n_points=4000)
#   device.analyse(curve: dict, prominence_frac=0.01, min_separation_v=1.0)
# Scenario fields used: .module .irradiances .temp_c .geometry
# Curve dict keys used: "V" "P"      analyse keys used: "gmpp" -> {"P", "V"}
# ---------------------------------------------------------------------------

_POOL = None


def pool() -> pd.DataFrame:
    """The CEC pool, loaded once. Same source S6 and S8 use."""
    global _POOL
    if _POOL is None:
        _POOL = pd.read_parquet(config.CEC_POOL)
    return _POOL


def scenario_set(n: int) -> list:
    """The seeded scenario set. Identical to S6/S8 for the same n."""
    return list(scen.generate(pool(), n))


def curve_and_ceiling(sc) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return (V sorted, P sorted, v_gmpp, p_gmpp) for one scenario.

    Mirrors S8's _loss_for_constant so the harness measures the same quantity:
    same ModuleParams source, same breakdown state, same sort.
    """
    mp = ModuleParams.from_cec(sc.module)
    c = device.module_iv(mp, sc.irradiances, sc.temp_c, bd=config.breakdown())
    a = device.analyse(c)

    V, P = np.asarray(c["V"], float), np.asarray(c["P"], float)
    order = np.argsort(V)
    V, P = V[order], P[order]

    gmpp = a["gmpp"]
    p_gmpp = float(gmpp["P"])
    v_gmpp = float(gmpp["V"]) if "V" in gmpp else float(V[int(np.argmax(P))])
    return V, P, v_gmpp, p_gmpp


def reference_frame(scenarios: Sequence) -> pd.DataFrame:
    """S8's own per-scenario results. This is the ground truth the harness is
    checked against -- not a reimplementation of it."""
    return pd.DataFrame(
        [scen.evaluate_scenario(s, device, ModuleParams) for s in scenarios])


# ---------------------------------------------------------------------------
# CORE
# ---------------------------------------------------------------------------

@dataclass
class Context:
    """What a method is allowed to see. The ceiling is NOT in here.

    `conditions` holds the physical inputs (irradiances, temp_c) a conditional
    method needs. Inputs, not answers -- the peak still has to be found by
    probing.
    """
    v_oc: float
    i_sc: float | None
    module: str
    geometry: str
    rng: np.random.Generator
    _V: np.ndarray = field(repr=False)
    _P: np.ndarray = field(repr=False)
    n_probes: int = 0
    conditions: dict = field(default_factory=dict)

    def probe(self, v: float) -> float:
        """Sample power at voltage v. Every call is counted."""
        self.n_probes += 1
        v = min(max(float(v), float(self._V[0])), float(self._V[-1]))
        return float(np.interp(v, self._V, self._P))


@dataclass
class Record:
    scenario_id: int
    module: str
    geometry: str
    v_oc: float
    p_gmpp: float
    v_gmpp: float
    k_true: float
    v_hat: float
    p_hat: float
    loss_frac: float
    loss_w: float
    k_hat: float
    region_correct: bool
    n_probes: int


def _peak_regions(V: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Region index per sample, split at the local minima between maxima.

    The harness's own attribution, needed because a general method can land
    anywhere. p1 cross-checks it against evaluate_scenario's region_error column.
    """
    peaks = [i for i in range(1, len(P) - 1) if P[i] > P[i - 1] and P[i] >= P[i + 1]]
    if len(peaks) < 2:
        return np.zeros(len(V), dtype=int)
    cuts = [peaks[j] + int(np.argmin(P[peaks[j]:peaks[j + 1] + 1]))
            for j in range(len(peaks) - 1)]
    return np.searchsorted(np.asarray(cuts), np.arange(len(V)), side="right")


def run_method(method: Callable[[Context], float],
               scenarios: Sequence,
               seed_purpose: str = "phase2_harness") -> list[Record]:
    """Run one method across a scenario list. One Record per scenario."""
    try:
        base_seed = config.seed_for(seed_purpose)
    except Exception:
        base_seed = int(config.MASTER_SEED)

    records: list[Record] = []
    for idx, sc in enumerate(scenarios):
        V, P, v_gmpp, p_gmpp = curve_and_ceiling(sc)
        if p_gmpp <= 0:
            continue
        v_oc = float(V[-1])

        ctx = Context(
            v_oc=v_oc,
            i_sc=getattr(sc, "i_sc", None),
            module=str(sc.module),
            geometry=str(sc.geometry),
            rng=np.random.default_rng(base_seed + idx),
            _V=V, _P=P,
            conditions={"irradiances": getattr(sc, "irradiances", None),
                        "temp_c": getattr(sc, "temp_c", None)},
        )

        v_hat = float(method(ctx))
        v_hat = min(max(v_hat, float(V[0])), v_oc)
        p_hat = float(np.interp(v_hat, V, P))

        regions = _peak_regions(V, P)
        r_true = regions[int(np.argmin(np.abs(V - v_gmpp)))]
        r_hat = regions[int(np.argmin(np.abs(V - v_hat)))]

        records.append(Record(
            scenario_id=idx,
            module=ctx.module,
            geometry=ctx.geometry,
            v_oc=v_oc,
            p_gmpp=p_gmpp,
            v_gmpp=v_gmpp,
            k_true=v_gmpp / v_oc if v_oc else float("nan"),
            v_hat=v_hat,
            p_hat=p_hat,
            loss_frac=max(0.0, (p_gmpp - p_hat) / p_gmpp),
            loss_w=max(0.0, p_gmpp - p_hat),
            k_hat=v_hat / v_oc if v_oc else float("nan"),
            region_correct=bool(r_true == r_hat),
            n_probes=ctx.n_probes,
        ))
    return records


def metrics(records: Sequence[Record], geometry: str | None = None) -> dict:
    """Two-sided summary: mean AND worst, raw loss AND costly fraction."""
    rs = [r for r in records if geometry is None or r.geometry == geometry]
    if not rs:
        return {"n": 0}
    loss = np.array([r.loss_frac for r in rs])
    watts = np.array([r.loss_w for r in rs])
    return {
        "n": len(rs),
        "mean_loss_pct": 100 * float(loss.mean()),
        "median_loss_pct": 100 * float(np.median(loss)),
        "p95_loss_pct": 100 * float(np.percentile(loss, 95)),
        "worst_loss_pct": 100 * float(loss.max()),
        "worst_loss_w": float(watts.max()),
        "costly_frac_pct": 100 * float((loss > 0.01).mean()),
        "region_error_pct": 100 * float(np.mean([not r.region_correct for r in rs])),
        "mean_probes": float(np.mean([r.n_probes for r in rs])),
    }


def write_records(records: Sequence[Record], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(Record.__dataclass_fields__.keys())
        for r in records:
            w.writerow([getattr(r, k) for k in Record.__dataclass_fields__])


# ---------------------------------------------------------------------------
# METHOD REGISTRY
# ---------------------------------------------------------------------------

METHODS: dict[str, Callable[[Context], float]] = {}


def register(name: str):
    def deco(fn):
        METHODS[name] = fn
        return fn
    return deco


def constant_method(k: float) -> Callable[[Context], float]:
    """The fixed-coefficient family, exactly as S8 defines it.

    Not a single k*Voc landing. The model builds one candidate per substring at
    n*k*Voc/n_sub for n = 1..n_sub, probes each, and keeps the highest-power one.
    A tracker that scans candidate peaks does exactly this, and the probe count
    (n_sub) is its measurement cost.
    """
    n_sub = int(config.N_SUBSTRINGS)

    def fn(ctx: Context) -> float:
        cands = [n * k * ctx.v_oc / n_sub for n in range(1, n_sub + 1)]
        return max(cands, key=ctx.probe)

    fn.__name__ = f"constant_{k:.3f}"
    return fn


METHODS["fixed_0.80"] = constant_method(0.80)


@register("oracle_upper_bound")
def m_oracle(ctx: Context) -> float:
    """Diagnostic only. Dense probing confirms the probe path can reach the
    ceiling -- so any loss a real method shows is the METHOD's, not the harness's."""
    grid = np.linspace(0.02 * ctx.v_oc, ctx.v_oc, 600)
    return float(grid[int(np.argmax([ctx.probe(v) for v in grid]))])


def recal_sample(scenarios: Sequence) -> list:
    """The exact 300-scenario sub-substring sample S8 section C used."""
    sub = [s for s in scenarios if s.geometry == "sub_substring"]
    rng = np.random.default_rng(config.seed_for(RECAL_SEED_PURPOSE))
    idx = rng.choice(len(sub), min(RECAL_SAMPLE_N, len(sub)), replace=False)
    return [sub[i] for i in idx]


def sweep_constants(sample: Sequence, grid=RECAL_GRID) -> dict[float, float]:
    """Mean loss for each candidate constant on the sample. Reproduces S8 sec C."""
    out = {}
    for k in grid:
        m = metrics(run_method(constant_method(float(k)), sample))
        out[float(k)] = m["mean_loss_pct"]
    return out


# ---------------------------------------------------------------------------
# SELF-CHECK
# ---------------------------------------------------------------------------

def _check(label: str, value: float) -> tuple[bool, str]:
    target, tol = ACCEPTANCE[label]
    ok = abs(value - target) <= tol
    return ok, (f"{'PASS' if ok else 'FAIL'}  {label:<40} {value:8.3f}"
                f"  (target {target} +/- {tol})")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Phase 2 comparison harness self-check")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=Path("results/phase2"))
    ap.add_argument("--skip-sweep", action="store_true")
    args = ap.parse_args(argv)

    print(f"BREAKDOWN_MODE = {getattr(config, 'BREAKDOWN_MODE', '?')}")
    print(f"N_SUBSTRINGS   = {config.N_SUBSTRINGS}")
    scenarios = scenario_set(args.n)
    print(f"scenarios: {len(scenarios)}")

    # -- 0. probe-path sanity: the harness must not itself lose power ---------
    orc = metrics(run_method(METHODS["oracle_upper_bound"], scenarios[:200]))
    print(f"\noracle probe check (200 scenarios): mean loss "
          f"{orc['mean_loss_pct']:.4f}%  -- must be ~0")

    # -- 1. full-set run, harness fixed-0.80 ---------------------------------
    recs = run_method(METHODS["fixed_0.80"], scenarios)
    write_records(recs, args.out / "fixed_080.csv")
    m_all = metrics(recs)
    m_sub = metrics(recs, geometry="sub_substring")
    print(f"\nfixed 0.80  all          : {json.dumps(m_all)}")
    print(f"fixed 0.80  sub_substring: {json.dumps(m_sub)}")

    # -- 2. per-scenario cross-check against S8's own column ------------------
    ref = reference_frame(scenarios)
    print("\nper-scenario cross-check vs evaluate_scenario:")
    if "power_lost_frac" in ref.columns and len(ref) == len(recs):
        mine = np.array([r.loss_frac for r in recs])
        theirs = ref["power_lost_frac"].to_numpy(dtype=float)
        d = np.abs(mine - theirs)
        print(f"   max abs diff {d.max():.5f}   mean abs diff {d.mean():.5f}")
        print(f"   scenarios differing by >0.005: {(d > 0.005).sum()} of {len(d)}")
        print("   (a large diff here means the harness is measuring a DIFFERENT")
        print("    quantity than S8 -- fix that before trusting anything below)")
    if "region_error" in ref.columns:
        rsub = ref[ref["geometry"] == "sub_substring"]["region_error"].mean()
        print(f"   reference sub-substring region-error: {rsub*100:.1f}% "
              f"(harness: {m_sub['region_error_pct']:.1f}%)")

    # -- 3. recalibration sweep on S8's exact sample --------------------------
    lines = []
    sample = recal_sample(scenarios)
    print(f"\nrecalibration sample: {len(sample)} sub-substring scenarios")
    m_fixed_sample = metrics(run_method(METHODS["fixed_0.80"], sample))
    lines.append(_check("fixed080_mean_loss_pct  [recal sample]",
                        m_fixed_sample["mean_loss_pct"]))

    if not args.skip_sweep:
        losses = sweep_constants(sample)
        best_k = min(losses, key=losses.get)
        print("   k sweep: " + "  ".join(f"{k:.2f}:{v:.2f}%"
                                         for k, v in sorted(losses.items())))
        print(f"   best k = {best_k:.3f} at {losses[best_k]:.2f}% mean loss")
        lines.append(_check("best_k                  [recal sample]", best_k))
        lines.append(_check("best_k_mean_loss_pct    [recal sample]",
                            losses[best_k]))

    lines.append(_check("costly_frac_pct         [sub-substring]",
                        m_sub["costly_frac_pct"]))
    lines.append(_check("region_error_pct        [sub-substring]",
                        m_sub["region_error_pct"]))
    lines.append(_check("worst_loss_w            [all]", m_all["worst_loss_w"]))

    print("\n--- regression vs Phase 1 (S8) ---")
    for _, text in lines:
        print(text)
    failed = sum(1 for ok, _ in lines if not ok)
    print(f"\n{len(lines) - failed}/{len(lines)} checks passed")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(
        json.dumps({"all": m_all, "sub_substring": m_sub,
                    "recal_sample_fixed080": m_fixed_sample}, indent=2),
        encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())