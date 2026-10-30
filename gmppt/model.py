"""
model.py  --  the C3 two-stage conditional coefficient, and its harness adapter.
REVISION 2 -- adds the keep-best landing rule. Overwrite the old file.

PLACE THIS FILE AT:   gmppt/model.py
                      beside features.py and dataset.py.

WHAT CHANGED IN REVISION 2
    as_method() gained a `keep_best` option. With it set, the adapter returns
    the best of the points it has ALREADY measured instead of unconditionally
    landing at k * V_oc. The probe count does not change.

    WHY. The baseline audit (paper/experiments/audit_baselines.py) found that
    ahmed_salam_scan probed 60 points, computed a staircase from them, then
    discarded the 60 measured powers and returned a formula prediction that
    was further from the true peak than points it had already visited -- in
    75% of scenarios, at a median 2.31x its own resolution bound. Keeping the
    best already-measured point cut its mean loss on whole_substring from
    1.47% to 0.07% at identical cost.

    This adapter committed the same error. It probes at PROBE_FRACTIONS to
    build the feature vector, then throws those five powers away and lands
    only at k * V_oc. E2 on the leakage-free design split (n=2000) showed
    what that costs: the seed beats a 60-probe exhaustive scan three to one
    in the median (0.0111% vs 0.0333%, Wilcoxon p 9.7e-06, winning 60.3% of
    scenarios) but carries a tail the scan structurally cannot have --
    worst 23.109% against 0.863%. The scan's worst case is bounded by its
    probe resolution; this adapter's was bounded by nothing.

    Having applied that standard to a competing method, applying it here is
    not optional. A reviewer who reads the audit will ask.

    GUARANTEE. The returned point is the argmax over a set that contains the
    landing, so its measured power is greater than or equal to the landing's.
    The rule cannot make any scenario worse. It is the same argument, and the
    same proof, as for ahmed_salam_scan_best.

    HONESTY OF THE MEASUREMENT. The rule uses only powers the method actually
    paid a probe for. It does so through a recording wrapper around ctx.probe
    rather than by reading ctx._V / ctx._P, which would be reaching past the
    sealed curve for values no deployed tracker could have. Probes are counted
    by the harness either way, so test_probe_cost still asserts exactly six.

    `keep_best` defaults to FALSE so that every number measured before this
    revision reproduces unchanged. Report both rows, exactly as the ladder
    reports ahmed_salam_scan beside ahmed_salam_scan_best.

WHY TWO STAGES
    P2.5 found that peak separation is quantised at about V_oc / N_SUBSTRINGS,
    because peaks sit at substring boundaries. The error structure is therefore
    DISCRETE, not continuous: the question is which of three intervals holds the
    peak, and then where inside it.

    P3.2 tested that hint against the alternatives on identical validation rows:

        plain regression on k        2.74% mean loss,  region-0 16.12%
        region-0 upweighted x6       2.34%             region-0  8.99%
        power-weighted               2.48%             region-0 15.03%
        two-stage region + offset    0.26%             region-0  0.51%

    Reweighting moved region 0 by 1-7 pt; the architecture moved it by 15.6. The
    answer was structural, not a matter of objective or class balance. That also
    reframes the supervised-versus-power-objective question: it matters far less
    than the output parameterisation.

    The mechanism is visible in the features. The p20 probe separates region 0
    from the rest at 4.69 standard deviations -- if the peak sits below V_oc/3,
    the probe at 0.20 V_oc lands near it, so its normalised power is close to 1.
    The signal was already in the vector; plain regression was not shaped to use
    it. The region classifier reaches 99.5% accuracy with 100% region-0 recall,
    against a 55.8% trivial floor.

WHAT THE MODEL DOES
    stage 1   classify which substring interval holds the GMPP  (0 .. n_sub-1)
    stage 2   regress the position within that interval          (0 .. 1)
    combine   k = (region + offset) / n_sub,  land at k * V_oc
    land      with keep_best, return the best measured point instead if the
              landing is worse than one already paid for

    One offset regressor per region, each trained only on rows from that region,
    so each solves a problem a third as wide as direct regression on k.

MEASUREMENT COST -- the other half of the claim
    Serving spends len(PROBE_FRACTIONS) probes to build the feature vector, plus
    one to land. Six, against Ahmed-Salam's ~33 (P2.4c showed 30 performs the
    same as 240; the default budget measures 62.5). The harness COUNTS these, so
    the figure in the results table is measured rather than asserted -- which
    matters, because the cost argument is half the contribution and an adapter
    that quietly probed more would not show up in the proxy metric.

WHY THE ADAPTER EXISTS AT ALL
    Until now the model was scored by dataset.power_loss_for_k, a fast proxy over
    stored curves. The baselines were scored by the harness. Two implementations
    of "power lost against the true peak" produce two numbers that cannot be
    placed in one table without argument -- exactly the error that made a
    reference row read 22.18% when the real baseline was 3.32%.

    as_method() turns trained weights into fn(ctx) -> voltage, so C3 goes through
    run_method like every other method and the final table has one definition of
    the metric.

SELF-TEST
    python -m gmppt.model
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from . import config
from .features import (FEATURE_NAMES, PROBE_FRACTIONS, extract_features,
                       module_scalars)  # noqa: F401
from .harness import Context

MODEL_DIR = config.RESULTS_DIR / "phase3"
N_SUB = int(config.N_SUBSTRINGS)

# Serving cost: features + one landing probe. Asserted against the harness count.
# keep_best does NOT change this -- it reuses measurements already taken.
EXPECTED_PROBES = len(PROBE_FRACTIONS) + 1


@dataclass
class TwoStageModel:
    """Region classifier plus one offset regressor per region."""
    classifier: object
    offsets: dict[int, object]
    feature_names: tuple[str, ...] = FEATURE_NAMES
    n_sub: int = N_SUB

    # -- prediction ------------------------------------------------------
    def predict_k(self, x: np.ndarray) -> np.ndarray:
        """Coefficient for a batch of feature vectors."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        regions = np.asarray(self.classifier.predict(x), dtype=int)
        k = np.empty(len(x), dtype=float)
        for r in range(self.n_sub):
            sel = regions == r
            if not sel.any():
                continue
            reg = self.offsets.get(int(r))
            off = (np.clip(reg.predict(x[sel]), 0.0, 1.0) if reg is not None
                   else np.full(int(sel.sum()), 0.5))
            k[sel] = (r + off) / self.n_sub
        return np.clip(k, 0.0, 1.0)

    # -- the harness contract --------------------------------------------
    def as_method(self, keep_best: bool = False) -> Callable[[Context], float]:
        """Wrap as fn(ctx) -> landing voltage, for run_method.

        Builds the feature vector through ctx.probe -- the SAME extractor used in
        training, so there is no second definition of what the model sees -- then
        lands once at k * V_oc. Total probes: len(PROBE_FRACTIONS) + 1.

        keep_best
            False (default)  land at k * V_oc unconditionally. This is the
                             behaviour every number measured before revision 2
                             was produced with, and it is kept so those numbers
                             reproduce.
            True             return the best of the points already measured if
                             the landing is worse than one of them. Same probe
                             count; cannot be worse in any scenario, because the
                             landing is itself a member of the candidate set.

        The candidate set is built by a recording wrapper around ctx.probe, so it
        holds only powers this method actually paid for. It is not read from the
        sealed curve.
        """
        def fn(ctx: Context) -> float:
            measured: list[tuple[float, float]] = []

            def probe(v: float) -> float:
                """ctx.probe, remembering what it cost us."""
                p = float(ctx.probe(v))
                measured.append((float(v), p))
                return p

            temp = ctx.conditions.get("temp_c")
            x = extract_features(probe, ctx.v_oc, ctx.module,
                                 float(temp) if temp is not None
                                 else config.STC_TEMPERATURE)
            k = float(self.predict_k(x.reshape(1, -1))[0])
            v = k * ctx.v_oc
            p = probe(v)          # the landing measurement, counted

            if keep_best and measured:
                v_best, p_best = max(measured, key=lambda vp: vp[1])
                if p_best > p:    # strict: a tie keeps the model's own landing
                    return v_best
            return v

        fn.__name__ = "c3_two_stage_best" if keep_best else "c3_two_stage"
        return fn

    # -- persistence -----------------------------------------------------
    def save(self, tag: str = "c3_two_stage") -> Path:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        path = MODEL_DIR / f"{tag}.pkl"
        with path.open("wb") as fh:
            pickle.dump(self, fh)
        return path

    @staticmethod
    def load(tag: str = "c3_two_stage") -> "TwoStageModel":
        with (MODEL_DIR / f"{tag}.pkl").open("rb") as fh:
            return pickle.load(fh)


# ---------------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------------

def _classifier(seed_tag: str):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, early_stopping=False,
        random_state=int(config.seed_for(seed_tag)) % 2**31)


def _regressor(seed_tag: str):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.06, early_stopping=False,
        random_state=int(config.seed_for(seed_tag)) % 2**31)


def train(tr: pd.DataFrame, min_rows_per_region: int = 20,
          verbose: bool = True) -> TwoStageModel:
    """Fit the classifier and the per-region offset regressors.

    A region with fewer than `min_rows_per_region` training examples gets no
    regressor and falls back to the interval midpoint, which is reported rather
    than hidden -- a silent fallback would look like a working model.
    """
    x = tr[list(FEATURE_NAMES)].to_numpy(float)
    r = tr["region"].to_numpy(int)
    off = tr["offset"].to_numpy(float)

    clf = _classifier("c3_clf")
    clf.fit(x, r)

    offsets: dict[int, object] = {}
    for reg in range(N_SUB):
        sel = r == reg
        if sel.sum() < min_rows_per_region:
            offsets[reg] = None
            if verbose:
                print(f"   region {reg}: only {int(sel.sum())} rows -- "
                      f"falling back to interval midpoint")
            continue
        m = _regressor(f"c3_off{reg}")
        m.fit(x[sel], off[sel])
        offsets[reg] = m
        if verbose:
            print(f"   region {reg}: offset regressor on {int(sel.sum())} rows")

    return TwoStageModel(classifier=clf, offsets=offsets)


# ---------------------------------------------------------------------------
# VERIFICATION
# ---------------------------------------------------------------------------

def _fresh_context(sc, V, P) -> Context:
    return Context(v_oc=float(V[-1]), i_sc=None, module=str(sc.module),
                   geometry=str(sc.geometry),
                   rng=np.random.default_rng(0), _V=V, _P=P,
                   conditions={"irradiances": sc.irradiances,
                               "temp_c": sc.temp_c})


def test_probe_cost(model: TwoStageModel, verbose: bool = True) -> bool:
    """Both variants must spend exactly EXPECTED_PROBES per call.

    The cost claim is half the contribution. An adapter that probed more would
    not show up in the proxy metric at all, so it is asserted here -- and
    keep_best is asserted alongside, because a landing rule that quietly cost an
    extra probe would corrupt the comparison it exists to improve.
    """
    from .harness import scenario_set, curve_and_ceiling

    ok = True
    for keep_best in (False, True):
        fn = model.as_method(keep_best=keep_best)
        counts = []
        for sc in scenario_set(20)[:8]:
            V, P, _, _ = curve_and_ceiling(sc)
            ctx = _fresh_context(sc, V, P)
            fn(ctx)
            counts.append(ctx.n_probes)
        good = all(c == EXPECTED_PROBES for c in counts)
        ok &= good
        if verbose:
            label = "keep_best" if keep_best else "plain    "
            print(f"   {label} probes per call: {sorted(set(counts))} "
                  f"(expected {EXPECTED_PROBES})")
    if verbose:
        print(f"   Ahmed-Salam realistic scan costs ~33 (P2.4c); the default "
              f"budget measures 62.5")
        print(f"   -> {'PASS' if ok else 'FAIL'}")
    return ok


def test_keep_best_dominates(model: TwoStageModel, n: int = 120,
                             verbose: bool = True) -> bool:
    """keep_best must never land on less power than plain, at equal cost.

    The candidate set contains the plain landing, so this is a theorem rather
    than a hope -- which is exactly why it is worth asserting: a failure here
    means the recording wrapper is not seeing what the method measured, or the
    tie-break is inverted.
    """
    from .harness import scenario_set, curve_and_ceiling

    plain = model.as_method(keep_best=False)
    best = model.as_method(keep_best=True)

    worse, improved, probes_differ = 0, 0, 0
    gain = []
    for sc in scenario_set(n):
        V, P, _, _ = curve_and_ceiling(sc)

        c1 = _fresh_context(sc, V, P)
        v1 = plain(c1)
        p1 = float(np.interp(min(max(v1, V[0]), V[-1]), V, P))

        c2 = _fresh_context(sc, V, P)
        v2 = best(c2)
        p2 = float(np.interp(min(max(v2, V[0]), V[-1]), V, P))

        if c1.n_probes != c2.n_probes:
            probes_differ += 1
        if p2 < p1 - 1e-9:
            worse += 1
        elif p2 > p1 + 1e-9:
            improved += 1
            if p1 > 0:
                gain.append(100.0 * (p2 - p1) / p1)

    ok = worse == 0 and probes_differ == 0
    if verbose:
        print(f"   scenarios {n}")
        print(f"   keep_best improved the landing in {improved} "
              f"({improved / n:.1%})")
        if gain:
            print(f"   when it improved: median {np.median(gain):.3f}% power "
                  f"recovered, max {max(gain):.3f}%")
        print(f"   keep_best was WORSE in {worse}  (must be 0)")
        print(f"   probe counts differed in {probes_differ}  (must be 0)")
        print(f"   -> {'PASS' if ok else 'FAIL'}")
    return ok


def test_proxy_matches_harness(model: TwoStageModel, n: int = 200,
                               verbose: bool = True) -> bool:
    """The harness and the stored-curve proxy must agree ON THE MODEL.

    They already agree on the constant baselines to a few hundredths (P3.2:
    0.80 three-candidate 3.36% proxy vs 3.39% harness). This checks the model,
    which is the number that will be reported.

    Checked against the PLAIN variant: the proxy predicts k and evaluates the
    curve there, which is what plain does. keep_best deliberately departs from
    k, so a disagreement there would be the rule working, not a defect.

    Tolerance declared here: mean loss within 0.15 pt. The residual is the
    256-point stored grid against the harness's full curve.
    """
    from . import dataset
    from .harness import metrics, run_method, scenario_set

    scenarios = scenario_set(n)
    recs = run_method(model.as_method(keep_best=False), scenarios)
    harness_mean = metrics(recs)["mean_loss_pct"]

    df, curves = dataset.build(scenarios, verbose=False)
    x = df[list(FEATURE_NAMES)].to_numpy(float)
    k = model.predict_k(x)
    loss = dataset.power_loss_for_k(curves, np.arange(len(df)), k,
                                    df["p_gmpp"].to_numpy(float))
    proxy_mean = 100 * float(loss.mean())

    d = abs(harness_mean - proxy_mean)
    ok = d < 0.15
    if verbose:
        print(f"   harness {harness_mean:.3f}%   proxy {proxy_mean:.3f}%   "
              f"difference {d:.3f} pt")
        print(f"   -> {'PASS' if ok else 'FAIL'} (tolerance 0.15 pt)")
    return ok


def test_k_in_range(model: TwoStageModel, verbose: bool = True) -> bool:
    """Predicted coefficients must be physically sensible.

    A model emitting k outside roughly 0.1-1.0 is producing a landing voltage
    that is not on the curve, and the harness clamp would hide it.
    """
    from . import dataset
    from .harness import scenario_set
    df, _ = dataset.build(scenario_set(120), verbose=False)
    k = model.predict_k(df[list(FEATURE_NAMES)].to_numpy(float))
    ok = bool(np.all(k > 0.05) and np.all(k <= 1.0))
    if verbose:
        print(f"   predicted k: min {k.min():.3f}, max {k.max():.3f}, "
              f"mean {k.mean():.3f}")
        print(f"   true k in the pilot data ranged 0.237-0.933")
        print(f"   -> {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    from . import dataset

    print("C3 two-stage model and harness adapter")
    try:
        print("  " + config.provenance())
    except Exception:
        pass
    print("=" * 70)

    try:
        df, _ = dataset.load(tag="pilot")
    except FileNotFoundError:
        print("pilot dataset not found. Run phase3/p1_dataset_sizing.py first.")
        raise SystemExit(1)

    df["row_id"] = np.arange(len(df))
    df = dataset.assign_split(df, dataset.module_split())
    tr = df[df["split"] == "train"].reset_index(drop=True)
    print(f"\ntraining on {len(tr)} rows (validation modules excluded)")
    print("   NOTE: this self-test uses the legacy NAME split, which leaks at")
    print("   design level. It checks the adapter's mechanics, not its accuracy.")
    print("   The leakage-free evaluation is paper/experiments/"
          "e2_rebaseline_design_split.py.")
    model = train(tr)
    path = model.save()
    print(f"   model -> {path}")

    print("\n1. PROBE COST  (the cost claim, measured, both variants)")
    a = test_probe_cost(model)

    print("\n2. KEEP-BEST DOMINANCE  (never worse, never more expensive)")
    b = test_keep_best_dominates(model)

    print("\n3. PREDICTED k IN RANGE")
    c = test_k_in_range(model)

    print("\n4. HARNESS vs PROXY  (do the two metrics agree on the MODEL?)")
    d = test_proxy_matches_harness(model)

    print("\n" + "=" * 70)
    passed = sum([a, b, c, d])
    print(f"{passed}/4 checks passed")
    if passed == 4:
        print("Adapter verified. Next: re-run E2 with keep_best=True and")
        print("compare the tail against the plain row, then open the held-out")
        print("test modules once.")
    raise SystemExit(0 if passed == 4 else 1)