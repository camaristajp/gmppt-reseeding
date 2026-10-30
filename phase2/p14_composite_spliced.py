"""P2.9d  The composite in the FAST regime, and what a quiet day costs it.

WHY THIS EXISTS
---------------
The two-regime claim has two halves and only one of them is measured:

  measured    on GRADUAL windows (p13, n=24, 6 modules) the power-drop
              detector fires 0 times at every published dP/P, so the
              triggered arm collapses onto never-reseed (21.79%), and a
              clock recovers it. The energy-vs-period curve reproduced
              across n=12 -> n=24 with an interior optimum at 300 s.

  asserted    "...and the detector handles the fast relocations, so the
              composite handles both." The composite has NEVER been run on
              p12's spliced windows. The only thing we have shown it doing
              is nothing: 0.04 detector fires over the whole gradual ladder.

This runner measures the asserted half. It also settles a soundness problem
that closing the gap creates.

THE SOUNDNESS PROBLEM, AND WHY THE WINDOW LENGTH DECIDES IT
-----------------------------------------------------------
Our cleanest result is 0 of 60 false alarms on the uniform control, against
the ratchet's 23 of 60. A clock fires on a quiet window BY CONSTRUCTION, so
putting a clock floor under the detector appears to destroy that claim.

It does not, and the reason is a number nobody had looked at: p12's window is
SETTLE_STEPS + WINDOW_STEPS control steps -- 500 steps, 50 s, of which 300 are
scored. The derived period is 3000 steps. The clock is SIX TIMES longer than
the entire window, relocation and control alike.

Two consequences, and they pull in opposite directions:

  the good one   At the derived period the clock cannot tick inside a p12
                 window, so the composite is identical to the detector alone
                 on both the relocation windows and the control. The floor
                 costs nothing in the fast regime and the 0/60 survives. That
                 is a real result -- it is the statement that the floor is
                 free where detection already works.

  the honest one A control shorter than the clock period cannot test the
                 clock. Reporting 0/60 for a composite whose clock never had
                 the chance to tick would be vacuous, and quoting it as
                 evidence of soundness would be dishonest by omission.

So this runner reports BOTH halves, and PART B builds the control p12 cannot:
a quiet window long enough to contain many clock periods. On that window the
detector should still fire 0 times and the clock should fire span/period
times, which is the floor's true cost -- priced in fires, in probes per day,
and in energy lost on a day when nothing happens at all.

That last number is the one the paper needs. Every energy figure so far is
measured on a window selected FOR containing a relocation. The cost of a
periodic re-seed is paid on the windows that contain nothing, and until now
nothing in the programme has measured one.

WHAT IS COMPARED
----------------
  hybrid, never reseed            the floor: seed once, hold
  hybrid, triggered reseed        OURS as published -- the reference arm
  hybrid, scheduled 24 s          Eltamaly's inherited period
  hybrid, scheduled <derived>     the period our own sweep selected
  clock floor 24 s + detector     the composite at the inherited period
  clock floor <derived> + detector  the composite we propose

USAGE
    python phase2/p14_composite_spliced.py --n 60
    python phase2/p14_composite_spliced.py --n 60 --quiet-steps 40200
    python phase2/p14_composite_spliced.py --n 24 --derived-period 3000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phase2 import p12_relocation_comparison as p12          # noqa: E402
from phase2.p13_gradual_sweep import (ClockFloorPolicy,       # noqa: E402
                                      make_clock, make_clock_floor,
                                      probes_per_day,
                                      gradual_energy_loss_pct)
from phase2.transition import (TRANSITION_FAMILIES,           # noqa: E402
                               pattern_sequence, validate_sequence)
from gmppt import config                                      # noqa: E402
from gmppt.model import TwoStageModel                         # noqa: E402
from gmppt.tracking import DEFAULT_STEP_FRAC                  # noqa: E402

OUT = config.RESULTS_DIR / "phase2"

SETTLE_STEPS = p12.SETTLE_STEPS
WINDOW_STEPS = p12.WINDOW_STEPS
DERIVED_PERIOD = 3000       # steps = 300 s, the optimum from the p13 sweep
QUIET_STEPS = 40_200        # scored steps for the long control; matches the
#                             SHORTEST gradual window (~67 min), so the quiet
#                             day is priced on the same timescale as the
#                             events, not on a timescale chosen to flatter it.


# ---------------------------------------------------------------------------
# ARMS
# ---------------------------------------------------------------------------

def build_arms(model, derived, step_frac=DEFAULT_STEP_FRAC,
               competitors=()):
    """Six arms. Every clock is phase-matched through rescale_phase, so no
    period is scored at a luckier alignment than another."""
    base = int(p12.SCHEDULE_PERIOD_STEPS)

    def never(traj, **_):
        p12.t_hybrid_never(traj, model, 25.0, step_frac)
        # t_hybrid_never seeds once and charges the five probe steps against
        # its own budget (p12:507, n = traj.n_steps - SEED_PROBE_COST) but
        # never reports them: it calls seed_voltage directly and discards the
        # returned cost, bypassing the estimator path that sets
        # estimator_probes, which is what line 522 reads. p12's table computes
        # the column analytically and reports 5 for this arm; without this
        # line p14 reports 0 for the same single seed.
        # REPORTING ONLY -- the simulation, the step budget and every energy
        # figure are unchanged.
        traj.estimator_probes = ((getattr(traj, "estimator_probes", 0) or 0)
                                 + p12.SEED_PROBE_COST)

    def trig(traj, **_):
        p12._t_hybrid_repolicy(traj, model, 25.0, step_frac,
                               p12._make_policy("power_drop"))

    def sched_base(traj, phase=0, **_):
        p12._t_hybrid_repolicy(traj, model, 25.0, step_frac,
                               make_clock(base, phase))

    def sched_derived(traj, phase=0, **_):
        p12._t_hybrid_repolicy(traj, model, 25.0, step_frac,
                               make_clock(derived, phase))

    def comp_base(traj, phase=0, **_):
        pol = make_clock_floor(base, phase, p12.TRIGGER_DROP_FRAC)
        p12._t_hybrid_repolicy(traj, model, 25.0, step_frac, pol)
        traj._policy = pol

    def comp_derived(traj, phase=0, **_):
        pol = make_clock_floor(derived, phase, p12.TRIGGER_DROP_FRAC)
        p12._t_hybrid_repolicy(traj, model, 25.0, step_frac, pol)
        traj._policy = pol

    arms = [("hybrid, never reseed", never),
            ("hybrid, triggered reseed", trig),
            (f"hybrid, scheduled {base * 0.1:.0f} s", sched_base),
            (f"hybrid, scheduled {derived * 0.1:.0f} s", sched_derived),
            (f"clock floor {base * 0.1:.0f} s + detector", comp_base),
            (f"clock floor {derived * 0.1:.0f} s + detector", comp_derived)]

    # Published competitors under the SAME two policies as ours: the detector
    # alone (p12's reference arm) and the composite at the derived period.
    # This is what attributes the composite result to the POLICY rather than
    # to our estimator -- if a published learned seed completes the same
    # policy to the same number, the re-seeding strategy is carrying it.
    #
    # Only those two. A competitor under a bare clock adds nothing here: p13's
    # ladder already carries that, and the detector fires 0 times on gradual
    # windows for every arm.
    #
    # APPENDED, never prepended. PART A2 and PART B draw a clock phase inside
    # an arm-major loop, so an arm's phases depend on how many arms precede
    # it. Appending leaves the six existing arms bit-identical; inserting
    # anywhere earlier would silently re-phase all of them.
    for label, est in competitors:
        def mk_trig(e=est):
            def f(traj, phase=0, **_):
                p12._t_estimator_repolicy(
                    traj, e, p12._make_policy("power_drop"), step_frac)
            return f

        def mk_comp(e=est):
            def f(traj, phase=0, **_):
                pol = make_clock_floor(derived, phase, p12.TRIGGER_DROP_FRAC)
                p12._t_estimator_repolicy(traj, e, pol, step_frac)
                traj._policy = pol
            return f

        arms.append((f"{label}, triggered", mk_trig()))
        arms.append((f"{label}, clock floor {derived * 0.1:.0f} s + detector",
                     mk_comp()))
    return arms


def attribution(traj):
    """(clock fires, detector fires) if the arm carried a composite policy."""
    pol = getattr(traj, "_policy", None)
    if isinstance(pol, ClockFloorPolicy):
        return int(pol.n_clock), int(pol.n_detector)
    return None, None


# ---------------------------------------------------------------------------
# SELECTION  (p12's own, with traj_full kept so PART B can reuse the curves)
# ---------------------------------------------------------------------------

def select(args):
    all_val, split_label = p12.load_val_modules(args.split, args.split_file)
    pool_rng = np.random.default_rng(int(config.seed_for("p12_pool")) % 2**31)
    all_val = list(all_val)
    pool_rng.shuffle(all_val)
    val_modules = all_val[:max(args.n * 12, args.n)]

    buckets, seen, flat = {}, set(), []
    discarded = duplicate = 0
    for mod in val_modules:
        if sum(len(v) for v in buckets.values()) >= args.n * 3:
            break
        seq = pattern_sequence(str(mod), 25.0, family=args.family)
        try:
            ok, traj_full, per_step, _ = validate_sequence(seq, verbose=False)
        except RuntimeError:
            discarded += 1
            continue
        if not ok:
            discarded += 1
            continue
        eps_list = p12.find_adjacent_transitions(per_step)
        if not eps_list:
            discarded += 1
            continue
        for ti, (pre_idx, post_idx, _ra, _rb) in enumerate(eps_list):
            widx = (int(config.seed_for("p12_noise")) % 2**31
                    + 1000 * len(flat) + ti)
            build, meta = p12.make_window(traj_full, per_step,
                                          pre_idx, post_idx, widx)
            # p12's fingerprint, unchanged, so this scenario set IS p12's and
            # the runs join scenario by scenario.
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
            ctrl = p12.make_uniform_control(traj_full, post_idx, widx)
            quiet = make_long_quiet(traj_full, post_idx, widx,
                                    args.quiet_steps)
            buckets.setdefault((meta["region_pre"], meta["region_post"]),
                               []).append((build, meta, ctrl, quiet))
            flat.append(1)

    if args.balance_directions and len(buckets) > 1:
        per = max(1, args.n // len(buckets))
        picked = []
        for d in sorted(buckets):
            picked.extend(buckets[d][:per])
        for d in sorted(buckets):
            if len(picked) >= args.n:
                break
            picked.extend(buckets[d][per:per + args.n - len(picked)])
        picked = picked[:args.n]
    else:
        picked = [w for d in sorted(buckets) for w in buckets[d]][:args.n]
    return picked, split_label, discarded, duplicate


def make_long_quiet(traj_full, idx, win_seed, quiet_steps):
    """A quiet window LONG ENOUGH TO CONTAIN THE CLOCK PERIOD.

    p12's uniform control holds one curve for WINDOW_STEPS = 300 steps. That
    is 30 s, and it is shorter than the derived 300 s period, so the control
    cannot test a clock -- it can only fail to give one the chance to tick.
    This builds the same control at the length of a gradual window, which is
    the timescale on which a clock's cost is actually paid.

    The curve is the window's POST-transition curve, so the quiet day is the
    same physics as the relocation it follows, not an easier one.
    """
    curve = traj_full.curves[idx + 1]
    module, geometry = traj_full.module, traj_full.geometry
    p_gmpp = float(curve[3])

    def build():
        t = p12.NoisyTrajectory(
            curves=[curve, curve],
            block_steps=np.array([SETTLE_STEPS, int(quiet_steps)], int),
            scored=np.array([False, True], bool),
            profile_name="long_quiet_control", module=module,
            geometry=geometry)
        return t.configure_noise(p12.NOISE_I_RMS, p12.NOISE_V_RMS,
                                 np.random.default_rng(win_seed + 2_000_000))

    return build, p_gmpp, (curve, module, geometry, win_seed)


def _scale_curve(curve, f):
    """The same curve at a lower uniform irradiance, to first order.

    A uniform irradiance change scales photocurrent, and therefore power at a
    fixed voltage, very nearly linearly; it also shifts V_oc and V_mp by the
    logarithm of the ratio, which this ignores. That approximation is the
    RIGHT one for this test and the wrong one for a yield study: the detector
    compares power against an exponentially-weighted baseline of power, so a
    clean multiplicative decline at a known rate is exactly the stimulus its
    firing bound was derived for. What the approximation gives up is peak
    MOVEMENT -- which is the point. A passing cloud dims a module without
    reassigning which substring carries the global peak, and the question
    here is whether the detector fires on that.
    """
    c = list(curve)
    c[1] = np.asarray(c[1], float) * float(f)
    c[3] = float(c[3]) * float(f)
    return tuple(c)


def make_declining_ramp(ctx, rate, floor_frac, hold_steps):
    """A quiet window whose irradiance declines at a fixed fractional rate.

    PART B holds irradiance constant, so it answers "does the detector fire
    when nothing happens". It does not answer the reviewers' actual question,
    which is whether it fires when the irradiance falls without the peak
    moving. Section 5.3 says so in as many words: "these control windows hold
    the irradiance constant; a slow uniform decline in irradiance, such as a
    passing cloud, was not tested."

    The ramp runs from 1.0 down to `floor_frac` at `rate` per control step,
    one curve per step, then holds at the floor for `hold_steps` so the
    detector's persistence and re-arm have room to act after the decline
    stops. The oracle moves with the curve, so energy here is measured
    against what was actually available.

    The rate to compare against is eps*alpha = 0.005 per step, the firing
    bound derived in Section 5.3. A rate below it should produce no detector
    fire however deep the decline goes; a rate above it should fire.
    """
    curve, module, geometry, win_seed = ctx
    r = float(rate)
    n_ramp = max(1, int(np.ceil(np.log(float(floor_frac)) / np.log(1.0 - r))))
    factors = [(1.0 - r) ** k for k in range(1, n_ramp + 1)]
    curves = ([curve]
              + [_scale_curve(curve, f) for f in factors]
              + [_scale_curve(curve, factors[-1])])
    block_steps = np.array([SETTLE_STEPS] + [1] * n_ramp + [int(hold_steps)],
                           int)
    scored = np.array([False] + [True] * n_ramp + [True], bool)
    p_oracle = np.array([float(c[3]) for c in curves[1:-1]]
                        + [float(curves[-1][3])] * int(hold_steps), float)

    def build():
        t = p12.NoisyTrajectory(
            curves=curves, block_steps=block_steps, scored=scored,
            profile_name="declining_ramp_control", module=module,
            geometry=geometry)
        return t.configure_noise(p12.NOISE_I_RMS, p12.NOISE_V_RMS,
                                 np.random.default_rng(win_seed + 3_000_000))

    return build, p_oracle, n_ramp


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--family", default=p12.FAMILY_DEFAULT,
                    choices=list(TRANSITION_FAMILIES))
    ap.add_argument("--derived-period", type=int, default=DERIVED_PERIOD,
                    metavar="STEPS",
                    help=f"the period the p13 sweep selected, in control "
                         f"steps at 0.1 s. Default {DERIVED_PERIOD} "
                         f"(= {DERIVED_PERIOD * 0.1:.0f} s).")
    ap.add_argument("--quiet-steps", type=int, default=QUIET_STEPS,
                    metavar="STEPS",
                    help=f"scored steps in the LONG quiet control. Default "
                         f"{QUIET_STEPS:,} (~{QUIET_STEPS * 0.1 / 60:.0f} min), "
                         f"matching the shortest gradual window.")
    ap.add_argument("--quiet-n", type=int, default=12, metavar="N",
                    help="windows used for PART B. A clock's tick count on a "
                         "quiet window is near-deterministic, so this needs "
                         "far fewer windows than PART A; it is the expensive "
                         "half (6 arms x N x quiet-steps).")
    ap.add_argument("--skip-quiet", action="store_true",
                    help="skip PART B. It is the expensive half and the only "
                         "part that can price the clock floor, so skip it "
                         "only for a quick check of PART A.")
    ap.add_argument("--split", default="design", choices=["design", "name"])
    ap.add_argument("--split-file", default=None)
    ap.add_argument("--model-tag", default="c3_two_stage_design")
    ap.add_argument("--no-balance-directions", dest="balance_directions",
                    action="store_false")
    ap.add_argument("--with-competitors", action="store_true",
                    help="add the published competitors as estimator arms, "
                         "each under the detector alone and under the "
                         "composite at the derived period. Without this the "
                         "six original arms run exactly as before.")
    ap.add_argument("--jamaa-model", default=None, metavar="PATH",
                    help="saved Jamaa net (results/phase3/jamaa_net_"
                         "keepbest.joblib is the variant behind Table 4). "
                         "Without it the jamaa arm is SKIPPED WITH A MESSAGE "
                         "rather than silently dropped -- see "
                         "save_jamaa_net.md.")
    ap.add_argument("--ramp-control", action="store_true",
                    help="run PART C: a quiet control whose irradiance "
                         "DECLINES, at rates spanning the eps*alpha firing "
                         "bound. PART B holds irradiance constant and so "
                         "cannot answer the cloud-transient question.")
    ap.add_argument("--ramp-rates", type=float, nargs="+", default=None,
                    metavar="FRAC",
                    help="per-step fractional decline rates for PART C. "
                         "Default: 0.1, 0.5, 1, 2 and 5 times eps*alpha, the "
                         "derived firing bound.")
    ap.add_argument("--ramp-floor", type=float, default=0.5, metavar="FRAC",
                    help="irradiance fraction the ramp descends to before "
                         "holding. Default 0.5.")
    ap.add_argument("--ramp-hold", type=int, default=2000, metavar="STEPS",
                    help="steps held at the floor after the ramp, so the "
                         "detector's persistence and re-arm can act after "
                         "the decline stops. Default 2,000 (200 s).")
    ap.add_argument("--ramp-n", type=int, default=12, metavar="N",
                    help="windows used for PART C.")
    args = ap.parse_args()

    rng = np.random.default_rng(int(config.seed_for("p14_stats")) % 2**31)
    base = int(p12.SCHEDULE_PERIOD_STEPS)
    derived = int(args.derived_period)
    total_window = SETTLE_STEPS + WINDOW_STEPS
    if derived == base:
        print(f"[FAIL] --derived-period equals Eltamaly's {base} steps, so "
              f"two arms would")
        print("       carry the same name and silently collide in the "
              "results dict.")
        print("       Pass a different period, or read the 24 s rows as both.")
        return 2

    print("P2.9d  the composite in the FAST regime + the cost of a quiet day")
    try:
        print("  " + config.provenance())
    except Exception:
        pass
    print("=" * 78)
    print(f"family {args.family}; spliced window {SETTLE_STEPS} settle + "
          f"{WINDOW_STEPS} scored = {total_window} steps "
          f"({total_window * 0.1:.0f} s)")
    print(f"trigger: power_drop, dP/P = {p12.TRIGGER_DROP_FRAC}, "
          f"EMA alpha {p12.TRIGGER_EMA_ALPHA}, persist {p12.TRIGGER_PERSIST}")
    print(f"clocks: {base} steps ({base * 0.1:.0f} s, Eltamaly T_PD) and "
          f"{derived} steps ({derived * 0.1:.0f} s, derived in p13)")
    print("DECLARED SYNTHETIC site (Jeju reference). Simulation study.")
    print("MEASUREMENT NOISE paired: current 1.0% RMS, voltage 0.5% RMS.")

    p_tick = min(1.0, total_window / derived)
    print("\nON RECORD BEFORE THIS RUN:")
    print(f"   The derived period ({derived} steps) is "
          f"{derived / total_window:.1f}x the whole spliced window "
          f"({total_window} steps).")
    print("   That does NOT mean the clock cannot tick inside one: the window")
    print("   is a slice of continuous operation, so the clock's PHASE decides")
    print("   whether a tick lands in it. With the phase drawn uniformly, one")
    print(f"   tick lands with probability ~ window/period = {p_tick:.2f}, and")
    print("   never two. Predictions:")
    print(f"     (a) the composite at the derived period ticks on ~"
          f"{100 * p_tick:.0f}% of")
    print("         windows and is EXACTLY equal to the triggered arm on every")
    print("         window where it does not. That is the real claim -- the")
    print("         floor is free where detection already works -- and it is")
    print("         tested per window below, not in aggregate.")
    print(f"     (b) on p12's control it likewise fires on ~"
          f"{100 * p_tick:.0f}% of windows,")
    print("         so the 0-of-n result does NOT survive as an 'any fire'")
    print("         count. It survives as a DETECTOR count, which is the only")
    print("         thing a false-alarm rate can mean. Attribution is what")
    print("         rescues the claim, not the window being short.")
    print("     (c) the composite at 24 s ticks ~"
          f"{total_window / base:.0f}x per window, so it")
    print("         fires on essentially every control window.")
    print("     (d) a 50 s control sees a 300 s clock at "
          f"{p_tick:.2f} of its true rate,")
    print("         so it cannot price the floor. PART B is the control that "
          "can.")

    model = TwoStageModel.load(tag=args.model_tag)
    picked, split_label, discarded, duplicate = select(args)
    print(f"\nsplit: {split_label}\n   model tag {args.model_tag}")
    if len(picked) < 3:
        print(f"only {len(picked)} clean windows; raise --n")
        return 1

    cnt = {}
    for _b, m, _c, _q in picked:
        cnt[(m["region_pre"], m["region_post"])] = cnt.get(
            (m["region_pre"], m["region_post"]), 0) + 1
    print(f"\n{len(picked)} SPLICED windows, {discarded} discarded, "
          f"{duplicate} duplicates rejected")
    print(f"   directions: "
          f"{', '.join(f'{a}->{b} (n={cnt[(a, b)]})' for a, b in sorted(cnt))}")
    dd = [m["drop_depth"] for _b, m, _c, _q in picked]
    print(f"   drop depth: median {np.median(dd):.3f}, "
          f"range {min(dd):.3f}-{max(dd):.3f}")
    print(f"   modules: {len({m['module'] for _b, m, _c, _q in picked})} "
          f"distinct")

    competitors = []
    if args.with_competitors:
        print("\nloading published competitors as estimator arms:")
        competitors = p12.load_competitors(args.jamaa_model)
        for lab, _ in competitors:
            print(f"   [OK] {lab}")
        if not competitors:
            print("   none loaded -- the composite table will have NO "
                  "published")
            print("   learned estimator in it, which is the comparison that "
                  "attributes")
            print("   the result to the policy rather than to our seed.")

    arms = build_arms(model, derived, competitors=competitors)
    names = [n for n, _ in arms]
    per = {n: {"reconv": [], "energy": [], "n_fires": [], "probes": [],
               "clk": [], "det": []} for n in names}

    # ---- PART A1: the relocation windows ----------------------------------
    for wi, (build, meta, _c, _q) in enumerate(picked):
        phase = int(rng.integers(0, base))
        for name, fn in arms:
            traj = build()
            fn(traj, phase=phase, seed=wi)
            fires = getattr(traj, "trigger_fires", None) or []
            c, d = attribution(traj)
            per[name]["reconv"].append(
                p12.reconvergence_steps(traj, meta["p_gmpp_new"]))
            per[name]["energy"].append(
                p12.transition_energy_loss_pct(traj, meta["p_gmpp_new"]))
            per[name]["n_fires"].append(len(fires))
            per[name]["probes"].append(
                int(getattr(traj, "estimator_probes", 0) or 0))
            if c is not None:
                per[name]["clk"].append(c)
                per[name]["det"].append(d)

    print("\n" + "=" * 78)
    print("PART A1  RE-CONVERGENCE AFTER THE JUMP  (spliced; 'cens' = never)")
    print(f"   {'arm':<34} {'median':>7} {'re-conv %':>10} "
          f"{'energy %':>9} {'fires':>7} {'probes':>7}")
    print("   " + "-" * 78)
    stats = {}
    for n in names:
        rc = per[n]["reconv"]
        got = [x for x in rc if x is not None]
        frac = 100.0 * len(got) / max(len(rc), 1)
        med = float(np.median(got)) if got else float("nan")
        stats[n] = {"median_reconv": med, "reconv_pct": frac,
                    "energy_loss_pct": float(np.mean(per[n]["energy"])),
                    "mean_fires": float(np.mean(per[n]["n_fires"])),
                    "mean_probes": float(np.mean(per[n]["probes"])),
                    "reconv": rc, "energy": per[n]["energy"],
                    "n_fires": per[n]["n_fires"]}
        if per[n]["clk"]:
            stats[n]["mean_fires_from_clock"] = float(np.mean(per[n]["clk"]))
            stats[n]["mean_fires_from_detector"] = float(np.mean(per[n]["det"]))
        print(f"   {n:<34} {med:>7.0f} {frac:>9.0f}% "
              f"{stats[n]['energy_loss_pct']:>9.2f} "
              f"{stats[n]['mean_fires']:>7.2f} "
              f"{stats[n]['mean_probes']:>7.0f}")

    ref = "hybrid, triggered reseed"
    comp_d = f"clock floor {derived * 0.1:.0f} s + detector"
    # Prediction (a), tested CONDITIONALLY on whether the clock ticked. The
    # unconditional "identical?" test would fail by design on the ~1-in-6
    # windows whose phase puts a tick inside them, and would read as a bug.
    ticked = [c > 0 for c in per[comp_d]["clk"]]
    quiet_idx = [i for i, t in enumerate(ticked) if not t]
    same_quiet = all(stats[comp_d]["reconv"][i] == stats[ref]["reconv"][i]
                     for i in quiet_idx)
    rate = (sum(ticked) / len(ticked)) if ticked else float("nan")
    print(f"\n   prediction (a): ticked on {sum(ticked)}/{len(ticked)} windows "
          f"(rate {rate:.2f}, expected {p_tick:.2f})")
    print(f"      on the {len(quiet_idx)} windows with NO tick, "
          f"re-convergence identical to triggered: "
          f"{'YES' if same_quiet else 'NO'}")
    if same_quiet:
        print("      -> the clock floor is FREE where detection already works,")
        print("         exactly and per window, not on average.")
    else:
        print("      -> NOT identical on the no-tick windows. That IS a bug:")
        print("         with no tick the composite reduces to the detector, so")
        print("         any difference is the composite changing the "
              "detector's")
        print("         own behaviour. Stop and find it.")
    if ticked and abs(rate - p_tick) > 3.0 * (
            (p_tick * (1 - p_tick) / max(len(ticked), 1)) ** 0.5 + 1e-9):
        print(f"      note: the tick rate is more than 3 SE from "
              f"window/period. With")
        print("         few windows that is unremarkable; at large n it would "
              "mean the")
        print("         phase is not uniform over the period.")

    cmp_ = p12.paired_compare(stats[comp_d]["reconv"], stats[ref]["reconv"],
                              WINDOW_STEPS, rng)
    print(f"\n   paired, composite vs triggered: "
          f"median {cmp_['median_diff']:+.0f} steps, "
          f"CI [{cmp_['ci_lo']:+.0f}, {cmp_['ci_hi']:+.0f}], "
          f"p = {cmp_['p']:.4f}  (n={cmp_['n']})")

    # ---- PART A2: p12's 50 s uniform control ------------------------------
    print("\n" + "=" * 78)
    print("PART A2  p12's UNIFORM CONTROL  (50 s -- nothing moves)")
    print("   Fires are ATTRIBUTED, because a false alarm is a DETECTOR's")
    print("   mistake and a clock makes no claim to detect. A clock's cost is")
    print("   probes, which is priced in PART B, not soundness.")
    print(f"\n   {'arm':<34} {'any fire':>9} {'det fire':>9} "
          f"{'clk/win':>8} {'det/win':>8}")
    print("   " + "-" * 72)
    ctrl = {}
    # Phases are drawn ONCE PER WINDOW and shared across every arm.
    # Drawn inside the arm loop instead, two arms carrying an identical
    # clock see different phases on the same window and are not matched
    # to each other -- measured on 27 Oct as 11/60, 8/60, 5/60 and 12/60
    # clock ticks for four arms with the same 360 s clock, two of which
    # moved again when an unrelated arm was added. PART A1 already did
    # this correctly; A2, B and C did not.
    ctrl_phases = [int(rng.integers(0, base)) for _ in picked]
    for name, fn in arms:
        any_f = det_f = 0
        clk_t, det_t = [], []
        for ui, (_b, _m, cbuild, _q) in enumerate(picked):
            phase = ctrl_phases[ui]
            traj = cbuild()
            fn(traj, phase=phase, seed=ui)
            fires = getattr(traj, "trigger_fires", None) or []
            c, d = attribution(traj)
            if fires:
                any_f += 1
            if d is None:
                # Not a composite: every fire is this arm's own mechanism.
                if "scheduled" in name:
                    clk_t.append(len(fires))
                    det_t.append(0)
                    d_this = 0
                else:
                    clk_t.append(0)
                    det_t.append(len(fires))
                    d_this = len(fires)
            else:
                clk_t.append(c)
                det_t.append(d)
                d_this = d
            if d_this:
                det_f += 1
        ctrl[name] = {"windows_any_fire": any_f,
                      "windows_detector_fire": det_f,
                      "mean_clock_per_window": float(np.mean(clk_t)),
                      "mean_detector_per_window": float(np.mean(det_t)),
                      "n_control": len(picked)}
        print(f"   {name:<34} {any_f:>4}/{len(picked):<4} "
              f"{det_f:>4}/{len(picked):<4} "
              f"{float(np.mean(clk_t)):>8.2f} {float(np.mean(det_t)):>8.2f}")

    print("\n   Read the 'det fire' column as the soundness result: it is the")
    print("   detector's false alarms, unchanged by whatever clock sits under")
    print("   it. The 'any fire' column is not a soundness number for an arm")
    print("   with a clock in it -- a clock firing on a quiet window is the")
    print("   clock working as specified, and its cost is probes.")
    print(f"\n   NOTE: this control is {total_window * 0.1:.0f} s long and the "
          f"derived period is {derived * 0.1:.0f} s, so a derived-period clock")
    print(f"   is seen here at only ~{p_tick:.2f} of its true rate. This table")
    print("   understates a slow clock's cost by that factor and cannot be "
          "used to")
    print("   price the floor. PART B is the control that can.")

    # ---- PART B: the long quiet control -----------------------------------
    quiet = {}
    if not args.skip_quiet:
        qs = int(args.quiet_steps)
        print("\n" + "=" * 78)
        print(f"PART B  LONG QUIET CONTROL  ({qs:,} scored steps = "
              f"{qs * 0.1 / 60:.0f} min, nothing moves)")
        print("   Every energy figure in the programme so far is measured on a")
        print("   window selected FOR containing a relocation. A periodic")
        print("   re-seed is paid for on the windows that contain nothing, and")
        print("   nothing until now has measured one. Here the oracle is flat,")
        print("   so energy lost IS the cost of re-seeding and nothing else.")
        print(f"\n   {'arm':<34} {'fires':>7} {'probes':>7} "
              f"{'probes/day':>11} {'energy %':>9} {'det fire':>9}")
        print("   " + "-" * 80)
        span_s = qs * 0.1
        q_windows = picked[:max(1, min(int(args.quiet_n), len(picked)))]
        print(f"   using {len(q_windows)} of {len(picked)} windows for this "
              f"part (--quiet-n)")
        # One phase per quiet window, shared across arms -- see PART A2.
        quiet_phases = [int(rng.integers(0, base)) for _ in q_windows]
        for name, fn in arms:
            nf, pr, prd, en = [], [], [], []
            det_f = 0
            for ui, (_b, _m, _c, (qbuild, p_gmpp, _rctx)) in \
                    enumerate(q_windows):
                phase = quiet_phases[ui]
                traj = qbuild()
                fn(traj, phase=phase, seed=ui)
                fires = getattr(traj, "trigger_fires", None) or []
                probes = int(getattr(traj, "estimator_probes", 0) or 0)
                c, d = attribution(traj)
                nf.append(len(fires))
                pr.append(probes)
                prd.append(probes_per_day(probes, span_s))
                en.append(p12.transition_energy_loss_pct(traj, p_gmpp))
                if d is None:
                    if "scheduled" not in name and fires:
                        det_f += 1
                elif d:
                    det_f += 1
            quiet[name] = {"mean_fires": float(np.mean(nf)),
                           "mean_probes": float(np.mean(pr)),
                           "mean_probes_per_day": float(np.mean(prd)),
                           "energy_loss_pct": float(np.mean(en)),
                           "energy_sd": float(np.std(en)),
                           "energy_max": float(np.max(en)),
                           "windows_detector_fire": det_f,
                           "n_control": len(q_windows), "scored_steps": qs,
                           "fires": nf, "energy": en}
            print(f"   {name:<34} {float(np.mean(nf)):>7.2f} "
                  f"{float(np.mean(pr)):>7.0f} "
                  f"{float(np.mean(prd)):>11,.0f} "
                  f"{float(np.mean(en)):>9.2f} "
                  f"{det_f:>4}/{len(q_windows):<4}")

        expect_base = qs / base
        expect_der = qs / derived
        print(f"\n   expected clock ticks at this length: "
              f"{expect_base:.0f} at {base} steps, "
              f"{expect_der:.1f} at {derived} steps")
        trig_q = quiet.get(ref, {})
        if trig_q.get("windows_detector_fire", 0) == 0:
            print("   the DETECTOR fires 0 times on a quiet window of this "
                  "length too, so")
            print("   its soundness is not an artifact of the 50 s control "
                  "being short.")
        else:
            print(f"   the detector fires on "
                  f"{trig_q['windows_detector_fire']}/{len(q_windows)} quiet "
                  f"windows at this length --")
            print("   the 0/n result DOES depend on the short control. Report "
                  "this, and")
            print("   re-state the soundness claim at the length that was "
                  "actually tested.")
        cd = quiet.get(comp_d, {})
        cb = quiet.get(f"clock floor {base * 0.1:.0f} s + detector", {})
        if cd and cb:
            print(f"\n   cost of the floor on a quiet window, in energy:")
            print(f"      at {base * 0.1:.0f} s   "
                  f"{cb['energy_loss_pct']:.2f}% "
                  f"({cb['mean_probes_per_day']:,.0f} probes/day)")
            print(f"      at {derived * 0.1:.0f} s  "
                  f"{cd['energy_loss_pct']:.2f}% "
                  f"({cd['mean_probes_per_day']:,.0f} probes/day)")
            print("   This is the number to quote when a reviewer asks what "
                  "the clock")
            print("   costs when nothing is happening, which is most of the "
                  "day.")

    # ---- PART C: the declining-irradiance control -------------------------
    ramp = {}
    if args.ramp_control:
        r_crit = p12.TRIGGER_DROP_FRAC * p12.TRIGGER_EMA_ALPHA
        rates = [float(x) for x in (args.ramp_rates
                 or [m * r_crit for m in (0.1, 0.5, 1.0, 2.0, 5.0)])]
        want = ("hybrid, never reseed", ref, comp_d)
        ramp_arms = [(n, fn) for n, fn in arms if n in want]
        r_windows = picked[:max(1, min(int(args.ramp_n), len(picked)))]

        print("\n" + "=" * 78)
        print("PART C  DECLINING-IRRADIANCE CONTROL  (the cloud question)")
        print("   PART B holds irradiance CONSTANT, so it answers 'does the")
        print("   detector fire when nothing happens'. It does not answer the")
        print("   reviewers' question, which is whether it fires when the")
        print("   irradiance FALLS without the peak moving. Here it falls, at")
        print("   rates spanning the derived firing bound.")
        print(f"\n   firing bound eps*alpha = {p12.TRIGGER_DROP_FRAC:.2f} x "
              f"{p12.TRIGGER_EMA_ALPHA:.2f} = {r_crit:.4f} per control step")
        print(f"   ramp descends to {args.ramp_floor:.2f} of initial "
              f"irradiance, then holds {args.ramp_hold:,} steps")
        print(f"   {len(r_windows)} windows, {len(ramp_arms)} arms, "
              f"{len(rates)} rates")
        print("\n   PREDICTION, on record before this run:")
        print("      a decline SLOWER than eps*alpha produces no detector fire")
        print("      however deep it goes, because the exponential baseline")
        print("      sinks with it. A decline FASTER than eps*alpha fires. If")
        print("      the boundary does not sit near 1.0x, the bound in Section")
        print("      5.3 is wrong and the structural argument goes with it.")
        print(f"\n   {'rate/step':>10} {'x bound':>8} {'ramp s':>8} "
              f"{'arm':<34} {'det/win':>8} {'det wins':>9} {'energy %':>9}")
        print("   " + "-" * 92)

        for rate in rates:
            # One phase per window at this rate, shared across arms.
            ramp_phases = [int(rng.integers(0, base)) for _ in r_windows]
            for name, fn in ramp_arms:
                det_per, det_wins, en, in_ramp, n_ramp_last = [], 0, [], 0, 0
                for ui, (_b, _m, _c, (_qb, _pg, rctx)) in \
                        enumerate(r_windows):
                    rbuild, p_or, n_ramp = make_declining_ramp(
                        rctx, rate, args.ramp_floor, args.ramp_hold)
                    n_ramp_last = n_ramp
                    phase = ramp_phases[ui]
                    traj = rbuild()
                    fn(traj, phase=phase, seed=ui)
                    fires = getattr(traj, "trigger_fires", None) or []
                    c, d = attribution(traj)
                    if d is None:
                        d_this = 0 if "scheduled" in name else len(fires)
                    else:
                        d_this = d
                    det_per.append(d_this)
                    if d_this:
                        det_wins += 1
                    # Fires landing inside the declining span, as opposed to
                    # during the hold that follows it. A fire after the ramp
                    # has stopped is a different event from one caused by it.
                    in_ramp += sum(1 for x in fires
                                   if int(x) < SETTLE_STEPS + n_ramp)
                    en.append(gradual_energy_loss_pct(traj, p_or))
                key = f"rate_{rate:.5f}_{name}"
                ramp[key] = {"rate_per_step": rate,
                             "rate_over_bound": rate / max(r_crit, 1e-12),
                             "arm": name,
                             "ramp_steps": int(n_ramp_last),
                             "ramp_seconds": float(n_ramp_last * 0.1),
                             "floor_frac": float(args.ramp_floor),
                             "hold_steps": int(args.ramp_hold),
                             "mean_detector_fires": float(np.mean(det_per)),
                             "windows_detector_fire": int(det_wins),
                             "fires_inside_ramp": int(in_ramp),
                             "n_control": len(r_windows),
                             "energy_loss_pct": float(np.mean(en)),
                             "detector_fires": det_per, "energy": en}
                print(f"   {rate:>10.4f} {rate / r_crit:>8.1f} "
                      f"{n_ramp_last * 0.1:>8.1f} {name:<34} "
                      f"{float(np.mean(det_per)):>8.2f} "
                      f"{det_wins:>4}/{len(r_windows):<4} "
                      f"{float(np.mean(en)):>9.2f}")

        # -- where does the boundary actually sit? ---------------------------
        det_rows = [r for r in ramp.values() if r["arm"] == ref]
        det_rows.sort(key=lambda r: r["rate_per_step"])
        fired = [r for r in det_rows if r["windows_detector_fire"] > 0]
        silent = [r for r in det_rows if r["windows_detector_fire"] == 0]
        print("\n   DETECTOR-ALONE ARM, the soundness result:")
        if not fired:
            print(f"      no detector fire at ANY rate tested, up to "
                  f"{det_rows[-1]['rate_over_bound']:.1f}x the bound.")
            print("      Either the bound is conservative by more than the")
            print("      ladder spans, or the ramp is not reaching the "
                  "detector.")
            print("      Check that a rate well above the bound fires before")
            print("      reporting this as a false-alarm result.")
        elif not silent:
            print("      the detector fires at EVERY rate tested, including")
            print(f"      {det_rows[0]['rate_over_bound']:.1f}x the bound. "
                  f"That contradicts")
            print("      the structural argument in Section 5.3. Do not "
                  "report the")
            print("      bound until this is explained.")
        else:
            lo = max(silent, key=lambda r: r["rate_per_step"])
            hi = min(fired, key=lambda r: r["rate_per_step"])
            print(f"      silent up to {lo['rate_per_step']:.4f}/step "
                  f"({lo['rate_over_bound']:.1f}x the bound)")
            print(f"      fires from  {hi['rate_per_step']:.4f}/step "
                  f"({hi['rate_over_bound']:.1f}x the bound)")
            print("      -> the boundary is bracketed. If it straddles 1.0x,")
            print("         the derived bound predicts the measured "
                  "behaviour and")
            print("         Section 5.3's structural argument is confirmed "
                  "rather")
            print("         than merely asserted.")
        print("\n   A cloud that dims a module by half in 30 s declines at")
        print(f"      {1 - 0.5 ** (1 / 300):.4f}/step = "
              f"{(1 - 0.5 ** (1 / 300)) / r_crit:.2f}x the bound -- which is "
              f"where")
        print("      this table should be read for the deployment question.")

    payload = {"runner": "p14_composite_spliced", "family": args.family,
               "n_windows": len(picked), "split": args.split,
               "split_description": split_label, "model_tag": args.model_tag,
               "jamaa_model": args.jamaa_model,
               "phase_matched_controls": True,
               "competitors": [lab for lab, _ in competitors],
               "settle_steps": SETTLE_STEPS, "window_steps": WINDOW_STEPS,
               "base_period_steps": base, "derived_period_steps": derived,
               "quiet_steps": int(args.quiet_steps),
               "derived_period_over_window":
                   float(derived / total_window),
               "trigger_drop_frac": p12.TRIGGER_DROP_FRAC,
               "trigger_persist": p12.TRIGGER_PERSIST,
               "trigger_ema_alpha": p12.TRIGGER_EMA_ALPHA,
               "window_meta": [m for _b, m, _c, _q in picked],
               "relocation": stats, "uniform_control_50s": ctrl,
               "long_quiet_control": quiet,
               "declining_ramp_control": ramp,
               "paired_composite_vs_triggered": cmp_}
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"composite_spliced_{args.family}.json"
    out.write_text(json.dumps(p12._json_safe(payload), indent=2,
                              default=float), encoding="utf-8")
    print(f"\nresults -> {out}")
    print("   scenario_sig in window_meta is p12's, so this joins p12 and p13")
    print("   scenario by scenario.")
    print("Validation modules; declared synthetic site (Jeju reference). "
          "All simulated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())