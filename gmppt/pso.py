"""
pso.py  --  particle swarm optimisation as a GMPPT tracker.

PLACE THIS FILE AT:   C:\\Users\\user\\gmppt\\gmppt\\pso.py
                      NEW FILE. gmppt/trackers.py keeps its gated pso() stub
                      until this is accepted; the stub then re-exports from here.

SELF-TEST:
    python -m gmppt.pso

WHY PSO IS THE LAST BASELINE, AND WHY IT WAS GATED

    The funded proposal commits to benchmarking against P&O, InC and PSO
    (sections 4.1 and 4.4), and the convergence-time target is stated as a >= 50%
    reduction AGAINST METAHEURISTICS. PSO is the metaheuristic that target refers
    to, which makes it the most consequential of the three baselines.

    It was gated not on the Month-3 meeting but on an unmade decision, because
    the decision determines the benchmark rather than merely the implementation.

THE COST MAPPING -- the decision, declared here before any number exists

    PSO evaluates a POPULATION of candidate voltages per iteration. A control
    step commands ONE voltage. How the population maps onto control steps decides
    what PSO costs, and therefore decides the comparison.

        SEQUENTIAL   M particles cost M control steps per iteration.
        PARALLEL     one iteration costs one control step.

    SEQUENTIAL IS PRIMARY. A module-level optimizer has one converter and can
    hold one voltage at a time; there is no way to sample five particles
    simultaneously. It is also the accounting already applied to every other
    method in this project -- the seed's five feature probes are charged as five
    control steps, and Ahmed-Salam was costed at its realistic ~33 probes rather
    than its default budget. Switching conventions for PSO alone would be the
    comparison-across-non-comparable-conditions error recorded as D2.

    PARALLEL IS REPORTED ALONGSIDE, because much of the PSO-MPPT literature
    reports iterations rather than samples, and omitting it invites the objection
    that PSO was over-costed. Both columns appear; neither is hidden.

    Under the sequential mapping PSO looks expensive and the hybrid wins on cost.
    That asymmetry is exactly why the parallel column must also be present.

PSO IS THE FIRST STOCHASTIC METHOD IN THIS PROJECT

    Every tracker measured so far is deterministic: the same scenario produces
    the same trajectory every time. PSO's particles are initialised randomly and
    move with random coefficients, so the same scenario produces a different
    answer on every run.

    Consequence for reporting: a single-seed PSO figure is one draw presented as
    a result. Every PSO number here is averaged over N_SEEDS runs and reported
    with its spread. Where the spread is comparable to a difference being
    claimed, the difference is not a difference.

    This also means PSO's reproducibility rests on config.seed_for rather than on
    determinism, which is why the seeds are derived from MASTER_SEED and not from
    the system clock.

WHAT HAPPENS AFTER THE SEARCH FINISHES -- declared

    PSO converges in far fewer control steps than the 400-step window. Left
    running, its particles continue to move and the steady-efficiency figure
    would measure a swarm still wandering rather than a tracker holding station.

    Declared behaviour: once the iteration budget is exhausted, the tracker HOLDS
    at the global best position for the remainder of the window. This is what a
    deployed PSO-MPPT does -- it searches, converges, and holds until a change is
    detected. Re-initialisation on change detection is a DYNAMIC feature and is
    out of scope here; it belongs to P2.8.

ALGORITHM

    For each particle i with position x_i (a voltage) and velocity v_i:

        v_i(k+1) = w*v_i(k) + c1*r1*(p_i - x_i(k)) + c2*r2*(g - x_i(k))
        x_i(k+1) = x_i(k) + v_i(k+1)

    where p_i is that particle's best-known position, g is the swarm's best, r1
    and r2 are uniform random draws, w is inertia, and c1, c2 are the cognitive
    and social coefficients. Positions are clamped to the usable voltage range;
    velocities are clamped to a fraction of V_oc to prevent particles leaving the
    curve in one step.

    Every position evaluation is a traj.step call, so the cost is COUNTED by the
    trajectory rather than asserted here.

PARAMETERS -- swept, not chosen

    Population size is swept, exactly as P&O's step size and InC's scaling
    constant were. That sweep is what forecloses the objection that a baseline
    was handicapped by a poorly chosen parameter, and it is the treatment both
    other baselines received.

    Inertia and the acceleration coefficients are set to values standard in the
    PSO-MPPT literature (w = 0.4, c1 = c2 = 1.2 in the widely used
    Ishaque/Salam parameterisation) and held fixed, with the population sweep
    carrying the fairness argument. Sweeping four parameters at once would
    produce a table nobody can read and would not change the conclusion, since
    arrival rate is the axis that matters and it is governed by whether the swarm
    covers the curve at all.

PREDICTION, DECLARED BEFORE THE RUN

    PSO should reach the global peak on AT LEAST 90% of multi-peak scenarios. It
    is a population method: particles scattered across the voltage range sample
    multiple regions, so it is not subject to the local-peak trapping that holds
    P&O and InC at 42.1%.

    Its cost should be high. With 5 particles over 20 iterations the sequential
    mapping spends 100 control steps against the hybrid's 5.

    IF PSO'S ARRIVAL COMES OUT NEAR 42%, THE IMPLEMENTATION IS WRONG. A
    metaheuristic that traps like a hill-climber has lost the property that
    defines it, and no number from it should be reported until that is resolved.

SCOPE
    Static shading only. All figures simulated.
"""

from __future__ import annotations

import numpy as np

from . import config
from .tracking import N_STEPS, Trajectory

N_SUB = int(config.N_SUBSTRINGS)

# Literature-standard PSO-MPPT coefficients, held fixed.
INERTIA = 0.4
C_COGNITIVE = 1.2
C_SOCIAL = 1.2

# Swept, as P&O's step and InC's scaling constant were.
POPULATION_SWEEP = (3, 4, 5, 6)
DEFAULT_POPULATION = 5
DEFAULT_ITERATIONS = 20

V_MIN_FRAC = 0.05        # particles never driven below this fraction of V_oc
V_MAX_FRAC = 0.98        # nor above this
VEL_MAX_FRAC = 0.25      # velocity clamp, fraction of V_oc per iteration

N_SEEDS = 5              # PSO is stochastic; one draw is not a result


def particle_swarm(traj: Trajectory, temp_c: float | None = None,
                   n_steps: int = N_STEPS,
                   population: int = DEFAULT_POPULATION,
                   iterations: int = DEFAULT_ITERATIONS,
                   seed: int = 0, **_) -> None:
    """PSO over the voltage range, then hold at the global best.

    Every position evaluation is a traj.step call, so the control-step cost is
    measured by the trajectory. This is the SEQUENTIAL mapping: M particles cost
    M steps per iteration. The parallel figure is derived in the runner by
    dividing, not by running a different algorithm.

    `temp_c` is accepted and unused so every tracker shares one runner signature.
    """
    rng = np.random.default_rng(
        (int(config.seed_for("pso")) + seed * 7919) % 2**31)

    v_lo = max(V_MIN_FRAC * traj.v_oc, traj.v_floor)
    v_hi = V_MAX_FRAC * traj.v_oc
    vel_max = VEL_MAX_FRAC * traj.v_oc

    # Spread the initial swarm evenly and jitter it, so the population covers
    # the curve on the first iteration rather than relying on luck.
    base = np.linspace(v_lo, v_hi, population)
    jitter = (rng.random(population) - 0.5) * (v_hi - v_lo) / (2 * population)
    x = np.clip(base + jitter, v_lo, v_hi)
    vel = np.zeros(population, dtype=float)

    budget = n_steps
    used = 0

    p_best_x = x.copy()
    p_best_p = np.full(population, -np.inf)
    g_best_x, g_best_p = float(x[0]), -np.inf

    for _ in range(iterations):
        for i in range(population):
            if used >= budget:
                break
            p = traj.step(float(x[i]))
            used += 1
            if p > p_best_p[i]:
                p_best_p[i], p_best_x[i] = p, float(x[i])
            if p > g_best_p:
                g_best_p, g_best_x = p, float(x[i])
        if used >= budget:
            break

        r1 = rng.random(population)
        r2 = rng.random(population)
        vel = (INERTIA * vel
               + C_COGNITIVE * r1 * (p_best_x - x)
               + C_SOCIAL * r2 * (g_best_x - x))
        vel = np.clip(vel, -vel_max, vel_max)
        x = np.clip(x + vel, v_lo, v_hi)

    # Hold at the global best for the remainder of the window. Declared: a
    # deployed PSO-MPPT searches, converges, and holds until a change is
    # detected. Re-initialisation on change is a P2.8 feature.
    while used < budget:
        traj.step(g_best_x)
        used += 1


def search_cost(population: int, iterations: int) -> dict:
    """Control-step cost of the search phase under both declared mappings."""
    return {"population": int(population), "iterations": int(iterations),
            "sequential_steps": int(population * iterations),
            "parallel_steps": int(iterations)}


# ---------------------------------------------------------------------------
# VERIFICATION
# ---------------------------------------------------------------------------

def test_cost_is_counted(verbose: bool = True) -> bool:
    """The search phase must spend exactly population * iterations steps.

    The cost claim decides the funded convergence-time comparison, so it is
    asserted against the trajectory's own count rather than calculated. The
    check is written so it CAN fail: it compares an independently computed
    expectation against a measured count, on a window large enough that the
    budget does not truncate the search.
    """
    from .harness import scenario_set
    from .tracking import trajectory_for

    ok = True
    for pop in (3, 5):
        iters = 10
        expected = pop * iters
        sc = scenario_set(4)[0]
        traj = trajectory_for(sc)
        # A window exactly the size of the search: any extra step would be the
        # hold phase, and any shortfall would be truncation.
        particle_swarm(traj, n_steps=expected, population=pop,
                       iterations=iters, seed=0)
        got = len(traj.v_hist)
        ok &= (got == expected)
        if verbose:
            print(f"   population {pop} x {iters} iterations: "
                  f"expected {expected} steps, measured {got}")
    if verbose:
        print(f"   -> {'PASS' if ok else 'FAIL'}")
    return ok


def test_swarm_covers_curve(verbose: bool = True) -> bool:
    """The initial swarm must sample more than one substring region.

    This is the property that distinguishes PSO from a hill-climber. A swarm
    initialised inside one region would trap exactly like P&O, and the arrival
    rate would then be a measure of the initialisation rather than of the
    algorithm. Written so it CAN fail -- a population of 3 spread over three
    regions is the minimum that passes.
    """
    from .harness import scenario_set
    from .tracking import trajectory_for

    sc = scenario_set(4)[0]
    traj = trajectory_for(sc)
    particle_swarm(traj, n_steps=DEFAULT_POPULATION, population=DEFAULT_POPULATION,
                   iterations=1, seed=0)
    width = traj.v_oc / N_SUB
    regions = {min(int(v / width), N_SUB - 1) for v in traj.v_hist}
    ok = len(regions) >= 2
    if verbose:
        print(f"   first iteration sampled regions {sorted(regions)} "
              f"of {N_SUB}")
        print(f"   -> {'PASS' if ok else 'FAIL'} (must sample >= 2 regions)")
    return ok


def test_stochastic(verbose: bool = True) -> bool:
    """Different seeds must give different trajectories.

    PSO is the first stochastic method in this project. If two seeds produced
    identical trajectories the randomness would not be reaching the algorithm,
    the multi-seed averaging would be measuring nothing, and the reported spread
    would be a false zero -- the defect pattern recorded at D5, D8 and D11.
    """
    from .harness import scenario_set
    from .tracking import trajectory_for

    sc = scenario_set(4)[0]
    hists = []
    for s in (0, 1):
        traj = trajectory_for(sc)
        particle_swarm(traj, n_steps=60, seed=s)
        hists.append(np.asarray(traj.v_hist))
    differ = not np.allclose(hists[0], hists[1])
    if verbose:
        print(f"   seed 0 and seed 1 trajectories differ: {differ}")
        print(f"   -> {'PASS' if differ else 'FAIL -- randomness is not '
                                            'reaching the algorithm'}")
    return differ


if __name__ == "__main__":
    print("PSO tracker: mechanical checks")
    try:
        print("  " + config.provenance())
    except Exception:
        pass
    print("=" * 74)
    print("cost mapping: SEQUENTIAL is primary (M particles = M control steps).")
    print("PARALLEL is reported alongside for comparability with the")
    print("PSO-MPPT literature, which counts iterations rather than samples.\n")
    print(f"coefficients held fixed: w={INERTIA}, c1={C_COGNITIVE}, "
          f"c2={C_SOCIAL}")
    print(f"population swept over {POPULATION_SWEEP}; {N_SEEDS} seeds per "
          f"configuration\n")

    print("1. COST IS COUNTED, NOT ASSERTED")
    a = test_cost_is_counted()

    print("\n2. THE SWARM COVERS MORE THAN ONE REGION")
    b = test_swarm_covers_curve()

    print("\n3. THE METHOD IS ACTUALLY STOCHASTIC")
    c = test_stochastic()

    print("\n" + "=" * 74)
    passed = sum([a, b, c])
    print(f"{passed}/3 mechanical checks passed")
    if passed == 3:
        print("\nPSO is ready to score. Run phase2/p9_pso_comparison.py for the")
        print("population sweep and the comparison against the other trackers.")
    else:
        print("\nDo NOT score PSO until every check passes. A swarm that does")
        print("not cover the curve, or that is not actually random, would")
        print("produce a plausible arrival rate that measures the wrong thing.")
    raise SystemExit(0 if passed == 3 else 1)