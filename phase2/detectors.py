"""Re-seed policies for the dynamic relocation comparison (E5).

One interface, five policies, so the p12 runner can hold everything else fixed
and vary only *when the seed re-fires*. That is the whole point: the A3 claim is
about re-prediction being cheap, and it can only be isolated if the trigger is a
swappable component rather than something welded into the tracker loop.

    det = PowerDropDetector(eps=0.05, persist=4)
    det.reset(p0=first_reading)
    for v, i, p in stream:              # p is the MEASURED (noisy) power
        if det.update(v, i, p):
            ... re-fire the seed ...

`update` returns True on the step the policy fires, and the policy re-arms
itself, so a caller never has to manage its state.

PROVENANCE OF EACH POLICY
-------------------------
PowerDropDetector
    The |ΔP|/P restart criterion. Originates with Miyatake et al. (2011),
    IEEE TAES 47(1), 367-380, who introduced re-initialisation for PSO-based
    MPPT. The canonical written form is Ahmed & Salam (2014), Applied Energy
    119, 118-130, eq. (15):

        |P(s^{i+1}) - P(s^i)| / P(s^i) > dP ,   with dP = 0.1

    Published thresholds span roughly 0.05-0.20 (see EPS_PUBLISHED below).
    Our addition is the EMA baseline and the persistence requirement, which
    exist because a bare rolling-max comparison false-fired on ~86% of windows
    once measurement noise was introduced (p12 rev 4 finding).

SignRunDetector
    Al Abri, Al Abri, Yousef & Al-Hinai (2021), Energies 14(16), 4938.
    Flags a shading change when N consecutive voltage changes share a sign,
    with a small voltage dead-band rejecting noise. A different detector class
    from power-drop: it keys on directed motion rather than on magnitude.
    !! VERIFY the two constants against the paper before reporting. The values
       below (N=6, dead-band 0.5 V) are as reported to us second-hand.

TimerLatchDetector
    Zaki, Shahin, Eskender & Elsayes (2023), Scientific Reports 13, 22197.
    A power threshold arms a fixed timer, and the timer forces the reset
    without the threshold having to be re-met. Exists to stop a slowly
    sweeping shadow edge from chattering the trigger on and off.
    !! VERIFY the 5-10% threshold band and the timer length against the paper.

ScheduledReseed
    The predefined-time (PDT) dispersal of Eltamaly, Farh & Al Saud (2019),
    Sustainability 11(7), 2091, and Eltamaly & Farh (2019), Solar Energy 177,
    306-316, who re-initialise every T_PD seconds with no detection at all.
    This is the arm our comparison was missing: without it we show that
    detection beats nothing, which is a far weaker statement than detection
    beats a clock.

NeverReseed
    The null arm. Fires once at step 0 and never again.

All thresholds are module-level constants so a sweep can import and vary them
rather than editing the file. Nothing here is tuned to make our method win; the
defaults are the published values, and where we depart (persistence) the reason
is a measured failure, stated above.
"""

from __future__ import annotations

# --- declared constants ------------------------------------------------------

# The published range for the dP/P restart threshold, for the epsilon sweep.
# Each entry is (value, source as cited).
EPS_PUBLISHED = [
    (0.05, "Eltamaly 2021, Energies 14(4) 953; Al-Ramaden & Smadi, JEEIT 2019"),
    (0.10, "Ahmed & Salam 2014, Appl. Energy 119, eq. 15; Wang, Li & Ruan"),
    (0.15, "Kim & Kim 2013, J. Int. Conf. Electr. Mach. Syst. 2, 65-68"),
    (0.20, "Rizzo & Scelba 2015, Appl. Energy 145 (upper end of 10-20%)"),
]

EPS_DEFAULT = 0.05          # our frozen value; the low end of the published range
PERSIST_DEFAULT = 4         # consecutive steps the drop must hold
EMA_ALPHA_DEFAULT = 0.10    # baseline smoothing; ~10-step memory
REARM_STEPS = 10            # dead time after a fire, so one event fires once

SIGN_RUN_N = 6              # Al Abri et al. 2021  -- VERIFY
SIGN_RUN_DEADBAND_V = 0.5   # volts; rejects dither  -- VERIFY

TIMER_EPS = 0.05            # Zaki et al. 2023 band is 5-10%  -- VERIFY
TIMER_STEPS = 20            # forced-reset delay after arming  -- VERIFY


class ReseedPolicy:
    """Common interface. Subclasses implement _step and may extend reset."""

    name = "base"

    def __init__(self):
        self._cool = 0
        self.n_fires = 0
        self.fire_steps: list[int] = []
        self._k = -1

    def reset(self, p0: float = 0.0, v0: float = 0.0) -> None:
        self._cool = 0
        self.n_fires = 0
        self.fire_steps = []
        self._k = -1

    def update(self, v: float, i: float, p: float) -> bool:
        self._k += 1
        if self._cool > 0:
            self._cool -= 1
            self._step(v, i, p)          # keep internal state warm
            return False
        fired = bool(self._step(v, i, p))
        if fired:
            self._cool = REARM_STEPS
            self.n_fires += 1
            self.fire_steps.append(self._k)
        return fired

    def _step(self, v: float, i: float, p: float) -> bool:
        raise NotImplementedError

    def describe(self) -> str:
        return self.name


class NeverReseed(ReseedPolicy):
    """Seed once at the start, then hold whatever happens."""

    name = "never"

    def _step(self, v, i, p):
        return False


class ScheduledReseed(ReseedPolicy):
    """Re-seed every `period` control steps, regardless of measurements.

    The predefined-time dispersal of Eltamaly et al. (2019). No detection, so
    it pays the re-seed cost on every period whether or not anything moved, and
    it cannot respond faster than one period.
    """

    name = "scheduled"

    def __init__(self, period: int = 240, phase: int = 0):
        super().__init__()
        # Eltamaly et al. (2019) disperse every 24 s. Set `period` from that in
        # control steps once the real loop rate is fixed -- at 0.1 s control
        # that is 240 steps, which is where the default comes from. It is their
        # number, not ours, and should be cited as such.
        self.period = int(period)
        # A fixed phase would make this arm's score depend on whether its clock
        # happens to land near the event -- luck, not merit, and unfair to a
        # published method in both directions. Callers vary `phase` per run and
        # report the mean, so the schedule is scored over all its alignments.
        self.phase = int(phase) % max(int(period), 1)

    def _step(self, v, i, p):
        return self._k > 0 and (self._k - self.phase) % self.period == 0

    def describe(self):
        return f"scheduled(period={self.period})"


class PowerDropDetector(ReseedPolicy):
    """|dP|/P against an EMA baseline, sustained for `persist` steps.

    Rationale for the two departures from the bare published rule:

    baseline
        The published form compares consecutive samples. Under ~1% RMS current
        noise that difference is dominated by noise, so we compare against an
        exponential moving average of recent power, which is the quantity the
        rule is morally about ("has the operating point got worse than it was").

    persistence
        A relocation is sustained: the peak has physically moved and the old
        operating point stays bad. A noise dip is transient. Requiring the drop
        to hold for `persist` consecutive steps separates them on that physical
        difference rather than by raising the threshold, which would have cost
        sensitivity to genuine small relocations.

    The baseline freezes while a candidate drop is in progress, so the EMA
    cannot chase the drop down and disarm the detector before it confirms.
    """

    name = "power_drop"

    def __init__(self, eps: float = EPS_DEFAULT,
                 persist: int = PERSIST_DEFAULT,
                 alpha: float = EMA_ALPHA_DEFAULT):
        super().__init__()
        self.eps, self.persist, self.alpha = float(eps), int(persist), float(alpha)

    def reset(self, p0=0.0, v0=0.0):
        super().reset(p0, v0)
        self._ema = float(p0)
        self._run = 0

    def _step(self, v, i, p):
        if self._ema <= 0:
            self._ema = max(p, 1e-9)
            return False
        drop = (self._ema - p) / self._ema
        if drop > self.eps:
            self._run += 1
            # baseline frozen while a candidate drop is open
            if self._run >= self.persist:
                self._run = 0
                self._ema = max(p, 1e-9)     # re-baseline at the new level
                return True
            return False
        self._run = 0
        self._ema = (1 - self.alpha) * self._ema + self.alpha * p
        return False

    def describe(self):
        return f"power_drop(eps={self.eps:g}, persist={self.persist}, alpha={self.alpha:g})"


class RatchetDropDetector(ReseedPolicy):
    """|dP|/P against a ratcheting baseline, sustained for `persist` steps.

    Motivated by a measured failure of the EMA form: on a SMOOTH relocation the
    EMA baseline sinks with the signal, so the ratio never crosses eps and the
    detector never fires. That is the failure Wellawatta & Choi (2019) predict
    for a fixed threshold on smooth transitions, and the bench reproduces it.

    The fix is to stop the baseline following the decay. The baseline here is
    the best power seen since the last fire, optionally bled off at `decay` per
    step so a genuinely lower operating level is eventually accepted. Noise
    robustness comes from `persist` rather than from a sinking reference, which
    is what the EMA was doing the job of.

    Known and accepted consequence: a ratcheting baseline cannot tell a
    relocation from a GLOBAL dimming -- in both the held power falls away from
    the best recently seen. It will therefore fire on an irradiance ramp where
    nothing has relocated. We accept that rather than engineer it away, because
    a re-seed costs ~6 probes: firing when nothing moved costs a few
    measurements, while failing to fire when the peak moved costs most of the
    array's output. The asymmetry is the whole argument for a cheap seed, and
    it is not available to a method whose re-search costs a full scan.
    """

    name = "ratchet_drop"

    def __init__(self, eps: float = EPS_DEFAULT,
                 persist: int = PERSIST_DEFAULT,
                 decay: float = 0.0):
        super().__init__()
        self.eps, self.persist, self.decay = float(eps), int(persist), float(decay)

    def reset(self, p0=0.0, v0=0.0):
        super().reset(p0, v0)
        self._best = float(p0)
        self._run = 0

    def _step(self, v, i, p):
        if self._best <= 0:
            self._best = max(p, 1e-9)
            return False
        if p > self._best:
            self._best = p
            self._run = 0
            return False
        self._best *= (1.0 - self.decay)
        if (self._best - p) / self._best > self.eps:
            self._run += 1
            if self._run >= self.persist:
                self._run = 0
                self._best = max(p, 1e-9)
                return True
        else:
            self._run = 0
        return False

    def describe(self):
        return (f"ratchet_drop(eps={self.eps:g}, persist={self.persist}, "
                f"decay={self.decay:g})")


class SignRunDetector(ReseedPolicy):
    """N consecutive voltage changes of the same sign (Al Abri et al. 2021).

    Changes smaller than `deadband` in magnitude are treated as no movement and
    break nothing -- they are skipped, not counted as a sign flip, so that a
    quiet patch does not silently reset a genuine run.

    Note what this detector keys on: P&O at steady state dithers up and down, so
    signs alternate and no run accumulates. A relocation drags the operating
    point consistently in one direction, which is a run. It therefore sees the
    *motion* rather than the *loss*, and should behave differently from the
    power rule on a slow sweep -- which is exactly why it is worth running.
    """

    name = "sign_run"

    def __init__(self, n: int = SIGN_RUN_N, deadband: float = SIGN_RUN_DEADBAND_V):
        super().__init__()
        self.n, self.deadband = int(n), float(deadband)

    def reset(self, p0=0.0, v0=0.0):
        super().reset(p0, v0)
        self._vprev = float(v0)
        self._sign = 0
        self._run = 0

    def _step(self, v, i, p):
        dv = v - self._vprev
        if abs(dv) < self.deadband:
            return False                     # no movement; leave the run intact
        self._vprev = v
        s = 1 if dv > 0 else -1
        if s == self._sign:
            self._run += 1
        else:
            self._sign, self._run = s, 1
        if self._run >= self.n:
            self._run, self._sign = 0, 0
            return True
        return False

    def describe(self):
        return f"sign_run(n={self.n}, deadband={self.deadband:g}V)"


class TimerLatchDetector(ReseedPolicy):
    """Threshold arms a timer; the timer fires regardless (Zaki et al. 2023).

    Once the power drop exceeds `eps` the detector latches and counts down
    `timer` steps, then fires whether or not the drop is still present. The
    point is a slowly developing change: with a pure threshold the drop can
    cross, recover below the line as the tracker partially adapts, and never
    fire, leaving the tracker on a local peak. The latch removes that failure
    mode at the cost of firing on transients the power rule would have
    dismissed -- which is the trade this ablation measures.
    """

    name = "timer_latch"

    def __init__(self, eps: float = TIMER_EPS, timer: int = TIMER_STEPS,
                 alpha: float = EMA_ALPHA_DEFAULT):
        super().__init__()
        self.eps, self.timer, self.alpha = float(eps), int(timer), float(alpha)

    def reset(self, p0=0.0, v0=0.0):
        super().reset(p0, v0)
        self._ema = float(p0)
        self._latched = -1

    def _step(self, v, i, p):
        if self._latched >= 0:
            self._latched -= 1
            if self._latched < 0:
                self._ema = max(p, 1e-9)
                return True
            return False
        if self._ema <= 0:
            self._ema = max(p, 1e-9)
            return False
        if (self._ema - p) / self._ema > self.eps:
            self._latched = self.timer
            return False
        self._ema = (1 - self.alpha) * self._ema + self.alpha * p
        return False

    def describe(self):
        return f"timer_latch(eps={self.eps:g}, timer={self.timer})"


# --- the arms the E5 comparison runs ----------------------------------------

def default_arms(scheduled_period: int = 240) -> list[ReseedPolicy]:
    """The five re-seed policies, in the order they should appear in a table."""
    return [
        NeverReseed(),
        ScheduledReseed(period=scheduled_period),
        PowerDropDetector(),
        RatchetDropDetector(),
        SignRunDetector(),
        TimerLatchDetector(),
    ]


def eps_sweep_arms() -> list[ReseedPolicy]:
    """The power-drop detector at each published threshold (E5c)."""
    return [PowerDropDetector(eps=e) for e, _ in EPS_PUBLISHED]