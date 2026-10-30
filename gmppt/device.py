"""
Device model (S3): multi-peak module I-V under partial shading.

A module = `n_substrings` cell groups in series, each with a bypass diode wired
anti-parallel. Each substring is a single-diode model (CEC parameters), extended
into reverse bias by the Bishop (1988) term via pvlib's bishop88, and shunted by
a bypass diode that conducts when the substring is driven negative.

Composition is done in the CURRENT domain:
  * for each substring, compute the terminal I-V of (substring || bypass diode)
    over a fine voltage grid: I_elem(V) = I_substring(V) + I_bypass(V);
  * I_elem(V) is strictly decreasing in V, so invert to V(I) on a common current
    grid shared by all substrings (series elements carry the same current);
  * module voltage V(I) = sum of substring voltages; power P = V * I.

The staircase / multiple peaks arise because as the string current rises past a
shaded substring's short-circuit current, its bypass diode turns on and clamps
that substring near -0.6 V, removing it from the voltage sum.

Reverse-bias (avalanche) parameters are an explicit, cited assumption (a named
limitation of the study). They default to OFF (breakdown_factor=0, i.e. ordinary
ohmic reverse conduction through R_sh) because the bypass diode clamps the
substring long before avalanche; they are exposed here so the sensitivity study
required by the plan can sweep them.
"""
from __future__ import annotations
import warnings
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from pvlib import pvsystem, singlediode

from . import config

# Boltzmann * T / q at a reference temperature, for the bypass-diode model
_Q = 1.602176634e-19
_K = 1.380649e-23


# --------------------------------------------------------------------------
# Parameter containers
# --------------------------------------------------------------------------
@dataclass
class ModuleParams:
    """CEC single-diode reference parameters for a whole module."""
    name: str
    N_s: int
    alpha_sc: float
    a_ref: float
    I_L_ref: float
    I_o_ref: float
    R_sh_ref: float
    R_s: float
    Adjust: float

    @classmethod
    def from_cec(cls, name: str) -> "ModuleParams":
        full = pvsystem.retrieve_sam("CECMod").T
        r = full.loc[name]
        num = lambda k: float(pd.to_numeric(r[k]))
        return cls(name=name, N_s=int(num("N_s")), alpha_sc=num("alpha_sc"),
                   a_ref=num("a_ref"), I_L_ref=num("I_L_ref"),
                   I_o_ref=num("I_o_ref"), R_sh_ref=num("R_sh_ref"),
                   R_s=num("R_s"), Adjust=num("Adjust"))


@dataclass
class Breakdown:
    """Reverse-bias avalanche parameters (Bishop term). Default: OFF."""
    factor: float = 0.0          # breakdown_factor (0 => no avalanche)
    voltage: float = -15.0       # breakdown_voltage per substring (V)
    exp: float = 3.28            # breakdown_exp


@dataclass
class Bypass:
    """Anti-parallel bypass diode across each substring."""
    I0: float = 1e-9             # saturation current (A) -> ~0.6 V clamp at ~8 A
    n: float = 1.0               # ideality
    temp_c: float = 25.0         # diode temperature (deg C)

    def current(self, v: np.ndarray) -> np.ndarray:
        """Bypass current vs substring voltage. Conducts (positive) for v<0."""
        vt = self.n * _K * (self.temp_c + 273.15) / _Q
        i = self.I0 * (np.exp(np.clip(-v / vt, -50, 50)) - 1.0)
        return np.maximum(i, 0.0)

    @property
    def clamp_voltage(self) -> float:
        """Approx forward drop (negative V) when the diode conducts a bypassed
        module's current. Used at string level, where a shaded module carrying no
        useful current is clamped to roughly this voltage by its bypass diode."""
        vt = self.n * _K * (self.temp_c + 273.15) / _Q
        return -float(vt * np.log(8.0 / self.I0))  # ~-0.6 V at ~8 A, 25 C


# --------------------------------------------------------------------------
# Substring scaling and I-V
# --------------------------------------------------------------------------
def scale_to_substring(mp: ModuleParams, cell_fraction: float) -> dict:
    """Scale module CEC params to a substring holding `cell_fraction` of cells.

    Series-count quantities scale with the number of series cells; current
    quantities do not.
    """
    return dict(
        alpha_sc=mp.alpha_sc,          # A/degC  (current-side, unchanged)
        a_ref=mp.a_ref * cell_fraction,        # n*Ns*k*T/q  (~ Ns)
        I_L_ref=mp.I_L_ref,            # A
        I_o_ref=mp.I_o_ref,            # A
        R_sh_ref=mp.R_sh_ref * cell_fraction,  # ohm (series cells)
        R_s=mp.R_s * cell_fraction,            # ohm (series cells)
        Adjust=mp.Adjust,
    )


def substring_operating_params(sub: dict, G: float, T: float):
    """(IL, I0, Rs, Rsh, nNsVth) for a substring at irradiance G, temp T."""
    return pvsystem.calcparams_cec(
        effective_irradiance=G, temp_cell=T,
        alpha_sc=sub["alpha_sc"], a_ref=sub["a_ref"], I_L_ref=sub["I_L_ref"],
        I_o_ref=sub["I_o_ref"], R_sh_ref=sub["R_sh_ref"], R_s=sub["R_s"],
        Adjust=sub["Adjust"], EgRef=config.EG_REF, dEgdT=config.DEG_DT)


def substring_element_iv(sub, G, T, bd: Breakdown, bp: Bypass, n_v=4000):
    """Terminal I-V of (substring || bypass) over a fine voltage grid.

    Returns (v, i_elem) with v ascending and i_elem strictly decreasing.

    The grid floor is adaptive: it reaches past the avalanche knee
    (bd.voltage, typically -10 to -30 V) whenever the avalanche term is active,
    so the breakdown region is actually representable. A fixed -2.0 V floor
    silently clips the knee off the grid and makes any breakdown-voltage sweep
    vacuous - the term is never sampled, so the GMPP looks invariant to it by
    construction rather than by physics. With avalanche off (bd.factor == 0) the
    bypass diode clamps near -0.6 V and a shallow floor is sufficient.
    """
    IL, I0, Rs, Rsh, nNsVth = substring_operating_params(sub, G, T)
    # forward open-circuit voltage sets the upper end of the grid
    fwd = pvsystem.singlediode(IL, I0, Rs, Rsh, nNsVth)
    v_oc = float(fwd["v_oc"])
    # Reach the breakdown knee when avalanche is on, but stop just ABOVE
    # bd.voltage: the Bishop term (1 - V/V_br)^-exp is only defined for
    # V > V_br, and below the knee the substring's bypass diode has long since
    # clamped it out of the operating path anyway. Sampling to ~95% of the knee
    # makes the onset representable without evaluating the term where it is
    # numerically undefined. With avalanche off, a shallow floor suffices
    # (bypass clamps near -0.6 V).
    v_floor = min(-2.0, 0.95 * bd.voltage) if bd.factor > 0 else -2.0
    v = np.linspace(v_floor, v_oc * 1.02, n_v)
    # The Bishop solver may not converge at the handful of samples right at the
    # avalanche knee (a stiff region far below the operating path). Those samples
    # sit in deep reverse bias where the substring is bypass-clamped and never
    # carries operating current, so their exact value is immaterial to the module
    # curve; suppress the benign convergence warning locally rather than letting
    # it mask a real one elsewhere. Verified: the operating region (V>=0) is
    # always finite (asserted downstream in module_iv).
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="some failed to converge", category=RuntimeWarning)
        warnings.filterwarnings(
            "ignore", message="invalid value encountered in power",
            category=RuntimeWarning)
        i_sub = singlediode.bishop88_i_from_v(
            v, IL, I0, Rs, Rsh, nNsVth,
            breakdown_factor=bd.factor, breakdown_voltage=bd.voltage,
            breakdown_exp=bd.exp)
    i_elem = np.asarray(i_sub) + bp.current(v)
    # Operating-path guarantee: currents at and above V=0 must be finite. If the
    # knee's non-convergence ever leaked upward into the operating region this
    # would catch it rather than letting a NaN propagate into the module curve.
    if not np.isfinite(i_elem[v >= 0.0]).all():
        raise FloatingPointError(
            "non-finite substring current in the operating region (V>=0); "
            "breakdown solver instability reached the operating path.")
    return v, i_elem


def substring_element_iv_subshaded(sub, cell_group_irr, T, bd: Breakdown,
                                   bp: Bypass, n_v=4000):
    """Terminal I-V of a substring whose CELLS are non-uniformly shaded.

    `cell_group_irr` is a sequence of irradiances (W/m^2), one per equal-sized
    cell group WITHIN this one substring. All groups are in series; there is ONE
    bypass diode across the whole substring, none per cell group.

    This is the sub-substring geometry S3 could not represent and that Gate A(i)
    tests. The physics that emerges here and nowhere else: when one cell group is
    shaded far below its neighbours, the brighter groups drive the shaded group
    into DEEP reverse bias (there is no bypass to protect it), and the reverse-
    bias / avalanche parameters finally govern where the substring's steps fall.
    The bypass across the whole substring still clamps the substring TERMINAL at
    ~-0.6 V, but internally a single cell can sit at -10..-20 V.

    Composition: each cell group is a fraction (1/n_groups) of the substring;
    compose the groups in series in the current domain (sum voltages at shared
    current), then place the substring bypass across the summed terminal.
    """
    irr = np.asarray(cell_group_irr, dtype=float)
    n_groups = len(irr)
    if n_groups == 1:
        # uniform substring -> identical to the standard path
        return substring_element_iv(sub, float(irr[0]), T, bd, bp, n_v=n_v)

    # each cell group is 1/n_groups of this substring's series cells
    group = dict(sub)
    group["a_ref"] = sub["a_ref"] / n_groups
    group["R_sh_ref"] = sub["R_sh_ref"] / n_groups
    group["R_s"] = sub["R_s"] / n_groups

    # per-group I-V WITHOUT bypass (no per-cell-group bypass diode); reverse bias
    # via Bishop. Build each on its own voltage grid, then invert to V(I) on a
    # shared current grid and sum.
    curves, isc = [], []
    for G in irr:
        IL, I0, Rs, Rsh, nNsVth = substring_operating_params(group, G, T)
        v_oc = float(pvsystem.singlediode(IL, I0, Rs, Rsh, nNsVth)["v_oc"])
        v_floor = min(-2.0, 0.95 * bd.voltage) if bd.factor > 0 else -2.0
        v = np.linspace(v_floor, v_oc * 1.02, n_v)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="some failed to converge",
                                    category=RuntimeWarning)
            warnings.filterwarnings("ignore",
                                    message="invalid value encountered in power",
                                    category=RuntimeWarning)
            i_grp = np.asarray(singlediode.bishop88_i_from_v(
                v, IL, I0, Rs, Rsh, nNsVth,
                breakdown_factor=bd.factor, breakdown_voltage=bd.voltage,
                breakdown_exp=bd.exp))
        curves.append((v, i_grp))
        isc.append(float(np.interp(0.0, v, i_grp)))
    isc = np.array(isc)

    # shared current grid 0..i_max; sum group voltages at each current (series).
    i_max = float(isc.max())
    I = np.linspace(0.0, i_max, n_v)
    Vsub = np.zeros_like(I)
    for (v, i_grp) in curves:
        i_asc = i_grp[::-1]
        v_asc = v[::-1]
        # a shaded group with low Isc cannot carry I>its own reverse-bias max;
        # np.interp clamps to the deepest reverse-bias voltage, which is the
        # physically correct "driven to its breakdown floor" behaviour here.
        Vsub += np.interp(I, i_asc, v_asc)

    # place the substring bypass across the summed terminal: where Vsub goes
    # negative past the clamp, the bypass conducts and holds the terminal.
    clamp = bp.clamp_voltage
    Vsub = np.maximum(Vsub, clamp)

    # return in the same (v ascending, i descending) convention as the standard
    # path: here we have (I ascending, Vsub). Convert to v-ascending form.
    order = np.argsort(Vsub)
    v_out = Vsub[order]
    i_out = I[order]
    # deduplicate on v for a clean monotone curve
    v_uni, idx = np.unique(v_out, return_index=True)
    return v_uni, i_out[idx]


# --------------------------------------------------------------------------
# Module I-V
# --------------------------------------------------------------------------
def module_iv(mp: ModuleParams, irradiances, T,
              n_substrings=config.N_SUBSTRINGS,
              bd: Breakdown | None = None, bp: Bypass | None = None,
              n_points=4000):
    """Compose the module I-V from per-substring irradiances.

    `irradiances` is a length-`n_substrings` sequence. Each entry is either:
      * a scalar W/m^2  -> that substring is uniformly lit (the original path); or
      * a sequence of W/m^2 -> that substring is SUB-SUBSTRING shaded, one value
        per equal cell group within it (the Gate A(i) geometry).
    Mixing is allowed: e.g. [1000, 1000, [1000, 1000, 200]] shades one cell group
    of the third substring only. Returns a dict with arrays I, V, P and the
    per-substring short-circuit currents.
    """
    bd = bd or Breakdown()
    bp = bp or Bypass(temp_c=T)
    assert len(irradiances) == n_substrings

    # split cells across substrings (handle non-divisible counts)
    base = mp.N_s // n_substrings
    counts = [base] * n_substrings
    for k in range(mp.N_s - base * n_substrings):
        counts[k] += 1
    fracs = [c / mp.N_s for c in counts]

    # per-substring element I-V and short-circuit current; route uniform vs
    # sub-substring per entry
    elems, isc = [], []
    for entry, frac in zip(irradiances, fracs):
        sub = scale_to_substring(mp, frac)
        if np.isscalar(entry):
            v, i_elem = substring_element_iv(sub, float(entry), T, bd, bp,
                                             n_v=n_points)
        else:
            v, i_elem = substring_element_iv_subshaded(sub, entry, T, bd, bp,
                                                       n_v=n_points)
        elems.append((v, i_elem))
        isc.append(float(np.interp(0.0, v, i_elem)))  # v ascending -> i at v=0
    isc = np.array(isc)

    # common current grid up to the brightest substring's Isc (inclusive, so the
    # V=0 point is representable for consistency checks)
    i_max = isc.max()
    I = np.linspace(0.0, i_max, n_points)

    # invert each substring's element I-V to V(I) and sum
    #
    # np.interp CLAMPS to the endpoint outside [i_asc.min, i_asc.max] rather
    # than raising. For a uniform substring the element curve runs deep into
    # reverse bias and covers the whole current range, so clamping never bites.
    # For a SUB-SUBSTRING-shaded substring whose terminal is bypass-clamped, the
    # curve stops near the bypass voltage and does NOT reach the brightest
    # substring's Isc: physically, when the module current exceeds this
    # substring's own Isc, its bypass diode conducts and holds the terminal at
    # the clamp voltage. So for currents beyond a substring's range we substitute
    # the bypass clamp rather than raising or silently clamping to a wrong value.
    clamp = bp.clamp_voltage
    V = np.zeros_like(I)
    for (v, i_elem) in elems:
        i_asc = i_elem[::-1]
        v_asc = v[::-1]
        Vsub = np.interp(I, i_asc, v_asc)
        over = I > i_asc.max() + 1e-9          # current beyond this substring's Isc
        if over.any():
            Vsub = np.where(over, clamp, Vsub)  # bypass conducts, terminal clamped
        V += Vsub

    P = V * I
    return dict(I=I, V=V, P=P, isc_substrings=isc, cell_counts=counts)


# --------------------------------------------------------------------------
# Curve analysis
# --------------------------------------------------------------------------
def analyse(curve: dict, prominence_frac=0.01, min_separation_v=1.0):
    """Global peak and local maxima of a module P-V/P-I curve.

    A raw index-comparison local-maximum test inflates the peak count under two
    conditions the study deliberately probes: numerical jitter on the dense grid,
    and near-threshold irradiance ratios where the curve flattens between shelves
    (a single physical shelf then reads as several adjacent maxima). Two criteria
    suppress both without discarding genuine peaks:

      * PROMINENCE. A candidate must rise at least `prominence_frac` of the
        global-peak power above the higher of the two troughs flanking it (the
        deepest point between it and the nearest higher peak on each side).
      * SEPARATION. Peaks within `min_separation_v` volts are merged, keeping the
        more powerful; distinct bypass shelves are ~10 V apart, so 1 V is safely
        below the physical spacing while removing grid-adjacent duplicates.

    Defaults (1% prominence, 1 V separation) leave the clean three-shelf demo
    untouched and only bite on flattened / jittery curves. Both are exposed so
    S7 can tune them against scenarios whose true peak count is known.
    """
    I, V, P = curve["I"], curve["V"], curve["P"]
    # restrict to the physical operating region (V >= 0)
    mask = V >= 0
    Im, Vm, Pm = I[mask], V[mask], P[mask]
    g = int(np.argmax(Pm))
    p_thresh = prominence_frac * float(Pm[g])

    # raw local maxima (index comparison)
    raw = []
    for j in range(1, len(Pm) - 1):
        if Pm[j] >= Pm[j - 1] and Pm[j] > Pm[j + 1]:
            raw.append(j)

    # prominence: rise above the deeper flanking trough out to the nearest
    # strictly-higher sample on each side (bounded local prominence).
    def prominence(j):
        pj = Pm[j]
        li = j
        while li > 0 and Pm[li - 1] <= pj:
            li -= 1
        ri = j
        while ri < len(Pm) - 1 and Pm[ri + 1] <= pj:
            ri += 1
        left_trough = Pm[li:j + 1].min() if j > li else pj
        right_trough = Pm[j:ri + 1].min() if ri > j else pj
        return pj - max(left_trough, right_trough)

    kept = [j for j in raw if prominence(j) >= p_thresh or j == g]

    # separation merge: within min_separation_v volts, keep the stronger.
    kept.sort(key=lambda j: Vm[j])
    merged = []
    for j in kept:
        if merged and abs(Vm[j] - Vm[merged[-1]]) < min_separation_v:
            if Pm[j] > Pm[merged[-1]]:
                merged[-1] = j
        else:
            merged.append(j)

    peaks = [(float(Vm[j]), float(Im[j]), float(Pm[j])) for j in merged]
    return dict(
        gmpp=dict(V=float(Vm[g]), I=float(Im[g]), P=float(Pm[g])),
        n_peaks=len(peaks),
        peaks=sorted(peaks, key=lambda t: -t[2]),
    )

# --------------------------------------------------------------------------
# Array generalisation (S5): modules in series -> a string
# --------------------------------------------------------------------------
def string_iv(mp: ModuleParams, module_irradiances, T,
              n_substrings=config.N_SUBSTRINGS,
              bd: Breakdown | None = None, bp: Bypass | None = None,
              n_points=4000):
    """Compose a string I-V from several modules in series.

    A module is substrings-in-series composed in the current domain (module_iv).
    A string is modules-in-series composed the SAME way, one level up: the string
    carries one current; each module's voltage at that current is summed. This is
    the array-capable generalisation (S5) — the substring-composition code S3
    wrote, reused unchanged at the module level.

    `module_irradiances` is a sequence of per-module irradiance patterns, each a
    length-`n_substrings` sequence. `[[1000,1000,1000]]` is the single-module
    N=1 special case and must reproduce `module_iv` exactly.

    Returns a dict with arrays I, V, P and the per-module short-circuit currents.
    """
    bd = bd or Breakdown()
    bp = bp or Bypass(temp_c=T)
    patterns = [np.asarray(p, dtype=float) for p in module_irradiances]
    assert all(len(p) == n_substrings for p in patterns), \
        "each module needs one irradiance per substring"

    # per-module V(I) curves and their short-circuit currents
    curves, isc = [], []
    for irr in patterns:
        c = module_iv(mp, irr, T, n_substrings=n_substrings, bd=bd, bp=bp,
                      n_points=n_points)
        curves.append(c)
        isc.append(float(c["isc_substrings"].max()))  # module Isc = brightest sub
    isc = np.array(isc)

    # common current grid up to the strongest module's Isc
    I = np.linspace(0.0, isc.max(), n_points)

    # Sum each module's voltage at the shared current (series composition), with a
    # MODULE-LEVEL bypass diode across each module. In a real string, when a
    # shaded module cannot carry the string current (I > its Isc), its bypass
    # diode conducts and clamps that module to ~-0.6 V rather than forcing it deep
    # into reverse bias. Without this, a dim module in series with a bright one is
    # unrepresentable (its curve does not reach the string current) - which is
    # exactly why real strings have module bypass diodes. This is the same
    # bypass mechanism S3 applied to substrings, applied one level up.
    V = np.zeros_like(I)
    for c in curves:
        i_asc = c["I"]                # module current 0..Isc (ascending)
        v_of_i = c["V"]               # module voltage Voc..0 (descending)
        i_mod_max = float(i_asc.max())
        Vmod = np.interp(I, i_asc, v_of_i)   # clamps above Isc, fixed next line
        # where string current exceeds this module's Isc, the module is bypassed
        bypassed = I > i_mod_max
        Vmod = np.where(bypassed, bp.clamp_voltage, Vmod)
        V += Vmod

    P = V * I
    return dict(I=I, V=V, P=P, isc_modules=isc, n_modules=len(patterns))


# --------------------------------------------------------------------------
# Array composition: strings in parallel  (additive; dashboard Scenario page)
# --------------------------------------------------------------------------
def array_iv(mp: ModuleParams, string_irradiances, T,
             n_substrings=config.N_SUBSTRINGS,
             bd: Breakdown | None = None, bp: Bypass | None = None,
             blocking_diodes=True, n_points=4000):
    """Compose an array I-V curve from strings connected in parallel.

    `string_irradiances` is one `module_irradiances` collection per parallel
    string, each exactly what `string_iv` accepts. Every string is composed by
    `string_iv` (modules in series, module-level bypass) and nothing about that
    composition is changed here; this function only adds the one operation a
    parallel connection needs.

    Strings in parallel share one terminal voltage, so the composition is done
    in the VOLTAGE domain: a common ascending voltage grid from 0 to the
    largest string V_oc, each string's current interpolated onto it, and the
    currents summed. Each string's curve is known only where its current is
    non-negative (V <= its own V_oc); `string_iv` never computes the forward-
    conduction branch above V_oc.

    blocking_diodes=True (the protected case): a blocking diode in series with
    each string stops it from sinking current from the others, so a string's
    contribution is clipped at zero — above its own V_oc it contributes
    nothing, and a weak string can never subtract from a healthier one.

    blocking_diodes=False is the explicitly unprotected comparison case, in
    which a weak string may carry reverse current. That needs each string's
    I(V) for V > V_oc, which the validated engine does not model, so it is
    refused rather than invented: NotImplementedError.

    A single-string array returns that string's own curve (sorted by ascending
    voltage), so it reproduces `string_iv` exactly.

    Returns a dict with arrays I, V, P (V ascending) plus per-string V_oc and
    short-circuit currents — the shape `analyse` expects.
    """
    bd = bd or Breakdown()
    bp = bp or Bypass(temp_c=T)
    if not blocking_diodes:
        raise NotImplementedError(
            "array_iv(blocking_diodes=False): reverse current through an unprotected "
            "string requires each string's I(V) above its V_oc (forward conduction), "
            "which the validated engine does not compute. Only the blocking-diode "
            "case is available.")
    strings = [string_iv(mp, s, T, n_substrings=n_substrings, bd=bd, bp=bp,
                         n_points=n_points) for s in string_irradiances]
    assert strings, "an array needs at least one string"

    vocs, iscs, sorted_curves = [], [], []
    for s in strings:
        o = np.argsort(s["V"])
        v_asc, i_asc = s["V"][o], s["I"][o]
        sorted_curves.append((v_asc, i_asc))
        vocs.append(float(s["V"].max()))          # V at I = 0, the first grid point
        iscs.append(float(np.interp(0.0, v_asc, i_asc)))

    if len(strings) == 1:                          # exact, not interpolated
        v_asc, i_asc = sorted_curves[0]
        return dict(I=i_asc, V=v_asc, P=v_asc * i_asc, voc_strings=vocs,
                    isc_strings=iscs, n_strings=1, blocking_diodes=True)

    V = np.linspace(0.0, max(vocs), n_points)
    I = np.zeros_like(V)
    for v_asc, i_asc in sorted_curves:
        # right=0.0: above this string's V_oc its blocking diode holds it at
        # zero; the clip covers any sign the interpolation leaves at the edge.
        I += np.maximum(np.interp(V, v_asc, i_asc, right=0.0), 0.0)
    P = V * I
    return dict(I=I, V=V, P=P, voc_strings=vocs, isc_strings=iscs,
                n_strings=len(strings), blocking_diodes=True)