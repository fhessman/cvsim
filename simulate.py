#!/usr/bin/env python
# simulate.py
"""
Command-line driver: build a system from a YAML config (optionally
overridden by per-parameter CLI flags) and produce any combination of
the outline plot, temperature/intensity renderings, and eclipse light
curve -- either displayed interactively (--show) or saved to --outdir.

Every phase-dependent output shares one phase spec: --phase-list's own
explicit phases, in the given order, if given; else --phase-min alone
(--phase-max ignored) if --phase-num<=1 (the default: a single, harmless
phase, see --phase-num's own help), else linspace(--phase-min,
--phase-max, --phase-num). "outline"/"temperature"/"intensity" write one
output file per phase -- a sequence when --phase-num>1 -- each named
<prefix>_<kind>_<phase>.png, underscore-separated, where kind is "plot"
(outline), "render" (temperature image), or "intensity" (band_intensity
image -- the one to use for checking limb-darkening effects visually, see
--u_1/--u_2/--u_d); prefix is --prefix if given, else the --config file's
basename, else omitted -- same value (blanks intact) is also shown as
every output plot's title. "lightcurve", "magnitude", "shadow", and "rv"
instead always plot every phase together on one plot each
(<prefix>_<kind>.png, no phase suffix). "rv" is the system's radial-
velocity curve -- see render.radial_velocity_curve. "lightcurve" and
"magnitude" are two views of the same
underlying computation -- relative flux vs. AB magnitude -- and either
one (or both) triggers writing a single FITS binary table
(<prefix>_lightcurve.fits) with both flux and magnitude columns,
phase-resolved, plus every system/model parameter used recorded in the
primary HDU's header. Pass --data-file (a CSV or FITS table, with
--data-phase-col and exactly one of --data-flux-col/--data-mag-col plus
--data-err-col naming its columns) to overlay observed data points on the
light-curve and/or magnitude plot(s); --ph_off shifts the data's own
phase column to correct for a bad ephemeris in the data, independent of
the model. Likewise, pass any of --data_rv_1_col/--data_rv_2_col/
--data_rv_stream_col/--data_rv_hotspot_col (plus their own
--data_rv_*_err_col, --data-rv-phase-col, and --data-rv-file if the RV
data lives in a separate table from --data-file) to overlay observed
radial-velocity data on the "rv" output -- each names the column for one
of the four model curves (primary/secondary/stream/magnetically-channeled
"hotspot"), and any subset may be given at once; --data_rv_gamma adds a
single systemic-velocity offset to every model curve before comparing.
--lsq_fit/--mcmc_fit fit against --data-file and/or any active
--data_rv_*_col -- whichever are given -- jointly, with
'dist'/'data_norm'/'rv_gamma' available as extra fittable
pseudo-parameters alongside any SystemParams/ModelParams field name.

Examples
--------
    python simulate.py --config ZCha.yaml
    python simulate.py --config ZCha.yaml --outputs lightcurve --T_h 6000 --prefix zcha_hot
    python simulate.py --config ZCha.yaml --outputs temperature --show
    python simulate.py --P_orb 0.0745 --a 5.1237e8 --q 0.1488 --R_1 0.0133 \\
        --T_1 14700 --T_2 3000 --incl 81.8 \\
        --R_in 0.1567 --R_out 0.3134 --T_0 6000 --beta_d -0.75 --T_h 8000 --L_h 30

All SystemParams/ModelParams fields (see params.py) are available as
--<field_name> flags; anything given on the command line overrides the
same field from --config.
"""

import argparse
import dataclasses
import os
import re
import sys

import numpy as np

from params import (add_param_args, params_from_args, build_system, raw_angle_acc_from_config,
                     _SYSTEM_FIELDS, _MODEL_FIELDS)
from stream import (integrate_stream, closest_approach_index, angle_acc_index, sample_points,
                     disc_impact_point, impact_incidence, lubow_shu_eps,
                     lubow_shu_stream_size)

# SystemParams/ModelParams fields that render.physical_light_curve takes as
# live, per-call arguments even when reusing an already-built temp_maps --
# safe for _run_lsq_fit to vary without rebuilding lobe/disc/temp_maps.
# Anything else (T_2, disc geometry/temperature, R_2, q, ...) is baked into
# temp_maps at build time, so fitting it needs a full rebuild every trial
# (see _run_lsq_fit).
_CHEAP_SYSTEM_FIT_FIELDS = {"T_1", "R_1", "incl", "wavelength", "u_1", "u_2", "a", "ph_off"}
_CHEAP_MODEL_FIT_FIELDS = {"u_d"}

OUTPUT_CHOICES = ("outline", "temperature", "intensity", "lightcurve", "magnitude", "shadow",
                   "rim projection", "primary", "fit system", "rv")
OUTPUT_KIND = {"outline": "plot", "temperature": "render", "intensity": "intensity",
               "lightcurve": "lightcurve", "magnitude": "magnitude", "shadow": "shadow",
               "rim projection": "rim_projection", "primary": "primary",
               "fit system": "fit_system", "rv": "rv",
               "corner": "corner"}  # --corner_plot (--mcmc_fit) -- not an --outputs choice itself

IMAGE_DPI = 100  # fixed, so (width_px, height_px) maps to figsize unambiguously

# The four RV data streams --data_rv_<suffix>_col can independently supply
# (see add_argument loop below): CLI flag suffix -> (which of
# render.radial_velocity_curve's four returned curves it's compared
# against, the "rv" plot's own color for that curve -- same colors
# render_system_image/the "rv" output block already use elsewhere).
# "1"/"2" follow the standard K1/K2 (primary/secondary velocity
# semi-amplitude) notation from the spectroscopic-binary literature;
# "hotspot" names what render.py itself calls "magnetic" (the
# magnetically-channeled continuation of the stream, see
# build_temperature_maps' angle_acc) in the more observationally-familiar term
# for where that emission actually comes from.
RV_DATA_COMPONENTS = (
    ("1", "primary", "#2e6f95"),
    ("2", "secondary", "#d1272e"),
    ("stream", "stream", "#1b9e77"),
    ("hotspot", "magnetic", "purple"),
)


_RECORDED_AXES_METHODS = {
    "plot", "fill", "fill_between", "scatter", "axhline", "axvline",
    "text", "legend", "set_xlabel", "set_ylabel", "set_title",
    "set_xlim", "set_ylim", "set_aspect", "minorticks_on", "tick_params",
}


class _RecordingAxes:
    """
    Transparent proxy around a real matplotlib Axes, for --pyplot (see
    its own --help): forwards every attribute access to the real axes
    (so a plotting function using this in place of a real `ax` behaves
    identically, and still builds the real figure normally), but for
    the curated set of methods in _RECORDED_AXES_METHODS -- the ones
    that actually define a plot's content/appearance -- also records
    each call (name, args, kwargs) in order in self.calls, for
    write_pyplot_script to replay verbatim afterward into a standalone
    script. NOT a general-purpose Axes mock: only the methods this
    package's own plotting functions actually call are worth
    intercepting; anything else (e.g. reading back .figure, .transAxes)
    just passes through untouched.
    """
    def __init__(self, ax):
        self._ax = ax
        self.calls = []

    def __getattr__(self, name):
        attr = getattr(self._ax, name)
        if name not in _RECORDED_AXES_METHODS or not callable(attr):
            return attr

        def wrapper(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return attr(*args, **kwargs)
        return wrapper


def _pyplot_number(v):
    """
    Python source repr for one float, rounded to 8 significant digits
    (not full ~17-digit float repr -- plenty of precision for anything
    plotted, and keeps a multi-thousand-point array from bloating the
    script into an unreadable wall of digits), and safe for non-finite
    values: repr() of a bare nan/inf float is just the unquoted word
    itself, not valid Python on its own (no such builtin name) --
    float('nan')/float('inf')/float('-inf') are.
    """
    v = float(v)
    if not np.isfinite(v):
        if v != v:  # nan
            return "float('nan')"
        return "float('inf')" if v > 0 else "float('-inf')"
    return repr(float(f"{v:.8g}"))


def _pyplot_repr(value):
    """Python source repr for one scalar-ish call argument/kwarg value."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.ndarray):
        return "np.array([" + ", ".join(_pyplot_number(v) for v in value.tolist()) + "])"
    if isinstance(value, float):
        return _pyplot_number(value)
    if isinstance(value, (list, tuple)):
        items = ", ".join(_pyplot_repr(v) for v in value)
        return f"[{items}]" if isinstance(value, list) else f"({items}{',' if len(value) == 1 else ''})"
    return repr(value)


def _pyplot_slug(text, fallback):
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", text.strip()).strip("_").lower() if text else ""
    return slug or fallback


def write_pyplot_script(calls, path, figsize=None):
    """
    Turn a _RecordingAxes' own recorded ax.<method>(...) call log into a
    standalone, runnable matplotlib script at `path` that reconstructs
    the same figure (see --pyplot's own --help for the intent): every
    numpy-array positional argument becomes its own named
    np.array([...]) variable -- named from that call's own `label=`
    kwarg where there is one (so e.g. the "ingress before" curve's data
    becomes `ingress_before_x`/`ingress_before_y`), a generic
    "<method><call index>" fallback otherwise -- and every call itself
    is re-emitted in original order with the same positional/keyword
    arguments (arrays replaced by their new variable name, everything
    else via its own Python repr). The result is not a black-box dump:
    every line is exactly the matplotlib call that produced that part
    of the original figure. Any array over 300 points is uniformly
    subsampled first (same indices across every array in one call, so a
    paired x/y curve keeps its shape) -- a plotted line doesn't need
    every one of a fine root-find's thousands of samples to look right,
    and full-resolution arrays would otherwise bloat the script into an
    unreadable wall of numbers for no visual benefit.
    """
    lines = ["import numpy as np", "import matplotlib.pyplot as plt", ""]
    if figsize is not None:
        lines.append(f"fig, ax = plt.subplots(figsize={_pyplot_repr(tuple(figsize))})")
    else:
        lines.append("fig, ax = plt.subplots()")
    lines.append("")

    max_points = 300
    used_names = set()
    for i, (name, args, kwargs) in enumerate(calls):
        base = _pyplot_slug(kwargs.get("label"), f"{name}{i}")
        # subsample every array in this call together (same indices, taken
        # from the first array's own length), so a paired x/y curve keeps
        # its shape -- a plot line doesn't need every one of a fine
        # bisection sweep's thousands of samples to look right, and this
        # keeps the script from bloating into an unreadable wall of
        # numbers for no visual benefit.
        idx = None
        for a in args:
            if isinstance(a, np.ndarray):
                if len(a) > max_points:
                    idx = np.unique(np.linspace(0, len(a) - 1, max_points).astype(int))
                break
        call_args = []
        for j, a in enumerate(args):
            if isinstance(a, np.ndarray):
                a_use = a[idx] if idx is not None else a
                suffix = "x" if j == 0 else ("y" if j == 1 else f"arr{j}")
                varname = f"{base}_{suffix}"
                n = 1
                while varname in used_names:
                    varname = f"{base}_{suffix}{n}"
                    n += 1
                used_names.add(varname)
                nums = ", ".join(_pyplot_number(v) for v in a_use.tolist())
                lines.append(f"{varname} = np.array([{nums}])")
                call_args.append(varname)
            else:
                call_args.append(_pyplot_repr(a))
        call_args.extend(f"{k}={_pyplot_repr(v)}" for k, v in kwargs.items())
        lines.append(f"ax.{name}(" + ", ".join(call_args) + ")")
        lines.append("")
    lines.append("plt.show()")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _output_path(outdir, prefix, kind, phase=None, ext="png"):
    parts = [p for p in (prefix, kind) if p]
    if phase is not None:
        parts.append(f"{phase:.6f}")
    return os.path.join(outdir, "_".join(parts) + "." + ext)


def _prefix_and_title_label(args):
    """
    --prefix (SYSTEM tab's global label; defaults to --config's basename,
    same as always) doubles as both the output filename prefix and every
    plot's title. Returns (prefix, title_label): `title_label` is the
    human-readable form (blanks intact) used for titles, `prefix` is the
    filesystem-safe form (any run of whitespace collapsed to a single "_")
    used in _output_path -- the only one of the two that ever touches a
    path.
    """
    prefix = args.prefix
    if prefix is None and args.config:
        prefix = os.path.splitext(os.path.basename(args.config))[0]
    title_label = prefix
    if prefix:
        prefix = re.sub(r"\s+", "_", prefix.strip())
    return prefix, title_label


def _save_lightcurve_fits(path, phases, star, discf, sec, total, system, model, args,
                           irradiate, n_disc):
    """
    Write the light curve to a FITS binary table (extension LIGHTCURVE:
    PHASE, PRIMARY, DISC, SECONDARY, TOTAL), with every SystemParams/
    ModelParams value used to compute it -- plus the run-specific
    irradiation/resolution/config settings -- recorded in the primary
    HDU's header, so the file is self-describing without needing the
    command line that produced it.
    """
    from astropy.io import fits
    import params
    from blackbody import mjy_to_ab_mag

    hdr = params.metadata_header(
        system, model,
        IRRADIAT=(irradiate, "secondary irradiation included"),
        NDISC_R=(n_disc[0], "disc temperature-grid radial resolution used"),
        NDISCNU=(n_disc[1], "disc temperature-grid azimuthal resolution used"),
        CONFIG=(args.config or "NONE", "source YAML config file"),
        DIST_PC=(args.dist, "pc, observer distance for the flux/mag columns"),
    )
    hdr.add_comment("Flux columns are a real spectral flux density [mJy] at DIST_PC")
    hdr.add_comment("parsecs -- see render.physical_light_curve. MAG_* columns are the")
    hdr.add_comment("same fluxes as AB magnitudes (Oke & Gunn 1983) -- see")
    hdr.add_comment("blackbody.mjy_to_ab_mag. A fully eclipsed component (zero flux)")
    hdr.add_comment("gives +inf, not an error.")

    cols = [
        fits.Column(name="PHASE", array=np.asarray(phases, dtype=np.float64), format="D"),
        fits.Column(name="PRIMARY", array=star, format="D", unit="mJy"),
        fits.Column(name="DISC", array=discf, format="D", unit="mJy"),
        fits.Column(name="SECONDARY", array=sec, format="D", unit="mJy"),
        fits.Column(name="TOTAL", array=total, format="D", unit="mJy"),
        fits.Column(name="MAG_PRIMARY", array=mjy_to_ab_mag(star), format="D"),
        fits.Column(name="MAG_DISC", array=mjy_to_ab_mag(discf), format="D"),
        fits.Column(name="MAG_SECONDARY", array=mjy_to_ab_mag(sec), format="D"),
        fits.Column(name="MAG_TOTAL", array=mjy_to_ab_mag(total), format="D"),
    ]
    table_hdu = fits.BinTableHDU.from_columns(cols, name="LIGHTCURVE")
    fits.HDUList([fits.PrimaryHDU(header=hdr), table_hdu]).writeto(path, overwrite=True)


def _print_table_preview(path, columns):
    """
    Print an astropy.table-like preview of a just-loaded data table:
    column headers, the first 20 rows, an elision line ("..."), then the
    last 20 rows -- so a data-loading mistake (wrong column, misparsed
    row, bad delimiter) is visible immediately in the console rather than
    only showing up much later as an inexplicable fit/plot result.

    columns: {label: array} for exactly the columns that were actually
    extracted (not necessarily every column in the file) -- reflects what
    the rest of the run will actually use. Tables of 40 rows or fewer are
    printed in full (no elision needed).
    """
    from astropy.table import Table

    table = Table(columns)
    n = len(table)
    print(f"-- {path}: {n} row{'s' if n != 1 else ''} read --")
    lines = table.pformat(max_lines=-1, max_width=-1)
    if n <= 40:
        print("\n".join(lines))
    else:
        # lines[0]/lines[1] are the column-name/dashes header rows (see
        # astropy.table.Table.pformat), shared by both halves below so the
        # column alignment stays consistent across the whole printout.
        header, data_lines = lines[:2], lines[2:]
        print("\n".join(header + data_lines[:20] + ["..."] + data_lines[-20:]))


_lightcurve_data_cache = {}


def _load_lightcurve_data(path, phase_col, value_col, err_col):
    """
    Read observed (phase, value, [error]) columns from a CSV or FITS
    table (dispatched by --data-file's extension) for overlay on the
    computed light curve/magnitude plot, and (via --lsq_fit) as the
    least-squares fit target. `value_col` is whichever of flux or
    magnitude the caller decided to use (see main()'s --data-mag-col/
    --data-flux-col handling); `err_col` is interpreted in those same
    units.

    err_col: pass '' or None if the data has no uncertainty column of its
    own -- err is then returned as None (not a made-up array), so a
    caller can tell "no error data" apart from "errors happen to be
    zero" and decide what that means for it (main() skips error bars
    when plotting; _run_lsq_fit falls back to equal weights when
    fitting).

    CSV column names are matched after stripping leading/trailing
    whitespace: a ", " (comma-space) delimiter style is common and
    otherwise silently produces header names like " phase" that a plain
    "phase" can never match.

    For FITS, searches every HDU for the first table extension that has
    phase_col/value_col (and err_col, if given). Returns (phase, value,
    error-or-None) as float arrays.

    Repeated calls with the same (path, phase_col, value_col, err_col) --
    e.g. --lsq_fit/--mcmc_fit's _prepare_fit loading the same RV table
    the "rv" output block then loads again for its own overlay -- reuse
    the first call's already-read arrays (each caller still gets its own
    copy, safe to modify) instead of re-reading the file and printing the
    same _print_table_preview a second time.
    """
    key = (os.path.abspath(path), phase_col, value_col, err_col or "")
    cached = _lightcurve_data_cache.get(key)
    if cached is not None:
        phase, value, err = cached
        return phase.copy(), value.copy(), (err.copy() if err is not None else None)

    ext = os.path.splitext(path)[1].lower()
    required = [phase_col, value_col] + ([err_col] if err_col else [])
    if ext == ".fits":
        from astropy.io import fits
        with fits.open(path) as hdul:
            table = None
            for hdu in hdul:
                names = getattr(getattr(hdu, "columns", None), "names", None)
                if names and all(c in names for c in required):
                    table = hdu.data
                    break
            if table is None:
                raise ValueError(
                    f"no table extension in {path!r} has all of columns "
                    f"{', '.join(map(repr, required))}")
            phase = np.asarray(table[phase_col], dtype=float)
            value = np.asarray(table[value_col], dtype=float)
            err = np.asarray(table[err_col], dtype=float) if err_col else None
    elif ext == ".csv":
        import csv
        with open(path, newline="") as f:
            rows = [{k.strip(): v for k, v in row.items()} for row in csv.DictReader(f)]
        if not rows or not set(required) <= set(rows[0]):
            found = list(rows[0].keys()) if rows else []
            raise ValueError(
                f"{path!r} is missing one of columns {', '.join(map(repr, required))} "
                f"(found: {found})")
        phase = np.array([float(r[phase_col]) for r in rows])
        value = np.array([float(r[value_col]) for r in rows])
        err = np.array([float(r[err_col]) for r in rows]) if err_col else None
    else:
        raise ValueError(f"--data-file must be .csv or .fits, got {path!r}")
    cols = {phase_col: phase, value_col: value}
    if err is not None:
        cols[err_col] = err
    _print_table_preview(path, cols)
    _lightcurve_data_cache[key] = (phase, value, err)
    return phase.copy(), value.copy(), (err.copy() if err is not None else None)


def _wrap_extend_phase(phase, phase_min, phase_max):
    """
    Index array (into `phase`) and matching integer shift array such
    that phase[idx] + shift lands inside [phase_min, phase_max] for
    every integer shift where it does -- i.e. replicate period-1
    (unity) phase-folded data across as many cycles as the requested
    display window spans, the same way the model curve itself already
    repeats periodically (phase only ever enters as 2*pi*phase, e.g.
    see eclipse.observer_frame, so it's exact for any integer shift).

    Used only for the observed-data overlay on the lightcurve/magnitude
    plots: if --phase-min/--phase-max spans more than one cycle -- e.g.
    -1..1 rather than the usual -0.5..0.5 -- data that was
    phase-folded into a single cycle would otherwise show up only once,
    leaving the rest of a multi-cycle display looking sparse next to the
    model curve's repeats. NOT applied to the fit itself (_run_lsq_fit
    residuals stay one-to-one against the real, unduplicated data
    points -- duplicating them there would double-count/reweight the
    chi^2 for no reason). Returns empty arrays for empty input.
    """
    phase = np.asarray(phase, dtype=float)
    if phase.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    k_lo = int(np.floor(phase_min - np.max(phase)))
    k_hi = int(np.ceil(phase_max - np.min(phase)))
    idx_parts, shift_parts = [], []
    for k in range(k_lo, k_hi + 1):
        mask = (phase + k >= phase_min) & (phase + k <= phase_max)
        if np.any(mask):
            idx_parts.append(np.nonzero(mask)[0])
            shift_parts.append(np.full(mask.sum(), float(k)))
    if not idx_parts:
        return np.array([], dtype=int), np.array([], dtype=float)
    return np.concatenate(idx_parts), np.concatenate(shift_parts)


def _parse_fit_names(param_str, flag_name):
    """
    Parse a --lsq_fit/--mcmc_fit comma-separated parameter list into
    (names, groups): each name is either 'dist' (render.physical_light_curve's
    distance_pc), 'data_norm' (--data_norm's own multiplicative flux
    factor/additive magnitude offset), 'rv_gamma' (--data_rv_gamma's own
    systemic-velocity offset, see its --help), or a SystemParams/ModelParams
    field name (params.py); `groups` is the parallel list of which.
    """
    names = [s.strip() for s in param_str.split(",") if s.strip()]
    if not names:
        raise SystemExit(f"{flag_name} given but no parameter names found")
    groups = []
    for name in names:
        if name == "dist":
            groups.append("dist")
        elif name == "data_norm":
            groups.append("data_norm")
        elif name == "rv_gamma":
            groups.append("rv_gamma")
        elif name in _SYSTEM_FIELDS:
            groups.append("system")
        elif name in _MODEL_FIELDS:
            groups.append("model")
        else:
            raise SystemExit(f"{flag_name}: unknown parameter {name!r} -- must be a "
                              f"SystemParams/ModelParams field name, or "
                              f"'dist'/'data_norm'/'rv_gamma'")
    return names, groups


def _prepare_fit(args, system, model, lobe, disc, disc_teff_func, temp_maps,
                  n_primary, n_disc, irradiate, ntheta, nphi, angle_acc_list, names, groups, flag_name):
    """
    Shared machinery for --lsq_fit/--mcmc_fit: loads --data-file and/or
    --data-rv-file once, and builds the closures both fitting methods
    evaluate a trial parameter vector `x` (ordered same as `names`) with.

    The model is evaluated at the DATA's own phases (shifted by whatever
    trial ph_off is current, if ph_off is being fit) rather than at the
    --phase-min/--phase-max/--phase-num plotting grid, since the fit needs a
    residual per actual data point, not a smooth curve.

    Fields in _CHEAP_SYSTEM_FIT_FIELDS/_CHEAP_MODEL_FIT_FIELDS are ones
    physical_light_curve takes as live arguments on top of a fixed
    temp_maps -- reused unchanged across every trial for speed. Fitting
    anything else (disc geometry/temperature, T_2, R_2, q, ...) needs
    lobe/disc/temp_maps rebuilt from scratch every trial, since those are
    baked into temp_maps at build time; slower, but correct.

    angle_acc_list: every --angle_acc entry (see params._parse_angle_acc), if it's a
    list of more than one -- only its first entry ever varies here (as
    "angle_acc", if that's a fit parameter; model.angle_acc/model2.angle_acc only
    ever hold that one value, see params._parse_angle_acc's own docstring),
    with any further entries rebuilt at their own fixed radius on every
    trial, same as when angle_acc isn't being fit at all.

    Photometric (flux/mag) and RV data are independent, both-optional fit
    targets, and at least one must be given. --data-file enables the
    photometric residual (unchanged from before RV support existed). Each
    of the four --data_rv_<1|2|stream|hotspot>_col flags (RV_DATA_COMPONENTS)
    independently enables that component's own RV residual, against
    whichever of --data-rv-file/--data-file has that column -- any subset
    may be given at once (e.g. just the secondary's RV curve, or all four
    simultaneously if the data distinguishes them). Every active data
    source's residual is concatenated into one combined chi-squared by
    residuals(x), i.e. a simultaneous photometric+RV fit -- possibly
    against multiple RV components at once -- same as e.g. JKTEBOP/PHOEBE.

    Returns (x0, apply, residuals, rebuild, n_obs):
      x0: each named parameter's starting value (system/model's current
        value, or args.dist/data_norm/data_rv_gamma), same order as `names`.
      apply(x) -> (system2, model2, dist2, data_norm2, rv_gamma2).
      residuals(x) -> (model - data)/data_err for every active data
        source, concatenated -- always takes the "cheap" branch when it
        can, however slow the alternative would be, since it runs on
        every trial.
      rebuild(x) -> (system2, model2, dist2, data_norm2, rv_gamma2, lobe2,
        disc2, temp_maps2), i.e. apply(x) plus a from-scratch lobe/disc/
        temp_maps rebuild if fitting isn't "cheap" -- lobe/disc/temp_maps
        themselves, unchanged, otherwise. Meant to be called once, on the
        final fitted x, by both _run_lsq_fit and _run_mcmc_fit.
      n_obs: total number of data points feeding residuals(x) (photometric
        plus RV), e.g. for a caller's own dof count.
    """
    from blackbody import ab_mag_to_mjy
    from render import build_temperature_maps, physical_light_curve, radial_velocity_curve

    have_lc = bool(args.data_file)
    active_rv = [(suffix, curve_name) for suffix, curve_name, _color in RV_DATA_COMPONENTS
                 if getattr(args, f"data_rv_{suffix}_col")]
    if not have_lc and not active_rv:
        raise SystemExit(f"{flag_name} requires --data-file (photometric) and/or at least one "
                          f"--data_rv_<1|2|stream|hotspot>_col (RV) -- nothing to fit against")

    cheap = all(
        g in ("dist", "data_norm", "rv_gamma")
        or (g == "system" and n in _CHEAP_SYSTEM_FIT_FIELDS)
        or (g == "model" and n in _CHEAP_MODEL_FIT_FIELDS)
        for n, g in zip(names, groups)
    )
    if not cheap:
        print(f"{flag_name}: one or more parameters affect disc geometry/irradiation, so "
              "lobe/disc/temperature maps are rebuilt on every trial -- this will be slow")

    n_obs = 0

    if have_lc:
        if args.data_mag_col:
            raw_phase, raw_value, raw_err = _load_lightcurve_data(
                args.data_file, args.data_phase_col, args.data_mag_col, args.data_err_col)
            is_mag = True
        else:
            raw_phase, raw_value, raw_err = _load_lightcurve_data(
                args.data_file, args.data_phase_col, args.data_flux_col, args.data_err_col)
            is_mag = False
        if raw_err is None:
            print(f"{flag_name}: no error column given (--data-err-col) -- fitting "
                  "photometry with equal weights")
            raw_err = np.ones_like(raw_value)
        n_obs += len(raw_phase)
        # --data_norm's no-op identity differs by branch (additive 0.0 for
        # magnitude, multiplicative 1.0 for flux) -- see its own --help
        data_norm = args.data_norm if args.data_norm is not None else (0.0 if is_mag else 1.0)
    else:
        data_norm = args.data_norm if args.data_norm is not None else 1.0

    rv_data = {}  # curve_name -> (phase, value, err)
    if active_rv:
        rv_file = args.data_rv_file or args.data_file
        if not rv_file:
            raise SystemExit("--data_rv_*_col requires --data-rv-file or --data-file")
        for suffix, curve_name in active_rv:
            col = getattr(args, f"data_rv_{suffix}_col")
            err_col = getattr(args, f"data_rv_{suffix}_err_col")
            phase, value, err = _load_lightcurve_data(rv_file, args.data_rv_phase_col, col, err_col)
            if err is None:
                print(f"{flag_name}: no error column given (--data_rv_{suffix}_err_col) -- "
                      f"fitting {curve_name} RV with equal weights")
                err = np.ones_like(value)
            rv_data[curve_name] = (phase, value, err)
            n_obs += len(phase)

    def effective_angle_acc(model_):
        # the trial's own model_.angle_acc (possibly the current fit value)
        # stands in for the list's first entry; any further --angle_acc
        # entries stay fixed at whatever they were given as.
        if angle_acc_list is not None and len(angle_acc_list) > 1:
            return np.concatenate([[model_.angle_acc], angle_acc_list[1:]])
        return model_.angle_acc

    def start_value(name, group):
        if group == "dist":
            return args.dist
        if group == "data_norm":
            return data_norm
        if group == "rv_gamma":
            return args.data_rv_gamma
        return getattr(system if group == "system" else model, name)

    x0 = np.array([start_value(n, g) for n, g in zip(names, groups)], dtype=float)

    def apply(x):
        sys_over, model_over = {}, {}
        dist_val, norm_val, gamma_val = args.dist, data_norm, args.data_rv_gamma
        for n, g, v in zip(names, groups, x):
            if g == "system":
                sys_over[n] = v
            elif g == "model":
                model_over[n] = v
            elif g == "dist":
                dist_val = v
            elif g == "data_norm":
                norm_val = v
            else:
                gamma_val = v
        sys2 = dataclasses.replace(system, **sys_over) if sys_over else system
        model2 = dataclasses.replace(model, **model_over) if model_over else model
        return sys2, model2, dist_val, norm_val, gamma_val

    def _temp_maps_for(sys2, model2):
        if cheap:
            return lobe, disc, temp_maps
        lobe2, disc2, teff2 = build_system(sys2, model2, ntheta=ntheta, nphi=nphi)
        tm2 = build_temperature_maps(
            lobe2, disc2, sys2.T_1, sys2.T_2, sys2.R_1, disc_teff_func=teff2,
            irradiate=irradiate, beta_grav=model2.beta_grav,
            hotspot_T_h=model2.T_h if model2.has_hotspot else None,
            hotspot_L_h_deg=model2.L_h, n_sec=args.n_areas_2, n_disc=n_disc,
            u_disc=model2.u_d, u_primary=sys2.u_1, n_primary=n_primary,
            theta_1=sys2.theta_1_rad, phi_1=sys2.phi_1_rad, angle_acc=effective_angle_acc(model2),
            spot_acc=model2.spot_acc_rad, T_acc=model2.T_acc, u_acc=model2.u_acc,
            incl_deg=sys2.incl, stream_angle_deg=args.stream_angle,
            stream_eps=lubow_shu_eps(sys2.T_2, sys2.P_orb, sys2.a_m))
        return lobe2, disc2, tm2

    def residuals(x):
        sys2, model2, dist2, norm2, gamma2 = apply(x)
        lobe2, disc2, tm2 = _temp_maps_for(sys2, model2)
        parts = []

        if have_lc:
            if is_mag:
                target_mag = raw_value + norm2
                target_flux = ab_mag_to_mjy(target_mag)
                target_err = target_flux * (np.log(10.0) / 2.5) * raw_err
            else:
                target_flux = raw_value * norm2
                target_err = raw_err * norm2
            eval_phase = raw_phase - sys2.ph_off

            _, _, _, total = physical_light_curve(
                lobe2, disc2, sys2.T_1, sys2.T_2, sys2.R_1, eval_phase, sys2.incl, sys2.a_m,
                wavelength_m=sys2.wavelength_m, temp_maps=tm2, n_workers=1,
                distance_pc=dist2, u_primary=sys2.u_1, u_secondary=sys2.u_2, u_disc=model2.u_d,
                T_acc=model2.T_acc, u_acc=model2.u_acc)
            parts.append((total - target_flux) / target_err)

        for curve_name, (phase, value, err) in rv_data.items():
            # each component's own phases (usually the same rows/phase
            # column, but not required to be -- see RV_DATA_COMPONENTS'
            # docstring), so a fresh call per active component; all four
            # curves come out of one radial_velocity_curve call regardless
            # of which single one is actually used here.
            eval_rv_phase = phase - sys2.ph_off
            rv_curves = radial_velocity_curve(
                lobe2, disc2, sys2.T_1, sys2.R_1, eval_rv_phase, sys2.incl, sys2.a_m, sys2.P_orb_s,
                tm2, wavelength_m=sys2.wavelength_m, u_primary=sys2.u_1, u_secondary=sys2.u_2,
                T_acc=model2.T_acc, u_acc=model2.u_acc, stream_angle_deg=args.stream_angle)
            model_rv = dict(zip(("primary", "secondary", "stream", "magnetic"), rv_curves[:4]))[
                curve_name]
            parts.append((model_rv + gamma2 - value) / err)

        return np.concatenate(parts)

    def rebuild(x):
        sys2, model2, dist2, norm2, gamma2 = apply(x)
        lobe2, disc2, tm2 = _temp_maps_for(sys2, model2)
        return sys2, model2, dist2, norm2, gamma2, lobe2, disc2, tm2

    return x0, apply, residuals, rebuild, n_obs


def _lsq_covariance(names, result, n_obs):
    """
    (cov, sigmas, chi2_reduced, dof) for a converged scipy.optimize.
    least_squares `result` -- the standard linearized (Gauss-Newton)
    estimate cov = (J^T J)^-1 * chi2/dof, J = result.jac (the residual
    Jacobian at the solution) -- the "unknown common error scale"
    convention (matches scipy.optimize.curve_fit's own absolute_sigma=
    False default: even where the residuals are already divided by a
    real per-point uncertainty, this still rescales so the reported
    chi^2/dof comes out consistent with the fitted errors, a
    conservative hedge against the given uncertainties themselves being
    off by a common unknown factor), used identically by every --lsq_fit
    variant in this module (the main light-curve fit, and the shadow/
    rim-projection/primary geometric ones).

    n_obs: number of independent data points the residual vector was
    built from (e.g. real photometric/RV points for the main fit; just
    len(result.fun) for the purely-geometric shadow/rim-projection/
    primary fits, which have no separate "data" of their own).

    cov is an all-NaN (len(names), len(names)) matrix (sigmas all NaN
    too) if J^T J is singular rather than raising -- e.g. a genuinely
    unconstrained direction in parameter space (see this module's own
    q/incl near-degeneracy discussions for "primary"/"shadow"-mode
    fits). dof is floored at 1 (never 0 or negative) purely so chi2/dof
    stays a finite, if not always meaningful, number to print.
    """
    dof = max(n_obs - len(names), 1)
    chi2 = float(np.sum(result.fun ** 2))
    chi2_reduced = chi2 / dof
    try:
        cov = np.linalg.inv(result.jac.T @ result.jac) * chi2_reduced
    except np.linalg.LinAlgError:
        cov = np.full((len(names), len(names)), np.nan)
    sigmas = np.sqrt(np.diag(cov))
    return cov, sigmas, chi2_reduced, dof


def _print_lsq_covariance(names, result, n_obs):
    """Print names' fitted value +/- sigma, the covariance matrix, and reduced chi^2 -- see _lsq_covariance."""
    cov, sigmas, chi2_reduced, dof = _lsq_covariance(names, result, n_obs)
    for name, val, sig in zip(names, result.x, sigmas):
        print(f"\t{name} = {val:.4g} +/- {sig:.4g}")
    print("\tcovariance matrix:")
    print("\t\t",cov)
    print(f"\treduced chi^2 = {chi2_reduced:.4f} (dof={dof})")


_M_CHANDRASEKHAR = 1.44  # Msun -- Nauenberg (1972)'s own zero-temperature limiting mass


def nauenberg_R1(M_1):
    """
    Nauenberg (1972, ApJ 175, 417) analytic zero-temperature white-dwarf
    mass-radius relation [SI: kg in, m out]:

        R_1 = 0.0114 Rsun * sqrt((M_1/M_ch)^(-2/3) - (M_1/M_ch)^(2/3))

    M_ch = 1.44 Msun (the Chandrasekhar mass) -- the closed-form fit
    (carbon-oxygen composition, mean molecular weight per electron
    mu_e=2) widely used in eclipsing-CV eclipse-timing analyses to tie a
    fitted white-dwarf radius to its mass. Undefined (imaginary) at or
    above M_ch, by construction (the relation's own limiting mass).
    """
    from render import MSUN, RSUN

    M_ch = _M_CHANDRASEKHAR * MSUN
    x = M_1 / M_ch
    return 0.0114 * RSUN * np.sqrt(x ** (-2.0 / 3.0) - x ** (2.0 / 3.0))


def _mass_and_a_from_R1_over_a(R1_over_a, q, P_orb_s):
    """
    The unique (M_1, a) [SI: kg, m] simultaneously satisfying Nauenberg's
    own R_1(M_1) (nauenberg_R1) and Kepler's third law
    (a^3 = G*M_1*(1+q)*P_orb_s^2/(4*pi^2), M_2=q*M_1) such that
    nauenberg_R1(M_1)/a equals the given dimensionless R1_over_a --
    i.e. the physical mass/separation this codebase's own fitted R_1
    (always R_1/a, see roche.py's module docstring) implies, given q
    and P_orb assumed independently known (not derived here; q may
    itself be a jointly fitted parameter, P_orb never is in this
    codebase).

    Monotonic, single-rooted by construction: nauenberg_R1(M_1) strictly
    DECREASES with M_1 (electron degeneracy pressure -- more massive
    white dwarfs are smaller), while Kepler's own a strictly INCREASES
    with M_1 at fixed q/P_orb, so their ratio is strictly decreasing
    across the whole physical mass range -- scipy.optimize.brentq needs
    only a wide, generic bracket (essentially 0 up to just short of the
    Chandrasekhar mass, where nauenberg_R1 itself is only defined).
    """
    from scipy.optimize import brentq
    from render import G, MSUN

    M_ch = _M_CHANDRASEKHAR * MSUN

    def a_of_M1(M_1):
        return (G * M_1 * (1.0 + q) * P_orb_s ** 2 / (4.0 * np.pi ** 2)) ** (1.0 / 3.0)

    def resid(M_1):
        return nauenberg_R1(M_1) / a_of_M1(M_1) - R1_over_a

    M_1 = brentq(resid, 1e-3 * MSUN, 0.999999 * M_ch, xtol=1.0, rtol=1e-13)
    return M_1, a_of_M1(M_1)


def _nauenberg_derived_quantities(R1_over_a, q, incl_deg, P_orb_s, lobe2):
    """
    The full physical (M_1, M_2, R_1, R_2, a, K_1, K_2) [SI] implied by a
    fitted R_1 (R1_over_a), q, incl_deg, and P_orb_s -- _mass_and_a_from_
    R1_over_a's own (M_1, a) solve, M_2=q*M_1 and R_1=nauenberg_R1(M_1)
    following directly (self-consistent by construction), R_2 -- the
    secondary's own mean (volume-equivalent) radius, lobe2.
    r_volume_equiv (dimensionless, Roche geometry, q alone) -- scaled by
    the resulting a, and K_1/K_2 -- the two stars' own projected radial-
    velocity semi-amplitudes for a CIRCULAR orbit (this whole codebase's
    own standing assumption): K_i = 2*pi*a_i*sin(incl)/P_orb, with each
    star's own distance from the center of mass a_1=a*q/(1+q),
    a_2=a/(1+q) (the usual lever-arm relation, M_1*a_1=M_2*a_2).
    lobe2 must already reflect the SAME q as given here (the caller's
    own responsibility, same as every other lobe2 this module's fit code
    threads through).

    Returns a dict {"M_1","M_2","a","R_1","R_2","K_1","K_2"}, all SI
    (kg, m, m/s).
    """
    M_1, a = _mass_and_a_from_R1_over_a(R1_over_a, q, P_orb_s)
    incl_rad = np.radians(incl_deg)
    v_scale = 2.0 * np.pi * a * np.sin(incl_rad) / P_orb_s
    return {"M_1": M_1, "M_2": q * M_1, "a": a,
            "R_1": nauenberg_R1(M_1), "R_2": lobe2.r_volume_equiv * a,
            "K_1": v_scale * q / (1.0 + q), "K_2": v_scale / (1.0 + q)}


def _propagate_scalar(func_of_x, x0, cov):
    """
    First-order (linearized) error propagation of a scalar func(x)
    through a fit's own parameter covariance matrix cov (same convention
    _lsq_covariance itself returns, or an MCMC chain's own sample
    covariance np.cov(chain, rowvar=False) -- either way, just "the
    covariance of whatever x0 is"): sigma_func^2 = grad^T @ cov @ grad,
    grad[j] = d(func)/d(x0[j]) by a small central difference in the j-th
    entry alone (holding the others at x0's own value) -- the same
    "propagate through the local slope" idea _primary_fit_distance_sigmas/
    _rim_fit_sigmas use for a single input (phase), generalized here to
    every entry of a fitted parameter vector at once, correlations
    included.

    Returns (func(x0), sigma) -- sigma is NaN if cov itself has any
    non-finite entry (e.g. a singular fit -- see _lsq_covariance) or the
    resulting variance comes out negative (a poorly-conditioned cov).
    """
    value = func_of_x(x0)
    if not np.all(np.isfinite(cov)):
        return value, float("nan")
    n = len(x0)
    grad = np.empty(n)
    for j in range(n):
        step = max(abs(x0[j]), 1e-8) * 1e-4
        x_hi = np.array(x0, dtype=float)
        x_hi[j] += step
        x_lo = np.array(x0, dtype=float)
        x_lo[j] -= step
        grad[j] = (func_of_x(x_hi) - func_of_x(x_lo)) / (2.0 * step)
    var = float(grad @ cov @ grad)
    return value, (np.sqrt(var) if var >= 0.0 else float("nan"))


def _print_nauenberg_derived(sys2_lobe_of, x0, cov, full):
    """
    Print the physical quantities a fitted R_1 implies via Nauenberg's
    own mass-radius relation (_nauenberg_derived_quantities), errors
    propagated (_propagate_scalar) through the fit's own covariance
    matrix -- called only when "R_1" is one of the fitted parameters (an
    R_1 held fixed carries no fit uncertainty to propagate, and this
    codebase's own --R_1/--a inputs are already usable directly with no
    Nauenberg solve needed).

    sys2_lobe_of(x) -> (sys2, lobe2) for one trial parameter vector x --
    each fit runner supplies its own (wrapping its own apply/lobe_for or
    apply/lobe_disc_for closures), so this helper doesn't need to know
    which fit it's being called from.

    full=False prints just the secondary's own mean radius R_2 and the
    separation a ("primary"-mode fitting's own request); full=True also
    prints M_1, M_2, R_1 itself (the Nauenberg-implied one, for
    comparison against the fitted R_1*a), and the two stars' own
    projected radial-velocity semi-amplitudes K_1/K_2 -- "fit system"'s
    own fuller physical summary.
    """
    from render import MSUN, RSUN

    sys2_0, lobe2_0 = sys2_lobe_of(x0)
    vals = _nauenberg_derived_quantities(sys2_0.R_1, sys2_0.q, sys2_0.incl, sys2_0.P_orb_s, lobe2_0)

    def deriv(name):
        def f(x):
            sys2x, lobe2x = sys2_lobe_of(x)
            return _nauenberg_derived_quantities(sys2x.R_1, sys2x.q, sys2x.incl,
                                                  sys2x.P_orb_s, lobe2x)[name]
        return f

    print("\tNauenberg (1972) white-dwarf mass-radius relation (R_1 fit -> M_1, a solved "
          "self-consistently via Kepler's third law):")
    names = ("M_1", "M_2", "R_1", "R_2", "a", "K_1", "K_2") if full else ("R_2", "a")
    units = {"M_1": ("Msun", MSUN), "M_2": ("Msun", MSUN), "R_1": ("Rsun", RSUN),
             "R_2": ("Rsun", RSUN), "a": ("Rsun", RSUN),
             "K_1": ("km/s", 1000.0), "K_2": ("km/s", 1000.0)}
    for name in names:
        _, sigma = _propagate_scalar(deriv(name), x0, cov)
        unit, scale = units[name]
        print(f"\t\t{name} = {vals[name] / scale:.4g} +/- {sigma / scale:.4g} {unit}")


def _run_lsq_fit(args, system, model, lobe, disc, disc_teff_func, temp_maps,
                  n_primary, n_disc, irradiate, ntheta, nphi, angle_acc_list=None):
    """
    Least-squares fit of --lsq_fit's comma-separated parameter names
    against --data-file and/or any active --data_rv_*_col, via
    scipy.optimize.least_squares -- see _prepare_fit for the shared parameter-name syntax/semantics
    ('dist'/'data_norm'/'rv_gamma' pseudo-parameters, "cheap" vs. rebuilt
    trials, angle_acc_list). Start values come straight from each named
    parameter's current (--config/CLI/default) value.

    Returns (system, model, dist, data_norm, rv_gamma, lobe, disc, temp_maps)
    updated to the best-fit values -- the last three are the *same*
    objects passed in when every fit field is "cheap", or freshly rebuilt
    ones (consistent with the fitted system/model) otherwise, so the
    caller can just keep using whichever it gets back.
    """
    from scipy.optimize import least_squares

    names, groups = _parse_fit_names(args.lsq_fit, "--lsq_fit")
    x0, apply, residuals, rebuild, n_obs = _prepare_fit(
        args, system, model, lobe, disc, disc_teff_func, temp_maps,
        n_primary, n_disc, irradiate, ntheta, nphi, angle_acc_list, names, groups, "--lsq_fit")

    result = least_squares(residuals, x0)

    print(f"\nLeast-squares fit to {', '.join(names)}:")
    _print_lsq_covariance(names, result, n_obs)

    return rebuild(result.x)


def _run_mcmc_fit(args, system, model, lobe, disc, disc_teff_func, temp_maps,
                   n_primary, n_disc, irradiate, ntheta, nphi, angle_acc_list, finish):
    """
    MCMC fit of --mcmc_fit's comma-separated parameter names against
    --data-file and/or any active --data_rv_*_col, via emcee -- see
    _prepare_fit for the shared parameter-name syntax/semantics
    ('dist'/'data_norm'/'rv_gamma'
    pseudo-parameters, "cheap" vs. rebuilt trials, angle_acc_list), identical
    to --lsq_fit's own.

    Walks the posterior with --walkers*<number of fitted parameters>
    walkers, started in a small Gaussian ball (relative dispersion
    --spread, default 1%) around each parameter's current (--config/CLI/
    default) value -- the same starting point --lsq_fit uses -- for
    --nburn steps (discarded as burn-in), then --nsample further steps
    kept as the posterior sample. Each step's walkers are evaluated across
    --workers threads (default 1, i.e. serial) -- the main stop-gap for
    slow fits, since a "cheap" fit's own per-walker cost is usually too
    small to be worth a process pool's startup/pickling overhead.
    The likelihood is the same chi-squared _prepare_fit's residuals()
    computes (Gaussian errors: log L = -0.5*sum(residuals**2)); there's
    no prior beyond that (flat/improper, matching --lsq_fit's own
    unbounded least-squares), except that a trial whose residuals aren't
    finite (an unphysical parameter combination the model can't evaluate,
    or can't evaluate at all) is rejected outright (log P = -inf) rather
    than raising and aborting the whole run.

    Each parameter's fit value is its marginal posterior MEDIAN. Also
    reports a 90% credible interval per parameter: +error/-error are the
    95th/5th percentile's distance from the median, and the "+/-" figure
    printed alongside them is their mean -- a quick symmetric summary
    next to that asymmetric pair, not an independent statistic.

    finish: --corner_plot's figure (the classic corner.py posterior
    pairwise-correlation plot) is handed to the SAME finish() closure
    main() uses for every other output (saved via _output_path under
    kind="corner", or left for the shared final plt.show() under --show)
    -- see simulate.py's OUTPUT_KIND.

    Returns (system, model, dist, data_norm, rv_gamma, lobe, disc, temp_maps)
    -- same shape/contract as _run_lsq_fit's return.
    """
    import emcee
    from multiprocessing.pool import ThreadPool

    names, groups = _parse_fit_names(args.mcmc_fit, "--mcmc_fit")
    x0, apply, residuals, rebuild, n_obs = _prepare_fit(
        args, system, model, lobe, disc, disc_teff_func, temp_maps,
        n_primary, n_disc, irradiate, ntheta, nphi, angle_acc_list, names, groups, "--mcmc_fit")

    ndim = len(names)
    nwalkers = args.walkers * ndim

    def log_probability(x):
        try:
            r = residuals(x)
        except Exception:
            return -np.inf
        if not np.all(np.isfinite(r)):
            return -np.inf
        return -0.5 * float(np.sum(r ** 2))

    rng = np.random.default_rng()
    spread = np.where(x0 != 0.0, np.abs(x0), 1.0) * args.spread
    pos = x0 + spread * rng.standard_normal((nwalkers, ndim))

    # a real process Pool can't pickle log_probability/residuals -- they're
    # closures over this call's own lobe/disc/temp_maps, not top-level
    # functions -- so a thread pool is used instead; numpy releases the GIL
    # during its own C-level work, so this still parallelizes the walkers'
    # per-step evaluations across --workers threads
    pool = ThreadPool(args.workers) if args.workers > 1 else None
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool)
    print(f"--mcmc_fit: {nwalkers} walkers ({args.walkers}/parameter), "
          f"{args.nburn} burn-in + {args.nsample} sample steps"
          + (f", {args.workers} worker threads" if pool is not None else ""))
    try:
        state = sampler.run_mcmc(pos, args.nburn, progress=True)
        sampler.reset()
        sampler.run_mcmc(state, args.nsample, progress=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    chain = sampler.get_chain(flat=True)
    medians = np.median(chain, axis=0)
    p05 = np.percentile(chain, 5, axis=0)
    p95 = np.percentile(chain, 95, axis=0)
    plus_err = p95 - medians
    minus_err = medians - p05
    mean90_err = 0.5 * (plus_err + minus_err)

    print(f"\nMCMC fit to {', '.join(names)} "
          f"({nwalkers} walkers x {args.nsample} steps = {len(chain)} samples):")
    for name, med, m9, pl, mi in zip(names, medians, mean90_err, plus_err, minus_err):
        print(f"\t{name} = {med:.4g} +/- {m9:.4g}  (+{pl:.4g} / -{mi:.4g})")
    print(f"\tmean acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")

    if args.corner_plot:
        import corner
        fig = corner.corner(chain, labels=names, truths=medians, show_titles=True)
        # unlike every other output (see plots.labeled_title), a corner
        # plot is an NxN grid of axes with no single "own" title to merge
        # the global label into -- show_titles=True already puts each
        # parameter's own fit result above its diagonal subplot -- so the
        # label goes on a plain figure-level suptitle instead.
        _, title_label = _prefix_and_title_label(args)
        if title_label:
            fig.suptitle(title_label)
        finish(fig, "corner")

    return rebuild(medians)


_SHADOW_FIT_FIELDS = ("q", "incl", "R_out")


def _parse_shadow_fit_names(param_str, flag_name):
    """
    Parse a --lsq_fit/--mcmc_fit comma-separated parameter list for
    shadow-mode fitting (see _run_shadow_lsq_fit/_run_shadow_mcmc_fit's
    own docstrings) -- only 'q'/'incl'/'R_out' are meaningful here: q
    and incl actually move the shadow curves and the ballistic stream
    and so are real free parameters; R_out has NO effect on the
    residual at all (the shadow/stream crossing this fits doesn't
    involve the disc), so it's never numerically varied -- naming it
    just means "also derive R_out from the fitted crossing point
    afterward, for free" (see _derive_R_out).

    Returns (active_names, derive_R_out): active_names is the ordered
    subset of q/incl actually being fit (possibly empty, if only R_out
    was named -- see both _run_shadow_* functions' own "R_out alone"
    shortcut, which skips optimization entirely then).
    """
    names = [s.strip() for s in param_str.split(",") if s.strip()]
    if not names:
        raise SystemExit(f"{flag_name} given but no parameter names found")
    bad = [n for n in names if n not in _SHADOW_FIT_FIELDS]
    if bad:
        raise SystemExit(f"{flag_name}: shadow-output fitting only supports "
                          f"q, incl, R_out (got {bad!r}) -- q/incl move the shadow "
                          f"curves/stream, R_out is only ever derived, never fit "
                          f"(see --help)")
    active_names = [n for n in names if n != "R_out"]
    return active_names, "R_out" in names


def _shadow_fit_phases(args, flag_name, output_name="shadow-output"):
    """
    The --phase-list phases shadow-mode/rim-mode fitting needs: the 1st
    (ingress-before) and 3rd (egress-before) of the 4 (see plots.
    shadow_constrained_regions' own ingress_phases/egress_phases) --
    exactly the pair whose shadow-boundary curves are matched against
    the stream (in the midplane, shadow_curve_crossings; on the rim
    wall, shadow_boundary_on_rim -- see _shadow_fit_target/
    _rim_fit_target respectively). Raises if --phase-list isn't exactly
    4 phases.
    """
    if args.phase_list is None or len(args.phase_list) != 4:
        raise SystemExit(f"{flag_name}: {output_name} fitting needs --phase-list with "
                          f"exactly 4 phases (ingress-before, ingress-after, "
                          f"egress-before, egress-after) -- the 1st/3rd of those are "
                          f"the two shadow curves being matched to the stream")
    return args.phase_list[0], args.phase_list[2]


def _shadow_fit_target(sys2, lobe2, phase_a, phase_b):
    """
    The shadow-mode fit's core geometric target, at one trial (q,incl)
    (sys2/lobe2 already reflect it): every crossing of the phase_a/
    phase_b shadow-boundary curves (plots.shadow_curve_crossings), each
    scored by its distance to the ballistic stream's own near-primary
    Lubow & Shu flanking line (stream.near_primary_flank_distance),
    keeping whichever crossing is CLOSEST to that line -- the real
    system's own shadow/stream crossing, if q/incl are anywhere near
    right, should be unambiguously closer to it than any other spurious
    crossing the two curves happen to have elsewhere.

    Returns (distance, point) -- point is the winning crossing's own
    (x,y) [corotating frame], or (None, None) if the two shadow curves
    never cross at all for this trial.
    """
    from plots import shadow_curve_crossings
    from stream import near_primary_flank_distance

    incl2 = np.radians(sys2.incl)
    crossings = shadow_curve_crossings(lobe2, phase_a, phase_b, incl2, lobe2.x1)
    if not crossings:
        return None, None
    dists = [near_primary_flank_distance(lobe2, sys2.T_2, sys2.P_orb, sys2.a_m, pt)
             for pt in crossings]
    i = int(np.argmin(dists))
    return dists[i], crossings[i]


def _derive_R_out(sys2, lobe2, phase_a, phase_b):
    """
    R_out "derived for free" (see _parse_shadow_fit_names) from the
    current q/incl's own best shadow/stream crossing point -- just that
    point's own distance from the primary, on the assumption a real
    disc's outer edge sits right where the eclipse-timing-derived
    crossing and the stream's own near edge actually meet. Returns None
    if no crossing exists at all for this q/incl (nothing to derive).
    """
    _, point = _shadow_fit_target(sys2, lobe2, phase_a, phase_b)
    if point is None:
        return None
    x, y = point
    return float(np.hypot(x - lobe2.x1, y))


def _run_shadow_lsq_fit(args, system, model, lobe, ntheta, nphi):
    """
    Shadow-mode --lsq_fit: q and/or incl are varied (scipy.optimize.
    least_squares) to make the --phase-list's 1st/3rd shadow-boundary
    curves cross as close as possible to the ballistic accretion
    stream's own near-primary Lubow & Shu flanking line (see
    _shadow_fit_target) -- a purely geometric consistency check between
    the eclipse-timing-derived hot-spot location and the physical stream
    model, with NO photometric/RV data involved at all (contrast
    _run_lsq_fit's own light-curve fit). simulate.py's main() picks
    between the two based on whether "shadow" is a requested output --
    see its own dispatch.

    R_out, if named (see _parse_shadow_fit_names), is never itself
    varied -- it has no effect on the residual above -- but is DERIVED
    from the best-fit q/incl's own crossing point afterward and folded
    into the returned model. Naming ONLY R_out (no q/incl) skips fitting
    entirely: R_out is derived directly from the CURRENT q/incl, a
    one-shot computation, no optimizer involved.

    Returns (system, model) updated to the fitted/derived values --
    unlike _run_lsq_fit, there's no lobe/disc/temp_maps to hand back:
    this fit never needed them (a bare RocheLobe is enough), so the
    caller rebuilds them from the returned system/model itself, same as
    after any other --config/CLI override.
    """
    from scipy.optimize import least_squares

    active_names, derive_R_out = _parse_shadow_fit_names(args.lsq_fit, "--lsq_fit")
    phase_a, phase_b = _shadow_fit_phases(args, "--lsq_fit")

    def apply(x):
        overrides = dict(zip(active_names, x))
        return dataclasses.replace(system, **overrides) if overrides else system

    def lobe_for(sys2):
        return lobe if "q" not in active_names else build_system(sys2, model, ntheta=ntheta, nphi=nphi)[0]

    def residuals(x):
        sys2 = apply(x)
        dist, _ = _shadow_fit_target(sys2, lobe_for(sys2), phase_a, phase_b)
        if dist is None:
            raise ValueError(f"--lsq_fit: the 1st/3rd shadow curves never cross for "
                              f"q={sys2.q:.6g}, incl={sys2.incl:.6g}")
        return np.array([dist])

    if active_names:
        x0 = np.array([getattr(system, n) for n in active_names], dtype=float)
        result = least_squares(residuals, x0)
        sys2 = apply(result.x)
        print(f"\nShadow-mode least-squares fit to {', '.join(active_names)}:")
        _print_lsq_covariance(active_names, result, len(result.fun))
        print(f"\tfinal shadow/stream distance = {result.fun[0]:.6g} (units of a)")
    else:
        sys2 = system
        print("Shadow-mode --lsq_fit: no free parameters (R_out alone) -- "
              "deriving R_out directly from the current q/incl")

    model2 = model
    if derive_R_out:
        r_out = _derive_R_out(sys2, lobe_for(sys2), phase_a, phase_b)
        if r_out is None:
            print("Shadow-mode --lsq_fit: R_out requested but the shadow curves "
                  "don't cross at the fitted q/incl -- R_out left unchanged")
        else:
            print(f"\tR_out = {r_out:.6g} (derived, not fit)")
            model2 = dataclasses.replace(model, R_out=r_out)

    return sys2, model2


def _run_shadow_mcmc_fit(args, system, model, lobe, ntheta, nphi, finish):
    """
    Shadow-mode --mcmc_fit: q and/or incl's posterior given the same
    shadow/stream-crossing distance _run_shadow_lsq_fit's own residual
    computes (Gaussian likelihood on that single distance -- no prior
    beyond the flat/improper default, same convention as _run_mcmc_fit's
    own light-curve fit), via emcee. See _run_shadow_lsq_fit's own
    docstring for R_out's "derived, not fit" handling and the "R_out
    alone" one-shot shortcut, both identical here.

    Returns (system, model) -- same contract as _run_shadow_lsq_fit.
    """
    import emcee
    from multiprocessing.pool import ThreadPool

    active_names, derive_R_out = _parse_shadow_fit_names(args.mcmc_fit, "--mcmc_fit")
    phase_a, phase_b = _shadow_fit_phases(args, "--mcmc_fit")

    def apply(x):
        overrides = dict(zip(active_names, x))
        return dataclasses.replace(system, **overrides) if overrides else system

    def lobe_for(sys2):
        return lobe if "q" not in active_names else build_system(sys2, model, ntheta=ntheta, nphi=nphi)[0]

    def residuals(x):
        sys2 = apply(x)
        dist, _ = _shadow_fit_target(sys2, lobe_for(sys2), phase_a, phase_b)
        if dist is None:
            raise ValueError("shadow curves don't cross")
        return np.array([dist])

    if not active_names:
        sys2 = system
        print("Shadow-mode --mcmc_fit: no free parameters (R_out alone) -- "
              "deriving R_out directly from the current q/incl")
    else:
        ndim = len(active_names)
        nwalkers = args.walkers * ndim

        def log_probability(x):
            try:
                r = residuals(x)
            except Exception:
                return -np.inf
            if not np.all(np.isfinite(r)):
                return -np.inf
            return -0.5 * float(np.sum(r ** 2))

        x0 = np.array([getattr(system, n) for n in active_names], dtype=float)
        rng = np.random.default_rng()
        spread = np.where(x0 != 0.0, np.abs(x0), 1.0) * args.spread
        pos = x0 + spread * rng.standard_normal((nwalkers, ndim))

        pool = ThreadPool(args.workers) if args.workers > 1 else None
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool)
        print(f"--mcmc_fit (shadow mode): {nwalkers} walkers ({args.walkers}/parameter), "
              f"{args.nburn} burn-in + {args.nsample} sample steps"
              + (f", {args.workers} worker threads" if pool is not None else ""))
        try:
            state = sampler.run_mcmc(pos, args.nburn, progress=True)
            sampler.reset()
            sampler.run_mcmc(state, args.nsample, progress=True)
        finally:
            if pool is not None:
                pool.close()
                pool.join()

        chain = sampler.get_chain(flat=True)
        medians = np.median(chain, axis=0)
        p05 = np.percentile(chain, 5, axis=0)
        p95 = np.percentile(chain, 95, axis=0)
        plus_err = p95 - medians
        minus_err = medians - p05
        mean90_err = 0.5 * (plus_err + minus_err)

        print(f"\nShadow-mode MCMC fit to {', '.join(active_names)} "
              f"({nwalkers} walkers x {args.nsample} steps = {len(chain)} samples):")
        for name, med, m9, pl, mi in zip(active_names, medians, mean90_err, plus_err, minus_err):
            print(f"\t{name} = {med:.6g} +/- {m9:.3g}  (+{pl:.3g} / -{mi:.3g})")
        print(f"\tmean acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")

        if args.corner_plot:
            import corner
            fig = corner.corner(chain, labels=active_names, truths=medians, show_titles=True)
            _, title_label = _prefix_and_title_label(args)
            if title_label:
                fig.suptitle(title_label)
            finish(fig, "corner")

        sys2 = apply(medians)

    model2 = model
    if derive_R_out:
        r_out = _derive_R_out(sys2, lobe_for(sys2), phase_a, phase_b)
        if r_out is None:
            print("Shadow-mode --mcmc_fit: R_out requested but the shadow curves "
                  "don't cross at the fitted q/incl -- R_out left unchanged")
        else:
            print(f"\tR_out = {r_out:.6g} (derived, not fit)")
            model2 = dataclasses.replace(model, R_out=r_out)

    return sys2, model2


_RIM_FIT_FIELDS = ("q", "R_out")


def _parse_rim_fit_names(param_str, flag_name):
    """
    Parse a --lsq_fit/--mcmc_fit comma-separated parameter list for
    rim-projection-mode fitting (see _run_rim_lsq_fit/_run_rim_mcmc_fit's
    own docstrings) -- only 'q'/'R_out' are meaningful here: unlike
    shadow-mode's own q/incl/R_out (see _parse_shadow_fit_names -- R_out
    has NO effect on that fit's own midplane residual), R_out DOES move
    the rim-projected shadow curves here (they're evaluated ON the
    disc's own wall, at r=disc.rim(nu)), so both q and R_out are genuine,
    independently meaningful free parameters -- neither is ever merely
    "derived after the fact".
    """
    names = [s.strip() for s in param_str.split(",") if s.strip()]
    if not names:
        raise SystemExit(f"{flag_name} given but no parameter names found")
    bad = [n for n in names if n not in _RIM_FIT_FIELDS]
    if bad:
        raise SystemExit(f"{flag_name}: rim-projection-output fitting only supports "
                          f"q, R_out (got {bad!r})")
    return names


def _rim_fit_target(sys2, lobe2, disc2, phases, n_nu=1000, half_width_deg=15.0):
    """
    The rim-projection-mode fit's core geometric target, at one trial
    (q, R_out) (sys2/lobe2/disc2 already reflect it): how close each of
    `phases`' own rim-projected shadow-boundary curves (plots.
    shadow_boundary_on_rim) comes to being a LOCAL TANGENT to the
    ballistic stream's own Lubow & Shu cross-section ellipse -- rim-
    projected exactly like plots.plot_disc_rim_projection's own overlay
    (same W/cos(incidence) stretch, see that function's own derivation
    comment). `phases` is any number of phases (2, for _run_rim_lsq_fit/
    _run_rim_mcmc_fit's own 1st/3rd-curve tangency target; 4, for
    "fit system"'s combined fit, one residual per given contact phase --
    the SAME per-curve computation either way, just called for more of
    them).

    For one curve, define the ellipse-normalized distance at every one
    of its own points, rho(s,z) = sqrt((s/W_eff)^2 + (z/H)^2) -- exactly
    1 on the ellipse boundary itself, >1 outside it, <1 inside -- and
    take the MINIMUM of rho along the whole curve. That minimum reaching
    exactly 1 is precisely tangency: the curve grazes the ellipse at its
    own single closest point without ever crossing inside it (rho>=1
    everywhere else along the curve, by definition of a minimum); >1
    means the curve passes clear of the ellipse everywhere; <1 means it
    cuts into it somewhere. No "from which side" qualifier is needed --
    unlike an ellipse-vs-z=0-crossing measure restricted to one side,
    this is well-defined and smooth wherever the curve has ANY point
    near the ellipse at all, on either side, which is what makes it
    usable as an optimizer target from an arbitrary starting q/R_out
    (including the common case where the curve currently sits entirely
    on one side of the ellipse, well before any fitting has happened).

    Returns a tuple of len(phases) residuals, each (min rho - 1) for that
    curve, or all-None (one per phase) if the stream doesn't reach this
    disc at all, or an individual entry None if that particular curve
    has no point at all in the +-half_width_deg window.
    """
    from plots import shadow_boundary_on_rim
    from stream import disc_impact_point, impact_incidence, lubow_shu_eps, lubow_shu_stream_size

    incl2 = np.radians(sys2.incl)
    eps = lubow_shu_eps(sys2.T_2, sys2.P_orb, sys2.a_m)
    impact = disc_impact_point(lobe2, disc2, eps=eps)
    if impact is None:
        return (None,) * len(phases)
    nu_imp, cos_incidence = impact_incidence(impact, disc2)
    H_ls, W_ls = lubow_shu_stream_size(impact["r1"], lobe2.q, eps)
    W_eff = W_ls / cos_incidence

    half_width = np.radians(half_width_deg)
    nu_dense = np.linspace(nu_imp - half_width, nu_imp + half_width, n_nu)
    r_dense = disc2.rim(nu_dense)
    drdnu = np.gradient(r_dense, nu_dense)
    ds_dnu = np.sqrt(r_dense ** 2 + drdnu ** 2)
    s_cum = np.concatenate([[0.0],
                             np.cumsum(0.5 * (ds_dnu[:-1] + ds_dnu[1:]) * np.diff(nu_dense))])
    s_dense = s_cum - np.interp(nu_imp, nu_dense, s_cum)

    def tangent_residual(phase):
        z = shadow_boundary_on_rim(disc2, lobe2, phase, incl2, nu_dense)
        rho = np.hypot(s_dense / W_eff, z / H_ls)  # NaN wherever z is (mirrors it exactly)
        finite = np.isfinite(rho)
        if not finite.any():
            return None
        i = int(np.nanargmin(rho))
        # sub-sample parabolic refinement: the raw discrete min jumps from
        # one nu_dense sample to its neighbor as q/R_out vary continuously
        # (whichever is closer keeps swapping), which puts a small kink in
        # an otherwise-smooth residual right at each swap -- exactly where
        # a gradient-based optimizer's finite-difference derivative gets
        # noisy/wrong. Fitting a parabola through the minimum and its two
        # immediate neighbors IN nu_dense (only when both are themselves
        # finite -- neighboring INDICES, not just nearby surviving values,
        # so a NaN gap never gets bridged) and using ITS OWN vertex instead
        # of the raw sample value removes that discreteness almost
        # entirely, at effectively no cost.
        if 0 < i < len(rho) - 1 and finite[i - 1] and finite[i + 1]:
            y0, y1, y2 = rho[i - 1], rho[i], rho[i + 1]
            denom = y0 - 2.0 * y1 + y2
            if abs(denom) > 1e-14:
                frac = 0.5 * (y0 - y2) / denom  # in [-0.5, 0.5] for a genuine local min
                if abs(frac) <= 0.5:
                    rho_min = y1 - 0.25 * (y0 - y2) * frac
                    return float(rho_min) - 1.0
        return float(rho[i]) - 1.0

    return tuple(tangent_residual(ph) for ph in phases)


def _rim_fit_lobe_disc(sys2, model2, active_names, lobe, disc, ntheta, nphi):
    """
    (lobe, disc) for one rim-mode fit trial: rebuilds only what the
    trial's active_names actually varies -- the disc alone (a plain
    constructor call, no iterative solve) if only R_out is active, both
    (via params.build_system) if q is active too (rebuilding the lobe on
    every trial can't be avoided once q itself is varying), or neither
    (the given lobe/disc reused as-is) if -- not expected in practice,
    since _parse_rim_fit_names requires at least one name -- somehow
    neither is.
    """
    if "q" in active_names:
        return build_system(sys2, model2, ntheta=ntheta, nphi=nphi)[:2]
    if "R_out" in active_names:
        from disc import Disc
        disc2 = Disc(lobe.x1, model2.R_out, e_disc=model2.e_d, omega_disc=model2.omega_d_rad,
                      r_in=model2.R_in, opening_angle=model2.alpha_d_rad)
        return lobe, disc2
    return lobe, disc


def _rim_fit_bounds(active_names, model):
    """
    (lower, upper) bound arrays, same order as active_names, for
    scipy.optimize.least_squares' own bounds= -- q and R_out both have a
    genuine "must stay physical" range (q<=0 makes roche.RocheLobe's own
    L1 solve fail outright -- see roche.lagrange1's brentq bracket;
    R_out must stay outside model.R_in, and can't sensibly approach the
    binary separation itself), and this fit's own residuals (see
    _rim_fit_target) can be large enough, this far from the fitted
    system's own real values, that an unbounded trial step tries a
    value outside that range and crashes rather than backing off --
    _run_rim_mcmc_fit doesn't need this (bad trials there are already
    caught and rejected via log_probability's own try/except), but
    least_squares has no equivalent built in.
    """
    lo, hi = [], []
    for n in active_names:
        if n == "q":
            lo.append(1e-3)
            hi.append(20.0)
        else:  # "R_out"
            lo.append(model.R_in * 1.01)
            hi.append(0.9)
    return lo, hi


def _run_rim_lsq_fit(args, system, model, lobe, disc, ntheta, nphi):
    """
    Rim-projection-mode --lsq_fit: q and/or R_out are varied (scipy.
    optimize.least_squares) to make the --phase-list's 1st/3rd rim-
    projected shadow-boundary curves each a local tangent to the
    stream's own Lubow & Shu cross-section ellipse (see
    _rim_fit_target) -- simulate.py's main() picks this over the usual
    light-curve fit whenever "rim projection" is the requested output
    (see its own dispatch; "shadow" takes priority if both are
    requested at once).

    Returns (system, model) updated to the fitted values -- like
    _run_shadow_lsq_fit, no lobe/disc/temp_maps to hand back: the
    caller rebuilds them from the returned system/model itself.
    """
    from scipy.optimize import least_squares

    active_names = _parse_rim_fit_names(args.lsq_fit, "--lsq_fit")
    phase_a, phase_b = _shadow_fit_phases(args, "--lsq_fit", "rim-projection-output")

    def apply(x):
        sys_over = {n: v for n, v in zip(active_names, x) if n == "q"}
        model_over = {n: v for n, v in zip(active_names, x) if n == "R_out"}
        sys2 = dataclasses.replace(system, **sys_over) if sys_over else system
        model2 = dataclasses.replace(model, **model_over) if model_over else model
        return sys2, model2

    def residuals(x):
        sys2, model2 = apply(x)
        lobe2, disc2 = _rim_fit_lobe_disc(sys2, model2, active_names, lobe, disc, ntheta, nphi)
        res_a, res_b = _rim_fit_target(sys2, lobe2, disc2, (phase_a, phase_b))
        if res_a is None or res_b is None:
            raise ValueError(f"--lsq_fit: the stream/rim geometry gives nothing to "
                              f"compare for q={sys2.q:.6g}, R_out={model2.R_out:.6g}")
        return np.array([res_a, res_b])

    x0 = np.array([getattr(system, n) if n == "q" else getattr(model, n) for n in active_names],
                   dtype=float)
    # an explicit, larger-than-default relative finite-difference step:
    # this residual (see _rim_fit_target) is smooth at realistic
    # parameter scales (its own grid scan varies meaningfully over
    # delta_q/delta_R_out ~ 0.01-0.05), but scipy's own default step
    # (~1e-8 relative) is far too small to resolve that on the fixed
    # n_nu=1000 rim sampling grid the residual is built from -- the
    # finite-difference Jacobian comes back ~0 there, and the optimizer
    # reports immediate (false) convergence right at x0.
    result = least_squares(residuals, x0, bounds=_rim_fit_bounds(active_names, model),
                            diff_step=0.02)
    sys2, model2 = apply(result.x)
    print(f"\nRim-projection-mode least-squares fit to {', '.join(active_names)}:")
    _print_lsq_covariance(active_names, result, len(result.fun))
    print(f"\tfinal (min ellipse-normalized distance - 1) for 1st/3rd shadow curves = "
          f"{result.fun[0]:.6g}, {result.fun[1]:.6g}")

    return sys2, model2


def _run_rim_mcmc_fit(args, system, model, lobe, disc, ntheta, nphi, finish):
    """
    Rim-projection-mode --mcmc_fit: q and/or R_out's posterior given the
    same two-curve touch residual _run_rim_lsq_fit's own residual
    computes (Gaussian likelihood on sum of squares -- no prior beyond
    the flat/improper default, same convention as _run_mcmc_fit's own
    light-curve fit and _run_shadow_mcmc_fit's own shadow-mode fit), via
    emcee.

    Returns (system, model) -- same contract as _run_rim_lsq_fit.
    """
    import emcee
    from multiprocessing.pool import ThreadPool

    active_names = _parse_rim_fit_names(args.mcmc_fit, "--mcmc_fit")
    phase_a, phase_b = _shadow_fit_phases(args, "--mcmc_fit", "rim-projection-output")

    def apply(x):
        sys_over = {n: v for n, v in zip(active_names, x) if n == "q"}
        model_over = {n: v for n, v in zip(active_names, x) if n == "R_out"}
        sys2 = dataclasses.replace(system, **sys_over) if sys_over else system
        model2 = dataclasses.replace(model, **model_over) if model_over else model
        return sys2, model2

    def residuals(x):
        sys2, model2 = apply(x)
        lobe2, disc2 = _rim_fit_lobe_disc(sys2, model2, active_names, lobe, disc, ntheta, nphi)
        res_a, res_b = _rim_fit_target(sys2, lobe2, disc2, (phase_a, phase_b))
        if res_a is None or res_b is None:
            raise ValueError("nothing to compare for this trial q/R_out")
        return np.array([res_a, res_b])

    def log_probability(x):
        try:
            r = residuals(x)
        except Exception:
            return -np.inf
        if not np.all(np.isfinite(r)):
            return -np.inf
        return -0.5 * float(np.sum(r ** 2))

    ndim = len(active_names)
    nwalkers = args.walkers * ndim
    x0 = np.array([getattr(system, n) if n == "q" else getattr(model, n) for n in active_names],
                   dtype=float)
    rng = np.random.default_rng()
    spread = np.where(x0 != 0.0, np.abs(x0), 1.0) * args.spread
    pos = x0 + spread * rng.standard_normal((nwalkers, ndim))

    pool = ThreadPool(args.workers) if args.workers > 1 else None
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool)
    print(f"--mcmc_fit (rim-projection mode): {nwalkers} walkers ({args.walkers}/parameter), "
          f"{args.nburn} burn-in + {args.nsample} sample steps"
          + (f", {args.workers} worker threads" if pool is not None else ""))
    try:
        state = sampler.run_mcmc(pos, args.nburn, progress=True)
        sampler.reset()
        sampler.run_mcmc(state, args.nsample, progress=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    chain = sampler.get_chain(flat=True)
    medians = np.median(chain, axis=0)
    p05 = np.percentile(chain, 5, axis=0)
    p95 = np.percentile(chain, 95, axis=0)
    plus_err = p95 - medians
    minus_err = medians - p05
    mean90_err = 0.5 * (plus_err + minus_err)

    print(f"\nRim-projection-mode MCMC fit to {', '.join(active_names)} "
          f"({nwalkers} walkers x {args.nsample} steps = {len(chain)} samples):")
    for name, med, m9, pl, mi in zip(active_names, medians, mean90_err, plus_err, minus_err):
        print(f"\t{name} = {med:.6g} +/- {m9:.3g}  (+{pl:.3g} / -{mi:.3g})")
    print(f"\tmean acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")

    if args.corner_plot:
        import corner
        fig = corner.corner(chain, labels=active_names, truths=medians, show_titles=True)
        _, title_label = _prefix_and_title_label(args)
        if title_label:
            fig.suptitle(title_label)
        finish(fig, "corner")

    return apply(medians)


_PRIMARY_FIT_FIELDS = ("q", "incl", "R_1")


def _parse_primary_fit_names(param_str, flag_name):
    """
    Parse a --lsq_fit/--mcmc_fit comma-separated parameter list for
    "primary"-output fitting (see _run_primary_lsq_fit/_run_primary_mcmc_fit's
    own docstrings): 1 or 2 of 'q'/'incl'/'R_1', in any combination --
    the 6 possibilities are [q], [incl], [R_1], [q,incl], [q,R_1],
    [incl,R_1] -- each well-determined against the 4 residuals (one per
    given contact phase), holding whichever of q/incl/R_1 wasn't named
    fixed at its current --q/--incl/--R_1 value. Not 3 at once (q,incl,R_1
    together): that leaves only 1 residual's worth of information beyond
    the 3 unknowns, too little to pin all three down at once.
    """
    names = [s.strip() for s in param_str.split(",") if s.strip()]
    if not names:
        raise SystemExit(f"{flag_name} given but no parameter names found")
    bad = [n for n in names if n not in _PRIMARY_FIT_FIELDS]
    if bad:
        raise SystemExit(f"{flag_name}: primary-output fitting only supports "
                          f"q, incl, R_1 (got {bad!r})")
    if len(names) > 2:
        raise SystemExit(f"{flag_name}: primary-output fitting takes at most 2 of "
                          f"q, incl, R_1 at a time (got {names!r}) -- only 4 residuals "
                          f"(one per contact phase) are available to constrain them")
    return names


def _primary_fit_phases(args, flag_name):
    """
    The --phase-list phases "primary"-mode fitting needs: all 4 (ingress-
    before, ingress-after, egress-before, egress-after) -- every one of
    the primary's own eclipse contact phases feeds its own residual (see
    _primary_fit_target), unlike shadow-/rim-mode fitting's own
    _shadow_fit_phases (which only ever needs 2 of the 4).
    """
    if args.phase_list is None or len(args.phase_list) != 4:
        raise SystemExit(f"{flag_name}: primary-output fitting needs --phase-list with "
                          f"exactly 4 phases (ingress-before, ingress-after, "
                          f"egress-before, egress-after) -- the primary's own eclipse "
                          f"contact phases")
    return tuple(args.phase_list)


def _primary_fit_distances(sys2, lobe2, phases):
    """
    The 4 closest-approach distances this fit's residual AND its derived
    R_1 are both built from: for each given eclipse contact phase, the
    closest distance from the primary's own phase-0 sky position to that
    phase's secondary-limb curve (plots.lobe_outline), shifted onto the
    phase-0 sky frame exactly as the "primary" output's own display
    overlay does (re-anchored by the primary's own phase-dependent
    apparent sky-position drift, see simulate.py's own --outputs primary
    block). Does NOT reference sys2.R_1 at all -- these are purely
    geometric, from the secondary's own limbs and the primary's own
    (q-dependent) position, nothing else.

    Returns an ndarray of 4 distances (units of a), one per given phase,
    in the same order.
    """
    incl_rad = np.radians(sys2.incl)
    primary_pt = np.array([[lobe2.x1, 0.0, 0.0]])
    from eclipse import project
    Xprim0, Yprim0, _ = project(primary_pt, 0.0, incl_rad)
    cx, cy = float(Xprim0[0]), float(Yprim0[0])
    return np.array([_primary_fit_distance_at_phase(lobe2, incl_rad, primary_pt, cx, cy, phase)
                      for phase in phases])


def _primary_fit_distance_at_phase(lobe2, incl_rad, primary_pt, cx, cy, phase):
    """
    _primary_fit_distances' own per-phase computation (the closest
    distance from the primary's own already-computed phase-0 sky
    position (cx,cy) to `phase`'s own shifted secondary-limb curve),
    factored out so it can also be called at phase +- a small step by
    _primary_fit_distance_sigmas (--phase_error's own error-propagation
    derivative) without duplicating this logic.
    """
    from plots import lobe_outline
    from eclipse import project

    Xsec, Ysec = lobe_outline(lobe2, phase, incl_rad)
    Xp, Yp, _ = project(primary_pt, phase, incl_rad)
    dx, dy = cx - float(Xp[0]), cy - float(Yp[0])
    dist = np.hypot(Xsec + dx - cx, Ysec + dy - cy)
    return float(dist.min())


_PRIMARY_FIT_PHASE_DERIV_STEP = 1e-5  # central-difference step [cycles] for d(distance)/d(phase)


def _primary_fit_distance_sigmas(sys2, lobe2, phases, phase_error):
    """
    Per-phase distance uncertainty [units of a] implied by --phase_error
    [cycles]: sigma_distance = |d(distance)/d(phase)| * phase_error, the
    standard linear error-propagation estimate (valid as long as
    phase_error is small compared to the local curvature scale -- the
    same small-timing-error regime classical eclipse-contact-phase
    analysis already assumes). The derivative itself is a small central
    difference in phase (_PRIMARY_FIT_PHASE_DERIV_STEP, independent of
    phase_error itself -- a numerical-differentiation step, not the
    actual timing uncertainty), reusing _primary_fit_distance_at_phase.
    Clamped away from exactly 0 (a perfectly flat local slope would
    otherwise blow up that phase's own residual under the 1/sigma
    weighting in _primary_fit_target) at a tiny floor.

    phase_error: an ndarray of len(phases) values [cycles], already
    resolved (see _resolve_phase_error) against exactly this many
    phases -- NOT a raw --phase_error value (which may be a single
    number meant for every phase alike).

    Returns an ndarray of len(phases) sigmas, one per given phase, in
    the same order.
    """
    incl_rad = np.radians(sys2.incl)
    primary_pt = np.array([[lobe2.x1, 0.0, 0.0]])
    from eclipse import project
    Xprim0, Yprim0, _ = project(primary_pt, 0.0, incl_rad)
    cx, cy = float(Xprim0[0]), float(Yprim0[0])

    step = _PRIMARY_FIT_PHASE_DERIV_STEP
    sigmas = []
    for phase, sigma_phase in zip(phases, phase_error):
        d_hi = _primary_fit_distance_at_phase(lobe2, incl_rad, primary_pt, cx, cy, phase + step)
        d_lo = _primary_fit_distance_at_phase(lobe2, incl_rad, primary_pt, cx, cy, phase - step)
        slope = abs(d_hi - d_lo) / (2.0 * step)
        sigmas.append(max(slope * sigma_phase, 1e-12))
    return np.array(sigmas)


def _primary_fit_target(sys2, lobe2, phases, phase_error=None):
    """
    The "primary"-mode fit's core geometric target, at one trial (q or
    incl, AND R_1 -- sys2/lobe2 already reflect all of it): each of the
    4 given eclipse contact phases' own closest-approach distance
    (_primary_fit_distances) minus sys2.R_1 -- the distance from the
    trial CIRCLE's own EDGE (not its center) to that phase's secondary
    limb, exactly 0 at tangency, positive if the circle sits clear of
    that limb, negative if it already overlaps it. Least-squares drives
    all 4 towards 0 at once, i.e. finds the (q or incl, R_1) pair whose
    circle best "fills the box" the 4 limbs bound.

    R_1 is read directly from sys2 -- a genuine trial value the fit
    varies jointly with q/incl, not a fixed input.

    phase_error: an ndarray of len(phases) values [cycles], already
    resolved against exactly this many phases (see _resolve_phase_error
    -- NOT a raw --phase_error value, which may be a single number
    meant for every phase alike), if given: each residual is divided by
    that phase's own propagated distance uncertainty (_primary_fit_
    distance_sigmas), turning the raw (distance - R_1) residuals into
    properly chi^2-weighted ones -- sum(residual**2) is then a real
    chi^2, not merely an unweighted sum of squares, and (via
    _run_primary_mcmc_fit's own log_probability = -0.5*sum(residual**2))
    a genuine Gaussian log-likelihood too. None (default) leaves
    residuals in raw distance units (units of a), unweighted, same as
    before --phase_error existed.

    Returns an ndarray of 4 residuals, one per given phase, in the same
    order.
    """
    residuals = _primary_fit_distances(sys2, lobe2, phases) - sys2.R_1
    if phase_error is not None:
        residuals = residuals / _primary_fit_distance_sigmas(sys2, lobe2, phases, phase_error)
    return residuals


def _primary_fit_bounds(active_names):
    """
    (lower, upper) bound arrays, same order as active_names, for
    scipy.optimize.least_squares' own bounds= -- same reasoning as
    _rim_fit_bounds: q<=0 makes roche.RocheLobe's own L1 solve fail
    outright, and an unbounded trial step can wander there; incl is
    clamped to a physically sane (0, 90] deg range (90 = edge-on, the
    conventional maximum in this codebase's own --incl convention); R_1
    to a generous (0, 0.5] range (well beyond any physically plausible
    stellar radius relative to the binary separation, for typical CV
    mass ratios) -- wide enough that it practically never binds, just
    keeping a stray trial step away from R_1<=0 (meaningless) or
    absurdly large. _run_primary_mcmc_fit doesn't need this (bad trials
    are already caught and rejected via its own log_probability
    try/except).
    """
    lo, hi = [], []
    for n in active_names:
        if n == "q":
            lo.append(1e-3)
            hi.append(20.0)
        elif n == "incl":
            lo.append(1e-3)
            hi.append(90.0)
        else:  # "R_1"
            lo.append(1e-4)
            hi.append(0.5)
    return lo, hi


def _run_primary_lsq_fit(args, system, model, lobe, ntheta, nphi):
    """
    "primary"-output --lsq_fit: 1-2 of q/incl/R_1 are varied jointly
    (scipy.optimize.least_squares, see _parse_primary_fit_names for which
    combinations are allowed) so the primary's own circle -- center fixed
    by q/incl, radius R_1 -- comes as close as possible to touching all 4
    of the --phase-list's eclipse-contact-phase secondary limbs from the
    inside (_primary_fit_target's own edge-of-circle-to-curve residual,
    which reads sys2.R_1 as a genuine trial value -- fit or not).
    Whichever of q/incl/R_1 isn't named is held fixed at its current
    --q/--incl/--R_1 value. simulate.py's main() picks this over the
    usual light-curve fit whenever "primary" is the requested output (see
    its own dispatch; "shadow"/"rim projection" take priority if either
    is also requested with a fit flag).

    Returns (system, model) updated to the fitted values (model unchanged
    -- this fit never touches it). No lobe/disc/temp_maps to hand back
    either, same contract as _run_rim_lsq_fit: the caller rebuilds them
    from the returned system itself.
    """
    from scipy.optimize import least_squares

    active_names = _parse_primary_fit_names(args.lsq_fit, "--lsq_fit")
    phases = _primary_fit_phases(args, "--lsq_fit")
    phase_error = _resolve_phase_error(args.phase_error, len(phases), "--lsq_fit")

    def apply(x):
        overrides = dict(zip(active_names, x))
        return dataclasses.replace(system, **overrides) if overrides else system

    def lobe_for(sys2):
        return lobe if "q" not in active_names else build_system(sys2, model, ntheta=ntheta, nphi=nphi)[0]

    def residuals(x):
        sys2 = apply(x)
        return _primary_fit_target(sys2, lobe_for(sys2), phases, phase_error=phase_error)

    x0 = np.array([getattr(system, n) for n in active_names], dtype=float)
    # explicit, larger-than-default finite-difference step -- same reason
    # as _run_rim_lsq_fit's own diff_step=0.02: the residual is smooth at
    # realistic parameter scales, but scipy's tiny default relative step
    # under-resolves it against lobe_outline's own (coarser than rim-
    # mode's 1000-point grid) hull-vertex sampling.
    result = least_squares(residuals, x0, bounds=_primary_fit_bounds(active_names),
                            diff_step=0.02)
    sys2 = apply(result.x)
    print(f"\nPrimary-mode least-squares fit to {', '.join(active_names)}:")
    _print_lsq_covariance(active_names, result, len(result.fun))
    if "R_1" not in active_names:
        print(f"\tR_1/a = {sys2.R_1:.6g} (held fixed, not fit)")
    else:
        cov, _, _, _ = _lsq_covariance(active_names, result, len(result.fun))

        def sys2_lobe_of(x):
            sys2x = apply(x)
            return sys2x, lobe_for(sys2x)

        _print_nauenberg_derived(sys2_lobe_of, result.x, cov, full=False)
    if args.phase_error is not None:
        print("\tfinal (weighted, dimensionless) residual per contact phase: "
              + ", ".join(f"{v:.6g}" for v in result.fun))
    else:
        print("\tfinal (closest distance - R_1) per contact phase (units of a, unweighted "
              "-- pass --phase_error for a real chi^2): "
              + ", ".join(f"{v:.6g}" for v in result.fun))

    return sys2, model


def _run_primary_mcmc_fit(args, system, model, lobe, ntheta, nphi, finish):
    """
    "primary"-output --mcmc_fit: the 1-2 active q/incl/R_1 parameters'
    joint posterior given the same edge-of-circle-to-curve residual
    _run_primary_lsq_fit's own residual computes (Gaussian likelihood on
    sum of squares -- no prior beyond the flat/improper default, same
    convention as this codebase's other MCMC fits), via emcee. With
    --phase_error given, that sum of squares is a real chi^2 (see
    _primary_fit_target), so this posterior is a properly calibrated one,
    not just a formal exploration of an arbitrary unweighted landscape.

    Returns (system, model) -- same contract as _run_primary_lsq_fit.
    """
    import emcee
    from multiprocessing.pool import ThreadPool

    active_names = _parse_primary_fit_names(args.mcmc_fit, "--mcmc_fit")
    phases = _primary_fit_phases(args, "--mcmc_fit")
    phase_error = _resolve_phase_error(args.phase_error, len(phases), "--mcmc_fit")

    def apply(x):
        overrides = dict(zip(active_names, x))
        return dataclasses.replace(system, **overrides) if overrides else system

    def lobe_for(sys2):
        return lobe if "q" not in active_names else build_system(sys2, model, ntheta=ntheta, nphi=nphi)[0]

    def residuals(x):
        sys2 = apply(x)
        return _primary_fit_target(sys2, lobe_for(sys2), phases, phase_error=phase_error)

    def log_probability(x):
        try:
            r = residuals(x)
        except Exception:
            return -np.inf
        if not np.all(np.isfinite(r)):
            return -np.inf
        return -0.5 * float(np.sum(r ** 2))

    ndim = len(active_names)
    nwalkers = args.walkers * ndim
    x0 = np.array([getattr(system, n) for n in active_names], dtype=float)
    rng = np.random.default_rng()
    spread = np.where(x0 != 0.0, np.abs(x0), 1.0) * args.spread
    pos = x0 + spread * rng.standard_normal((nwalkers, ndim))

    pool = ThreadPool(args.workers) if args.workers > 1 else None
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool)
    print(f"--mcmc_fit (primary mode): {nwalkers} walkers ({args.walkers}/parameter), "
          f"{args.nburn} burn-in + {args.nsample} sample steps"
          + (f", {args.workers} worker threads" if pool is not None else ""))
    try:
        state = sampler.run_mcmc(pos, args.nburn, progress=True)
        sampler.reset()
        sampler.run_mcmc(state, args.nsample, progress=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    chain = sampler.get_chain(flat=True)
    medians = np.median(chain, axis=0)
    p05 = np.percentile(chain, 5, axis=0)
    p95 = np.percentile(chain, 95, axis=0)
    plus_err = p95 - medians
    minus_err = medians - p05
    mean90_err = 0.5 * (plus_err + minus_err)

    print(f"\nPrimary-mode MCMC fit to {', '.join(active_names)} "
          f"({nwalkers} walkers x {args.nsample} steps = {len(chain)} samples):")
    for name, med, m9, pl, mi in zip(active_names, medians, mean90_err, plus_err, minus_err):
        print(f"\t{name} = {med:.6g} +/- {m9:.3g}  (+{pl:.3g} / -{mi:.3g})")
    print(f"\tmean acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")

    if args.corner_plot:
        import corner
        fig = corner.corner(chain, labels=active_names, truths=medians, show_titles=True)
        _, title_label = _prefix_and_title_label(args)
        if title_label:
            fig.suptitle(title_label)
        finish(fig, "corner")

    sys2 = apply(medians)
    if "R_1" not in active_names:
        print(f"\tR_1/a = {sys2.R_1:.6g} (held fixed, not fit)")
    else:
        # no closed-form covariance from an MCMC run -- the posterior
        # chain's own sample covariance is the natural substitute (same
        # "covariance of whatever x0 is" contract _propagate_scalar
        # documents), evaluated at the median posterior point.
        cov = np.cov(chain, rowvar=False).reshape(ndim, ndim)

        def sys2_lobe_of(x):
            sys2x = apply(x)
            return sys2x, lobe_for(sys2x)

        _print_nauenberg_derived(sys2_lobe_of, medians, cov, full=False)
    if args.phase_error is not None:
        chi2 = float(np.sum(residuals(medians) ** 2))
        dof = len(phases) - len(active_names)
        print(f"\tchi^2 at the median fit = {chi2:.6g} ({len(phases)} phases, "
              f"{len(active_names)} fit parameters, {dof} dof -> reduced chi^2 = "
              f"{chi2 / dof if dof > 0 else float('nan'):.6g})")

    return sys2, model


_SYSTEM_FIT_FIELDS = ("q", "incl", "R_1", "R_out")


def _parse_system_fit_names(param_str, flag_name):
    """
    Parse a --lsq_fit/--mcmc_fit comma-separated parameter list for
    "fit system" -- any 1-4 of 'q'/'incl'/'R_1'/'R_out', any combination:
    unlike "primary"-output fitting's own restrictions (see _parse_
    primary_fit_names), "fit system" always has 8 residuals available
    (4 from the primary's own eclipse contacts, 4 from the bright spot's
    -- see _system_fit_phases), comfortably more than the at-most-4
    unknowns here, so there's no equivalent "too little information"
    combination to refuse.
    """
    names = [s.strip() for s in param_str.split(",") if s.strip()]
    if not names:
        raise SystemExit(f"{flag_name} given but no parameter names found")
    bad = [n for n in names if n not in _SYSTEM_FIT_FIELDS]
    if bad:
        raise SystemExit(f"{flag_name}: fit-system fitting only supports "
                          f"q, incl, R_1, R_out (got {bad!r})")
    return names


def _system_fit_phases(args, flag_name):
    """
    The --phase-list phases "fit system" needs: exactly 8 -- the first 4
    the primary's own eclipse contact phases (ingress-before, ingress-
    after, egress-before, egress-after -- same convention as the
    "primary" output's own --phase-list), the last 4 the bright spot's
    own (same convention as "rim projection"'s). Returns (phases_primary,
    phases_rim), each a 4-tuple, in that order.
    """
    if args.phase_list is None or len(args.phase_list) != 8:
        raise SystemExit(f"{flag_name}: fit-system fitting needs --phase-list with exactly "
                          f"8 phases -- the primary's own 4 eclipse contact phases (ingress-"
                          f"before, ingress-after, egress-before, egress-after), followed by "
                          f"the bright spot's own 4 (same convention as the 'primary'/'rim "
                          f"projection' outputs' own --phase-list, back to back)")
    return tuple(args.phase_list[:4]), tuple(args.phase_list[4:])


_RIM_FIT_PHASE_DERIV_STEP = 1e-5  # central-difference step [cycles] for d(rho-1)/d(phase)


# "fit system"'s own 4 bright-spot phases can legitimately span much
# more of the rim than _run_rim_lsq_fit/_run_rim_mcmc_fit's own 2-phase
# (1st/3rd curve) tangency fit ever needed to look at -- _rim_fit_target's
# own half_width_deg=15.0 default window (tuned for that narrower,
# closely-spaced use) can leave one or more of the 4 curves with no
# point in range at all (confirmed directly: phases 0.1 apart can need
# 45 deg to all resolve). "Fit system"'s own calls use this wider
# default instead; the original 2-phase callers are untouched.
_SYSTEM_FIT_RIM_HALF_WIDTH_DEG = 45.0


def _rim_fit_sigmas(sys2, lobe2, disc2, phases, phase_error,
                     half_width_deg=_SYSTEM_FIT_RIM_HALF_WIDTH_DEG):
    """
    Per-phase sigma [dimensionless, same rho-1 units as _rim_fit_target's
    own residual] implied by --phase_error [cycles] -- the rim-mode
    analogue of _primary_fit_distance_sigmas, same linear error-
    propagation idea (sigma = |d(residual)/d(phase)| * phase_error, the
    derivative a small central difference in phase independent of
    phase_error itself, reusing _rim_fit_target at phase +- that step)
    applied to a differently-scaled residual. NaN wherever either
    +-step trial has nothing to compare (see _rim_fit_target); clamped
    away from exactly 0 for the same reason _primary_fit_distance_sigmas
    is (a flat local slope would otherwise blow up that phase's own
    residual under 1/sigma weighting).

    phase_error: an ndarray of len(phases) values [cycles], already
    resolved against exactly this many phases (see _resolve_phase_error
    -- NOT a raw --phase_error value, which may be a single number
    meant for every phase alike).
    """
    step = _RIM_FIT_PHASE_DERIV_STEP
    res_hi = _rim_fit_target(sys2, lobe2, disc2, tuple(ph + step for ph in phases),
                              half_width_deg=half_width_deg)
    res_lo = _rim_fit_target(sys2, lobe2, disc2, tuple(ph - step for ph in phases),
                              half_width_deg=half_width_deg)
    sigmas = []
    for hi, lo, sigma_phase in zip(res_hi, res_lo, phase_error):
        if hi is None or lo is None:
            sigmas.append(np.nan)
        else:
            slope = abs(hi - lo) / (2.0 * step)
            sigmas.append(max(slope * sigma_phase, 1e-12))
    return np.array(sigmas)


def _system_fit_residuals(sys2, lobe2, disc2, phases_primary, phases_rim, phase_error=None,
                           half_width_deg=_SYSTEM_FIT_RIM_HALF_WIDTH_DEG):
    """
    "fit system"'s own combined residual vector at one trial (sys2/
    lobe2/disc2 already reflect it): _primary_fit_target's own 4
    primary-eclipse residuals (edge-of-circle-to-limb distance, minus
    R_1), followed by _rim_fit_target's own 4 bright-spot residuals
    (ellipse-normalized rho-1, tangent-to-stream-cross-section) -- both
    the SAME "closest distance from a projected limb to a fixed
    reference shape" idea (see each target's own docstring), just each
    in its own natural units. With phase_error given, BOTH halves are
    divided by their own propagated per-phase sigma (_primary_fit_
    distance_sigmas / _rim_fit_sigmas), turning the combined 8-residual
    vector into one coherent chi^2 regardless of those differing native
    units -- without it, the two halves are simply concatenated as-is
    (an unweighted combination, same caveat as every other unweighted
    geometric fit in this module).

    phase_error: an ndarray of 8 values [cycles], already resolved
    against all 8 combined phases (see _resolve_phase_error -- NOT a
    raw --phase_error value, which may be a single number meant for
    every phase alike), or None -- the first 4 apply to phases_primary,
    the last 4 to phases_rim (split here, not by the caller).

    Returns an 8-element ndarray, primary's own 4 residuals first, then
    rim-projection's own 4, in the given phase order. Raises ValueError
    if the bright-spot half has nothing to compare (the stream never
    reaches this disc, or some curve entirely misses its own +-window,
    even at this wider half_width_deg) -- same "can't proceed with this
    trial" signal _run_rim_lsq_fit's own residuals() raises.
    """
    if phase_error is not None:
        phase_error_primary, phase_error_rim = phase_error[:4], phase_error[4:]
    else:
        phase_error_primary = phase_error_rim = None
    prim_res = np.array(_primary_fit_target(sys2, lobe2, phases_primary,
                                              phase_error=phase_error_primary))
    rim_res = _rim_fit_target(sys2, lobe2, disc2, phases_rim, half_width_deg=half_width_deg)
    if any(r is None for r in rim_res):
        raise ValueError(f"fit-system: the stream/rim geometry gives nothing to compare "
                          f"for this trial (q={sys2.q:.6g})")
    rim_res = np.array(rim_res)
    if phase_error_rim is not None:
        rim_res = rim_res / _rim_fit_sigmas(sys2, lobe2, disc2, phases_rim, phase_error_rim,
                                             half_width_deg=half_width_deg)
    return np.concatenate([prim_res, rim_res])


def _system_fit_bounds(active_names, model):
    """
    (lower, upper) bound arrays, same order as active_names, for
    scipy.optimize.least_squares' own bounds= -- the union of _primary_
    fit_bounds' own q/incl/R_1 ranges and _rim_fit_bounds' own q/R_out
    range (q appears in both with the same range, so no conflict).
    _run_system_mcmc_fit doesn't need this (bad trials are already
    caught and rejected via its own log_probability try/except).
    """
    lo, hi = [], []
    for n in active_names:
        if n == "q":
            lo.append(1e-3)
            hi.append(20.0)
        elif n == "incl":
            lo.append(1e-3)
            hi.append(90.0)
        elif n == "R_1":
            lo.append(1e-4)
            hi.append(0.5)
        else:  # "R_out"
            lo.append(model.R_in * 1.01)
            hi.append(0.9)
    return lo, hi


def _run_system_lsq_fit(args, system, model, lobe, disc, ntheta, nphi):
    """
    "fit system" --lsq_fit: 1-4 of q/incl/R_1/R_out are varied jointly
    (scipy.optimize.least_squares) against the COMBINED 8-residual
    target _system_fit_residuals builds from both the "primary" output's
    own technique (4 residuals, the primary's own eclipse contacts) and
    the "rim projection" output's own (4 more, the bright spot's) --
    the grand fit using both eclipses' own geometry at once, reusing
    each mode's own machinery unchanged rather than a separate
    implementation. Whichever of q/incl/R_1/R_out isn't named is held
    fixed at its current --q/--incl/--R_1/--R_out value. simulate.py's
    main() picks this over the usual light-curve fit whenever "fit
    system" is the requested output (see its own dispatch; it takes
    PRIORITY over shadow/rim-projection/primary-mode fitting if more
    than one is somehow requested together with a fit flag, being the
    most specific/comprehensive of the four).

    Returns (system, model) updated to the fitted values. No lobe/disc/
    temp_maps to hand back either, same contract as _run_rim_lsq_fit:
    the caller rebuilds them from the returned system/model itself.
    """
    from scipy.optimize import least_squares

    active_names = _parse_system_fit_names(args.lsq_fit, "--lsq_fit")
    phases_primary, phases_rim = _system_fit_phases(args, "--lsq_fit")
    phase_error = _resolve_phase_error(args.phase_error, 8, "--lsq_fit")

    def apply(x):
        sys_over = {n: v for n, v in zip(active_names, x) if n in ("q", "incl", "R_1")}
        model_over = {n: v for n, v in zip(active_names, x) if n == "R_out"}
        sys2 = dataclasses.replace(system, **sys_over) if sys_over else system
        model2 = dataclasses.replace(model, **model_over) if model_over else model
        return sys2, model2

    def lobe_disc_for(sys2, model2):
        return _rim_fit_lobe_disc(sys2, model2, active_names, lobe, disc, ntheta, nphi)

    def residuals(x):
        sys2, model2 = apply(x)
        lobe2, disc2 = lobe_disc_for(sys2, model2)
        return _system_fit_residuals(sys2, lobe2, disc2, phases_primary, phases_rim,
                                      phase_error=phase_error)

    x0 = np.array([getattr(system, n) if n in ("q", "incl", "R_1") else getattr(model, n)
                   for n in active_names], dtype=float)
    # explicit, larger-than-default finite-difference step -- same
    # reason as _run_rim_lsq_fit's/_run_primary_lsq_fit's own
    # diff_step=0.02: these residuals are smooth at realistic parameter
    # scales, but scipy's tiny default relative step under-resolves
    # them against the underlying fixed-resolution sampling grids
    # (shadow_boundary_on_rim's n_nu, lobe_outline's own hull vertices).
    result = least_squares(residuals, x0, bounds=_system_fit_bounds(active_names, model),
                            diff_step=0.02)
    sys2, model2 = apply(result.x)
    print(f"\nFit-system least-squares fit to {', '.join(active_names)}:")
    _print_lsq_covariance(active_names, result, len(result.fun))
    for n in ("q", "incl", "R_1"):
        if n not in active_names:
            print(f"\t{n} = {getattr(sys2, n):.6g} (held fixed, not fit)")
    if "R_out" not in active_names:
        print(f"\tR_out/a = {model2.R_out:.6g} (held fixed, not fit)")
    cov, _, _, _ = _lsq_covariance(active_names, result, len(result.fun))

    def sys2_lobe_of(x):
        sys2x, model2x = apply(x)
        return sys2x, lobe_disc_for(sys2x, model2x)[0]

    if "R_1" in active_names:
        _print_nauenberg_derived(sys2_lobe_of, result.x, cov, full=True)

    def d_L1_of(x):
        _, lobe2x = sys2_lobe_of(x)
        return lobe2x.x_L1 - lobe2x.x1

    d_L1, d_L1_sigma = _propagate_scalar(d_L1_of, result.x, cov)
    print(f"\td_L1/a = {d_L1:.4g} +/- {d_L1_sigma:.4g}")
    print("\tfinal residuals, primary phases 1-4 then bright-spot phases 5-8"
          + (" (weighted, dimensionless)" if args.phase_error is not None else " (unweighted, "
             "mixed units -- pass --phase_error for a real chi^2)") + ": "
          + ", ".join(f"{v:.6g}" for v in result.fun))

    return sys2, model2


def _run_system_mcmc_fit(args, system, model, lobe, disc, ntheta, nphi, finish):
    """
    "fit system" --mcmc_fit: the 1-4 active q/incl/R_1/R_out parameters'
    joint posterior given the same combined residual _run_system_lsq_fit's
    own residual computes (Gaussian likelihood on sum of squares -- no
    prior beyond the flat/improper default, same convention as this
    module's other MCMC fits), via emcee.

    Returns (system, model) -- same contract as _run_system_lsq_fit.
    """
    import emcee
    from multiprocessing.pool import ThreadPool

    active_names = _parse_system_fit_names(args.mcmc_fit, "--mcmc_fit")
    phases_primary, phases_rim = _system_fit_phases(args, "--mcmc_fit")
    phase_error = _resolve_phase_error(args.phase_error, 8, "--mcmc_fit")

    def apply(x):
        sys_over = {n: v for n, v in zip(active_names, x) if n in ("q", "incl", "R_1")}
        model_over = {n: v for n, v in zip(active_names, x) if n == "R_out"}
        sys2 = dataclasses.replace(system, **sys_over) if sys_over else system
        model2 = dataclasses.replace(model, **model_over) if model_over else model
        return sys2, model2

    def lobe_disc_for(sys2, model2):
        return _rim_fit_lobe_disc(sys2, model2, active_names, lobe, disc, ntheta, nphi)

    def residuals(x):
        sys2, model2 = apply(x)
        lobe2, disc2 = lobe_disc_for(sys2, model2)
        return _system_fit_residuals(sys2, lobe2, disc2, phases_primary, phases_rim,
                                      phase_error=phase_error)

    def log_probability(x):
        try:
            r = residuals(x)
        except Exception:
            return -np.inf
        if not np.all(np.isfinite(r)):
            return -np.inf
        return -0.5 * float(np.sum(r ** 2))

    ndim = len(active_names)
    nwalkers = args.walkers * ndim
    x0 = np.array([getattr(system, n) if n in ("q", "incl", "R_1") else getattr(model, n)
                   for n in active_names], dtype=float)
    rng = np.random.default_rng()
    spread = np.where(x0 != 0.0, np.abs(x0), 1.0) * args.spread
    pos = x0 + spread * rng.standard_normal((nwalkers, ndim))

    pool = ThreadPool(args.workers) if args.workers > 1 else None
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool)
    print(f"--mcmc_fit (fit system): {nwalkers} walkers ({args.walkers}/parameter), "
          f"{args.nburn} burn-in + {args.nsample} sample steps"
          + (f", {args.workers} worker threads" if pool is not None else ""))
    try:
        state = sampler.run_mcmc(pos, args.nburn, progress=True)
        sampler.reset()
        sampler.run_mcmc(state, args.nsample, progress=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    chain = sampler.get_chain(flat=True)
    medians = np.median(chain, axis=0)
    p05 = np.percentile(chain, 5, axis=0)
    p95 = np.percentile(chain, 95, axis=0)
    plus_err = p95 - medians
    minus_err = medians - p05
    mean90_err = 0.5 * (plus_err + minus_err)

    print(f"\nFit-system MCMC fit to {', '.join(active_names)} "
          f"({nwalkers} walkers x {args.nsample} steps = {len(chain)} samples):")
    for name, med, m9, pl, mi in zip(active_names, medians, mean90_err, plus_err, minus_err):
        print(f"\t{name} = {med:.4g} +/- {m9:.4g}  (+{pl:.4g} / -{mi:.4g})")
    print(f"mean acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")

    if args.corner_plot:
        import corner
        fig = corner.corner(chain, labels=active_names, truths=medians, show_titles=True)
        _, title_label = _prefix_and_title_label(args)
        if title_label:
            fig.suptitle(title_label)
        finish(fig, "corner")

    sys2, model2 = apply(medians)
    for n in ("q", "incl", "R_1"):
        if n not in active_names:
            print(f"\t{n} = {getattr(sys2, n):.6g} (held fixed, not fit)")
    if "R_out" not in active_names:
        print(f"\tR_out/a = {model2.R_out:.6g} (held fixed, not fit)")
    cov = np.cov(chain, rowvar=False).reshape(ndim, ndim)

    def sys2_lobe_of(x):
        sys2x, model2x = apply(x)
        return sys2x, lobe_disc_for(sys2x, model2x)[0]

    if "R_1" in active_names:
        _print_nauenberg_derived(sys2_lobe_of, medians, cov, full=True)

    def d_L1_of(x):
        _, lobe2x = sys2_lobe_of(x)
        return lobe2x.x_L1 - lobe2x.x1

    d_L1, d_L1_sigma = _propagate_scalar(d_L1_of, medians, cov)
    print(f"\td_L1/a = {d_L1:.4g} +/- {d_L1_sigma:.4g}")
    if args.phase_error is not None:
        chi2 = float(np.sum(residuals(medians) ** 2))
        dof = max(8 - len(active_names), 1)
        print(f"\tchi^2 at the median fit = {chi2:.6g} (8 phases, {len(active_names)} fit "
              f"parameters, {dof} dof -> reduced chi^2 = {chi2 / dof:.6g})")

    return sys2, model2


def _parse_image_size(s):
    """Parse a 'WIDTHxHEIGHT' pixel-dimension string, e.g. '788x644'."""
    parts = s.lower().split("x")
    try:
        if len(parts) != 2:
            raise ValueError
        w, h = int(parts[0]), int(parts[1])
        if w <= 0 or h <= 0:
            raise ValueError
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--image-size must be WIDTHxHEIGHT in pixels (e.g. 788x644), got {s!r}")
    return w, h


def _splice_equals(argv, flags):
    """
    Work around argparse's "expected one argument" error when a flag's
    value starts with '-' -- e.g. --bounds -1.2,1.5,-0.8,0.8 or
    --phase-list -0.02,-0.01,0.07,0.08: argparse only recognizes a token
    as a value (rather than another option) when it's ENTIRELY a bare
    negative number ("-1.2"), not a comma-separated list merely starting
    with one, so it misidentifies the whole thing as an attempted
    (unrecognized) option instead of the flag's own value. Rewrites two
    argv entries ("--flag", "value") into one ("--flag=value"), which
    argparse always accepts regardless of what the value looks like --
    a no-op for any flag in `flags` that was already given that way, or
    not given at all.

    flags: an iterable of flag strings (e.g. ("--bounds", "--phase-list"))
    -- every flag whose value might start with '-' needs to be listed
    here, since this rewrite has to happen before argparse ever sees argv.
    """
    flags = set(flags)
    out = []
    i = 0
    while i < len(argv):
        if argv[i] in flags and i + 1 < len(argv):
            out.append(f"{argv[i]}={argv[i + 1]}")
            i += 2
        else:
            out.append(argv[i])
            i += 1
    return out


def _parse_bounds(s):
    """Parse an 'xleft,xright,ybottom,ytop' sky-coordinate bounds string."""
    parts = s.split(",")
    try:
        if len(parts) != 4:
            raise ValueError
        xleft, xright, ybottom, ytop = (float(p) for p in parts)
        if not (xright > xleft and ytop > ybottom):
            raise ValueError
    except ValueError:
        raise argparse.ArgumentTypeError(
            "--bounds must be xleft,xright,ybottom,ytop with xright>xleft and "
            f"ytop>ybottom (e.g. -1.2,1.2,-0.8,0.8), got {s!r}")
    return xleft, xright, ybottom, ytop


def _parse_phase_list(s):
    """Parse a comma-separated list of explicit orbital phases, e.g. '0.1,0.15,0.85,0.9'."""
    try:
        vals = [float(p) for p in s.split(",") if p.strip() != ""]
        if not vals:
            raise ValueError
    except ValueError:
        raise argparse.ArgumentTypeError(
            "--phase-list must be a comma-separated list of phases "
            f"(e.g. 0.1,0.15,0.85,0.9), got {s!r}")
    return vals


def _parse_phase_error(s):
    """
    Parse --phase_error: either a single number (applied to every phase
    alike) or a comma-separated list (one entry per --phase-list phase,
    same order) -- see _resolve_phase_error, which expands/validates
    this against however many phases a given caller actually has (4 for
    "primary"/"shadow"/"rim projection", 8 for "fit system"). Just the
    raw parsed numbers here, in whatever count was actually given (1 or
    more) -- no length checking yet, since that depends on context this
    parser doesn't have.
    """
    try:
        vals = [float(p) for p in s.split(",") if p.strip() != ""]
        if not vals:
            raise ValueError
    except ValueError:
        raise argparse.ArgumentTypeError(
            "--phase_error must be a single number (applied to every phase) or a "
            f"comma-separated list of numbers (one per phase), got {s!r}")
    return tuple(vals)


def _resolve_phase_error(phase_error, n_phases, flag_name="--phase_error"):
    """
    None (--phase_error not given) stays None -- "residuals unweighted"
    throughout. Otherwise, --phase_error's own raw parsed value
    (_parse_phase_error's tuple) is either a single number, broadcast to
    all n_phases here, or already exactly n_phases numbers, one per
    phase in the same order the caller's own phase list uses -- any
    other length is a usage error, not silently truncated/padded.

    Returns an ndarray of exactly n_phases values, or None.
    """
    if phase_error is None:
        return None
    if len(phase_error) == 1:
        return np.full(n_phases, phase_error[0], dtype=float)
    if len(phase_error) != n_phases:
        raise SystemExit(f"{flag_name}: expected 1 value (applied to all {n_phases} phases) "
                          f"or exactly {n_phases} (one per phase, same order as --phase-list), "
                          f"got {len(phase_error)}")
    return np.array(phase_error, dtype=float)


def _hotspot_dphi_max(disc, model, phi_h, n_probe=2000, max_scales=20.0):
    """
    Downstream angular distance [deg] from phi_h at which the hot spot's
    own T_h*exp(-dphi/L_h) formula (render.disc_surfaces_with_teff) drops
    to (or below) the disc's own base temperature at that azimuth's rim
    -- beyond this point the real max(T_base, hot) formula is just
    T_base, i.e. no boost left at all, so stopping the outline's own
    grid (plots.disc_hotspot_cells) there gives a physically
    meaningful sense of how far the bright spot actually extends, rather
    than an arbitrary fixed angular cutoff. Recomputes the disc's base
    temperature directly from model.T_0/R_in/beta_d (render.
    disc_powerlaw_teff) rather than reusing the model/system's own
    disc_teff_func closure, which isn't available at all when
    --load-irradiation supplied temp_maps straight from a cache.

    Capped at max_scales*L_h if the crossing is never reached that far
    out (e.g. an unrealistically hot/slow-decaying spot) -- probed at
    n_probe points over [0, max_scales*L_h], not solved in closed form,
    since disc_powerlaw_teff has no simple inverse.
    """
    from render import disc_powerlaw_teff

    L_h = np.radians(model.L_h)
    dphi = np.linspace(0.0, max_scales * L_h, n_probe)
    hot = model.T_h * np.exp(-dphi / L_h)
    T_base = disc_powerlaw_teff(disc.rim(phi_h + dphi), model.T_0, model.R_in, model.beta_d)
    below = np.nonzero(hot <= T_base)[0]
    dphi_max = dphi[below[0]] if len(below) else max_scales * L_h
    return np.degrees(dphi_max)


def main():
    # stdout is fully block-buffered (not line-buffered) whenever it's not
    # a terminal -- e.g. piped through gui.py's QProcess -- so without
    # this, every "wrote <path>"/progress print below sits in Python's own
    # internal buffer and only actually reaches the reader in one lump
    # when the buffer fills or the process exits, instead of as each file
    # is actually saved.
    sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_param_args(parser)
    parser.add_argument("--outputs", type=str, default="outline",
                         help=f"comma-separated subset of {{{','.join(OUTPUT_CHOICES)}}}")
    parser.add_argument("--phase-min", type=float, default=0.25,
                         help="orbital phase (--phase-num=1) or phase range start "
                              "(--phase-num>1) for every phase-dependent output. Default 0.25, "
                              "deliberately paired with --phase-max=0.25 and --phase-num=1 "
                              "(a single, harmless phase) so a bare invocation can't "
                              "accidentally kick off a full multi-phase sweep")
    parser.add_argument("--phase-max", type=float, default=0.25,
                         help="phase range end; ignored if --phase-num=1 (--phase-min alone "
                              "is used then)")
    parser.add_argument("--phase-num", type=int, default=1,
                         help="number of phases: 1 (the default) uses --phase-min alone as a "
                              "single orbital phase; >1 uses linspace(--phase-min, --phase-max, "
                              "--phase-num). For 'outline'/'temperature'/'intensity', >1 writes "
                              "one output file per phase (a sequence, each named with its own "
                              "phase, see this module's docstring) instead of just one; "
                              "'lightcurve'/'magnitude'/'shadow' always plot every phase "
                              "together on one plot regardless. Ignored if --phase-list is given")
    parser.add_argument("--phase-list", type=_parse_phase_list, default=None,
                         help="explicit comma-separated list of orbital phases (e.g. "
                              "'0.1,0.15,0.85,0.9'), used in the given order -- a third way to "
                              "pick the phase(s) for every phase-dependent output, alongside "
                              "--phase-min/--phase-max/--phase-num's contiguous range (the same "
                              "choice as printing 'these specific pages' vs. 'pages A-B'). "
                              "Takes precedence over --phase-min/--phase-max/--phase-num when "
                              "given, e.g. to display exactly the phases relevant to an eclipse "
                              "ingress/egress constrained-area analysis")
    parser.add_argument("--phase_error", type=_parse_phase_error, default=None,
                         help="uncertainty [same units as --phase-list, orbital cycles] on the "
                              "--phase-list phases -- a single number (applied to every phase "
                              "alike), or a comma-separated list with exactly one entry per "
                              "phase, same order (4 for \"primary\"/\"shadow\"/\"rim projection\", "
                              "8 for \"fit system\"). Used by those outputs' own --lsq_fit/"
                              "--mcmc_fit (see _resolve_phase_error/_primary_fit_target/"
                              "_rim_fit_sigmas) to convert each phase's own geometric residual "
                              "into a properly chi^2-weighted one (residual / sigma, sigma "
                              "propagated from this via that phase's own local d(residual)/"
                              "d(phase) slope) instead of an unweighted sum of squares, and by "
                              "the same outputs' own display to flank each solid contact-phase "
                              "curve with two faint dashed ones at phase +/- its own error. Left "
                              "unset (default), residuals stay unweighted and no dashed curves "
                              "are drawn, same as before this option existed.")
    parser.add_argument("--outdir", type=str, default="demo_output",
                         help="output directory (created if missing); ignored with --show")
    parser.add_argument("--prefix", type=str, default=None,
                         help="global label for this run: shown as every output plot's title, "
                              "and used (with any blanks replaced by '_') as the output "
                              "filename prefix. Defaults to --config's basename "
                              "(no prefix/title at all if neither is given)")
    parser.add_argument("--no-irradiate", action="store_true",
                         help="skip the (expensive) irradiation flux calculation entirely -- "
                              "every temperature build uses gravity darkening only, whether the "
                              "initial one or (via --lsq_fit/--mcmc_fit) any not-'cheap' fit "
                              "trial's own rebuild, so this can speed up fitting considerably. "
                              "No effect together with --load-irradiation: a loaded cache's own "
                              "irradiation (baked in when it was saved) is always used exactly "
                              "as-is, regardless of this flag")
    parser.add_argument("--show", action="store_true",
                         help="display each output interactively instead of saving it to --outdir")
    parser.add_argument("--pyplot", action="store_true",
                         help="save each output as a standalone, runnable matplotlib .py "
                              "script instead of a .png -- every plotted x/y array becomes "
                              "its own named np.array(...) variable, and every ax.<method>(...) "
                              "call (plot/fill/axhline/legend/set_xlabel/...) that built the "
                              "figure is re-emitted verbatim in the same order, so the script "
                              "IS the plot's own recipe: copy it, combine it with other data, "
                              "or edit it directly to build something more complex. Currently "
                              "only the 'shadow'/'rim projection' outputs support this; other "
                              "outputs ignore it and save a normal .png")
    parser.add_argument("--show-points", action="store_true",
                         help="in the outline output, also scatter each body's actual "
                              "rendering sample points (secondary/disc/primary, at their "
                              "respective --n_areas_* resolutions) -- a way to visually judge "
                              "whether that resolution is dense enough without running a full "
                              "temperature render")
    parser.add_argument("--pixelmapping", choices=("direct", "indirect"), default="direct",
                         help="how surface samples become image pixels for the temperature "
                              "rendering: 'direct' (default) splats each sample across every "
                              "pixel its own projected area actually covers, with a depth "
                              "buffer to resolve overlaps (no holes, slower); 'indirect' "
                              "scatters each sample into the one pixel it projects onto "
                              "(faster, but can leave holes where the sample grid is coarser "
                              "than the image)")
    parser.add_argument("--n_areas_2", type=int, default=10000,
                         help="number of roughly equal-area surface sample points used for "
                              "the secondary (see roche.RocheLobe.equal_area_sample) -- shared "
                              "by every output (outline/temperature/lightcurve/magnitude); "
                              "lower it for a faster light curve, raise it for a denser render")
    parser.add_argument("--n_areas_d", type=int, default=2000,
                         help="total number of roughly equal-area surface elements used for "
                              "the disc (see disc.Disc.equal_area_annulus) -- shared by every "
                              "output; lower it for a faster light curve, raise it for a denser "
                              "render")
    parser.add_argument("--n_areas_1", type=int, default=200,
                         help="total number of roughly equal-area surface elements used for "
                              "the primary (see render.build_temperature_maps' n_primary) -- "
                              "shared by every output; raise it when the limb-darkening "
                              "coefficient (--u_1) matters to your use case, e.g. studying "
                              "its effect on ingress/egress shape")
    parser.add_argument("--n_field_1", type=int, default=0,
                         help="number of dipole field-line loops drawn from the primary's "
                              "magnetic axis (--theta_1/--phi_1), geometry only -- see "
                              "magnetic.py. Currently only the 'outline' output draws them; "
                              "0 (default) disables")
    parser.add_argument("--stream_angle", type=float, default=None,
                         help="how far around the primary (cumulative swept azimuth, in "
                              "degrees, 180=directly behind the primary, may exceed 360 to "
                              "loop around more than once) the ballistic accretion stream is "
                              "integrated before stopping -- see stream.integrate_stream. "
                              "Default (not given): stop at the trajectory's first closest "
                              "approach to the primary, as before. Every time the stream is "
                              "integrated, its closest approach and final radius/angle are "
                              "printed, to help pick a --angle_acc that actually connects to "
                              "the trajectory")
    parser.add_argument("--image-size", type=_parse_image_size, default="1280x720",
                         help="output image pixel dimensions, as WIDTHxHEIGHT (e.g. 1280x720 "
                              "or 1200x900); applies to all three output figures")
    parser.add_argument("--vmin", type=float, default=None,
                         help="colorbar minimum for whichever rendering is produced -- [K] for "
                              "temperature, [W/m^2/sr/m] for intensity (default: the rendered "
                              "data's own minimum). The primary usually dominates the intensity "
                              "scale by orders of magnitude, so clipping this (and/or --vmax) is "
                              "usually necessary to see limb-darkening effects on the fainter "
                              "bodies there")
    parser.add_argument("--vmax", type=float, default=None,
                         help="colorbar maximum for whichever rendering is produced -- [K] for "
                              "temperature, [W/m^2/sr/m] for intensity (default: the rendered "
                              "data's own maximum)")
    parser.add_argument("--bounds", type=_parse_bounds, default=None,
                         help="fixed plot bounds 'xleft,xright,ybottom,ytop' (units of a), "
                              "applied to the outline/temperature/intensity/shadow outputs' X/Y "
                              "axes -- note 'shadow' plots in its own (Y/a, -X/a) rotated "
                              "face-on frame, not outline/temperature/intensity's plain (X/a, "
                              "Y/a), so the same numeric box means something different there. "
                              "Equal x/y scaling is always kept, so these bounds also set the "
                              "plotted region's shape. Default (not given): outline/shadow "
                              "autoscale to whatever's actually visible (so plotted area/shape "
                              "can drift from phase to phase, e.g. across a movie's frames), and "
                              "temperature/intensity already default to a phase-independent "
                              "envelope (render.auto_extent) instead. Useful for a movie's "
                              "consistent field of view, or to crop in on a specific region")
    parser.add_argument("--data-file", type=str, default=None,
                         help="CSV or FITS table of observed (phase, flux or magnitude, error) "
                              "data to overlay on the light curve/magnitude plot")
    parser.add_argument("--data-phase-col", type=str, default="phase",
                         help="column name for orbital phase in --data-file")
    parser.add_argument("--data-flux-col", type=str, default="flux",
                         help="column name for flux [mJy] in --data-file; ignored if "
                              "--data-mag-col is given")
    parser.add_argument("--data-mag-col", type=str, default=None,
                         help="column name for AB magnitude in --data-file, as an alternative "
                              "to --data-flux-col -- give exactly one of the two")
    parser.add_argument("--data-err-col", type=str, default=None,
                         help="column name for the uncertainty on whichever of "
                              "--data-flux-col/--data-mag-col is actually used, in the same "
                              "units as that column. Leave unset (the default) if the data has "
                              "no uncertainty column -- error bars are then omitted from the "
                              "plot, and --lsq_fit falls back to equal weights")
    parser.add_argument("--data-rv-file", type=str, default=None,
                         help="CSV or FITS table of observed radial-velocity data to overlay "
                              "on the 'rv' output and/or fit against (--lsq_fit/--mcmc_fit) -- "
                              "defaults to --data-file itself (a single table with both "
                              "photometric and RV columns) if not given")
    parser.add_argument("--data-rv-phase-col", type=str, default="phase",
                         help="column name for orbital phase in --data-rv-file")
    for _suffix, _curve_name, _color in RV_DATA_COMPONENTS:
        parser.add_argument(f"--data_rv_{_suffix}_col", type=str, default=None,
                             help=f"column name for the {_curve_name} radial velocity [km/s] "
                                  f"in --data-rv-file (see render.radial_velocity_curve's "
                                  f"rv_{_curve_name} curve) -- enables that component's RV "
                                  f"data overlay and fit. Any subset of the four "
                                  f"--data_rv_*_col flags may be given at once, each "
                                  f"contributing its own residual block to a joint fit; left "
                                  f"unset (the default), that component's RV data is not used")
        parser.add_argument(f"--data_rv_{_suffix}_err_col", type=str, default=None,
                             help=f"column name for the uncertainty on --data_rv_{_suffix}_col, "
                                  f"in km/s. Leave unset (the default) if the data has no "
                                  f"uncertainty column -- error bars are then omitted from the "
                                  f"plot, and --lsq_fit/--mcmc_fit fall back to equal weights "
                                  f"for this component")
    parser.add_argument("--data_rv_gamma", type=float, default=0.0,
                         help="systemic (gamma) velocity [km/s] added to every model RV curve "
                              "before comparing to any --data_rv_*_col. The model curves "
                              "themselves have zero systemic velocity (computed in the "
                              "system's own center-of-mass frame), so real spectroscopic data "
                              "-- which includes the system's own motion along the line of "
                              "sight -- needs this single, shared offset to compare correctly "
                              "(one systemic velocity for the whole system, not per-component). "
                              "Fittable via --lsq_fit/--mcmc_fit as 'rv_gamma'")
    parser.add_argument("--save-irradiation", type=str, default=None,
                         help="save the (expensive, phase-independent) temperature-map "
                              "calculation to this FITS path, along with all system/model "
                              "metadata, for reuse via --load-irradiation -- e.g. more light "
                              "curves or renders at other phases/wavelengths without "
                              "recomputing irradiation")
    parser.add_argument("--load-irradiation", type=str, default=None,
                         help="load a previously-saved --save-irradiation FITS file instead of "
                              "rebuilding the temperature maps from scratch; only --wavelength "
                              "and --u_2 may still safely override the loaded values -- "
                              "any other differing --config/per-field override (including "
                              "--u_1/--u_d, which change the cached irradiation, not "
                              "just the observer-facing flux) is dropped with a warning")
    parser.add_argument("--workers", type=int, default=1,
                         help="split the light-curve phase loop across this many worker "
                              "processes (default 1, i.e. serial). Only worth it once the "
                              "phase loop itself takes more than about a second -- a "
                              "low-resolution/few-phase preview can finish before a process "
                              "pool even starts up, making this a net loss there. Also used "
                              "by --mcmc_fit, which instead spreads each step's walkers across "
                              "this many threads (not processes, since the fit's own residuals "
                              "function can't be pickled)")
    parser.add_argument("--dist", type=float, default=10.0,
                         help="observer distance in parsecs used to scale the light curve "
                              "from a relative flux to a real spectral flux density [mJy] "
                              "(default 10 pc)")
    parser.add_argument("--data_norm", type=float, default=None,
                         help="normalization applied to --data-file's own values before "
                              "comparing/overlaying: multiplicative flux factor "
                              "(flux = table_flux*data_norm) if --data-flux-col is used, or "
                              "additive magnitude offset (mag = table_mag+data_norm) if "
                              "--data-mag-col is used -- for data that's normalized/relative "
                              "rather than already on the model's absolute scale. Left unset "
                              "(the default), no normalization is applied -- 1.0 could not "
                              "serve as that default for both cases at once, since it's a "
                              "no-op for the multiplicative flux use but shifts every "
                              "magnitude by a full magnitude")
    parser.add_argument("--lsq_fit", type=str, default=None,
                         help="comma-separated SystemParams/ModelParams field names (plus "
                              "'dist'/'data_norm'/'rv_gamma') to fit by least squares against "
                              "--data-file and/or any active --data_rv_*_col (whichever are "
                              "given -- jointly if more than one is), e.g. 'ph_off,dist' -- "
                              "requires at least one of --data-file/--data_rv_*_col; starts "
                              "from each parameter's current value (--config/CLI/default) and "
                              "updates it in place to the best fit before any output is produced")
    parser.add_argument("--mcmc_fit", type=str, default=None,
                         help="same parameter-name syntax as --lsq_fit (comma-separated "
                              "SystemParams/ModelParams field names, plus "
                              "'dist'/'data_norm'/'rv_gamma'), but fit by MCMC (emcee) instead "
                              "of least squares -- mutually exclusive with --lsq_fit. Walks "
                              "--walkers*<n parameters> walkers for --nburn burn-in steps then "
                              "--nsample production steps; each parameter's fit value is its "
                              "median posterior sample")
    parser.add_argument("--nburn", type=int, default=500,
                         help="--mcmc_fit burn-in steps, discarded before sampling (default 500)")
    parser.add_argument("--nsample", type=int, default=1000,
                         help="--mcmc_fit production steps kept for the posterior (default 1000)")
    parser.add_argument("--walkers", type=int, default=5,
                         help="--mcmc_fit walkers per fitted parameter (default 5) -- total "
                              "walker count is this times the number of --mcmc_fit parameters")
    parser.add_argument("--corner_plot", action="store_true",
                         help="--mcmc_fit: also produce a corner.py posterior "
                              "pairwise-correlation plot")
    parser.add_argument("--spread", type=float, default=0.01,
                         help="--mcmc_fit: relative dispersion of each walker's starting "
                              "position around its parameter's initial value (default 0.01, "
                              "i.e. 1%%) -- just needs to seed a small, non-degenerate cloud "
                              "for the sampler to diffuse outward from during burn-in, not "
                              "already span the true posterior width")
    # --bounds/--phase-list's values routinely start with '-' (a negative
    # xleft/ybottom, or a negative phase, e.g. -1.2,1.5,-0.8,0.8 or
    # -0.02,-0.01,0.07,0.08) -- splice them to --flag=value first so
    # argparse doesn't misidentify them as attempted options (see
    # _splice_equals).
    args = parser.parse_args(_splice_equals(sys.argv[1:], ("--bounds", "--phase-list")))

    # --angle_acc may be a comma-separated list of connection angles -- one
    # accretion spot per entry, all sharing spot_acc/T_acc/u_acc (see
    # its own --help, params._parse_angle_acc). A CLI --angle_acc is already that
    # list (or None); a --config file's own angle_acc default is NOT (it went
    # through load_yaml's ordinary single-float cast), so recover it
    # there too when the CLI didn't override it -- same "CLI beats config"
    # precedence params_from_args itself uses for every other field.
    # Either way, normalize args.angle_acc to a plain float (the first entry)
    # before params_from_args or anything else touches it, so
    # ModelParams.angle_acc (--lsq_fit, the FITS header round-trip) only ever
    # sees a single value, same as any other field. angle_acc_list itself
    # (every entry, including that first one) is what actually drives the
    # accretion-spot physics and outline drawing below.
    angle_acc_list = args.angle_acc
    if angle_acc_list is None and args.config:
        angle_acc_list = raw_angle_acc_from_config(args.config)
    if angle_acc_list is not None:
        args.angle_acc = float(angle_acc_list[0])

    outputs = {s.strip() for s in args.outputs.split(",") if s.strip()}
    unknown = outputs - set(OUTPUT_CHOICES)
    if unknown:
        parser.error(f"unknown --outputs entries: {sorted(unknown)}")
    if "fit system" in outputs and not (args.lsq_fit or args.mcmc_fit):
        parser.error("--outputs 'fit system' requires --lsq_fit or --mcmc_fit -- there is no "
                      "plain display mode for it (it only ever runs the combined primary + "
                      "bright-spot fit and prints the result)")

    # matplotlib's backend must be chosen before pyplot is imported: Agg
    # (headless, fast) when saving to disk, the normal interactive
    # backend when the user wants windows popped up via --show.
    if not args.show:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    from plots import (plot_component_outlines, plot_topdown_shadows, style_axes, labeled_title,
                        tighten_external_legend, fit_content_to_canvas, shadow_constrained_regions,
                        shadow_boundary_radial, lobe_outline, plot_disc_rim_projection)
    from eclipse import project
    from render import (build_temperature_maps, render_system_image, physical_light_curve,
                         radial_velocity_curve, split_area_count, disc_aspect_ratio,
                         PRIMARY_ASPECT, save_temperature_maps, load_temperature_maps,
                         G, MSUN, RSUN)

    if args.load_irradiation:
        lobe, disc, temp_maps, system, model, irradiate = load_temperature_maps(args.load_irradiation)
        # the cached temperature maps were computed for exactly this
        # system/model. u_2 is safe to override -- it only
        # reshapes the secondary's own emission toward the observer, not
        # how much it absorbs -- but u_1/u_d are NOT: they
        # change how much the primary/disc irradiate the secondary (see
        # irradiation.star_irradiation_flux/disc_irradiation_flux), which
        # is baked into the cached Tsec, just like R_in or any other
        # geometry field. ph_off is likewise always safe -- it never
        # reaches build_system/build_temperature_maps at all, only
        # shifting the observed-data overlay at comparison time (see the
        # "lightcurve"/"magnitude"/"rv" blocks below), same as why it's
        # in _CHEAP_SYSTEM_FIT_FIELDS for --lsq_fit/--mcmc_fit. So
        # wavelength, u_2, ph_off, and the separately-handled --phase*
        # flags (not SystemParams/ModelParams fields at all) can safely
        # differ without invalidating the cache -- any other differing
        # CLI override is dropped, with a warning.
        system, model = params_from_args(args, base=(system, model), strict=True,
                                          exempt=("wavelength", "u_2", "ph_off"))
        if disc.is_empty:
            n_disc = (1, 1)
        else:
            n_r = round((disc.a - disc.r_in) / temp_maps.disc_cell_size)
            n_disc = (n_r, len(temp_maps.disc_pts) // n_r)
    else:
        system, model = params_from_args(args)
        irradiate = not args.no_irradiate
        temp_maps = None

    prefix, title_label = _prefix_and_title_label(args)

    if not args.show:
        os.makedirs(args.outdir, exist_ok=True)

    figsize = (args.image_size[0] / IMAGE_DPI, args.image_size[1] / IMAGE_DPI)

    # The one phase spec for every phase-dependent output (see --phase-num's
    # help): --phase-list's own explicit phases, in the given order, if
    # given (takes precedence -- see its own --help); else a single phase
    # (--phase-max ignored) if --phase-num<=1; else linspace(--phase-min,
    # --phase-max, --phase-num). "outline"/"temperature"/"intensity" loop
    # over `phases`, writing one output per entry (a sequence when
    # there's more than one); "lightcurve"/"magnitude"/"shadow" always
    # plot every entry together on one plot.
    phases = (np.array(args.phase_list) if args.phase_list is not None else
              np.array([args.phase_min]) if args.phase_num <= 1 else
              np.linspace(args.phase_min, args.phase_max, args.phase_num))

    def finish(fig, kind, phase=None):
        # title_label (SYSTEM tab's global label, --prefix) is merged into
        # each output's own title at its own plotting call site (see
        # plots.labeled_title, and this module's render_and_save/
        # plot_flux_mag_view/"rv" block below) -- a single "LABEL : ..."
        # line rather than a separate figure-level one, except the MCMC
        # corner plot (_run_mcmc_fit), which has no single per-axis title
        # to merge into and uses fig.suptitle directly instead.
        fig.tight_layout()
        if kind in ("outline", "shadow", "primary"):
            # outline/shadow's legend sits below the axes via
            # ax.legend(bbox_to_anchor=(x, negative axes-fraction), ...)
            # (see plots.py) -- an axes-fraction offset that ax.set_aspect
            # ("equal") can badly distort once the axes' own fractional
            # height gets shrunk on a wide canvas (e.g. the default
            # 1280x720 --image-size), leaving a huge gap below the plot.
            # Re-anchor it a small, fixed distance (in real inches) below
            # the axes' actual rendered extent instead, now that layout is
            # otherwise final.
            tighten_external_legend(fig, fig.axes[0])
        # last: nothing (title, legend, axis labels) should overflow the
        # fixed --image-size canvas -- see fit_content_to_canvas's own
        # docstring for why tight_layout's own margins can't be trusted
        # to already guarantee that.
        fit_content_to_canvas(fig)
        if args.show:
            # leave the figure open, undrawn -- the single plt.show() call
            # at the end of main() displays every requested output at once,
            # only once everything has finished computing. Progress prints
            # (with elapsed time, see utils.with_progress) are the feedback
            # during a long "temperature"/"lightcurve" wait instead.
            return
        path = _output_path(args.outdir, prefix, OUTPUT_KIND[kind], phase)
        # NOT bbox_inches="tight": that crops the canvas to each frame's
        # own content extent, which varies phase to phase (e.g. outline's
        # autoscale, or how wide the legend ends up) -- fine for a single
        # image, but it means a --phase-num>1 sequence comes out as
        # differently-sized PNGs instead of the fixed --image-size every
        # frame needs to share for a movie or any other automated/batch
        # use. tighten_external_legend above already keeps outline/
        # shadow's legend from drifting off the bottom of the fixed
        # canvas, which was bbox_inches="tight"'s only real job here.
        fig.savefig(path, dpi=IMAGE_DPI)
        plt.close(fig)
        print("wrote", path)

    # split the disc's (and, for the outline plot only, the primary's)
    # total element count into their own grid dimensions so individual
    # elements come out roughly equal-area, not a fixed (and often
    # wrong-for-this-body) aspect ratio -- see render.split_area_count.
    # Neither the secondary nor the primary's own light-curve/image
    # sampling need this: both take a single count directly (equal-area
    # sampling over the whole surface, see roche.RocheLobe.equal_area_sample
    # / lightcurve.sphere_points_full), no aspect ratio to split.
    def _report_split(label, requested, n1, n2):
        actual = n1 * n2
        if actual != requested:
            print(f"{label}: requested {requested}, using {actual} "
                  f"({n1}x{n2}, nearest equal-area split)")

    # a plain count now (render.build_temperature_maps/sphere_points_full
    # sample the whole sphere directly, no aspect-ratio grid split needed --
    # see render.TemperatureMaps' n_primary docstring); only the outline
    # plot below still needs the old (n_rho,n_phi) split, since it draws
    # via lightcurve.star_points directly.
    n_primary = args.n_areas_1
    n_primary_outline = split_area_count(args.n_areas_1, PRIMARY_ASPECT)
    _report_split("primary outline elements", args.n_areas_1, *n_primary_outline)

    if not args.load_irradiation:
        if model.has_disc:
            n_disc = split_area_count(args.n_areas_d, disc_aspect_ratio(model.R_in, model.R_out))
            _report_split("disc surface elements", args.n_areas_d, *n_disc)
        else:
            n_disc = (1, 1)  # unused (see below) -- disc_aspect_ratio would
                              # divide by zero on a None R_in/R_out anyway
        lobe, disc, disc_teff_func = build_system(system, model)

    # sanity-check printout: convert this run's internal-unit parameters
    # (P_orb/a/q, plus R_1/R_2 as fractions of a -- see roche.py's module
    # docstring for the a=1, G(M1+M2)=1 convention) to the equivalent
    # physical masses/radii/separation, via Kepler's third law for
    # M1+M2. lobe.r_volume_equiv is the secondary's actual
    # volume-equivalent radius (fraction of a) whether it came from an
    # explicit system.R_2 or from R_2's default lobe-filling behavior.
    # system.a is itself already in Rsun (see SystemParams' own field
    # comment/a_m property) -- R_1*a, r_volume_equiv*a are Rsun directly,
    # no further scaling needed; only the Kepler mass calc below needs
    # the SI (meters) a_m.
    M_total = 4.0 * np.pi ** 2 * system.a_m ** 3 / (G * system.P_orb_s ** 2)
    M_1 = M_total / (1.0 + system.q)
    M_2 = system.q * M_1
    print(f"\nsystem check:"
        f"\n\tM_1    = {M_1 / MSUN:.3g} Msun"
        f"\n\tR_1    = {system.R_1 * system.a:.6g} Rsun"
        f"\n\tM_2    = {M_2 / MSUN:.3g} Msun"
        f"\n\tR_2    = {lobe.r_volume_equiv * system.a:.4g} Rsun"
        f"\n\ta      = {system.a:.4g} Rsun")
    # inner Lagrange point's own distance from the primary -- lobe.x_L1
    # and lobe.x1 are both already in this package's own a=1 corotating-
    # frame convention (roche.py's module docstring), so their
    # difference is directly d_L1/a, no further scaling needed.
    print(f"\td_L1/a = {lobe.x_L1 - lobe.x1:.6g}")

    # disc.is_empty reflects model.has_disc (see build_system): the disc
    # and hot spot are used only when their inputs were actually given.
    # The accretion stream is a separate, independent knob -- tied to
    # whether the secondary actually overflows its Roche lobe
    # (lobe.fill_factor), not to whether a disc was configured to catch
    # it: a magnetic CV (see AMHer.yaml) can have a real stream with no
    # disc at all, channeled onto the primary's magnetic pole instead.
    if lobe.fill_factor < 1.0:
        print(f"secondary underfills its Roche lobe (fill_factor={lobe.fill_factor:.4f}): "
              "no accretion stream")
    elif system.R_2 > 0.0 and system.R_2 > lobe.r_volume_equiv_full:
        print(f"secondary effective radius R_2={system.R_2:g} exceeds the Roche-lobe-filling "
              f"radius, so it was clipped to exactly filling; enter "
              f"R_2={lobe.r_volume_equiv_full:.6g} instead for that precise value")
    if disc.is_empty:
        print("no disc configured (R_in/R_out/T_0/beta_d not all given): no disc or hot spot")

    if args.lsq_fit and args.mcmc_fit:
        parser.error("--lsq_fit and --mcmc_fit are mutually exclusive -- pick one")

    # shadow-mode fitting (see _run_shadow_lsq_fit/_run_shadow_mcmc_fit's
    # own docstrings) replaces the usual light-curve/RV fit entirely
    # whenever "shadow" is a requested output: it needs neither
    # photometric/RV data nor temp_maps (its own residual is purely
    # geometric -- shadow-boundary curves vs. the ballistic stream), so
    # it's dispatched BEFORE the temp_maps build below, and excluded from
    # what triggers that build. rim-projection-mode fitting (_run_rim_
    # lsq_fit/_run_rim_mcmc_fit) is the same idea, for "rim projection"
    # instead, primary-mode fitting (_run_primary_lsq_fit/_run_primary_
    # mcmc_fit) the same again for "primary", and fit-system fitting
    # (_run_system_lsq_fit/_run_system_mcmc_fit) the same once more for
    # "fit system" (the combined primary+bright-spot grand fit) -- fit
    # system takes priority over shadow, which takes priority over rim
    # projection, which takes priority over primary, if more than one is
    # somehow requested together with a fit flag, rather than picking
    # one arbitrarily (fit system being the most specific/comprehensive
    # of the four).
    system_fit_mode = "fit system" in outputs and bool(args.lsq_fit or args.mcmc_fit)
    shadow_fit_mode = ("shadow" in outputs and not system_fit_mode
                        and bool(args.lsq_fit or args.mcmc_fit))
    rim_fit_mode = ("rim projection" in outputs and not system_fit_mode and not shadow_fit_mode
                     and bool(args.lsq_fit or args.mcmc_fit))
    primary_fit_mode = ("primary" in outputs and not system_fit_mode and not shadow_fit_mode
                         and not rim_fit_mode and bool(args.lsq_fit or args.mcmc_fit))

    # the (expensive) irradiation/disc-temperature calculation is
    # phase-independent, so build it once and share it between every
    # output that needs it (outline, temperature, lightcurve, magnitude --
    # also the shape to follow for rendering movie frames: build this
    # once, then call render_system_image per frame). Skipped entirely if
    # --load-irradiation already supplied temp_maps. Built before the
    # outline block below (rather than after, its more natural position)
    # so the outline can draw temp_maps.accretion_field_line (the
    # accretion-spot connection, see PRIMARY tab's angle_acc) if active.
    if temp_maps is None and ("temperature" in outputs or "intensity" in outputs
                               or "outline" in outputs
                               or "lightcurve" in outputs or "magnitude" in outputs or "rv" in outputs
                               or args.save_irradiation
                               or (not system_fit_mode and not shadow_fit_mode and not rim_fit_mode
                                   and not primary_fit_mode and (args.lsq_fit or args.mcmc_fit))):
        # outline (like shadow, which doesn't even reach this point -- see
        # above) never actually reads the secondary's temperature VALUES:
        # it only uses len(temp_maps.Tsec) (a point count, identical
        # either way) and temp_maps.accretion_field_lines (irradiate-
        # independent geometry). So when outline is the only reason
        # temp_maps is being built at all -- no temperature/intensity/
        # lightcurve/magnitude/rv output, no --save-irradiation/--lsq_fit/
        # --mcmc_fit to actually want the real physics -- there's nothing
        # to lose by skipping the (expensive) irradiation calculation
        # regardless of --no-irradiate, only time.
        needs_real_temps = bool({"temperature", "intensity", "lightcurve", "magnitude", "rv"}
                                 & outputs) or args.save_irradiation \
            or (not system_fit_mode and not shadow_fit_mode and not rim_fit_mode
                and not primary_fit_mode and (args.lsq_fit or args.mcmc_fit))
        temp_maps = build_temperature_maps(
            lobe, disc, system.T_1, system.T_2, system.R_1, disc_teff_func=disc_teff_func,
            irradiate=irradiate and needs_real_temps, beta_grav=model.beta_grav,
            hotspot_T_h=model.T_h if model.has_hotspot else None, hotspot_L_h_deg=model.L_h,
            n_sec=args.n_areas_2, n_disc=n_disc, u_disc=model.u_d,
            u_primary=system.u_1, n_primary=n_primary,
            theta_1=system.theta_1_rad, phi_1=system.phi_1_rad,
            angle_acc=(angle_acc_list if angle_acc_list is not None else model.angle_acc),
            spot_acc=model.spot_acc_rad, T_acc=model.T_acc, u_acc=model.u_acc,
            incl_deg=system.incl, stream_angle_deg=args.stream_angle,
            stream_eps=lubow_shu_eps(system.T_2, system.P_orb, system.a_m))

    if system_fit_mode:
        if disc.is_empty:
            parser.error("--outputs 'fit system' needs a real disc "
                          "(R_in/R_out/T_0/beta_d all given)")
        if args.lsq_fit:
            system, model = _run_system_lsq_fit(args, system, model, lobe, disc,
                                                 lobe._ntheta, lobe._nphi)
        else:
            system, model = _run_system_mcmc_fit(args, system, model, lobe, disc,
                                                  lobe._ntheta, lobe._nphi, finish)
        # same "rebuild lobe/disc/disc_teff_func from the fitted system/
        # model" contract as shadow_fit_mode's own branch below.
        lobe, disc, disc_teff_func = build_system(system, model, ntheta=lobe._ntheta, nphi=lobe._nphi)
        temp_maps = None
    elif shadow_fit_mode:
        if args.lsq_fit:
            system, model = _run_shadow_lsq_fit(args, system, model, lobe, lobe._ntheta, lobe._nphi)
        else:
            system, model = _run_shadow_mcmc_fit(args, system, model, lobe,
                                                   lobe._ntheta, lobe._nphi, finish)
        # q and/or R_out may have changed -- rebuild lobe/disc/disc_teff_func
        # to match, same as _run_lsq_fit/_run_mcmc_fit's own "rebuild"
        # contract, just done here since this fit never needed them itself.
        lobe, disc, disc_teff_func = build_system(system, model, ntheta=lobe._ntheta, nphi=lobe._nphi)
        temp_maps = None
    elif rim_fit_mode:
        if disc.is_empty:
            parser.error("--outputs 'rim projection' needs a real disc "
                          "(R_in/R_out/T_0/beta_d all given)")
        if args.lsq_fit:
            system, model = _run_rim_lsq_fit(args, system, model, lobe, disc,
                                              lobe._ntheta, lobe._nphi)
        else:
            system, model = _run_rim_mcmc_fit(args, system, model, lobe, disc,
                                               lobe._ntheta, lobe._nphi, finish)
        # same "rebuild lobe/disc/disc_teff_func from the fitted system/
        # model" contract as shadow_fit_mode's own branch above.
        lobe, disc, disc_teff_func = build_system(system, model, ntheta=lobe._ntheta, nphi=lobe._nphi)
        temp_maps = None
    elif primary_fit_mode:
        if args.lsq_fit:
            system, model = _run_primary_lsq_fit(args, system, model, lobe,
                                                  lobe._ntheta, lobe._nphi)
        else:
            system, model = _run_primary_mcmc_fit(args, system, model, lobe,
                                                   lobe._ntheta, lobe._nphi, finish)
        # same "rebuild lobe/disc/disc_teff_func from the fitted system/
        # model" contract as shadow_fit_mode's own branch above -- the
        # later "primary" output block (unchanged) then draws the fitted
        # q/incl's own view, R_1 circle included, same as any other
        # --config/CLI override.
        lobe, disc, disc_teff_func = build_system(system, model, ntheta=lobe._ntheta, nphi=lobe._nphi)
        temp_maps = None
    elif args.lsq_fit:
        (system, model, args.dist, args.data_norm, args.data_rv_gamma,
         lobe, disc, temp_maps) = _run_lsq_fit(
            args, system, model, lobe, disc, disc_teff_func, temp_maps,
            n_primary, n_disc, irradiate, lobe._ntheta, lobe._nphi, angle_acc_list)
    elif args.mcmc_fit:
        (system, model, args.dist, args.data_norm, args.data_rv_gamma,
         lobe, disc, temp_maps) = _run_mcmc_fit(
            args, system, model, lobe, disc, disc_teff_func, temp_maps,
            n_primary, n_disc, irradiate, lobe._ntheta, lobe._nphi, angle_acc_list, finish)

    def _outline_draw_state():
        """
        Everything plot_component_outlines needs that's phase-independent
        (the truncated stream trajectory, hotspot azimuth, stream-impact
        cross-section) -- computed once, shared by the "outline" output's
        own per-phase loop below AND the "primary" output (which draws
        the very same base view, just at phase=0 with a shadow-box
        overlay) -- each an independent call (matching "shadow"'s own
        standalone re-computation of stream_impact above) rather than
        threading shared state between output blocks, so either can run
        alone with no dependency on the other.
        """
        # eps_ls (the real, T_2/P_orb/a-derived Lubow & Shu sound speed,
        # not integrate_stream's own generic 0.02 default) is needed for
        # every cross-section's own H/W below, computed once up front
        # since it depends only on `system`, not on the trajectory itself.
        eps_ls = lubow_shu_eps(system.T_2, system.P_orb, system.a_m)
        stream_begin = None
        stream_end = None
        if lobe.fill_factor < 1.0:
            xs, ys = np.zeros(0), np.zeros(0)
        else:
            # extend the integration limit (if needed) to guarantee the
            # trajectory actually sweeps far enough to reach every active
            # --angle_acc entry -- same reasoning as
            # render.build_temperature_maps'/accretion_connection_line's own
            # matching extension (which governs the *red* field-line curves
            # drawn from temp_maps.accretion_field_lines below); this one
            # governs the *green* ballistic-stream curve drawn from this
            # block's own, separately-integrated traj.
            eff_stream_angle = args.stream_angle
            if angle_acc_list is not None:
                max_angle_acc = float(np.max(angle_acc_list))
                eff_stream_angle = max(args.stream_angle, max_angle_acc) \
                    if args.stream_angle is not None else max_angle_acc
            traj = integrate_stream(lobe, stream_angle_deg=eff_stream_angle)
            if args.stream_angle is not None:
                # an explicit --stream_angle is the user's own authoritative
                # instruction for how far to show the stream -- e.g. to
                # visualize overflow past where it would otherwise hit the
                # disc (see STREAM tab's own note) -- so display the WHOLE
                # computed trajectory (up to wherever integration actually
                # stopped: exactly stream_angle, or earlier if the particle
                # plunged into the primary first) rather than second-guessing
                # it with the closest-approach heuristic below, which exists
                # only to pick a sensible endpoint when stream_angle wasn't
                # given at all.
                idx = len(traj["x"]) - 1
            else:
                idx = closest_approach_index(traj, lobe.x1)
                if idx is None:
                    # no turnaround found within the trajectory (e.g. t_max
                    # cut it off too early -- see closest_approach_index's
                    # own docstring) -- fall back to the trajectory's own
                    # last point, same as every other closest_approach_index
                    # call site in this package.
                    idx = len(traj["x"]) - 1
            if angle_acc_list is not None:
                # don't let the plotted stream curve stop short of the
                # furthest active accretion-connection point -- otherwise
                # the red field-line curves (drawn separately, from the
                # already-extended temp_maps.accretion_field_lines) branch
                # off from a point the green stream curve never visibly
                # reaches, an inexplicable-looking gap. Only ever extends
                # idx (never shortens it below the closest-approach/
                # fallback choice above).
                for ang in angle_acc_list:
                    ang_idx = angle_acc_index(traj, float(ang))
                    if ang_idx is not None:
                        idx = max(idx, ang_idx)
            s_vals = np.linspace(0.0, traj["s"][idx], 200)
            xs, ys = sample_points(traj, s_vals)
            # the stream's own starting cross-section, right at L1 (the
            # trajectory's own first point/velocity, before any of the
            # truncation above) -- see plot_component_outlines' own
            # stream_begin docstring. r1_begin is the distance from the
            # PRIMARY (this package's universal convention for
            # lubow_shu_stream_size's own r1, same as stream_impact's
            # impact["r1"] below), not from L1 itself.
            r1_begin = float(np.hypot(traj["x"][0] - lobe.x1, traj["y"][0]))
            H_begin, W_begin = lubow_shu_stream_size(r1_begin, lobe.q, eps_ls)
            stream_begin = {"x": float(traj["x"][0]), "y": float(traj["y"][0]),
                             "vx": float(traj["vx"][0]), "vy": float(traj["vy"][0]),
                             "H": float(H_begin), "W": float(W_begin)}
            # a free-ellipse fallback "ending" cross-section, at wherever
            # this truncated trajectory itself stops -- only actually used
            # (see stream_impact/stream_end below) when the stream never
            # reaches a real disc impact, so there's no disc-flush ending
            # to show instead.
            r1_end = float(np.hypot(xs[-1] - lobe.x1, ys[-1]))
            H_end, W_end = lubow_shu_stream_size(r1_end, lobe.q, eps_ls)
            stream_end = {"x": float(xs[-1]), "y": float(ys[-1]),
                           "vx": float(traj["vx"][idx]), "vy": float(traj["vy"][idx]),
                           "H": float(H_end), "W": float(W_end)}
        # if a cache was loaded, show the sample count it actually has
        # (n_areas_2 is meaningless there -- the secondary wasn't
        # rebuilt from it) rather than the CLI's (possibly stale) default.
        n_sec_points = len(temp_maps.Tsec) if temp_maps is not None else args.n_areas_2
        # one line per active --angle_acc entry (see ModelParams.angle_acc, built
        # above from the *full* angle_acc_list, not just its first value) --
        # every one of them now really does heat its own spot.
        accretion_field_lines = temp_maps.accretion_field_lines if temp_maps is not None else []
        # the stream's own physical cross-section at the disc-impact
        # point (Hessman 1999's Lubow & Shu fits -- see
        # plot_component_outlines' own stream_impact docstring) AND the
        # disc's own stream-impact hot spot (T_h/L_h, DISC tab -- not the
        # magnetic accretion_spot above) both derive from the SAME
        # disc_impact_point call/nu_imp below, rather than hotspot_phi_h
        # separately re-deriving its own azimuth via impact_azimuth's own
        # (coarser, discrete-trajectory-index) estimate -- that used to
        # leave the hot-spot striping starting at a visibly different
        # azimuth than the stream's own impact point/cross-section
        # ellipse, even after matching their eps (impact_azimuth has no
        # sub-step interpolation; disc_impact_point does). Both
        # phase-independent, so computed once here, same as
        # accretion_field_lines/n_sec_points above.
        stream_impact = None
        nu_imp = None
        hotspot_footprint = None
        if not disc.is_empty:
            impact = disc_impact_point(lobe, disc, eps=eps_ls)
            if impact is not None:
                H, W = lubow_shu_stream_size(impact["r1"], lobe.q, eps_ls)
                stream_impact = {"x": impact["x"], "y": impact["y"],
                                  "vx": impact["vx"], "vy": impact["vy"],
                                  "H": float(H), "W": float(W)}
                # H,W are scaleheights (see lubow_shu_stream_size's own
                # docstring) -- the stream's actual total width/height is
                # twice that.
                # a real disc-flush ending is available -- drop the
                # free-ellipse fallback "ending" computed above, since
                # plot_component_outlines only wants one or the other
                # (stream_impact's disc-flush footprint is the more
                # physically meaningful "ending" when both exist).
                stream_end = None
                nu_imp, cos_incidence = impact_incidence(impact, disc)
                r_imp = disc.rim(nu_imp)
                disc_height = r_imp * np.tan(disc.opening_angle) \
                    if disc.opening_angle > 0.0 else 0.0
                incidence_deg = np.degrees(np.arccos(cos_incidence))
                print(f"at disc impact (units of a): disc rim wall height={disc_height:.6g}, "
                      f"stream total width={2.0 * W:.6g}, stream total height={2.0 * H:.6g}; "
                      f"stream/disc incidence angle={incidence_deg:.6g} deg "
                      f"(0=radial/head-on, 90=tangential/grazing)")
                # the stream's own physical footprint on the wall -- the
                # SAME oblique-projection-stretched ellipse
                # stream_impact_ellipse_outline draws, W_eff tangential
                # (grazing incidence spreads it further, same
                # cos_incidence floor) and H vertical -- gets a flat T_h
                # (see render.disc_surfaces_with_teff's own matching
                # `footprint` override), centered on nu_imp itself; the
                # T_h*exp(-dphi/L_h) stripe's own dphi=0 reference stays
                # nu_imp too (not shifted), so both pieces share one
                # anchor.
                W_eff = W / max(cos_incidence, 0.05)
                hotspot_footprint = (float(W_eff), float(H))
        # None (no disc, or the stream never reaches its rim -- same
        # "nothing to show" case build_temperature_maps itself already
        # handles for T_h) draws no hot spot at all.
        hotspot_phi_h = nu_imp if (model.has_hotspot and nu_imp is not None) else None
        hotspot_dphi_max_deg = (_hotspot_dphi_max(disc, model, hotspot_phi_h)
                                 if hotspot_phi_h is not None else None)
        hotspot_footprint = hotspot_footprint if hotspot_phi_h is not None else None
        return {"xs": xs, "ys": ys, "n_sec_points": n_sec_points,
                "accretion_field_lines": accretion_field_lines,
                "hotspot_phi_h": hotspot_phi_h, "hotspot_dphi_max_deg": hotspot_dphi_max_deg,
                "hotspot_footprint": hotspot_footprint,
                "stream_impact": stream_impact, "stream_begin": stream_begin,
                "stream_end": stream_end, "stream_eps": eps_ls}

    def _draw_outline(phase, state, primary_faint_lw_scale=0.5, primary_faint_alpha=0.3):
        """
        One outline-view figure at `phase`, from _outline_draw_state's own
        dict. primary_faint_lw_scale/primary_faint_alpha: see plot_
        component_outlines' own docstring -- "outline" mode below leaves
        these at their normal faint default; "primary" mode raises them,
        since that output usually shows the primary fully eclipsed (this
        faint circle the only thing showing it at all) and needs it to
        actually stand out against the eclipse-derived curves it's being
        compared to.
        """
        fig, ax = plt.subplots(figsize=figsize)
        plot_component_outlines(lobe, disc, {"x": state["xs"], "y": state["ys"]}, system.R_1,
                                 phase, system.incl, ax=ax,
                                 show_points=args.show_points,
                                 n_sec=state["n_sec_points"], n_disc=n_disc,
                                 n_primary=n_primary_outline,
                                 theta_1=system.theta_1_rad, phi_1=system.phi_1_rad,
                                 n_field_1=args.n_field_1,
                                 accretion_field_lines=state["accretion_field_lines"],
                                 hotspot_phi_h=state["hotspot_phi_h"], hotspot_L_h_deg=model.L_h,
                                 hotspot_dphi_max_deg=state["hotspot_dphi_max_deg"],
                                 hotspot_footprint=state["hotspot_footprint"],
                                 stream_impact=state["stream_impact"],
                                 stream_begin=state["stream_begin"],
                                 stream_end=state["stream_end"],
                                 stream_eps=state["stream_eps"],
                                 label=title_label,
                                 primary_faint_lw_scale=primary_faint_lw_scale,
                                 primary_faint_alpha=primary_faint_alpha)
        return fig, ax

    if "outline" in outputs:
        state = _outline_draw_state()
        for phase in phases:
            fig, ax = _draw_outline(phase, state)
            if args.bounds is not None:
                # override the default per-phase autoscale (fit to
                # whatever's currently visible/drawn) with a fixed field
                # of view -- ax.set_aspect("equal") above already keeps
                # x/y scaling equal, so this box also fixes the plotted
                # region's shape, same as every frame of a movie sharing
                # one field of view
                xleft, xright, ybottom, ytop = args.bounds
                ax.set_xlim(xleft, xright)
                ax.set_ylim(ybottom, ytop)
            finish(fig, "outline", phase)

    if "primary" in outputs:
        if args.phase_list is None or len(args.phase_list) != 4:
            parser.error("--outputs 'primary' needs --phase-list with exactly 4 phases "
                          "(ingress-before, ingress-after, egress-before, egress-after -- the "
                          "primary star's own eclipse contact phases)")
        p1, p2, p3, p4 = args.phase_list
        incl_rad = np.radians(system.incl)
        # The secondary's own projected LIMB (plots.lobe_outline -- the
        # same silhouette "outline" mode's own "secondary" curve draws)
        # at each of the primary's own 4 eclipse contact phases, re-
        # anchored onto the phase=0 view's own primary position -- NOT
        # each phase's raw shadow region (a first attempt at this output
        # tried that, plots.shadow_boundary_radial/shadow_constrained_
        # regions, but that machinery assumes the 4 phases bracket a
        # single POINT's near-instantaneous transition; real 1st-4th
        # contact phases instead span the star's whole finite ingress/
        # egress duration, which broke that combination outright). What's
        # actually wanted is simpler: the secondary's own limb, literally,
        # at each contact moment -- four nearly-straight chords sweeping
        # past the primary, the classical eclipse-geometry picture, and
        # exactly the "box" a fitted R_1 circle should sit tangent to.
        #
        # Each phase has its own projection (the observer direction
        # itself rotates with phase), so even though the primary sits at
        # the same corotating-frame point (lobe.x1,0,0) always, ITS OWN
        # projected position also drifts from one phase to the next --
        # shifting each limb curve by (primary's phase=0 position minus
        # its position at that limb's own phase) removes that drift,
        # leaving the 4 limbs directly comparable to each other and to
        # the phase=0 base view, as if the primary had stayed fixed on
        # the sky throughout (it doesn't, but the geometry that matters
        # here -- how close each limb comes to the primary -- is relative
        # to the primary's own position at each moment, not to some
        # arbitrary shared sky frame).
        colors = ["#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
        curve_labels = ["ingress before", "ingress after", "egress before", "egress after"]
        primary_pt = np.array([[lobe.x1, 0.0, 0.0]])
        Xprim0, Yprim0, _ = project(primary_pt, 0.0, incl_rad)
        phase_error = _resolve_phase_error(args.phase_error, 4, "--outputs primary")
        state = _outline_draw_state()
        # the primary usually sits fully eclipsed at this output's own
        # phase=0 base view (between the 2nd and 3rd of the 4 given
        # contact phases), so its faint "always shown, even eclipsed"
        # limb circle (see plot_component_outlines' own docstring) is
        # normally the ONLY trace of it at all -- raised well past that
        # circle's usual barely-visible default so a --lsq_fit/--mcmc_fit
        # result is actually legible against the 4 limb curves it's
        # being compared to, not just implied by the printed numbers.
        fig, ax = _draw_outline(0.0, state, primary_faint_lw_scale=1.5, primary_faint_alpha=0.6)

        def shifted_limb(phase):
            """(Xsec, Ysec) for `phase`'s own secondary limb, shifted onto
            the phase=0 view exactly like the main 4 curves below."""
            Xsec, Ysec = lobe_outline(lobe, phase, incl_rad)
            Xprim, Yprim, _ = project(primary_pt, phase, incl_rad)
            dx, dy = float(Xprim0[0] - Xprim[0]), float(Yprim0[0] - Yprim[0])
            return Xsec + dx, Ysec + dy

        for i, (phase, color, lbl) in enumerate(zip((p1, p2, p3, p4), colors, curve_labels)):
            Xs, Ys = shifted_limb(phase)
            print(f"secondary limb ({lbl}, phase={phase})")
            ax.plot(Xs, Ys, "-", color=color, lw=1.0, label=lbl)
            # --phase_error: two faint dashed limbs flanking this one, at
            # phase -+ that same uncertainty -- a visual timing-error
            # band around each solid curve above, using the identical
            # shift-onto-phase-0 construction, just at phase_i+-dphase
            # instead of phase_i itself.
            if phase_error is not None:
                dphase = phase_error[i]
                for j, sign in enumerate((-1.0, 1.0)):
                    Xs2, Ys2 = shifted_limb(phase + sign * dphase)
                    ax.plot(Xs2, Ys2, "--", color=color, lw=0.6, alpha=0.6,
                            label="+/- phase_error" if i == 0 and j == 0 else None)
        # plot_component_outlines already finalized its own legend
        # (outline mode's usual handles only) before returning -- redo it
        # now that the curves above have added their own handles, same
        # placement/style it used internally.
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.45), ncol=4,
                  fontsize=9, framealpha=0.9)
        if args.bounds is not None:
            xleft, xright, ybottom, ytop = args.bounds
            ax.set_xlim(xleft, xright)
            ax.set_ylim(ybottom, ytop)
        finish(fig, "primary")

    if args.save_irradiation:
        save_temperature_maps(args.save_irradiation, temp_maps, system, model,
                               lobe._ntheta, lobe._nphi, irradiate)
        print("wrote", args.save_irradiation)

    def render_and_save(quantity, kind, label, vmin, vmax, log_scale=False):
        # An unclipped linear scale is dominated by the primary (hottest by
        # a wide margin, e.g. a 14700K white dwarf against a ~2000K
        # accretion stream -- and even more so in intensity, which goes
        # roughly as T^4): everything else -- disc, stream, secondary --
        # washes out to near-black. Rather than guess a clipping range,
        # leave it to the user via --vmin/--vmax, applied to whichever of
        # temperature/intensity is actually being rendered; unset (the
        # default) falls back to the actual data range, same as plain
        # imshow.
        #
        # log_scale: "temperature"'s own dynamic range (a ~14700K white
        # dwarf next to a ~2000K stream/disc) still washes out everything
        # but the primary even with --vmin/--vmax clipping -- log10(T)
        # compresses that range so the disc/stream/secondary's own
        # structure stays visible alongside it. --vmin/--vmax stay in
        # plain K (unchanged CLI meaning); only the image actually shown
        # (and its own vmin/vmax) are converted to log10 here.
        #
        # The empty sky around the system has no real temperature at all
        # (render_system_image leaves it NaN) -- log10(NaN) is still NaN,
        # so it would stay transparent (showing the black facecolor below)
        # regardless, but filling it with a real, physically-motivated
        # floor (2.7K, the cosmic microwave background) instead lets it
        # flow through the same colormap as an actual data value rather
        # than a special-cased hole. Left at the default vmin=None it
        # would then wrongly dominate the autoscaled low end (2.7K is
        # far colder than any real structure, crushing the disc/stream/
        # secondary's own range into a sliver at the top) -- so
        # log_scale's own default vmin floor is a representative coolest-
        # visible-structure value (1000K) instead of the data's own
        # minimum, unless the user gives --vmin explicitly. 2.7K sits far
        # below that floor either way, so the background simply clips to
        # the colormap's own coldest color -- visually indistinguishable
        # from plain black, same as it looked before.
        if log_scale:
            vmin = np.log10(vmin) if vmin is not None else np.log10(1000.0)
            vmax = np.log10(vmax) if vmax is not None else None
        extend = {(False, False): "neither", (True, False): "min",
                  (False, True): "max", (True, True): "both"}[(vmin is not None, vmax is not None)]
        for phase in phases:
            img, xedges, yedges = render_system_image(
                lobe, disc, system.T_1, system.R_1, phase, system.incl, temp_maps,
                quantity=quantity, pixel_mapping=args.pixelmapping,
                u_primary=system.u_1, u_secondary=system.u_2, u_disc=model.u_d, T_acc=model.T_acc,
                u_acc=model.u_acc, extent=args.bounds)
            if log_scale:
                img = np.where(np.isnan(img), 2.7, img)  # empty sky -> cosmic microwave background
                with np.errstate(divide="ignore", invalid="ignore"):
                    img = np.log10(img)
            fig, ax = plt.subplots(figsize=figsize)
            ax.set_facecolor("black")  # unrendered (NaN) sky pixels show through as black, not white
            im = ax.imshow(img, origin="lower", extent=(xedges[0], xedges[-1], yedges[0], yedges[-1]),
                            cmap="inferno", vmin=vmin, vmax=vmax)
            # imshow's own default equal-aspect (1 data unit = 1 data unit
            # in x and y) "letterboxes" the actual image within ax's own
            # (roughly square) bounding box for a system this much wider
            # than it is tall -- a plain fig.colorbar(im, ax=ax) sizes
            # itself to ax's FULL box regardless, towering over the
            # actual visible image. make_axes_locatable instead attaches
            # a colorbar axes that tracks the image's own true displayed
            # bounding box, so the two end up the same height.
            cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.15)
            fig.colorbar(im, cax=cax, label=label, extend=extend)
            ax.set_xlabel("X / a")
            ax.set_ylabel("Y / a")
            ax.set_title(labeled_title(title_label, f"phase={phase:.2f}"))
            # lock the view to the rendered image's own extent -- imshow
            # alone shouldn't need this, but pins it explicitly against
            # any future addition (a legend, an overlay, ...) silently
            # widening the autoscaled view past --bounds.
            ax.set_xlim(xedges[0], xedges[-1])
            ax.set_ylim(yedges[0], yedges[-1])
            style_axes(ax)
            finish(fig, kind, phase)

    if "temperature" in outputs:
        render_and_save("temperature", "temperature", "log$_{10}$ T [K]", args.vmin, args.vmax,
                         log_scale=True)

    if "intensity" in outputs:
        # the quantity to render for checking limb-darkening effects
        # visually (--u_1/--u_2/--u_d) -- unlike
        # "temperature", this actually reflects them, since limb
        # darkening is a brightness effect (see render_system_image's
        # "v_band" docstring).
        render_and_save("v_band", "intensity", "I [W/m^2/sr/m]", args.vmin, args.vmax)


    # "lightcurve" (relative flux) and "magnitude" (AB magnitude) are two
    # views of the same underlying physical_light_curve computation, so
    # it's only run once (and only one FITS file written, with both flux
    # and magnitude columns) no matter which of the two -- or both -- are
    # requested.
    if "lightcurve" in outputs or "magnitude" in outputs:
        from blackbody import mjy_to_ab_mag, ab_mag_to_mjy

        # unlike every other phase-dependent output, "lightcurve"/
        # "magnitude" always sample their own curve from a plain
        # --phase-min/--phase-max/--phase-num sweep, ignoring
        # --phase-list even when it's given -- a curve needs an evenly
        # (or at least monotonically) sampled phase axis to plot as a
        # continuous line; --phase-list's own explicit, possibly
        # out-of-order set of phases is instead shown as vertical dashed
        # markers on top of that curve below (see plot_flux_mag_view),
        # the same "where in phase are we" role it plays for the diagram
        # outputs, without standing in for the curve's own sampling.
        curve_phases = (np.array([args.phase_min]) if args.phase_num <= 1 else
                         np.linspace(args.phase_min, args.phase_max, args.phase_num))
        star, discf, sec, total = physical_light_curve(
            lobe, disc, system.T_1, system.T_2, system.R_1, curve_phases, system.incl, system.a_m,
            wavelength_m=system.wavelength_m, temp_maps=temp_maps, n_workers=args.workers,
            distance_pc=args.dist,
            u_primary=system.u_1, u_secondary=system.u_2, u_disc=model.u_d,
            n_primary=n_primary, T_acc=model.T_acc, u_acc=model.u_acc)
        # --flux_offset: light this model doesn't capture (e.g. third
        # light/background contamination), added to the TOTAL only (not
        # to the individual star/disc/secondary components, which stay
        # exactly what the model itself predicts) -- applied here, after
        # physical_light_curve's own --dist scaling, so it's independent
        # of distance unlike every other flux in this package. Everything
        # downstream (the saved FITS total, both flux/magnitude plots)
        # reads `total` fresh, so one addition here covers all of them.
        total = total + system.flux_offset
        if not args.show:
            fits_path = _output_path(args.outdir, prefix, OUTPUT_KIND["lightcurve"], ext="fits")
            _save_lightcurve_fits(fits_path, curve_phases, star, discf, sec, total, system, model,
                                   args, irradiate, n_disc)
            print("wrote", fits_path)

        # load the observed data (if any) once, in whichever of flux/magnitude
        # the user actually gave (exactly one of --data-mag-col/--data-flux-col
        # is expected; magnitude takes priority if both happen to be set),
        # apply --data_norm to that native representation (multiplicative flux
        # factor or additive magnitude offset -- for data that's normalized/
        # relative rather than already on the model's absolute scale), then
        # derive both representations via blackbody's flux<->mag conversions so
        # either output plot below can overlay it regardless of which units it
        # was supplied in. system.ph_off (fraction of P_orb) is subtracted from
        # the data's own phase column to correct for a bad ephemeris in the
        # data, independent of the model's own phase convention. (If --lsq_fit
        # was given, system.ph_off/args.data_norm are already the fitted
        # values by this point.)
        obs_phase = obs_flux = obs_flux_err = obs_mag = obs_mag_err = obs_label = None
        if args.data_file:
            if args.data_mag_col:
                obs_phase, obs_mag, obs_mag_err = _load_lightcurve_data(
                    args.data_file, args.data_phase_col, args.data_mag_col, args.data_err_col)
                # additive identity is 0.0, not args.data_norm's other branch's
                # 1.0 -- see --data_norm's help
                obs_mag = obs_mag + (args.data_norm if args.data_norm is not None else 0.0)
                # the data's own column label, not "AB magnitude" -- these are
                # typically catalog magnitudes in some specific photometric
                # system (e.g. Johnson V), which the model's AB-magnitude
                # axis only approximates, so the legend shouldn't imply
                # they're the same unit
                obs_label = (args.data_mag_col if "mag" in args.data_mag_col.lower()
                             else f"{args.data_mag_col} [mag]")
                obs_flux = ab_mag_to_mjy(obs_mag)
                # first-order error propagation: d(flux)/d(mag) = -flux*ln(10)/2.5.
                # obs_mag_err is None if the data has no error column (see
                # _load_lightcurve_data) -- propagate that through rather
                # than fabricating a value, so the plot below just omits
                # error bars instead of drawing made-up ones.
                obs_flux_err = (obs_flux * (np.log(10.0) / 2.5) * obs_mag_err
                                 if obs_mag_err is not None else None)
            else:
                obs_phase, obs_flux, obs_flux_err = _load_lightcurve_data(
                    args.data_file, args.data_phase_col, args.data_flux_col, args.data_err_col)
                obs_label = args.data_flux_col
                # multiplicative identity is 1.0 -- see --data_norm's help
                norm = args.data_norm if args.data_norm is not None else 1.0
                obs_flux = obs_flux * norm
                obs_flux_err = obs_flux_err * norm if obs_flux_err is not None else None
                obs_mag = mjy_to_ab_mag(obs_flux)
                obs_mag_err = ((2.5 / np.log(10.0)) * obs_flux_err / obs_flux
                                if obs_flux_err is not None else None)
            obs_phase = obs_phase - system.ph_off

            # extend the overlay to repeat across cycles if the display
            # window (curve_phases, from --phase-min/--phase-max/
            # --phase-num) spans more than the single period the data was
            # folded into -- see _wrap_extend_phase's docstring; a no-op
            # (idx = 0..N-1, shift all 0) when the window is one cycle or
            # narrower.
            idx, shift = _wrap_extend_phase(obs_phase, float(np.min(curve_phases)),
                                              float(np.max(curve_phases)))
            obs_phase = obs_phase[idx] + shift
            obs_flux = obs_flux[idx]
            obs_flux_err = obs_flux_err[idx] if obs_flux_err is not None else None
            obs_mag = obs_mag[idx]
            obs_mag_err = obs_mag_err[idx] if obs_mag_err is not None else None

        def plot_flux_mag_view(primary):
            """
            Draw one flux/magnitude light-curve view: `primary` ('flux' or
            'mag') picks which unit is the main (left) axis and which is
            mirrored on the right via secondary_yaxis -- everything else
            (curves, observed-data overlay, styling) is identical either
            way. Used for both the "lightcurve" and "magnitude" outputs; see
            the call site below for how `primary` is chosen (the format the
            observed data was given in, when there is any, takes precedence
            over each output's historical default).
            """
            fig, ax = plt.subplots(figsize=figsize)
            if primary == "flux":
                y_total, y_star, y_disc, y_sec = total, star, discf, sec
                y_obs, y_obs_err = obs_flux, obs_flux_err
                ylabel = "flux [mJy]"
            else:
                y_total, y_star, y_disc, y_sec = (mjy_to_ab_mag(v) for v in (total, star, discf, sec))
                y_obs, y_obs_err = obs_mag, obs_mag_err
                ylabel = f"AB magnitude @ {args.dist:g} pc"
            ax.plot(curve_phases, y_total, "-", color="black", lw=1.3, label="total")
            ax.plot(curve_phases, y_star, "--", color="#2e6f95", lw=1.0, label="primary")
            ax.plot(curve_phases, y_disc, "--", color="#e08214", lw=1.0, label="disc")
            ax.plot(curve_phases, y_sec, "--", color="#d1272e", lw=1.0, label="secondary")
            if system.flux_offset != 0.0:
                # a flat, phase-independent 4th "component" curve, same
                # plot/label convention as primary/disc/secondary above --
                # makes the offset's own contribution to `total` visible
                # and identifiable, rather than only present as an
                # invisible shift baked silently into the total curve.
                # mjy_to_ab_mag has no meaningful value for a non-positive
                # flux, so the magnitude view simply omits this line for
                # a negative offset (the flux view still shows it).
                if primary == "flux" or system.flux_offset > 0.0:
                    y_offset = (system.flux_offset if primary == "flux"
                                else mjy_to_ab_mag(system.flux_offset))
                    ax.plot(curve_phases, np.full_like(curve_phases, y_offset), "--",
                            color="gray", lw=1.0, label="flux offset")
            if args.phase_list is not None:
                # --phase-list plays no part in sampling the curve itself
                # here (see curve_phases' own comment above) -- shown
                # instead as a reference marker for "where in phase are
                # we", the same role it plays for the diagram outputs.
                for i, p in enumerate(args.phase_list):
                    ax.axvline(p, color="black", lw=0.8, ls=":", alpha=0.6,
                               label="--phase-list" if i == 0 else None)
            if obs_phase is not None:
                # plotted as-is (matching the model curves' own units) -- no
                # longer normalized to its own peak, now that the model
                # itself is in real physical units rather than an arbitrary
                # relative scale
                ax.errorbar(obs_phase, y_obs, yerr=y_obs_err,
                             fmt="o", ms=3, color="black", ecolor="gray", elinewidth=0.8,
                             capsize=1.5, label=obs_label)
            if primary == "mag":
                ax.invert_yaxis()  # brighter (lower mag) on top, standard convention
            ax.set_xlabel("orbital phase")
            ax.set_ylabel(ylabel)
            ax.set_title(labeled_title(title_label))
            ax.legend(fontsize=9)
            style_axes(ax, right=False)
            if primary == "flux":
                # secondary_yaxis transforms the PRIMARY axis's own
                # (autoscaled) view limits, not just the plotted data, every
                # time it draws -- so the floor below has to work for
                # whatever that view happens to be, not just today's data.
                # Two problems follow from mjy_to_ab_mag's log10 otherwise:
                # (1) the default autoscale margin can pad the lower flux
                # limit to (or below) zero even when every actual data point
                # is positive, and a +-inf axis limit makes matplotlib raise
                # outright, not just draw an odd tick; (2) an
                # unclipped-but-tiny floor still maps to some enormously
                # faint magnitude, and matplotlib's tick locator then spreads
                # ticks across that whole range -- cramming most of them,
                # illegibly, into the sliver of pixels near zero flux where
                # mag(flux) changes almost vertically. Flooring at a fixed
                # *fraction* of the axis's own current max (rather than a
                # tiny absolute constant) keeps the mirrored magnitude range
                # within a sane number of magnitudes of the brightest point,
                # self-scaling to whatever flux units this particular
                # system/distance uses. Curves that legitimately dip to/
                # through this floor (e.g. a total eclipse) are simply
                # clipped at the visible axis edge on the magnitude side,
                # same as any plot's y-limits -- the flux axis itself is
                # untouched.
                _, ymax_flux = ax.get_ylim()
                flux_floor = max(ymax_flux * 1e-2, 1e-300)
                secax = ax.secondary_yaxis(
                    "right",
                    functions=(lambda f: mjy_to_ab_mag(np.clip(f, flux_floor, None)), ab_mag_to_mjy))
                secax.set_ylabel(f"AB magnitude @ {args.dist:g} pc")
            else:
                secax = ax.secondary_yaxis("right", functions=(ab_mag_to_mjy, mjy_to_ab_mag))
                secax.set_ylabel("flux [mJy]")
            secax.minorticks_on()
            secax.yaxis.set_major_locator(MaxNLocator(6))
            return fig

        # The main (left) axis shows whichever unit the observed data was
        # actually given in (--data-mag-col vs --data-flux-col), regardless
        # of which of "lightcurve"/"magnitude" was requested -- a user
        # plotting magnitude data expects to see magnitudes up front, not
        # mJy, and vice versa. Absent any data to take a cue from, each
        # output falls back to its own name's traditional unit.
        data_primary = ("mag" if args.data_mag_col else "flux") if args.data_file else None

        if "lightcurve" in outputs:
            finish(plot_flux_mag_view(data_primary or "flux"), "lightcurve")

        if "magnitude" in outputs:
            finish(plot_flux_mag_view(data_primary or "mag"), "magnitude")

    if "shadow" in outputs:
        # stream's own total width/height (Hessman 1999's Lubow & Shu
        # fits) at the disc-impact point -- same computation/print as
        # the "outline" output above, independent of it since this
        # output runs standalone.
        if not disc.is_empty:
            eps_ls = lubow_shu_eps(system.T_2, system.P_orb, system.a_m)
            impact = disc_impact_point(lobe, disc, eps=eps_ls)
            if impact is not None:
                H, W = lubow_shu_stream_size(impact["r1"], lobe.q, eps_ls)
                nu_imp, cos_incidence = impact_incidence(impact, disc)
                disc_height = disc.rim(nu_imp) * np.tan(disc.opening_angle) \
                    if disc.opening_angle > 0.0 else 0.0
                incidence_deg = np.degrees(np.arccos(cos_incidence))
                print(f"at disc impact (units of a): disc rim wall height={disc_height:.6g}, "
                      f"stream total width={2.0 * W:.6g}, stream total height={2.0 * H:.6g}; "
                      f"stream/disc incidence angle={incidence_deg:.6g} deg "
                      f"(0=radial/head-on, 90=tangential/grazing)")
        fig, real_ax = plt.subplots(figsize=figsize)
        # --pyplot (see its own --help): a recording proxy in place of
        # the real axes, so every ax.<method>(...) call plot_topdown_
        # shadows/the constrained-area overlay/--bounds handling below
        # make is captured for write_pyplot_script -- the real figure
        # still gets built normally either way, since the proxy forwards
        # every call through to the real axes underneath it.
        ax = _RecordingAxes(real_ax) if args.pyplot else real_ax
        plot_topdown_shadows(lobe, disc, system.R_1, phases, system.incl, ax=ax,
                              theta_1=system.theta_1_rad, phi_1=system.phi_1_rad,
                              n_field_1=args.n_field_1,
                              angle_acc=(angle_acc_list if angle_acc_list is not None else model.angle_acc),
                              T_eff1=system.T_1, T_acc=model.T_acc,
                              stream_angle_deg=args.stream_angle,
                              T_2=system.T_2, P_orb=system.P_orb, a_m=system.a_m,
                              label=title_label)
        if args.phase_list is not None and len(args.phase_list) == 4:
            # exactly 4 phases via --phase-list: (ingress_before,
            # ingress_after, egress_before, egress_after) -- the "2 pairs
            # of eclipse contact times" plots.shadow_constrained_regions
            # is built around. Overlay the resulting constrained area(s),
            # derived PURELY from the secondary's own shadow (no
            # disc/stream assumption -- see that function's own
            # docstring) -- a general geometric constraint on where
            # SOMETHING eclipsed at those phases must be, not specific to
            # the disc's own hot spot; for visual/numeric comparison
            # against wherever the current disc/stream model actually
            # puts whatever's being checked.
            p1, p2, p3, p4 = args.phase_list
            incl_rad = np.radians(system.incl)
            # the 4 raw shadow-boundary curves the constrained-area
            # polygon below is itself built from (shadow_constrained_
            # regions' own internal shadow_boundary_radial calls,
            # duplicated here since that function only ever returns the
            # combined polygon, not the 4 curves themselves) -- drawn
            # directly so the fit is actually visible against them, not
            # just implied by the polygon's own edges; --phase_error, if
            # given, additionally flanks each with two faint dashed
            # curves at that same phase +- phase_error, the same visual
            # timing-uncertainty band the "primary"/"rim projection"
            # outputs draw around their own 4 curves.
            colors = ["#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
            curve_labels = ["ingress before", "ingress after", "egress before", "egress after"]
            shadow_phase_error = _resolve_phase_error(args.phase_error, 4, "--outputs shadow")

            def shadow_curve_xy(ph):
                theta, r = shadow_boundary_radial(lobe, ph, incl_rad, disc.x1)
                x = disc.x1 + r * np.cos(theta)
                y = r * np.sin(theta)
                return y, -x  # plot_topdown_shadows' own (Y,-X) convention

            for i, (ph, color, lbl) in enumerate(zip((p1, p2, p3, p4), colors, curve_labels)):
                Xp, Yp = shadow_curve_xy(ph)
                ax.plot(Xp, Yp, "-", color=color, lw=1.0, label=lbl)
                if shadow_phase_error is not None:
                    dphase = shadow_phase_error[i]
                    for j, sign in enumerate((-1.0, 1.0)):
                        Xp2, Yp2 = shadow_curve_xy(ph + sign * dphase)
                        ax.plot(Xp2, Yp2, "--", color=color, lw=0.6, alpha=0.6,
                                label="+/- phase_error" if i == 0 and j == 0 else None)

            regions = shadow_constrained_regions(lobe, (p1, p2), (p3, p4), incl_rad, disc.x1)
            if not regions:
                print("constrained area: the ingress/egress phase pairs' own shadow "
                      "regions never overlap anywhere -- no constrained area found")
            for i, region in enumerate(regions):
                cx, cy = region["center"]
                r = np.hypot(cx - disc.x1, cy)
                az = np.degrees(np.arctan2(cy, cx - disc.x1))
                print(f"constrained area {i + 1}: center r={r:.4f} (units of a), "
                      f"azimuth={az:.4g} deg  (x={cx:.4f}, y={cy:.4f})")
                # machine-parseable block for gathering multiple runs'
                # results together: the 4 defining phases, one x,y line
                # per polygon vertex (corotating-frame midplane, same
                # coordinates as disc.rim/azimuth), then the centroid.
                print(f"{p1},{p2},{p3},{p4}")
                for vx, vy in zip(region["x"], region["y"]):
                    print(f"{vx:.6f},{vy:.6f}")
                print(f"{cx:.6f},{cy:.6f}")
                # plot_topdown_shadows' own (X,Y)_plot = (y,-x) convention.
                Xp, Yp = region["y"], -region["x"]
                ax.fill(Xp, Yp, facecolor="purple", edgecolor="none", alpha=0.12,
                        label="constrained area" if i == 0 else None)
            # always re-drawn now (not just "if regions"): the 4 raw
            # curves above are worth labeling even when they never
            # overlap anywhere.
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3,
                      fontsize=9, framealpha=0.9)
        if args.bounds is not None:
            # shadow's own axes are (Y/a, -X/a) -- a rotated face-on frame,
            # not outline's plain (X/a, Y/a) -- so a --bounds box means
            # something different here than for outline/temperature/
            # intensity; still the user's own call how to set it for this
            # output, not something to silently withhold.
            xleft, xright, ybottom, ytop = args.bounds
            ax.set_xlim(xleft, xright)
            ax.set_ylim(ybottom, ytop)
        if args.pyplot:
            path = _output_path(args.outdir, prefix, OUTPUT_KIND["shadow"], ext="py")
            write_pyplot_script(ax.calls, path, figsize=fig.get_size_inches())
            plt.close(fig)
            print("wrote", path)
        else:
            finish(fig, "shadow")

    if "rim projection" in outputs:
        if disc.is_empty:
            parser.error("--outputs 'rim projection' needs a real disc "
                          "(R_in/R_out/T_0/beta_d all given)")
        if args.phase_list is None or len(args.phase_list) != 4:
            parser.error("--outputs 'rim projection' needs --phase-list with exactly 4 "
                          "phases (ingress-before, ingress-after, egress-before, "
                          "egress-after) -- the same 4-phase shadow this projects onto "
                          "the disc's own outer rim wall")
        p1, p2, p3, p4 = args.phase_list
        rim_phase_error = _resolve_phase_error(args.phase_error, 4, "--outputs rim projection")
        fig, real_ax = plt.subplots(figsize=figsize)
        # --pyplot (see its own --help): a recording proxy in place of
        # the real axes, so every ax.<method>(...) call plot_disc_rim_
        # projection/the --bounds handling below make is captured for
        # write_pyplot_script -- the real figure still gets built
        # normally either way, since the proxy forwards every call
        # through to the real axes underneath it.
        plot_ax = _RecordingAxes(real_ax) if args.pyplot else real_ax
        fig, ax = plot_disc_rim_projection(lobe, disc, (p1, p2), (p3, p4), system.incl,
                                            system.T_2, system.P_orb, system.a_m,
                                            ax=plot_ax, label=title_label,
                                            phase_error=rim_phase_error)
        if args.bounds is not None:
            # this output's own axes are (distance along rim, height
            # along rim) -- not equal-scaled (see plot_disc_rim_
            # projection's own docstring) -- so --bounds here means yet
            # another thing than for outline/shadow; still honored the
            # same way, the user's own call.
            xleft, xright, ybottom, ytop = args.bounds
            ax.set_xlim(xleft, xright)
            ax.set_ylim(ybottom, ytop)
        if args.pyplot:
            path = _output_path(args.outdir, prefix, OUTPUT_KIND["rim projection"], ext="py")
            write_pyplot_script(ax.calls, path, figsize=fig.get_size_inches())
            plt.close(fig)
            print("wrote", path)
        else:
            finish(fig, "rim projection")

    if "rv" in outputs:
        rv_primary, rv_secondary, rv_stream, rv_magnetic, rv1, rv2 = radial_velocity_curve(
            lobe, disc, system.T_1, system.R_1, phases, system.incl, system.a_m, system.P_orb_s,
            temp_maps, wavelength_m=system.wavelength_m, u_primary=system.u_1, u_secondary=system.u_2,
            T_acc=model.T_acc, u_acc=model.u_acc, stream_angle_deg=args.stream_angle)

        # the model curves themselves are computed in the system's own
        # center-of-mass frame (zero systemic velocity); shift all of them
        # by the (possibly fitted, see --data_rv_gamma) systemic velocity
        # so they land in the same frame as real spectroscopic data, which
        # includes the system's own motion along the line of sight -- same
        # gamma2 offset _prepare_fit's residuals() adds to the model side.
        gamma = args.data_rv_gamma
        rv_primary, rv_secondary, rv_stream, rv_magnetic, rv1, rv2 = (
            rv_primary + gamma, rv_secondary + gamma, rv_stream + gamma,
            rv_magnetic + gamma, rv1 + gamma, rv2 + gamma)

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(phases, rv_primary, "-", color="#2e6f95", lw=1.3, label="primary")
        ax.plot(phases, rv_secondary, "-", color="#d1272e", lw=1.3, label="secondary")
        ax.plot(phases, rv_stream, "-", color="#1b9e77", lw=1.3, label="accretion stream")
        ax.plot(phases, rv_magnetic, "-", color="purple", lw=1.3, label="magnetic stream")
        ax.plot(phases, rv1, "--", color="#2e6f95", lw=0.8, alpha=0.5,
                label="primary Roche point (circular)")
        ax.plot(phases, rv2, "--", color="#d1272e", lw=0.8, alpha=0.5,
                label="secondary Roche point (circular)")

        # observed RV data (if any) -- any subset of the four
        # --data_rv_<1|2|stream|hotspot>_col columns may be active at
        # once, each overlaid in its own matching model curve's color;
        # same phase-shift/ph_off/multi-cycle handling as the lightcurve/
        # magnitude overlay above (see _wrap_extend_phase), just for a
        # single (phase, velocity, error) triple instead of flux/mag.
        for suffix, curve_name, color in RV_DATA_COMPONENTS:
            col = getattr(args, f"data_rv_{suffix}_col")
            if not col:
                continue
            err_col = getattr(args, f"data_rv_{suffix}_err_col")
            rv_file = args.data_rv_file or args.data_file
            if not rv_file:
                raise SystemExit(f"--data_rv_{suffix}_col requires --data-rv-file or --data-file")
            obs_phase, obs_rv, obs_rv_err = _load_lightcurve_data(
                rv_file, args.data_rv_phase_col, col, err_col)
            obs_phase = obs_phase - system.ph_off
            idx, shift = _wrap_extend_phase(obs_phase, float(np.min(phases)), float(np.max(phases)))
            obs_phase = obs_phase[idx] + shift
            obs_rv = obs_rv[idx]
            obs_rv_err = obs_rv_err[idx] if obs_rv_err is not None else None
            ax.errorbar(obs_phase, obs_rv, yerr=obs_rv_err,
                        fmt="o", ms=3, color=color, ecolor="gray", elinewidth=0.8,
                        capsize=1.5, label=f"{col} ({curve_name})")

        ax.axhline(0.0, color="gray", lw=0.5, alpha=0.5)
        if gamma != 0.0:
            # the systemic velocity itself -- distinct from the plain 0.0
            # reference line above, and from the gamma-shifted curves
            # (which already include it): this just marks where that
            # shift landed, e.g. to sanity-check a --data_rv_gamma fit
            # against the data's own apparent offset from zero.
            ax.axhline(gamma, color="black", lw=0.8, ls="--", alpha=0.6,
                       label=f"rv_gamma={gamma:.3g} km/s")
        ax.set_xlabel("orbital phase")
        ax.set_ylabel("radial velocity [km/s]")
        ax.set_title(labeled_title(title_label))
        ax.legend(fontsize=9)
        style_axes(ax)
        finish(fig, "rv")

    if args.show:
        if sys.platform == "darwin":
            # a GUI window spawned from a Terminal-launched process (rather
            # than double-clicking an app icon) doesn't automatically become
            # the frontmost/active app on macOS -- the matplotlib window is
            # real and mapped, just invisible behind whatever else is up
            # front, which looks exactly like "nothing happened". Best-effort
            # nudge it forward via System Events; silently no-ops if
            # Automation permission for this terminal/interpreter hasn't been
            # granted (System Settings > Privacy & Security > Automation) --
            # Cmd+Tab still finds the window either way.
            try:
                import subprocess
                subprocess.run(
                    ["osascript", "-e",
                     f'tell application "System Events" to set frontmost of every process '
                     f'whose unix id is {os.getpid()} to true'],
                    timeout=5, capture_output=True)
            except Exception:
                pass
        plt.show()


if __name__ == "__main__":
    main()
