#!/usr/bin/env python
# simulate.py
"""
Command-line driver: build a system from a YAML config (optionally
overridden by per-parameter CLI flags) and produce any combination of
the outline plot, temperature/intensity renderings, and eclipse light
curve -- either displayed interactively (--show) or saved to --outdir.

Every phase-dependent output shares one phase spec: --phase-min alone
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
                     impact_azimuth)

# SystemParams/ModelParams fields that render.physical_light_curve takes as
# live, per-call arguments even when reusing an already-built temp_maps --
# safe for _run_lsq_fit to vary without rebuilding lobe/disc/temp_maps.
# Anything else (T_2, disc geometry/temperature, R_2, q, ...) is baked into
# temp_maps at build time, so fitting it needs a full rebuild every trial
# (see _run_lsq_fit).
_CHEAP_SYSTEM_FIT_FIELDS = {"T_1", "R_1", "incl", "wavelength", "u_1", "u_2", "a", "ph_off"}
_CHEAP_MODEL_FIT_FIELDS = {"u_d"}

OUTPUT_CHOICES = ("outline", "temperature", "intensity", "lightcurve", "magnitude", "shadow", "rv")
OUTPUT_KIND = {"outline": "plot", "temperature": "render", "intensity": "intensity",
               "lightcurve": "lightcurve", "magnitude": "magnitude", "shadow": "shadow", "rv": "rv",
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
            incl_deg=sys2.incl, stream_angle_deg=args.stream_angle)
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
                lobe2, disc2, sys2.T_1, sys2.T_2, sys2.R_1, eval_phase, sys2.incl, sys2.a,
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
                lobe2, disc2, sys2.T_1, sys2.R_1, eval_rv_phase, sys2.incl, sys2.a, sys2.P_orb_s,
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

    dof = max(n_obs - len(names), 1)
    chi2 = float(np.sum(result.fun ** 2))
    chi2_reduced = chi2 / dof
    try:
        cov = np.linalg.inv(result.jac.T @ result.jac) * chi2_reduced
    except np.linalg.LinAlgError:
        cov = np.full((len(names), len(names)), np.nan)
    sigmas = np.sqrt(np.diag(cov))

    print(f"Least-squares fit to {', '.join(names)}:")
    for name, val, sig in zip(names, result.x, sigmas):
        print(f"\t{name} = {val:.6g} +/- {sig:.3g}")
    print("covariance matrix:")
    print(cov)
    print(f"reduced chi^2 = {chi2_reduced:.4f} (dof={dof})")

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

    print(f"MCMC fit to {', '.join(names)} "
          f"({nwalkers} walkers x {args.nsample} steps = {len(chain)} samples):")
    for name, med, m9, pl, mi in zip(names, medians, mean90_err, plus_err, minus_err):
        print(f"\t{name} = {med:.6g} +/- {m9:.3g}  (+{pl:.3g} / -{mi:.3g})")
    print(f"mean acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")

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


def _splice_equals(argv, flag):
    """
    Work around argparse's "expected one argument" error when a flag's
    value starts with '-' -- e.g. --bounds -1.2,1.5,-0.8,0.8: argparse
    only recognizes a token as a value (rather than another option) when
    it's ENTIRELY a bare negative number ("-1.2"), not a comma-separated
    list merely starting with one, so it misidentifies the whole thing as
    an attempted (unrecognized) option instead of --bounds' value.
    Rewrites two argv entries ("--flag", "value") into one
    ("--flag=value"), which argparse always accepts regardless of what
    the value looks like -- a no-op if `flag` was already given that way,
    or not given at all.
    """
    out = []
    i = 0
    while i < len(argv):
        if argv[i] == flag and i + 1 < len(argv):
            out.append(f"{flag}={argv[i + 1]}")
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


def _hotspot_dphi_max(disc, model, phi_h, n_probe=2000, max_scales=20.0):
    """
    Downstream angular distance [deg] from phi_h at which the hot spot's
    own T_h*exp(-dphi/L_h) formula (render.disc_surfaces_with_teff) drops
    to (or below) the disc's own base temperature at that azimuth's rim
    -- beyond this point the real max(T_base, hot) formula is just
    T_base, i.e. no boost left at all, so stopping the outline's wedge
    sequence (plots.disc_hotspot_wedges) there gives a physically
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
                              "together on one plot regardless")
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
                         help="fixed sky-plane plot bounds 'xleft,xright,ybottom,ytop' (units "
                              "of a), applied to the outline/temperature/intensity outputs' X/Y "
                              "axes (not 'shadow', which uses its own distinct face-on frame) "
                              "-- equal x/y scaling is always kept, so these bounds also set the "
                              "plotted region's shape. Default (not given): outline autoscales "
                              "to whatever's actually visible (so its plotted area/shape can "
                              "drift from phase to phase, e.g. across a movie's frames), and "
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
    # --bounds' value routinely starts with '-' (a negative xleft/ybottom,
    # e.g. -1.2,1.5,-0.8,0.8) -- splice it to --bounds=value first so
    # argparse doesn't misidentify it as an attempted option (see
    # _splice_equals).
    args = parser.parse_args(_splice_equals(sys.argv[1:], "--bounds"))

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

    # matplotlib's backend must be chosen before pyplot is imported: Agg
    # (headless, fast) when saving to disk, the normal interactive
    # backend when the user wants windows popped up via --show.
    if not args.show:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    from plots import (plot_component_outlines, plot_topdown_shadows, style_axes, labeled_title,
                        tighten_external_legend, fit_content_to_canvas)
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
    # help): a single phase (--phase-max ignored) if --phase-num<=1, else
    # linspace(--phase-min, --phase-max, --phase-num). "outline"/
    # "temperature"/"intensity" loop over `phases`, writing one output per
    # entry (a sequence when there's more than one); "lightcurve"/
    # "magnitude"/"shadow" always plot every entry together on one plot.
    phases = (np.array([args.phase_min]) if args.phase_num <= 1 else
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
        if kind in ("outline", "shadow"):
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
    from blackbody import AU_M
    M_total = 4.0 * np.pi ** 2 * system.a ** 3 / (G * system.P_orb_s ** 2)
    M_1 = M_total / (1.0 + system.q)
    M_2 = system.q * M_1
    print(f"system check: M_1={M_1 / MSUN:.6g} Msun, R_1={system.R_1 * system.a / RSUN:.6g} Rsun, "
          f"M_2={M_2 / MSUN:.6g} Msun, R_2={lobe.r_volume_equiv * system.a / RSUN:.6g} Rsun, "
          f"a={system.a / AU_M:.6g} AU")

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
                               or args.save_irradiation or args.lsq_fit or args.mcmc_fit):
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
                                 & outputs) or args.save_irradiation or args.lsq_fit or args.mcmc_fit
        temp_maps = build_temperature_maps(
            lobe, disc, system.T_1, system.T_2, system.R_1, disc_teff_func=disc_teff_func,
            irradiate=irradiate and needs_real_temps, beta_grav=model.beta_grav,
            hotspot_T_h=model.T_h if model.has_hotspot else None, hotspot_L_h_deg=model.L_h,
            n_sec=args.n_areas_2, n_disc=n_disc, u_disc=model.u_d,
            u_primary=system.u_1, n_primary=n_primary,
            theta_1=system.theta_1_rad, phi_1=system.phi_1_rad,
            angle_acc=(angle_acc_list if angle_acc_list is not None else model.angle_acc),
            spot_acc=model.spot_acc_rad, T_acc=model.T_acc, u_acc=model.u_acc,
            incl_deg=system.incl, stream_angle_deg=args.stream_angle)

    if args.lsq_fit and args.mcmc_fit:
        parser.error("--lsq_fit and --mcmc_fit are mutually exclusive -- pick one")

    if args.lsq_fit:
        (system, model, args.dist, args.data_norm, args.data_rv_gamma,
         lobe, disc, temp_maps) = _run_lsq_fit(
            args, system, model, lobe, disc, disc_teff_func, temp_maps,
            n_primary, n_disc, irradiate, lobe._ntheta, lobe._nphi, angle_acc_list)

    if args.mcmc_fit:
        (system, model, args.dist, args.data_norm, args.data_rv_gamma,
         lobe, disc, temp_maps) = _run_mcmc_fit(
            args, system, model, lobe, disc, disc_teff_func, temp_maps,
            n_primary, n_disc, irradiate, lobe._ntheta, lobe._nphi, angle_acc_list, finish)

    if "outline" in outputs:
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
        # if a cache was loaded, show the sample count it actually has
        # (n_areas_2 is meaningless there -- the secondary wasn't
        # rebuilt from it) rather than the CLI's (possibly stale) default.
        n_sec_points = len(temp_maps.Tsec) if temp_maps is not None else args.n_areas_2
        # one line per active --angle_acc entry (see ModelParams.angle_acc, built
        # above from the *full* angle_acc_list, not just its first value) --
        # every one of them now really does heat its own spot.
        accretion_field_lines = temp_maps.accretion_field_lines if temp_maps is not None else []
        # the disc's own stream-impact hot spot (T_h/L_h, DISC tab -- not
        # the magnetic accretion_spot above): phase-independent, so
        # computed once here, same as accretion_field_lines/n_sec_points.
        # impact_azimuth returns None if the stream never actually reaches
        # this disc's rim (e.g. r_in too large), same "nothing to show"
        # case build_temperature_maps itself already handles for T_h.
        hotspot_phi_h = impact_azimuth(lobe, disc) if model.has_hotspot else None
        hotspot_dphi_max_deg = (_hotspot_dphi_max(disc, model, hotspot_phi_h)
                                 if hotspot_phi_h is not None else None)
        for phase in phases:
            fig, ax = plt.subplots(figsize=figsize)
            plot_component_outlines(lobe, disc, {"x": xs, "y": ys}, system.R_1,
                                     phase, system.incl, ax=ax,
                                     show_points=args.show_points,
                                     n_sec=n_sec_points, n_disc=n_disc, n_primary=n_primary_outline,
                                     theta_1=system.theta_1_rad, phi_1=system.phi_1_rad,
                                     n_field_1=args.n_field_1,
                                     accretion_field_lines=accretion_field_lines,
                                     hotspot_phi_h=hotspot_phi_h, hotspot_L_h_deg=model.L_h,
                                     hotspot_dphi_max_deg=hotspot_dphi_max_deg,
                                     label=title_label)
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

    if args.save_irradiation:
        save_temperature_maps(args.save_irradiation, temp_maps, system, model,
                               lobe._ntheta, lobe._nphi, irradiate)
        print("wrote", args.save_irradiation)

    def render_and_save(quantity, kind, label, vmin, vmax):
        # An unclipped linear scale is dominated by the primary (hottest by
        # a wide margin, e.g. a 14700K white dwarf against a ~2000K
        # accretion stream -- and even more so in intensity, which goes
        # roughly as T^4): everything else -- disc, stream, secondary --
        # washes out to near-black. Rather than guess a clipping range,
        # leave it to the user via --vmin/--vmax, applied to whichever of
        # temperature/intensity is actually being rendered; unset (the
        # default) falls back to the actual data range, same as plain
        # imshow.
        extend = {(False, False): "neither", (True, False): "min",
                  (False, True): "max", (True, True): "both"}[(vmin is not None, vmax is not None)]
        for phase in phases:
            img, xedges, yedges = render_system_image(
                lobe, disc, system.T_1, system.R_1, phase, system.incl, temp_maps,
                quantity=quantity, pixel_mapping=args.pixelmapping,
                u_primary=system.u_1, u_secondary=system.u_2, u_disc=model.u_d, T_acc=model.T_acc,
                u_acc=model.u_acc, extent=args.bounds)
            fig, ax = plt.subplots(figsize=figsize)
            ax.set_facecolor("black")  # unrendered (NaN) sky pixels show through as black, not white
            im = ax.imshow(img, origin="lower", extent=(xedges[0], xedges[-1], yedges[0], yedges[-1]),
                            cmap="inferno", vmin=vmin, vmax=vmax)
            fig.colorbar(im, ax=ax, label=label, extend=extend)
            ax.set_xlabel("X / a")
            ax.set_ylabel("Y / a")
            ax.set_title(labeled_title(title_label, f"phase={phase:.2f}"))
            style_axes(ax)
            finish(fig, kind, phase)

    if "temperature" in outputs:
        render_and_save("temperature", "temperature", "T [K]", args.vmin, args.vmax)

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

        star, discf, sec, total = physical_light_curve(
            lobe, disc, system.T_1, system.T_2, system.R_1, phases, system.incl, system.a,
            wavelength_m=system.wavelength_m, temp_maps=temp_maps, n_workers=args.workers,
            distance_pc=args.dist,
            u_primary=system.u_1, u_secondary=system.u_2, u_disc=model.u_d,
            n_primary=n_primary, T_acc=model.T_acc, u_acc=model.u_acc)
        if not args.show:
            fits_path = _output_path(args.outdir, prefix, OUTPUT_KIND["lightcurve"], ext="fits")
            _save_lightcurve_fits(fits_path, phases, star, discf, sec, total, system, model,
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
            # window (phases, from --phase-min/--phase-max/--phase-num) spans
            # more than the single period the data was folded into -- see
            # _wrap_extend_phase's docstring; a no-op (idx = 0..N-1, shift
            # all 0) when the window is one cycle or narrower.
            idx, shift = _wrap_extend_phase(obs_phase, float(np.min(phases)), float(np.max(phases)))
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
            ax.plot(phases, y_total, "-", color="black", lw=1.3, label="total")
            ax.plot(phases, y_star, "--", color="#2e6f95", lw=1.0, label="primary")
            ax.plot(phases, y_disc, "--", color="#e08214", lw=1.0, label="disc")
            ax.plot(phases, y_sec, "--", color="#d1272e", lw=1.0, label="secondary")
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
        fig, ax = plt.subplots(figsize=figsize)
        plot_topdown_shadows(lobe, disc, system.R_1, phases, system.incl, ax=ax,
                              theta_1=system.theta_1_rad, phi_1=system.phi_1_rad,
                              n_field_1=args.n_field_1,
                              angle_acc=(angle_acc_list if angle_acc_list is not None else model.angle_acc),
                              T_eff1=system.T_1, T_acc=model.T_acc,
                              stream_angle_deg=args.stream_angle,
                              label=title_label)
        finish(fig, "shadow")

    if "rv" in outputs:
        rv_primary, rv_secondary, rv_stream, rv_magnetic, rv1, rv2 = radial_velocity_curve(
            lobe, disc, system.T_1, system.R_1, phases, system.incl, system.a, system.P_orb_s,
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
