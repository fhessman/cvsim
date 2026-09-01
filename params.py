# params.py
"""
The simulation's input parameters, split into two groups:

  SystemParams -- the physically "given" parameters of the binary,
  normally fixed by independent measurements (spectroscopy, parallax,
  eclipse timing, ...) rather than solved for from the eclipse light
  curve itself: P_orb, a, q, R_1, T_1, T_2, incl, wavelength.

  ModelParams -- the disc/accretion-structure parameters that actually
  get varied to fit an eclipse light curve: R_in, R_out, T_0, beta_d,
  T_h, L_h, plus the optional disc-shape extras (beta_grav, e_d,
  omega_d, alpha_d).

Field names favor brevity (R_1/R_2, u_1/u_2/u_d, beta_d, e_d, omega_d,
alpha_d, n_areas_1/2/d, dist, ...) over descriptiveness, since these
double as fit-parameter names/labels once a parameter search is built on
top of this module -- see each field's inline comment for what it means.

Disc temperature model (see render.disc_powerlaw_teff):

    T_d(R) = T_0 * (R/R_in)^beta_d

Stream-impact hot spot on the outer rim (see render.disc_surfaces_with_teff):

    T(phi) = max(T_d(rim(phi)), T_h * exp(-(phi-phi_h)/L_h))

phi_h -- the azimuth where the ballistic stream from L1 crosses the
disc's outer rim -- is NOT a free parameter: it is determined by q (via
the L1 nozzle/Roche geometry) and the disc's own shape, and is computed
automatically by stream.impact_azimuth(lobe, disc).

beta_d is kept distinct from beta_grav (the secondary's Lucy 1967
gravity-darkening exponent, roche.RocheLobe.gravity_darkened_teff) --
they are unrelated physical quantities that happen to both be power-law
indices, hence the different names.

Units: everywhere else in this package (disc.Disc, blackbody.py, ...)
time is seconds, angles are radians, and wavelength is meters -- SI, the
physics layer's convention. Here, at the user-facing config/CLI/GUI
boundary, P_orb is days, wavelength is Angstrom, and omega_d/alpha_d are
degrees (more convenient to type/read); build_system(),
SystemParams.P_orb_s, and SystemParams.wavelength_m do the conversion,
so nothing below this module ever sees the user-facing units.
"""

import argparse
import dataclasses
import math
from dataclasses import dataclass

import numpy as np
import yaml

from blackbody import V_WAVELENGTH_M


@dataclass
class SystemParams:
    """The binary's physically "given" parameters."""
    P_orb: float                              # orbital period [d]
    a: float                                  # orbital separation [m]
    q: float                                  # mass ratio, M2/M1
    R_1: float                                # primary radius, units of a
    T_1: float                                # primary effective temperature [K]
    T_2: float                                # secondary effective temperature [K]
    incl: float                               # orbital inclination [deg]
    wavelength: float = V_WAVELENGTH_M / 1e-10  # observing wavelength [Angstrom]
    u_1: float = 0.0                          # primary limb-darkening coeff. (linear law), 0=off
    u_2: float = 0.0                          # secondary limb-darkening coeff.
    R_2: float = 0.0                          # secondary volume-equiv. radius, units of a;
                                               # 0 (default) = exactly Roche-lobe-filling. A value
                                               # less than the lobe-filling volume-equivalent radius
                                               # makes the secondary underfill its Roche lobe (see
                                               # roche.RocheLobe's R2 parameter); values at or above
                                               # the lobe-filling radius are clipped to it (see
                                               # roche.RocheLobe.__init__), so there's no need to know
                                               # the exact filling radius to get a filling secondary.
    ph_off: float = 0.0                       # phase offset, same units as orbital phase elsewhere
                                               # (fraction of P_orb); subtracted from observed data's
                                               # phase column before it's compared against/overlaid on
                                               # the model (see simulate.py), to correct for a bad
                                               # ephemeris in the data rather than in the model itself.
    theta_1: float = 0.0                      # primary magnetic-axis obliquity [deg]: angle between
                                               # the magnetic axis and the orbital spin axis (z_hat).
                                               # 0 (default) = untilted/aligned. Assumes synchronous
                                               # rotation (the standard "polar" case), so the axis is
                                               # fixed in the corotating frame -- see magnetic.py.
    phi_1: float = 0.0                        # primary magnetic-axis azimuth [deg]: orientation of the
                                               # tilt, measured the same way as every other in-plane
                                               # azimuth in this package (from +x, the sub-secondary
                                               # direction at phase 0, increasing counterclockwise as
                                               # seen from +z). See magnetic.py.

    @property
    def wavelength_m(self):
        """wavelength converted to meters, for blackbody.band_intensity etc."""
        return self.wavelength * 1e-10

    @property
    def P_orb_s(self):
        """orbital period converted to seconds, for any physics-layer use."""
        return self.P_orb * 86400.0

    @property
    def theta_1_rad(self):
        return np.radians(self.theta_1)

    @property
    def phi_1_rad(self):
        return np.radians(self.phi_1)


@dataclass
class ModelParams:
    """The disc/hot-spot parameters fitted to an eclipse light curve.

    R_in/R_out/T_0/beta_d and T_h/L_h all default to None (rather than
    being required) so that "not given" is a real, distinguishable state:
    the disc is built at all only if R_in/R_out/T_0/beta_d are *all*
    given (see has_disc/build_system), and the hot spot only if T_h/L_h
    are *also* both given (has_hotspot) -- an explicit, user-controlled
    flag rather than something inferred from a default/placeholder value
    or from the secondary's fill factor (which is an independent knob,
    see SystemParams.R_2).
    """
    R_in: float = None                # disc inner radius, units of a
    R_out: float = None                # disc outer radius (semi-major axis if e_d>0), units of a
    T_0: float = None                   # disc temperature at R_in [K]
    beta_d: float = None                 # disc temperature power-law index
    T_h: float = None                     # hot-spot peak temperature [K]
    L_h: float = None                      # hot-spot azimuthal decay length [deg]
    beta_grav: float = 0.08          # secondary gravity-darkening exponent (Lucy 1967)
    e_d: float = 0.0                 # disc eccentricity (0 = circular)
    omega_d: float = 0.0             # disc periastron orientation [deg]
    alpha_d: float = 0.0             # disc half-opening angle from midplane [deg] (0 = flat)
    u_d: float = 0.0                 # disc limb-darkening coefficient (linear law), 0=off
    angle_acc: float = None           # cumulative swept azimuth [deg] at which the ballistic
                                       # stream connects to a magnetic field line (magnetic CV
                                       # accretion spot; see magnetic.py) -- same convention as
                                       # --stream_angle (0=facing secondary, 180=directly behind
                                       # the primary, may exceed 360 to loop around more than
                                       # once; see stream.integrate_stream/angle_acc_index).
                                       # None (default) = feature off -- unlike T_h/L_h's disc
                                       # hot spot, spot_acc/T_acc below already have real
                                       # defaults, so angle_acc alone gates has_accretion_spot.
    spot_acc: float = 5.0             # accretion spot's angular radius on the primary's
                                       # surface around the field line's footpoint [deg]
    T_acc: float = 100000.0           # accretion spot temperature [K]
    u_acc: float = 0.0                # accretion spot limb-darkening coefficient (linear
                                       # law), independent of u_1 (the rest of the primary);
                                       # 0=flat (no limb law of its own). Negative values give
                                       # limb-BRIGHTENING (I increases toward the limb) instead
                                       # of the usual darkening -- a crude stand-in for
                                       # cyclotron beaming, not a real angle-resolved emission
                                       # pattern.

    @property
    def omega_d_rad(self):
        return np.radians(self.omega_d)

    @property
    def alpha_d_rad(self):
        return np.radians(self.alpha_d)

    @property
    def spot_acc_rad(self):
        return np.radians(self.spot_acc)

    @property
    def has_disc(self):
        """True only if every disc-defining input was actually given --
        the sole trigger build_system uses for whether to build a disc
        at all."""
        return None not in (self.R_in, self.R_out, self.T_0, self.beta_d)

    @property
    def has_hotspot(self):
        """True only if there's a disc for it to sit on, and both
        hot-spot inputs were actually given."""
        return self.has_disc and None not in (self.T_h, self.L_h)

    @property
    def has_accretion_spot(self):
        """True only if angle_acc was actually given -- see angle_acc's own
        comment for why it alone gates this (spot_acc/T_acc already have
        real defaults). Does NOT check T_acc>T_1 or whether angle_acc is
        ever actually reached by the stream trajectory -- those need the
        built lobe/stream trajectory, so they're checked (and reported) in
        render.build_temperature_maps instead."""
        return self.angle_acc is not None


def build_system(system: SystemParams, model: ModelParams, ntheta=181, nphi=361):
    """
    Build (lobe, disc, disc_teff_func) from a SystemParams/ModelParams
    pair, ready to pass into render.render_system_image,
    render.physical_light_curve, and plots.plot_component_outlines
    (which all still take T_1, T_2, R_1, incl, and beta_grav
    directly from `system`/`model`, and system.wavelength_m for
    wavelength_m -- this factory only builds the geometry + disc
    temperature closure). model.omega_d/alpha_d [deg] are
    converted to the radians disc.Disc itself expects.

    ntheta/nphi: the secondary's own Roche-lobe surface grid resolution
    (RocheLobe's constructor default is 181x361); a numerical/rendering
    knob, not a physical parameter, so it isn't part of SystemParams --
    see render.split_area_count for turning a single "how many surface
    elements" count into this (ntheta,nphi) pair.

    system.R_2 (0 = lobe-filling, the default) sets the secondary's
    volume-equivalent radius, passed straight through to
    roche.RocheLobe's own R2 parameter, which resolves it to a
    fill_factor internally -- clipping it to exactly 1.0 (lobe-filling)
    if system.R_2 is at or above the true lobe-filling radius
    (lobe.r_volume_equiv_full), so the caller doesn't need to already
    know that radius to ask for a filling secondary.

    Whether a disc is built at all is a separate, independent knob:
    model.has_disc (all of R_in/R_out/T_0/beta_d actually given).
    When it's False, a lobe-independent "null disc" (r_in=a=0) is built
    instead, whose existing rim/contains/visible_from/equal_area_annulus
    interface already degrades to "occults nothing, renders nothing"
    without any special-casing at the many call sites that hold a `disc`
    object elsewhere in the package (the hot spot is separately gated on
    model.has_hotspot, which requires has_disc too -- no rim for it to
    sit on otherwise).

    The accretion stream is a THIRD, independent knob, gated purely on
    lobe.fill_factor (Roche-lobe overflow at L1) -- NOT on model.has_disc
    -- so render.py's build_temperature_maps computes it whenever
    fill_factor>=1, disc or no: a magnetic CV (an AM Her-type "polar")
    has a real accretion stream with no disc at all, channeled by the
    primary's magnetic field onto its pole instead of spreading into a
    disc (see AMHer.yaml, magnetic.py). A detached (fill_factor<1)
    secondary has no stream regardless of the disc knob, since there's no
    Roche-lobe overflow to feed one.
    """
    from roche import RocheLobe
    from disc import Disc
    from render import disc_powerlaw_teff

    lobe = RocheLobe(system.q, ntheta=ntheta, nphi=nphi,
                      R2=(system.R_2 if system.R_2 > 0.0 else None))

    if model.has_disc:
        disc = Disc(lobe.x1, model.R_out, e_disc=model.e_d, omega_disc=model.omega_d_rad,
                    r_in=model.R_in, opening_angle=model.alpha_d_rad)

        def disc_teff_func(r):
            return disc_powerlaw_teff(r, model.T_0, model.R_in, model.beta_d)
    else:
        disc = Disc(lobe.x1, 0.0, r_in=0.0, opening_angle=0.0)

        def disc_teff_func(r):
            return np.zeros_like(np.asarray(r, dtype=float))

    return lobe, disc, disc_teff_func


# ---- FITS header metadata (shared by simulate.py's light-curve output
# and render.py's save_temperature_maps, so both stay in sync and either
# kind of FITS file can be read back as a parameter source) ----

_HEADER_FIELDS = {
    # keyword: (dataclass, field name, comment)
    "P_ORB": ("system", "P_orb", "d, orbital period"),
    "A": ("system", "a", "m, orbital separation"),
    "Q": ("system", "q", "mass ratio M2/M1"),
    "R1": ("system", "R_1", "primary radius, units of a"),
    "T1": ("system", "T_1", "K, primary effective temperature"),
    "T2": ("system", "T_2", "K, secondary effective temperature"),
    "INCL": ("system", "incl", "deg, orbital inclination"),
    "WAVELEN": ("system", "wavelength", "Angstrom, observing wavelength"),
    "U_PRI": ("system", "u_1", "primary limb-darkening coefficient (linear law)"),
    "U_SEC": ("system", "u_2", "secondary limb-darkening coeff (linear law)"),
    "R2": ("system", "R_2", "units of a, secondary radius (0=lobe-filling)"),
    "PH_OFF": ("system", "ph_off", "phases, subtracted from data before overlay"),
    "THETA1": ("system", "theta_1", "deg, magnetic obliquity from the spin axis"),
    "PHI1": ("system", "phi_1", "deg, primary magnetic-axis azimuth"),
    "R_IN": ("model", "R_in", "units of a, disc inner radius"),
    "R_OUT": ("model", "R_out", "units of a, disc outer radius (semi-major axis)"),
    "T0": ("model", "T_0", "K, disc temperature at R_in"),
    "BETADISC": ("model", "beta_d", "disc temperature power-law index"),
    "T_H": ("model", "T_h", "K, hot-spot peak temperature"),
    "L_H": ("model", "L_h", "deg, hot-spot azimuthal decay length"),
    "BETAGRAV": ("model", "beta_grav", "secondary gravity-darkening exponent"),
    "E_DISC": ("model", "e_d", "disc eccentricity"),
    "OMEGADSC": ("model", "omega_d", "deg, disc periastron orientation"),
    "OPENANG": ("model", "alpha_d", "deg, disc half-opening angle from midplane"),
    "U_DISC": ("model", "u_d", "disc limb-darkening coefficient (linear law)"),
    "ANGLACC": ("model", "angle_acc", "deg, stream angle at field-line connection"),
    "SPOTACC": ("model", "spot_acc", "deg, accretion spot angular radius"),
    "T_ACC": ("model", "T_acc", "K, accretion spot temperature"),
    "U_ACC": ("model", "u_acc", "accretion spot limb-darkening coeff (linear)"),
}

# A standard FITS card is a fixed 80 chars: an <=8-char keyword padded to 8,
# "= " (2), a numeric/bool value right-justified in a fixed 20-char field,
# " / " (3), then the comment -- 47 chars left over regardless of the
# value's own magnitude (still true for a 6-digit float or a bare "T"/"F"
# alike, since the field is padding, not sized to the value). A comment
# over that budget gets silently truncated by astropy (with only a runtime
# VerifyWarning to notice by) rather than raising -- checked here, at
# import time, so a too-long comment added to this table is instead a
# loud, immediate failure.
_HEADER_COMMENT_BUDGET = 80 - 8 - 2 - 20 - 3
_overlong = {kw: comment for kw, (_, _, comment) in _HEADER_FIELDS.items()
             if len(comment) > _HEADER_COMMENT_BUDGET}
assert not _overlong, (
    f"_HEADER_FIELDS comment(s) too long for a FITS card (budget "
    f"{_HEADER_COMMENT_BUDGET} chars) and would be silently truncated: {_overlong}")


def metadata_header(system: SystemParams, model: ModelParams, **extra):
    """
    Build a fits.Header recording every SystemParams/ModelParams value
    used, plus any caller-supplied extra cards. `extra` values are
    either a bare value or a (value, comment) tuple, matching
    astropy.io.fits.Header's own convention -- e.g.
    metadata_header(system, model, IRRADIAT=(True, "irradiation included")).

    Shared by simulate.py's light-curve FITS output and
    render.save_temperature_maps: both are then readable back by
    params_from_fits_header, so either kind of FITS file this package
    writes can be used as a --config-equivalent parameter source.
    """
    from astropy.io import fits

    hdr = fits.Header()
    for keyword, (group, field, comment) in _HEADER_FIELDS.items():
        value = getattr(system if group == "system" else model, field)
        # astropy rejects NaN in a FITS header value, and there's no other
        # native "missing float" card -- so a None field (disc/hot-spot
        # input not given) simply omits its card, and params_from_fits_header
        # reads a missing card back as None the same way.
        if value is not None:
            hdr[keyword] = (value, comment)
    for keyword, value in extra.items():
        hdr[keyword] = value
    return hdr


def params_from_fits_header(header):
    """
    Reconstruct a (SystemParams, ModelParams) pair from a FITS header
    written by metadata_header (in this file's own FITS output, or
    render.save_temperature_maps') -- the reverse of metadata_header.
    """
    sys_kwargs, model_kwargs = {}, {}
    for keyword, (group, field, _comment) in _HEADER_FIELDS.items():
        target = sys_kwargs if group == "system" else model_kwargs
        if keyword in header:
            target[field] = float(header[keyword])
    return SystemParams(**sys_kwargs), ModelParams(**model_kwargs)


def load_fits_header(path):
    """Read a (SystemParams, ModelParams) pair from a FITS file's primary
    HDU header at `path` -- see params_from_fits_header."""
    from astropy.io import fits
    with fits.open(path) as hdul:
        return params_from_fits_header(hdul[0].header)


# ---- YAML persistence ----

def save_yaml(system: SystemParams, model: ModelParams, path):
    """
    Write `system`/`model` to `path` as a two-section YAML file. Values
    are cast through plain float() -- a dataclass's "float" annotation
    isn't enforced at construction, so e.g. a numpy.float64 slipping in
    from some np.sqrt(...) upstream would otherwise reach yaml.safe_dump
    (which doesn't know how to represent it) and crash.
    """
    def _plain(d):
        return {k: (float(v) if v is not None else None) for k, v in d.items()}
    data = {"system": _plain(dataclasses.asdict(system)), "model": _plain(dataclasses.asdict(model))}
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def load_yaml(path):
    """
    Read a YAML file into a (SystemParams, ModelParams) pair. Two formats
    are accepted, auto-detected by top-level shape:

    - a plain 'system'/'model' file (see save_yaml) -- either section may
      be omitted, falling back to that dataclass's own defaults where
      possible (both still require their non-default fields to be
      present somewhere).
    - a gui.py GUI-config file (script_path/panes/arguments, see
      simulation.yaml): since gui.py flags are already named after these
      same dataclass fields (--P_orb, --T_h, ...), each pane argument's
      "default" is picked up directly by matching flag name, so a
      simulation.yaml populated in the GUI and saved via its "Save
      Config..." button can be handed straight back to --config here.
      Arguments whose flag doesn't match a SystemParams/ModelParams field
      (--outputs, --show, --config, ...) are simply ignored, as are
      blank/omitted defaults (same convention gui.py itself uses for
      skipping empty optional fields).
    """
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if "panes" in data:
        return _params_from_gui_config(data)
    # cast explicitly: PyYAML only recognizes scientific notation as a
    # float when the exponent has an explicit sign (5e+8, not 5e8), so an
    # easy-to-write config can otherwise silently hand a *string* to a
    # numeric field. angle_acc alone may be a comma-separated list (see
    # _parse_angle_acc_values) -- _cast_field takes just its first value,
    # same as an explicit --angle_acc CLI override would (see
    # raw_angle_acc_from_config for recovering the rest).
    sys_kwargs = {n: (_cast_field(n, v) if v is not None else None) for n, v in data.get("system", {}).items()}
    model_kwargs = {n: (_cast_field(n, v) if v is not None else None) for n, v in data.get("model", {}).items()}
    return SystemParams(**sys_kwargs), ModelParams(**model_kwargs)


def _cast_field(name, value):
    """
    Cast one SystemParams/ModelParams field's raw config value (a YAML
    scalar, possibly a string -- see load_yaml/_params_from_gui_config)
    to what the dataclass actually stores: a plain float for every field
    except angle_acc, which may be a comma-separated list
    (_parse_angle_acc_values) -- only its first value is a real
    ModelParams.angle_acc (see raw_angle_acc_from_config for recovering a
    config-provided list the same way an explicit --angle_acc CLI
    override's extra entries are).
    """
    if name == "angle_acc":
        return float(_parse_angle_acc_values(str(value))[0])
    return float(value)


def _params_from_gui_config(data):
    """Extract {flag: default} from a gui.py config's panes and cast onto
    whichever SystemParams/ModelParams fields it matches (see load_yaml)."""
    defaults = {}
    for pane in data.get("panes", []):
        for arg in pane.get("arguments", []):
            if arg.get("type") == "text":
                continue
            flag = arg.get("flag", "").lstrip("-")
            default = arg.get("default", "")
            if flag and default not in ("", None):
                defaults[flag] = default

    sys_kwargs = {n: _cast_field(n, defaults[n]) for n in _SYSTEM_FIELDS if n in defaults}
    model_kwargs = {n: _cast_field(n, defaults[n]) for n in _MODEL_FIELDS if n in defaults}
    return SystemParams(**sys_kwargs), ModelParams(**model_kwargs)


def raw_angle_acc_from_config(path):
    """
    Re-read --angle_acc's raw value directly from a YAML config file
    (either format load_yaml accepts) -- bypassing load_yaml/_cast_field's
    "only the first value" reduction, which is what every other consumer
    of a loaded ModelParams needs, but not what simulate.py wants for its
    "outline" output's extra accretion spots/field lines: an --angle_acc
    list given as a config file's own default (not an explicit CLI
    override) is otherwise invisible past ModelParams.angle_acc's single
    float.

    Returns a float array (see _parse_angle_acc_values), or None if the
    file doesn't set angle_acc at all.
    """
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if "panes" in data:
        for pane in data.get("panes", []):
            for arg in pane.get("arguments", []):
                if arg.get("flag", "").lstrip("-") == "angle_acc":
                    default = arg.get("default", "")
                    return _parse_angle_acc_values(str(default)) if default not in ("", None) else None
        return None
    raw = data.get("model", {}).get("angle_acc")
    return _parse_angle_acc_values(str(raw)) if raw is not None else None


# ---- command-line interface ----

_FIELD_HELP = {
    "P_orb": "orbital period [d]",
    "a": "orbital separation [m]",
    "q": "mass ratio, M2/M1",
    "R_1": "primary radius [units of a]",
    "T_1": "primary effective temperature [K]",
    "T_2": "secondary effective temperature [K]",
    "incl": "orbital inclination [deg]",
    "wavelength": "observing wavelength [Angstrom]",
    "u_1": "primary limb-darkening coefficient (linear law, I=I0*(1-u+u*mu)); 0=off",
    "u_2": "secondary limb-darkening coefficient (linear law); 0=off",
    "R_2": "secondary volume-equiv. radius [units of a]; 0 (default) = exactly "
           "Roche-lobe-filling; large values clipped to Roche-lobe filling, "
           "less than the (unknown a priori) filling radius underfills the lobe",
    "ph_off": "phase offset (fraction of P_orb), subtracted from observed data's phase "
              "column before overlay/fitting -- corrects for a bad ephemeris in the data",
    "theta_1": "primary magnetic-axis obliquity from the orbital spin axis [deg] (0=aligned); "
               "assumes synchronous rotation, so the axis is fixed in the corotating frame "
               "-- see magnetic.py. Currently unused (no magnetic field model yet)",
    "phi_1": "primary magnetic-axis azimuth [deg], measured from +x (the sub-secondary "
             "direction at phase 0) -- see magnetic.py. Currently unused",
    "R_in": "disc inner radius [units of a]; leave this and R_out/T_0/beta_d "
            "all unset to skip the disc (and hot spot) entirely",
    "R_out": "disc outer radius (semi-major axis if e_d>0) [units of a]; see R_in",
    "T_0": "disc temperature at R_in [K]; see R_in",
    "beta_d": "disc temperature power-law index, T_d(R)=T_0*(R/R_in)^beta_d; see R_in",
    "T_h": "hot-spot peak temperature [K]; leave this and L_h both unset to skip the hot spot",
    "L_h": "hot-spot azimuthal decay length [deg]; see T_h",
    "beta_grav": "secondary gravity-darkening exponent (Lucy 1967)",
    "e_d": "disc eccentricity (0 = circular)",
    "omega_d": "disc periastron orientation [deg]",
    "alpha_d": "disc half-opening angle from midplane [deg] (0 = flat)",
    "u_d": "disc limb-darkening coefficient (linear law); 0=off",
    "angle_acc": "cumulative swept azimuth [deg] at which the ballistic stream connects "
                 "to a magnetic field line (magnetic CV accretion spot) -- same "
                 "convention as --stream_angle (0=facing secondary, 180=directly behind "
                 "the primary, may exceed 360 to loop around more than once; see "
                 "stream.integrate_stream/angle_acc_index). Unset (default) disables the "
                 "feature. Ignored with a warning if the stream trajectory never actually "
                 "sweeps that far. May be a comma-separated list (e.g. 90,150,200) for "
                 "multiple accretion spots -- one per angle, each drawn as its own field "
                 "line in the outline output and heating its own patch of the primary "
                 "(sharing spot_acc/T_acc/u_acc). Only the first value is used for "
                 "--lsq_fit/the FITS header round-trip",
    "spot_acc": "accretion spot's angular radius on the primary's surface around the "
                "field line's footpoint [deg]; only meaningful if angle_acc is set",
    "T_acc": "accretion spot temperature [K]; only meaningful if angle_acc is set. Ignored "
             "with a warning if not hotter than T_1",
    "u_acc": "accretion spot limb-darkening coefficient (linear law, I=I0*(1-u+u*mu)); "
             "independent of u_1; only meaningful if angle_acc is set. Negative values give "
             "limb-brightening instead of darkening, a crude stand-in for cyclotron beaming",
}

_SYSTEM_FIELDS = [f.name for f in dataclasses.fields(SystemParams)]
_MODEL_FIELDS = [f.name for f in dataclasses.fields(ModelParams)]


def _parse_angle_acc_values(s):
    """
    Shared parsing for --angle_acc's value, wherever it comes from (the
    CLI, via _parse_angle_acc below, or a YAML config's own
    angle_acc/default, via raw_angle_acc_from_config): a single float, or
    a comma-separated list of floats (e.g. '90,150,200') -- see
    _FIELD_HELP['angle_acc']. Always returns a 1D float array (length 1
    for a single value). Raises a plain ValueError on a malformed value --
    callers wrap that in whatever's appropriate for their own context
    (argparse.ArgumentTypeError for _parse_angle_acc, a SystemExit for
    raw_angle_acc_from_config).
    """
    values = [float(v) for v in s.split(",") if v.strip() != ""]
    if not values:
        raise ValueError(f"empty --angle_acc value: {s!r}")
    return np.array(values)


def _parse_angle_acc(s):
    """
    --angle_acc's CLI type: see _parse_angle_acc_values. The caller
    (simulate.py's main()) pulls out the first entry for
    ModelParams.angle_acc (every other --angle_acc-touching thing --
    --lsq_fit, the FITS header round-trip -- only ever sees that single
    float, unchanged from before this list-supporting type existed) and
    treats every entry, including that first one, as its own accretion
    spot: one heated patch (and one drawn field line, in simulate.py's
    "outline" output) per angle, all sharing ModelParams'
    spot_acc/T_acc/u_acc.
    """
    try:
        return _parse_angle_acc_values(s)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--angle_acc must be a float or comma-separated list of floats, got {s!r}")


def add_param_args(parser):
    """
    Register --config plus one --<field> flag per SystemParams/ModelParams
    field (all float-valued, default None so "not given on the command
    line" is distinguishable from "given as 0" -- except --angle_acc,
    which takes a comma-separated list, see _parse_angle_acc). Call
    params_from_args on the resulting namespace to resolve config-file +
    CLI-override values into a (SystemParams, ModelParams) pair.
    """
    parser.add_argument("--config", type=str, default=None,
                         help="YAML file with 'system'/'model' sections "
                              "(base values; the flags below override it)")
    for name in _SYSTEM_FIELDS + _MODEL_FIELDS:
        parser.add_argument(f"--{name}", type=(_parse_angle_acc if name == "angle_acc" else float),
                             default=None, help=_FIELD_HELP.get(name, name))
    return parser


def _drop_mismatched_overrides(overrides, base_obj, exempt, source_label):
    """
    Used by params_from_args(..., strict=True): any override in
    `overrides` that isn't in `exempt` and differs (beyond float
    round-off) from base_obj's own value for that field is dropped, with
    a warning -- rather than silently changing a value that the cached
    data in `source_label` was computed for.
    """
    import sys as _sys

    kept = {}
    for name, value in overrides.items():
        if name in exempt:
            kept[name] = value
            continue
        base_value = getattr(base_obj, name)
        # base_value is None means that field wasn't part of the cached
        # run at all (no disc/hot spot there) -- any override supplying a
        # real value for it is a categorical change, so it's dropped too.
        if base_value is not None and math.isclose(value, base_value, rel_tol=1e-9, abs_tol=1e-12):
            kept[name] = value
        else:
            print(f"warning: --{name}={value} differs from {source_label}'s stored value "
                  f"({base_value}) and was ignored -- the loaded irradiation results were "
                  f"computed for the stored value, so only wavelength (and phase) may safely "
                  f"differ from it", file=_sys.stderr)
    return kept


def params_from_args(args, base=None, strict=False, exempt=()):
    """
    Resolve a (SystemParams, ModelParams) pair from an argparse namespace
    built with add_param_args: start from `base` if given, else --config
    (if given), then overwrite with any field whose CLI flag was actually
    set. If neither is given, every required field must be supplied on
    the command line.

    base: an optional pre-resolved (SystemParams, ModelParams) pair to
    use instead of --config -- e.g. loaded from a FITS file via
    render.load_temperature_maps, so a cached irradiation calculation's
    own system/model can still be overridden field-by-field (say,
    --wavelength for a different-band light curve) without needing a
    --config file too.

    strict: only meaningful together with `base`. If True, any CLI
    override that differs from base's own value for that field is
    dropped (with a warning to stderr) instead of applied -- except
    fields named in `exempt`, which always apply normally. Used by
    --load-irradiation: the loaded system/model describes exactly what
    the cached temperature maps were computed for, so changing anything
    but wavelength (downstream/cheap, per render.build_temperature_maps)
    would silently invalidate the cache without this guard.
    """
    sys_overrides = {n: getattr(args, n) for n in _SYSTEM_FIELDS if getattr(args, n) is not None}
    model_overrides = {n: getattr(args, n) for n in _MODEL_FIELDS if getattr(args, n) is not None}

    try:
        if base is not None:
            system, model = base
            if strict:
                sys_overrides = _drop_mismatched_overrides(
                    sys_overrides, system, exempt, "the loaded irradiation cache")
                model_overrides = _drop_mismatched_overrides(
                    model_overrides, model, exempt, "the loaded irradiation cache")
            system = dataclasses.replace(system, **sys_overrides)
            model = dataclasses.replace(model, **model_overrides)
        elif args.config:
            system, model = load_yaml(args.config)
            system = dataclasses.replace(system, **sys_overrides)
            model = dataclasses.replace(model, **model_overrides)
        else:
            system = SystemParams(**sys_overrides)
            model = ModelParams(**model_overrides)
    except TypeError as e:
        raise SystemExit(f"missing required parameter(s) (pass --config or the CLI flag): {e}")

    return system, model
