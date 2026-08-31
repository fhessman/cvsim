# cvsim

A Roche-lobe binary-star eclipse simulator for cataclysmic variables (CVs),
including magnetic CVs ("polars"). 

This collection of python scripts was written with the help of Claude Code
to be a tool for various tasks associated with observations of CVs, including
the graphical representation of the geometries, the calculation of realistic
lightcurves and radial velocity curves including the realistic effects on
eclipses, fitting lightcurve or rv data, and even creating primitive
"photographic" images of the systems.

The main script -- simulation.py -- can either be invoked as a classic python
script or it can be run using a convenient GUI.  The parameters for a 
particular simulation are kept in YAML files that can be read either by the
script directly or indirectly via the GUI.

This document summarizes the physical
model, how the YAML config files work, the available
outputs, and the parameters behind every pane in those configs.

## 1. Physical background

Everything is computed in the frame **corotating** with the binary (units
`a=1`, `G(M1+M2)=1`, orbital angular velocity `Omega=1`, origin at the
center of mass); physical values (meters, Kelvin, seconds) only appear at
the config/CLI boundary and get converted internally.

- **Roche geometry** (`roche.py`) -- the primary (accretor) sits at
  `x1 = -q/(1+q)`, the secondary at `x2 = 1/(1+q)`, `q = M2/M1`. The
  secondary's surface is either its Roche lobe (the `Phi = Phi(L1)`
  equipotential through the inner Lagrangian point L1) or, if `R_2` is set
  below the lobe-filling volume-equivalent radius, a smaller detached
  star -- `R_2=0` (the default) always means exactly lobe-filling, the
  defining condition for a mass-transferring CV.
- **Accretion stream** (`stream.py`) -- a ballistic stream leaving
  the L1 nozzle, following Lubow & Shu (1975). The initial velocity is set by
  the local sound speed, then integrated under gravity + Coriolis +
  centrifugal forces in the corotating frame. By default, the trajectory
  is drawn/used up to its first closest approach to the primary (a stand-in
  for "where the disc/field would take over"); `--stream_angle` can extend
  it further (even multiple loops around the primary) when the stream should
  be stopped at the disc rim or at particular magnetic field line, e.g. when a magnetic
  connection point lies beyond that first approach.
- **Accretion disc** (`disc.py`, optional) -- considered only if `R_in`,
  `R_out`, `T_0`, and `beta_d` are *all* given. The rim is a Kepler
  ellipse focused on the primary (semi-major axis, eccentricity,
  periapsis angle), and surface temperature follows a radial power law
  `T(R) = T_0*(R/R_in)^beta_d`. A disc can optionally be given nonzero
  thickness defined by non-zero constant opening angle `beta_d`.
  If the `T_acc`? and `L_acc`?
  parameters are given, then a "hot spot" on the disc's outer rim starting
  at the accretion stream's impact point is created, where the 
  disc rim's temperature is exponentially decreasing with an angular scale ?.
- **Magnetic channeling / "polar" accretion** (`magnetic.py`) -- used for
  simulating a system where the
  primary's magnetic field disrupts disc formation (AM Her being
  the prototype). The modelled field is a simple tilted dipole (obliquity `theta_1`,
  azimuth `phi_1`, ususally fixed in the corotating frame under the standard
  synchronous-rotation assumption). If an "accretion connection radius"
  (`r_acc`) is set, the ballistic stream is assumed to hand off to the
  field there, and the connecting field line carries material onward to a
  heated spot on the primary's surface -- this spot has its own
  temperature, angular size, and (independent) limb-darkening/brightening
  law, and irradiates the secondary like any other hot component.
- **Irradiation** (`irradiation.py`) -- full view-factor integrals (not a
  point-source approximation) for how much the disc/primary heat the
  secondary's surface, including partial occultation of the primary by
  the disc as seen from a given secondary point. This is the expensive
  part of a calculation, so it can be cached to a FITS file and reused
  (`--save-irradiation`/`--load-irradiation`).
- **Sky projection / eclipses** (`eclipse.py`) -- at phase 0 the secondary
  is between the primary and the observer (mid-eclipse convention). Every
  phase-dependent output projects the corotating-frame geometry onto the
  sky plane for the given inclination and tests occultation by the
  secondary's Roche lobe (and, if present, the disc).

## 2. Output possibilities (`--outputs`, comma-separated)

| Output | What it produces |
| --- | --- |
| `outline` | Sky-projected geometry (Roche lobe, disc, stream, dipole field-line loops, accretion-spot connection) at each requested phase -- one PNG per phase. |
| `temperature` | Rendered surface-temperature image (K) at each phase -- one PNG per phase. |
| `intensity` | Rendered surface-brightness image (band intensity, W/m^2/sr/m), the one that actually shows limb-darkening effects -- one PNG per phase. |
| `lightcurve` | The eclipse light curve as relative flux [mJy] vs. orbital phase, all phases on one plot, with a mirrored AB-magnitude axis; overlays observed data if `--data-file` is given. |
| `magnitude` | Same underlying computation as `lightcurve`, plotted with AB magnitude as the primary axis instead (whichever the observed data's own units are takes the primary axis automatically, when data is given). `lightcurve`/`magnitude` together (or either alone) also write one FITS binary table with both units, phase-resolved, plus every system/model parameter used in the header. |
| `shadow` | Top-down (face-on) view of the eclipse footprint: which parts of the disc/stream are shadowed/occulted, swept across the requested phase range. |
| `rv` | Radial-velocity curves (km/s) for the primary, secondary, accretion stream, and magnetically-channeled stream, computed *separately* (intensity-and-limb-darkening-weighted median line-of-sight velocity of each component's visible material), plus the two stars' circular Roche-point velocities as reference. Overlays observed RV data if any `--data_rv_*_col` is given. |
| `corner` | Not a `--outputs` choice itself -- produced automatically by `--mcmc_fit` when `--corner_plot` is set: the classic posterior pairwise-correlation plot. |

`--show` displays each output interactively instead of writing it to
`--outdir`. `--phase-num=1` (the default) uses a single phase;
`--phase-num>1` sweeps `linspace(--phase-min, --phase-max, --phase-num)` --
`outline`/`temperature`/`intensity` then write one file per phase, while
`lightcurve`/`magnitude`/`shadow`/`rv` always plot every phase together on
one figure regardless.

## 3. How the YAML config files work

Each config (`examples/*.yaml`, once published) is consumed two ways:

- **`gui.py`** (a generic PySide6 form-builder that can be used for 
  running any python script) reads the *entire* file: a
  `script_path` (the script to run, resolved relative to the config file's
  own directory if not absolute), and a list of `panes`, each with a
  `title` and an `arguments` list. Each argument becomes one form field
  (`flag`, `label`, `type`, `default`, optional `tip`/`choices`), and
  pressing "Run" assembles `python <script_path> <flag1> <value1> ...`
  from whatever the form currently holds, using every field's live value
  -- not just the YAML's own recorded defaults.
- **`simulate.py --config file.yaml`** (bare CLI, no GUI) reads the same
  file but only pulls defaults for fields that are actual
  `SystemParams`/`ModelParams` dataclass fields (see `params.py`):
  `P_orb, a, q, R_1, T_1, T_2, incl, wavelength, u_1, u_2, R_2, ph_off,
  theta_1, phi_1` (SystemParams) and `R_in, R_out, T_0, beta_d, T_h, L_h,
  beta_grav, e_d, omega_d, alpha_d, u_d, r_acc, angle_acc, T_acc, u_acc`
  (ModelParams).

**The gotcha:** every *other* field in a config -- `--outputs`,
`--phase-min/-max/-num`, `--n_areas_*`, `--n_field_1`, `--stream_angle`,
`--outdir`, `--prefix`, `--save-irradiation`/`--load-irradiation`, every
`--data*`/`--data_rv_*` flag, `--lsq_fit`/`--mcmc_fit` and its knobs,
`--no-irradiate`, `--workers`, `--pixelmapping`, `--image-size`,
`--vmin`/`--vmax` -- is a **plain** argparse flag, not a dataclass field.
A bare `python simulate.py --config file.yaml` does **not** pick up that
field's YAML default; those only take effect when the config is driven
through `gui.py`, or when you pass the flag explicitly on the command
line. (Every `SystemParams`/`ModelParams` field is also available as its
own `--<field_name>` flag on the plain CLI, config or not.)

## 4. Parameters by pane

### SYSTEM
The binary's physically "given" parameters: `P_orb` [d], `a` [m] (orbital
separation), `q` (mass ratio M2/M1), `incl` [deg], `dist` [pc] (scales the
light curve to a real flux density), `wavelength` [Angstrom].

### PHASES
Shared by every phase-dependent output: `phase-min`/`phase-max` (a single
phase if `phase-num=1`, else the sweep range), `phase-num`.

### PRIMARY
`R_1` [units of a], `T_1` [K], `u_1` (limb-darkening coefficient -- also
governs how strongly the primary irradiates the secondary, so it's baked
into a saved irradiation cache), `n_areas_1` (surface-sampling
resolution). Magnetic/accretion-spot knobs: `theta_1`/`phi_1` [deg]
(dipole obliquity/azimuth), `n_field_1` (cosmetic field-line-loop count
for the outline output), `r_acc` [units of a] (accretion connection
radius/radii -- comma-separated for more than one; the feature-enabling
field), `angle_acc` [deg] (spot angular radius), `T_acc` [K] (spot
temperature -- ignored with a warning if not hotter than `T_1`), `u_acc`
(spot's own, independent limb-darkening/brightening coefficient).

### SECONDARY
`R_2` [units of a] (0 = exactly Roche-lobe-filling; larger values are
clipped to lobe-filling), `T_2` [K], `u_2`, `beta_grav` (gravity-darkening
exponent, Lucy 1967), `n_areas_2` (surface-sampling resolution).

### DISC
`R_in`/`R_out` [units of a], `T_0` [K], `beta_d` (temperature power-law
index) -- the disc is built only if all four are given. Optional shape
extras: `alpha_d` [deg] (half-opening angle, 0 = flat), `e_d`
(eccentricity), `omega_d` [deg] (periastron orientation), `u_d`
(limb-darkening), `n_areas_d` (surface-sampling resolution).

### STREAM
`T_h`/`L_h` -- the disc-rim stream-impact hot spot (needs a disc, plus
both of these given). `stream_angle` [deg] -- how far around the primary
the ballistic stream is integrated before stopping (blank = stop at first
closest approach; set higher, up to and beyond 360, to reach an
accretion-connection radius further along the trajectory). Every stream
integration prints its closest approach and final radius to help pick
that radius.

### OUTPUT
`outputs` (comma-separated choice list, see section 3), `show`,
`show-points` (scatter each body's actual sample points, to judge
resolution), `outdir`, `prefix` (defaults to the config's basename),
`save-irradiation`/`load-irradiation` (FITS cache of the expensive
irradiation calculation -- loading one only lets `wavelength`/`u_2` still
safely vary; anything else that would change the cached irradiation is
ignored with a warning).

### DATA
Photometric overlay/fit: `data-file`, `data-phase-col`, exactly one of
`data-flux-col`/`data-mag-col`, `data-err-col`. Radial-velocity
overlay/fit (independent, any subset may be active): `data-rv-file`
(defaults to `data-file` if blank), `data-rv-phase-col`, and four
column/error-column pairs -- `data_rv_1` (primary), `data_rv_2`
(secondary, the usual source of a CV's measured RV curve), `data_rv_stream`,
`data_rv_hotspot` (the magnetically-channeled stream) -- each independently
enabling that component's overlay and fit contribution. `data_rv_gamma`
[km/s] -- a single systemic-velocity offset added to every model RV curve
before comparing to any of the above (the model itself has zero systemic
velocity).

### FIT
`ph_off` (phase offset subtracted from the data before
overlay/fitting -- corrects a bad ephemeris in the data, not the model),
`data_norm` (multiplicative flux factor or additive magnitude offset
applied to the data before comparing). `lsq_fit`/`mcmc_fit`
(comma-separated parameter names -- any `SystemParams`/`ModelParams`
field, plus the pseudo-parameters `dist`, `data_norm`, `rv_gamma` -- fit
by least squares or MCMC against whichever of `data-file`/any
`data_rv_*_col` is active, jointly if more than one; mutually exclusive
with each other). MCMC-only: `nburn`, `nsample`, `walkers` (per
parameter), `spread` (starting-ball dispersion), `corner_plot`.

### MISC
`no-irradiate` (cheaper gravity-darkening-only preview), `workers`
(parallelize the light-curve phase loop), `pixelmapping` (`direct` vs.
`indirect` sample-to-pixel rendering strategy), `image-size` (output PNG
pixel dimensions), `vmin`/`vmax` (colorbar limits for the
temperature/intensity renders).
