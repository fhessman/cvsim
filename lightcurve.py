# lightcurve.py
"""
Flux components (central star, accretion disc, accretion stream) and
their assembly into a synthetic eclipse light curve, using the Roche-lobe
occultation test in eclipse.py.

Each component is represented as a set of discrete emitting sample points
(a "flat" limb-darkened disc for the star, an (r,nu) grid for the disc,
points along the ballistic trajectory for the stream); at a given orbital
phase, each point's visibility is determined by eclipse.visible_fraction
(a smooth 0..1 ramp across the eclipse limb, occulted by the secondary's
Roche lobe) and the total flux is the occultation-weighted sum.
"""

import numpy as np

from eclipse import observer_frame, visible_fraction, visible_fraction_bulk
from disc import Disc
from stream import integrate_stream, disc_impact_index, sample_points


def star_points(center, radius, phase, incl, n_rho=10, n_phi=24, limb_u=0.0,
                 return_area=False, frame=None):
    """
    Sample points + limb-darkening-and-area weights on the true visible
    hemisphere of a limb-darkened sphere of `radius` (units a) centered at
    `center` (corotating 3-vector). Weights are normalized so
    sum(weights) = 1.

    Equal-projected-area point cloud rather than a regular (alpha,beta)
    grid: alpha is the angle from the sub-observer point (0 at disc
    center, pi/2 at the limb), beta the azimuth around the line of sight,
    and the projected (foreshortened) area element is
    R^2*sin(alpha)*cos(alpha) dalpha dbeta -- exactly the same annulus
    structure (in the projected radius rho=R*sin(alpha)) as
    disc.equal_area_annulus, so it gets the same golden-angle-spiral
    treatment: beta swept via the golden angle, mu=cos(alpha)=sqrt(1-t)
    placed via a uniform area-fraction t=(i+0.5)/n (exact, since a sphere
    -- unlike an eccentric disc rim -- has no direction-dependent outer
    boundary). A fixed (n_rho,n_phi) grid puts far more physical area per
    cell near the limb (where sin(alpha)cos(alpha) is small) than near
    disc-center, giving a non-uniform sky-plane point density once
    rendered (the same "moire" problem fixed elsewhere -- see
    disc.equal_area_annulus's docstring); this doesn't. mu is exactly the
    standard limb-darkening cosine, and I0*(1-u+u*mu) summed over n equal-
    area cells (each pi*R^2/n) is a quadrature estimate of the standard
    limb-darkened-disc total flux I0*pi*R^2*(1-u/3).

    n_rho, n_phi are kept only as the calling convention -- their product
    is the total point count that matters now.

    Deliberately uses true 3D sphere positions (bounded |offset| <= radius
    in every direction, for any viewing angle) rather than a flat
    tangent-plane facet: the facet's linearized height off the sphere's
    true surface is fine for Roche-lobe ray-marching (insensitive to that
    O(radius^2) error) but gets catastrophically amplified by the
    disc-plane intersection test in disc.Disc.visible_from (which divides
    by cos(inclination)), producing unphysical results near edge-on.

    return_area (default False): also return each point's own raw
    (un-normalized, limb-darkening-independent) projected area element
    pi*R^2/n [units a^2] -- e.g. for render.py's area-aware "direct"
    pixel-mapping image binning, which needs actual physical footprint
    sizes rather than a flux-normalized weight.

    frame: pass a caller's already-computed eclipse.observer_frame(phase,
    incl) to skip recomputing it (e.g. render._phase_chunk_flux already
    has one on hand for every phase).

    Returns (points (N,3), weights (N,)), or (points, weights, area) if
    return_area.
    """
    n = n_rho * n_phi
    n_hat, eX, eY = frame if frame is not None else observer_frame(phase, incl)
    i = np.arange(n)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    beta = np.mod(i * golden_angle, 2.0 * np.pi)
    t = (i + 0.5) / n
    mu = np.sqrt(1.0 - t)
    sinA = np.sqrt(t)

    I = (1.0 - limb_u + limb_u * mu)
    proj_area = np.full(n, np.pi * radius ** 2 / n)
    w = I * proj_area
    w = w / w.sum()

    pts = (np.asarray(center)
           + radius * mu[:, None] * n_hat
           + radius * (sinA * np.cos(beta))[:, None] * eX
           + radius * (sinA * np.sin(beta))[:, None] * eY)
    if return_area:
        return pts, w, proj_area
    return pts, w


def sphere_points_full(center, radius, n):
    """
    Roughly equal-area point cloud over a sphere's WHOLE surface -- unlike
    star_points, which only samples the hemisphere visible from a given
    observer direction (fine for a flux-toward-Earth calculation, wrong
    for irradiating another body, where every direction the sphere emits
    in matters, not just the Earth-facing half). Same golden-angle-spiral
    equal-area technique as roche.RocheLobe.equal_area_sample, specialized
    to a sphere's constant radius (no direction-dependent surface to look
    up). Returns (points (N,3), outward unit normals (N,3), areas (N,),
    each area exactly 4*pi*radius**2/n).
    """
    i = np.arange(n)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    z = 1.0 - 2.0 * (i + 0.5) / n
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    phi = np.mod(i * golden_angle, 2.0 * np.pi)
    sinT = np.sin(theta)
    normal = np.stack([sinT * np.cos(phi), sinT * np.sin(phi), z], axis=-1)
    points = np.asarray(center) + radius * normal
    areas = np.full(n, 4.0 * np.pi * radius ** 2 / n)
    return points, normal, areas


def star_flux(center, radius, phase, incl, lobe, disc=None, flux_norm=1.0,
              limb_u=0.0, n_rho=10, n_phi=24, frame=None):
    """
    Occulted flux of the central star (0..flux_norm): eclipsed by the
    secondary's Roche lobe, and -- if `disc` is given -- also by the
    disc's own material (disc.Disc.visible, which dispatches between the
    flat zero-thickness plane test and the flared 3D solid's ray march
    depending on disc.opening_angle -- see disc.py). Both tests are
    independent "is this bit of star hidden" checks, so their visible
    fractions multiply. If the disc's inner radius is large enough that
    it never reaches the star's vicinity, `disc.visible` naturally
    returns all-visible and this reduces to Roche-lobe eclipse only.

    frame: pass a caller's already-computed eclipse.observer_frame(phase,
    incl) to skip recomputing it here, in star_points, and in
    visible_fraction_bulk -- see their docstrings.
    """
    if frame is None:
        frame = observer_frame(phase, incl)
    pts, w = star_points(center, radius, phase, incl, n_rho, n_phi, limb_u, frame=frame)
    cell_size = radius / n_rho
    # the primary's own points never enter the L1 funnel, so the fast
    # hull-based shortcut is exact here (see visible_fraction_bulk) --
    # unlike visible_fraction's stream-safe general ray march
    vis = visible_fraction_bulk(pts, phase, incl, lobe, cell_size, frame=frame)
    if disc is not None:
        vis = vis * disc.visible(pts, phase, incl)
    return flux_norm * np.sum(w * vis)


def disc_flux(disc, phase, incl, lobe, n_r=40, n_nu=80, n_z=15, frame=None):
    """
    Occulted flux of the accretion disc: the flat zero-thickness sheet if
    disc.opening_angle==0 (unchanged original behavior), else the full
    flared solid (four surfaces, self-occlusion-aware, one-sided
    Lambertian emission foreshortened by the local surface normal) -- see
    System.prepare()/component_fluxes_prepared() for the phase-dependence
    this implies (the flared disc's per-point flux weight depends on the
    observer direction, unlike the flat disc's phase-independent one).

    frame: see star_flux's docstring.
    """
    if frame is None:
        frame = observer_frame(phase, incl)
    if disc.opening_angle > 0.0:
        x, y, z, normals, areas, r, _sid = disc.all_surfaces_grid(n_r=n_r, n_nu=n_nu, n_z=n_z)
        pts = np.stack([x, y, z], axis=-1)
        n_hat = frame[0]
        cos_obs = np.clip(np.einsum("ij,j->i", normals, n_hat), 0.0, None)
        f = disc.surface_brightness(r) * areas * cos_obs
        pts_test = pts + 1e-6 * normals
        # disc points never enter the L1 funnel either -- same shortcut as
        # star_flux (see visible_fraction_bulk)
        vis = visible_fraction_bulk(pts, phase, incl, lobe, (disc.a - disc.r_in) / n_r, frame=frame)
        # disc's own surface points are always inside the disc's own hull,
        # so the hull pre-filter in occluded_by_disc can never reject any
        # of them -- skip it (use_hull=False) to avoid pure overhead
        vis = vis * disc.visible(pts_test, phase, incl, use_hull=False)
        return np.sum(f * vis)
    x, y, f = disc.grid(n_r=n_r, n_nu=n_nu)
    f = f * np.cos(incl)  # flat disc's constant foreshortening (normal=z_hat everywhere)
    pts = np.stack([x, y, np.zeros_like(x)], axis=-1)
    cell_size = (disc.a - disc.r_in) / n_r
    vis = visible_fraction_bulk(pts, phase, incl, lobe, cell_size, frame=frame)
    return np.sum(f * vis)


def stream_points_and_weights(traj, x1, s_impact, brightness=1.0,
                               hotspot_amp=2.0, hotspot_width=0.05, n=40):
    """
    Sample points along the ballistic stream from L1 up to the disc-impact
    arclength s_impact, with a uniform "stream" brightness plus an
    additional Gaussian "hot spot" concentrated near the impact point
    (the classic two-component stream + shock-heated hot-spot picture).

    Returns (points (n,3), weights (n,)); weights integrate (trapezoidally
    in arclength) to `brightness` (stream) + `hotspot_amp*brightness`
    (spot), i.e. flux_norm-style total when unocculted.
    """
    s = np.linspace(0.0, s_impact, n)
    x, y = sample_points(traj, s)
    pts = np.stack([x, y, np.zeros_like(x)], axis=-1)

    ds = np.gradient(s)
    uniform = ds / s_impact * brightness
    spot = np.exp(-0.5 * ((s - s_impact) / hotspot_width) ** 2)
    spot = spot / np.sum(spot * ds) * hotspot_amp * brightness
    weights = uniform + spot * ds
    return pts, weights


def stream_flux(points, weights, phase, incl, lobe, cell_size):
    vis = visible_fraction(points, phase, incl, lobe, cell_size)
    return np.sum(weights * vis)


class System:
    """
    Convenience wrapper bundling a fixed RocheLobe (set by q, expensive to
    rebuild) with the flux-component parameters, producing full model
    light curves. `q` and `incl_deg` are the "given" (not fitted)
    parameters; everything passed via `params` in light_curve() is
    typically what gets fitted to data.
    """

    def __init__(self, lobe, incl_deg):
        self.lobe = lobe
        self.incl = np.radians(incl_deg)

    def prepare(self, p):
        """
        Build everything that does NOT depend on orbital phase: the disc
        surface grid (and its pre-eclipse flux weights) and the stream
        sample points/weights (which require integrating the ballistic
        trajectory and locating the disc-impact point). Reused across all
        phases in light_curve()/component_fluxes(), and across repeated
        calls during a fit as long as p['a_disc'] & friends are unchanged.

        p: dict with keys
            R1, Flux1, limb_u1                     (central star)
            a_disc, e_disc, omega_disc, r_in, p_disc, i_norm  (disc)
            opening_angle                           (disc thickness; 0=flat)
            eps, stream_brightness, hotspot_amp, hotspot_width  (stream)
            n_r, n_nu, n_z                          (disc grid resolution)

        For a flared disc (opening_angle>0), the per-point flux weight
        depends on the observer direction (each surface's own tilted
        normal foreshortens differently as the system rotates), unlike
        the flat disc's phase-independent weight -- so here we only
        precompute the phase-independent geometry (positions, normals,
        areas*surface_brightness(r)); component_fluxes_prepared() applies
        the phase-dependent foreshortening and occultation.
        """
        x1 = self.lobe.x1
        disc = Disc(x1, p["a_disc"], p.get("e_disc", 0.0), p.get("omega_disc", 0.0),
                    r_in=p.get("r_in", 2 * p["R1"]), brightness_index=p.get("p_disc", 0.75),
                    i_norm=p["i_norm"], opening_angle=p.get("opening_angle", 0.0))
        n_r, n_nu = p.get("n_r", 40), p.get("n_nu", 80)

        if disc.opening_angle > 0.0:
            n_z = p.get("n_z", 15)
            dx, dy, dz, dnormals, dareas, dr, _dsid = disc.all_surfaces_grid(
                n_r=n_r, n_nu=n_nu, n_z=n_z)
            disc_pts = np.stack([dx, dy, dz], axis=-1)
            disc_normals = dnormals
            disc_base = disc.surface_brightness(dr) * dareas
        else:
            dx, dy, df = disc.grid(n_r=n_r, n_nu=n_nu)
            disc_pts = np.stack([dx, dy, np.zeros_like(dx)], axis=-1)
            disc_normals = None
            disc_base = df

        stream_pts, stream_w = None, None
        if p.get("stream_brightness", 0.0) > 0:
            traj = integrate_stream(self.lobe, eps=p.get("eps", 0.02))
            idx = disc_impact_index(traj, disc.rim, x1)
            if idx is not None and traj["s"][idx] > 0:
                stream_pts, stream_w = stream_points_and_weights(
                    traj, x1, traj["s"][idx],
                    brightness=p["stream_brightness"],
                    hotspot_amp=p.get("hotspot_amp", 2.0),
                    hotspot_width=p.get("hotspot_width", 0.05),
                )

        cell_disc = (disc.a - disc.r_in) / n_r
        cell_stream = (traj["s"][idx] / len(stream_pts)) if stream_pts is not None else None

        return {"x1": x1, "disc": disc, "disc_pts": disc_pts, "disc_normals": disc_normals,
                "disc_base": disc_base, "cell_disc": cell_disc,
                "stream_pts": stream_pts, "stream_w": stream_w, "cell_stream": cell_stream}

    def component_fluxes_prepared(self, phase, p, prep):
        """Component fluxes at one phase, given prep=self.prepare(p)."""
        frame = observer_frame(phase, self.incl)
        star_center = np.array([prep["x1"], 0.0, 0.0])
        f_star = star_flux(star_center, p["R1"], phase, self.incl, self.lobe,
                            disc=prep["disc"], flux_norm=p["Flux1"], limb_u=p.get("limb_u1", 0.0),
                            n_rho=p.get("n_rho", 10), n_phi=p.get("n_phi", 24), frame=frame)

        disc = prep["disc"]
        disc_pts = prep["disc_pts"]
        if prep["disc_normals"] is not None:
            n_hat = frame[0]
            cos_obs = np.clip(np.einsum("ij,j->i", prep["disc_normals"], n_hat), 0.0, None)
            f_disc_weighted = prep["disc_base"] * cos_obs
            pts_test = disc_pts + 1e-6 * prep["disc_normals"]
            # disc points, so the fast shortcut applies (see star_flux/
            # visible_fraction_bulk) -- unlike the stream below.
            # use_hull=False: these are the disc's OWN points (self-occlusion),
            # always inside its own hull by construction, so the hull
            # pre-filter in occluded_by_disc would only add overhead here.
            vis = (visible_fraction_bulk(disc_pts, phase, self.incl, self.lobe, prep["cell_disc"], frame=frame)
                   * disc.visible(pts_test, phase, self.incl, use_hull=False))
        else:
            f_disc_weighted = prep["disc_base"] * np.cos(self.incl)
            vis = visible_fraction_bulk(disc_pts, phase, self.incl, self.lobe, prep["cell_disc"], frame=frame)
        f_disc = np.sum(f_disc_weighted * vis)

        f_stream = 0.0
        if prep["stream_pts"] is not None:
            vis_s = visible_fraction(prep["stream_pts"], phase, self.incl, self.lobe, prep["cell_stream"])
            f_stream = np.sum(prep["stream_w"] * vis_s)
        return f_star, f_disc, f_stream

    def component_fluxes(self, phase, p):
        """Convenience one-off (rebuilds disc/stream setup every call)."""
        return self.component_fluxes_prepared(phase, p, self.prepare(p))

    def light_curve(self, phases, p):
        prep = self.prepare(p)
        star = np.empty(len(phases))
        disc = np.empty(len(phases))
        stream = np.empty(len(phases))
        for i, ph in enumerate(phases):
            star[i], disc[i], stream[i] = self.component_fluxes_prepared(ph, p, prep)
        return star, disc, stream, star + disc + stream
