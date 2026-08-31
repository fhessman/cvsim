# eclipse.py
"""
Sky projection and eclipse (occultation-by-the-secondary) geometry.

Observer frame
--------------
At orbital phase 0 (cycles, i.e. phase in [0,1)) the secondary is between
the primary and the observer (mid-eclipse of the primary/disc), the
standard convention for eclipsing CVs/LMXBs.  The observer direction in
the corotating (x,y,z) frame is

    n(phase, incl) = (sin(incl) cos(2*pi*phase),
                      -sin(incl) sin(2*pi*phase),
                       cos(incl))

i.e. n sweeps clockwise (as seen from +z) as phase increases -- the
correct sense for a fixed external observer viewed from a frame
corotating at Omega=+1 (CCW as seen from +z, roche.py/stream.py's
convention: a point fixed in the inertial frame appears, from within a
frame rotating with the system, to sweep in the *opposite* sense to the
system's own rotation). Getting this backwards is invisible to
eclipse-timing (a circular orbit's ingress/egress are phase-symmetric
either way) but shows up as a mirrored light curve for any genuinely
phase-asymmetric feature, e.g. the accretion-stream hot spot.

with incl the orbital inclination [rad] (90 deg = edge-on).  Sky-plane axes
(e_X, e_Y) are built by Gram-Schmidt against the orbital rotation axis
z-hat, so e_Y is the sky-projected pole ("north") and e_X = e_Y x n
completes a right-handed (e_X, e_Y, n) frame; "depth" along n increases
towards the observer.

Occultation test
-----------------
A point P (corotating frame) is eclipsed by the secondary's Roche lobe at
a given phase iff the sky ray from P towards the observer, P + t*n_hat
(t>0), passes through the lobe's solid volume.  Using the lobe's tabulated
radius-vs-direction r_lobe(theta,phi) (roche.RocheLobe), define along the
ray

    g(t) = |P + t*n - center2| - r_lobe(direction of (P+t*n-center2))

g<0 inside the lobe, g>0 outside.  The point is occulted iff min_{t>0} g(t)
< 0.  This is a well-defined, continuous (though non-smooth via the min)
function of orbital phase, so ingress/egress are simply its zero-crossings
-- found here by bracketing + Brent's method for high-precision contact
times.
"""

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.spatial import ConvexHull
from matplotlib.path import Path


def observer_frame(phase, incl):
    """
    Return (n_hat, eX_hat, eY_hat) for orbital phase [cycles] and
    inclination [rad].

    eY (the sky-projected pole, "north") is built by Gram-Schmidt against
    z_hat -- eY = z_hat - (z_hat.n)*n, normalized -- which algebraically
    reduces (since z_hat.n = cos(incl) identically) to exactly

        eY = (-cos(incl)*cos(ph), cos(incl)*sin(ph), sin(incl))

    already unit length for every (phase, incl), so it's used directly
    rather than via a normalized-difference division: at incl=0 or pi
    (observer looking straight down the rotation axis) z_hat is parallel
    to n, so the difference vector vanishes identically and the naive
    division is 0/0 -- this closed form has no such singularity, and
    matches the Gram-Schmidt result exactly everywhere it IS defined, so
    it's not a special-cased fallback, just a nicer way to compute the
    same thing. It keeps the projected image continuous through i=0/180
    (still rotating with phase, as the i->0 limit of the general formula
    does) rather than freezing to some arbitrary fixed frame there.
    """
    ph = 2.0 * np.pi * np.asarray(phase, dtype=float)
    si, ci = np.sin(incl), np.cos(incl)
    n = np.stack([si * np.cos(ph), -si * np.sin(ph), np.full_like(ph, ci)], axis=-1)
    eY = np.stack([-ci * np.cos(ph), ci * np.sin(ph), np.full_like(ph, si)], axis=-1)
    eX = np.cross(eY, n)
    return n, eX, eY


def project(points, phase, incl, frame=None):
    """
    Project points (...,3) [corotating frame] to observer-frame (X,Y,depth).
    phase, incl are scalars (one orbital configuration for all points).

    frame: pass a caller's already-computed observer_frame(phase, incl)
    (the full (n_hat, eX, eY) triple) to skip recomputing it -- e.g. a
    per-phase loop that calls project() more than once at the same
    (phase, incl), like visible_fraction_bulk's two calls (one for the
    lobe's own hull, one for the query points) at every phase an eclipse
    is even possible.
    """
    n, eX, eY = frame if frame is not None else observer_frame(phase, incl)
    X = np.dot(points, eX)
    Y = np.dot(points, eY)
    depth = np.dot(points, n)
    return X, Y, depth


def _gap_along_los(points, n, lobe, t_max, n_t, refine=True, max_refine=64):
    """
    min_t [ |P + t*n - center2| - r_lobe(direction) ], t in [0,t_max],
    for an array of points (...,3). Returns array of shape points.shape[:-1].

    The coarse scan is fully vectorized (cheap even for large point sets,
    e.g. a disc surface grid). If `refine` is set, points are further
    refined with a local per-point bounded minimization for accurate
    zero-crossings in the phase domain -- but only up to `max_refine`
    points (chosen nearest the coarse minimum's sign boundary), since this
    step is not vectorizable and is only needed for precise single-point
    contact-time work, not bulk visibility masks over a whole surface grid.
    """
    pts = np.asarray(points, dtype=float)
    lead_shape = pts.shape[:-1]
    flat = pts.reshape(-1, 3)
    ts = np.linspace(0.0, t_max, n_t)

    ray = flat[:, None, :] + ts[None, :, None] * n[None, None, :]
    delta = ray - lobe.center
    r = np.linalg.norm(delta, axis=-1)
    theta = np.arccos(np.clip(delta[..., 2] / np.maximum(r, 1e-300), -1.0, 1.0))
    phidir = np.arctan2(delta[..., 1], delta[..., 0])
    rl = lobe.radius(theta, phidir)
    g = r - rl  # shape (Npts, n_t)

    imin = np.argmin(g, axis=1)
    gmin_coarse = g[np.arange(g.shape[0]), imin]

    if not refine or flat.shape[0] > max_refine:
        return gmin_coarse.reshape(lead_shape)

    # local refinement in a bracket around the coarse minimum
    dt = ts[1] - ts[0]
    refined = np.empty(flat.shape[0])
    for k in range(flat.shape[0]):
        i0 = imin[k]
        lo = ts[max(i0 - 1, 0)]
        hi = ts[min(i0 + 1, n_t - 1)]
        if hi <= lo:
            refined[k] = gmin_coarse[k]
            continue

        def gfun(t, p=flat[k]):
            d = p + t * n - lobe.center
            rr = np.linalg.norm(d)
            th = np.arccos(np.clip(d[2] / max(rr, 1e-300), -1.0, 1.0))
            pd = np.arctan2(d[1], d[0])
            return rr - float(lobe.radius(th, pd))

        res = minimize_scalar(gfun, bounds=(lo, hi), method="bounded",
                               options={"xatol": 1e-10})
        refined[k] = res.fun

    return refined.reshape(lead_shape)


def eclipse_signal(points, phase, incl, lobe, t_max=3.0, n_t=80, refine=True):
    """
    Signed occultation function at given phase(s): negative where occulted
    by the secondary's Roche lobe, positive where visible.  `points` may be
    a single point (3,) or an array (...,3); `phase` a scalar.

    `refine` (default True) adds a precise per-point local minimization on
    top of the coarse scan; for a handful of points (e.g. contact-time
    root-finding on a single point) this is cheap and gives high-precision
    zero-crossings. For bulk visibility over large point sets (a disc
    surface grid), refinement is automatically skipped once the point
    count exceeds `_gap_along_los`'s max_refine, and the coarse n_t grid
    alone determines visibility -- prefer `visible_mask` for that use case.
    """
    n, _, _ = observer_frame(phase, incl)
    return _gap_along_los(np.asarray(points, dtype=float), n, lobe, t_max, n_t, refine=refine)


def visible_mask(points, phase, incl, lobe, t_max=3.0, n_t=120):
    """
    Fast boolean visibility for a (potentially large) array of points at a
    single phase: True where not occulted by the secondary's Roche lobe.
    Purely vectorized coarse scan (no per-point refinement) -- intended for
    bulk flux integration (disc/star/stream surface grids), where a few
    misclassified points near the exact eclipse edge are negligible
    against the grid's own spatial resolution.
    """
    n, _, _ = observer_frame(phase, incl)
    g = _gap_along_los(np.asarray(points, dtype=float), n, lobe, t_max, n_t, refine=False)
    return g > 0.0


_L1_DIR = np.array([-1.0, 0.0, 0.0])  # from lobe.center toward L1, in the corotating frame


def visible_mask_bulk(points, phase, incl, lobe):
    """
    Fast visibility test against occultation by the secondary's Roche
    lobe -- valid ONLY for points that stay well clear of the L1 funnel
    (the primary and the disc). NOT valid for the accretion stream: its
    near-L1 points sit exactly in the region this shortcut ignores, and
    need the full ray march (visible_mask) instead.

    Two simplifications, both verified numerically against visible_mask's
    ray march (0 mismatches across hundreds of randomized primary/disc
    points over the full eclipse-capable phase range):

    1. Whenever L1 itself faces the observer (n_hat . L1_dir > 0, L1_dir
       pointing from the lobe's center toward L1), the secondary's bulk
       is swung to the observer's far side and cannot be eclipsing
       anything at the primary's position -- return all-visible
       immediately, skipping all per-point work (true for ~half the
       orbit).
    2. Otherwise, a point's sky-plane (X,Y) landing within the lobe's
       projected convex hull (see plots.lobe_outline) is by itself
       sufficient -- no depth check against the lobe's near surface is
       needed. The hull's one real inaccuracy is a ~13x R1 bulge past the
       lobe's true, concave near-L1 waist, but that bulge sits right at
       L1's own projected position, which primary/disc points -- unlike
       the stream -- never reach.
    """
    n_hat, _, _ = observer_frame(phase, incl)
    pts = np.asarray(points, dtype=float)
    lead_shape = pts.shape[:-1]

    if np.dot(n_hat, _L1_DIR) > 0.0:
        return np.ones(lead_shape, dtype=bool)

    surf = lobe.surface_points()
    Xs, Ys, _ = project(surf, phase, incl)
    hull = ConvexHull(np.column_stack([Xs, Ys]))
    poly = Path(np.column_stack([Xs, Ys])[np.append(hull.vertices, hull.vertices[0])])
    X, Y, _ = project(pts.reshape(-1, 3), phase, incl)
    inside = poly.contains_points(np.column_stack([X, Y])).reshape(lead_shape)
    return ~inside


def visible_fraction_bulk(points, phase, incl, lobe, cell_size, frame=None):
    """
    Fast, smooth (0..1 ramp) visibility fraction against occultation by
    the secondary's Roche lobe -- a drop-in replacement for
    visible_fraction wherever visible_mask_bulk's own validity condition
    holds: points that stay well clear of the L1 funnel, i.e. the
    primary and the disc, NOT the accretion stream (whose points sit
    exactly in the region this is unsafe for -- use visible_fraction
    there instead).

    frame: pass a caller's already-computed observer_frame(phase, incl)
    to skip recomputing it here and in each of this function's own two
    project() calls -- see sphere_visible_mask's docstring for why this
    matters once per-phase overhead like this stops being negligible
    next to the (now much cheaper) hull test.

    Skips the 3D ray-march (_gap_along_los, dominated by
    roche.radius's bilinear interpolation) entirely rather than merely
    approximating it, for two reasons that make this exact, not just a
    fast heuristic, everywhere it's actually used:

    1. Whenever L1 faces the observer, the secondary's bulk is swung to
       the far side and cannot be occulting the primary/disc at all --
       an exact geometric fact (see visible_mask_bulk), so this returns
       all-visible (1.0) immediately, no further work.
    2. Whenever it does not face the observer -- the only regime where
       an eclipse is even geometrically possible -- the lobe's near-L1
       concave waist (the one feature a convex hull can't represent) is
       necessarily on the observer's far side too, i.e. not part of the
       projected silhouette being tested against at all. So exactly
       where occultation can happen, the true silhouette has no
       concavity, and the convex hull is exact for primary/disc points,
       not merely a well-tested approximation.

    Distance to the hull boundary, needed for the smooth ramp, is the
    signed distance to a convex polygon: the max over each edge's signed
    perpendicular distance (ConvexHull.equations gives [normal, offset]
    per facet in Qhull's A.x+b<=0-inside convention, so normal.point+
    offset is exactly that edge's signed distance, positive outside).
    This is exact when the nearest boundary feature is an edge, and a
    minor, immaterial overestimate of the true distance near a hull
    vertex -- immaterial because it only ever feeds the same
    clip(0.5 + g/cell_size, 0, 1) ramp visible_fraction itself uses: points
    more than about half a cell away from the boundary land at a hard 0
    or 1 regardless, and only points within about their own size of the
    true limb see any smooth transition at all.

    The hull is built from a decimated subsample of the lobe's surface
    grid (~_HULL_TARGET_N points, picked directly from the stored
    (theta,phi) grid before building any Cartesian coordinates -- not
    lobe.surface_points()'s full mesh then thrown away after) rather than
    the full ntheta*nphi grid: the full grid was measured to cost ~50-130ms
    per ConvexHull call (dominant cost of this whole function, an order of
    magnitude worse than the ray-march it's meant to replace), while a few
    thousand well-spread points already reproduce the same hull to well
    within one grid cell (0 classification mismatches, sub-cell fraction
    error, against a full ray-march ground truth over hundreds of random
    primary/disc points and phases) at a small fraction of the cost --
    the hull only needs enough points to bound a smooth, already-convex-
    in-this-regime silhouette, not the full surface resolution.
    """
    if frame is None:
        frame = observer_frame(phase, incl)
    n_hat = frame[0]
    pts = np.asarray(points, dtype=float)
    lead_shape = pts.shape[:-1]

    if np.dot(n_hat, _L1_DIR) > 0.0:
        return np.ones(lead_shape)

    surf = _coarse_lobe_surface(lobe)
    Xs, Ys, _ = project(surf, phase, incl, frame=frame)
    hull = ConvexHull(np.column_stack([Xs, Ys]))
    normals = hull.equations[:, :2]
    offsets = hull.equations[:, 2]

    X, Y, _ = project(pts.reshape(-1, 3), phase, incl, frame=frame)
    xy = np.column_stack([X, Y])
    g = (xy @ normals.T + offsets).max(axis=1)

    half = max(cell_size, 1e-6) / 2.0
    return np.clip(0.5 + g / (2.0 * half), 0.0, 1.0).reshape(lead_shape)


_HULL_TARGET_N = 2000  # see visible_fraction_bulk's docstring -- deliberately modest:
                        # ConvexHull's cost was measured (repeatedly, with warmup) to jump
                        # sharply (~25-100x) between ~2500 and ~3000 points on this data
                        # (likely a Qhull internal algorithm/precision threshold), so this
                        # stays safely clear of that cliff rather than chasing marginal
                        # accuracy gains right up against it.


def _coarse_lobe_surface(lobe):
    """
    A ~_HULL_TARGET_N-point Cartesian subsample of the lobe's surface,
    picked by decimating the stored (theta,phi) grid *before* building
    any (x,y,z) -- unlike lobe.surface_points(), which builds the full
    ntheta*nphi mesh first and would be no cheaper to subsample after the
    fact. Stride is chosen from the target point count rather than fixed,
    so this stays adequately dense (and doesn't silently under-sample
    down to a handful of points) for a RocheLobe built at any ntheta/nphi,
    not just the 181x361 default.
    """
    ntheta, nphi = lobe._ntheta, lobe._nphi
    stride = max(1, int(np.sqrt(ntheta * nphi / _HULL_TARGET_N)))
    r = lobe.r_grid[::stride, ::stride]
    theta = lobe._theta[::stride]
    phi = lobe._phi[::stride]
    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    dx = np.sin(TH) * np.cos(PH)
    dy = np.sin(TH) * np.sin(PH)
    dz = np.cos(TH)
    X = lobe.x2 + r * dx
    Y = r * dy
    Z = r * dz
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)


def sphere_visible_mask(points, phase, incl, center, radius, n_hat=None):
    """
    Exact (closed-form, no marching) boolean visibility for points against
    occultation by a sphere (`center`, `radius`) -- e.g. the primary star.
    True where NOT occulted. The occluding silhouette is trivially a
    circle (a sphere's projection, from any direction), so unlike the
    secondary's Roche lobe there's no hull/ray-march machinery here at
    all to begin with -- this has always been O(1) per point, just dot
    products and a comparison.

    A point P's sky ray toward the observer has constant sky-plane (X,Y)
    (by construction of the projection), so occultation reduces to: is P's
    own sky position within the sphere's circular silhouette, and does P
    sit behind the sphere's near (away-from-observer) surface at that
    (X,Y). With d = P - center and b = d . n_hat (P's depth relative to
    the sphere's center), the sky-plane offset squared is |d|^2 - b^2, and
    the sphere's near-surface depth offset is -sqrt(radius^2 - offset^2)
    (real iff the point is within the silhouette) -- occluded iff P's own
    depth offset b is less than that.

    n_hat: pass the caller's own already-computed observer_frame(phase,
    incl)[0] to skip recomputing it -- cheap individually, but callers
    like render._phase_chunk_flux invoke this (and visible_fraction_bulk)
    several times per phase, and observer_frame's sin/cos/cross ends up a
    surprisingly large fraction of the loop once the Roche-lobe occlusion
    test is no longer the dominant cost (see visible_fraction_bulk).
    Whether the primary is itself in eclipse at this phase (tested
    completely separately, via visible_fraction_bulk/disc.visible against
    the primary's own points) has no bearing on this function at all --
    it only ever answers "is `points` behind the primary", independent of
    what else may or may not be visible.
    """
    n = n_hat if n_hat is not None else observer_frame(phase, incl)[0]
    d = np.asarray(points, dtype=float) - np.asarray(center, dtype=float)
    b = np.einsum("...i,i->...", d, n)
    perp2 = np.maximum(np.einsum("...i,...i->...", d, d) - b ** 2, 0.0)
    inside_silhouette = perp2 < radius ** 2
    near_side = -np.sqrt(np.maximum(radius ** 2 - perp2, 0.0))
    occluded = inside_silhouette & (b < near_side)
    return ~occluded


def visible_fraction(points, phase, incl, lobe, cell_size, t_max=3.0, n_t=120):
    """
    Smoothly-varying (0..1) visibility fraction for a (potentially large)
    array of points, using a linear ramp of width `cell_size` around the
    eclipse edge (g=0) instead of a hard g>0 threshold.

    Each point here stands in for a finite-area grid cell (disc/star/
    stream surface element) of roughly that size, whose true unocculted
    fraction genuinely transitions smoothly across the eclipse limb -- so
    this is not just a numerical convenience but the more physically
    correct treatment. It also keeps flux(params) continuous and
    differentiable enough for reliable finite-difference gradients during
    least-squares fitting, where a hard per-cell boolean would otherwise
    make the model flux change in discrete "staircase" jumps as
    continuous parameters (e.g. a_disc) are varied infinitesimally.

    `cell_size` should be of order the point spacing (e.g. dr for a disc
    grid, or the ray step used to build the point set); it need not be
    exact, just the right order of magnitude for the grid resolution used.
    """
    n, _, _ = observer_frame(phase, incl)
    g = _gap_along_los(np.asarray(points, dtype=float), n, lobe, t_max, n_t, refine=False)
    half = max(cell_size, 1e-6) / 2.0
    return np.clip(0.5 + g / (2.0 * half), 0.0, 1.0)


def is_occulted(points, phase, incl, lobe, **kw):
    return eclipse_signal(points, phase, incl, lobe, **kw) < 0.0


def contact_phases(point, incl, lobe, phase_center=0.0, half_window=0.25,
                    n_scan=180, t_max=3.0, n_t=60):
    """
    Ingress/egress orbital phase(s) at which a single fixed point
    (corotating frame) is occulted by the secondary's Roche lobe, searched
    within [phase_center-half_window, phase_center+half_window].

    Returns a sorted array of phases where eclipse_signal crosses zero
    (empty if the point is never occulted in that window). For a simple
    single eclipse this has length 2: [ingress, egress].
    """
    phases = np.linspace(phase_center - half_window, phase_center + half_window, n_scan)
    point = np.asarray(point, dtype=float)
    sig = np.array([eclipse_signal(point, p, incl, lobe, t_max=t_max, n_t=n_t)
                     for p in phases])

    roots = []
    for i in range(len(phases) - 1):
        if sig[i] == 0.0:
            roots.append(phases[i])
        elif sig[i] * sig[i + 1] < 0.0:
            def f(p):
                return eclipse_signal(point, p, incl, lobe, t_max=t_max, n_t=n_t)
            roots.append(brentq(f, phases[i], phases[i + 1], xtol=1e-10))
    return np.array(roots)
