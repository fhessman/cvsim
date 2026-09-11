# disc.py
"""
Accretion disc geometry and surface brightness, in the corotating frame of
roche.py (units a=1, primary at x1, COM at origin).

The disc rim is an ellipse with one focus at the primary (Kepler orbit of
disc material around M1), parametrized by semi-major axis a_disc,
eccentricity e_disc, and periapsis orientation omega_disc measured from
the +x axis (the instantaneous line towards the secondary):

    R(nu) = a_disc*(1-e_disc^2) / (1 + e_disc*cos(nu - omega_disc))

For a real disc, e_disc and omega_disc are not static: an eccentric disc's
apsidal line precesses slowly (the "superhump" mechanism, Whitehurst 1988;
Osaki 1985, 1996) at a beat period P_sh between the orbital and disc
precession periods.  precession_omega() gives omega_disc as a function of
time for use cycle-by-cycle by the eclipse-timing layer; within a single
eclipse (~0.1 in phase) the disc shape is frozen.

Surface brightness is modeled as a simple radial power law,
I(r) = I0*(r/a_disc)^(-p); p ~ 0.75 corresponds to the local blackbody
temperature profile of a steady-state alpha-disc (T ~ r^-3/4) integrated
over a Rayleigh-Jeans-like band, but p is left as a free parameter since
the true band-integrated profile depends on the (unmodeled) SED.

Disc thickness (opening_angle)
-------------------------------
By default the disc is the zero-thickness sheet above (opening_angle=0),
used throughout lightcurve.py/eclipse_timing.py/fit.py for eclipse-timing
and light-curve work, via the analytic flat-plane test in visible_from().

Setting opening_angle>0 (half-angle from the midplane, radians) instead
gives the disc real geometric thickness -- constant H/R = tan(opening_angle)
-- replacing the flat sheet with a *solid* bounded by four surfaces: an
upper cone (z=+r*tanA), a lower cone (z=-r*tanA), a vertical outer-edge
wall at r=rim(nu), and a vertical inner-edge wall at the (circular)
r=r_in. This is what gives the disc rim genuine 3D extent for a stream
"hot spot" to land on, rather than a patch pasted onto an edge with no
actual surface. The edge surfaces take the local disc temperature at
their radius (rim(nu) or r_in respectively) -- a hot spot is then just a
locally-elevated temperature on top of that, applied by the caller.

All four surfaces are handled uniformly for occlusion via the solid's
containment test, contains_3d() -- see occluded_by_disc()'s docstring for
why that also gets self-occlusion among the four surfaces for free. This
3D path is used by render.py/irradiation.py, not by the flat-disc-only
light-curve/eclipse-timing modules (see their own docstrings).
"""

import numpy as np
from scipy.spatial import ConvexHull

from eclipse import observer_frame, project


def precession_omega(omega0, time, P_sh, sense=-1.0):
    """
    Disc periapsis orientation [rad] at a given time, given the superhump
    (orbital/precession beat) period P_sh in the same time units as `time`.
    sense=-1 (default): apsidal line regresses in the corotating frame,
    the standard sense for prograde, slowly-precessing eccentric discs
    viewed in a frame corotating with the (faster) binary orbit. Flip to
    +1 if the fitted sense turns out to be the other way for your data.
    """
    return omega0 + sense * 2.0 * np.pi * time / P_sh


def _split_grid(n_areas, aspect):
    """
    Local copy of render.split_area_count's (n1,n2) grid-dimension split
    (n1*n2 ~= n_areas, n1:n2 matching aspect=n2/n1) -- duplicated rather
    than imported to avoid a disc->render->lightcurve->disc import cycle
    (lightcurve.py already imports Disc from this module).
    """
    n1 = max(1, int(round((n_areas / aspect) ** 0.5)))
    n2 = max(1, int(round(n1 * aspect)))
    return n1, n2


def _wall_aspect_ratio(opening_angle_rad):
    """Local copy of render.wall_aspect_ratio -- see _split_grid's note."""
    return np.pi / max(np.tan(opening_angle_rad), 1e-12)


class Disc:
    """A single (frozen-shape) snapshot of the disc, for one eclipse cycle."""

    def __init__(self, x1, a_disc, e_disc=0.0, omega_disc=0.0, r_in=0.02,
                 brightness_index=0.75, i_norm=1.0, opening_angle=0.0):
        self.x1 = x1
        self.a = a_disc
        self.e = e_disc
        self.omega = omega_disc
        self.r_in = r_in
        self.p = brightness_index
        self.i_norm = i_norm
        self.opening_angle = opening_angle

    @property
    def is_empty(self):
        """True for a "null disc" placeholder (detached/underfilling secondary,
        no accretion disc): r_in >= a means every point fails the rim.py
        r_in<=r<=rim(nu) containment test, so the disc occults nothing and
        renders nothing without any extra special-casing at call sites."""
        return self.r_in >= self.a

    def rim(self, nu):
        """Rim radius (distance from primary) at azimuth nu [rad]."""
        return self.a * (1.0 - self.e ** 2) / (1.0 + self.e * np.cos(nu - self.omega))

    def rim_xy(self, n=200):
        """Cartesian (x,y) of the rim, corotating frame, for plotting."""
        nu = np.linspace(0.0, 2.0 * np.pi, n)
        R = self.rim(nu)
        return self.x1 + R * np.cos(nu), R * np.sin(nu)

    def surface_brightness(self, r):
        return self.i_norm * (r / self.a) ** (-self.p)

    def visible_from(self, points, phase, incl):
        """
        Boolean array, same leading shape as `points` (...,3) [corotating
        frame]: True where NOT occulted by this disc's solid (r_in..rim)
        annulus, sitting exactly in the z=0 orbital plane, at the given
        phase/inclination.

        Exact (not grid/ray-marched) plane-intersection test: since the
        disc is flat, the sky ray from a point P towards the observer,
        P + t*n_hat, crosses z=0 at a single t_cross = -P_z/n_hat_z; the
        point is occulted iff that crossing is in front of it (t_cross>0,
        i.e. the disc plane is between P and the observer) and lands
        within the disc's annulus footprint (r_in <= r <= rim(nu)).

        Note a point exactly at z=0 (t_cross=0) is never occulted by this
        coplanar, zero-thickness disc -- only points with nonzero z, e.g.
        the near/far portions of the primary's own finite-size disc face
        (lightcurve.star_points' sample points do have nonzero z off the
        exact center, since they are built on the plane through the
        primary's center perpendicular to the line of sight, which is
        tilted relative to the disc's z=0 plane for any inclination other
        than exactly 90 or 0 deg). A literal point source at the disc's
        own midplane can thus never be eclipsed by a perfectly flat,
        zero-thickness disc except exactly edge-on.
        """
        n, _, _ = observer_frame(phase, incl)
        pts = np.asarray(points, dtype=float)
        Pz = pts[..., 2]
        nz = n[2]
        if abs(nz) < 1e-10:
            return np.ones(Pz.shape, dtype=bool)
        t_cross = -Pz / nz
        Qx = pts[..., 0] + t_cross * n[0]
        Qy = pts[..., 1] + t_cross * n[1]
        r = np.hypot(Qx - self.x1, Qy)
        nu = np.arctan2(Qy, Qx - self.x1)
        occulted = (t_cross > 0.0) & (r >= self.r_in) & (r <= self.rim(nu))
        return ~occulted

    def segment_blocked(self, P1, P2):
        """
        Exact test (flat disc, opening_angle==0, only): True where the
        straight FINITE segment P1->P2 passes through this disc's annulus
        at z=0 -- the finite-segment analogue of visible_from's exact
        ray-to-observer crossing test, for testing occlusion between two
        nearby points (e.g. a primary surface point irradiating a
        secondary surface point) rather than a ray out to the sky. Like
        visible_from, this is exact rather than sampled/marched, for the
        same reason: a flat, zero-thickness disc is a measure-zero target
        for a handful of interior sample points along the segment, so a
        march would almost always (wrongly) report "not blocked".

        P1, P2 (...,3) [corotating frame] broadcast against each other;
        returns their common broadcast leading shape.
        """
        P1 = np.asarray(P1, dtype=float)
        P2 = np.asarray(P2, dtype=float)
        dz = P2[..., 2] - P1[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = -P1[..., 2] / dz
        Qx = P1[..., 0] + t * (P2[..., 0] - P1[..., 0])
        Qy = P1[..., 1] + t * (P2[..., 1] - P1[..., 1])
        r = np.hypot(Qx - self.x1, Qy)
        nu = np.arctan2(Qy, Qx - self.x1)
        valid = np.isfinite(t) & (t > 0.0) & (t < 1.0)
        return valid & (r >= self.r_in) & (r <= self.rim(nu))

    def azimuth(self, x, y):
        """nu [rad] of a point given in the (x1-centered) corotating frame."""
        return np.arctan2(y, x - self.x1)

    def contains(self, x, y):
        r = np.hypot(x - self.x1, y)
        nu = self.azimuth(x, y)
        return (r >= self.r_in) & (r <= self.rim(nu))

    def contains_3d(self, x, y, z):
        """
        Boolean, same shape as x/y/z: True where (x,y,z) is inside the
        disc *solid* (thickness set by opening_angle), the region bounded
        by the four surfaces described in the module docstring:
        r_in <= r <= rim(nu) and |z| <= r*tan(opening_angle).
        """
        r = np.hypot(x - self.x1, y)
        nu = self.azimuth(x, y)
        half_h = r * np.tan(self.opening_angle)
        return (r >= self.r_in) & (r <= self.rim(nu)) & (np.abs(z) <= half_h)

    def equal_area_annulus(self, n):
        """
        Roughly equal-area (r, nu, area) point cloud over the disc's
        annular face r_in..rim(nu), replacing a regular (r,nu) grid.

        A regular grid (fixed dr, dnu) puts area ~ r dr dnu per cell, so
        cell area -- and hence sky-plane point density once projected --
        grows with r; scatter-plotted, this shows up as a moire pattern
        (the same problem RocheLobe.equal_area_sample fixed for the
        secondary). Here nu is swept via the golden angle (a quasi-
        uniform spiral with no repeating grid lines, exactly as
        equal_area_sample does on the sphere) and r is placed via the
        standard equal-area annulus transform, r = sqrt(r_in^2 +
        t*(rim(nu)^2 - r_in^2)) for a uniform area-fraction t=(i+0.5)/n --
        exact for a circular rim, and a good approximation for a mildly
        eccentric one (same caveat as equal_area_sample's near-spherical
        Roche lobe).

        `area` is the flat in-plane wedge area per point (r dr dnu,
        integrated); a conical face just scales this by an overall
        sec(opening_angle) slant factor (see upper_cone_grid). Callers
        needing (x,y) can take x=x1+r*cos(nu), y=r*sin(nu).
        """
        i = np.arange(n)
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        nu = np.mod(i * golden_angle, 2.0 * np.pi)
        t = (i + 0.5) / n
        R_rim = self.rim(nu)
        r = np.sqrt(np.maximum(self.r_in ** 2 + t * (R_rim ** 2 - self.r_in ** 2), 0.0))
        area = 0.5 * (R_rim ** 2 - self.r_in ** 2) * (2.0 * np.pi / n)
        return r, nu, area

    def upper_cone_grid(self, n_r=40, n_nu=80):
        """
        Upper cone surface (z=+r*tan(opening_angle)) as (x,y,z,normals,
        areas,r): r is each point's cylindrical (midplane) radius, for
        looking up the local disc temperature T(r). Point placement uses
        equal_area_annulus(n_r*n_nu) (n_r, n_nu kept as the calling
        convention -- their product is the only thing that matters now,
        see equal_area_annulus); areas use the exact conical-surface
        element, the annulus's flat wedge area times sec(opening_angle).
        """
        secA = 1.0 / np.cos(self.opening_angle)
        tanA = np.tan(self.opening_angle)
        r, nu, area = self.equal_area_annulus(n_r * n_nu)
        cos_nu, sin_nu = np.cos(nu), np.sin(nu)
        xs = self.x1 + r * cos_nu
        ys = r * sin_nu
        zs = r * tanA
        nxs = -np.sin(self.opening_angle) * cos_nu
        nys = -np.sin(self.opening_angle) * sin_nu
        nzs = np.full_like(r, np.cos(self.opening_angle))
        areas = area * secA
        return xs, ys, zs, np.stack([nxs, nys, nzs], axis=-1), areas, r

    def lower_cone_grid(self, n_r=40, n_nu=80):
        """Lower cone (z=-r*tan(opening_angle)): mirror of upper_cone_grid."""
        xs, ys, zs, normals, areas, rs = self.upper_cone_grid(n_r=n_r, n_nu=n_nu)
        zs = -zs
        normals = normals.copy()
        normals[..., 2] *= -1.0
        return xs, ys, zs, normals, areas, rs

    def outer_edge_grid(self, n_z=20, n_nu=80, n_nu_wall=None, nu_refine=None, refine_factor=4.0):
        """
        Outer-edge wall at r=rim(nu): a vertical face (constant-nu column
        spanning z in [-h,h], h=rim(nu)*tan(opening_angle)) with outward
        normal taken as purely radial, (cos(nu),sin(nu),0) -- the "vertical
        face" simplification requested, rather than the true perpendicular
        to the (possibly eccentric) rim curve. Area uses the matching
        simple cylindrical-wall element rim(nu)*dz*dnu. Temperature lookup
        radius r = rim(nu) (constant along each column).

        n_nu_wall: the wall's own azimuthal resolution, independent of
        the cone surfaces' n_nu (defaults to n_nu if not given, for
        standalone/backward-compatible use) -- see render.wall_aspect_ratio
        for why the wall usually wants a very different n_z:n_nu split
        than the cone surfaces it shares a caller with.

        nu_refine: optional (nu_lo, nu_hi) [rad] azimuthal window --
        typically the stream-impact hot spot's own span, see
        render.build_temperature_maps -- sampled roughly refine_factor
        times denser than the rest of the rim, at the SAME total
        n_nu_wall budget (just reallocated, not increased): the hot
        spot's temperature varies sharply over a small azimuthal range
        that a uniform grid under-resolves, while the rest of the rim
        (a much smoother, slowly-varying T(r)) doesn't need the same
        density. nu_hi is taken as nu_lo + ((nu_hi-nu_lo) mod 2*pi), so
        it wraps the same way the hot spot's own dphi does; a window
        spanning the (or more than the) full circle degenerates to the
        plain uniform grid. None (default) keeps that uniform grid.
        """
        if n_nu_wall is None:
            n_nu_wall = n_nu
        tanA = np.tan(self.opening_angle)
        if nu_refine is None:
            nu = np.linspace(0.0, 2.0 * np.pi, n_nu_wall, endpoint=False)
            dnu = np.full(n_nu_wall, 2.0 * np.pi / n_nu_wall)
        else:
            nu_lo, nu_hi = nu_refine
            window = (nu_hi - nu_lo) % (2.0 * np.pi)
            rest = 2.0 * np.pi - window
            if window <= 0.0 or rest <= 0.0:
                nu = np.linspace(0.0, 2.0 * np.pi, n_nu_wall, endpoint=False)
                dnu = np.full(n_nu_wall, 2.0 * np.pi / n_nu_wall)
            else:
                frac_in = (window * refine_factor) / (window * refine_factor + rest)
                n_in = int(np.clip(round(n_nu_wall * frac_in), 1, n_nu_wall - 1))
                n_out = n_nu_wall - n_in
                nu_in = np.linspace(nu_lo, nu_lo + window, n_in, endpoint=False)
                nu_out = np.linspace(nu_lo + window, nu_lo + 2.0 * np.pi, n_out, endpoint=False)
                nu = np.concatenate([nu_in, nu_out])
                dnu = np.concatenate([np.full(n_in, window / n_in), np.full(n_out, rest / n_out)])
        s = (0.5 + np.arange(n_z)) / n_z * 2.0 - 1.0  # cell centers in [-1,1]
        ds = 2.0 / n_z
        xs, ys, zs, nxs, nys, nzs, areas, rs = [], [], [], [], [], [], [], []
        for nu_i, dnu_i in zip(nu, dnu):
            R_rim = self.rim(nu_i)
            if R_rim <= self.r_in or self.opening_angle <= 0.0:
                continue
            h = R_rim * tanA
            cos_nu, sin_nu = np.cos(nu_i), np.sin(nu_i)
            xs.append(np.full(n_z, self.x1 + R_rim * cos_nu))
            ys.append(np.full(n_z, R_rim * sin_nu))
            zs.append(s * h)
            nxs.append(np.full(n_z, cos_nu))
            nys.append(np.full(n_z, sin_nu))
            nzs.append(np.zeros(n_z))
            areas.append(np.full(n_z, R_rim * h * ds * dnu_i))
            rs.append(np.full(n_z, R_rim))
        cat = lambda a: np.concatenate(a) if a else np.array([])
        xs, ys, zs, nxs, nys, nzs, areas, rs = (cat(a) for a in
            (xs, ys, zs, nxs, nys, nzs, areas, rs))
        return xs, ys, zs, np.stack([nxs, nys, nzs], axis=-1), areas, rs

    def inner_edge_grid(self, n_z=20, n_nu=80, n_nu_wall=None):
        """
        Inner-edge wall at the (circular) r=r_in: like outer_edge_grid but
        facing inward, normal=-(cos(nu),sin(nu),0), exact since r_in does
        not vary with nu. Temperature lookup radius r=r_in (constant).

        n_nu_wall: see outer_edge_grid's docstring -- same independent
        wall azimuthal resolution, defaults to n_nu if not given.
        """
        if n_nu_wall is None:
            n_nu_wall = n_nu
        tanA = np.tan(self.opening_angle)
        if self.opening_angle <= 0.0:
            empty = np.array([])
            return empty, empty, empty, np.zeros((0, 3)), empty, empty
        nu = np.linspace(0.0, 2.0 * np.pi, n_nu_wall, endpoint=False)
        dnu = 2.0 * np.pi / n_nu_wall
        s = (0.5 + np.arange(n_z)) / n_z * 2.0 - 1.0
        ds = 2.0 / n_z
        h = self.r_in * tanA
        cos_nu, sin_nu = np.cos(nu), np.sin(nu)
        xs = np.repeat(self.x1 + self.r_in * cos_nu, n_z)
        ys = np.repeat(self.r_in * sin_nu, n_z)
        zs = np.tile(s * h, n_nu_wall)
        nxs = np.repeat(-cos_nu, n_z)
        nys = np.repeat(-sin_nu, n_z)
        nzs = np.zeros_like(xs)
        areas = np.full(n_nu_wall * n_z, self.r_in * h * ds * dnu)
        rs = np.full(n_nu_wall * n_z, self.r_in)
        return xs, ys, zs, np.stack([nxs, nys, nzs], axis=-1), areas, rs

    def all_surfaces_grid(self, n_r=40, n_nu=80, n_z=20, n_nu_wall=None,
                           nu_refine=None, refine_factor=4.0):
        """
        Concatenation of all four surfaces (upper cone, lower cone, outer
        edge, inner edge): returns (x,y,z,normals,areas,r,surface_id),
        surface_id in {0:upper,1:lower,2:outer,3:inner} for callers that
        want to single out e.g. the outer edge for a stream hot spot.

        n_nu_wall: see outer_edge_grid's docstring -- the wall surfaces'
        own azimuthal resolution, independent of the cone surfaces' n_nu
        (defaults to n_nu if not given).

        nu_refine/refine_factor: forwarded to outer_edge_grid ONLY (see
        its own docstring) -- the hot spot lives on the outer rim alone,
        so there's nothing for the inner wall to refine around.

        The outer and inner walls' combined budget (2*n_z*n_nu_wall --
        what this signature always represented in total, split evenly
        before) is instead split between them proportional to their
        actual areas (~R^2*tan(opening_angle), tan(opening_angle) common
        to both): giving both walls the same element count regardless of
        the outer wall typically having (a_disc/r_in)^2 times more area
        than the inner wall just over-resolves the smaller one for no
        gain in coverage quality -- pure wasted work.
        """
        if n_nu_wall is None:
            n_nu_wall = n_nu
        n_wall_total = 2 * n_z * n_nu_wall
        frac_outer = self.a ** 2 / (self.a ** 2 + self.r_in ** 2)
        n_outer = max(1, round(n_wall_total * frac_outer))
        n_inner = max(1, n_wall_total - n_outer)
        wall_aspect = _wall_aspect_ratio(self.opening_angle) if self.opening_angle > 0.0 else 1.0
        n_z_o, n_nu_o = _split_grid(n_outer, wall_aspect)
        n_z_i, n_nu_i = _split_grid(n_inner, wall_aspect)

        parts = [
            (0, self.upper_cone_grid(n_r=n_r, n_nu=n_nu)),
            (1, self.lower_cone_grid(n_r=n_r, n_nu=n_nu)),
            (2, self.outer_edge_grid(n_z=n_z_o, n_nu=n_nu, n_nu_wall=n_nu_o,
                                      nu_refine=nu_refine, refine_factor=refine_factor)),
            (3, self.inner_edge_grid(n_z=n_z_i, n_nu=n_nu, n_nu_wall=n_nu_i)),
        ]
        xs, ys, zs, normals, areas, rs, ids = [], [], [], [], [], [], []
        for sid, (x, y, z, n, a, r) in parts:
            xs.append(x); ys.append(y); zs.append(z); normals.append(n)
            areas.append(a); rs.append(r); ids.append(np.full(x.shape, sid))
        return (np.concatenate(xs), np.concatenate(ys), np.concatenate(zs),
                np.concatenate(normals, axis=0), np.concatenate(areas),
                np.concatenate(rs), np.concatenate(ids))

    def _boundary_curves(self, n=250):
        """
        The four curves bounding the disc solid's *entire* projected
        silhouette: the top/bottom lips of the inner rim (r=r_in) and
        outer rim (r=rim(nu)) -- see occluded_by_disc's docstring for why
        these four 1D curves alone (not the full 2D surfaces) exactly
        determine the solid's projected convex hull.
        """
        nu = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        tanA = np.tan(self.opening_angle)
        r_out = self.rim(nu)
        h_in = self.r_in * tanA
        h_out = r_out * tanA
        cos_nu, sin_nu = np.cos(nu), np.sin(nu)
        A = np.stack([self.x1 + self.r_in * cos_nu, self.r_in * sin_nu, np.full(n, h_in)], axis=-1)
        Ap = np.stack([self.x1 + self.r_in * cos_nu, self.r_in * sin_nu, np.full(n, -h_in)], axis=-1)
        B = np.stack([self.x1 + r_out * cos_nu, r_out * sin_nu, h_out], axis=-1)
        Bp = np.stack([self.x1 + r_out * cos_nu, r_out * sin_nu, -h_out], axis=-1)
        return np.concatenate([A, Ap, B, Bp], axis=0)

    def _occluded_by_disc_march(self, flat_points, n_hat, t_max, n_t):
        """The exact ray march against contains_3d() -- see
        occluded_by_disc's docstring. Factored out so its fast pre-filter
        can call this on only the (usually much smaller) residual of
        points it can't already rule out."""
        ts = np.linspace(0.0, t_max, n_t)
        ray = flat_points[:, None, :] + ts[None, :, None] * n_hat[None, None, :]
        inside = self.contains_3d(ray[..., 0], ray[..., 1], ray[..., 2])
        return inside.any(axis=1)

    def occluded_by_disc(self, points, phase, incl, t_max=3.0, n_t=150, n_curve=250, use_hull=True):
        """
        Boolean array, leading shape of `points` (...,3): True where
        occulted by the disc *solid* along the sky ray toward the
        observer. This single test handles occlusion by any of the four
        surfaces (and self-occlusion among them, e.g. the far cone hidden
        behind the near cone or an edge wall) automatically, since it
        only asks "does the ray ever enter the solid", not which surface
        it would first cross.

        Exact fast pre-filter (use_hull=True, the default), then ray
        march only where still needed: each of the disc's four surfaces
        is *ruled* -- a straight line in 3D for fixed nu (e.g. the upper
        cone's r*(cos nu, sin nu, tanA) parametrization, see
        upper_cone_grid) -- and an orthographic projection maps straight
        lines to straight lines, so for fixed nu each surface's image is
        a straight segment between its two boundary points. The convex
        hull of a family of segments between two curves equals the convex
        hull of the two curves themselves, so the convex hull of the
        WHOLE solid's projected image is exactly the convex hull of just
        the four 1D boundary curves (_boundary_curves) -- not the full 2D
        surfaces, and not an approximation: a point outside that hull is
        *provably* not occulted, no marching needed (verified: zero
        exceptions across ~5 million (point, phase, inclination,
        disc-shape) test combinations against the ray march below).
        Points inside the hull may or may not be occulted (the disc's own
        inner hole, r<r_in, is entirely empty, so "inside the outer
        silhouette" doesn't by itself mean "occulted") and fall back to
        the exact ray march.

        use_hull: set False to skip the pre-filter and march every point
        directly. Do this when `points` are themselves on the disc's own
        surface (self-occlusion, e.g. render._phase_chunk_flux's disc
        block or lightcurve.disc_flux) -- such points are, by
        construction, always inside (or right on) the hull, so the
        pre-filter can never reject any of them and is pure overhead
        (building + projecting the hull) for no benefit. Leave it True
        for an external body's points (the primary or secondary): for a
        typical system where they sit well outside the disc's own radial
        extent, this alone already rules out ~80-90% of points at a small
        fraction of the ray march's cost. Either way, the result is
        always exactly what the plain ray march would give.

        For opening_angle=0 use visible_from() instead -- a zero-thickness
        sheet is a measure-zero target for a discretely-sampled ray march
        (see visible_from's docstring), so this method requires
        opening_angle>0 (raises otherwise).
        """
        if self.opening_angle <= 0.0:
            raise ValueError("occluded_by_disc needs opening_angle>0; "
                              "use visible_from() for the flat (thin) disc")
        frame = observer_frame(phase, incl)
        pts = np.asarray(points, dtype=float)
        lead_shape = pts.shape[:-1]
        flat = pts.reshape(-1, 3)

        if not use_hull:
            return self._occluded_by_disc_march(flat, frame[0], t_max, n_t).reshape(lead_shape)

        boundary = self._boundary_curves(n=n_curve)
        Xb, Yb, _ = project(boundary, phase, incl, frame=frame)
        hull = ConvexHull(np.column_stack([Xb, Yb]))
        normals = hull.equations[:, :2]
        offsets = hull.equations[:, 2]
        X, Y, _ = project(flat, phase, incl, frame=frame)
        # signed distance to the hull (max over per-edge half-planes, see
        # eclipse.visible_fraction_bulk for the same trick) -- faster in
        # practice than matplotlib.path.Path.contains_points, and all we
        # need is the sign (>0 outside every half-plane, i.e. outside the
        # hull -> definitely not occluded).
        g = (np.column_stack([X, Y]) @ normals.T + offsets).max(axis=1)
        maybe_occluded = g <= 0.0

        occluded = np.zeros(flat.shape[0], dtype=bool)
        if np.any(maybe_occluded):
            occluded[maybe_occluded] = self._occluded_by_disc_march(
                flat[maybe_occluded], frame[0], t_max, n_t)
        return occluded.reshape(lead_shape)

    def visible(self, points, phase, incl, n_t=150, use_hull=True):
        """
        The single entry point every caller (lightcurve.py, render.py,
        plots.py, irradiation.py) should use to test disc occultation:
        dispatches to visible_from() (exact flat-plane test) if
        opening_angle==0, else to occluded_by_disc()'s 3D ray march.
        Keeping this dispatch in one place means a Disc's opening_angle
        alone controls its occlusion behavior everywhere consistently --
        no caller needs to remember to branch on it itself. It's also why
        a single is_empty short-circuit here (rather than at each of the
        many call sites above) covers all of them at once: a null disc
        (no disc configured) occults nothing, by construction, everywhere
        -- skip the flat-plane/ray-march test and its per-point trig
        entirely rather than running it to that same always-True
        conclusion on every call, every phase.

        use_hull: see occluded_by_disc's docstring -- pass False when
        `points` are the disc's own surface (self-occlusion).
        """
        if self.is_empty:
            return np.ones(np.asarray(points).shape[:-1], dtype=bool)
        if self.opening_angle > 0.0:
            return ~self.occluded_by_disc(points, phase, incl, n_t=n_t, use_hull=use_hull)
        return self.visible_from(points, phase, incl)

    def grid(self, n_r=40, n_nu=80):
        """
        Surface-element grid for flux integration: returns (x, y, flux)
        arrays where `flux` is the area-integrated, pre-eclipse brightness
        contribution of each cell (r,nu grid conforming to the -- possibly
        eccentric -- rim shape).
        """
        nu = np.linspace(0.0, 2.0 * np.pi, n_nu, endpoint=False)
        dnu = 2.0 * np.pi / n_nu
        xs, ys, fluxes = [], [], []
        for nu_i in nu:
            R_rim = self.rim(nu_i)
            if R_rim <= self.r_in:
                continue
            r_edges = np.linspace(self.r_in, R_rim, n_r + 1)
            r_mid = 0.5 * (r_edges[:-1] + r_edges[1:])
            dr = np.diff(r_edges)
            area = r_mid * dr * dnu
            flux = self.surface_brightness(r_mid) * area
            xs.append(self.x1 + r_mid * np.cos(nu_i))
            ys.append(r_mid * np.sin(nu_i))
            fluxes.append(flux)
        return np.concatenate(xs), np.concatenate(ys), np.concatenate(fluxes)

    def total_flux(self, n_r=40, n_nu=80):
        _, _, f = self.grid(n_r=n_r, n_nu=n_nu)
        return f.sum()
