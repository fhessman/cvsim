# roche.py
"""
Roche geometry for a Roche-lobe-filling secondary in a circular binary.

Convention
----------
Units: separation a = 1, G(M1+M2) = 1, orbital angular velocity Omega = 1
(synchronous rotation).  Coordinates (x,y,z) are in the frame corotating
with the binary, origin at the center of mass:

    M1 (primary, non-mass-losing star / accretor) at x1 = -q/(1+q)
    M2 (secondary, Roche-lobe-filling star)       at x2 = +1/(1+q)

    q = M2/M1,  mu1 = M1/(M1+M2) = 1/(1+q),  mu2 = M2/(M1+M2) = q/(1+q)

Roche potential per unit mass (Kopal 1959; e.g. Hilditch 2001, Sect. 4.3):

    Phi(x,y,z) = -mu1/r1 - mu2/r2 - 0.5*(x^2+y^2)

with r1, r2 the distances to M1, M2.  L1 is the saddle point of Phi on the
x-axis between the stars; the secondary's Roche lobe is the equipotential
Phi = Phi(L1) enclosing M2, found here by radial shooting from the
secondary's center (valid because the lobe is star-convex about that
center for the mass ratios of interest).
"""

import numpy as np
from scipy.optimize import brentq


def mass_fractions(q):
    """Return (mu1, mu2, x1, x2) for mass ratio q = M2/M1."""
    mu1 = 1.0 / (1.0 + q)
    mu2 = q / (1.0 + q)
    x1 = -mu2
    x2 = mu1
    return mu1, mu2, x1, x2


def eggleton_radius(q):
    """
    Eggleton (1983) approximation to a Roche lobe's volume-equivalent
    radius, in units of the orbital separation a, for a star with mass
    ratio q = M_star/M_other -- e.g. q itself (M2/M1) for the secondary's
    own lobe (RocheLobe.eggleton_radius, which wraps this), or 1/q for the
    primary's (see magnetic.field_line_points' r_max_factor, the primary's
    dipole field lines' outer size cap).
    """
    q13 = q ** (1.0 / 3.0)
    q23 = q13 * q13
    return 0.49 * q23 / (0.6 * q23 + np.log(1.0 + q13))


def potential(x, y, z, q):
    """Roche potential Phi(x,y,z) for mass ratio q (see module docstring)."""
    mu1, mu2, x1, x2 = mass_fractions(q)
    r1 = np.sqrt((x - x1) ** 2 + y ** 2 + z ** 2)
    r2 = np.sqrt((x - x2) ** 2 + y ** 2 + z ** 2)
    return -mu1 / r1 - mu2 / r2 - 0.5 * (x ** 2 + y ** 2)


def gravity(x, y, z, q):
    """Roche acceleration vector -grad(Phi), shape (...,3)."""
    mu1, mu2, x1, x2 = mass_fractions(q)
    r1 = np.sqrt((x - x1) ** 2 + y ** 2 + z ** 2)
    r2 = np.sqrt((x - x2) ** 2 + y ** 2 + z ** 2)
    gx = -mu1 * (x - x1) / r1 ** 3 - mu2 * (x - x2) / r2 ** 3 + x
    gy = -mu1 * y / r1 ** 3 - mu2 * y / r2 ** 3 + y
    gz = -mu1 * z / r1 ** 3 - mu2 * z / r2 ** 3
    return np.stack([gx, gy, gz], axis=-1)


def lagrange1(q):
    """x-coordinate of the L1 point (root of dPhi/dx=0 between the stars)."""
    mu1, mu2, x1, x2 = mass_fractions(q)

    def dphidx(x):
        return gravity(x, 0.0, 0.0, q)[0] * -1.0  # dPhi/dx = -gx

    return brentq(dphidx, x1 + 1e-8, x2 - 1e-8, xtol=1e-12, rtol=1e-13)


class RocheLobe:
    """
    Precomputed Roche lobe of the secondary for a fixed mass ratio q.

    The lobe surface radius r(theta, phi), measured from the secondary's
    center (x2,0,0) in the usual physics spherical convention
    (theta = polar angle from +z, phi = azimuth from +x), is tabulated on
    a regular grid via vectorized bisection on Phi(r) = Phi_L1, then wrapped
    in a RegularGridInterpolator for fast repeated queries.  This is the
    only q-dependent, expensive step; everything else (orbital phase,
    inclination, disc/stream geometry) is cheap and can vary freely once a
    RocheLobe is built.
    """

    def __init__(self, q, ntheta=181, nphi=361, tol_iter=60, fill_factor=1.0, R2=None):
        """
        fill_factor (default 1.0, i.e. exactly Roche-lobe-filling): the
        secondary's volume as a fraction of its full Roche lobe's volume,
        1.0 = today's lobe-filling behavior exactly, <1.0 = underfilling.
        Volume, not radius: the lobe is far from spherical (especially
        near the L1 cusp), so scaling r(theta,phi) by a linear factor
        would distort its shape unphysically; instead this finds the
        *equipotential* Phi=Phi_target < Phi_L1 whose enclosed volume
        matches fill_factor*V_full, via a 1D root-find (see below) wrapped
        around the same per-direction bisection used for the full lobe --
        r_l1 remains a valid bisection bracket for ANY Phi_target<=Phi_L1
        (a deeper potential can only enclose a smaller region, since Phi
        decreases monotonically inward from the critical surface in every
        direction), so no new bracket logic is needed there.

        R2: alternative to fill_factor -- the secondary's desired
        volume-equivalent radius (units of a, same convention as R1),
        converted internally to the equivalent fill_factor using this
        same lobe's own r_volume_equiv_full (computed once, as part of
        building the reference full lobe fill_factor<1 needs anyway -- no
        redundant extra lobe build). Takes precedence over `fill_factor`
        if given. R2 >= the full lobe's own r_volume_equiv_full is
        clamped to fill_factor=1 (can't overfill in this model).
        """
        self.q = q
        self.mu1, self.mu2, self.x1, self.x2 = mass_fractions(q)
        self.center = np.array([self.x2, 0.0, 0.0])
        self.x_L1 = lagrange1(q)
        self.phi_L1 = potential(self.x_L1, 0.0, 0.0, q)

        theta = np.linspace(0.0, np.pi, ntheta)
        phi = np.linspace(0.0, 2.0 * np.pi, nphi)  # inclusive endpoint for periodic wrap
        TH, PH = np.meshgrid(theta, phi, indexing="ij")
        dx = np.sin(TH) * np.cos(PH)
        dy = np.sin(TH) * np.sin(PH)
        dz = np.cos(TH)

        # r_l1 (the lobe's radius in the L1 direction, theta=pi/2,phi=pi)
        # is an upper bound on the radius in EVERY direction, for ANY
        # equipotential Phi_target<=Phi_L1 -- the lobe is star-convex
        # about the secondary's center and L1 is the critical surface's
        # single most-distant reach, the cusp the whole surface tapers
        # towards, and a deeper (more underfilling) equipotential sits
        # strictly inside the critical one everywhere -- so it doubles as
        # a bisection bracket that's always valid (f(r_l1)>=0 everywhere,
        # tight at Phi_target=Phi_L1 with equality only exactly at the L1
        # direction itself) and needs no separate growth phase to find.
        # That matters because growing r_hi from a small guess (an
        # earlier approach) can fail entirely near the L1 direction:
        # Phi(r) is tangent to Phi_L1 there (a touch, not a crossing), so
        # f(r) stays negative (or numerically indistinguishable from it)
        # all the way out past the true radius, and growth alone never
        # detects a sign change -- which left a multi-grid-cell-wide patch
        # of nonsense results around the cusp (patched, previously, by
        # locally averaging neighbors that were frequently *also* bad,
        # producing a flat plateau at the fallback value instead of the
        # true surface's smooth taper). Bisecting directly against the
        # known r_l1 bracket has no such failure mode: it's simple
        # sign-based halving against a bracket that's valid everywhere by
        # construction, cusp included (when Phi_target=Phi_L1 -- an
        # underfilling surface never reaches the cusp at all, so this
        # doesn't arise there).
        r_l1 = self.x2 - self.x_L1

        def solve_r_grid(phi_target):
            def f(r):
                x = self.x2 + r * dx
                y = r * dy
                z = r * dz
                return potential(x, y, z, q) - phi_target

            r_lo = np.full(TH.shape, 1e-5)
            r_hi = np.full(TH.shape, r_l1)
            for _ in range(tol_iter):
                r_mid = 0.5 * (r_lo + r_hi)
                neg = f(r_mid) < 0.0
                r_lo = np.where(neg, r_mid, r_lo)
                r_hi = np.where(neg, r_hi, r_mid)
            return 0.5 * (r_lo + r_hi)

        def volume_of(r_grid):
            dV = (r_grid ** 3 / 3.0) * np.sin(TH) * (theta[1] - theta[0]) * (phi[1] - phi[0])
            return np.sum(dV)

        # the full (fill_factor=1) lobe is needed either way: directly,
        # if not underfilling, or as the volume/radius reference an
        # underfilling target (fill_factor<1 or R2) is measured against.
        r_grid_full = solve_r_grid(self.phi_L1)
        V_full = volume_of(r_grid_full)
        r_volume_equiv_full = (3.0 * V_full / (4.0 * np.pi)) ** (1.0 / 3.0)

        if R2 is not None:
            fill_factor = min((R2 / r_volume_equiv_full) ** 3, 1.0)

        if fill_factor >= 1.0:
            r_grid = r_grid_full
            phi_surface = self.phi_L1
            fill_factor = 1.0
        else:
            target_volume = fill_factor * V_full
            # Phi -> -inf as r->0 (approaching the secondary's own point
            # mass), and volume(Phi) -> 0 in that same limit, so a root
            # always exists below Phi_L1 for any fill_factor in (0,1];
            # bracket it by stepping down from Phi_L1 until the enclosed
            # volume undershoots the target, then bisect (scipy.brentq)
            # within that bracket.
            phi_hi = self.phi_L1
            step = max(abs(self.phi_L1), 1.0) * 0.5
            phi_lo = self.phi_L1 - step
            for _ in range(200):
                if volume_of(solve_r_grid(phi_lo)) < target_volume:
                    break
                step *= 2.0
                phi_lo = self.phi_L1 - step
            else:
                raise RuntimeError("could not bracket an equipotential for the requested "
                                    "fill_factor/R2 -- volume never dropped below target")

            def volume_error(phi_target):
                return volume_of(solve_r_grid(phi_target)) - target_volume

            phi_surface = brentq(volume_error, phi_lo, phi_hi, xtol=1e-10, rtol=1e-10)
            r_grid = solve_r_grid(phi_surface)

        self.r_grid = r_grid
        self.fill_factor = fill_factor
        self.phi_surface = phi_surface
        self._theta = theta
        self._phi = phi
        self._ntheta = ntheta
        self._nphi = nphi
        self._dtheta = theta[1] - theta[0]
        self._dphi = phi[1] - phi[0]

        # volume-equivalent radius, for a sanity check against Eggleton (1983)
        # R_L/a = 0.49 q^(2/3) / [0.6 q^(2/3) + ln(1+q^(1/3))] when
        # fill_factor=1 (r_volume_equiv_full always matches that check;
        # r_volume_equiv itself only does when not underfilling)
        self.volume = volume_of(self.r_grid)
        self.r_volume_equiv = (3.0 * self.volume / (4.0 * np.pi)) ** (1.0 / 3.0)
        self.r_volume_equiv_full = r_volume_equiv_full

    def radius(self, theta, phi):
        """
        Bilinearly interpolated lobe radius at spherical angles (theta,phi)
        [rad]. Hand-rolled on the known-regular (theta,phi) grid (theta in
        [0,pi], phi periodic in [0,2pi]) rather than via
        RegularGridInterpolator, since this is called many millions of
        times during light-curve/eclipse-timing evaluation and the
        generic bisection-based index search dominates runtime otherwise.
        """
        theta_a = np.asarray(theta, dtype=float)
        phi_a = np.mod(np.asarray(phi, dtype=float), 2.0 * np.pi)
        theta_a, phi_a = np.broadcast_arrays(theta_a, phi_a)

        ti = np.clip(theta_a / self._dtheta, 0.0, self._ntheta - 1 - 1e-12)
        pi_ = phi_a / self._dphi  # phi grid spans [0,2pi] inclusive (periodic wrap already applied)

        i0 = ti.astype(np.int64)
        j0 = pi_.astype(np.int64)
        i1 = np.minimum(i0 + 1, self._ntheta - 1)
        j1 = np.minimum(j0 + 1, self._nphi - 1)
        ft = ti - i0
        fp = pi_ - j0

        r = self.r_grid
        val = ((1 - ft) * (1 - fp) * r[i0, j0]
               + (1 - ft) * fp * r[i0, j1]
               + ft * (1 - fp) * r[i1, j0]
               + ft * fp * r[i1, j1])
        return val

    def surface_points(self):
        """Cartesian (N,3) mesh of the lobe surface, for plotting/rendering."""
        X, Y, Z = self.surface_grid_xyz()
        return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)

    def eggleton_radius(self):
        """Eggleton (1983) approximation to the volume-equivalent lobe radius."""
        return eggleton_radius(self.q)

    def l1_grid_index(self):
        """
        (i,j) index of the surface grid cell closest to the L1 point:
        the grid point where the local gravity magnitude is smallest,
        since L1 is the one genuine saddle point of the potential on the
        lobe surface (gravity -> 0 there in the limit). Used wherever a
        per-grid-cell quantity (temperature, normal) needs a robust
        stand-in value at L1, which is otherwise a numerical artifact of
        which direction the grid approaches the cusp from -- see
        irradiation.fix_l1_cusp.
        """
        X, Y, Z = self.surface_grid_xyz()
        gmag = np.linalg.norm(gravity(X, Y, Z, self.q), axis=-1)
        return np.unravel_index(np.argmin(gmag), gmag.shape)

    def l1_nearest_index(self, points):
        """
        Flat index into `points` (N,3) closest to the L1 point, by the same
        minimum-|gravity| criterion as l1_grid_index -- the point-cloud
        (equal_area_sample) analogue, for an arbitrary (unstructured)
        sample rather than the regular (theta,phi) grid.
        """
        gmag = np.linalg.norm(gravity(points[..., 0], points[..., 1], points[..., 2], self.q),
                               axis=-1)
        return int(np.argmin(gmag))

    def equal_area_sample(self, n):
        """
        ~N roughly equal-AREA sample points on the lobe surface, as a flat
        point cloud (points, normals, areas) rather than surface_grid_xyz's
        rectangular (theta,phi) grid.

        The rectangular grid's cell area dA ~ r^2*sin(theta)*dtheta*dphi
        varies enormously between the poles (sin(theta)->0, vanishing area)
        and the equator (sin(theta)=1, maximum area) -- for ntheta=181, the
        equatorial ring alone has about (2/pi)*ntheta/2 ~= 60x the area-per-
        cell of a near-polar ring, so a huge fraction of any fixed sample
        budget is wasted on tiny near-polar cells that contribute almost
        nothing to an area-weighted sum (irradiation flux, gravity-darkened
        luminosity, ...), while the equator -- doing most of the physical
        work -- is comparatively under-resolved.

        Instead, place N directions with equal solid angle (4*pi/N each) via
        a Fibonacci/golden-angle spiral on the unit sphere -- a standard,
        cheap way to distribute points quasi-uniformly over a sphere -- and
        look up each one's actual lobe radius via self.radius (the same
        fast interpolator used everywhere else), so area ~= r^2*(4*pi/N)
        varies only with the lobe's own (much milder) radius variation,
        not an arbitrary grid artifact. Same "ignore the radial-slope
        contribution to the true differential area" approximation already
        used by surface_area_elements -- adequate for rendering/irradiation,
        not precision photometry.

        Returns (points (n,3), normals (n,3), areas (n,)).
        """
        i = np.arange(n)
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        z = 1.0 - 2.0 * (i + 0.5) / n
        theta = np.arccos(np.clip(z, -1.0, 1.0))
        phi = np.mod(i * golden_angle, 2.0 * np.pi)

        r = self.radius(theta, phi)
        dx = np.sin(theta) * np.cos(phi)
        dy = np.sin(theta) * np.sin(phi)
        dz = np.cos(theta)
        X = self.x2 + r * dx
        Y = r * dy
        Z = r * dz
        points = np.stack([X, Y, Z], axis=-1)

        g = gravity(X, Y, Z, self.q)
        gmag = np.linalg.norm(g, axis=-1, keepdims=True)
        gmag = np.maximum(gmag, 1e-6 * np.median(gmag))
        normals = -g / gmag

        areas = r ** 2 * (4.0 * np.pi / n)
        return points, normals, areas

    def surface_grid_xyz(self):
        """Cartesian (X,Y,Z), each shaped like r_grid (ntheta,nphi)."""
        TH, PH = np.meshgrid(self._theta, self._phi, indexing="ij")
        dx = np.sin(TH) * np.cos(PH)
        dy = np.sin(TH) * np.sin(PH)
        dz = np.cos(TH)
        X = self.x2 + self.r_grid * dx
        Y = self.r_grid * dy
        Z = self.r_grid * dz
        return X, Y, Z

    def surface_normals(self):
        """
        Outward unit normal at each surface grid point, shaped like
        r_grid. The Roche lobe surface is an equipotential of Phi, so its
        outward normal is exactly anti-parallel to the local gravity
        vector (which points down the potential gradient, i.e. into the
        star): n_hat = -g_vec / |g_vec|.
        """
        X, Y, Z = self.surface_grid_xyz()
        g = gravity(X, Y, Z, self.q)
        gmag = np.linalg.norm(g, axis=-1, keepdims=True)
        # gravity -> 0 exactly at the L1 grid point (a saddle of the
        # potential); floor it so the normal there stays well-defined
        # instead of amplifying floating-point noise.
        gmag = np.maximum(gmag, 1e-6 * np.median(gmag))
        return -g / gmag

    def surface_area_elements(self):
        """
        Approximate surface area element at each grid point,
        dA ~ r^2 sin(theta) dtheta dphi (i.e. treating the surface as
        locally spherical). This ignores the extra area contributed by
        the surface's radial slope (dr/dtheta, dr/dphi), which is a
        reasonable approximation away from the elongated tip near L1 but
        somewhat underestimates area there; adequate for illustrative
        rendering, not for precision photometry.
        """
        TH, _ = np.meshgrid(self._theta, self._phi, indexing="ij")
        return self.r_grid ** 2 * np.sin(TH) * self._dtheta * self._dphi

    def gravity_darkened_teff(self, T_eff, beta_grav=0.08):
        """
        Von Zeipel / Lucy (1967) gravity-darkened effective temperature
        map: T_local proportional to g_local^beta_grav, with the
        proportionality constant fixed by requiring the area-weighted mean
        of T_local^4 equal T_eff^4:

            <T_local^4>_area = T_eff^4
            => T_local = T_eff * g_local^beta_grav / <g_local^(4*beta_grav)>_area^(1/4)

        This is the flux-conserving convention -- sigma*T_eff^4 then
        correctly represents the star's mean emergent flux (~luminosity /
        area) for ANY beta_grav, including beta_grav=0 (uniform T_eff
        exactly) as a check that this reduces to the trivial case. Simply
        normalizing by mean gravity (T_local = T_eff*(g/g_mean)^beta_grav,
        an earlier version of this method) is not luminosity-conserving in
        general, since <(g/g_mean)^beta_grav> != 1 unless g is uniform.

        beta_grav=0.08 is the standard value for stars with convective
        envelopes (Lucy 1967), appropriate for a low-mass, Roche-lobe-
        filling secondary; beta_grav=0.25 is the classical von Zeipel
        (1924) value for radiative envelopes. Named beta_grav (not just
        beta) to keep it distinct from the disc temperature profile's own
        power-law index -- see render.disc_powerlaw_teff.

        Uses the (theta,phi) grid's own points/areas; for an
        equal_area_sample point cloud instead, use the module-level
        gravity_darkened_teff(points, area, q, T_eff, beta_grav) directly.
        """
        X, Y, Z = self.surface_grid_xyz()
        points = np.stack([X, Y, Z], axis=-1)
        area = self.surface_area_elements()
        return gravity_darkened_teff(points, area, self.q, T_eff, beta_grav=beta_grav)


def gravity_darkened_teff(points, area, q, T_eff, beta_grav=0.08):
    """
    Von Zeipel / Lucy (1967) gravity-darkened effective temperature at
    arbitrary surface points (N,3) with per-point area (N,) -- the generic
    form of RocheLobe.gravity_darkened_teff, usable with either the
    (theta,phi) grid's own points or an equal_area_sample point cloud. See
    RocheLobe.gravity_darkened_teff's docstring for the physics/formula.
    """
    g = gravity(points[..., 0], points[..., 1], points[..., 2], q)
    gmag = np.linalg.norm(g, axis=-1)
    mean_g_4beta = np.sum(gmag ** (4.0 * beta_grav) * area) / np.sum(area)
    return T_eff * gmag ** beta_grav / mean_g_4beta ** 0.25
