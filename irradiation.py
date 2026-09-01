# irradiation.py
"""
Irradiation of the secondary's surface by the (partially disc-occulted)
primary and the (partially star-occulted) disc, assuming full
thermalization: absorbed flux adds directly to the local energy budget,

    sigma*T_local^4 = sigma*T_intrinsic^4 + F_star_irrad + F_disc_irrad

All geometric factors (R1/r, areas/r^2) are dimensionless ratios, so they
can be evaluated directly in units of the orbital separation a -- no
conversion to physical length units is needed; only T (Kelvin) and sigma
carry physical units, giving F in W/m^2.

Star -> secondary
------------------
The primary is a genuine extended, Lambertian (optionally limb-darkened)
sphere here, not a point source: the disc sits close enough to it (its
inner radius is often only a handful of primary radii) that the disc can
occult PART of the primary's surface as seen from a given secondary point
-- e.g. the equatorial band facing the disc while the poles stay visible
-- which a single "is the whole primary blocked or not" test cannot
represent. So star_irradiation_flux samples the primary's own surface and
sums a full view-factor integral, exactly mirroring the disc's own
treatment below, with per-(primary point, secondary point) disc-blocking
tests (exact either way: a solid-containment march for a flared disc, an
exact finite-segment plane crossing for a flat one). In the far-field,
no-occlusion limit this still reduces exactly to the point-source result
(a uniformly-emitting Lambertian sphere looks like an isotropic point
source from far away), so nothing changes for systems where the disc
never gets close enough to the primary to matter.

Disc -> secondary
------------------
The disc is not small, so it needs a genuine surface (view-factor)
integral, not a point-source shortcut -- the flux at a secondary point
depends on both the emitting disc element's foreshortening (cosine of
its own tilt relative to the line to the secondary point) and the
receiving secondary element's foreshortening (cosine of the angle
between its outward normal and the line to the disc element), plus the
star possibly blocking that sightline ("disc behind the star").
"""

import numpy as np

from roche import gravity
from utils import with_progress

SIGMA_SB = 5.670374419e-8  # W / (m^2 K^4)


def star_irradiation_flux(sec_points, sec_normals, star_center, R1, T_star,
                           disc=None, n_primary=200, u_primary=0.0, chunk=150, n_t=40,
                           accretion_footpoint_dirs=None, spot_acc=0.0, T_acc=None,
                           u_acc=0.0):
    """
    Absorbed flux [W/m^2] at each secondary surface point from the
    primary, via the full view-factor surface integral (Lambertian,
    optionally limb-darkened primary elements; Lambertian receiver) --
    the primary analogue of disc_irradiation_flux, NOT a point source.
    A point-source treatment can only say "the whole primary is visible"
    or "blocked": but the disc sits close to the primary (its inner
    radius is often only a handful of primary radii), so it can occult
    PART of the primary's surface as seen from a given secondary point --
    e.g. the equatorial band facing the disc, while the poles stay
    visible -- which needs each primary surface element's own occlusion
    tested individually. (In the limit R1<<r with no occlusion, this
    correctly reduces to the point-source result SIGMA_SB*T^4*(R1/r)^2*
    cos_theta2 -- the standard "a uniform Lambertian sphere looks like an
    isotropic point source from far away" result -- so nothing changes
    for systems where the disc never gets close enough to matter.)

    Disc shadowing, tested per (primary point, secondary point) pair:
    flared (opening_angle>0) discs use the exact solid-containment
    segment march (_segment_blocked_by_disc, disc.contains_3d); flat
    discs use the exact finite-segment z=0-plane-crossing test
    (disc.segment_blocked) instead of a sampled march, which would almost
    always miss a flat, zero-thickness disc (see disc.visible_from's
    docstring for the same issue).

    n_primary: number of roughly equal-area points sampled over the
    primary's WHOLE surface (lightcurve.sphere_points_full) -- independent
    of --n_areas_primary (the observer-facing render/flux resolution),
    which has no bearing on this irradiation integral; see
    disc_irradiation_flux's own n_disc_irrad for the analogous split on
    the disc side.

    u_primary: linear limb-darkening coefficient (0=off), applied to the
    primary's own emission angle exactly as u_disc is in
    disc_irradiation_flux -- a limb-darkened primary element emits less
    toward a secondary point that sees it near its own limb.

    sec_points, sec_normals: arrays shaped (Nsec,3), outward unit normals.

    accretion_footpoint_dirs/spot_acc/T_acc/u_acc: the primary's
    accretion spot(s) (see magnetic.field_line_to_point/accretion_spot_mask,
    render.build_temperature_maps' angle_acc) -- a per-point temperature AND
    limb-darkening-coefficient override near each field line's footpoint,
    baked into this integral (T_acc/u_acc instead of the uniform
    T_star/u_primary everywhere) the same way it's baked into the
    primary's own emission map, so every spot irradiates the secondary too
    (including its own limb law -- e.g. a cyclotron-beaming-like negative
    u_acc changes how much of a spot's flux reaches a given secondary
    point, not just how bright the spot looks to the observer).
    accretion_footpoint_dirs=None/empty (default, i.e. no active spot)
    reduces exactly to the old uniform-T_star/u_primary behavior.
    """
    from lightcurve import sphere_points_full
    from magnetic import primary_surface_temperature, primary_limb_coefficient

    star_pts, star_normals, star_areas = sphere_points_full(star_center, R1, n_primary)
    star_T = primary_surface_temperature(star_normals, T_star, accretion_footpoint_dirs,
                                          spot_acc, T_acc)
    star_u = primary_limb_coefficient(star_normals, u_primary, accretion_footpoint_dirs,
                                       spot_acc, u_acc)
    n_p = star_pts.shape[0]
    flux = np.zeros(sec_points.shape[0])

    chunk_starts = list(range(0, n_p, chunk))
    for i0 in with_progress(chunk_starts, len(chunk_starts), "primary irradiation"):
        i1 = min(i0 + chunk, n_p)
        P1 = star_pts[i0:i1][:, None, :]              # (c,1,3), broadcasts below
        N1 = star_normals[i0:i1][:, None, :]           # (c,1,3)
        A1 = star_areas[i0:i1][:, None]                # (c,1)
        T1 = star_T[i0:i1][:, None]                     # (c,1)
        U1 = star_u[i0:i1][:, None]                     # (c,1)
        P2 = sec_points[None, :, :]                     # (1,Nsec,3)
        N2 = sec_normals[None, :, :]                     # (1,Nsec,3)

        r_vec = P2 - P1                                 # (c,Nsec,3), broadcast
        r = np.linalg.norm(r_vec, axis=-1)              # (c,Nsec)
        r_hat = r_vec / r[..., None]

        cos_theta1 = np.clip(np.sum(N1 * r_hat, axis=-1), 0.0, None)
        cos_theta2 = np.clip(np.sum(N2 * (-r_hat), axis=-1), 0.0, None)

        if disc is not None and disc.opening_angle > 0.0:
            blocked = _segment_blocked_by_disc(P1, P2, disc, n_t=n_t)
        elif disc is not None:
            blocked = disc.segment_blocked(P1, P2)
        else:
            blocked = False

        limb = 1.0 - U1 + U1 * cos_theta1
        dF = SIGMA_SB * T1 ** 4 / np.pi * cos_theta1 * limb * cos_theta2 / r ** 2 * A1
        dF = np.where(blocked, 0.0, dF)
        flux += dF.sum(axis=0)

    return flux


def _segment_blocked_by_disc(P1, P2, disc, n_t=40):
    """
    True where the straight segment P1->P2 passes through the disc SOLID
    (disc.contains_3d) at any sampled interior parameter -- the disc
    analogue of _segment_blocked_by_sphere, needed because the disc solid
    isn't a simple analytic shape a closest-approach formula can handle.
    P1, P2 broadcast against each other; returns their common broadcast
    shape (leading dims, i.e. without the trailing size-3 axis).
    """
    ts = np.linspace(0.02, 0.98, n_t)  # exclude the endpoints themselves
    lead_shape = np.broadcast(P1[..., 0], P2[..., 0]).shape
    blocked = np.zeros(lead_shape, dtype=bool)
    for t in ts:
        seg = P1 * (1.0 - t) + P2 * t
        blocked |= disc.contains_3d(seg[..., 0], seg[..., 1], seg[..., 2])
    return blocked


def _segment_blocked_by_sphere(P1, P2, center, radius):
    """
    True where the straight segment P1->P2 passes within `radius` of
    `center` at a parameter strictly between the endpoints (i.e. the
    sphere sits between them, not beyond either end). P1, P2 broadcast
    against each other; returns an array of their common broadcast shape.
    """
    d = P2 - P1
    dlen2 = np.einsum("...i,...i->...", d, d)
    v = center - P1
    t = np.clip(np.einsum("...i,...i->...", v, d) / np.maximum(dlen2, 1e-300), 0.0, 1.0)
    closest = P1 + t[..., None] * d
    dist2 = np.einsum("...i,...i->...", closest - center, closest - center)
    return (dist2 < radius ** 2) & (t > 0.0) & (t < 1.0)


def disc_irradiation_flux(sec_points, sec_normals, disc_pos, disc_area, disc_T,
                           star_center, R1, disc_normal=None, chunk=150, u_disc=0.0):
    """
    Absorbed flux [W/m^2] at each secondary surface point from the disc,
    via the full view-factor surface integral (Lambertian disc elements,
    Lambertian receiver), excluding disc elements whose sightline to a
    given secondary point passes through the primary ("disc behind the
    star"). Processed in chunks over the disc cells to bound memory.

    sec_points, sec_normals: (Nsec,3) arrays (already flattened).
    disc_pos: (Ndisc,3), disc_area: (Ndisc,), disc_T: (Ndisc,) [K].
    disc_normal: (Ndisc,3) outward unit normal per disc cell. If None,
    falls back to the flat zero-thickness-disc assumption (normal=+-z_hat,
    i.e. cos_theta1=|r_hat_z|, emitting from whichever face the secondary
    point is on) -- pass the real per-cell normals from
    disc.all_surfaces_grid for the flared (opening_angle>0) disc, where
    the cone/edge surfaces are not all z_hat-normal.

    u_disc: linear limb-darkening coefficient (I=I0*(1-u+u*mu), 0=off),
    applied to the disc's own emission angle cos_theta1 -- a limb-darkened
    element emits less toward a secondary point that sees it near its own
    limb, exactly as it would emit less toward Earth from the same angle.
    This is the same coefficient render.physical_light_curve uses for the
    disc's observer-facing flux, so it must also change the secondary's
    absorbed (and hence Tsec) here -- unlike u_secondary (the secondary's
    own emission pattern), which has no effect on what it absorbs.

    Note: this does NOT check whether a disc cell's own sightline to the
    secondary point is blocked by other disc material (self-shadowing of
    the flared disc's irradiation of the secondary) -- only star-blocks-
    disc is excluded. For a modestly flared disc this is a second-order
    effect; see disc.occluded_by_disc for the (cheaper, O(Nsec) rather
    than O(Ndisc*Nsec)) self-occlusion test used in the observer-facing
    renders.
    """
    n_disc = disc_pos.shape[0]
    flux = np.zeros(sec_points.shape[0])

    chunk_starts = list(range(0, n_disc, chunk))
    for i0 in with_progress(chunk_starts, len(chunk_starts), "secondary irradiation"):
        i1 = min(i0 + chunk, n_disc)
        P1 = disc_pos[i0:i1][:, None, :]            # (c,1,3), broadcasts below
        A1 = disc_area[i0:i1][:, None]               # (c,1)
        T1 = disc_T[i0:i1][:, None]                  # (c,1)
        P2 = sec_points[None, :, :]                   # (1,Nsec,3)
        N2 = sec_normals[None, :, :]                   # (1,Nsec,3)

        r_vec = P2 - P1                               # (c,Nsec,3), broadcast
        r = np.linalg.norm(r_vec, axis=-1)            # (c,Nsec)
        r_hat = r_vec / r[..., None]

        if disc_normal is None:
            cos_theta1 = np.abs(r_hat[..., 2])        # disc normal = +-z_hat, thin two-sided disc
        else:
            # cone/edge surfaces have a single well-defined outward face
            # (real solid material behind them) -- one-sided emission,
            # unlike the idealized flat sheet's two independent faces.
            N1 = disc_normal[i0:i1][:, None, :]       # (c,1,3)
            cos_theta1 = np.clip(np.sum(N1 * r_hat, axis=-1), 0.0, None)
        cos_theta2 = np.clip(np.sum(N2 * (-r_hat), axis=-1), 0.0, None)

        blocked = _segment_blocked_by_sphere(P1, P2, star_center, R1)

        limb = 1.0 - u_disc + u_disc * cos_theta1
        dF = SIGMA_SB * T1 ** 4 / np.pi * cos_theta1 * limb * cos_theta2 / r ** 2 * A1
        dF = np.where(blocked, 0.0, dF)
        flux += dF.sum(axis=0)

    return flux


def secondary_irradiated_teff(lobe, sec_points, sec_normals, sec_areas, T_eff2, T_eff1, R1,
                               disc_pos, disc_area, disc_T,
                               disc=None, disc_normal=None, beta_grav=0.08, disc_chunk=150,
                               u_disc=0.0, u_primary=0.0, n_primary_irrad=200, star_chunk=150,
                               accretion_footpoint_dirs=None, spot_acc=0.0, T_acc=None,
                               u_acc=0.0):
    """
    Full local effective-temperature map of the secondary: gravity
    darkening (intrinsic) plus full-thermalization irradiation from the
    (disc-occulted) primary and the (star-occulted) disc. Returns a flat
    (N,) array matching sec_points/sec_normals/sec_areas's own N -- e.g.
    RocheLobe.equal_area_sample's output, so callers control the surface
    sampling (equal-area point cloud, or the (theta,phi) grid via
    surface_grid_xyz/surface_normals/surface_area_elements) rather than
    this function assuming one particular representation.

    disc: the actual disc.Disc object, used for star_irradiation_flux's
    EXACT shadow test against its real solid geometry (r_in, r_out,
    opening_angle).

    disc_normal: per-disc-cell outward unit normals (Ndisc,3); pass the
    real normals from disc.all_surfaces_grid for a flared (opening_angle>0)
    disc, or omit for the flat zero-thickness disc (see
    disc_irradiation_flux's docstring).

    u_disc/u_primary: each body's limb-darkening coefficient -- see
    star_irradiation_flux/disc_irradiation_flux's docstrings; both reduce
    how much that body irradiates the secondary, not just how it looks to
    the observer.

    n_primary_irrad: number of surface points sampled over the primary
    for its own (coarser, cheaper) irradiation integral -- independent of
    the primary's render/light-curve resolution, exactly like n_disc_irrad
    is for the disc (see build_temperature_maps).

    accretion_footpoint_dirs/spot_acc/T_acc/u_acc: forwarded straight to
    star_irradiation_flux -- the primary's accretion spot(s) (if any are
    active) irradiate the secondary too, not just their own emission map.
    """
    from roche import gravity_darkened_teff as _gravity_darkened_teff

    star_center = np.array([lobe.x1, 0.0, 0.0])
    T_intrinsic = _gravity_darkened_teff(sec_points, sec_areas, lobe.q, T_eff2, beta_grav=beta_grav)

    F_star = star_irradiation_flux(sec_points, sec_normals, star_center, R1, T_eff1, disc=disc,
                                    n_primary=n_primary_irrad, u_primary=u_primary, chunk=star_chunk,
                                    accretion_footpoint_dirs=accretion_footpoint_dirs,
                                    spot_acc=spot_acc, T_acc=T_acc, u_acc=u_acc)
    if disc is not None and disc.is_empty:
        # a null disc (no disc configured, see disc.Disc.is_empty) has zero
        # area everywhere, so its irradiation contribution is always exactly
        # zero -- skip the O(Ndisc*Nsec) view-factor/occlusion integral
        # entirely rather than running it over degenerate zero-area cells.
        F_disc = np.zeros(sec_points.shape[0])
    else:
        F_disc = disc_irradiation_flux(sec_points, sec_normals, disc_pos, disc_area, disc_T,
                                        star_center, R1, disc_normal=disc_normal, chunk=disc_chunk,
                                        u_disc=u_disc)

    T4 = T_intrinsic ** 4 + (F_star + F_disc) / SIGMA_SB
    Tmap = T4 ** 0.25
    return fix_l1_cusp(Tmap, lobe, points=sec_points)


def fix_l1_cusp(Tmap, lobe, points=None):
    """
    The L1 point is a genuine geometric cusp of the Roche lobe (the two
    lobes meet at a single point there): the surface has no well-defined
    local outward normal in the limit approaching it, so both the
    gravity-darkened T and the irradiation-derived normal at that one
    sample are artifacts of which direction the numerical sampling
    happens to approach it from, not real surface temperature. Its area
    is a single sample's worth (negligible), so replace it with the
    median of its immediate neighbors -- a cosmetic fix for display, not
    a physical claim about that single point. Used both to clean up the
    secondary's own temperature map and (render.py) to get a sensible
    constant temperature for the accretion stream, which is assigned
    "the temperature of the L1 point".

    points: if given, `Tmap` is a flat (N,) array over this (N,3) point
    cloud (e.g. RocheLobe.equal_area_sample's output) rather than the
    (theta,phi) grid -- neighbors are then the k nearest points by
    Euclidean distance, instead of the grid's (theta,phi) adjacency.
    """
    if getattr(lobe, "fill_factor", 1.0) < 1.0:
        return Tmap  # underfilling: surface never reaches L1, no cusp to fix

    if points is not None:
        i_l1 = lobe.l1_nearest_index(points)
        gmag = np.linalg.norm(gravity(points[:, 0], points[:, 1], points[:, 2], lobe.q), axis=-1)
        if gmag[i_l1] > 1e-3 * np.median(gmag):
            return Tmap  # no genuine near-singular point found
        d2 = np.sum((points - points[i_l1]) ** 2, axis=-1)
        k = min(6, len(points) - 1)
        cand = np.argpartition(d2, k)[:k + 1]
        neighbor_idx = cand[cand != i_l1]
        out = Tmap.copy()
        out[i_l1] = np.median(Tmap[neighbor_idx])
        return out

    i, j = lobe.l1_grid_index()
    X, Y, Z = lobe.surface_grid_xyz()
    gmag = np.linalg.norm(gravity(X, Y, Z, lobe.q), axis=-1)
    if gmag[i, j] > 1e-3 * np.median(gmag):
        return Tmap  # no genuine near-singular point found
    jm1, jp1 = (j - 1) % Tmap.shape[1], (j + 1) % Tmap.shape[1]
    neighbors = [Tmap[i, jm1], Tmap[i, jp1]]
    if i > 0:
        neighbors.append(Tmap[i - 1, j])
    if i < Tmap.shape[0] - 1:
        neighbors.append(Tmap[i + 1, j])
    out = Tmap.copy()
    out[i, j] = np.median(neighbors)
    return out
