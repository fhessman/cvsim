# magnetic.py
"""
Coordinate-system groundwork for a magnetic primary (white dwarf), in
preparation for layering a magnetic (polar/AM Her-type) accretion model on
top of the existing Roche-lobe/disc/stream light-curve machinery.

The magnetic axis is tilted away from the orbital spin axis (z_hat in the
corotating (x,y,z) frame used throughout this package -- see eclipse.py's
module docstring) by an obliquity theta_1, oriented in azimuth by phi_1
(params.SystemParams.theta_1/phi_1, degrees at the config/CLI boundary,
radians here and everywhere else in the physics layer, same convention as
disc.py/eclipse.py).

Assumes synchronous rotation (the standard "polar" case): the WD's spin
axis is aligned with the orbital angular momentum axis, so the magnetic
axis is FIXED in the corotating frame -- no separate spin phase to track,
and no dependence on orbital phase beyond what the corotating-frame
points/normals it's applied to already carry.
"""

import numpy as np


def magnetic_frame(theta_1, phi_1):
    """
    Return (m_hat, e1_hat, e2_hat): a right-handed orthonormal basis for
    the primary's tilted magnetic coordinate system, in the corotating
    frame.

    m_hat is the magnetic axis (north pole) direction:

        m_hat = (sin(theta_1)*cos(phi_1), sin(theta_1)*sin(phi_1), cos(theta_1))

    e1_hat/e2_hat span the magnetic equator and set the magnetic-longitude
    origin: e1_hat is z_hat's component perpendicular to m_hat, renormalized
    (Gram-Schmidt -- the same construction eclipse.observer_frame uses to
    build eX/eY against the orbital axis), i.e. e1_hat points along the
    meridian containing both m_hat and +z. In the degenerate untilted case
    (m_hat parallel to +/-z_hat, theta_1 ~ 0 or pi), that meridian isn't
    defined, so e1_hat falls back to +x directly -- consistent with phi_1's
    own azimuth reference (measured from +x, see this module's docstring),
    so the longitude origin varies continuously as theta_1 -> 0 rather than
    jumping. e2_hat = m_hat x e1_hat completes the right-handed set.
    """
    m_hat = np.array([np.sin(theta_1) * np.cos(phi_1),
                       np.sin(theta_1) * np.sin(phi_1),
                       np.cos(theta_1)])
    z_hat = np.array([0.0, 0.0, 1.0])
    e1 = z_hat - np.dot(z_hat, m_hat) * m_hat
    norm = np.linalg.norm(e1)
    if norm < 1e-9:
        e1 = np.array([1.0, 0.0, 0.0])
    else:
        e1 = e1 / norm
    e2 = np.cross(m_hat, e1)
    return m_hat, e1, e2


def magnetic_colatitude_longitude(points, theta_1, phi_1, center=None):
    """
    Magnetic colatitude (angle from the magnetic axis m_hat, [0,pi]) and
    longitude (azimuth about m_hat from e1_hat, [-pi,pi]) of each point (or
    outward unit normal -- both give the same angles for points on a
    sphere centered at `center`) in the corotating frame.

    points: (...,3) array -- either primary surface points (pass `center`,
    the primary's center, to recenter first) or unit outward normals
    (center=None, the default: points are used as-is, e.g.
    render.TemperatureMaps.primary_normals, already direction-only and
    R1-independent -- see build_temperature_maps' n_primary docstring).

    Returns (colat, long), each shape points.shape[:-1].
    """
    m_hat, e1_hat, e2_hat = magnetic_frame(theta_1, phi_1)
    v = np.asarray(points, dtype=float)
    if center is not None:
        v = v - np.asarray(center)
    v = v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-300)
    mu = np.clip(np.einsum("...i,i->...", v, m_hat), -1.0, 1.0)
    colat = np.arccos(mu)
    x1 = np.einsum("...i,i->...", v, e1_hat)
    x2 = np.einsum("...i,i->...", v, e2_hat)
    long = np.arctan2(x2, x1)
    return colat, long


DEFAULT_FIELD_LINE_R_MAX_FACTOR = 10.0  # see field_line_points' docstring


def field_line_footpoints(n, r_max_factor=DEFAULT_FIELD_LINE_R_MAX_FACTOR):
    """
    n magnetic footpoints (colatitude theta_m0, longitude phi_m0) on a
    regular latitude/longitude grid over the northern magnetic hemisphere:
    n_theta colatitude rings, evenly spaced between the pole-proximal cutoff
    theta_min (below) and the magnetic equator, each carrying the same
    n_phi footpoints evenly spaced all the way around in azimuth --
    n_theta*n_phi as close to n as a whole-number grid allows. This makes
    both marginal distributions genuinely uniform (n_phi equally-spaced
    azimuths per ring; n_theta equally-spaced colatitudes), unlike a
    quasi-random spiral, which is what a small n most visibly needs.

    EXCLUDES the region nearest the magnetic pole (small theta_m0), where
    a real dipole field line's equatorial-crossing radius r_eq =
    R1/sin^2(theta_m0) diverges (an "open" field line): capped here at
    r_eq <= r_max_factor*R1, i.e. theta_m0 >= arcsin(1/sqrt(r_max_factor)).
    r_max_factor is a geometry-only placeholder cap -- see this module's
    docstring and field_line_points' own r_max_factor -- typically set by
    the caller (e.g. plots.field_line_outlines) from the actual system
    geometry: the primary's own Roche lobe if there's no disc to truncate
    the magnetosphere sooner, else (2x) the disc's inner radius.

    Returns (theta_m0, phi_m0), each length n_theta*n_phi (may differ
    slightly from n).
    """
    theta_min = np.arcsin(1.0 / np.sqrt(r_max_factor))
    n_theta = max(1, int(round(np.sqrt(n))))
    n_phi = max(1, int(round(n / n_theta)))
    # theta_min itself is a perfectly good (the largest, r_eq=r_max_factor*R1)
    # loop, so the innermost ring sits exactly there rather than being
    # cell-centered away from it -- otherwise, since r_eq ~ 1/sin(theta_m0)^2
    # is so steep near theta_min, even a modest half-cell offset falls well
    # short of the intended outer cap (e.g. the primary's actual Roche lobe
    # extent -- see field_line_outlines). The equator (theta_m0=pi/2) *is*
    # degenerate (a zero-length loop), so it's excluded via endpoint=False.
    theta_rings = np.linspace(theta_min, np.pi / 2.0, n_theta, endpoint=False)
    phis = (np.arange(n_phi) + 0.5) / n_phi * 2.0 * np.pi
    theta_m0, phi_m0 = np.meshgrid(theta_rings, phis, indexing="ij")
    return theta_m0.ravel(), phi_m0.ravel()


def field_line_points(center, R1, theta_1, phi_1, n_field_1, n_per_line=100,
                       r_max_factor=DEFAULT_FIELD_LINE_R_MAX_FACTOR):
    """
    ~n_field_1 closed dipole field-line loops (see field_line_footpoints:
    the actual count is n_theta*n_phi, the nearest whole-number grid),
    each both feet planted on the primary's surface (radius R1, center
    `center`, corotating frame). Returns a list of those arrays, each
    (n_per_line,3).

    Each loop lies entirely within one magnetic meridian plane (constant
    magnetic longitude phi_m0, see field_line_footpoints) and follows the
    standard dipole field-line equation

        r(theta_m) = r_eq * sin(theta_m)^2,  theta_m in [theta_m0, pi-theta_m0]

    where r_eq = R1/sin(theta_m0)^2 is set by the footpoint colatitude
    theta_m0 -- the loop's other, "mirror" footpoint is at pi-theta_m0,
    the same longitude, symmetric across the magnetic equator.

    r_max_factor: caps each loop's equatorial-crossing radius at
    r_max_factor*R1 (see field_line_footpoints) -- the caller should pass
    one derived from the actual system geometry (e.g.
    plots.field_line_outlines does: the primary's own Roche lobe extent
    if there's no disc, else the disc's inner edge), not the bare
    DEFAULT_FIELD_LINE_R_MAX_FACTOR placeholder, wherever that geometry is
    available.

    Geometry only: field strength/direction along the line, and any
    truncation by an actual magnetosphere, aren't modeled yet.
    """
    theta_m0, phi_m0 = field_line_footpoints(n_field_1, r_max_factor=r_max_factor)
    m_hat, e1_hat, e2_hat = magnetic_frame(theta_1, phi_1)
    center = np.asarray(center, dtype=float)
    s = np.linspace(0.0, 1.0, n_per_line)

    lines = []
    for th0, ph0 in zip(theta_m0, phi_m0):
        theta_m = th0 + s * (np.pi - 2.0 * th0)
        r = R1 / np.sin(th0) ** 2 * np.sin(theta_m) ** 2
        pos = (r * np.cos(theta_m))[:, None] * m_hat \
            + (r * np.sin(theta_m) * np.cos(ph0))[:, None] * e1_hat \
            + (r * np.sin(theta_m) * np.sin(ph0))[:, None] * e2_hat
        lines.append(center + pos)
    return lines


def field_line_to_point(center, R1, theta_1, phi_1, target_point, n=100):
    """
    Trace the single dipole field line connecting the primary's surface
    (radius R1) to `target_point` (corotating frame, any radius >= R1
    from `center`) -- the accretion-spot connection: an accretion stream
    reaches swept angle angle_acc at `target_point`, and is assumed to
    then follow this field line down onto the surface (see
    stream.angle_acc_index, render.build_temperature_maps' angle_acc).

    Unlike field_line_points (a full pole-to-pole loop), this traces only
    the shorter arc from target_point's own (nearer) surface footpoint out
    to target_point itself -- not the rest of the loop beyond it.

    Returns (points, footpoint_dir): points is (n,3) with points[0] exactly
    the surface footpoint (r=R1) and points[-1] exactly target_point (set
    directly, not just algebraically implied, to avoid any trig round-trip
    drift); footpoint_dir is footpoint_dir = (points[0]-center)/R1, the
    footpoint's unit outward normal -- e.g. for accretion_spot_mask.
    """
    target = np.asarray(target_point, dtype=float)
    center = np.asarray(center, dtype=float)
    m_hat, e1_hat, e2_hat = magnetic_frame(theta_1, phi_1)

    theta_m, phi_m = magnetic_colatitude_longitude(target, theta_1, phi_1, center=center)
    theta_m = float(theta_m)
    phi_m = float(phi_m)
    r_conn = float(np.linalg.norm(target - center))
    sin_tm = np.sin(theta_m)

    if abs(sin_tm) < 1e-9:
        # degenerate: target sits (anti)parallel to the magnetic axis --
        # r_eq is undefined there (a purely radial field line straight out
        # from the pole); fall back to a straight radial segment.
        sign = 1.0 if np.cos(theta_m) > 0.0 else -1.0
        footpoint = center + sign * R1 * m_hat
        s = np.linspace(0.0, 1.0, n)[:, None]
        points = footpoint + s * (target - footpoint)
        points[-1] = target
        return points, sign * m_hat

    r_eq = r_conn / sin_tm ** 2
    theta_m0 = np.arcsin(min(1.0, np.sqrt(R1 / r_eq)))  # northern reference footpoint
    theta_start = theta_m0 if theta_m <= np.pi / 2.0 else (np.pi - theta_m0)

    s = np.linspace(0.0, 1.0, n)
    theta = theta_start + s * (theta_m - theta_start)
    r = r_eq * np.sin(theta) ** 2
    pos = (r * np.cos(theta))[:, None] * m_hat \
        + (r * np.sin(theta) * np.cos(phi_m))[:, None] * e1_hat \
        + (r * np.sin(theta) * np.sin(phi_m))[:, None] * e2_hat
    points = center + pos
    points[-1] = target
    footpoint_dir = (points[0] - center) / R1
    return points, footpoint_dir


def accretion_spot_mask(normals, footpoint_dir, spot_acc):
    """
    Boolean mask over primary-centered unit direction vectors `normals`
    (...,3): True within angular radius spot_acc [rad] of footpoint_dir
    (a unit vector) -- the accretion spot's footprint on the primary's
    surface (see field_line_to_point).
    """
    cos_angle = np.einsum("...i,i->...", normals, footpoint_dir)
    return cos_angle >= np.cos(spot_acc)


def primary_surface_temperature(normals, T_1, footpoint_dirs=None, spot_acc=0.0, T_acc=None):
    """
    Per-point primary surface temperature over primary-centered unit
    direction vectors `normals` (...,3): uniform T_1 everywhere, except
    within angular radius spot_acc [rad] of any of footpoint_dirs (a
    list of unit vectors, one per active accretion angle -- see
    params._parse_angle_acc) where it's T_acc -- the accretion spot(s) where
    field-line-channeled material lands (see field_line_to_point). Every
    spot shares the same spot_acc/T_acc (only the connection angle, and
    so the footpoint, differs between them). Returns uniform T_1 (no
    spot) if footpoint_dirs is None/empty or T_acc is None -- the
    caller's single gate for "no active accretion spot this run" (see
    render.build_temperature_maps' angle_acc/T_acc validity checks), so every
    consumer (the primary's own observer-facing flux, and its irradiation
    of the secondary) sees the identical spot(s), or none.
    """
    T = np.full(np.asarray(normals).shape[:-1], float(T_1))
    if footpoint_dirs and T_acc is not None:
        mask = np.zeros(T.shape, dtype=bool)
        for footpoint_dir in footpoint_dirs:
            mask |= accretion_spot_mask(normals, footpoint_dir, spot_acc)
        T[mask] = float(T_acc)
    return T


def primary_limb_coefficient(normals, u_1, footpoint_dirs=None, spot_acc=0.0, u_acc=0.0):
    """
    Per-point primary limb-darkening coefficient (I=I0*(1-u+u*mu), see
    lightcurve.py's module docstring) over primary-centered unit direction
    vectors `normals` (...,3): uniform u_1 everywhere, except within
    angular radius spot_acc [rad] of any of footpoint_dirs (a list of
    unit vectors, see field_line_to_point) where it's u_acc -- every
    accretion spot's own limb law, independent of the rest of the
    primary's.

    A negative u_acc mimics limb-BRIGHTENING (I increases toward the limb,
    mu->0, rather than the usual darkening) -- a crude phenomenological
    stand-in for cyclotron beaming, not a real angle-resolved emission
    pattern.

    Unlike primary_surface_temperature's T_acc, u_acc has no None-gated
    "off" state of its own (0.0, like u_1/u_2/u_d elsewhere, already means
    "flat/no limb law") -- footpoint_dirs is the sole gate for "no active
    accretion spot," matching spot_acc's own always-real-valued
    convention. Returns uniform u_1 if footpoint_dirs is None/empty.
    """
    u = np.full(np.asarray(normals).shape[:-1], float(u_1))
    if footpoint_dirs:
        mask = np.zeros(u.shape, dtype=bool)
        for footpoint_dir in footpoint_dirs:
            mask |= accretion_spot_mask(normals, footpoint_dir, spot_acc)
        u[mask] = float(u_acc)
    return u
