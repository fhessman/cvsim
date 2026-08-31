# stream.py
"""
Ballistic accretion stream from L1, following Lubow & Shu (1975, ApJ 198,
383).  A test particle leaves the L1 nozzle with a small velocity set by
the local sound speed eps = c_s/(Omega*a) and is then integrated under
gravity + Coriolis + centrifugal forces in the corotating frame (same
convention as roche.py: units a=1, G(M1+M2)=1, Omega=1, COM at origin).

The initial velocity direction (angle theta from the -x axis, i.e. from
the line towards the primary) is set by the L1-nozzle analysis of Lubow &
Shu (their eqns 13 and 24):

    A = mu1/|x_L1-x1|^3 + mu2/|x_L1-x2|^3
    cos(2*theta) = -4/(3A) + sqrt(1 - 8/(9A))

with initial velocity (vx,vy) = eps*(-cos(theta), +sin(theta)).  The
trajectory is confined to the orbital plane (z=0); this is adequate for
eclipse-timing purposes since the stream lies close to the orbital plane
until it plunges into the disc.
"""

import numpy as np
from scipy.integrate import solve_ivp


def lubow_shu_angle(lobe):
    """Return (A, theta [rad]) for the L1 nozzle, given a roche.RocheLobe."""
    mu1, mu2, x1, x2, xL1 = lobe.mu1, lobe.mu2, lobe.x1, lobe.x2, lobe.x_L1
    A = mu1 / abs(xL1 - x1) ** 3 + mu2 / abs(xL1 - x2) ** 3
    theta = 0.5 * np.arccos(-4.0 / (3.0 * A) + np.sqrt(1.0 - 8.0 / (9.0 * A)))
    return A, theta


def _eom(t, state, mu1, mu2, x1, x2):
    x, y, vx, vy = state
    r1 = np.hypot(x - x1, y)
    r2 = np.hypot(x - x2, y)
    ax = 2.0 * vy + x - mu1 * (x - x1) / r1 ** 3 - mu2 * (x - x2) / r2 ** 3
    ay = -2.0 * vx + y - mu1 * y / r1 ** 3 - mu2 * y / r2 ** 3
    return [vx, vy, ax, ay]


def _eom_with_angle(t, state, mu1, mu2, x1, x2):
    """
    _eom plus a 5th state, the primary-centered azimuth (radians) swept
    since t=0, integrated directly (d(theta)/dt = (dx*vy - dy*vx)/r1^2,
    the standard angular-velocity-about-a-point formula) rather than
    recovered from arctan2(y, x-x1) after the fact, so it accumulates
    past +-pi instead of wrapping -- see integrate_stream's
    stream_angle_deg.
    """
    x, y, vx, vy = state[0], state[1], state[2], state[3]
    dxdt, dydt, ax, ay = _eom(t, state[:4], mu1, mu2, x1, x2)
    r1_sq = (x - x1) ** 2 + y ** 2
    dtheta = ((x - x1) * vy - y * vx) / r1_sq
    return [dxdt, dydt, ax, ay, dtheta]


def integrate_stream(lobe, eps=0.02, t_max=8.0, r_min_primary=0.02, max_step=0.01,
                      stream_angle_deg=None):
    """
    Integrate the ballistic trajectory from L1 in the orbital plane.

    eps : c_s(T2)/(Omega*a), the dimensionless sound speed setting the
          initial nozzle velocity (typically ~0.01-0.05 for CV secondaries).

    stream_angle_deg: how far around the primary the trajectory is
    allowed to sweep before stopping -- the cumulative (unwrapped)
    primary-centered azimuth, tracked from t=0, 0 deg nominally facing
    the secondary (+x, the usual convention -- see e.g.
    disc_impact_index's own nu) and 180 deg directly behind the primary;
    may exceed 360 to let the trajectory loop around the primary more
    than once. None (the default) instead keeps the original behavior:
    stop as soon as the particle first comes within r_min_primary of the
    primary. r_min_primary remains an active safety floor even when
    stream_angle_deg is given (the particle plunging into the primary
    before ever accumulating that much angle -- it would then be
    accreted directly / the ballistic approximation breaks down); either
    way, also stops at t_max if neither condition is reached first --
    scaled up automatically when stream_angle_deg asks for more than
    half an orbit around the primary, since the default t_max is sized
    for the original single-pass behavior (a generous heuristic, not a
    real estimate: how long "once around" actually takes depends on how
    deep/eccentric the periapsis passage is, which isn't known in
    advance).

    Prints the trajectory's closest approach to the primary and its
    final radius, every time this runs -- e.g. to help pick a sensible
    --r_acc (the field-line connection radius) by seeing where the
    ballistic trajectory itself actually reaches.

    Returns dict with arrays t, x, y, vx, vy, s (arclength from L1),
    plus the scalar A, theta, x_L1 used to start the integration.
    """
    mu1, mu2, x1, x2 = lobe.mu1, lobe.mu2, lobe.x1, lobe.x2
    A, theta = lubow_shu_angle(lobe)
    vx0 = -eps * np.cos(theta)
    vy0 = eps * np.sin(theta)

    def hit_primary(t, state, *_args):
        return np.hypot(state[0] - x1, state[1]) - r_min_primary
    hit_primary.terminal = True
    hit_primary.direction = -1

    if stream_angle_deg is None:
        state0 = [lobe.x_L1, 0.0, vx0, vy0]
        eom = _eom
        events = [hit_primary]
        t_span_max = t_max
    else:
        state0 = [lobe.x_L1, 0.0, vx0, vy0, 0.0]
        eom = _eom_with_angle
        target_theta = np.radians(stream_angle_deg)

        def hit_angle(t, state, *_args):
            return abs(state[4]) - target_theta
        hit_angle.terminal = True
        hit_angle.direction = 1

        events = [hit_primary, hit_angle]
        t_span_max = max(t_max, t_max * max(1.0, stream_angle_deg / 180.0) * 4.0)

    sol = solve_ivp(eom, [0.0, t_span_max], state0, args=(mu1, mu2, x1, x2),
                     method="DOP853", max_step=max_step,
                     rtol=1e-10, atol=1e-12, dense_output=True,
                     events=events)

    x, y, vx, vy = sol.y[0], sol.y[1], sol.y[2], sol.y[3]
    ds = np.hypot(np.diff(x), np.diff(y))
    s = np.concatenate([[0.0], np.cumsum(ds)])

    r1_all = np.hypot(x - x1, y)
    print(f"stream: closest approach to primary r={r1_all.min():.6g}, "
          f"final r={r1_all[-1]:.6g} (units of a)")

    return {
        "t": sol.t, "x": x, "y": y, "vx": vx, "vy": vy, "s": s,
        "A": A, "theta": theta, "x_L1": lobe.x_L1,
        "hit_primary": len(sol.t_events[0]) > 0,
    }


def disc_impact_index(traj, rim_radius_func, x1):
    """
    Index of the first point along the trajectory that lies inside a disc
    of primary-centered rim radius R(azimuth) = rim_radius_func(nu), with
    nu measured from the +x direction (line towards the secondary).

    rim_radius_func(nu) -> R  [array-safe: nu in radians, returns radius]
    Returns None if the trajectory never enters the disc.
    """
    xs = traj["x"] - x1
    ys = traj["y"]
    r = np.hypot(xs, ys)
    nu = np.arctan2(ys, xs)
    inside = r <= rim_radius_func(nu)
    idx = np.argmax(inside)  # first True, or 0 if none
    if not inside[idx]:
        return None
    return int(idx)


def impact_azimuth(lobe, disc, eps=0.02):
    """
    Azimuth [rad, measured from +x through the primary as usual] where the
    ballistic stream from L1 crosses the disc's outer rim -- the natural,
    geometry-determined location of the impact ("hot spot"). Not a free
    parameter: it is fixed by the mass ratio (via the L1 nozzle and the
    Roche-lobe geometry that shapes the trajectory) together with the
    disc's own shape (r_in, and the rim(nu) it must reach). Returns None
    if the trajectory never reaches the disc.
    """
    traj = integrate_stream(lobe, eps=eps)
    idx = disc_impact_index(traj, disc.rim, lobe.x1)
    if idx is None:
        return None
    return float(np.arctan2(traj["y"][idx], traj["x"][idx] - lobe.x1))


def closest_approach_index(traj, x1):
    """
    Index of the trajectory's first pericenter passage around the primary
    (first local minimum of r1 = distance from the primary), i.e. where
    the free-falling stream stops approaching and starts receding again.
    Beyond a real disc's outer edge, the stream is no longer physical (a
    ballistic test particle would keep going, loop around, and
    self-intersect its earlier path -- see integrate_stream's docstring),
    so this -- not the far end of the integrated trajectory -- is the
    natural place to stop when the disc doesn't intercept it, or when
    tracing the trajectory past the disc rim for illustration.

    Returns None if r1 is monotonically decreasing over the whole
    trajectory (no turnaround found, e.g. t_max cut off too early).
    """
    r1 = np.hypot(traj["x"] - x1, traj["y"])
    dr = np.diff(r1)
    turn = np.nonzero((dr[:-1] < 0.0) & (dr[1:] >= 0.0))[0]
    if len(turn) == 0:
        return None
    return int(turn[0] + 1)


def r_acc_index(traj, x1, r_acc):
    """
    Index of the first point along the (infalling) trajectory where the
    distance from the primary (x1) drops to r_acc or less -- the "magnetic
    field takes over" connection point for a magnetic CV's accretion spot
    (see magnetic.field_line_to_point), the disc-less analogue of
    disc_impact_index.

    Returns None if the trajectory never gets that close, i.e. r_acc is
    smaller than the stream's minimum approach to the primary (see
    closest_approach_index) -- the caller should report this and skip the
    accretion-spot feature rather than silently taking the trajectory's
    last point.
    """
    r1 = np.hypot(traj["x"] - x1, traj["y"])
    inside = r1 <= r_acc
    idx = np.argmax(inside)
    if not inside[idx]:
        return None
    return int(idx)


def sample_points(traj, s_values):
    """Interpolate (x,y) along the trajectory at given arclength values."""
    x = np.interp(s_values, traj["s"], traj["x"])
    y = np.interp(s_values, traj["s"], traj["y"])
    return x, y
