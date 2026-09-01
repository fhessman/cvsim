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
    """
    State is (x, y, vx, vy, theta): the orbital-plane ballistic equations
    of motion, plus a 5th state, the primary-centered azimuth (radians)
    swept since t=0, integrated directly (d(theta)/dt = (dx*vy - dy*vx)/r1^2,
    the standard angular-velocity-about-a-point formula) rather than
    recovered from arctan2(y, x-x1) after the fact, so it accumulates
    past +-pi instead of wrapping. Tracked unconditionally (its cost is
    one extra scalar derivative) so every trajectory integrate_stream
    returns carries angle_deg, whether or not stream_angle_deg is used to
    stop early on it -- see integrate_stream/angle_acc_index.
    """
    x, y, vx, vy = state[0], state[1], state[2], state[3]
    r1 = np.hypot(x - x1, y)
    r2 = np.hypot(x - x2, y)
    ax = 2.0 * vy + x - mu1 * (x - x1) / r1 ** 3 - mu2 * (x - x2) / r2 ** 3
    ay = -2.0 * vx + y - mu1 * y / r1 ** 3 - mu2 * y / r2 ** 3
    dtheta = ((x - x1) * vy - y * vx) / r1 ** 2
    return [vx, vy, ax, ay, dtheta]


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

    The trajectory's cumulative swept azimuth (angle_deg, see the
    returned dict below) is tracked unconditionally regardless of
    stream_angle_deg -- stream_angle_deg only controls whether/where
    integration stops early because of it, not whether the angle itself
    is computed. angle_acc_index looks up a connection point along it the
    same way this function's own stream_angle_deg does.

    Prints the trajectory's closest approach to the primary (and the
    swept angle there) and its final radius/angle, every time this runs
    -- e.g. to help pick a sensible --angle_acc (the field-line
    connection angle) by seeing where the ballistic trajectory itself
    actually reaches.

    Returns dict with arrays t, x, y, vx, vy, s (arclength from L1),
    angle_deg (cumulative signed swept azimuth [deg] from L1, 0=facing
    the secondary, positive/negative by direction of travel -- see
    angle_acc_index), plus the scalar A, theta, x_L1 used to start the
    integration.
    """
    mu1, mu2, x1, x2 = lobe.mu1, lobe.mu2, lobe.x1, lobe.x2
    A, theta = lubow_shu_angle(lobe)
    vx0 = -eps * np.cos(theta)
    vy0 = eps * np.sin(theta)
    state0 = [lobe.x_L1, 0.0, vx0, vy0, 0.0]

    def hit_primary(t, state, *_args):
        return np.hypot(state[0] - x1, state[1]) - r_min_primary
    hit_primary.terminal = True
    hit_primary.direction = -1

    events = [hit_primary]
    t_span_max = t_max
    if stream_angle_deg is not None:
        target_theta = np.radians(stream_angle_deg)

        def hit_angle(t, state, *_args):
            return abs(state[4]) - target_theta
        hit_angle.terminal = True
        hit_angle.direction = 1

        events.append(hit_angle)
        t_span_max = max(t_max, t_max * max(1.0, stream_angle_deg / 180.0) * 4.0)

    sol = solve_ivp(_eom, [0.0, t_span_max], state0, args=(mu1, mu2, x1, x2),
                     method="DOP853", max_step=max_step,
                     rtol=1e-10, atol=1e-12, dense_output=True,
                     events=events)

    x, y, vx, vy, theta_swept = sol.y[0], sol.y[1], sol.y[2], sol.y[3], sol.y[4]
    ds = np.hypot(np.diff(x), np.diff(y))
    s = np.concatenate([[0.0], np.cumsum(ds)])
    angle_deg = np.degrees(theta_swept)

    r1_all = np.hypot(x - x1, y)
    i_min = int(np.argmin(r1_all))
    print(f"stream: closest approach to primary r={r1_all[i_min]:.6g} (units of a) "
          f"at angle={angle_deg[i_min]:.6g} deg, "
          f"final r={r1_all[-1]:.6g} (units of a) at angle={angle_deg[-1]:.6g} deg")

    return {
        "t": sol.t, "x": x, "y": y, "vx": vx, "vy": vy, "s": s, "angle_deg": angle_deg,
        "A": A, "theta": theta, "x_L1": lobe.x_L1,
        "hit_primary": len(sol.t_events[0]) > 0,
    }


def disc_impact_index(traj, rim_radius_func, x1, report=False):
    """
    Index of the first point along the trajectory that lies inside a disc
    of primary-centered rim radius R(azimuth) = rim_radius_func(nu), with
    nu measured from the +x direction (line towards the secondary).

    rim_radius_func(nu) -> R  [array-safe: nu in radians, returns radius]
    Returns None if the trajectory never enters the disc (in particular,
    always None for disc.Disc.is_empty's "null disc" placeholder, since
    its rim sits beyond where the stream ever reaches).

    report: if True and an impact IS found, print the impact radius and
    azimuth [deg, units of a] -- the disc-impact analogue of
    integrate_stream's own closest-approach print, for callers where a
    real disc is expected to matter (e.g. render.build_temperature_maps).
    """
    xs = traj["x"] - x1
    ys = traj["y"]
    r = np.hypot(xs, ys)
    nu = np.arctan2(ys, xs)
    inside = r <= rim_radius_func(nu)
    idx = np.argmax(inside)  # first True, or 0 if none
    if not inside[idx]:
        return None
    idx = int(idx)
    if report:
        print(f"stream: hits disc at r={r[idx]:.6g} (units of a), "
              f"azimuth={np.degrees(nu[idx]):.6g} deg")
    return idx


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


def angle_acc_index(traj, angle_acc_deg):
    """
    Index of the first point along the trajectory where the cumulative
    swept azimuth (traj["angle_deg"], see integrate_stream) reaches
    angle_acc_deg in magnitude -- the "magnetic field takes over"
    connection point for a magnetic CV's accretion spot (see
    magnetic.field_line_to_point), matching integrate_stream's own
    stream_angle_deg convention (0=facing the secondary, 180=directly
    behind the primary, may exceed 360). The disc-less analogue of
    disc_impact_index, and the angle-based replacement for the old
    radius-based r_acc_index.

    Returns None if the trajectory never actually sweeps that far --
    e.g. angle_acc_deg exceeds however far stream_angle_deg let
    integrate_stream run, or the particle plunged into the primary
    (traj["hit_primary"]) before reaching it -- the caller should report
    this and skip the accretion-spot feature rather than silently taking
    the trajectory's last point.

    A small absolute tolerance (1e-6 deg, far below any physically
    meaningful angle_acc granularity) absorbs solve_ivp's own event
    root-finding precision: when integrate_stream was itself given
    stream_angle_deg == angle_acc_deg (the trajectory was extended
    exactly to reach this connection point, see accretion_connection_line/
    build_temperature_maps' auto-extension), the event-terminated
    trajectory's own final angle can land a few ULPs short of the exact
    target (e.g. 94.99999999999996 instead of 95.0) -- without this, an
    exact-strict ">=" would spuriously find no crossing at all.
    """
    inside = np.abs(traj["angle_deg"]) >= angle_acc_deg - 1e-6
    idx = np.argmax(inside)
    if not inside[idx]:
        return None
    return int(idx)


def sample_points(traj, s_values):
    """Interpolate (x,y) along the trajectory at given arclength values."""
    x = np.interp(s_values, traj["s"], traj["x"])
    y = np.interp(s_values, traj["s"], traj["y"])
    return x, y
