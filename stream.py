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
    than once. None (the default) instead stops at the trajectory's own
    FIRST turnaround -- the first local minimum of r1 (distance from the
    primary), d(r1)/dt crossing from negative to positive, typically
    around half an orbit (~180 deg) after leaving L1 -- rather than
    continuing on regardless (which would otherwise, for many systems,
    loop around the primary several more times before r_min_primary or
    t_max finally stopped it, none of which is the physically relevant
    part of the trajectory: a free-falling test particle isn't a real
    accretion stream once it's past its own first pericenter passage).
    r_min_primary remains an active safety floor either way (the
    particle plunging into the primary before ever turning around, or
    before reaching stream_angle_deg -- it would then be accreted
    directly / the ballistic approximation breaks down); either way,
    also stops at t_max if neither condition is reached first -- scaled
    up automatically when stream_angle_deg asks for more than half an
    orbit around the primary, since the default t_max is sized for the
    original single-pass behavior (a generous heuristic, not a real
    estimate: how long "once around" actually takes depends on how
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
    if stream_angle_deg is None:
        # d(r1)/dt = ((x-x1)*vx + y*vy) / r1 -- negative while
        # approaching the primary, crossing zero (direction=1: from
        # negative to positive) exactly at the first local minimum of
        # r1, i.e. the trajectory's own first pericenter passage. Not
        # added when stream_angle_deg is given -- that's an explicit
        # request to let the trajectory continue past this point
        # (possibly around the primary more than once), which this
        # event would otherwise cut short right away.
        def turnaround(t, state, *_args):
            x, y, vx, vy = state[0], state[1], state[2], state[3]
            r1 = np.hypot(x - x1, y)
            return ((x - x1) * vx + y * vy) / max(r1, 1e-300)
        turnaround.terminal = True
        turnaround.direction = 1
        events.append(turnaround)
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
    # angle_deg itself is the cumulative (unwrapped) swept azimuth (see
    # this function's own docstring) -- genuinely past 360 deg whenever
    # the ballistic trajectory loops around the primary more than once
    # before settling into its closest approach (common for a disc-
    # accreting system, since integrate_stream is deliberately NOT
    # stopped by a disc -- "the stream is not stopped by a disc by
    # default", see the README). The bare cumulative value alone reads
    # as a bug at a glance, so also report the equivalent [0,360) azimuth
    # (same convention as disc_impact_index's own nu) alongside it.
    print(f"stream: closest approach to primary r={r1_all[i_min]:.6g} (units of a) "
          f"at angle={angle_deg[i_min]:.6g} deg ({angle_deg[i_min] % 360.0:.6g} deg mod 360), "
          f"final r={r1_all[-1]:.6g} (units of a) "
          f"at angle={angle_deg[-1]:.6g} deg ({angle_deg[-1] % 360.0:.6g} deg mod 360)")

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


def disc_impact_point(lobe, disc, eps=0.02):
    """
    The richer version of impact_azimuth: the full state (x, y, vx, vy,
    r1) of the ballistic stream at the point it first enters the disc's
    rim, not just its azimuth -- vx,vy give the stream's own local
    direction of travel there (needed to orient e.g. a cross-section
    ellipse perpendicular to it, see plots.stream_impact_ellipse_outline),
    r1 its distance from the primary (units of a, this module's
    convention -- the argument lubow_shu_stream_size's own
    h1(r1)/w1(r1) fits expect). Returns None in the same cases
    impact_azimuth/disc_impact_index do (trajectory never reaches this
    disc).

    disc_impact_index only ever returns the first discrete trajectory
    sample already inside the rim, which can overshoot the true r ==
    disc.rim(nu) crossing by a full integration step -- for a coarsely
    sampled trajectory that overshoot can exceed the stream's own
    cross-sectional H/W (Hessman 1999's Lubow & Shu fits), burying a
    cross-section ellipse drawn around the raw sample entirely inside the
    disc's own solid and making it permanently self-occluded. So refine:
    linearly interpolate x,y,vx,vy between that sample and the one just
    before it (still outside the rim, by disc_impact_index's own "first
    inside" definition) to the sub-step point where r(t) == rim(nu(t)),
    treating both r-rim(nu) and the state itself as linear over the one
    step in between -- exact for a straight sub-step, and a good
    approximation otherwise since the step is already the trajectory's
    own finest resolution there.
    """
    traj = integrate_stream(lobe, eps=eps)
    idx = disc_impact_index(traj, disc.rim, lobe.x1)
    if idx is None:
        return None
    x1 = lobe.x1
    x, y = float(traj["x"][idx]), float(traj["y"][idx])
    vx, vy = float(traj["vx"][idx]), float(traj["vy"][idx])
    if idx > 0:
        x0, y0 = float(traj["x"][idx - 1]), float(traj["y"][idx - 1])
        vx0, vy0 = float(traj["vx"][idx - 1]), float(traj["vy"][idx - 1])
        f0 = np.hypot(x0 - x1, y0) - float(disc.rim(np.arctan2(y0, x0 - x1)))
        f1 = np.hypot(x - x1, y) - float(disc.rim(np.arctan2(y, x - x1)))
        if f0 >= 0.0 and f1 <= 0.0 and (f0 - f1) > 0.0:
            t = f0 / (f0 - f1)
            x, y = x0 + t * (x - x0), y0 + t * (y - y0)
            vx, vy = vx0 + t * (vx - vx0), vy0 + t * (vy - vy0)
    return {"x": x, "y": y, "vx": vx, "vy": vy, "r1": float(np.hypot(x - x1, y))}


def impact_incidence(impact, disc):
    """
    (nu_imp, cos_incidence) at a disc_impact_point() result -- shared by
    every caller that needs the stream's own incidence angle there
    (plots.plot_disc_rim_projection's ellipse-width stretch, simulate.py's
    rim-projection-mode fit target, simulate.py's own impact-point
    diagnostic print). nu_imp [rad] is the impact azimuth, from +x
    through the primary; cos_incidence is |cos| of the angle between the
    stream's own velocity there and the disc rim's own LOCAL OUTWARD
    NORMAL at that azimuth -- 1.0 for a perfectly radial ("head-on")
    impact, 0.0 for a perfectly tangential ("grazing") one. Clamped away
    from exactly 0 (a genuinely tangential impact would otherwise blow up
    any 1/cos_incidence use, e.g. the ellipse stretch above) at 1e-3.

    The rim's local outward normal is only the pure radial direction
    r_hat=(cos nu, sin nu) when the rim itself is circular (disc.e==0):
    for an eccentric rim R(nu)=disc.rim(nu), the curve's own tangent has
    a radial component too (proportional to dR/dnu, its "pitch"), zero
    only exactly at periastron/apastron. Writing r_hat/t_hat for the
    ordinary radial/azimuthal unit vectors at nu_imp, the curve's own
    local tangent/normal are instead
        T_hat = (R'*r_hat + R*t_hat) / sqrt(R^2+R'^2)
        N_hat = (R*r_hat - R'*t_hat) / sqrt(R^2+R'^2)
    (T_hat.N_hat=0 by construction; both reduce to t_hat/r_hat exactly
    when R'=0) -- R' from a small central-difference step of disc.rim
    itself, so this holds for whatever rim shape disc.rim implements, not
    just the eccentric-ellipse formula it currently is.
    """
    nu_imp = np.arctan2(impact["y"], impact["x"] - disc.x1)
    r_hat = np.array([np.cos(nu_imp), np.sin(nu_imp)])
    t_hat = np.array([-np.sin(nu_imp), np.cos(nu_imp)])
    dnu = 1e-6
    R = float(disc.rim(nu_imp))
    Rprime = float(disc.rim(nu_imp + dnu) - disc.rim(nu_imp - dnu)) / (2.0 * dnu)
    n_hat_imp = (R * r_hat - Rprime * t_hat) / np.hypot(R, Rprime)
    v_imp = np.array([impact["vx"], impact["vy"]])
    v_imp = v_imp / np.hypot(*v_imp)
    cos_incidence = max(abs(np.dot(v_imp, n_hat_imp)), 1e-3)
    return nu_imp, cos_incidence


def lubow_shu_eps(T_2, P_orb_d, a_m):
    """
    The dimensionless stream sound speed eps = c_s(T_2)/(Omega*a) -- this
    module's own `eps` parameter above (integrate_stream's initial-
    velocity scale) -- computed from real system parameters via Hessman
    (1999)'s own fit (that paper's Eq. 1), rather than left at the
    assumed-constant default (0.02) integrate_stream otherwise uses:

        eps = 0.013 * sqrt(T_2/4000 K) * (P_orb/4 h) / (a/1e11 cm)

    T_2 [K], P_orb_d [d] (converted to hours here), a_m [m] (converted to
    cm here) -- SystemParams' own native units (see params.py's own
    docstring), so callers can pass system.T_2/system.P_orb/system.a_m
    directly (system.a itself is in Rsun, not meters -- a_m is its SI
    conversion).
    """
    P_orb_h = P_orb_d * 24.0
    a_cm = a_m * 100.0
    return 0.013 * np.sqrt(T_2 / 4000.0) * (P_orb_h / 4.0) / (a_cm / 1.0e11)


def lubow_shu_stream_size(r1, q, eps):
    """
    The ballistic stream's own transverse size at a point r1/a from the
    primary (r1 already in units of a, this module's convention -- see
    e.g. disc_impact_point's own "r1"), via Hessman (1999)'s fits (that
    paper's Fig. 1 and Eq. 3-4) to the Lubow & Shu (1975) stream
    hydrodynamics -- separable in r1/a and mass ratio q:

        H(r1,q) = h1(r1)*h2(q)*a*eps   (vertical, perpendicular to the
                                         orbital plane)
        W(r1,q) = w1(r1)*w2(q)*a*eps   (horizontal, in the orbital plane,
                                         transverse to the stream's own
                                         direction of travel)

    with (Hessman 1999's own Eq. 4, base-10 log):
        h1(r1) = 0.060 + 3.17*r1 - 2.90*r1^2
        w1(r1) = 0.084 + 3.09*r1 - 3.08*r1^2
        h2(q)  = 10**(0.031*log10(q) + 0.095*log10(q)**2)
        w2(q)  = 10**(-0.021*log10(q) + 0.087*log10(q)**2)

    r1 already in units of a makes the "*a" above implicit -- H and W
    are returned directly in units of a, this codebase's own convention
    for every other geometric quantity. Both are SCALEHEIGHTS -- already
    a one-sided, half-width/half-height quantity, appropriate directly
    as an ellipse's own semi-axis (see plots.stream_impact_ellipse_outline)
    or as a single-sided offset from the stream's own centerline (see
    plot_topdown_shadows' T_2/P_orb/a_m handling) -- NOT half of that;
    the stream's own FULL transverse extent is 2*H, 2*W.

    r1/q accept arrays; returns (H, W), each the same shape as r1.
    """
    h1 = 0.060 + 3.17 * r1 - 2.90 * r1 ** 2
    w1 = 0.084 + 3.09 * r1 - 3.08 * r1 ** 2
    logq = np.log10(q)
    h2 = 10.0 ** (0.031 * logq + 0.095 * logq ** 2)
    w2 = 10.0 ** (-0.021 * logq + 0.087 * logq ** 2)
    return h1 * h2 * eps, w1 * w2 * eps


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


def near_primary_flank_distance(lobe, T_2, P_orb_d, a_m, point, s_max=None, n=600):
    """
    Minimum distance from `point` (x,y) [corotating frame] to the
    ballistic stream's own NEAR-PRIMARY Lubow & Shu flanking line (see
    lubow_shu_stream_size's own half-width W, and plots.
    plot_topdown_shadows' matching +-W overlay): of the two +-W offset
    lines flanking the trajectory's centerline, "near-primary" means
    whichever one sits at the smaller distance from the primary AT THE
    POINT ALONG THE CENTERLINE closest to `point` -- the physically
    meaningful edge for simulate.py's shadow-mode --lsq_fit/--mcmc_fit
    (see its own docstring): the eclipse-timing-derived shadow crossing
    this measures against should sit on the stream's own near edge, not
    its far one.

    Approximated via n dense, evenly-arclength-spaced samples along the
    trajectory (not a further local refinement beyond that) -- called
    many times per fit trial, so this stays a single vectorized pass;
    finer sampling (raise n) rather than a slower local polish is the
    lever available if a fit needs more precision than this gives.

    s_max: truncate the trajectory's own arclength there (None: its
    first closest approach to the primary, the same default
    plot_topdown_shadows/plot_component_outlines use).

    Returns the scalar distance (units of a).
    """
    traj = integrate_stream(lobe)
    if s_max is None:
        idx = closest_approach_index(traj, lobe.x1)
        if idx is None:
            idx = len(traj["x"]) - 1
        s_max = traj["s"][idx]
    s_vals = np.linspace(0.0, s_max, n)
    xc, yc = sample_points(traj, s_vals)
    px, py = point

    tangent = np.gradient(np.stack([xc, yc], axis=-1), axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=-1, keepdims=True), 1e-300)
    perp = np.stack([-tangent[:, 1], tangent[:, 0]], axis=-1)
    eps = lubow_shu_eps(T_2, P_orb_d, a_m)
    r1 = np.hypot(xc - lobe.x1, yc)
    _, half = lubow_shu_stream_size(r1, lobe.q, eps)

    i0 = int(np.argmin(np.hypot(xc - px, yc - py)))
    r1_plus = np.hypot(xc[i0] + half[i0] * perp[i0, 0] - lobe.x1,
                        yc[i0] + half[i0] * perp[i0, 1])
    r1_minus = np.hypot(xc[i0] - half[i0] * perp[i0, 0] - lobe.x1,
                         yc[i0] - half[i0] * perp[i0, 1])
    sign = 1.0 if r1_plus < r1_minus else -1.0

    xf = xc + sign * half * perp[:, 0]
    yf = yc + sign * half * perp[:, 1]
    return float(np.min(np.hypot(xf - px, yf - py)))
