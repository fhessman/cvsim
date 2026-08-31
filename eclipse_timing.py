# eclipse_timing.py
"""
Per-component eclipse contact-time (ingress/egress) extraction, and the
multi-cycle diagnostic that relates the accretion stream's eclipse timing
to the projected area of the disc it strikes -- the standard observational
test for an eccentric, precessing accretion disc (cf. the superhump
picture of Whitehurst 1988 and Osaki 1985, 1996): if the disc is circular
and non-precessing, the stream/hot-spot impact point (and hence its
ingress/egress phase) is fixed cycle to cycle; if the disc is eccentric
and precessing, the impact radius and azimuth -- and thus the timing --
vary periodically with the superhump beat phase.
"""

import numpy as np

from eclipse import contact_phases
from disc import Disc, precession_omega
from stream import integrate_stream, disc_impact_index, sample_points


def star_contacts(lobe, incl, R1, half_window=0.15, n_edge=12):
    """
    (first_contact, last_contact) of the central star, found from the
    envelope of contact phases of points around its limb (radius R1).
    Falls back to the point-source contact if R1<=0.
    """
    x1 = lobe.x1
    if R1 <= 0:
        roots = contact_phases(np.array([x1, 0.0, 0.0]), incl, lobe, half_window=half_window)
        return (roots.min(), roots.max()) if len(roots) else (None, None)

    angles = np.linspace(0.0, 2.0 * np.pi, n_edge, endpoint=False)
    firsts, lasts = [], []
    for a in angles:
        pt = np.array([x1 + R1 * np.cos(a), R1 * np.sin(a), 0.0])
        roots = contact_phases(pt, incl, lobe, half_window=half_window)
        if len(roots):
            firsts.append(roots.min())
            lasts.append(roots.max())
    if not firsts:
        return None, None
    return min(firsts), max(lasts)


def disc_contacts(lobe, incl, disc, half_window=0.2, n_az=36):
    """
    Per-azimuth ingress/egress of the disc rim, plus the overall first/last
    contact.  Returns dict: nu (n_az,), ingress (n_az,), egress (n_az,)
    [nan where that rim point is never eclipsed], first_contact, last_contact.
    """
    nu = np.linspace(0.0, 2.0 * np.pi, n_az, endpoint=False)
    R = disc.rim(nu)
    x = disc.x1 + R * np.cos(nu)
    y = R * np.sin(nu)

    ingress = np.full(n_az, np.nan)
    egress = np.full(n_az, np.nan)
    for i in range(n_az):
        roots = contact_phases(np.array([x[i], y[i], 0.0]), incl, lobe, half_window=half_window)
        if len(roots):
            ingress[i] = roots.min()
            egress[i] = roots.max()

    valid = ~np.isnan(ingress)
    first_contact = np.nanmin(ingress) if valid.any() else None
    last_contact = np.nanmax(egress) if valid.any() else None
    return {"nu": nu, "ingress": ingress, "egress": egress,
            "first_contact": first_contact, "last_contact": last_contact}


def stream_contacts(lobe, incl, disc, eps=0.02, half_window=0.25, n_points=20):
    """
    Ingress/egress phase of `n_points` points sampled along the ballistic
    stream from L1 up to the disc-impact point, plus the impact point's
    own (radius, azimuth) on the disc rim -- this is the array to compare,
    cycle by cycle, against the disc's assumed shape to test for
    ellipticity/precession (see multi_cycle_diagnostic).

    Returns dict: s (n_points,) arclength from L1, x,y (corotating frame),
    ingress, egress (n_points,), impact_radius, impact_nu, impact_ingress,
    impact_egress (the last, disc-impact point specifically -- the
    "hot spot", usually the most observationally accessible feature).
    """
    x1 = lobe.x1
    traj = integrate_stream(lobe, eps=eps)
    idx = disc_impact_index(traj, disc.rim, x1)
    if idx is None or traj["s"][idx] <= 0:
        return None

    s_impact = traj["s"][idx]
    s = np.linspace(0.0, s_impact, n_points)
    x, y = sample_points(traj, s)

    ingress = np.full(n_points, np.nan)
    egress = np.full(n_points, np.nan)
    for i in range(n_points):
        roots = contact_phases(np.array([x[i], y[i], 0.0]), incl, lobe, half_window=half_window)
        if len(roots):
            ingress[i] = roots.min()
            egress[i] = roots.max()

    impact_r = np.hypot(x[-1] - x1, y[-1])
    impact_nu = np.arctan2(y[-1], x[-1] - x1)
    return {"s": s, "x": x, "y": y, "ingress": ingress, "egress": egress,
            "impact_radius": impact_r, "impact_nu": impact_nu,
            "impact_ingress": ingress[-1], "impact_egress": egress[-1]}


def hotspot_contact(lobe, incl, disc, eps=0.02, half_window=0.25):
    """
    Cheap single-point version of stream_contacts: only the disc-impact
    ("hot spot") point's ingress/egress and (radius,azimuth), without the
    full per-point stream profile. Used for the (potentially many-cycle)
    diagnostic sweep, where the hot-spot timing is the primary observable.
    """
    x1 = lobe.x1
    traj = integrate_stream(lobe, eps=eps)
    idx = disc_impact_index(traj, disc.rim, x1)
    if idx is None or traj["s"][idx] <= 0:
        return None
    x, y = traj["x"][idx], traj["y"][idx]
    roots = contact_phases(np.array([x, y, 0.0]), incl, lobe, half_window=half_window)
    if not len(roots):
        return None
    r = np.hypot(x - x1, y)
    nu = np.arctan2(y, x - x1)
    return {"impact_radius": r, "impact_nu": nu,
            "ingress": roots.min(), "egress": roots.max()}


def multi_cycle_diagnostic(lobe, incl_deg, P_orb, a_disc, e_disc, omega0, P_sh,
                            r_in=0.02, p_disc=0.75, eps=0.02,
                            n_cycles=60, cycle0=0, sense=-1.0,
                            n_az=24, half_window=0.25):
    """
    Simulate `n_cycles` consecutive orbital cycles of an eccentric,
    precessing disc (semi-major axis a_disc, eccentricity e_disc,
    precessing with beat period P_sh -- see disc.precession_omega) and
    extract, each cycle: the disc's first/last contact, and the stream's
    disc-impact ("hot spot") ingress/egress and impact (radius,azimuth).

    This is the core diagnostic: if e_disc=0 the impact geometry -- and
    hence the hot-spot ingress/egress phase -- is identical every cycle;
    for e_disc>0 it is modulated with the superhump phase
    (cycle*P_orb mod P_sh)/P_sh, tracing out the disc-rim shape as the
    ellipse precesses under the fixed stream trajectory.

    Returns a dict of per-cycle arrays: cycle, superhump_phase,
    disc_first_contact, disc_last_contact, impact_radius, impact_nu,
    hotspot_ingress, hotspot_egress.
    """
    incl = np.radians(incl_deg)
    cycles = cycle0 + np.arange(n_cycles)
    t = cycles * P_orb
    sh_phase = np.mod(t, P_sh) / P_sh

    out = {k: np.full(n_cycles, np.nan) for k in
           ["disc_first_contact", "disc_last_contact", "impact_radius",
            "impact_nu", "hotspot_ingress", "hotspot_egress"]}

    for k in range(n_cycles):
        omega = precession_omega(omega0, t[k], P_sh, sense=sense)
        disc = Disc(lobe.x1, a_disc, e_disc, omega, r_in=r_in, brightness_index=p_disc)

        dc = disc_contacts(lobe, incl, disc, half_window=half_window, n_az=n_az)
        out["disc_first_contact"][k] = dc["first_contact"] if dc["first_contact"] is not None else np.nan
        out["disc_last_contact"][k] = dc["last_contact"] if dc["last_contact"] is not None else np.nan

        hs = hotspot_contact(lobe, incl, disc, eps=eps, half_window=half_window)
        if hs is not None:
            out["impact_radius"][k] = hs["impact_radius"]
            out["impact_nu"][k] = hs["impact_nu"]
            out["hotspot_ingress"][k] = hs["ingress"]
            out["hotspot_egress"][k] = hs["egress"]

    out["cycle"] = cycles
    out["superhump_phase"] = sh_phase
    return out
