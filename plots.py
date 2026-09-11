# plots.py
"""
Reusable sky-projected outline plots of the binary components (secondary
Roche lobe, primary, disc, stream), built on the same projection used for
the eclipse geometry (eclipse.project/observer_frame).
"""

import numpy as np
from scipy.spatial import ConvexHull
from matplotlib.path import Path

from eclipse import project, observer_frame, visible_mask, visible_mask_bulk, _L1_DIR
from roche import gravity, RocheLobe
from stream import (closest_approach_index, integrate_stream, disc_impact_index, sample_points,
                     lubow_shu_eps, lubow_shu_stream_size, impact_incidence)
from lightcurve import star_points


def tighten_external_legend(fig, ax, margin_in=0.12):
    """
    Re-anchor `ax`'s legend (created via ax.legend(bbox_to_anchor=(x, negative
    axes-fraction), ...) -- see plot_component_outlines/plot_topdown_shadows'
    own below-the-axes legends) a small, FIXED distance (in real figure
    inches, not axes-fraction) below the axes' actual rendered extent
    (tick labels/xlabel included). Call once, after the figure's own
    final layout pass (simulate.py's finish(), right before savefig) --
    ax.set_aspect("equal") can shrink the axes' own fractional height a
    lot on a wide canvas (e.g. simulate.py's default 1280x720
    --image-size), and a bbox_to_anchor offset expressed in axes-fraction
    (tuned for a squarer figure) then translates to a much larger -- or,
    for a more squashed axes, potentially too small/clipped -- absolute
    gap than intended. No-op if `ax` has no legend.
    """
    leg = ax.get_legend()
    if leg is None:
        return
    # get_tightbbox() includes the legend itself (a child artist of ax) at
    # wherever it's currently anchored -- which, before this call, is
    # exactly the badly-offset position this function exists to fix, so
    # measuring with it still attached would just reproduce the same gap.
    # Hide it for the measurement, then restore it at the freshly computed
    # position.
    leg.set_visible(False)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tight = ax.get_tightbbox(renderer)
    inv = fig.transFigure.inverted()
    (x0, y0), (x1, _y1) = inv.transform([(tight.x0, tight.y0), (tight.x1, tight.y1)])
    margin = margin_in / fig.get_size_inches()[1]
    leg.set_visible(True)
    leg.set_bbox_to_anchor(((x0 + x1) / 2.0, y0 - margin), transform=fig.transFigure)


def fit_content_to_canvas(fig, pad_frac=0.008):
    """
    Nudge the figure's subplot margins so nothing -- title, legend, axis
    labels -- overflows the canvas edge, without changing the canvas's
    own pixel size (simulate.py's finish() deliberately does NOT use
    savefig(bbox_inches="tight"): every frame of a --phase-num>1 sequence
    needs to share the same --image-size dimensions for a movie or any
    other automated/batch use, not each get cropped to its own content).

    fig.tight_layout()'s own margins are picked before matplotlib
    finalizes anything whose actual rendered size/position depends on
    the draw itself -- ax.set_aspect("equal") shrinking/repositioning the
    axes box (see tighten_external_legend's own docstring for the same
    issue on the legend side), or a title long enough to wrap -- so with
    a fixed canvas the title or legend can end up genuinely clipped
    against the top/bottom edge instead of just oddly spaced, at
    whichever phases/content happen to push the real layout past what
    tight_layout guessed. Call once everything else (including
    tighten_external_legend, if used) is already in its final position.

    Measures the whole figure's actual rendered tight bbox and, on
    whichever side(s) it overflows the canvas, shrinks the axes box by
    exactly that much (plus a small pad) -- a no-op if nothing overflows.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Figure.get_tightbbox (unlike Axes.get_tightbbox, used elsewhere in
    # this module) returns its Bbox in figure INCHES, not display pixels.
    tight = fig.get_tightbbox(renderer)
    fig_w, fig_h = fig.get_size_inches()
    sp = fig.subplotpars
    top, bottom, left, right = sp.top, sp.bottom, sp.left, sp.right
    over_top = tight.y1 - fig_h
    if over_top > 0.0:
        top -= over_top / fig_h + pad_frac
    over_bottom = -tight.y0
    if over_bottom > 0.0:
        bottom += over_bottom / fig_h + pad_frac
    over_right = tight.x1 - fig_w
    if over_right > 0.0:
        right -= over_right / fig_w + pad_frac
    over_left = -tight.x0
    if over_left > 0.0:
        left += over_left / fig_w + pad_frac
    if (top, bottom, left, right) != (sp.top, sp.bottom, sp.left, sp.right):
        fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right)


def _ensure_ccw(X, Y):
    """
    Reverse a closed 2D loop (X,Y) if its signed area is negative
    (clockwise), so the returned loop is always counterclockwise.

    Needed before filling an annulus as one path via "outer forward +
    inner reversed" (matplotlib leaves the enclosed hole unfilled only
    when the two loops wind in OPPOSITE senses) -- reliable only if both
    loops start from a known, consistent orientation. That's not a given
    here: the outer loop can be either a plain parametric sweep
    (disc_outline's flat-disc rim, angle increasing -- whose screen-space
    winding after projection depends on phase/inclination) or a
    scipy.spatial.ConvexHull silhouette (the flared-disc case), which
    scipy always returns counterclockwise in whatever 2D coordinates it's
    given, regardless of the original points' order -- so the two loops'
    relative winding can flip from one phase/case to another if left
    alone. Forcing both to this same canonical (CCW) orientation first,
    then always reversing just the inner one, makes the hole subtraction
    correct unconditionally.
    """
    area = np.sum(X[:-1] * Y[1:] - X[1:] * Y[:-1])
    if area < 0.0:
        return X[::-1], Y[::-1]
    return X, Y


def labeled_title(label, title=""):
    """
    Prefix a plot's own title with simulate.py's --prefix global label
    (see its own --help), single-line ("LABEL : title") -- shared by every
    plotting function (here and in simulate.py) that sets its own
    ax.set_title, so the two always combine the same way. `title` may be
    omitted for a plot with no title of its own beyond the label itself
    (e.g. simulate.py's lightcurve/magnitude/rv outputs); label may
    likewise be empty/None (--prefix not given) -- either combination
    degrades gracefully to whichever of the two is actually non-empty, or
    "" (a harmless no-op title) if neither is.
    """
    if label and title:
        return f"{label} : {title}"
    return label or title


def style_axes(ax, right=True):
    """
    Standard tick styling applied to every axes this package plots on:
    minor ticks enabled, and both major and minor ticks drawn on all four
    sides -- matplotlib's own default only shows major ticks along the
    labeled bottom/left edges, and no minor ticks at all. Called once per
    axes, right before returning it, regardless of whether the caller
    passed in an existing `ax` or a new figure was created here.

    right: pass False when the caller adds its own secondary_yaxis on the
    right edge (see simulate.py's lightcurve/magnitude plots) -- its own
    ticks/label would otherwise compete with this axis's plain tick marks
    on the same spine.
    """
    ax.minorticks_on()
    ax.tick_params(which="both", top=True, right=right)


_DASH_PATTERNS = {"--": (4, 2), ":": (1, 2)}  # absolute (points), see docstring below


def _hidden_split_plot(ax, X, Y, visible, color, lw=1.0, alpha=1.0, label=None, ls="-"):
    """
    Draw a curve that may pass behind solid material: the hidden portions
    (visible=False) as an even-fainter continuous guide (so the curve's
    full path stays legible), the visible portions at the caller's given
    lw/alpha on top, using NaN gaps so matplotlib skips the hidden
    stretches in that overlay rather than needing explicit run-splitting.

    The two passes are drawn at different linewidths (lw*0.5 vs lw), so
    for a dashed/dotted `ls` (see _DASH_PATTERNS) an explicit, absolute
    (points, not lw-scaled) dash pattern is used instead of the bare
    "--"/":" string -- matplotlib's named linestyles scale their dash
    length with linewidth, so the same style at two different linewidths
    otherwise renders with visibly different (finer vs. coarser) dashing
    instead of matching up.
    """
    dash_kwargs = {"dashes": _DASH_PATTERNS[ls]} if ls in _DASH_PATTERNS else {"linestyle": "-"}
    ax.plot(X, Y, color=color, lw=lw * 0.5, alpha=alpha * 0.3, **dash_kwargs)
    Xv = np.where(visible, X, np.nan)
    Yv = np.where(visible, Y, np.nan)
    ax.plot(Xv, Yv, color=color, lw=lw, alpha=alpha, label=label, **dash_kwargs)


def lobe_outline(lobe, phase, incl):
    """
    Projected silhouette of the secondary's Roche lobe, via the convex
    hull of the projected surface mesh. The Roche lobe is star-convex
    about the secondary's center and its projected silhouette is, to
    excellent approximation, itself convex for any viewing direction, so
    this gives an accurate, cheap limb curve (as opposed to scatter-
    plotting the whole surface mesh). Returns closed (X,Y) arrays.
    """
    surf = lobe.surface_points()
    X, Y, _ = project(surf, phase, incl)
    pts = np.column_stack([X, Y])
    hull = ConvexHull(pts)
    idx = np.append(hull.vertices, hull.vertices[0])
    return pts[idx, 0], pts[idx, 1]


def circle_outline(center_xyz, radius, phase, incl, n=100):
    """
    Projected outline (limb) of a sphere of given `radius` centered at
    `center_xyz` (corotating frame). These points sit on the sphere's true
    3D surface (not a flat facet) -- at the limb the two coincide exactly
    -- so they're valid input to disc.Disc.visible_from for occultation
    testing, unlike an interior facet point (see lightcurve.star_points).
    """
    n_hat, eX, eY = observer_frame(phase, incl)
    ang = np.linspace(0.0, 2.0 * np.pi, n)
    pts = (np.asarray(center_xyz)
           + radius * np.cos(ang)[:, None] * eX
           + radius * np.sin(ang)[:, None] * eY)
    X = np.dot(pts, eX)
    Y = np.dot(pts, eY)
    return pts, X, Y


def primary_visibility_grid(center_xyz, radius, phase, incl, disc=None, lobe=None, n=140):
    """
    Visibility of the primary's projected disc as a fine (X,Y) grid of
    booleans, general enough to handle any disc geometry -- flat with
    finite r_in/r_out, flared (opening_angle>0), or a disc whose r_in is
    large enough that it never reaches the primary at all (in which case
    this correctly comes out all-True, i.e. a plain, unoccluded circle).

    This replaces an earlier version that used a closed-form ellipse
    (valid only for an idealized disc extending across the whole
    occultable region, ignoring r_in/r_out) -- that shortcut no longer
    generalizes once the disc has finite extent or is flared, so this
    instead tests each grid point's true 3D position directly against
    disc.Disc.visible (which itself dispatches flat-plane vs 3D-solid
    occlusion) and, if given, the secondary's Roche lobe.

    Returns (X, Y, visible) as (n,n) arrays; feed to ax.contour/contourf
    at level 0.5 for an outline/fill that handles disconnected or
    partially-occulted regions -- and the plain circle case -- uniformly.
    """
    n_hat, eX, eY = observer_frame(phase, incl)
    lin = np.linspace(-radius, radius, n)
    Xrel, Yrel = np.meshgrid(lin, lin, indexing="xy")
    R2 = Xrel ** 2 + Yrel ** 2
    inside_circle = R2 <= radius ** 2
    depth = np.sqrt(np.clip(radius ** 2 - R2, 0.0, None))
    pts = (np.asarray(center_xyz)
           + Xrel[..., None] * eX + Yrel[..., None] * eY + depth[..., None] * n_hat)

    # absolute sky-plane coordinates (Xrel/Yrel above are center-relative)
    Xabs = pts @ eX
    Yabs = pts @ eY

    vis = inside_circle.copy()
    if disc is not None:
        vis &= disc.visible(pts, phase, incl)
    if lobe is not None:
        vis &= visible_mask_bulk(pts, phase, incl, lobe)
    return Xabs, Yabs, vis & inside_circle


def disc_outline(disc, phase, incl, lobe=None, n=200, primary_center=None, R1=None):
    """
    Projected outline of the disc rim. For the flat (opening_angle=0)
    disc this is just the projected rim curve at z=0 (with `visible`
    marking any part occulted by the Roche lobe, if `lobe` given, or the
    primary star's sphere, if `primary_center`/`R1` given, for
    _hidden_split_plot). For a flared disc it's the convex-hull silhouette
    of all four surfaces' sample points (same technique as lobe_outline,
    and similarly a good approximation since the disc solid is close to
    convex in projection for realistic opening angles) -- a silhouette is
    "visible everywhere" by construction, so `visible` is all-True there;
    a single 2D curve can't show the cone/edge structure itself anyway,
    see disc_cross_section_lines for that.
    """
    if disc.opening_angle <= 0.0:
        x, y = disc.rim_xy(n=n)
        pts = np.stack([x, y, np.zeros_like(x)], axis=-1)
        vis = visible_mask_bulk(pts, phase, incl, lobe) if lobe is not None else np.ones(n, dtype=bool)
        if primary_center is not None and R1 is not None:
            n_hat, _, _ = observer_frame(phase, incl)
            vis = vis & ~_occluded_by_sphere(pts, primary_center, R1, n_hat)
        X, Y, _ = project(pts, phase, incl)
        return X, Y, vis
    n_r = max(n // 4, 10)
    xs, ys, zs, _normals, _areas, _r, _sid = disc.all_surfaces_grid(n_r=n_r, n_nu=n, n_z=8)
    pts = np.stack([xs, ys, zs], axis=-1)
    X, Y, _ = project(pts, phase, incl)
    hull = ConvexHull(np.column_stack([X, Y]))
    idx = np.append(hull.vertices, hull.vertices[0])
    return X[idx], Y[idx], np.ones(len(idx), dtype=bool)


def disc_inner_wall_outline(disc, phase, incl, n=200):
    """
    Projected silhouette of the disc's own INNER edge wall alone
    (all_surfaces_grid's surface_id==3), by the identical convex-hull
    technique disc_outline uses for the WHOLE disc -- the correct
    boundary of the disc's actual "hole" (the region with no disc
    material at all, inside the inner wall), for a flared disc
    (opening_angle>0) properly spanning that wall's own vertical extent
    (z=-h_in..+h_in, h_in=r_in*tan(opening_angle)), not just its flat
    midplane circle. plot_component_outlines' own fill uses this to cut
    that hole out of the disc's fill; disc_inner_rim_line's plain flat
    circle remains exactly what it already was (a separate, deliberately
    simple faint reference LINE, documented as flat regardless of
    opening_angle) -- unrelated, unaffected by this.

    For a flat disc (opening_angle<=0, where inner_edge_grid returns no
    points at all -- there's no wall to sample) this instead returns the
    same flat midplane circle disc_inner_rim_line draws (the inner wall
    truly has no vertical extent then, so the two coincide exactly).
    """
    if disc.opening_angle <= 0.0:
        ang = np.linspace(0.0, 2.0 * np.pi, n)
        x = disc.x1 + disc.r_in * np.cos(ang)
        y = disc.r_in * np.sin(ang)
        pts = np.stack([x, y, np.zeros(n)], axis=-1)
        X, Y, _ = project(pts, phase, incl)
        return X, Y
    n_r = max(n // 4, 10)
    xs, ys, zs, _normals, _areas, _r, sid = disc.all_surfaces_grid(n_r=n_r, n_nu=n, n_z=8)
    mask = sid == 3
    pts = np.stack([xs[mask], ys[mask], zs[mask]], axis=-1)
    X, Y, _ = project(pts, phase, incl)
    hull = ConvexHull(np.column_stack([X, Y]))
    idx = np.append(hull.vertices, hull.vertices[0])
    return X[idx], Y[idx]


def disc_hotspot_cells(disc, phi_h, L_h_deg, dphi_max_deg, phase, incl, n_z=8, n_nu=300,
                        footprint=None, flat_band_frac=0.08, lobe=None, primary_center=None, R1=None):
    """
    The disc's own stream-impact hot spot, drawn as the actual simulated
    grid cells themselves -- each cell's own true projected footprint
    (area), colored by its own local intensity (T_h*exp(-dphi/L_h)/T_h,
    or flat 1.0 inside `footprint` -- see below) -- "what you see is
    what's simulated" -- rather than a small number of coarse,
    illustrative wedges (the earlier disc_hotspot_wedges this replaces).
    Occlusion is tested per CELL (a few hundred, at this function's own
    default resolution) instead of per (much larger) wedge, so the
    visible boundary tracks the true occluding silhouette (the
    secondary's limb, the disc's own far side, the primary) to within
    one cell's own size rather than one wedge's -- the old "drop the
    WHOLE wedge if ANY of its corners is hidden" rule could erase up to
    one full wedge's worth of genuinely visible area right at the
    terminator, coarse enough (at only ~60 wedges) to look like the hot
    spot's own visible edge jumping away from its true position as an
    occulting limb swept across it, rather than smoothly eroding from
    the correct edge.

    Each cell spans an angular slice of azimuth and, for a flared disc
    (opening_angle>0), a slice of the wall's own vertical height
    z=-h(nu)..+h(nu), h=rim(nu)*tan(opening_angle) -- exactly
    disc.outer_edge_grid's own cell geometry, just built directly here
    (restricted to the hot spot's own narrow azimuthal window, not the
    whole rim) rather than reused wholesale. For a flat disc
    (opening_angle==0) a single ring of thin radial slivers instead,
    (1-flat_band_frac)*rim(nu) to rim(nu) at z=0, same as before.

    dphi_max_deg: downstream extent (same prograde/increasing-phi sense
    as disc_surfaces_with_teff's own dphi=mod(phi-phi_h,2*pi)) the
    T_h*exp(-dphi/L_h) stripe covers, starting exactly at the central
    impact azimuth phi_h -- normally simulate.py's own hot-spot-vs-disc-
    temperature crossing point (see its _hotspot_dphi_max), so the grid
    stops exactly where the real max(T_base, hot) formula drops to
    T_base (no boost left at all) instead of an arbitrary fixed cutoff.

    footprint: optional (W_eff, H) -- the stream's own physical
    footprint on the wall (W_eff tangential half-width, oblique-
    projection-stretched by the incidence angle; H vertical half-height
    -- same construction as stream_impact_ellipse_outline's disc-flush
    ellipse), centered on phi_h. Every cell actually covered by that
    footprint (an ellipse in the local (arc-length-along-rim, z) plane,
    straddling phi_h on BOTH sides -- the grid's own azimuthal window is
    extended upstream to cover it) gets a flat alpha=1.0 (i.e. exactly
    T_h), overriding whatever the stripe gave there -- applied strictly
    AFTER the stripe (matching render.disc_surfaces_with_teff's own
    matching `footprint` override there), so the physically-struck patch
    always reads as fully "hot" regardless of how that compares to the
    stripe's own value at zero downstream distance. None (the default)
    skips this, leaving the plain stripe with no footprint carve-out.

    lobe/primary_center/R1: same occlusion inputs disc_hotspot_wedges
    took (self-occlusion by the disc's own solid, occultation by the
    secondary's Roche lobe, occultation by the primary star), tested
    once per cell via _disc_curve_visible. Left all None (the default)
    skips occlusion testing entirely -- e.g. plot_topdown_shadows' own
    face-on view has no use for it, by design (see its own docstring).

    Returns (verts, facecolors): `verts` a list of (4,2) arrays, one per
    visible cell, `facecolors` a matching list of RGBA tuples -- ready
    for a single matplotlib.collections.PolyCollection(verts,
    facecolors=facecolors, edgecolors="none"), far cheaper at this cell
    count than one ax.fill call each.
    """
    # Fully vectorized (one batched _disc_curve_visible/project call for
    # every cell corner at once, not one Python-level call per cell):
    # at a few hundred cells, a per-cell call is dominated by
    # visible_mask_bulk's own ConvexHull(lobe surface) recomputation --
    # paid ONCE here instead of once per cell -- the difference between
    # a sub-second plot and a multi-minute one at this function's own
    # default resolution.
    L_h = np.radians(L_h_deg)
    dphi_max = np.radians(dphi_max_deg)
    if footprint is not None:
        # extend the grid's own start upstream far enough to cover the
        # footprint ellipse's own widest (z=0) point -- a rough,
        # angle-from-a-single-radius conversion of W_eff, just to size
        # the grid; the actual membership test below uses each cell's
        # own local R_rim, not this approximation.
        W_eff, H = footprint
        nu_start = phi_h - W_eff / max(disc.rim(phi_h), 1e-300)
    else:
        nu_start = phi_h
    nu_edges = np.linspace(nu_start, phi_h + dphi_max, n_nu + 1)
    nu_lo, nu_hi = nu_edges[:-1], nu_edges[1:]
    nu_c = 0.5 * (nu_lo + nu_hi)
    R_rim = disc.rim(nu_c)
    valid = R_rim > disc.r_in
    cos_lo, sin_lo = np.cos(nu_lo), np.sin(nu_lo)
    cos_hi, sin_hi = np.cos(nu_hi), np.sin(nu_hi)
    x_lo, y_lo = disc.x1 + R_rim * cos_lo, R_rim * sin_lo
    x_hi, y_hi = disc.x1 + R_rim * cos_hi, R_rim * sin_hi

    flared = disc.opening_angle > 0.0
    if flared:
        tanA = np.tan(disc.opening_angle)
        h = R_rim * tanA
        s_edges = np.linspace(-1.0, 1.0, n_z + 1)
        s_lo, s_hi = s_edges[:-1], s_edges[1:]
        Z_LO = s_lo[:, None] * h[None, :]
        Z_HI = s_hi[:, None] * h[None, :]
        X_LO = np.broadcast_to(x_lo, (n_z, n_nu))
        Y_LO = np.broadcast_to(y_lo, (n_z, n_nu))
        X_HI = np.broadcast_to(x_hi, (n_z, n_nu))
        Y_HI = np.broadcast_to(y_hi, (n_z, n_nu))
        NU_C = np.broadcast_to(nu_c, (n_z, n_nu))
        VALID = np.broadcast_to(valid, (n_z, n_nu))
        R_RIM = np.broadcast_to(R_rim, (n_z, n_nu))
        Z_C = 0.5 * (Z_LO + Z_HI)
        pts3 = np.stack([np.stack([X_LO, Y_LO, Z_LO], axis=-1),
                          np.stack([X_HI, Y_HI, Z_LO], axis=-1),
                          np.stack([X_HI, Y_HI, Z_HI], axis=-1),
                          np.stack([X_LO, Y_LO, Z_HI], axis=-1)], axis=-2)  # (n_z,n_nu,4,3)
    else:
        r_in_edge = R_rim * (1.0 - flat_band_frac)
        xi_lo, yi_lo = disc.x1 + r_in_edge * cos_lo, r_in_edge * sin_lo
        xi_hi, yi_hi = disc.x1 + r_in_edge * cos_hi, r_in_edge * sin_hi
        zeros = np.zeros_like(x_lo)
        pts3 = np.stack([np.stack([x_lo, y_lo, zeros], axis=-1),
                          np.stack([x_hi, y_hi, zeros], axis=-1),
                          np.stack([xi_hi, yi_hi, zeros], axis=-1),
                          np.stack([xi_lo, yi_lo, zeros], axis=-1)], axis=-2)[None, ...]  # (1,n_nu,4,3)
        NU_C = nu_c[None, :]
        VALID = valid[None, :]
        R_RIM = R_rim[None, :]
        Z_C = np.zeros_like(NU_C)

    normal = np.stack([np.cos(NU_C), np.sin(NU_C), np.zeros_like(NU_C)], axis=-1)
    normals4 = np.broadcast_to(normal[..., None, :], pts3.shape)

    if lobe is not None or primary_center is not None:
        vis = _disc_curve_visible(pts3, normals4, phase, incl, disc, lobe=lobe,
                                   primary_center=primary_center, R1=R1)
        cell_visible = vis.all(axis=-1) & VALID
    else:
        cell_visible = VALID

    X, Y, _ = project(pts3, phase, incl)
    # the stripe: T_h*exp(-dphi/L_h) normalized by T_h, dphi the
    # DOWNSTREAM-only (mod-wrapped, always >=0) angular distance from
    # phi_h -- upstream cells (possible now that the grid can start
    # before phi_h, to cover the footprint below) wrap to dphi near
    # 2*pi, correctly reading as ~0 stripe contribution there.
    dphi_downstream = np.mod(NU_C - phi_h, 2.0 * np.pi)
    alpha = np.clip(np.exp(-dphi_downstream / L_h), 0.0, 1.0)

    if footprint is not None:
        # THEN (overriding the stripe) the footprint itself: an ellipse
        # in the local (arc-length-along-rim, z) plane centered on
        # phi_h, using each cell's own SIGNED angular offset (can be
        # negative, unlike dphi_downstream above) converted to arc
        # length via that cell's own local R_rim.
        W_eff, H = footprint
        dnu_signed = np.mod(NU_C - phi_h + np.pi, 2.0 * np.pi) - np.pi
        s = R_RIM * dnu_signed
        in_footprint = (s / W_eff) ** 2 + (Z_C / H) ** 2 <= 1.0
        alpha = np.where(in_footprint, 1.0, alpha)

    verts, colors = [], []
    for j, i in zip(*np.nonzero(cell_visible)):
        verts.append(np.column_stack([X[j, i], Y[j, i]]))
        colors.append((0.0, 0.5, 0.0, float(alpha[j, i])))
    return verts, colors


def _occluded_by_sphere(pts, center, radius, n_hat):
    """
    True where a point sits behind a sphere (`center`, `radius`) along the
    sky-ward ray pts + t*n_hat (t>0) -- used to hide disc reference curves
    that pass behind the primary star.
    """
    v = center - pts
    t_closest = np.einsum("...i,i->...", v, n_hat)
    closest = pts + t_closest[..., None] * n_hat
    dist2 = np.einsum("...i,...i->...", closest - center, closest - center)
    return (t_closest > 0.0) & (dist2 < radius ** 2)


def _disc_curve_visible(pts, normal_dir, phase, incl, disc, lobe=None,
                         primary_center=None, R1=None, push=1e-6):
    """
    Visibility of points that lie exactly on the disc's own solid
    boundary (so a naive self-occlusion test would trivially call them
    "inside" right at the curve itself): push each point slightly outward
    along its known local `normal_dir` before testing self-occlusion
    (disc.visible), occultation by the secondary's Roche lobe (if `lobe`
    given), and occultation by the primary star's sphere (if
    `primary_center`/`R1` given -- needed for e.g. the inner disc rim,
    which can pass directly behind the primary).
    """
    pts_test = pts + push * normal_dir
    if disc.opening_angle > 0.0:
        vis = disc.visible(pts_test, phase, incl)
    else:
        vis = np.ones(pts.shape[0], dtype=bool)
    if lobe is not None:
        vis = vis & visible_mask_bulk(pts, phase, incl, lobe)
    if primary_center is not None and R1 is not None:
        n_hat, _, _ = observer_frame(phase, incl)
        vis = vis & ~_occluded_by_sphere(pts, primary_center, R1, n_hat)
    return vis


def disc_cross_section_lines(disc, phase, incl, lobe=None, n_nu=200,
                              primary_center=None, R1=None):
    """
    For a flared disc, four extra curves that make the cone/edge
    structure actually visible in an outline plot (the silhouette alone
    looks like a flat ellipse): the projected top/bottom lips of both the
    outer edge wall (r=rim(nu), z=+-rim(nu)*tan(opening_angle)) and the
    inner edge wall (r=r_in, z=+-r_in*tan(opening_angle), always
    circular). Returns [(X,Y,visible), ...] x4 (outer-up, outer-lo,
    inner-up, inner-lo), or [] if opening_angle==0. `visible` marks
    points hidden behind the disc's own solid, the Roche lobe, or the
    primary star (whichever of `lobe`/`primary_center`+`R1` are given)
    for _hidden_split_plot.

    The inner-up/inner-lo pair is omitted (only the outer two are
    returned) when R1 is given and disc.r_in <= R1 -- same reasoning as
    disc_inner_rim_line's own matching guard: the inner edge wall sits at
    or inside the primary's own surface, so there's no disc material
    there to mark a distinct boundary for.
    """
    if disc.opening_angle <= 0.0:
        return []
    nu = np.linspace(0.0, 2.0 * np.pi, n_nu)
    lines = []
    edges = [(disc.rim(nu), 1.0)]
    if R1 is None or disc.r_in > R1:
        edges.append((np.full(n_nu, disc.r_in), -1.0))
    for R, outward_sign in edges:
        h = R * np.tan(disc.opening_angle)
        x = disc.x1 + R * np.cos(nu)
        y = R * np.sin(nu)
        normal_dir = outward_sign * np.stack([np.cos(nu), np.sin(nu), np.zeros(n_nu)], axis=-1)
        for z in (h, -h):
            pts = np.stack([x, y, z], axis=-1)
            vis = _disc_curve_visible(pts, normal_dir, phase, incl, disc, lobe=lobe,
                                       primary_center=primary_center, R1=R1)
            X, Y, _ = project(pts, phase, incl)
            lines.append((X, Y, vis))
    return lines


def _periastron_line(disc):
    """
    (2,3) corotating-frame points from the primary (disc.x1, 0, 0) out to
    the disc rim's own periastron (disc.rim's minimum, at true anomaly
    nu=disc.omega -- see disc.Disc.rim's own R(nu) formula) -- a straight
    segment in this frame, so project() alone (linear) gives a straight
    line in any projection too, with no need to sample it more finely.
    Only meaningful for an eccentric disc (disc.e>0); callers gate on
    that themselves rather than this always returning something trivial.
    """
    r_peri = disc.rim(disc.omega)
    peri = np.array([disc.x1 + r_peri * np.cos(disc.omega), r_peri * np.sin(disc.omega), 0.0])
    return np.stack([np.array([disc.x1, 0.0, 0.0]), peri])


def stream_impact_ellipse_outline(x_imp, y_imp, vx_imp, vy_imp, H, W, phase, incl, disc,
                                   n=100, lobe=None, primary_center=None, R1=None):
    """
    Outline of the ballistic accretion stream's own cross-section, at the
    point (x_imp, y_imp, 0) [corotating frame] where it strikes the
    disc's outer rim wall -- projected ALONG the stream's own local
    direction of travel onto that wall's local tangent plane (spanned by
    the rim's azimuthal direction and +-z), rather than left as a free
    ellipse in the plane perpendicular to travel: a beam of
    circular/elliptical cross-section striking a surface at an angle
    leaves an elongated footprint there (an oblique projection, stretched
    in the direction of travel's in-plane component), exactly like a
    flashlight's circular beam making an ellipse on a tilted wall. An
    earlier version left the raw perpendicular-to-travel ellipse
    unprojected, which could dip into/out of the disc's own solid on
    either side of the impact point -- _disc_curve_visible then
    (correctly) rendered that part fainter, an occlusion artifact of the
    wrong shape rather than the stream's real footprint.

    Travel is confined to the orbital midplane (z=0, see
    stream.integrate_stream), and the wall's own local outward normal is
    purely radial (r_hat=(cos nu_imp, sin nu_imp), matching
    disc_hotspot_cells' own convention for this same role -- the small
    rim-slope correction impact_incidence's own n_hat_imp adds is not
    worth a second, inconsistent convention here). With the projection
    direction confined to that plane, the algebra collapses cleanly: the
    vertical semi-axis H is untouched (the projection has no
    z-component), while the in-plane semi-axis becomes W/cos_incidence,
    entirely along the wall's azimuthal tangent
    t_hat=(-sin nu_imp, cos nu_imp) -- true for ANY incidence angle, not
    just the special case where the old ellipse's own perpendicular-to-
    travel axis happened to already line up with t_hat. cos_incidence
    here is the SIGNED v_hat.r_hat (not impact_incidence's own abs()),
    floored in magnitude to keep the projection finite as incidence
    approaches grazing (cos_incidence -> 0), where the true footprint
    really does smear out indefinitely along the wall.

    Occlusion-tested exactly like disc_cross_section_lines'/
    disc_hotspot_cells' own wall-lip curves, via _disc_curve_visible
    (self-occlusion by `disc`'s own solid, `lobe`'s Roche lobe, the
    primary sphere via primary_center/R1), using the wall's own outward
    radial direction (constant across the curve, like
    disc_hotspot_cells' normal_dir) as the occlusion push.

    Returns (X, Y, visible) for _hidden_split_plot, same convention as
    disc_outline/disc_cross_section_lines.
    """
    v = np.array([vx_imp, vy_imp])
    vnorm = np.hypot(*v)
    v_hat = v / vnorm if vnorm > 0.0 else np.array([1.0, 0.0])
    nu_imp = np.arctan2(y_imp, x_imp - disc.x1)
    r_hat = np.array([np.cos(nu_imp), np.sin(nu_imp)])
    t_hat = np.array([-np.sin(nu_imp), np.cos(nu_imp)])
    cos_incidence = v_hat[0] * r_hat[0] + v_hat[1] * r_hat[1]
    cos_incidence = (max(cos_incidence, 0.05) if cos_incidence >= 0.0
                      else min(cos_incidence, -0.05))
    psi = np.linspace(0.0, 2.0 * np.pi, n)
    ct = (W / cos_incidence) * np.cos(psi)
    x = x_imp + ct * t_hat[0]
    y = y_imp + ct * t_hat[1]
    z = H * np.sin(psi)
    pts = np.stack([x, y, z], axis=-1)
    normal_dir = np.broadcast_to(np.array([r_hat[0], r_hat[1], 0.0]), pts.shape)
    vis = _disc_curve_visible(pts, normal_dir, phase, incl, disc, lobe=lobe,
                               primary_center=primary_center, R1=R1)
    X, Y, _ = project(pts, phase, incl)
    return X, Y, vis


def stream_cross_section_outline(x, y, vx, vy, H, W, phase, incl, disc, n=100,
                                  lobe=None, primary_center=None, R1=None):
    """
    A free ellipse in the plane perpendicular to the stream's own local
    direction of travel at (x, y, 0) [corotating frame] -- semi-axis W
    transverse to travel in the orbital midplane, semi-axis H along +-z
    -- for use anywhere the stream ISN'T striking a surface to project
    onto (contrast stream_impact_ellipse_outline's disc-flush footprint
    at the actual disc-impact point): the stream's own starting
    cross-section near L1, or its terminal cross-section where a
    discless system's trajectory simply ends (primary impact or
    truncation at closest approach) rather than hitting a disc rim.

    Occlusion-tested via _disc_curve_visible (self-occlusion by `disc`'s
    own solid -- relevant if this cross-section happens to sit near/
    behind it, occultation by `lobe`'s Roche lobe -- relevant since the
    stream's own starting cross-section, right at L1, sits flush against
    the secondary's own surface, and the primary sphere via
    primary_center/R1), using each point's own outward (center-relative)
    direction as its local normal for the occlusion push -- same
    construction stream_impact_ellipse_outline used before it was
    changed to project onto the disc surface instead.

    Returns (X, Y, visible) for _hidden_split_plot, same convention as
    disc_outline/disc_cross_section_lines.
    """
    v = np.array([vx, vy])
    vnorm = np.hypot(*v)
    v_hat = v / vnorm if vnorm > 0.0 else np.array([1.0, 0.0])
    perp = np.array([-v_hat[1], v_hat[0]])
    psi = np.linspace(0.0, 2.0 * np.pi, n)
    cw, sh = W * np.cos(psi), H * np.sin(psi)
    x_pts = x + cw * perp[0]
    y_pts = y + cw * perp[1]
    z_pts = sh
    pts = np.stack([x_pts, y_pts, z_pts], axis=-1)
    normal_dir = np.stack([np.cos(psi) * perp[0], np.cos(psi) * perp[1], np.sin(psi)], axis=-1)
    vis = _disc_curve_visible(pts, normal_dir, phase, incl, disc, lobe=lobe,
                               primary_center=primary_center, R1=R1)
    X, Y, _ = project(pts, phase, incl)
    return X, Y, vis


def stream_tube_outline(traj_x, traj_y, lobe, eps, phase, incl):
    """
    Two curves tracing the outer silhouette of the ballistic accretion
    stream's own volume, swept along the whole trajectory (traj_x,
    traj_y) [corotating frame, z=0, arclength-sampled -- e.g.
    simulate.py's own state["xs"]/state["ys"]] -- unlike
    stream_impact_ellipse_outline's single representative cross-section
    at the disc-impact point alone.

    At each trajectory point, build that point's own local elliptical
    cross-section (same convention as stream_impact_ellipse_outline:
    semi-axis W transverse to travel in the orbital midplane, H along
    +-z, from stream.lubow_shu_stream_size's own r1-dependent
    scaleheights) and find its two envelope points -- where the ellipse
    is tangent to a line parallel to the path's own local (projected)
    tangent direction, the standard construction for an extruded tube's
    true projected silhouette. This is deliberately the ellipse's
    extreme points along the local path-PERPENDICULAR direction, not
    simply its two points farthest from the projected center: those
    coincide for a genuinely elongated projected ellipse, but a plain
    farthest-from-center search is ill-conditioned (and flips
    unpredictably from one trajectory point to the next) wherever H~W
    makes the projected cross-section nearly circular, since then most
    points on the ring are almost equally far from center. Anchoring to
    the path's own smoothly-varying tangent direction instead keeps the
    two flanking curves smooth even through such near-circular stretches.

    Since project() is linear, the projected ring is again an ellipse,
    so this has a closed form: writing the projected offset from center
    as W*cos(psi)*Proj(perp) + H*sin(psi)*Proj(z_hat), its dot product
    with the local projected normal direction is A*cos(psi)+B*sin(psi),
    maximized at psi*=atan2(B,A) and minimized at psi*+pi -- no sampling
    or search needed, and the two points are then an exact reflection of
    one another through the projected center.

    No occlusion test (contrast stream_impact_ellipse_outline) -- this
    is meant as a *fill* boundary, not a line plot, and every other
    filled body in plot_component_outlines (fill_secondary,
    fill_disc_and_primary) is likewise a simple unconditional shape,
    with only whole-body painter's-algorithm depth ordering between
    fills, not per-point self-occlusion within any one of them.

    Returns (X1, Y1, X2, Y2): one flanking curve per side, same length
    as traj_x/traj_y.
    """
    from stream import lubow_shu_stream_size

    traj_x = np.asarray(traj_x, dtype=float)
    traj_y = np.asarray(traj_y, dtype=float)
    n = len(traj_x)
    tx, ty = np.gradient(traj_x), np.gradient(traj_y)
    tnorm = np.hypot(tx, ty)
    tnorm = np.where(tnorm > 0.0, tnorm, 1.0)
    tx, ty = tx / tnorm, ty / tnorm
    perp_x, perp_y = -ty, tx
    r1 = np.hypot(traj_x - lobe.x1, traj_y)
    H, W = lubow_shu_stream_size(r1, lobe.q, eps)

    Xc, Yc, _ = project(np.stack([traj_x, traj_y, np.zeros(n)], axis=-1), phase, incl)
    # Proj(tangent), Proj(perp) vary along the path (local direction);
    # Proj(z_hat) is the same fixed 2D vector everywhere (project() has
    # no translation term, so projecting any vector -- not just a point
    # -- is well-defined and position-independent).
    Xt, Yt, _ = project(np.stack([tx, ty, np.zeros(n)], axis=-1), phase, incl)
    Xe1, Ye1, _ = project(np.stack([perp_x, perp_y, np.zeros(n)], axis=-1), phase, incl)
    zX, zY, _ = project(np.array([0.0, 0.0, 1.0]), phase, incl)

    tnorm2 = np.hypot(Xt, Yt)
    tnorm2 = np.where(tnorm2 > 0.0, tnorm2, 1.0)
    # local path-perpendicular direction in the projected (sky) plane
    nX, nY = -Yt / tnorm2, Xt / tnorm2

    A = W * (Xe1 * nX + Ye1 * nY)
    B = H * (zX * nX + zY * nY)
    psi_star = np.arctan2(B, A)
    cos_s, sin_s = np.cos(psi_star), np.sin(psi_star)

    dX = W * cos_s * Xe1 + H * sin_s * zX
    dY = W * cos_s * Ye1 + H * sin_s * zY
    X1, Y1 = Xc + dX, Yc + dY
    X2, Y2 = Xc - dX, Yc - dY
    return X1, Y1, X2, Y2


def disc_inner_rim_line(disc, phase, incl, lobe=None, n=150,
                         primary_center=None, R1=None):
    """
    The disc's inner-edge circle (r=r_in, always circular in this model)
    at z=0, projected -- the midplane trace of the inner hole, shown
    separately since disc_midplane_line/disc_outline only trace the outer
    rim. Present regardless of opening_angle (a flat disc's inner edge is
    just this circle; a flared disc additionally gets the inner edge's
    up/down lips from disc_cross_section_lines). Returns (X,Y,visible).

    Returns empty arrays when R1 is given and disc.r_in <= R1 -- the
    inner rim sits at or inside the primary's own surface, so there's no
    disc material between the star and its own equator to mark a distinct
    boundary for; drawing this circle anyway would just retrace the
    star's own equator (a real 3D circle on its surface, but not its
    projected silhouette except exactly edge-on), an artifact with no
    physical meaning of its own.
    """
    if R1 is not None and disc.r_in <= R1:
        return np.zeros(0), np.zeros(0), np.zeros(0, dtype=bool)
    ang = np.linspace(0.0, 2.0 * np.pi, n)
    x = disc.x1 + disc.r_in * np.cos(ang)
    y = disc.r_in * np.sin(ang)
    pts = np.stack([x, y, np.zeros(n)], axis=-1)
    normal_dir = -np.stack([np.cos(ang), np.sin(ang), np.zeros(n)], axis=-1)
    vis = _disc_curve_visible(pts, normal_dir, phase, incl, disc, lobe=lobe,
                               primary_center=primary_center, R1=R1)
    X, Y, _ = project(pts, phase, incl)
    return X, Y, vis


def lobe_midplane_line(lobe, phase, incl, n=300):
    """
    The secondary Roche lobe's equatorial cross-section (theta=pi/2, i.e.
    z=0 in the corotating frame -- the orbital midplane), projected.
    Distinct from lobe_outline's observer-facing SILHOUETTE (a convex
    hull over the whole 3D surface, which is generally NOT the same curve
    as this actual in-plane cut except by coincidence at special phases.
    `visible` (for _hidden_split_plot) is a simple back-face test: the
    lobe's own local outward normal (from its potential gradient) must
    face the observer, since the far half of this equatorial ring is
    self-occluded by the lobe's own bulk.

    phi=pi (the L1 direction) is inserted explicitly rather than left to
    a plain linspace: L1 is a genuine cusp of this curve (a vertex, not a
    smooth point), and a generic evenly-spaced phi grid generally doesn't
    land a sample exactly there, leaving the drawn polyline's nearest
    two points straddling it -- which looks like the curve rounds the
    corner off/avoids the point instead of actually reaching it.
    """
    n_hat, _, _ = observer_frame(phase, incl)
    phi = np.sort(np.concatenate([np.linspace(0.0, 2.0 * np.pi, n - 1), [np.pi]]))
    r = lobe.radius(np.full(n, np.pi / 2.0), phi)
    x = lobe.x2 + r * np.cos(phi)
    y = r * np.sin(phi)
    g = gravity(x, y, np.zeros(n), lobe.q)
    normal = -g / np.linalg.norm(g, axis=-1, keepdims=True)
    vis = np.einsum("ij,j->i", normal, n_hat) > 0.0
    pts = np.stack([x, y, np.zeros(n)], axis=-1)
    X, Y, _ = project(pts, phase, incl)
    return X, Y, vis


def full_lobe_midplane_line(lobe, phase, incl, n=300):
    """
    The secondary's FULL (fill_factor=1) Roche lobe's equatorial (z=0)
    cross-section, projected -- a purely geometric reference curve,
    independent of the secondary's own actual current shape
    (lobe_midplane_line), which is smaller/rounder whenever it's
    underfilling (lobe.fill_factor<1, see roche.RocheLobe). This is the
    secondary's counterpart to primary_lobe_midplane_line (the primary's
    own, never-actually-filled reference lobe) and is drawn the same
    way: no occlusion/back-face treatment, since it isn't a real opaque
    surface, just a comparison curve.

    Built from a fresh RocheLobe at the same q but fill_factor=1
    (ignoring `lobe`'s own, possibly underfilling, r_grid) -- ntheta=3
    is enough for this (theta=pi/2 then lands exactly on the middle grid
    row, so RocheLobe.radius's bilinear interpolation is exact in theta,
    not approximate), so this stays cheap even though it means building
    a second lobe.

    phi=pi (the L1 direction) is inserted explicitly; see
    lobe_midplane_line's docstring for why a plain linspace generally
    misses it.
    """
    full_lobe = RocheLobe(lobe.q, ntheta=3, nphi=361)
    phi = np.sort(np.concatenate([np.linspace(0.0, 2.0 * np.pi, n - 1), [np.pi]]))
    r = full_lobe.radius(np.full(n, np.pi / 2.0), phi)
    x = full_lobe.x2 + r * np.cos(phi)
    y = r * np.sin(phi)
    pts = np.stack([x, y, np.zeros(n)], axis=-1)
    X, Y, _ = project(pts, phase, incl)
    return X, Y


def primary_lobe_midplane_line(lobe, phase, incl, n=300):
    """
    The primary Roche lobe's own equatorial (z=0) cross-section, derived
    by the same q -> 1/q mirror as the primary lobe itself. Drawn at
    uniform strength (no back-face/occlusion treatment) since it is a
    purely geometric reference curve, not a physical opaque surface.

    phi=pi (the L1 direction, this lobe's own cusp too -- L1 is the one
    point shared by both lobes' equipotential surfaces) is inserted
    explicitly; see lobe_midplane_line's docstring for why a plain
    linspace generally misses it.
    """
    inv_lobe = RocheLobe(1.0 / lobe.q)
    phi = np.sort(np.concatenate([np.linspace(0.0, 2.0 * np.pi, n - 1), [np.pi]]))
    r = inv_lobe.radius(np.full(n, np.pi / 2.0), phi)
    x = -(inv_lobe.x2 + r * np.cos(phi))
    y = r * np.sin(phi)
    pts = np.stack([x, y, np.zeros(n)], axis=-1)
    X, Y, _ = project(pts, phase, incl)
    return X, Y


def primary_midplane_line(center_xyz, radius, phase, incl, disc=None, lobe=None, n=150):
    """
    The primary's TRUE equatorial great circle (z=0 in the corotating
    frame, i.e. its intersection with the orbital midplane) -- distinct
    from the sky-facing limb circle used elsewhere for occultation tests.
    `visible` combines the sphere's own back-face test (its outward
    normal at that point is just the radial direction) with occultation
    by the disc and/or Roche lobe, if given.
    """
    n_hat, _, _ = observer_frame(phase, incl)
    ang = np.linspace(0.0, 2.0 * np.pi, n)
    radial = np.stack([np.cos(ang), np.sin(ang), np.zeros(n)], axis=-1)
    pts = np.asarray(center_xyz) + radius * radial
    vis = np.einsum("ij,j->i", radial, n_hat) > 0.0
    if disc is not None:
        vis &= disc.visible(pts, phase, incl)
    if lobe is not None:
        vis &= visible_mask_bulk(pts, phase, incl, lobe)
    X, Y, _ = project(pts, phase, incl)
    return X, Y, vis


def disc_midplane_line(disc, phase, incl, lobe=None, n=200,
                        primary_center=None, R1=None):
    """
    The disc rim's own midplane (z=0) curve, projected. For a flared disc
    this differs from disc_outline's convex-hull envelope of the whole 3D
    solid; for a flat disc it coincides with disc_outline exactly. This
    z=0 line sits at half-height on the outer edge wall, so it shares
    that surface's outward normal, (cos(nu),sin(nu),0).
    """
    nu = np.linspace(0.0, 2.0 * np.pi, n)
    R = disc.rim(nu)
    x = disc.x1 + R * np.cos(nu)
    y = R * np.sin(nu)
    pts = np.stack([x, y, np.zeros(n)], axis=-1)
    normal_dir = np.stack([np.cos(nu), np.sin(nu), np.zeros(n)], axis=-1)
    vis = _disc_curve_visible(pts, normal_dir, phase, incl, disc, lobe=lobe,
                               primary_center=primary_center, R1=R1)
    X, Y, _ = project(pts, phase, incl)
    return X, Y, vis


def stream_outline(traj, x1, phase, incl, disc=None, lobe=None, primary_center=None, R1=None):
    """
    Stream outline, drawn exactly as given in `traj` (x1 is unused by this
    function itself -- kept for call-site symmetry with other outline
    helpers) -- the caller is expected to have already truncated `traj` to
    wherever the ballistic trajectory stops being physically meaningful
    (its first pericenter passage around the primary/"closest approach",
    see stream.closest_approach_index, or -- for a magnetic CV -- the
    accretion-spot connection point, see stream.angle_acc_index; simulate.py's
    "outline" output does this once, up front). Re-deriving that
    truncation here from an already-truncated, resampled curve is not
    just redundant: interpolation can make r1(s) wobble non-monotonically
    right at the far endpoint (its true minimum), so re-running
    closest_approach_index on the resampled array occasionally clips a
    few points short of the caller's actual intended endpoint -- visible
    as a gap where this curve should meet another one starting exactly
    there (e.g. the accretion-spot field line, see field_line_outlines).
    `visible` marks points hidden behind the disc, the Roche lobe, and/or
    (if `primary_center`/`R1` given) the primary star's sphere -- relevant
    right near the far endpoint, where the trajectory passes nearest the
    primary.
    """
    pts = np.stack([traj["x"], traj["y"], np.zeros_like(traj["x"])], axis=-1)
    vis = np.ones(pts.shape[0], dtype=bool)
    if disc is not None:
        vis &= disc.visible(pts, phase, incl)
    if lobe is not None:
        vis &= visible_mask(pts, phase, incl, lobe)
    if primary_center is not None and R1 is not None:
        n_hat, _, _ = observer_frame(phase, incl)
        vis &= ~_occluded_by_sphere(pts, primary_center, R1, n_hat)
    X, Y, _ = project(pts, phase, incl)
    return X, Y, vis


def _field_line_r_max_factor(R1, disc=None, lobe=None):
    """
    The dipole field loops'/outline's outer extent (field_line_points'
    r_max_factor), set from the actual system geometry rather than
    magnetic.py's bare placeholder default -- shared by field_line_outlines
    and plot_topdown_shadows so the two views always agree: if there's a
    real disc (disc is not None and not disc.is_empty), the magnetosphere
    is assumed truncated at 2x the disc's own inner radius -- inside that,
    accretion would be magnetically channeled rather than forming a disc,
    so the field lines shouldn't visually extend past roughly where the
    disc actually starts. Without a disc (e.g. a polar like AM Her, which
    has none), there's no such inner truncation to go by, so the lines
    instead fill the primary's own Roche lobe (roche.eggleton_radius(1/
    lobe.q) -- the primary's mass ratio relative to its companion is the
    inverse of lobe.q = M2/M1).
    """
    from magnetic import DEFAULT_FIELD_LINE_R_MAX_FACTOR

    if disc is not None and not disc.is_empty:
        r_max_factor = 2.0 * disc.r_in / R1
    elif lobe is not None:
        from roche import eggleton_radius
        r_max_factor = eggleton_radius(1.0 / lobe.q) / R1
    else:
        r_max_factor = DEFAULT_FIELD_LINE_R_MAX_FACTOR
    return max(r_max_factor, 1.0001)  # loops must at least reach the surface


def _magnetic_pole_points(primary_center, R1, theta_1, phi_1):
    """
    The two points where the primary's magnetic axis (magnetic_frame's
    m_hat) meets its surface -- shared by plot_component_outlines and
    plot_topdown_shadows so both draw the same north/south pole markers
    from one definition. Returns (pole_dirs, pole_pts): pole_dirs is
    (2,3), [m_hat, -m_hat]; pole_pts is pole_dirs*R1 offset from
    `primary_center`.
    """
    from magnetic import magnetic_frame

    m_hat, _, _ = magnetic_frame(theta_1, phi_1)
    pole_dirs = np.stack([m_hat, -m_hat])
    return pole_dirs, primary_center + R1 * pole_dirs


def field_line_outlines(primary_center, R1, theta_1, phi_1, n_field_1, phase, incl,
                         disc=None, lobe=None, n_per_line=100):
    """
    Projected (X,Y,visible) curves for n_field_1 dipole field-line loops
    anchored on the primary's surface (magnetic.field_line_points) --
    geometry only, no field strength (see magnetic.py). `visible` marks
    points occulted by the secondary's Roche lobe (visible_mask_bulk --
    the primary's own surroundings stay well clear of the L1 funnel, same
    reasoning as primary_visibility_grid's own occlusion test), the disc
    (disc.visible), and/or the primary star's own sphere (_occluded_by_sphere
    -- large loops pass behind the star itself).

    The loops' outer extent is set from the actual system geometry -- see
    _field_line_r_max_factor.

    Returns a list of n_field_1 (X, Y, visible) tuples, or [] if
    n_field_1<=0.
    """
    if n_field_1 <= 0:
        return []
    from magnetic import field_line_points

    r_max_factor = _field_line_r_max_factor(R1, disc=disc, lobe=lobe)

    lines = field_line_points(primary_center, R1, theta_1, phi_1, n_field_1,
                               n_per_line=n_per_line, r_max_factor=r_max_factor)
    n_hat, _, _ = observer_frame(phase, incl)
    curves = []
    for pts in lines:
        vis = np.ones(pts.shape[0], dtype=bool)
        if lobe is not None:
            vis &= visible_mask_bulk(pts, phase, incl, lobe)
        if disc is not None:
            vis &= disc.visible(pts, phase, incl)
        vis &= ~_occluded_by_sphere(pts, primary_center, R1, n_hat)
        X, Y, _ = project(pts, phase, incl)
        curves.append((X, Y, vis))
    return curves


def _plot_sample_points(ax, lobe, disc, R1, phase, incl, n_sec, n_disc, n_primary,
                         secondary_color, primary_color):
    """
    Scatter each body's actual rendering sample points (tiny markers,
    near-hemisphere/unocculted only, same criteria render_system_image
    itself uses) on top of the outline -- lets --n_areas_secondary/
    --n_areas_disc/--n_areas_primary be judged visually (too sparse:
    visible gaps between dots; wastefully dense: a solid smear) without
    running a full temperature render. n_sec is a plain count (see
    roche.RocheLobe.equal_area_sample); n_disc, n_primary are (n1,n2)
    grid-dimension pairs (see render.split_area_count) -- the same
    resolution knobs build_temperature_maps/render_system_image use.
    Stream points aren't shown: the outline's own curve already samples
    it densely, and unlike the other three bodies it has no separate
    --n_areas_* resolution knob to judge.
    """
    from render import disc_grid_with_teff, disc_surfaces_with_teff, disc_face_rim_split
    style = dict(marker="o", ls="none", ms=5.0, alpha=0.8, zorder=0, markeredgewidth=0)
    n_hat, _, _ = observer_frame(phase, incl)
    primary_center = np.array([lobe.x1, 0.0, 0.0])

    pts, normals, _areas = lobe.equal_area_sample(n_sec)
    mu_sec = np.einsum("...i,i->...", normals, n_hat) > 0.0
    vis = (mu_sec & disc.visible(pts, phase, incl)
           & ~_occluded_by_sphere(pts, primary_center, R1, n_hat))
    X, Y, _ = project(pts[vis], phase, incl)
    ax.plot(X, Y, color=secondary_color, **style)

    no_teff = lambda r: np.zeros_like(np.asarray(r, dtype=float))
    n_r, n_nu = n_disc
    if disc.opening_angle > 0.0:
        # split the face/rim budget by projected (inclination-dependent)
        # area, same as build_temperature_maps, so these scattered points
        # actually reflect what the real render/light-curve uses -- not a
        # separate, uncorrected face:rim ratio.
        (n_r, n_nu), n_wall = disc_face_rim_split(
            n_r * n_nu, disc.r_in, disc.a, disc.opening_angle, np.degrees(incl))
        xs, ys, zs, ns, _a, _T, _sid = disc_surfaces_with_teff(
            disc, no_teff, n_r=n_r, n_nu=n_nu, n_wall=n_wall)
        pts_d = np.stack([xs, ys, zs], axis=-1)
        # nudge just outside the solid along its own normal, same as
        # render_system_image -- disc.visible's self-occlusion ray march
        # otherwise immediately registers a point sitting exactly on the
        # solid's boundary as "inside" at t=0, near-randomly depending on
        # floating-point noise (which is what made the disc's points look
        # scattered rather than following the actual grid).
        pts_d_test = pts_d + 1e-6 * ns
        mu_disc = np.einsum("ij,j->i", ns, n_hat) > 0.0
    else:
        xs, ys, _a, _T = disc_grid_with_teff(disc, no_teff, n_r=n_r, n_nu=n_nu)
        pts_d = np.stack([xs, ys, np.zeros_like(xs)], axis=-1)
        pts_d_test = pts_d
        mu_disc = np.full(xs.shape, True)
    vis_d = (visible_mask_bulk(pts_d, phase, incl, lobe) & disc.visible(pts_d_test, phase, incl)
             & mu_disc & ~_occluded_by_sphere(pts_d, primary_center, R1, n_hat))
    Xd, Yd, _ = project(pts_d[vis_d], phase, incl)
    ax.plot(Xd, Yd, color="#e08214", **style)

    n_rho, n_phi = n_primary
    pts1, _, _ = star_points(primary_center, R1, phase, incl, n_rho=n_rho, n_phi=n_phi,
                              limb_u=0.0, return_area=True)
    vis1 = visible_mask_bulk(pts1, phase, incl, lobe) & disc.visible(pts1, phase, incl)
    X1, Y1, _ = project(pts1[vis1], phase, incl)
    ax.plot(X1, Y1, color=primary_color, **style)


def plot_component_outlines(lobe, disc, traj, R1, phase, incl_deg, ax=None,
                             fill=True, show_points=False,
                             n_sec=20000, n_disc=(80, 160), n_primary=(60, 120),
                             theta_1=0.0, phi_1=0.0, n_field_1=0, accretion_field_lines=(),
                             hotspot_phi_h=None, hotspot_L_h_deg=None, hotspot_dphi_max_deg=None,
                             hotspot_footprint=None,
                             stream_impact=None, stream_begin=None, stream_end=None,
                             stream_eps=None, label=None,
                             primary_faint_lw_scale=0.5, primary_faint_alpha=0.3):
    """
    Draw the projected outlines of the secondary (Roche lobe), primary
    (only its disc-visible limb arc(s), radius R1), disc rim, and stream
    (L1 to first pericenter) at a given orbital phase and inclination, on
    `ax` (a new figure if None).

    show_points: also scatter each body's actual rendering sample points
    (see _plot_sample_points) -- a way to visually judge whether
    n_sec/n_disc/n_primary (the same resolution knobs
    build_temperature_maps/render_system_image take, just plumbed through
    here too) are dense enough for a given body, without running a full
    temperature render.

    theta_1/phi_1 [rad]: the primary's magnetic-axis obliquity/azimuth
    (params.SystemParams.theta_1_rad/phi_1_rad -- see magnetic.py).
    n_field_1: number of dipole field-line loops to draw (0 = none, the
    default); geometry only, see field_line_outlines/magnetic.py. Whenever
    this is >0, a small dot is also drawn at each of the two points where
    the magnetic axis meets the primary's surface (its north/south poles),
    occlusion-tested the same way as everything else here (hidden, but
    still faintly shown, behind the primary's own far side, the
    secondary's Roche lobe, or the disc).

    accretion_field_lines: (n,3) curves from the primary's surface to an
    accretion stream connection point (see
    render.TemperatureMaps.accretion_field_line/render.accretion_connection_line,
    build_temperature_maps'/--angle_acc's own docstrings) -- one per --angle_acc
    entry when it's a list, e.g. via simulate.py -- each drawn as a solid
    red line, occlusion-tested the same way as the dipole loops above;
    empty (the default) draws nothing. Only the first actually heats a
    spot on the primary or affects the light curve -- the rest are
    geometry only, same as the dipole loops.

    hotspot_phi_h/hotspot_L_h_deg/hotspot_dphi_max_deg/hotspot_footprint:
    the disc's own stream-impact hot spot (model.T_h/model.L_h, DISC tab
    -- not the magnetic accretion_spot above), shown as the disc's own
    simulated grid cells (see disc_hotspot_cells) sitting on its outer
    edge (never spread radially inward onto its face): the
    T_h*exp(-dphi/L_h) stripe downstream of the central impact azimuth
    hotspot_phi_h [rad] out to hotspot_dphi_max_deg, THEN (overriding
    the stripe) a flat-T_h patch whenever hotspot_footprint (a
    (W_eff, H) pair, see disc_hotspot_cells' own docstring) is also
    given -- the stream's own actual footprint on the wall. Any of
    hotspot_phi_h/hotspot_L_h_deg/hotspot_dphi_max_deg left None/non-
    positive (the default) disables the whole feature;
    hotspot_footprint left None (the default) just skips the footprint
    override, keeping the plain stripe.

    stream_impact: dict {"x","y","vx","vy","H","W"} (see
    stream.disc_impact_point/lubow_shu_stream_size) giving the stream's
    own state at the disc-impact point and its Lubow & Shu transverse
    size there -- drawn (and, if `fill`, filled with the same light green
    as stream_eps's own tube below) as the stream's actual footprint on
    the disc surface there, via stream_impact_ellipse_outline. None (the
    default) draws nothing -- e.g. no disc, or the stream never reaches it.

    stream_begin: dict {"x","y","vx","vy","H","W"}, same shape as
    stream_impact but at the OTHER end of the stream -- its own starting
    cross-section near L1 -- drawn via stream_cross_section_outline (the
    free, non-disc-flush ellipse, since there's no surface to project
    onto at this end) and, if `fill`, filled the same light green. None
    (the default) draws nothing.

    stream_end: dict {"x","y","vx","vy","H","W"}, same shape/drawing as
    stream_begin -- the stream's own terminal cross-section for systems
    where it DOESN'T strike a disc (no disc, or the disc's rim is never
    reached), at wherever the plotted trajectory truncates (closest
    approach to the primary, or an explicit stream_angle cutoff -- see
    simulate.py's own caller). Mutually exclusive with stream_impact in
    practice (a caller with a real disc impact passes that instead, since
    it's the more physically meaningful "ending"); None (the default)
    draws nothing.

    stream_eps: the real (T_2/P_orb/a-derived, see stream.lubow_shu_eps)
    Lubow & Shu sound speed -- when given (not None), the stream's own
    outer volume (swept elliptical cross-sections along the whole `traj`,
    see stream_tube_outline) is drawn as two flanking boundary curves and,
    if `fill`, filled between them with the same light green used for the
    "shadow" output's own stream-width band (plot_topdown_shadows,
    facecolor "#1b9e77", alpha 0.15). None (the default) draws nothing.

    label: simulate.py's --prefix global label (see labeled_title), if
    any -- prepended to this plot's own "q=..., i=..., phase=..." title.

    primary_faint_lw_scale/primary_faint_alpha: the primary's own faint
    "always present" full-limb circle (drawn even when it's entirely
    eclipsed -- see that circle's own comment below) is normally drawn at
    style["lw"]*0.5, alpha=0.3, faint enough to be hard to make out
    against everything else on the same axes. Some callers (e.g.
    simulate.py's "primary" output, comparing this circle directly
    against eclipse-derived shadow/limb curves at a phase where the
    primary usually IS fully eclipsed, so this faint circle is the ONLY
    thing showing it at all) need it to actually stand out -- raise
    these instead of hand-editing the hardcoded defaults every other
    caller (outline mode's own phase sequence, mid-eclipse only
    incidentally) still relies on.
    """
    import matplotlib.pyplot as plt

    incl = np.radians(incl_deg)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
    else:
        fig = ax.figure

    # colors match effective temperature: secondary (cool) red, primary
    # (hot) blue
    secondary_color = "#d1272e"
    primary_color = "#2e6f95"

    if show_points:
        _plot_sample_points(ax, lobe, disc, R1, phase, incl, n_sec, n_disc, n_primary,
                             secondary_color, primary_color)

    primary_center = np.array([lobe.x1, 0.0, 0.0])
    Xs, Ys = lobe_outline(lobe, phase, incl)
    Xd, Yd, vis_d = disc_outline(disc, phase, incl, lobe=lobe,
                                  primary_center=primary_center, R1=R1)
    Xst, Yst, vis_st = stream_outline(traj, lobe.x1, phase, incl, disc=disc, lobe=lobe,
                                       primary_center=primary_center, R1=R1)
    Xg, Yg, vis_g = primary_visibility_grid(primary_center, R1, phase, incl, disc=disc, lobe=lobe)

    style = dict(lw=1.0)
    ax.plot(Xs, Ys, "-", color=secondary_color, label="secondary", **style)
    _hidden_split_plot(ax, Xd, Yd, vis_d, "#e08214", lw=style["lw"], label="accretion disc")
    for Xe, Ye, vis_e in disc_cross_section_lines(disc, phase, incl, lobe=lobe,
                                                   primary_center=primary_center, R1=R1):
        _hidden_split_plot(ax, Xe, Ye, vis_e, "#e08214", lw=style["lw"])
    if disc.e > 0.0:
        Xpa, Ypa, _ = project(_periastron_line(disc), phase, incl)
        ax.plot(Xpa, Ypa, ":", color="#e08214", lw=0.6, alpha=0.5, label="periastron")
    _hidden_split_plot(ax, Xst, Yst, vis_st, "#1b9e77", lw=style["lw"], label="accretion stream")
    if stream_impact is not None:
        Xse, Yse, vis_se = stream_impact_ellipse_outline(
            stream_impact["x"], stream_impact["y"], stream_impact["vx"], stream_impact["vy"],
            stream_impact["H"], stream_impact["W"], phase, incl, disc,
            lobe=lobe, primary_center=primary_center, R1=R1)
        _hidden_split_plot(ax, Xse, Yse, vis_se, "#1b9e77", lw=style["lw"],
                            label="stream cross-section (Lubow & Shu)")
    if stream_begin is not None:
        Xsb, Ysb, vis_sb = stream_cross_section_outline(
            stream_begin["x"], stream_begin["y"], stream_begin["vx"], stream_begin["vy"],
            stream_begin["H"], stream_begin["W"], phase, incl, disc,
            lobe=lobe, primary_center=primary_center, R1=R1)
        _hidden_split_plot(ax, Xsb, Ysb, vis_sb, "#1b9e77", lw=style["lw"])
    if stream_end is not None:
        Xsn, Ysn, vis_sn = stream_cross_section_outline(
            stream_end["x"], stream_end["y"], stream_end["vx"], stream_end["vy"],
            stream_end["H"], stream_end["W"], phase, incl, disc,
            lobe=lobe, primary_center=primary_center, R1=R1)
        _hidden_split_plot(ax, Xsn, Ysn, vis_sn, "#1b9e77", lw=style["lw"])
    stream_tube = None
    if stream_eps is not None and len(traj["x"]) > 1:
        # no occlusion test (see stream_tube_outline's own docstring) --
        # a plain, unconditional pair of curves, same convention as the
        # periastron reference line just above (ax.plot, not
        # _hidden_split_plot).
        X1t, Y1t, X2t, Y2t = stream_tube_outline(traj["x"], traj["y"], lobe, stream_eps,
                                                   phase, incl)
        ax.plot(X1t, Y1t, "-", color="#1b9e77", lw=style["lw"] * 0.7, alpha=0.6)
        ax.plot(X2t, Y2t, "-", color="#1b9e77", lw=style["lw"] * 0.7, alpha=0.6,
                label="stream boundary")
        stream_tube = (X1t, Y1t, X2t, Y2t)
    # faint full limb circle, always drawn underneath (even when the
    # primary is entirely hidden, e.g. eclipsed behind the secondary or
    # the disc) -- the same "always-present faint guide" every other
    # body's outline gets from _hidden_split_plot's own unconditional
    # first pass (disc/stream/field lines above); primary_visibility_grid
    # can't reuse that helper directly (it needs a 2D contour, not a 1D
    # curve, to correctly trace a possibly-complex partial-occlusion
    # shape -- e.g. the disc cutting a chord across the primary's face,
    # not just its outer limb), so this circle is drawn separately, at
    # this same faint styling every other body's own unconditional first
    # pass uses by default (primary_faint_lw_scale/primary_faint_alpha,
    # see this function's own docstring for why a caller might raise
    # them), with the solid contour below drawn on top of it for
    # whatever's actually visible.
    _, Xp_full, Yp_full = circle_outline(primary_center, R1, phase, incl, n=100)
    ax.plot(Xp_full, Yp_full, color=primary_color,
            lw=style["lw"] * primary_faint_lw_scale, alpha=primary_faint_alpha)
    vis_f = vis_g.astype(float)
    if vis_f.any():
        ax.contour(Xg, Yg, vis_f, levels=[0.5], colors=[primary_color], linewidths=style["lw"])
    # proxy artist so "primary" still gets a legend entry (contour objects
    # don't hand matplotlib a legend-friendly handle the way plot() does)
    ax.plot([], [], "-", color=primary_color, label="primary", **style)

    # dipole field-line loops (geometry only -- see magnetic.py): very
    # thin blue dotted, one legend entry for the whole family.
    field_curves = field_line_outlines(primary_center, R1, theta_1, phi_1, n_field_1,
                                        phase, incl, disc=disc, lobe=lobe)
    for i, (Xf, Yf, vis_f_line) in enumerate(field_curves):
        _hidden_split_plot(ax, Xf, Yf, vis_f_line, "blue", lw=0.4, ls=":",
                            label="field lines" if i == 0 else None)

    # magnetic north/south poles (where the magnetic axis m_hat meets the
    # primary's own surface): same "field lines shown" gate as above. A
    # pole is plotted only if actually visible -- a back-face test (on the
    # primary's own far side, self-occluded, same as
    # primary_visibility_grid's contour) plus the disc/Roche-lobe tests
    # every other outline feature already uses -- nothing drawn at all
    # otherwise, unlike the curves above's faint-when-hidden treatment.
    if n_field_1 > 0:
        pole_dirs, pole_pts = _magnetic_pole_points(primary_center, R1, theta_1, phi_1)
        n_hat_pole, _, _ = observer_frame(phase, incl)
        vis_pole = ((np.einsum("ij,j->i", pole_dirs, n_hat_pole) > 0.0)
                    & visible_mask_bulk(pole_pts, phase, incl, lobe)
                    & disc.visible(pole_pts, phase, incl))
        Xp, Yp, _ = project(pole_pts, phase, incl)
        ax.plot(Xp[vis_pole], Yp[vis_pole], "o", color="blue", ms=2.5, label="magnetic poles")

    # accretion-spot connection(s) (stream -> field line -> primary
    # surface, see render.build_temperature_maps'/--angle_acc's own
    # docstrings): solid red, same occlusion tests as the field lines
    # above; one legend entry for the whole family, same as those.
    n_hat_acc, _, _ = observer_frame(phase, incl)
    for i, line in enumerate(accretion_field_lines):
        vis_acc = (visible_mask_bulk(line, phase, incl, lobe)
                   & disc.visible(line, phase, incl)
                   & ~_occluded_by_sphere(line, primary_center, R1, n_hat_acc))
        Xa, Ya, _ = project(line, phase, incl)
        _hidden_split_plot(ax, Xa, Ya, vis_acc, "red", lw=style["lw"],
                            label="accretion spot" if i == 0 else None)

    # faint reference lines: where the orbital midplane (z=0) actually
    # cuts through each object (the stream already lies exactly in that
    # plane by construction, so its outline above already IS this trace
    # for the stream); hidden portions of these get weakened further
    # still by _hidden_split_plot, same as the main outlines above.
    faint_lw, faint_alpha = 0.7, 0.5
    # the secondary's own midplane cut sits entirely within its own
    # silhouette, so unlike the other bodies' reference lines (which cross
    # in and out of other objects' solids) a hidden/visible split here
    # only makes the line look inconsistently dashed against itself, with
    # no other object for the "hidden" treatment to explain -- draw it as
    # one uniform dash throughout instead.
    Xsm, Ysm, _vis_sm = lobe_midplane_line(lobe, phase, incl)
    ax.plot(Xsm, Ysm, color=secondary_color, lw=faint_lw, alpha=faint_alpha, dashes=(4, 2))
    Xdm, Ydm, vis_dm = disc_midplane_line(disc, phase, incl, lobe=lobe,
                                           primary_center=primary_center, R1=R1)
    _hidden_split_plot(ax, Xdm, Ydm, vis_dm, "#e08214", lw=faint_lw, alpha=faint_alpha, ls="--")
    Xdi, Ydi, vis_di = disc_inner_rim_line(disc, phase, incl, lobe=lobe,
                                            primary_center=primary_center, R1=R1)
    _hidden_split_plot(ax, Xdi, Ydi, vis_di, "#e08214", lw=faint_lw, alpha=faint_alpha, ls="--")
    Xpm, Ypm, vis_pm = primary_midplane_line(primary_center, R1, phase, incl, disc=disc, lobe=lobe)
    _hidden_split_plot(ax, Xpm, Ypm, vis_pm, primary_color, lw=faint_lw, alpha=faint_alpha, ls="--")

    # primary Roche lobe's own midplane cut (mirror-derived, see
    # primary_lobe_midplane_line) as a simple reference line, drawn at
    # uniform strength -- no occlusion/back-face treatment applies to it,
    # and no projected silhouette (that convex hull can extend far beyond
    # the disc for extreme mass ratios and blows out the plot's autoscale).
    Xs1m, Ys1m = primary_lobe_midplane_line(lobe, phase, incl)
    ax.plot(Xs1m, Ys1m, "--", color="black", lw=faint_lw, alpha=faint_alpha)

    # the secondary's own FULL (fill_factor=1) Roche lobe midplane cut
    # (see full_lobe_midplane_line) -- same role as primary_lobe_midplane_line
    # just above (a purely geometric reference, independent of the
    # secondary's actual current shape, drawn the same uniform-strength
    # way with no occlusion/back-face treatment), useful now that the
    # secondary's own midplane line (Xsm/Ysm above) may be smaller if
    # it's underfilling its lobe.
    Xsfm, Ysfm = full_lobe_midplane_line(lobe, phase, incl)
    ax.plot(Xsfm, Ysfm, "--", color=secondary_color, lw=faint_lw, alpha=faint_alpha)

    # "+" markers: secondary's point mass and the system center of mass
    # (always at the origin in this corotating, COM-centered frame)
    Xcm2, Ycm2, _ = project(np.array([lobe.x2, 0.0, 0.0]), phase, incl)
    Xcom, Ycom, _ = project(np.array([0.0, 0.0, 0.0]), phase, incl)
    ax.plot(Xcm2, Ycm2, "+", color=secondary_color, markersize=6, markeredgewidth=style["lw"])
    ax.plot(Xcom, Ycom, "+", color="black", markersize=6, markeredgewidth=style["lw"])

    if fill:
        def fill_stream_begin():
            ax.fill(Xsb, Ysb, facecolor="#1b9e77", edgecolor="none", alpha=0.15)

        def fill_stream_impact():
            ax.fill(Xse, Yse, facecolor="#1b9e77", edgecolor="none", alpha=0.15)

        def fill_stream_end():
            ax.fill(Xsn, Ysn, facecolor="#1b9e77", edgecolor="none", alpha=0.15)

        def fill_stream():
            # X1t/X2t (stream_tube_outline's own two flanking curves)
            # filled as one ribbon ("forward" + "reversed" side, same
            # concatenation trick as fill_disc_and_primary's annulus
            # below, just without needing _ensure_ccw first -- a ribbon
            # between two open curves has no inherent winding ambiguity
            # the way a closed annulus does). Same light green as the
            # "shadow" output's own stream-width band
            # (plot_topdown_shadows, facecolor "#1b9e77", alpha 0.15).
            ax.fill(np.concatenate([X1t, X2t[::-1]]), np.concatenate([Y1t, Y2t[::-1]]),
                    facecolor="#1b9e77", edgecolor="none", alpha=0.15)

        def fill_secondary():
            ax.fill(Xs, Ys, color=secondary_color, alpha=0.15)

        def fill_disc_and_primary():
            # Xd/Yd (outer rim -- disc_outline's flat-disc parametric
            # sweep, or its flared-disc convex-hull silhouette) and the
            # disc's own actual "hole" boundary (disc_inner_wall_outline
            # -- the inner wall's OWN convex-hull silhouette, spanning its
            # full vertical extent for a flared disc, not the flat
            # midplane circle Xdi/Ydi above -- that one stays exactly the
            # separate faint reference line it's documented as) filled as
            # one "outer forward + inner reversed" path so the hole
            # between them is left unfilled instead of painted over (same
            # trick as plot_topdown_shadows' disc annulus) -- _ensure_ccw
            # first, since the two loops' relative winding otherwise isn't
            # guaranteed consistent (see its own docstring), which would
            # silently double-fill the hole instead of leaving it empty.
            # Gated on the SAME "len(Xdi)>0" (disc.r_in>R1, see disc_
            # inner_rim_line) check already computed above: when that's
            # empty there's no distinct hole at all, the primary's own
            # fill (below) already covers that region, so just fill
            # Xd/Yd whole.
            Xd_ccw, Yd_ccw = _ensure_ccw(Xd, Yd)
            if len(Xdi) > 0:
                Xdih, Ydih = disc_inner_wall_outline(disc, phase, incl)
                Xdih_ccw, Ydih_ccw = _ensure_ccw(Xdih, Ydih)
                ax.fill(np.concatenate([Xd_ccw, Xdih_ccw[::-1]]),
                        np.concatenate([Yd_ccw, Ydih_ccw[::-1]]),
                        color="#e08214", alpha=0.12)
            else:
                ax.fill(Xd_ccw, Yd_ccw, color="#e08214", alpha=0.12)
            # disc's own stream-impact hot spot (see this function's
            # docstring, disc_hotspot_cells): drawn here, with the rest
            # of the disc's fill, so it shares the same back-to-front
            # ordering against the secondary below. disc_hotspot_cells
            # itself drops any cell occluded by the disc's own far side,
            # the secondary's Roche lobe, or the primary star -- passing
            # lobe/primary_center/R1 through is what makes that filtering
            # active here (plot_topdown_shadows' own face-on view leaves
            # them unpassed, since that view has no occlusion by design).
            if (hotspot_phi_h is not None and hotspot_L_h_deg is not None
                    and hotspot_dphi_max_deg is not None and hotspot_dphi_max_deg > 0.0):
                from matplotlib.collections import PolyCollection
                hs_verts, hs_colors = disc_hotspot_cells(
                    disc, hotspot_phi_h, hotspot_L_h_deg, hotspot_dphi_max_deg, phase, incl,
                    footprint=hotspot_footprint, lobe=lobe, primary_center=primary_center, R1=R1)
                if hs_verts:
                    ax.add_collection(PolyCollection(hs_verts, facecolors=hs_colors,
                                                      edgecolors="none"))
                    # PolyCollection doesn't register a legend entry on its
                    # own (no single facecolor to show) -- an invisible
                    # proxy patch stands in for it.
                    ax.fill([], [], facecolor="green", alpha=0.5, label="hot spot")
            # primary's own fill is already occlusion-masked (vis_g
            # accounts for both disc and secondary), so its position
            # relative to the disc specifically doesn't matter -- only
            # relative to the secondary, handled by the back-to-front
            # ordering below.
            if vis_f.any():
                ax.contourf(Xg, Yg, vis_f, levels=[0.5, 1.5], colors=[primary_color], alpha=0.35)

        # back-to-front draw order (painter's algorithm) so whichever of
        # {secondary} vs {disc+primary, both centered on the primary} is
        # actually nearer the observer at this phase has its fill drawn
        # last, on top -- rather than the disc/primary fill unconditionally
        # covering the secondary's regardless of which one is truly in
        # front. depth (from eclipse.project) increases towards the
        # observer, same convention used everywhere else in this package.
        # the stream's fill is drawn first, underneath both -- its own
        # depth varies continuously all along its path (L1 to the disc),
        # unlike the two roughly-rigid bodies below, so there's no single
        # "nearer/farther" choice that would be correct throughout; this
        # at least gets the common case right, since the disc/primary
        # bodies' own opaque fills then naturally paint over whatever
        # part of the stream actually passes behind them.
        if stream_tube is not None:
            fill_stream()
        if stream_begin is not None:
            fill_stream_begin()
        if stream_impact is not None:
            fill_stream_impact()
        if stream_end is not None:
            fill_stream_end()

        _, _, depth_primary = project(primary_center, phase, incl)
        _, _, depth_secondary = project(np.array([lobe.x2, 0.0, 0.0]), phase, incl)
        if depth_secondary > depth_primary:
            fill_disc_and_primary()
            fill_secondary()
        else:
            fill_secondary()
            fill_disc_and_primary()

    ax.set_aspect("equal")
    ax.set_xlabel("X / a  (sky-plane)")
    ax.set_ylabel("Y / a  (sky-plane)")
    ax.set_title(labeled_title(
        label, rf"q={lobe.q:.4f}, i={incl_deg:.2f}$^\circ$, R$_1$={R1:.4f}, phase={phase:.4f}"))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.45), ncol=4,
              fontsize=9, framealpha=0.9)
    style_axes(ax)
    # NOTE: the legend sits outside the axes; save this figure with
    # fig.savefig(path, bbox_inches="tight") so it isn't clipped/overlapped.
    return fig, ax


def _lobe_limb_points_3d(lobe, phase, incl):
    """
    Like lobe_outline, but returns the actual (corotating-frame) 3D
    surface points at the projected silhouette's convex-hull vertices,
    rather than their projected 2D (X,Y) -- needed to extend the limb
    along the line of sight onto the orbital midplane, see
    secondary_shadow_on_midplane.
    """
    surf = lobe.surface_points()
    X, Y, _ = project(surf, phase, incl)
    hull = ConvexHull(np.column_stack([X, Y]))
    idx = np.append(hull.vertices, hull.vertices[0])
    return surf[idx]


def secondary_shadow_on_midplane(lobe, phase, incl):
    """
    The secondary's true limb (its projected silhouette at this orbital
    phase and inclination) extended along the observer's line of sight
    to wherever each limb point crosses the orbital midplane (z=0) --
    i.e. the footprint the secondary occults on the disc plane at this
    phase, the geometric mechanism behind the eclipse light curve's
    shape (see render.physical_light_curve).

    For limb point P, extending along the full line P + t*n_hat crosses
    z=0 at a single t = -P_z/n_hat_z regardless of sign -- but only
    t<0 is physically a "shadow on the disc": eclipse.project's
    convention has depth = dot(point, n_hat) increasing *towards* the
    observer, so a t<0 extension moves *away* from the observer (deeper
    into the scene, where disc material behind the star could actually
    sit); t>0 would place the "shadow" point nearer the observer than
    the star itself, which cannot be shadowed by it. Since the limb loop
    crosses z=0 (t=0) at exactly the two points where the sight line is
    tangent to the star's own equatorial rim, filtering to t<=0 keeps a
    single contiguous open arc between those two points -- the shadow
    extends in one direction from one side of the star, out across the
    midplane, and back to the star's other side, not a closed loop
    encircling it.

    Returns (x, y) in the corotating frame (an open arc, not closed), or
    None if the line of sight lies exactly in the orbital plane (i=90
    deg, edge-on -- n_hat has no z-component, so "extend to z=0" is
    degenerate) or no point of the limb satisfies t<=0.
    """
    n_hat, _, _ = observer_frame(phase, incl)
    if abs(n_hat[2]) < 1e-8:
        return None
    P = _lobe_limb_points_3d(lobe, phase, incl)
    t = -P[:, 2] / n_hat[2]
    keep = t <= 1e-9
    if not keep.any():
        return None
    if not keep.all():
        # P is a closed loop (first point repeated at the end); roll it
        # so index 0 is excluded, breaking the circular wrap into a
        # single contiguous run of kept points to slice out.
        start = np.argmax(~keep)
        order = (np.arange(len(keep)) + start) % len(keep)
        keep_rolled = keep[order]
        idx = np.nonzero(keep_rolled)[0]
        order = order[idx[0]:idx[-1] + 1]
        P, t = P[order], t[order]
    Q = P + t[:, None] * n_hat
    return Q[:, 0], Q[:, 1]


def _polygon_centroid(X, Y):
    """Centroid of a simple closed polygon given as open (X,Y) arrays
    (the closing edge from the last point back to the first is implicit,
    same convention as _ensure_ccw/the annulus-fill helpers) -- the
    standard shoelace-weighted formula, not just the vertices' mean
    (which is biased toward whichever side has denser points)."""
    Xc, Yc = np.append(X, X[0]), np.append(Y, Y[0])
    cross = Xc[:-1] * Yc[1:] - Xc[1:] * Yc[:-1]
    A = 0.5 * np.sum(cross)
    if abs(A) < 1e-300:
        return float(np.mean(X)), float(np.mean(Y))
    Cx = np.sum((Xc[:-1] + Xc[1:]) * cross) / (6.0 * A)
    Cy = np.sum((Yc[:-1] + Yc[1:]) * cross) / (6.0 * A)
    return float(Cx), float(Cy)


_lobe_hull_test_cache = {}


def _lobe_hull_test(lobe, phase, incl):
    """
    A fast (...,3)-points -> visible boolean test for a FIXED phase,
    equivalent to eclipse.visible_mask_bulk but computing the lobe's own
    projected convex hull only ONCE per (lobe, phase, incl) and caching
    it, since that hull computation (over the WHOLE lobe surface, see
    visible_mask_bulk's own docstring) is by far its dominant per-call
    cost -- independent of how many points are actually being tested,
    so calling visible_mask_bulk directly once per bisection iteration
    (as shadow_boundary_radial's occlusion-based root-finding does, many
    times over for the very same handful of phases) recomputes an
    identical hull thousands of times for nothing. Not a general-purpose
    replacement for visible_mask_bulk elsewhere -- just local plumbing
    for this module's own repeated-same-phase bisection use, where the
    caching pays off.

    Keyed on the lobe's own CONTENT (q, fill_factor, ntheta, nphi) --
    everything surface_points() actually depends on -- not id(lobe):
    a fit that rebuilds a fresh RocheLobe every trial (e.g. simulate.py's
    shadow-mode/rim-projection-mode --lsq_fit/--mcmc_fit) constantly
    creates and discards lobe objects, and CPython is free to reuse a
    just-freed object's memory address for the next one -- id(lobe) can
    silently collide between two DIFFERENT (different q!) lobes across
    such a loop, returning a stale, wrong-geometry cached hull for the
    new one with no error at all. Content keying makes that impossible:
    two lobes with the same q/fill_factor/resolution really do have the
    same surface, so sharing the cached hull between them is correct
    (not just accidentally collision-free), and different q always gets
    its own entry.
    """
    key = (round(float(lobe.q), 12), round(float(lobe.fill_factor), 12),
           lobe._ntheta, lobe._nphi, float(phase), float(incl))
    test = _lobe_hull_test_cache.get(key)
    if test is not None:
        return test
    n_hat, _, _ = observer_frame(phase, incl)
    if np.dot(n_hat, _L1_DIR) > 0.0:
        def test(pts):
            return np.ones(np.asarray(pts).shape[:-1], dtype=bool)
    else:
        surf = lobe.surface_points()
        Xs, Ys, _ = project(surf, phase, incl)
        hull = ConvexHull(np.column_stack([Xs, Ys]))
        poly = Path(np.column_stack([Xs, Ys])[np.append(hull.vertices, hull.vertices[0])])

        def test(pts):
            pts = np.asarray(pts, dtype=float)
            lead_shape = pts.shape[:-1]
            X, Y, _ = project(pts.reshape(-1, 3), phase, incl)
            inside = poly.contains_points(np.column_stack([X, Y])).reshape(lead_shape)
            return ~inside
    _lobe_hull_test_cache[key] = test
    return test


def shadow_boundary_radial(lobe, phase, incl, primary_x, theta=None,
                            n_theta=720, r_max=1.0, n_iter=50, n_scan=32):
    """
    The secondary's shadow boundary at a given phase, as r(theta) --
    radius from the primary at (primary_x, 0) -- found by bisection along
    each ray in `theta` against the real occlusion test (eclipse.
    visible_mask_bulk, via the locally-cached _lobe_hull_test), the same
    one the eclipse light curve itself uses.

    Deliberately NOT built from secondary_shadow_on_midplane's own open
    reference curve: that one is explicitly documented as "not a closed
    loop encircling" the shadow (just a visual guide for plot_topdown_
    shadows), and empirically does not bound the actual occluded region
    at all -- the real shadow, verified directly against
    visible_mask_bulk, covers points nowhere near that curve's own
    coordinate range. A uniform-grid boolean occlusion mask (the more
    obvious alternative) was also tried and rejected: the region swept
    between two close-in-phase eclipse contacts can be far thinner than
    even a very fine grid reliably samples, so it can be missed
    completely by chance. Bisecting along rays instead finds the
    boundary by root-finding, so it can't be missed regardless of how
    thin the swept region between two phases turns out to be.

    theta: explicit ray angles to bisect along, e.g. a single-element
    array for one arbitrary azimuth (see shadow_constrained_regions'
    own scalar re-evaluations of this at exact corner/edge thetas) --
    default (None) is an n_theta-point sweep of the full circle.

    A ray can cross the shadow boundary more than once (near AND far
    side of a silhouette that doesn't contain the primary) -- naively
    bisecting straight between r=1e-6 and r_max, comparing only those
    two endpoints' visibility, silently finds nothing whenever r_max
    happens to land back on the SAME visibility as the near end (an
    even number of crossings in between, invisible to a two-point
    check), and can converge on the wrong one otherwise. So each ray is
    first scanned at n_scan points to bracket the FIRST (innermost)
    crossing specifically, then that bracket alone is bisected -- this
    also means r_max can safely be generous without a real near-disc
    crossing going missing just because some other, farther crossing
    also exists.

    Returns (theta, r): r is NaN at any theta where no transition was
    found within [0, r_max] (that ray is visible, or hidden, throughout).
    """
    if theta is None:
        theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    ct, st = np.cos(theta), np.sin(theta)

    test = _lobe_hull_test(lobe, phase, incl)

    def vis_at(r):
        pts = np.stack([primary_x + r * ct, r * st, np.zeros_like(theta)], axis=-1)
        return test(pts)

    r_scan = np.linspace(1e-6, r_max, n_scan)
    vis_scan = np.stack([vis_at(np.full_like(theta, r)) for r in r_scan], axis=0)
    vis0 = vis_scan[0]
    flips = vis_scan != vis0[np.newaxis, :]
    has_crossing = flips.any(axis=0)
    first = np.argmax(flips, axis=0)  # first True along axis 0; meaningless where has_crossing is False
    lo = np.where(has_crossing, r_scan[np.maximum(first - 1, 0)], 1e-6)
    hi = np.where(has_crossing, r_scan[first], r_max)
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        matches0 = vis_at(mid) == vis0
        lo = np.where(matches0, mid, lo)
        hi = np.where(matches0, hi, mid)
    return theta, np.where(has_crossing, 0.5 * (lo + hi), np.nan)


def shadow_boundary_on_rim(disc, lobe, phase, incl, nu, n_iter=50, n_scan=8):
    """
    The secondary's shadow boundary as it crosses the disc's own outer
    RIM WALL (the vertical surface at r=rim(nu), z=-h(nu)..+h(nu) with
    h=rim(nu)*tan(opening_angle) -- the same wall disc_cross_section_lines/
    disc_hotspot_cells draw/paint on) at a given phase, as z(nu) --
    height above the midplane where visibility transitions -- found
    exactly the way shadow_boundary_radial finds its own r(theta):
    bisection against the real occlusion test (the locally-cached
    _lobe_hull_test), here sweeping z at fixed nu instead of r at fixed
    theta. Same multi-point first-crossing scan too, for the same
    reason (a column can cross more than once in principle, and z's own
    range is tiny -- h(nu) -- compared to shadow_boundary_radial's r_max,
    so missing the one real crossing by only checking the two endpoints
    would be an even easier mistake to make here).

    nu: rim azimuths [rad, measured from +x through the primary] to
    sweep -- explicit, not an n_nu/None default, since callers (e.g.
    simulate.py's "rim projection" output) need this concentrated
    finely over just the narrow azimuthal window actually relevant, not
    spread thinly over the whole circle.

    Returns z (same shape as nu): NaN wherever no transition was found
    within [-h(nu), +h(nu)] (that whole wall column is visible, or
    hidden, throughout). All-NaN if disc.opening_angle<=0 (no wall to
    speak of -- see disc_cross_section_lines' own matching guard).
    """
    nu = np.asarray(nu, dtype=float)
    if disc.opening_angle <= 0.0:
        return np.full_like(nu, np.nan)
    r = disc.rim(nu)
    x = disc.x1 + r * np.cos(nu)
    y = r * np.sin(nu)
    h = r * np.tan(disc.opening_angle)

    test = _lobe_hull_test(lobe, phase, incl)

    def vis_at(z):
        pts = np.stack([x, y, z], axis=-1)
        return test(pts)

    frac = np.linspace(-1.0, 1.0, n_scan)
    z_scan = frac[:, None] * h[None, :]
    vis_scan = np.stack([vis_at(z_scan[i]) for i in range(n_scan)], axis=0)
    vis0 = vis_scan[0]
    flips = vis_scan != vis0[np.newaxis, :]
    has_crossing = flips.any(axis=0)
    first = np.argmax(flips, axis=0)
    cols = np.arange(len(nu))
    lo = np.where(has_crossing, z_scan[np.maximum(first - 1, 0), cols], -h)
    hi = np.where(has_crossing, z_scan[first, cols], h)
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        matches0 = vis_at(mid) == vis0
        lo = np.where(matches0, mid, lo)
        hi = np.where(matches0, hi, mid)
    return np.where(has_crossing, 0.5 * (lo + hi), np.nan)


def _line_intersection(p1, p2, p3, p4):
    """
    Exact intersection of the infinite line through p1,p2 and the
    infinite line through p3,p4 (Cramer's rule on the 2x2 linear
    system) -- used here to intersect two ADJACENT-SAMPLE CHORDS of two
    different shadow-boundary curves, i.e. shadow_constrained_regions'
    own "each corner is the intersection of two shadow lines"
    construction, applied directly to their (x,y) points rather than
    via iterative root-finding against the occlusion test: a single
    closed-form solve per corner, exact regardless of how coarse the
    theta sampling that bracketed it was (given two DIFFERENT curves at
    a real crossing, p1-p2 and p3-p4 are never parallel, so D==0 is not
    handled specially here).
    """
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = p1, p2, p3, p4
    D = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / D
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / D
    return px, py


def shadow_curve_crossings(lobe, phase_a, phase_b, incl, primary_x,
                            n_theta=720, r_max=1.0, n_iter=50):
    """
    Every point where the secondary's shadow-boundary curves at phase_a
    and phase_b (shadow_boundary_radial) cross each other -- the same
    "intersection of two shadow lines" construction shadow_constrained_
    regions builds its own polygon corners from (coarse theta scan to
    bracket each sign change of r_a(theta)-r_b(theta), then the exact
    closed-form _line_intersection on the bracketing chords), just for
    one arbitrary pair of phases in isolation rather than the 4-phase
    combined hi/lo constraint -- e.g. for simulate.py's shadow-mode
    --lsq_fit/--mcmc_fit, which needs the raw 1st/3rd-shadow crossing
    itself, not a constrained-area polygon built around it.

    Returns a list of (x,y) corotating-frame points, one per crossing
    found (possibly empty, if the two curves' own domains -- see
    shadow_boundary_radial's own NaN-outside-domain convention -- never
    overlap, or overlap without actually crossing).
    """
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    ct, st = np.cos(theta), np.sin(theta)
    _, ra = shadow_boundary_radial(lobe, phase_a, incl, primary_x, theta=theta,
                                    r_max=r_max, n_iter=n_iter)
    _, rb = shadow_boundary_radial(lobe, phase_b, incl, primary_x, theta=theta,
                                    r_max=r_max, n_iter=n_iter)
    Xa, Ya = primary_x + ra * ct, ra * st
    Xb, Yb = primary_x + rb * ct, rb * st
    both = ~np.isnan(ra) & ~np.isnan(rb)
    diff = ra - rb

    points = []
    for k in range(n_theta):
        j = (k + 1) % n_theta
        if not (both[k] and both[j]):
            continue
        if np.sign(diff[k]) == np.sign(diff[j]):
            continue
        x, y = _line_intersection((Xa[k], Ya[k]), (Xa[j], Ya[j]),
                                   (Xb[k], Yb[k]), (Xb[j], Yb[j]))
        points.append((x, y))
    return points


def shadow_constrained_regions(lobe, ingress_phases, egress_phases, incl, primary_x,
                                n_theta=720, r_max=1.0, n_iter=50):
    """
    Where a point on the disc plane must be, given its own eclipse
    ingress AND egress contact phases -- derived PURELY from the
    secondary's own shadow boundary (shadow_boundary_radial); no disc or
    stream geometry enters this at all, by design. This is a general
    geometric constraint, not specific to any one feature -- a disc's
    stream-impact hot spot is the motivating example (and the one used
    to sanity-check it), but the same technique constrains any point
    whose own eclipse contact phases are known -- so the result is meant
    to be used to CONSTRAIN a disc's outer radius, or a stream's impact
    azimuth, not derived FROM an assumed one. `primary_x` (the one
    number this needs about "the system") is just the primary's own
    corotating-frame position (lobe.x1 == disc.x1) -- the natural center
    for an (r,theta) description of the disc plane, not a disc/stream
    assumption. `r_max` is a generous, likewise disc-independent default
    search radius (a fraction of the separation, well beyond any
    physically plausible disc for typical CV mass ratios) -- not the
    disc's own R_out.

    A single ingress (or egress) pair alone only brackets the point's
    radius AT WHICHEVER AZIMUTH IT ACTUALLY SITS AT -- an elongated,
    degenerate "banana" region along the shadow's advancing edge at that
    phase pair, not a small patch (radius and azimuth trade off against
    each other along it). Combining BOTH an ingress and an egress pair
    resolves that degeneracy: at each azimuth theta, the four phases'
    own shadow-boundary radii r1(theta)..r4(theta) (one per phase) give
    hi(theta) = min(max(r1,r2), max(r3,r4)) and
    lo(theta) = max(min(r1,r2), min(r3,r4)) -- the interval [lo,hi]
    (where hi>lo) is exactly where both pairs' own constraints hold at
    once. This is the same "2 pairs of eclipse times... to define the
    semi-rectangular areas that overlap" idea the technique started
    from, just as per-azimuth radius intervals (all four curves share
    the same primary-centered (r,theta) parametrization) rather than a
    general 2D polygon intersection.

    Exact corners, regardless of sampling: hi(theta) and lo(theta) are
    each, at every theta, EXACTLY one of the four raw curves r1..r4 (the
    binding one there) -- so every polygon vertex is a genuine
    intersection of two of those four curves (or of hi and lo meeting,
    at the two ends of the constrained azimuth range), not an
    approximation, and not a sample snapped to whichever one happens to
    be nearest. A coarse scan over n_theta rays only has to locate
    roughly where each such crossing falls -- between which two adjacent
    theta samples -- since each vertex is then found from THOSE TWO
    CURVES' OWN (x,y) points there via a single closed-form line-
    segment intersection (_line_intersection: two points from each
    curve in, the exact crossing of the (straight) chords through them
    out), the literal "intersection of two shadow lines" construction,
    not an iterative refinement against the occlusion test. So n_theta
    only has to be fine enough to bracket each crossing somewhere -- a
    handful of samples per degree is already far finer than any real
    crossing needs -- not to resolve its exact location, which the
    closed-form solve gives directly regardless of how wide that
    bracket was. The resulting polygon is the most PRIMITIVE constrained
    region: straight edges directly between consecutive true corners,
    not a fine polyline tracing each curve's own curvature in between.

    ingress_phases/egress_phases: each a (phase_before, phase_after)
    pair, phase_before < phase_after (ingress: "just before totality" ->
    "just after totality starts"; egress, pair reversed in time: "just
    before the point re-emerges" -> "just after").

    Returns a list of dicts, one per contiguous azimuth run where the
    two phase pairs' constraints overlap, each {"x": ndarray,
    "y": ndarray, "center": (cx, cy)} in the corotating frame's midplane
    (x,y) -- the same coordinates disc.rim/azimuth use, so directly
    comparable to an assumed disc's R_out or a stream's impact_azimuth.
    Empty if the two pairs never overlap anywhere (e.g. inconsistent
    timing, or a genuine model/data mismatch).
    """
    phases = (ingress_phases[0], ingress_phases[1], egress_phases[0], egress_phases[1])
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    ct, st = np.cos(theta), np.sin(theta)

    # each of the 4 raw shadow-boundary curves, computed ONCE (a single
    # vectorized call per phase, over all n_theta rays at once) -- no
    # further occlusion-test evaluations happen anywhere below.
    r_curves = [shadow_boundary_radial(lobe, ph, incl, primary_x, theta=theta,
                                        r_max=r_max, n_iter=n_iter)[1] for ph in phases]
    X_curves = [primary_x + r * ct for r in r_curves]
    Y_curves = [r * st for r in r_curves]

    r1, r2, r3, r4 = r_curves
    # A sample only constrains anything if ALL FOUR curves reach it --
    # if even one is NaN (that phase's ray never crosses within r_max at
    # this azimuth, i.e. it's outside that phase's own shadow's angular
    # extent), its pair's [lo,hi] is genuinely undefined there, not
    # recoverable from the other three. (Comparing against a NaN partner
    # below, e.g. r1 >= r2 with r2=NaN, is always False -- so a naive
    # np.where would silently fall back to treating the DEFINED one of a
    # pair as if it alone bounded that pair, which is wrong: it produces
    # a spuriously wide, physically meaningless "valid" stretch out at
    # each curve's own angular edge, not a genuine two-curve crossing.)
    all_def = ~np.isnan(r1) & ~np.isnan(r2) & ~np.isnan(r3) & ~np.isnan(r4)

    # hi/lo, plus WHICH of r1..r4 (index 0-3) each came from at every
    # sample -- needed to know which two curves' chords to intersect
    # wherever the winning one switches. Only meaningful where all_def.
    ing_hi_idx = np.where(r1 >= r2, 0, 1)
    ing_hi = np.maximum(r1, r2)
    egr_hi_idx = np.where(r3 >= r4, 2, 3)
    egr_hi = np.maximum(r3, r4)
    hi_from_ing = ing_hi <= egr_hi
    hi = np.where(hi_from_ing, ing_hi, egr_hi)
    hi_idx = np.where(hi_from_ing, ing_hi_idx, egr_hi_idx)

    ing_lo_idx = np.where(r1 <= r2, 0, 1)
    ing_lo = np.minimum(r1, r2)
    egr_lo_idx = np.where(r3 <= r4, 2, 3)
    egr_lo = np.minimum(r3, r4)
    lo_from_ing = ing_lo >= egr_lo
    lo = np.where(lo_from_ing, ing_lo, egr_lo)
    lo_idx = np.where(lo_from_ing, ing_lo_idx, egr_lo_idx)

    valid = all_def & (hi > lo)
    if not valid.any():
        return []

    def corner(a, b, i, j):
        # exact intersection of curve a's and curve b's chords over the
        # SAME two adjacent theta samples i,j (mod n_theta, so this also
        # works across the theta=0 wraparound) -- the literal
        # "intersection of two shadow lines" corner.
        i, j = i % n_theta, j % n_theta
        return _line_intersection(
            (X_curves[a][i], Y_curves[a][i]), (X_curves[a][j], Y_curves[a][j]),
            (X_curves[b][i], Y_curves[b][i]), (X_curves[b][j], Y_curves[b][j]))

    valid_ext = np.concatenate([[False], valid, [False]])
    edges = np.diff(valid_ext.astype(int))
    starts, ends = np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]

    regions = []
    for s, e in zip(starts, ends):
        last = e - 1  # index of this run's last valid sample

        # A run boundary should be a genuine pinch: hi meets lo while all
        # four curves are still defined just outside the run. If instead
        # one of the four curves' own angular extent ends right there
        # (all_def false just outside), r_max wasn't generous enough to
        # resolve where this run's own true pinch actually is -- skip
        # rather than fabricate a corner that isn't really a crossing of
        # two curves (see shadow_boundary_radial's own r_max discussion).
        if not all_def[(s - 1) % n_theta] or not all_def[e % n_theta]:
            continue

        X, Y = [], []
        cx, cy = corner(hi_idx[s], lo_idx[s], s - 1, s)
        X.append(cx); Y.append(cy)
        for k in range(s, last):
            if hi_idx[k] != hi_idx[k + 1]:
                cx, cy = corner(hi_idx[k], hi_idx[k + 1], k, k + 1)
                X.append(cx); Y.append(cy)
        cx, cy = corner(hi_idx[last], lo_idx[last], last, e)
        X.append(cx); Y.append(cy)
        for k in range(last, s, -1):
            if lo_idx[k] != lo_idx[k - 1]:
                cx, cy = corner(lo_idx[k], lo_idx[k - 1], k, k - 1)
                X.append(cx); Y.append(cy)

        X, Y = np.array(X), np.array(Y)
        regions.append({"x": X, "y": Y, "center": _polygon_centroid(X, Y)})
    return regions


def plot_disc_rim_projection(lobe, disc, ingress_phases, egress_phases, incl_deg,
                              T_2, P_orb, a_m, ax=None, label=None,
                              n_nu=2000, window_pad=2.5, phase_error=None):
    """
    The secondary's 4-phase shadow (see shadow_constrained_regions'
    own ingress_phases/egress_phases) projected onto the disc's own
    OUTER RIM WALL -- "unrolled" into a flat (x,y) = (distance along the
    rim, height above the midplane) strip, rather than shadow_
    constrained_regions/plot_topdown_shadows' own flat-midplane (x,y).
    Useful precisely because the midplane view can't show the rim
    wall's own finite height at all -- here that's the y-axis.

    x is true arc length along disc.rim(nu) (not nu*R_out -- exact even
    for an eccentric disc, via numerical integration of
    ds/dnu = sqrt(rim(nu)^2 + (drim/dnu)^2)), referenced to zero at the
    stream's own disc-impact azimuth (stream.disc_impact_point) if the
    stream reaches this disc, else to the plotted window's own left
    edge. y is height above the midplane, z -h(nu)..+h(nu) with
    h=rim(nu)*tan(opening_angle) -- disc_cross_section_lines/
    disc_hotspot_cells' own wall extent, drawn here too as a faint
    reference pair of curves.

    Only a narrow azimuthal WINDOW around where something actually
    happens is swept (not the full circle) -- the 4-phase constrained
    area's own azimuthal span (shadow_constrained_regions, converting
    each vertex back to its own nu) padded by `window_pad`, folded in
    with the impact azimuth if there is one -- both so n_nu points buys
    much finer effective resolution there than spreading them over 360
    degrees would, and so the plot itself only ever shows the region
    where there's something to see. Nothing plotted (a placeholder
    message instead) if the disc has no wall (opening_angle<=0) or
    there's neither a shadow overlap nor a stream impact to center the
    window on at all.

    T_2/P_orb/a_m: SystemParams' own native units, for the stream's
    Lubow & Shu eps (stream.lubow_shu_eps) -- needed for both the
    impact-point window-centering above and the cross-section ellipse
    below.

    At the impact point (if any), the stream's own cross-section
    ellipse (stream.lubow_shu_stream_size's H,W -- see plot_component_
    outlines' stream_impact_ellipse_outline for the same ellipse in 3D)
    is drawn as it actually lands on this FLAT unrolled wall: its
    vertical extent H is unchanged (both the ellipse's own plane and the
    wall share the vertical direction exactly), but its horizontal
    extent widens to W/|cos(incidence)|, incidence being the angle
    between the stream's own velocity there and the wall's LOCAL
    outward normal (radial) direction -- a grazing impact (velocity
    nearly tangential to the rim) spreads the footprint much wider than
    a head-on one (velocity nearly radial) does; see this function's own
    derivation comment below for why this projection is exactly an
    ellipse, not some more complicated conic.

    label: simulate.py's --prefix global label (see labeled_title).

    phase_error: a length-4 sequence [cycles] (simulate.py's own
    --phase_error, already resolved against these 4 phases -- see
    simulate._resolve_phase_error, one entry per phase in ingress_phases
    +egress_phases order), if given: each of the 4 solid ingress/egress
    curves below also gets two faint dashed flanking curves, at that
    same phase +- its OWN phase_error entry -- the same visual timing-
    uncertainty band simulate.py's own "primary" output draws around its
    own 4 curves, here reusing shadow_boundary_on_rim (at phase+-
    phase_error[i] instead of phase) over the identical nu_dense/s_dense
    grid already built for the solid curve, rather than a separate
    computation.
    """
    import matplotlib.pyplot as plt
    from stream import disc_impact_point, lubow_shu_eps, lubow_shu_stream_size

    incl = np.radians(incl_deg)
    if ax is None:
        fig, ax = plt.subplots(figsize=(9.0, 4.0))
    else:
        fig = ax.figure

    if disc.opening_angle <= 0.0:
        ax.text(0.5, 0.5, "disc has no wall (opening_angle=0) -- nothing to project",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title(labeled_title(
            label, rf"q={lobe.q:.4f}, i={incl_deg:.1f}$^\circ$, R_out={disc.a:.4f}, rim projection"))
        style_axes(ax)
        return fig, ax

    phases = (ingress_phases[0], ingress_phases[1], egress_phases[0], egress_phases[1])
    primary_x = disc.x1

    eps = lubow_shu_eps(T_2, P_orb, a_m)
    impact = disc_impact_point(lobe, disc, eps=eps)
    if impact is not None:
        nu_imp, cos_incidence_imp = impact_incidence(impact, disc)
    else:
        nu_imp, cos_incidence_imp = None, None

    # window: every vertex of the 4-phase constrained area(s) (converted
    # back to its own azimuth), PLUS the raw 1st/3rd-shadow crossing(s)
    # (shadow_curve_crossings) -- not redundant with the constrained
    # area's own vertices: which pair of curves actually forms each of
    # THAT polygon's corners depends on the full 4-curve hi/lo
    # combinatorics (see shadow_constrained_regions' own docstring), so
    # the 1st/3rd crossing specifically (the one simulate.py's shadow-
    # mode --lsq_fit/--mcmc_fit targets, see its own docstring) isn't
    # guaranteed to still be one of those vertices for an arbitrary
    # q/incl -- but it's exactly the point this plot exists to show
    # alongside the stream, so it must be in the window regardless.
    # Plus the impact azimuth itself -- "where there's an effect" (see
    # this function's own docstring).
    regions = shadow_constrained_regions(lobe, ingress_phases, egress_phases, incl, primary_x)
    nus = [np.arctan2(y, x - primary_x) for region in regions
           for x, y in zip(region["x"], region["y"])]
    crossings_13 = shadow_curve_crossings(lobe, phases[0], phases[2], incl, primary_x)
    if crossings_13:
        if impact is not None:
            # keep only the crossing nearest the stream impact -- the
            # physically relevant one; an arbitrary q/incl can give the
            # two curves spurious further-out crossings too, which would
            # otherwise blow the window out to no purpose.
            d2 = [(x - impact["x"]) ** 2 + (y - impact["y"]) ** 2 for x, y in crossings_13]
            x13, y13 = crossings_13[int(np.argmin(d2))]
            nus.append(np.arctan2(y13, x13 - primary_x))
        else:
            nus.extend(np.arctan2(y, x - primary_x) for x, y in crossings_13)
    if nu_imp is not None:
        nus.append(nu_imp)
    if not nus:
        ax.text(0.5, 0.5, "no shadow overlap or stream impact found -- nothing to project",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title(labeled_title(
            label, rf"q={lobe.q:.4f}, i={incl_deg:.2f}$^\circ$, R_out={disc.a:.4f}, rim projection"))
        style_axes(ax)
        return fig, ax

    nu_arr = np.unwrap(np.array(nus))
    nu_c = 0.5 * (nu_arr.min() + nu_arr.max())
    nu_half = max(0.5 * (nu_arr.max() - nu_arr.min()), np.radians(2.0)) * window_pad
    nu_dense = np.linspace(nu_c - nu_half, nu_c + nu_half, n_nu)

    # true arc length along the rim (exact for an eccentric rim too),
    # zeroed at the impact azimuth if there is one
    r_dense = disc.rim(nu_dense)
    drdnu = np.gradient(r_dense, nu_dense)
    ds_dnu = np.sqrt(r_dense ** 2 + drdnu ** 2)
    s_cum = np.concatenate([[0.0],
                             np.cumsum(0.5 * (ds_dnu[:-1] + ds_dnu[1:]) * np.diff(nu_dense))])
    s0 = np.interp(nu_imp, nu_dense, s_cum) if nu_imp is not None else 0.0
    s_dense = s_cum - s0

    h_dense = r_dense * np.tan(disc.opening_angle)
    ax.plot(s_dense, h_dense, "-", color="#e08214", lw=1.0, alpha=0.5, label="rim wall edge")
    ax.plot(s_dense, -h_dense, "-", color="#e08214", lw=1.0, alpha=0.5)
    ax.axhline(0.0, color="gray", ls=":", lw=0.8, alpha=0.5)

    colors = ["#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
    curve_labels = ["ingress before", "ingress after", "egress before", "egress after"]
    for i, (ph, color, lbl) in enumerate(zip(phases, colors, curve_labels)):
        z = shadow_boundary_on_rim(disc, lobe, ph, incl, nu_dense)
        ax.plot(s_dense, z, "-", color=color, lw=1.2, label=lbl)
        if phase_error is not None:
            for j, sign in enumerate((-1.0, 1.0)):
                z2 = shadow_boundary_on_rim(disc, lobe, ph + sign * phase_error[i], incl, nu_dense)
                ax.plot(s_dense, z2, "--", color=color, lw=0.6, alpha=0.6,
                        label="+/- phase_error" if i == 0 and j == 0 else None)

    if impact is not None:
        H_ls, W_ls = lubow_shu_stream_size(impact["r1"], lobe.q, eps)
        # The stream's cross-section ellipse (perp_hat: W, z_hat: H, in
        # the plane perpendicular to its own velocity v_hat, which is
        # purely horizontal -- see stream_impact_ellipse_outline) meets
        # the wall's own flat (t_hat, z_hat) plane (t_hat: horizontal,
        # tangent to the rim; z_hat shared with the ellipse's own plane)
        # along the curve t(psi) = W*cos(psi)*[perp_hat.t_hat -
        # (perp_hat.n_hat)*(v_hat.t_hat)/(v_hat.n_hat)], z(psi) =
        # H*sin(psi) (z unaffected, since z_hat is common to both
        # planes and everything else in it is horizontal). Writing
        # v_hat = cos(a)*n_hat + sin(a)*t_hat (a = incidence angle from
        # the radial/normal direction) and perp_hat as v_hat rotated 90
        # degrees in-plane reduces that bracket to exactly 1/cos(a) --
        # i.e. this is still a plain ellipse, just W stretched to
        # W/|cos(a)|, not some more general conic.
        W_eff = W_ls / cos_incidence_imp
        psi = np.linspace(0.0, 2.0 * np.pi, 200)
        Xcs, Ycs = W_eff * np.cos(psi), H_ls * np.sin(psi)
        ax.plot(Xcs, Ycs, "-", color="#1b9e77", lw=1.0,
                label="stream cross-section (Lubow & Shu)")
        ax.fill(Xcs, Ycs, facecolor="#1b9e77", edgecolor="none", alpha=0.15)

    ax.set_aspect("equal")
    ax.set_xlabel("distance along rim (units of a)")
    ax.set_ylabel("height along rim (units of a)")
    ax.set_title(labeled_title(
        label, rf"q={lobe.q:.4f}, i={incl_deg:.2f}$^\circ$, R_out={disc.a:.4f}, rim projection"))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=9, framealpha=0.9)
    style_axes(ax)
    return fig, ax


def plot_topdown_shadows(lobe, disc, R1, phases, incl_deg, ax=None,
                          theta_1=0.0, phi_1=0.0, n_field_1=0,
                          angle_acc=None, T_eff1=None, T_acc=None, stream_angle_deg=None,
                          T_2=None, P_orb=None, a_m=None, label=None):
    """
    Pole-on (face-on, "effective inclination 0") diagram of the system,
    oriented at this "apparent phase zero" so the secondary sits below
    and the primary/disc sit above -- i.e. mapping the corotating (x,y)
    frame to image coordinates via (X,Y) = (y, -x), so +x (towards the
    secondary) points down and -x (towards the primary) points up.

    Every body (secondary, primary, disc, accretion stream, dipole field
    lines, magnetic poles, accretion spot) shares its geometry-computation
    machinery with plot_component_outlines' outline output --
    lobe_outline/stream_outline/field_line_outlines/_magnetic_pole_points
    here, render.accretion_connection_line for the spot -- called at
    PHASE0=0.0, INCL0=0.0 with occlusion arguments (disc/lobe/
    primary_center/R1) omitted wherever those functions take them
    optionally, rather than a second, separately-maintained
    implementation of the same shapes. That specific (phase, incl) pair
    isn't arbitrary: eclipse.observer_frame's own closed form reduces
    there to exactly eX=(0,1,0), eY=(-1,0,0) -- i.e. project(pts, 0.0,
    0.0) IS this diagram's fixed "(X,Y) = (y, -x)" face-on frame, so
    reusing project() directly (instead of a hand-rolled equivalent) is
    both simpler and guaranteed to agree with it. The disc's inner rim
    and the primary's own circle are the only shapes still built by hand
    here -- both are exact circles from any viewing angle (the primary
    because it is one; the disc because plot_component_outlines' own
    disc_outline doesn't expose the inner boundary as its own curve), so
    there's no shared machinery to gain by routing them through anything
    else.

    Overlaid on top of that shared, occlusion-free base view: the
    secondary's shadow footprint on the orbital midplane
    (secondary_shadow_on_midplane), at the system's real inclination
    `incl_deg`, once for every phase in `phases` (typically the same
    phase array used for the light curve -- see simulate.py's
    --phase-min/--phase-max/--n-phases) -- each drawn as a faint black
    line, so the overlaid set shows how the eclipse's occulted footprint
    sweeps across the disc over the orbit. This sweep is the one thing
    genuinely specific to this view -- outline mode has no equivalent.

    theta_1/phi_1/n_field_1: the primary's magnetic-axis obliquity/azimuth
    and dipole field-line-loop count (params.SystemParams.theta_1_rad/
    phi_1_rad, --n_field_1) -- draws the same blue dotted loops (plus a
    small dot at each of the two magnetic poles) as plot_component_outlines'
    outline output, whenever n_field_1>0, with no occlusion test (see
    above).

    angle_acc/T_eff1/T_acc: the accretion spot's connection angle(s) (see
    params._parse_angle_acc) and the temperatures accretion_connection_line
    needs to decide whether each one is actually active -- draws one
    solid red field line per active entry, same as plot_component_outlines'
    "accretion spot" curves. All three must be given (not None) for this
    to draw anything.

    stream_angle_deg: forwarded to integrate_stream (see its docstring)
    for both the accretion-stream curve and the accretion-spot connection
    line(s) above -- extended (if needed) to cover angle_acc's own
    largest entry, so a trajectory that needs to sweep past its default
    closest-approach stopping point (e.g. to reach a --angle_acc beyond that)
    is available here too.

    T_2/P_orb/a_m: the secondary's temperature [K], orbital period [d],
    and separation [m] (SystemParams' own native units) -- given all
    three (the default None skips this entirely), two faint green lines
    are drawn flanking the accretion-stream centerline, offset by +-W
    at every point (W is itself the stream's own transverse scaleheight,
    already a half-width -- the full flanked span is 2W), from Hessman
    (1999)'s fits to the Lubow & Shu (1975) stream hydrodynamics (see
    stream.lubow_shu_eps/lubow_shu_stream_size) -- a visual measure of
    the systematic error in treating the stream/shadow boundary as an
    infinitely thin line, this face-on view's own particular use case
    (see shadow_constrained_regions). Only W (the in-plane transverse
    size) is meaningful in this face-on projection; the stream's
    perpendicular-to-orbital-plane extent H has no analogue here -- see
    plot_component_outlines' own stream_impact_ellipse_outline for where
    H matters instead.

    label: simulate.py's --prefix global label (see labeled_title), if
    any -- prepended to this plot's own "q=..., i=... -- face-on, ..." title.
    """
    import matplotlib.pyplot as plt
    from render import accretion_connection_line

    incl = np.radians(incl_deg)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
    else:
        fig = ax.figure

    secondary_color, primary_color, disc_color = "#d1272e", "#2e6f95", "#e08214"
    PHASE0, INCL0 = 0.0, 0.0
    n = 300
    primary_center = np.array([lobe.x1, 0.0, 0.0])

    # secondary: true face-on silhouette, same machinery the outline
    # output's own "secondary" curve uses
    Xs, Ys = lobe_outline(lobe, PHASE0, INCL0)
    ax.plot(Xs, Ys, "-", color=secondary_color, lw=1.0, label="secondary")
    ax.fill(Xs, Ys, color=secondary_color, alpha=0.15)

    # primary: a plain circle of radius R1 -- exact from any viewing
    # angle (a sphere's silhouette never depends on projection), so no
    # shared machinery is needed to get this right
    ang = np.linspace(0.0, 2.0 * np.pi, n)
    xp = lobe.x1 + R1 * np.cos(ang)
    yp = R1 * np.sin(ang)
    Xp, Yp, _ = project(np.stack([xp, yp, np.zeros(n)], axis=-1), PHASE0, INCL0)
    ax.plot(Xp, Yp, "-", color=primary_color, lw=1.0, label="primary")
    ax.fill(Xp, Yp, color=primary_color, alpha=0.35)

    # disc: outer rim + inner rim, face-on -- filled as a proper annulus
    # (outer boundary + reversed inner boundary in one path, so the hole
    # is left unfilled rather than painted over). The inner rim is (like
    # the primary above) an exact circle regardless of viewing angle; the
    # outer rim is disc.rim_xy's own curve, projected the same way.
    xo, yo = disc.rim_xy(n=n)
    Xo, Yo, _ = project(np.stack([xo, yo, np.zeros(n)], axis=-1), PHASE0, INCL0)
    ang_i = np.linspace(0.0, 2.0 * np.pi, n)
    xi = disc.x1 + disc.r_in * np.cos(ang_i)
    yi = disc.r_in * np.sin(ang_i)
    Xi, Yi, _ = project(np.stack([xi, yi, np.zeros(n)], axis=-1), PHASE0, INCL0)
    ax.plot(Xo, Yo, "-", color=disc_color, lw=1.0, label="accretion disc")
    ax.plot(Xi, Yi, "-", color=disc_color, lw=1.0)
    ax.fill(np.concatenate([Xo, Xi[::-1]]), np.concatenate([Yo, Yi[::-1]]),
            color=disc_color, alpha=0.12)
    if disc.e > 0.0:
        Xpa, Ypa, _ = project(_periastron_line(disc), PHASE0, INCL0)
        ax.plot(Xpa, Ypa, ":", color=disc_color, lw=0.6, alpha=0.5, label="periastron")

    # accretion stream: L1 to the disc impact point (or closest approach,
    # if it never reaches the disc), same convention as elsewhere, drawn
    # via stream_outline (see plot_component_outlines) once truncated.
    # Extend the integration limit (if needed) to guarantee the
    # trajectory sweeps far enough to reach every requested
    # accretion-connection angle -- same reasoning as
    # render.build_temperature_maps' own matching extension, needed here
    # too since this traj is passed pre-built into accretion_connection_line
    # below, bypassing its own fallback extension.
    eff_stream_angle = stream_angle_deg
    if angle_acc is not None:
        max_angle_acc = float(np.max(np.atleast_1d(angle_acc)))
        eff_stream_angle = max(stream_angle_deg, max_angle_acc) \
            if stream_angle_deg is not None else max_angle_acc
    traj = integrate_stream(lobe, stream_angle_deg=eff_stream_angle)
    # reported purely as information (see disc_impact_index's own report=
    # option) -- NOT used to truncate the stream below: how far the stream
    # is drawn/extends is regulated only by an explicit stream_angle_deg,
    # or (default) the trajectory's own closest approach to the primary,
    # regardless of whether it happens to pass through the disc first.
    disc_impact_index(traj, disc.rim, lobe.x1, report=True)
    if stream_angle_deg is not None:
        # an explicit stream_angle_deg (see simulate.py's --stream_angle) is
        # the user's own authoritative instruction for how far to show the
        # stream -- so display the WHOLE computed trajectory rather than
        # second-guessing it with the closest-approach heuristic below,
        # which exists only to pick a sensible endpoint when
        # stream_angle_deg wasn't given at all.
        idx = len(traj["x"]) - 1
    else:
        idx = closest_approach_index(traj, lobe.x1)
        if idx is None:
            idx = len(traj["x"]) - 1
    if angle_acc is not None:
        # don't let the plotted stream curve stop short of the furthest
        # active accretion-connection point -- see simulate.py's matching
        # "outline" output extension for why (avoids the red field-line
        # curves below branching off from a point the green stream curve
        # never visibly reaches).
        from stream import angle_acc_index
        for ang in np.atleast_1d(angle_acc):
            ang_idx = angle_acc_index(traj, float(ang))
            if ang_idx is not None:
                idx = max(idx, ang_idx)
    s_vals = np.linspace(0.0, traj["s"][idx], 300)
    xst, yst = sample_points(traj, s_vals)
    Xst, Yst, _ = stream_outline({"x": xst, "y": yst}, lobe.x1, PHASE0, INCL0)
    ax.plot(Xst, Yst, "-", color="#1b9e77", lw=1.0, label="accretion stream")

    # stream width (see this function's own T_2/P_orb/a_m docstring):
    # two lines flanking the centerline above, offset by +-W (W itself is
    # already the stream's own scaleheight/half-width, Hessman 1999 --
    # see lubow_shu_stream_size's own docstring, so the full flanked
    # span is 2W) along the LOCAL in-plane perpendicular to the stream's
    # own direction of travel there (a finite-difference tangent along
    # the already-sampled centerline, rotated 90 degrees) -- not the
    # stream's own true 3D cross-section (see
    # stream_impact_ellipse_outline for that, only meaningful at a
    # single point, the disc impact), just its in-plane extent swept
    # along the whole visible trajectory, the one thing a face-on view
    # can actually show.
    if T_2 is not None and P_orb is not None and a_m is not None:
        eps = lubow_shu_eps(T_2, P_orb, a_m)
        r1 = np.hypot(xst - lobe.x1, yst)
        _, half = lubow_shu_stream_size(r1, lobe.q, eps)
        tangent = np.gradient(np.stack([xst, yst], axis=-1), axis=0)
        tangent /= np.maximum(np.linalg.norm(tangent, axis=-1, keepdims=True), 1e-300)
        perp = np.stack([-tangent[:, 1], tangent[:, 0]], axis=-1)
        sides = []
        for sign in (1.0, -1.0):
            xw = xst + sign * half * perp[:, 0]
            yw = yst + sign * half * perp[:, 1]
            Xw, Yw, _ = stream_outline({"x": xw, "y": yw}, lobe.x1, PHASE0, INCL0)
            sides.append((Xw, Yw))
        (Xw1, Yw1), (Xw2, Yw2) = sides
        ax.fill(np.concatenate([Xw1, Xw2[::-1]]), np.concatenate([Yw1, Yw2[::-1]]),
                facecolor="#1b9e77", edgecolor="none", alpha=0.15)
        ax.plot(Xw1, Yw1, "-", color="#1b9e77", lw=1.0,
                label="stream width (Lubow & Shu)")
        ax.plot(Xw2, Yw2, "-", color="#1b9e77", lw=1.0)

        # beginning (near L1) and ending (disc impact, or wherever this
        # truncated trajectory itself stops) cross-sections -- this face-
        # on view has no use for the stream's true vertical (H) extent
        # (see this function's own T_2/P_orb/a_m docstring), so each is
        # simply a small filled disc of radius `half` (the same in-plane
        # W used for the width band above) around that end's own point,
        # rather than a true 3D ellipse -- a deliberately simplified cap,
        # not a claim about the unresolvable vertical extent.
        psi_cap = np.linspace(0.0, 2.0 * np.pi, 60)
        for i in (0, -1):
            xc = xst[i] + half[i] * np.cos(psi_cap)
            yc = yst[i] + half[i] * np.sin(psi_cap)
            Xc, Yc, _ = stream_outline({"x": xc, "y": yc}, lobe.x1, PHASE0, INCL0)
            ax.fill(Xc, Yc, facecolor="#1b9e77", edgecolor="none", alpha=0.15)
            ax.plot(Xc, Yc, "-", color="#1b9e77", lw=0.7, alpha=0.6)

    # dipole field-line loops + magnetic poles -- see field_line_outlines/
    # _magnetic_pole_points, the same functions plot_component_outlines
    # uses; disc/lobe are left unpassed here (no occlusion, see above)
    if n_field_1 > 0:
        for i, (Xf, Yf, _vis) in enumerate(
                field_line_outlines(primary_center, R1, theta_1, phi_1, n_field_1, PHASE0, INCL0)):
            ax.plot(Xf, Yf, ":", color="blue", lw=0.4, label="field lines" if i == 0 else None)

        _, pole_pts = _magnetic_pole_points(primary_center, R1, theta_1, phi_1)
        Xpole, Ypole, _ = project(pole_pts, PHASE0, INCL0)
        ax.plot(Xpole, Ypole, "o", color="blue", ms=2.5, label="magnetic poles")

    # accretion-spot connection(s) -- one solid red line per active
    # --angle_acc entry, same render.accretion_connection_line every other
    # "where does this field line go" question in this package uses
    if angle_acc is not None and T_eff1 is not None and T_acc is not None:
        _, spot_lines = accretion_connection_line(lobe, R1, theta_1, phi_1, angle_acc, T_eff1, T_acc,
                                                    traj=traj, stream_angle_deg=stream_angle_deg)
        for i, line in enumerate(spot_lines):
            Xa, Ya, _ = project(line, PHASE0, INCL0)
            ax.plot(Xa, Ya, "-", color="red", lw=1.0, label="accretion spot" if i == 0 else None)

    # secondary's shadow footprint on the midplane, one faint black curve
    # per lightcurve phase
    shadow_label = "secondary's midplane shadow"
    for phase in phases:
        shadow = secondary_shadow_on_midplane(lobe, phase, incl)
        if shadow is None:
            continue
        Xsh, Ysh, _ = project(np.stack([shadow[0], shadow[1], np.zeros_like(shadow[0])], axis=-1),
                               PHASE0, INCL0)
        ax.plot(Xsh, Ysh, "-", color="black", lw=0.6, alpha=0.25, label=shadow_label)
        shadow_label = None  # only the first curve gets a legend entry

    # Near edge-on, grazing limb points well off the true midplane can
    # extend their shadow far past the disc (dividing by the small
    # n_hat_z amplifies it) -- physically real, but the interesting part
    # is the sweep near the disc, so default the view there rather than
    # autoscaling out to occasional far-flung excursions.
    extent = 1.3 * max(disc.a, abs(lobe.x1) + R1, lobe.x2 - lobe.x1)
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)

    ax.set_aspect("equal")
    ax.set_xlabel("Y / a  (face-on)")
    ax.set_ylabel("-X / a  (face-on)")
    title = rf"q={lobe.q:.3f}, i={incl_deg:.2f}$^\circ$"
    if not disc.is_empty:
        title += f", R_out={disc.a:.4f}"
    ax.set_title(labeled_title(label, title))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3,
              fontsize=9, framealpha=0.9)
    style_axes(ax)
    return fig, ax
