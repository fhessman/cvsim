# plots.py
"""
Reusable sky-projected outline plots of the binary components (secondary
Roche lobe, primary, disc, stream), built on the same projection used for
the eclipse geometry (eclipse.project/observer_frame).
"""

import numpy as np
from scipy.spatial import ConvexHull

from eclipse import project, observer_frame, visible_mask, visible_mask_bulk
from roche import gravity, RocheLobe
from stream import closest_approach_index, integrate_stream, disc_impact_index, sample_points
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


def disc_hotspot_wedges(disc, phi_h, L_h_deg, dphi_max_deg, phase, incl, n_steps=24, n_pts=15,
                         flat_band_frac=0.08):
    """
    A sequence of thin, progressively fainter wedge-shaped patches sitting
    entirely on the disc's own outer edge, downstream of the stream-impact
    azimuth phi_h [rad] -- an illustrative (not photometrically accurate
    -- see below) stand-in for the render.disc_surfaces_with_teff hot
    spot, for the outline output. Returns a list of (X, Y, alpha) tuples,
    one per step, each ready for ax.fill(X, Y, alpha=alpha).

    Each step spans an angular slice [dphi0, dphi1) of downstream azimuth
    (same prograde/increasing-phi sense as disc_surfaces_with_teff's own
    dphi = mod(phi-phi_h, 2*pi)), covering 0 to dphi_max_deg in n_steps
    equal slices. The patch never spreads radially inward onto the disc
    face -- it stays fixed at r=rim(nu), the true outer edge -- but for a
    flared disc (opening_angle>0) does span that edge's own full vertical
    wall height, z=-h(nu) to +h(nu) with h=rim(nu)*tan(opening_angle),
    the exact geometry disc_cross_section_lines already draws as the
    rim's upper/lower lip curves (so the fill lines up with them, "as
    tall as the outer disc" rather than sitting only at its midplane).
    For a flat disc (opening_angle==0, no wall height to use) a thin
    radial sliver instead, (1-flat_band_frac)*rim(nu) to rim(nu) at z=0.
    Only alpha (not the patch's own extent) follows the real
    T_h*exp(-dphi/L_h) falloff, evaluated at each slice's own midpoint
    dphi. This is a deliberately rough visual cue for where the bright
    spot sits and how fast it fades, not a real surface-brightness map --
    "a set of different fillings that get weaker and weaker", not a
    smooth/quantitatively correct gradient.

    dphi_max_deg: total downstream extent to cover -- normally
    simulate.py's own hot-spot-vs-disc-temperature crossing point (see
    its _hotspot_dphi_max), so the wedge sequence stops exactly where the
    real max(T_base, T_h*exp(-dphi/L_h)) formula drops to T_base (no
    boost left at all) instead of an arbitrary fixed angular cutoff --
    giving a physically meaningful sense of how far the bright spot
    actually extends.
    """
    L_h = np.radians(L_h_deg)
    edges = np.linspace(0.0, np.radians(dphi_max_deg), n_steps + 1)
    wedges = []
    for i in range(n_steps):
        dphi0, dphi1 = edges[i], edges[i + 1]
        nu = np.linspace(phi_h + dphi0, phi_h + dphi1, n_pts)
        r = disc.rim(nu)
        x, y = disc.x1 + r * np.cos(nu), r * np.sin(nu)
        if disc.opening_angle > 0.0:
            h = r * np.tan(disc.opening_angle)
            pts = np.stack([np.concatenate([x, x[::-1]]),
                             np.concatenate([y, y[::-1]]),
                             np.concatenate([h, -h[::-1]])], axis=-1)
        else:
            r_in = r * (1.0 - flat_band_frac)
            x_in, y_in = disc.x1 + r_in * np.cos(nu), r_in * np.sin(nu)
            pts = np.stack([np.concatenate([x, x_in[::-1]]),
                             np.concatenate([y, y_in[::-1]]),
                             np.zeros(2 * n_pts)], axis=-1)
        X, Y, _ = project(pts, phase, incl)
        alpha = float(np.exp(-0.5 * (dphi0 + dphi1) / L_h))
        wedges.append((X, Y, alpha))
    return wedges


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
                             label=None):
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

    hotspot_phi_h/hotspot_L_h_deg/hotspot_dphi_max_deg: the disc's own
    stream-impact hot spot (model.T_h/model.L_h, DISC tab -- not the
    magnetic accretion_spot above), shown as a sequence of green,
    progressively fainter wedge patches sitting on the disc's own outer
    edge (never spread radially inward onto its face), downstream of the
    impact azimuth hotspot_phi_h [rad] (stream.impact_azimuth's own
    convention) out to hotspot_dphi_max_deg -- see disc_hotspot_wedges.
    Any of the three left None/non-positive (the default) disables it.

    label: simulate.py's --prefix global label (see labeled_title), if
    any -- prepended to this plot's own "q=..., i=..., phase=..." title.
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
    _hidden_split_plot(ax, Xst, Yst, vis_st, "#1b9e77", lw=style["lw"], label="accretion stream")
    # faint full limb circle, always drawn underneath (even when the
    # primary is entirely hidden, e.g. eclipsed behind the secondary or
    # the disc) -- the same "always-present faint guide" every other
    # body's outline gets from _hidden_split_plot's own unconditional
    # first pass (disc/stream/field lines above); primary_visibility_grid
    # can't reuse that helper directly (it needs a 2D contour, not a 1D
    # curve, to correctly trace a possibly-complex partial-occlusion
    # shape -- e.g. the disc cutting a chord across the primary's face,
    # not just its outer limb), so this circle is drawn separately, at
    # the identical (lw*0.5, alpha*0.3) faint styling, with the solid
    # contour below drawn on top of it for whatever's actually visible.
    _, Xp_full, Yp_full = circle_outline(primary_center, R1, phase, incl, n=100)
    ax.plot(Xp_full, Yp_full, color=primary_color, lw=style["lw"] * 0.5, alpha=0.3)
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
        def fill_secondary():
            ax.fill(Xs, Ys, color=secondary_color, alpha=0.15)

        def fill_disc_and_primary():
            # Xd/Yd (outer rim -- disc_outline's flat-disc parametric
            # sweep, or its flared-disc convex-hull silhouette) and
            # Xdi/Ydi (inner rim -- disc_inner_rim_line's own midplane
            # circle, computed above regardless of opening_angle) filled
            # as one "outer forward + inner reversed" path so the hole
            # between them is left unfilled instead of painted over (same
            # trick as plot_topdown_shadows' disc annulus) -- _ensure_ccw
            # first, since the two loops' relative winding otherwise isn't
            # guaranteed consistent (see its own docstring), which would
            # silently double-fill the hole instead of leaving it empty.
            # Xdi is empty when disc.r_in <= R1 (see disc_inner_rim_line):
            # there's no distinct hole then, the primary's own fill
            # (below) already covers that region, so just fill Xd/Yd whole.
            Xd_ccw, Yd_ccw = _ensure_ccw(Xd, Yd)
            if len(Xdi) > 0:
                Xdi_ccw, Ydi_ccw = _ensure_ccw(Xdi, Ydi)
                ax.fill(np.concatenate([Xd_ccw, Xdi_ccw[::-1]]),
                        np.concatenate([Yd_ccw, Ydi_ccw[::-1]]),
                        color="#e08214", alpha=0.12)
            else:
                ax.fill(Xd_ccw, Yd_ccw, color="#e08214", alpha=0.12)
            # disc's own stream-impact hot spot (see this function's
            # docstring, disc_hotspot_wedges): drawn here, with the rest
            # of the disc's fill, so it shares the same back-to-front
            # ordering against the secondary below -- not occlusion-tested
            # against the disc/primary's own geometry any further than
            # that, same simplification as the disc annulus fill above.
            if (hotspot_phi_h is not None and hotspot_L_h_deg is not None
                    and hotspot_dphi_max_deg is not None and hotspot_dphi_max_deg > 0.0):
                for i, (Xh, Yh, alpha) in enumerate(
                        disc_hotspot_wedges(disc, hotspot_phi_h, hotspot_L_h_deg,
                                             hotspot_dphi_max_deg, phase, incl)):
                    # facecolor/edgecolor (not the color= shorthand, which
                    # sets both) -- otherwise each wedge's own edge stroke
                    # shows as a visible seam against its neighbors,
                    # instead of the sequence reading as one smooth fade.
                    ax.fill(Xh, Yh, facecolor="green", edgecolor="none", alpha=alpha,
                            label="hot spot" if i == 0 else None)
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
    ax.set_title(labeled_title(label, f"q={lobe.q:.2f}, i={incl_deg:.1f} deg, phase={phase:.3f}"))
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


def plot_topdown_shadows(lobe, disc, R1, phases, incl_deg, ax=None,
                          theta_1=0.0, phi_1=0.0, n_field_1=0,
                          angle_acc=None, T_eff1=None, T_acc=None, stream_angle_deg=None,
                          label=None):
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
    # always reported (see disc_impact_index's own report= option), even
    # when its result isn't used for truncation below
    disc_idx = disc_impact_index(traj, disc.rim, lobe.x1, report=True)
    if stream_angle_deg is not None:
        # an explicit stream_angle_deg (see simulate.py's --stream_angle) is
        # the user's own authoritative instruction for how far to show the
        # stream -- e.g. to visualize overflow past where it would
        # otherwise hit the disc -- so display the WHOLE computed
        # trajectory rather than second-guessing it with the disc-impact/
        # closest-approach heuristics below, which exist only to pick a
        # sensible endpoint when stream_angle_deg wasn't given at all.
        idx = len(traj["x"]) - 1
    else:
        idx = disc_idx
        if idx is None:
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
    ax.set_title(labeled_title(label, f"q={lobe.q:.2f}, i={incl_deg:.1f} deg -- face-on, "
                                       f"secondary shadow vs. phase"))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3,
              fontsize=9, framealpha=0.9)
    style_axes(ax)
    return fig, ax
