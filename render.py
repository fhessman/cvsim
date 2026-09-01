# render.py
"""
Local-temperature and specific-intensity image synthesis for the whole
system (secondary + primary + disc), combining:

  - roche.RocheLobe.gravity_darkened_teff / irradiation.py  (secondary T map)
  - a steady-state (Shakura & Sunyaev 1973) or power-law disc T(r) profile
  - blackbody.band_intensity                (T -> surface brightness at a
                                              given wavelength, V-band by
                                              default)
  - eclipse.py / disc.Disc.visible          (occultation)

render_system_image() builds one per-point local effective temperature
map -- geometry, occlusion, and (optionally) secondary irradiation are
identical either way -- and `quantity` picks whether the returned image
holds that temperature directly or band_intensity(T, wavelength_m); the
two outputs can never drift out of sync with each other's physics since
they share every step except the final conversion.

Unlike the flux-integration code in lightcurve.py (which needs
foreshortened *projected* area per sample, dA*cos(angle-to-observer), to
get the total collected flux right), an *image* is a map of specific
intensity I_nu(x,y) as seen by the observer. For a blackbody, I_nu itself
does not depend on the emission angle -- only the projected area does --
so each pixel here is simply the (sample-density-weighted) average of
whatever surface points project into it. This is why a uniform-
temperature sphere renders as a flat, uniformly bright disc with no
artificial limb darkening: that's the physically correct appearance of a
blackbody surface.
"""

from dataclasses import dataclass

import numpy as np

from eclipse import (project, observer_frame, visible_mask, visible_mask_bulk,
                      visible_fraction, visible_fraction_bulk, sphere_visible_mask)
from blackbody import band_intensity, V_WAVELENGTH_M, PARSEC_M, flux_lambda_to_mjy
from lightcurve import sphere_points_full
from utils import with_progress

SIGMA_SB = 5.670374419e-8   # W / (m^2 K^4)
G = 6.67430e-11             # m^3 / (kg s^2)
MSUN = 1.98892e30           # kg
RSUN = 6.957e8              # m


def split_area_count(n_areas, aspect):
    """
    Turn a single "how many surface elements" target into the (n1, n2)
    grid-dimension pair each body's own rendering grid actually takes
    (RocheLobe's (ntheta,nphi), disc.all_surfaces_grid's (n_r,n_nu),
    star_points' (n_rho,n_phi)): n1*n2 ~= n_areas, split so that n1:n2
    matches `aspect` = n2/n1.

    `aspect` should be the body's own physical extent2/extent1 ratio (see
    disc_aspect_ratio, PRIMARY_ASPECT below), not an arbitrary constant:
    individual cells are each roughly
    (extent1/n1) x (extent2/n2) in physical size, so matching n1:n2 to
    extent1:extent2 is what makes cells come out close to square --
    i.e. all n_areas elements end up with roughly the same area --
    rather than badly elongated in whichever direction is under-resolved.
    A fixed aspect (e.g. always 2:1) is only right by coincidence: the
    disc's radial extent (R_out-R_in) vs. its azimuthal circumference can
    differ by an order of magnitude depending on r_in/r_out, and using
    2:1 regardless left the disc's rendering grid badly elongated in
    azimuth -- the direct cause of the "spoke"-shaped holes at low
    resolution.
    """
    n1 = max(1, int(round((n_areas / aspect) ** 0.5)))
    n2 = max(1, int(round(n1 * aspect)))
    return n1, n2


# primary (star_points' alpha/beta polar parametrization of the
# projected disc): "radial" extent (rho=R*sin(alpha)) is R, "azimuthal"
# extent (circumference at the limb) is 2*pi*R -- their ratio is
# R-independent, always 2*pi.
PRIMARY_ASPECT = 2.0 * np.pi


def disc_aspect_ratio(R_in, R_out):
    """
    Disc equal-area aspect (n_nu/n_r): radial extent is R_out-R_in,
    azimuthal extent is the circumference 2*pi*R_mean at the mean radius
    -- unlike the secondary/primary this genuinely depends on the disc's
    own r_in/r_out (a thin annulus, R_out~R_in, wants almost all of its
    elements in nu; a wide one wants relatively more in r).
    """
    R_mean = 0.5 * (R_in + R_out)
    return 2.0 * np.pi * R_mean / max(R_out - R_in, 1e-12)


def wall_aspect_ratio(opening_angle_rad):
    """
    Equal-area aspect (n_nu/n_z) for the disc's edge "wall" surfaces
    (disc.Disc.outer_edge_grid/inner_edge_grid): a cell's height is
    2*R*tan(opening_angle)/n_z, its azimuthal width R*dnu = 2*pi*R/n_nu
    -- R cancels in the ratio, so unlike disc_aspect_ratio this is a
    constant depending only on the opening angle, not on R_in/R_out or
    eccentricity. For a realistic (few-degree) opening angle this is
    large (tens to ~100): the wall is a short, all-the-way-around ribbon,
    the opposite shape from the cone surfaces sharing its n_nu by
    default, so reusing the cone's own (n_r,n_nu) split for the wall
    leaves cells wildly elongated -- short azimuthally, and, wrongly,
    however tall n_z happens to be set independently of that -- so the
    apparent size of an occultation depends on which surface (and hence
    which direction) it's on, rather than on the actual occulting area.
    """
    return np.pi / max(np.tan(opening_angle_rad), 1e-12)


def disc_face_rim_split(n_areas_disc, R_in, R_out, opening_angle_rad, incl_deg,
                         f_min=0.1, f_max=0.9, rim_boost=1.1, incl_power=1.0):
    """
    Split a flared disc's total observer-facing surface-element budget
    between its face (the upper/lower cones) and its rim (the inner/outer
    edge walls), weighted by PROJECTED (sky-plane) area rather than true
    area: what determines whether occultation/rendering looks equally
    well resolved everywhere is projected point DENSITY, and the face and
    rim foreshorten oppositely with inclination -- the (near-horizontal)
    face's projected area shrinks with cos(incl) (maximal face-on,
    vanishing edge-on), while the (near-vertical) rim's shrinks with
    sin(incl) (the opposite). So for true-area ratio A_surf:A_rim, the
    equal-projected-density point-count ratio is

        N_surf / N_rim ~ (A_surf/A_rim) * (cos(incl)/sin(incl))^incl_power / rim_boost

    Splitting by true area alone (the original behavior) puts far too
    many points on the rim relative to the face at any inclination not
    extremely close to edge-on, since a realistic (few-degree) opening
    angle already makes the rim's true area much smaller than the face's
    -- but the *projected* comparison is what actually governs how well
    resolved each surface looks.

    rim_boost (default 1.1): an extra multiplicative factor on N_surf/N_rim,
    on top of the plain projected-area ratio -- >1 favors the face, <1
    favors the rim (0.5 doubles the face's share of the ratio relative to
    the plain projected-area argument above).

    incl_power (default 1, i.e. the plain cos/sin ratio): an exponent on
    the inclination term to soften (incl_power<1) or sharpen (>1) how
    hard the split swings between face-heavy and rim-heavy as inclination
    moves away from the (A_surf/A_rim)-balanced angle -- lower it if the
    split still feels too inclination-sensitive even after rim_boost.

    f_min/f_max clamp the face's resulting share of the budget so neither
    surface starves to ~0 points at a near-face-on or near-edge-on
    inclination -- the "invisible" surface's occultation geometry (e.g.
    the disc self-shadowing across its far side) still needs some
    resolution regardless of viewing angle.

    Returns ((n_r, n_nu), (n_z, n_nu_wall)): aspect-correct grid
    dimensions ready to pass to disc_surfaces_with_teff (n_r*n_nu points
    on EACH of the upper/lower cone faces, so 2*n_r*n_nu total face
    points; disc.all_surfaces_grid similarly gives 2*n_z*n_nu_wall total
    wall points, split further by its own inner/outer area ratio -- see
    that method's docstring). If opening_angle_rad<=0 (flat disc, no
    rim), the whole budget goes to the face and the rim tuple is (0, 0).
    """
    if opening_angle_rad <= 0.0:
        return split_area_count(n_areas_disc, disc_aspect_ratio(R_in, R_out)), (0, 0)

    A_surf = np.pi * (R_out ** 2 - R_in ** 2) / np.cos(opening_angle_rad)  # one face
    A_rim = np.pi * np.tan(opening_angle_rad) * (R_out ** 2 + R_in ** 2)  # inner+outer

    incl = np.radians(incl_deg)
    eps = 1e-6
    cot_i = (np.cos(incl) / max(np.sin(incl), eps)) ** incl_power
    ratio = (A_surf / max(A_rim, eps)) * cot_i / max(rim_boost, eps)
    f_surf = np.clip(ratio / (ratio + 1.0), f_min, f_max)

    n_face = max(2, round(n_areas_disc * f_surf))
    n_rim = max(2, n_areas_disc - n_face)

    n_r, n_nu = split_area_count(max(n_face // 2, 1), disc_aspect_ratio(R_in, R_out))
    n_z, n_nu_wall = split_area_count(max(n_rim // 2, 1), wall_aspect_ratio(opening_angle_rad))
    return (n_r, n_nu), (n_z, n_nu_wall)


def disc_steady_state_teff(r_over_a, a_meters, M1_kg, Mdot_kgps, Rin_over_a):
    """
    Standard steady-state, optically-thick accretion-disc temperature
    profile (Shakura & Sunyaev 1973; e.g. Frank, King & Raine "Accretion
    Power in Astrophysics", eq. 5.43):

        sigma_SB * T(r)^4 = (3 G M1 Mdot)/(8 pi r^3) * [1 - sqrt(Rin/r)]

    r_over_a, Rin_over_a: radius / disc-inner-radius in units of the
    orbital separation a; a_meters converts to physical units.
    """
    r = np.asarray(r_over_a, dtype=float) * a_meters
    Rin = Rin_over_a * a_meters
    with np.errstate(invalid="ignore"):
        bracket = np.clip(1.0 - np.sqrt(Rin / np.maximum(r, Rin)), 0.0, None)
    flux = (3.0 * G * M1_kg * Mdot_kgps) / (8.0 * np.pi * r ** 3) * bracket
    return (flux / SIGMA_SB) ** 0.25


def disc_powerlaw_teff(r_over_a, T_0, R_in_over_a, beta_disc):
    """
    Disc temperature model, referenced to the inner radius:

        T_d(R) = T_0 * (R/R_in)^beta_disc

    beta_disc ~ -3/4 is the standard steady-state (Shakura & Sunyaev)
    scaling away from the inner boundary; beta_disc=0 gives a uniform-
    temperature disc, a useful idealized test case since it isolates the
    irradiation geometry from the disc's own radial temperature
    structure (see e.g. the cold-primary/hot-uniform-disc sanity test).
    """
    return T_0 * (np.asarray(r_over_a, dtype=float) / R_in_over_a) ** beta_disc


def disc_grid_with_teff(disc, teff_func, n_r=80, n_nu=160):
    """
    Disc surface-element point cloud with per-point (x, y, area, T): x,y
    in the corotating frame (z=0), area in units of a^2, T [K] =
    teff_func(r) for each point's radius r (units of a, distance from
    primary). Shared by the image renderer and the secondary-irradiation
    calculation so both use the same disc temperature model, whichever
    `teff_func` is passed in (steady-state, power-law, ...).

    Uses disc.equal_area_annulus(n_r*n_nu) rather than a regular (r,nu)
    grid -- n_r, n_nu are kept only as the calling convention (their
    product is the total point count that matters now); see that
    method's docstring for why a regular grid's sky-plane point density
    is non-uniform (a "moire" artifact once scatter-plotted).
    """
    x1 = disc.x1
    r, nu, areas = disc.equal_area_annulus(n_r * n_nu)
    xs = x1 + r * np.cos(nu)
    ys = r * np.sin(nu)
    Ts = teff_func(r)
    return xs, ys, areas, Ts


def disc_grid_teff(disc, a_meters, M1_kg, Mdot_kgps, n_r=80, n_nu=160):
    """Steady-state-disc convenience wrapper around disc_grid_with_teff."""
    def teff_func(r):
        return disc_steady_state_teff(r, a_meters, M1_kg, Mdot_kgps, disc.r_in)
    return disc_grid_with_teff(disc, teff_func, n_r=n_r, n_nu=n_nu)


def disc_surfaces_with_teff(disc, teff_func, n_r=40, n_nu=80, n_z=20, n_wall=None,
                             T_h=None, phi_h=0.0, L_h_deg=5.0):
    """
    Flared (opening_angle>0) disc's four surfaces (upper/lower cone,
    outer/inner edge) as (x,y,z,normals,areas,T,surface_id), T=teff_func(r)
    using each point's own lookup radius (see disc.all_surfaces_grid) --
    the edge surfaces get teff_func(rim(nu)) / teff_func(r_in), "the
    temperature of the local disc" at their radius.

    Stream-impact hot spot on the outer rim (surface_id==2): if T_h is
    given, the rim's base temperature there is replaced by
    max(T_base, T_h*exp(-dphi/L_h)), where dphi is the azimuthal distance
    downstream (prograde, increasing phi -- the same sense disc/stream
    material actually flows in this frame; see stream._eom) from the
    impact site phi_h [rad] -- where the stream crosses the outer disc
    radius -- wrapped to [0, 2*pi) so it decays smoothly all the way
    around instead of blowing up on the upstream side. L_h_deg is the
    decay length in degrees (~5 deg for a typical CV bright spot); T_h is
    a measure of the kinetic energy the stream dissipates on impact.

    Confined to the wall deliberately, not spread onto the neighboring
    upper/lower cone surfaces: the impact is physically a feature of the
    rim, and any rule for how far onto the disc face it "should" spread
    has no natural physical scale to anchor it to -- tying it to the
    radial grid spacing instead makes the hot spot's apparent size
    resolution-dependent (finer grid -> smaller boosted patch), which is
    worse than the wall being a thin sliver. If the wall is barely
    visible at the current resolution/pixel_mapping, that's a rendering
    (sampling/binning) concern to fix there, not a reason to paint the
    hot spot onto geometry it doesn't belong on.
    """
    # the wall surfaces (outer/inner edge) get their own n_z:n_nu split
    # matching their own aspect ratio, not the cone surfaces' n_nu -- see
    # wall_aspect_ratio -- so a cell's height and azimuthal width come out
    # comparable instead of the wall being resolved far more finely
    # around than up, which would make an occultation's apparent size
    # depend on which surface -- and hence which direction -- it's
    # actually on.
    #
    # n_wall: pass an explicit (n_z_wall, n_nu_wall) -- e.g. from
    # disc_face_rim_split's inclination-aware face/rim budget split -- to
    # use it directly instead of deriving one from n_z*n_nu (the default,
    # kept for backward compatibility with callers that don't care about
    # the face/rim split and just want *a* reasonable wall shape).
    # Skipped entirely for a flat disc, which never builds wall geometry
    # in the first place (opening_angle<=0 short-circuits inside
    # outer_edge_grid/inner_edge_grid) -- feeding wall_aspect_ratio's
    # tan(0)-driven huge aspect into split_area_count there would be a
    # wasted (and, at n_nu_wall in the billions, a very large)
    # computation for geometry that's discarded anyway.
    if n_wall is not None:
        n_z_wall, n_nu_wall = n_wall
    elif disc.opening_angle > 0.0:
        n_z_wall, n_nu_wall = split_area_count(n_z * n_nu, wall_aspect_ratio(disc.opening_angle))
    else:
        n_z_wall, n_nu_wall = n_z, n_nu
    x, y, z, normals, areas, r, sid = disc.all_surfaces_grid(
        n_r=n_r, n_nu=n_nu, n_z=n_z_wall, n_nu_wall=n_nu_wall)
    T = teff_func(r)
    if T_h is not None:
        outer = sid == 2
        phi = np.arctan2(y[outer], x[outer] - disc.x1)
        dphi = np.mod(phi - phi_h, 2.0 * np.pi)
        hot = T_h * np.exp(-dphi / np.radians(L_h_deg))
        T = T.copy()
        T[outer] = np.maximum(T[outer], hot)
    return x, y, z, normals, areas, T, sid


def auto_extent(lobe, disc, R1, incl_deg, temp_maps, n_phase_samples=73):
    """
    (xmin, xmax, ymin, ymax) tightly bounding everywhere any part of the
    system (secondary, disc, stream, primary) can ever project to, across
    every orbital phase -- i.e. depends only on the system's own geometry
    and inclination, not on which specific phase is being rendered.

    render_system_image calls this itself when extent=None, so a single
    still image already gets a tight (not phase-dependent) fit; pass the
    result explicitly as `extent` to a sequence of render_system_image
    calls at different phases (a movie) so every frame shares the same
    fixed field of view, computed once, rather than each one silently
    recomputing (and potentially getting a slightly different answer at
    the phase-sampling resolution below, though it's the same calculation
    either way) -- what actually varies frame to frame is occlusion
    (which points are currently *visible*), not the geometric envelope
    all points could ever reach, so refitting per frame to only-currently-
    visible content would zoom/pan between frames instead of holding still.

    Works by projecting every body's full point cloud (not filtered by
    visibility -- a point hidden at one phase may be the one setting the
    extent at another) at `n_phase_samples` phases spanning a full orbit,
    and taking the union of each phase's bounding box. The primary (a
    sphere, not tabulated as a point cloud) is approximated by 6 points
    +-R1 along each axis from its center -- exact for a sphere's
    silhouette extent under any viewing angle.
    """
    incl = np.radians(incl_deg)
    center1 = np.array([lobe.x1, 0.0, 0.0])
    primary_pts = center1 + R1 * np.array([
        [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
    ])
    all_pts = np.concatenate([temp_maps.sec_pts, temp_maps.disc_pts,
                               temp_maps.stream_pts, primary_pts], axis=0)

    xmin = ymin = np.inf
    xmax = ymax = -np.inf
    for ph in np.linspace(0.0, 1.0, n_phase_samples, endpoint=False):
        X, Y, _ = project(all_pts, ph, incl)
        xmin, xmax = min(xmin, X.min()), max(xmax, X.max())
        ymin, ymax = min(ymin, Y.min()), max(ymax, Y.max())
    return xmin, xmax, ymin, ymax


def _pixel_edges(xmin, xmax, ymin, ymax, image_size, pad_frac=0.05):
    """
    (xedges, yedges) for a pixel grid tightly covering [xmin,xmax] x
    [ymin,ymax] plus pad_frac (default 5%) margin on each side -- unlike a
    single square extent sized to the larger of the system's true X/Y
    extents (wasting most of the image on empty space whenever the two
    differ a lot, e.g. a near-edge-on system's sky-projected Y extent is
    typically much smaller than its X extent), this puts every pixel
    somewhere the rendered content can actually reach. Pixels stay square
    (same physical size in X and Y, so shapes aren't stretched): the
    longer padded axis gets `image_size` pixels, the shorter one
    proportionally fewer at the same pixel size.
    """
    xspan, yspan = xmax - xmin, ymax - ymin
    xpad, ypad = pad_frac * max(xspan, 1e-300), pad_frac * max(yspan, 1e-300)
    xmin, xmax = xmin - xpad, xmax + xpad
    ymin, ymax = ymin - ypad, ymax + ypad
    xspan, yspan = xmax - xmin, ymax - ymin

    pixel_size = max(xspan, yspan) / image_size
    nx = max(1, int(round(xspan / pixel_size)))
    ny = max(1, int(round(yspan / pixel_size)))
    return np.linspace(xmin, xmax, nx + 1), np.linspace(ymin, ymax, ny + 1)


def _bin_image_indirect(X, Y, values, xedges, yedges):
    """
    "indirect" pixel mapping: average `values` of points landing in each
    pixel -- i.e. each surface *sample* independently picks the one pixel
    it happens to land in (a scatter), so a pixel with no sample landing
    in it goes empty (NaN), regardless of whether the actual surface
    geometrically covers that pixel. This shows up as "holes" once the
    pixel grid is finer than the sample spacing -- see _bin_image_direct
    for the area-aware alternative.

    Returns img[iy,ix] (imshow-ready with origin="lower"), by transposing
    numpy.histogram2d's [ix,iy]-ordered output.
    """
    total, _, _ = np.histogram2d(X, Y, bins=[xedges, yedges], weights=values)
    count, _, _ = np.histogram2d(X, Y, bins=[xedges, yedges])
    with np.errstate(invalid="ignore", divide="ignore"):
        img = total / count
    return img.T


def _bin_image_direct(X, Y, values, depths, half_x, half_y, xedges, yedges):
    """
    "direct" pixel mapping: rather than each surface sample picking the
    single pixel it happens to land in, each sample is compared against
    the pixel grid's own pixel area and splatted across every pixel its
    own projected footprint (a half_x x half_y rectangle centered on
    (X,Y)) actually covers -- so a surface element bigger than a pixel
    fills all of the pixels it geometrically overlaps instead of just
    one, closing the "holes" _bin_image_indirect leaves behind. A
    per-pixel depth buffer (nearest sample wins, using the same "depth
    increases towards the observer" convention as eclipse.project)
    resolves overlaps between different bodies' splats. Much more
    time-intensive than the histogram-based indirect approach: a Python
    loop over every sample, each writing a small pixel patch.

    half_x, half_y: per-sample screen-space half-widths (physical units,
    same as X/Y/xedges) along each axis -- NOT necessarily equal: an
    isotropic sample (half_x=half_y=0.5*sqrt(area)) is fine for a body
    whose local shape doesn't have a strongly preferred orientation
    relative to the observer, but a genuinely anisotropic footprint (see
    _disc_splat_halfwidths) needs the two axes sized independently, or
    its narrow direction's "should be near-zero" width leaks into the
    wide direction's splat size too.

    xedges, yedges must share the same pixel size (see _pixel_edges) --
    this assumes square pixels, using xedges' spacing for both axes.

    Returns img[iy,ix] (imshow-ready with origin="lower").
    """
    nx, ny = len(xedges) - 1, len(yedges) - 1
    pixel_size = xedges[1] - xedges[0]

    depth_buf = np.full((ny, nx), -np.inf)
    value_buf = np.full((ny, nx), np.nan)

    px = (X - xedges[0]) / pixel_size
    py = (Y - yedges[0]) / pixel_size
    hx = half_x / pixel_size
    hy = half_y / pixel_size

    ix_lo = np.clip(np.floor(px - hx).astype(int), 0, nx - 1)
    ix_hi = np.clip(np.floor(px + hx).astype(int), 0, nx - 1)
    iy_lo = np.clip(np.floor(py - hy).astype(int), 0, ny - 1)
    iy_hi = np.clip(np.floor(py + hy).astype(int), 0, ny - 1)
    in_bounds = (px >= 0.0) & (px < nx) & (py >= 0.0) & (py < ny)
    idx = np.nonzero(in_bounds)[0]

    for i in with_progress(idx, len(idx), "direct pixel mapping"):
        ysl = slice(iy_lo[i], iy_hi[i] + 1)
        xsl = slice(ix_lo[i], ix_hi[i] + 1)
        dview = depth_buf[ysl, xsl]
        nearer = depths[i] > dview
        if nearer.any():
            dview[nearer] = depths[i]
            value_buf[ysl, xsl][nearer] = values[i]

    return value_buf


def _bin_image(X, Y, values, xedges, yedges, mode="indirect", depths=None,
                half_x=None, half_y=None):
    """Dispatch to _bin_image_indirect or _bin_image_direct; see pixel_mapping
    in render_system_image's docstring."""
    if mode == "indirect":
        return _bin_image_indirect(X, Y, values, xedges, yedges)
    if mode == "direct":
        if depths is None or half_x is None or half_y is None:
            raise ValueError("pixel_mapping='direct' needs depths, half_x, and half_y")
        return _bin_image_direct(X, Y, values, depths, half_x, half_y, xedges, yedges)
    raise ValueError("pixel_mapping must be 'indirect' or 'direct'")


def _disc_splat_halfwidths(pts, normals, areas, x1, eX, eY):
    """
    Anisotropic "direct" pixel-mapping splat half-widths for disc surface
    points (face or wall) -- the projected-shape counterpart to the
    isotropic 0.5*sqrt(area) every other body still uses.

    A disc surface cell is treated as a square of side sqrt(area) in its
    own local tangent plane, spanned by two orthogonal directions: the
    azimuthal direction u_hat = (-sin(nu),cos(nu),0) (always well-defined
    and always exactly perpendicular to the surface normal, for any point
    on any of the four disc surfaces -- none of their normals ever have
    an azimuthal component) and v_hat = normal x u_hat (the remaining
    in-tangent-plane direction, also always unit length since normal and
    u_hat are always exactly orthogonal). Projecting each of these two
    edge vectors through (eX,eY) and taking a bounding box gives a real
    anisotropic footprint, unlike sqrt(area) alone.

    This matters because an isotropic splat blends both local directions
    into one size regardless of how each one actually foreshortens: at
    or near i=90 deg the disc's near-flat face is seen exactly edge-on,
    so the azimuthal direction collapses to near-zero screen width while
    the vertical (z_hat) direction stays fully visible -- an isotropic
    splat leaks the "should be near-zero" azimuthal width into the
    height too, making disc material appear to stick up past its true
    local envelope (most visible for a large opening angle, or few
    sample points, where each cell's area -- and hence its isotropic
    splat size -- is largest).

    Returns (half_x, half_y), each the projected half-extent along the
    image's eX/eY axis.
    """
    side = np.sqrt(np.maximum(areas, 0.0))
    nu = np.arctan2(pts[..., 1], pts[..., 0] - x1)
    u_hat = np.stack([-np.sin(nu), np.cos(nu), np.zeros_like(nu)], axis=-1)
    v_hat = np.cross(normals, u_hat)
    v_hat /= np.maximum(np.linalg.norm(v_hat, axis=-1, keepdims=True), 1e-12)
    du = 0.5 * side[..., None] * u_hat
    dv = 0.5 * side[..., None] * v_hat
    half_x = np.abs(du @ eX) + np.abs(dv @ eX)
    half_y = np.abs(du @ eY) + np.abs(dv @ eY)
    return half_x, half_y


@dataclass
class TemperatureMaps:
    """
    The phase-independent half of render_system_image: the secondary's
    (irradiated) temperature map, the disc's own temperature map
    (including the stream-impact hot spot), and the accretion stream's
    sample points, all fixed patterns in the corotating frame. Build once
    with build_temperature_maps() and reuse across as many phases as you
    like (e.g. movie frames) without repeating the
    O(n_disc_irrad*n_secondary) irradiation calculation on every frame --
    only occlusion and observer-angle foreshortening actually depend on
    phase, and those are cheap.
    """
    Tsec: np.ndarray            # (Nsec,) secondary temperature [K]
    sec_areas: np.ndarray       # (Nsec,) secondary area elements, units of a^2
    sec_pts: np.ndarray         # (Nsec,3) secondary sample points, corotating frame
    sec_normals: np.ndarray     # (Nsec,3) secondary outward unit normals
    primary_normals: np.ndarray    # (Nprim,3) primary sample directions == outward unit
                                    # normals on the unit sphere -- direction pattern only,
                                    # not scaled by R1 (see build_temperature_maps' n_primary):
                                    # actual points/areas are R1*primary_normals /
                                    # primary_base_areas*R1**2, computed live at each call
                                    # site (render_system_image/_phase_chunk_flux) so a
                                    # fitted R_1 (a "cheap" live parameter, see simulate.py's
                                    # _CHEAP_SYSTEM_FIT_FIELDS) is honored without rebuilding
                                    # this whole (expensive) TemperatureMaps.
    primary_base_areas: np.ndarray  # (Nprim,) per-point area weight at unit radius
    primary_spot_mask: np.ndarray   # (Nprim,) bool -- True where the accretion spot (see
                                     # build_temperature_maps' angle_acc/spot_acc/T_acc,
                                     # magnetic.py) covers this primary sample point,
                                     # all-False if inactive. Deliberately a MASK, not a
                                     # baked temperature array: the primary's own
                                     # observer-facing flux/image reconstructs
                                     # np.where(mask, T_acc, T_eff1) live at each call
                                     # (see _phase_chunk_flux/render_system_image) so
                                     # T_1 stays a cheap live fit parameter there (see
                                     # simulate.py's _CHEAP_SYSTEM_FIT_FIELDS) exactly as
                                     # before this feature -- only the spot's overall
                                     # heating of the SECONDARY (via irradiation, baked
                                     # into Tsec below at build time, same as every other
                                     # irradiation input) doesn't get this live treatment.
    accretion_footpoint_dirs: list  # list of (3,) unit vectors, one per usable --angle_acc
                                     # entry (see render.accretion_connection_line),
                                     # possibly empty (no active spot)
    accretion_field_lines: list     # parallel list of (n,3) curves from the primary's
                                     # surface to each entry's connection point
    disc_pts: np.ndarray        # (Ndisc,3) disc sample points, corotating frame
    disc_normals: np.ndarray    # (Ndisc,3) outward normals, or None for a flat disc
    disc_T: np.ndarray          # (Ndisc,) disc temperature [K] at disc_pts
    disc_areas: np.ndarray      # (Ndisc,) disc cell areas, units of a^2
    disc_cell_size: float       # radial disc grid spacing actually used, units of a
    stream_pts: np.ndarray      # (Nstream,3) stream sample points, L1 to disc impact
    stream_areas: np.ndarray    # (Nstream,) per-point footprint area, units of a^2
    stream_T: float             # stream temperature [K] -- the (cusp-fixed) T at L1


def accretion_connection_line(lobe, R1, theta_1, phi_1, angle_acc, T_eff1, T_acc, traj=None,
                               stream_angle_deg=None):
    """
    The field line(s) connecting the ballistic accretion stream to the
    primary's surface at swept angle angle_acc (cumulative primary-centered
    azimuth from L1, same convention as stream_angle_deg) -- a single
    float, or an array-like of them (see params._parse_angle_acc; every
    entry connects to the same stream/magnetic geometry, just at a
    different point along it) -- the geometry-only half of
    _accretion_spot_data, split out so simulate.py's "outline" output and
    _accretion_spot_data can share it.

    traj: see _accretion_spot_data. stream_angle_deg: forwarded to
    stream.integrate_stream (see its own docstring) if `traj` isn't
    already given -- extended to cover angle_acc's own largest entry (so
    the trajectory is always integrated far enough to actually reach
    every requested connection point) when it would otherwise fall
    short; has no effect when `traj` is already given, since the
    trajectory's own extent was decided wherever it was actually
    integrated.

    Returns (footpoint_dirs, field_lines): parallel lists (each possibly
    empty, if angle_acc is None or every entry turns out unusable), one entry
    per angle_acc value that IS usable -- footpoint_dirs are (3,) unit
    vectors, field_lines are (n,3) curves from the surface to that
    entry's connection point. Prints a one-line explanation for every
    entry that ends up unusable, except simply not being requested (angle_acc
    is None) -- see angle_acc/T_acc's own docstrings in params.py.
    """
    footpoint_dirs, field_lines = [], []
    if angle_acc is not None:
        angles = np.atleast_1d(angle_acc)
        if traj is None and lobe.fill_factor >= 1.0:
            from stream import integrate_stream
            eff_stream_angle = max(stream_angle_deg, float(np.max(angles))) \
                if stream_angle_deg is not None else float(np.max(angles))
            traj = integrate_stream(lobe, stream_angle_deg=eff_stream_angle)
        if traj is None:
            print("no accretion stream (secondary underfills its Roche lobe): "
                  "ignoring angle_acc")
        elif T_acc <= T_eff1:
            print(f"T_acc={T_acc:g}K is not hotter than T_1={T_eff1:g}K: "
                  f"ignoring accretion spot")
        else:
            from stream import angle_acc_index
            from magnetic import field_line_to_point
            center1 = np.array([lobe.x1, 0.0, 0.0])
            for ang in angles:
                i_acc = angle_acc_index(traj, ang)
                if i_acc is None:
                    print(f"angle_acc={ang:g} deg is never actually swept by the stream "
                          f"trajectory (max reached: {np.max(np.abs(traj['angle_deg'])):.6g} deg): "
                          f"ignoring that accretion spot")
                    continue
                stream_conn_pt = np.array([traj["x"][i_acc], traj["y"][i_acc], 0.0])
                # n well above field_line_to_point's own default (100):
                # radial_velocity_curve's weighted median is dominated by
                # just the hottest few points near the surface end (see
                # render_system_image's own docstring on the log(T)
                # taper), so a coarse line leaves too few points there to
                # shift smoothly as visibility/geometry change from phase
                # to phase -- the weighted median jumps between a small
                # set of discrete candidates instead of sliding
                # continuously, a visibly jagged (not just noisy) curve.
                field_line, footpoint_dir = field_line_to_point(
                    center1, R1, theta_1, phi_1, stream_conn_pt, n=400)
                footpoint_dirs.append(footpoint_dir)
                field_lines.append(field_line)
    return footpoint_dirs, field_lines


def _accretion_spot_data(lobe, R1, T_eff1, theta_1, phi_1, angle_acc, spot_acc, T_acc,
                          primary_normals, traj=None, stream_angle_deg=None):
    """
    The primary's accretion spot(s) (see magnetic.py): connect the
    ballistic stream to a magnetic field line at each swept angle in
    angle_acc (via accretion_connection_line), and find which of
    the primary's own sample points (primary_normals, unit directions)
    fall within spot_acc of any of the field lines' surface footpoints
    -- every spot shares the same spot_acc/T_acc/u_acc, so this comes
    down to a single union mask, not one per spot.

    traj: pass an already-integrated stream.integrate_stream(lobe) result
    to skip repeating that integration (build_temperature_maps already has
    one on hand from its own stream_pts section); left None (the default)
    to integrate one fresh here (e.g. load_temperature_maps, which has no
    other reason to integrate the stream) -- stream_angle_deg is then
    forwarded to that integration (see its own docstring); has no effect
    when `traj` is already given.

    Returns (spot_mask, footpoint_dirs, field_lines):
      spot_mask: (Nprim,) bool, all-False if inactive.
      footpoint_dirs: list of (3,) unit vectors, one per usable angle_acc
        entry (see accretion_connection_line), possibly empty.
      field_lines: parallel list of (n,3) curves, possibly empty.
    """
    from magnetic import accretion_spot_mask

    footpoint_dirs, field_lines = accretion_connection_line(
        lobe, R1, theta_1, phi_1, angle_acc, T_eff1, T_acc, traj=traj,
        stream_angle_deg=stream_angle_deg)

    spot_mask = np.zeros(primary_normals.shape[0], dtype=bool)
    for footpoint_dir in footpoint_dirs:
        spot_mask |= accretion_spot_mask(primary_normals, footpoint_dir, spot_acc)
    return spot_mask, footpoint_dirs, field_lines


def build_temperature_maps(lobe, disc, T_eff1, T_eff2, R1,
                            disc_teff_func=None, a_meters=None, M1_kg=None, Mdot_kgps=None,
                            irradiate=True, beta_grav=0.08,
                            hotspot_T_h=None, hotspot_L_h_deg=5.0,
                            n_sec=20000, n_disc=(80, 160), n_disc_z=20,
                            n_disc_irrad=(30, 60), irrad_chunk=150, u_disc=0.0,
                            u_primary=0.0, n_primary_irrad=200, n_primary=200,
                            theta_1=0.0, phi_1=0.0, angle_acc=None, spot_acc=np.radians(5.0),
                            T_acc=100000.0, u_acc=0.0, incl_deg=None, stream_angle_deg=None):
    """
    Build the TemperatureMaps consumed by render_system_image and (for a
    single call spanning many phases) physical_light_curve.

    stream_angle_deg: forwarded to stream.integrate_stream (see its own
    docstring) for the ballistic-stream integration this builds
    internally -- how far around the primary the trajectory is allowed
    to extend before stopping. Doesn't change how much of it actually
    gets RENDERED (still truncated at the disc impact point or first
    closest approach, whichever's found -- see the stream section
    below), only how far it's available for the accretion-spot
    connection (angle_acc, possibly beyond where the default stopping
    condition would have reached) and for the closest-approach/final-
    radius diagnostic integrate_stream itself prints.

    irradiate (default True): give the secondary its full irradiated
    temperature map (gravity darkening plus absorbed flux from the
    disc-occulted star and the star-occulted disc, fully thermalized --
    see irradiation.secondary_irradiated_teff), rather than gravity
    darkening alone. Costs an O(n_disc_irrad*n_sec) calculation; set
    False for a cheaper preview that ignores irradiation.

    n_sec: number of roughly equal-area sample points on the secondary's
    surface (see roche.RocheLobe.equal_area_sample) -- a flat point
    cloud, not a (theta,phi) grid, so a fixed sample budget isn't wasted
    on tiny near-polar cells the way a rectangular grid's would be.

    Disc temperature model: pass `disc_teff_func(r_over_a) -> T[K]`
    directly (e.g. a disc_powerlaw_teff(..., T_0=..., beta_disc=...)
    closure) for full control; if omitted, falls back to the steady-state
    profile built from a_meters/M1_kg/Mdot_kgps (all three then required).

    hotspot_T_h/hotspot_L_h_deg: stream-impact hot spot on the outer rim
    (only meaningful for a flared disc); see disc_surfaces_with_teff's
    docstring for the max(T_base, T_h*exp(-dphi/L_h)) formula.
    hotspot_T_h=None (default) disables it. The impact azimuth phi_h is
    NOT a free parameter here -- it's determined by the mass ratio (via
    the L1 nozzle/Roche geometry) and the disc's own shape, so it is
    computed automatically from stream.impact_azimuth(lobe, disc).

    n_disc_irrad (r,nu) and n_disc_z set the (coarser) disc grid used for
    the irradiation integral -- independent of n_disc, which sets the
    (finer) grid used for the disc's own temperature map (disc_T).

    u_disc/u_primary: each body's limb-darkening coefficient -- affects
    not just its own observer-facing flux (render.physical_light_curve)
    but also how strongly it irradiates the secondary
    (irradiation.disc_irradiation_flux/star_irradiation_flux), so both
    feed into Tsec here.

    n_primary_irrad: number of surface points sampled over the primary
    for its own (coarser, cheaper) irradiation integral -- independent of
    n_primary (the observer-facing render/flux resolution below); see
    irradiation.star_irradiation_flux, the primary
    analogue of n_disc_irrad on the disc side.

    n_primary: number of roughly equal-area sample points over the
    primary's WHOLE surface (lightcurve.sphere_points_full -- the same
    golden-angle-spiral equal-area technique as
    roche.RocheLobe.equal_area_sample, used for the secondary), replacing
    the old per-phase, hemisphere-only, foreshortened-projected-area
    resampling (lightcurve.star_points): a fixed point cloud computed
    once here and reused every phase (render_system_image/
    physical_light_curve backface-cull it via normal.n_hat>0, exactly
    like the secondary's sec_pts/sec_normals). Only the DIRECTION pattern
    (primary_normals below) is stored at unit radius, not scaled by R1,
    so that R_1 stays a cheap live parameter at every call site (points =
    center1 + R1*primary_normals, areas = primary_base_areas*R1**2) --
    consistent with simulate.py's _CHEAP_SYSTEM_FIT_FIELDS, which lets
    R_1 vary across fit trials without rebuilding this TemperatureMaps.

    theta_1/phi_1 [rad]: the primary's magnetic-axis obliquity/azimuth
    (params.SystemParams.theta_1_rad/phi_1_rad -- see magnetic.py). Only
    matter here if angle_acc is also given (they orient the accretion spot);
    otherwise unused by this function (still used by the outline output's
    field-line display, independent of temp_maps -- see plots.py).

    angle_acc/spot_acc/T_acc/u_acc: the accretion spot(s) (params.ModelParams'
    fields of the same name -- see magnetic.field_line_to_point,
    accretion_spot_mask). angle_acc [units of a] is the primary-centered
    radius where the ballistic stream is assumed to hand off to the
    primary's magnetic field -- a single float, or an array-like of them
    (params._parse_angle_acc, e.g. simulate.py's --angle_acc), one heated spot
    per entry, all sharing the same spot_acc/T_acc/u_acc below (only the
    connection radius differs between them); None (default) disables the
    whole feature. spot_acc [rad] is each heated spot's angular radius
    around its field line's surface footpoint. T_acc [K] is every spot's
    temperature, u_acc its own limb-darkening coefficient (independent of
    u_primary -- negative values give limb-brightening, a crude stand-in
    for cyclotron beaming) -- both heat/shade both the primary's own
    observer-facing flux/image (live, so T_1/u_primary themselves stay
    cheap fit parameters -- see TemperatureMaps.primary_spot_mask, a
    single mask that's the union of every spot) and, baked in here at
    build time (same as every other irradiation input), the secondary's
    irradiated temperature map below. Any entry is individually ignored,
    with an explanatory print(), if there's no accretion stream to
    connect (detached/underfilling secondary), that entry is smaller than
    the stream's own minimum approach to the primary, or T_acc is not
    hotter than T_eff1 -- see _accretion_spot_data.

    incl_deg: if given, the disc's own (observer-facing) n_disc budget is
    re-split between its face and rim by disc_face_rim_split -- weighted
    by PROJECTED (inclination-dependent) area rather than true area, so
    the rim isn't over-resolved relative to the face just because a
    realistic opening angle makes its true area small (see that
    function's docstring). Only affects the disc's own temperature map
    (disc_pts/disc_T/disc_areas below), not n_disc_irrad's irradiation
    grid, which has no dependence on the observer's viewing angle at all.
    Omit (the default) to keep the old area-only n_z*n_nu-derived wall
    shape, e.g. for callers that don't have an inclination on hand.
    """
    if disc_teff_func is None:
        if a_meters is None or M1_kg is None or Mdot_kgps is None:
            raise ValueError("pass disc_teff_func, or all of a_meters/M1_kg/Mdot_kgps")
        def disc_teff_func(r):
            return disc_steady_state_teff(r, a_meters, M1_kg, Mdot_kgps, disc.r_in)

    flared = disc.opening_angle > 0.0

    hotspot_phi_h = None
    if hotspot_T_h is not None:
        from stream import impact_azimuth
        hotspot_phi_h = impact_azimuth(lobe, disc)
        if hotspot_phi_h is None:  # stream never reaches this disc's rim
            hotspot_T_h = None

    # --- primary: fixed direction pattern (unit sphere) for the observer-
    #     facing render/flux resolution -- see n_primary's docstring above
    #     for why only the direction, not the R1-scaled points, is stored ---
    _, primary_normals, primary_base_areas = sphere_points_full(np.zeros(3), 1.0, n_primary)

    # --- accretion stream: L1 to the disc impact point (or, failing an
    #     impact -- e.g. no disc at all, see disc_impact_index's None
    #     return -- the ballistic trajectory's closest approach to the
    #     primary). Moved before the secondary's irradiation below (rather
    #     than after, its more natural narrative position) so the
    #     accretion-spot connection point found from `traj` here (see
    #     _accretion_spot_data) can also irradiate the secondary, not just
    #     be baked into the primary's own temperature.
    # skipped entirely for a detached/underfilling secondary
    # (lobe.fill_factor<1): there's no Roche-lobe overflow at L1, so no
    # stream integration corresponds to any real mass flow. Deliberately
    # NOT gated on disc.is_empty: a magnetic CV (see AMHer.yaml) can have
    # a real stream with no disc at all, channeled by the primary's
    # magnetic field onto its pole instead of spreading into a disc.
    if lobe.fill_factor < 1.0:
        stream_pts = np.zeros((0, 3))
        stream_areas = np.zeros(0)
        traj = None
    else:
        from stream import integrate_stream, disc_impact_index, closest_approach_index, sample_points
        # extend the integration limit (if needed) to guarantee the
        # trajectory actually sweeps far enough to reach every requested
        # accretion-connection angle -- see accretion_connection_line's
        # own matching extension, which has no effect here since the
        # traj built below is passed into _accretion_spot_data already
        # built, not left for accretion_connection_line to build itself.
        eff_stream_angle = stream_angle_deg
        if angle_acc is not None:
            max_angle_acc = float(np.max(np.atleast_1d(angle_acc)))
            eff_stream_angle = max(stream_angle_deg, max_angle_acc) \
                if stream_angle_deg is not None else max_angle_acc
        traj = integrate_stream(lobe, stream_angle_deg=eff_stream_angle)
        # always reported (see disc_impact_index's own report= option),
        # even when its result isn't used for truncation below
        disc_idx = disc_impact_index(traj, disc.rim, lobe.x1, report=True)
        if stream_angle_deg is not None:
            # an explicit stream_angle_deg (see simulate.py's --stream_angle)
            # is the user's own authoritative instruction for how far the
            # stream extends -- e.g. to simulate overflow past where it
            # would otherwise hit the disc (see STREAM tab's own note) --
            # so use the WHOLE computed trajectory (as both a visible
            # ribbon and a physical contributor to the temperature map/
            # light curve) rather than second-guessing it with the
            # disc-impact/closest-approach heuristics below, which exist
            # only to pick a sensible endpoint when stream_angle_deg wasn't
            # given at all.
            idx = len(traj["x"]) - 1
        else:
            idx = disc_idx
            if idx is None:
                idx = closest_approach_index(traj, lobe.x1)
            if idx is None:
                idx = len(traj["x"]) - 1
        n_along = 300
        s_max = traj["s"][idx]
        s_vals = np.linspace(0.0, s_max, n_along)
        xs_s, ys_s = sample_points(traj, s_vals)

        # a real stream has finite cross-section; give it a small but nonzero
        # width (R1 is a convenient, physically-motivated scale) sampled as a
        # few parallel strands, rather than a literal zero-width line -- a
        # single-pixel-wide line is nearly invisible against the disc/
        # secondary (especially since stream_T, set by the L1 cusp, tends to
        # be the coldest point in the whole image, i.e. the darkest color).
        n_strands = 5
        stream_width = R1
        tangent = np.gradient(np.stack([xs_s, ys_s], axis=-1), axis=0)
        tangent /= np.maximum(np.linalg.norm(tangent, axis=-1, keepdims=True), 1e-300)
        perp = np.stack([-tangent[:, 1], tangent[:, 0]], axis=-1)
        offsets = np.linspace(-0.5, 0.5, n_strands) * stream_width
        xs_ribbon = xs_s[:, None] + offsets[None, :] * perp[:, 0:1]
        ys_ribbon = ys_s[:, None] + offsets[None, :] * perp[:, 1:2]
        stream_pts = np.stack([xs_ribbon.ravel(), ys_ribbon.ravel(),
                                np.zeros(n_along * n_strands)], axis=-1)
        stream_areas = np.full(n_along * n_strands,
                                (stream_width / n_strands) * (s_max / n_along))

    # --- accretion spot: connect the stream to a magnetic field line at
    #     radius angle_acc (see this function's own docstring, magnetic.py) ---
    primary_spot_mask, accretion_footpoint_dirs, accretion_field_lines = _accretion_spot_data(
        lobe, R1, T_eff1, theta_1, phi_1, angle_acc, spot_acc, T_acc, primary_normals, traj=traj,
        stream_angle_deg=stream_angle_deg)

    # --- secondary: irradiated (or plain gravity-darkened) temperature map ---
    sec_pts, sec_normals, sec_areas = lobe.equal_area_sample(n_sec)
    if irradiate:
        from irradiation import secondary_irradiated_teff
        if flared:
            xi, yi, zi, ni, areai, Ti_disc, _sid = disc_surfaces_with_teff(
                disc, disc_teff_func, n_r=n_disc_irrad[0], n_nu=n_disc_irrad[1], n_z=n_disc_z,
                T_h=hotspot_T_h, phi_h=hotspot_phi_h, L_h_deg=hotspot_L_h_deg)
            disc_pos_irrad = np.stack([xi, yi, zi], axis=-1)
            disc_normal_irrad = ni
        else:
            xi, yi, areai, Ti_disc = disc_grid_with_teff(disc, disc_teff_func,
                                                          n_r=n_disc_irrad[0], n_nu=n_disc_irrad[1])
            disc_pos_irrad = np.stack([xi, yi, np.zeros_like(xi)], axis=-1)
            disc_normal_irrad = None
        Tsec = secondary_irradiated_teff(lobe, sec_pts, sec_normals, sec_areas, T_eff2, T_eff1, R1,
                                          disc_pos_irrad, areai, Ti_disc,
                                          disc=disc, disc_normal=disc_normal_irrad,
                                          beta_grav=beta_grav, disc_chunk=irrad_chunk, u_disc=u_disc,
                                          u_primary=u_primary, n_primary_irrad=n_primary_irrad,
                                          accretion_footpoint_dirs=accretion_footpoint_dirs,
                                          spot_acc=spot_acc, T_acc=T_acc, u_acc=u_acc)
    else:
        from roche import gravity_darkened_teff
        Tsec = gravity_darkened_teff(sec_pts, sec_areas, lobe.q, T_eff2, beta_grav=beta_grav)

    # the L1 sample is a genuine cusp of the potential (no well-defined
    # local temperature); fix_l1_cusp() is otherwise only applied inside
    # secondary_irradiated_teff, so re-apply it unconditionally here to
    # also cover the irradiate=False path, and to get a sensible constant
    # "temperature of the L1 point" for the accretion stream below.
    from irradiation import fix_l1_cusp
    Tsec = fix_l1_cusp(Tsec, lobe, points=sec_pts)
    i_l1 = lobe.l1_nearest_index(sec_pts)
    stream_T = float(Tsec[i_l1])

    # --- disc: temperature map (incl. hot spot), fixed in the corotating frame ---
    n_r, n_nu = n_disc
    if flared:
        n_wall = None
        if incl_deg is not None:
            (n_r, n_nu), n_wall = disc_face_rim_split(
                n_r * n_nu, disc.r_in, disc.a, disc.opening_angle, incl_deg)
        xs, ys, zs, ns, disc_areas, Td, _sid = disc_surfaces_with_teff(
            disc, disc_teff_func, n_r=n_r, n_nu=n_nu, n_z=n_disc_z, n_wall=n_wall,
            T_h=hotspot_T_h, phi_h=hotspot_phi_h, L_h_deg=hotspot_L_h_deg)
        disc_pts = np.stack([xs, ys, zs], axis=-1)
        disc_normals = ns
    else:
        xs, ys, disc_areas, Td = disc_grid_with_teff(disc, disc_teff_func, n_r=n_r, n_nu=n_nu)
        disc_pts = np.stack([xs, ys, np.zeros_like(xs)], axis=-1)
        disc_normals = None

    return TemperatureMaps(Tsec=Tsec, sec_areas=sec_areas, sec_pts=sec_pts, sec_normals=sec_normals,
                            primary_normals=primary_normals, primary_base_areas=primary_base_areas,
                            primary_spot_mask=primary_spot_mask,
                            accretion_footpoint_dirs=accretion_footpoint_dirs,
                            accretion_field_lines=accretion_field_lines,
                            disc_pts=disc_pts, disc_normals=disc_normals, disc_T=Td,
                            disc_areas=disc_areas, disc_cell_size=(disc.a - disc.r_in) / n_r,
                            stream_pts=stream_pts, stream_areas=stream_areas, stream_T=stream_T)


def save_temperature_maps(path, temp_maps, system, model, ntheta, nphi, irradiate):
    """
    Save a TemperatureMaps (the expensive, phase-independent result of
    build_temperature_maps) plus every SystemParams/ModelParams value and
    the grid-resolution knobs needed to reconstruct lobe/disc, to a
    multi-HDU FITS file. Load it back with load_temperature_maps() to
    produce more light curves/renders (any phase, any wavelength -- both
    cheap, downstream steps) without repeating the irradiation
    calculation.

    HDUs: PRIMARY (header only -- metadata), SEC (bintable:
    X,Y,Z,NX,NY,NZ,TEMP,AREA -- the secondary's equal-area sample point
    cloud, see roche.RocheLobe.equal_area_sample), DISC (bintable:
    X,Y,Z,NX,NY,NZ,TEMP,AREA -- NX/NY/NZ columns omitted for a flat
    disc), STREAM (bintable: X,Y,Z,AREA).
    """
    import os
    from astropy.io import fits
    import params as _params

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    hdr = _params.metadata_header(
        system, model,
        IRRADIAT=(bool(irradiate), "secondary irradiation included"),
        NTHETA=(ntheta, "secondary radius() grid, theta resolution"),
        NPHI=(nphi, "secondary radius() grid, phi resolution"),
        NSEC=(len(temp_maps.Tsec), "secondary equal-area sample count"),
        DISCCELL=(temp_maps.disc_cell_size, "units of a, disc radial grid spacing"),
        STREAMT=(temp_maps.stream_T, "K, stream temperature (T at L1)"),
    )
    hdus = [fits.PrimaryHDU(header=hdr)]

    sec_cols = [
        fits.Column(name="X", format="D", array=temp_maps.sec_pts[:, 0]),
        fits.Column(name="Y", format="D", array=temp_maps.sec_pts[:, 1]),
        fits.Column(name="Z", format="D", array=temp_maps.sec_pts[:, 2]),
        fits.Column(name="NX", format="D", array=temp_maps.sec_normals[:, 0]),
        fits.Column(name="NY", format="D", array=temp_maps.sec_normals[:, 1]),
        fits.Column(name="NZ", format="D", array=temp_maps.sec_normals[:, 2]),
        fits.Column(name="TEMP", format="D", array=temp_maps.Tsec),
        fits.Column(name="AREA", format="D", array=temp_maps.sec_areas),
    ]
    hdus.append(fits.BinTableHDU.from_columns(sec_cols, name="SEC"))

    # named "STAR", not "PRIMARY" -- the latter is reserved by FITS/astropy
    # for HDU 0 (the metadata header above) and would collide with it under
    # hdul["PRIMARY"] lookup.
    # primary_spot_mask/accretion_footpoint_dirs/accretion_field_lines are
    # NOT saved: unlike Tsec/disc_T (an expensive O(Ndisc_irrad*Nsec)
    # integral, the whole reason this cache exists), the accretion spot(s)
    # are cheap to recompute (one stream integration + a closed-form
    # field-line trace each), so load_temperature_maps just reruns
    # _accretion_spot_data from the header's own angle_acc/theta_1/phi_1/
    # spot_acc/T_acc -- one source of truth, no risk of the cached mask
    # drifting from a newer definition. Note this means only the FIRST
    # --angle_acc entry survives a save/load round-trip (the header only
    # stores ModelParams.angle_acc's single value, see params._parse_angle_acc) --
    # a --load-irradiation cache never reconstructs more than one spot.
    star_cols = [
        fits.Column(name="NX", format="D", array=temp_maps.primary_normals[:, 0]),
        fits.Column(name="NY", format="D", array=temp_maps.primary_normals[:, 1]),
        fits.Column(name="NZ", format="D", array=temp_maps.primary_normals[:, 2]),
        fits.Column(name="AREA", format="D", array=temp_maps.primary_base_areas),
    ]
    hdus.append(fits.BinTableHDU.from_columns(star_cols, name="STAR"))

    disc_cols = [
        fits.Column(name="X", format="D", array=temp_maps.disc_pts[:, 0]),
        fits.Column(name="Y", format="D", array=temp_maps.disc_pts[:, 1]),
        fits.Column(name="Z", format="D", array=temp_maps.disc_pts[:, 2]),
    ]
    if temp_maps.disc_normals is not None:
        disc_cols += [
            fits.Column(name="NX", format="D", array=temp_maps.disc_normals[:, 0]),
            fits.Column(name="NY", format="D", array=temp_maps.disc_normals[:, 1]),
            fits.Column(name="NZ", format="D", array=temp_maps.disc_normals[:, 2]),
        ]
    disc_cols += [
        fits.Column(name="TEMP", format="D", array=temp_maps.disc_T),
        fits.Column(name="AREA", format="D", array=temp_maps.disc_areas),
    ]
    hdus.append(fits.BinTableHDU.from_columns(disc_cols, name="DISC"))

    stream_cols = [
        fits.Column(name="X", format="D", array=temp_maps.stream_pts[:, 0]),
        fits.Column(name="Y", format="D", array=temp_maps.stream_pts[:, 1]),
        fits.Column(name="Z", format="D", array=temp_maps.stream_pts[:, 2]),
        fits.Column(name="AREA", format="D", array=temp_maps.stream_areas),
    ]
    hdus.append(fits.BinTableHDU.from_columns(stream_cols, name="STREAM"))

    fits.HDUList(hdus).writeto(path, overwrite=True)


def load_temperature_maps(path, stream_angle_deg=None):
    """
    Load a TemperatureMaps saved by save_temperature_maps(), rebuilding
    lobe/disc at the same grid resolution used when it was saved.

    stream_angle_deg: forwarded to the accretion-spot's own (re-integrated,
    not persisted -- see below) stream trajectory; see
    stream.integrate_stream's own docstring.

    Returns (lobe, disc, temp_maps, system, model, irradiate). `system`
    and `model` are ready to hand to params.params_from_args(args,
    base=(system, model)) so CLI flags (e.g. --wavelength) can still
    override individual fields before use.
    """
    from astropy.io import fits
    import params as _params

    with fits.open(path) as hdul:
        hdr = hdul[0].header
        system, model = _params.params_from_fits_header(hdr)
        irradiate = bool(hdr["IRRADIAT"])
        ntheta, nphi = int(hdr["NTHETA"]), int(hdr["NPHI"])
        disc_cell_size = float(hdr["DISCCELL"])
        stream_T = float(hdr["STREAMT"])

        sec_data = hdul["SEC"].data
        sec_pts = np.stack([sec_data["X"], sec_data["Y"], sec_data["Z"]], axis=-1)
        sec_normals = np.stack([sec_data["NX"], sec_data["NY"], sec_data["NZ"]], axis=-1)
        Tsec = np.asarray(sec_data["TEMP"])
        sec_areas = np.asarray(sec_data["AREA"])

        if "STAR" not in hdul:
            raise ValueError(
                f"{path!r} has no STAR HDU -- it was saved before the primary's full-sphere "
                "sample cloud was added to TemperatureMaps; regenerate the cache "
                "(--save-irradiation) with the current code")
        star_data = hdul["STAR"].data
        primary_normals = np.stack([star_data["NX"], star_data["NY"], star_data["NZ"]], axis=-1)
        primary_base_areas = np.asarray(star_data["AREA"])

        disc_data = hdul["DISC"].data
        disc_pts = np.stack([disc_data["X"], disc_data["Y"], disc_data["Z"]], axis=-1)
        if "NX" in disc_data.columns.names:
            disc_normals = np.stack([disc_data["NX"], disc_data["NY"], disc_data["NZ"]], axis=-1)
        else:
            disc_normals = None
        disc_T = np.asarray(disc_data["TEMP"])
        disc_areas = np.asarray(disc_data["AREA"])

        stream_data = hdul["STREAM"].data
        stream_pts = np.stack([stream_data["X"], stream_data["Y"], stream_data["Z"]], axis=-1)
        stream_areas = np.asarray(stream_data["AREA"])

    lobe, disc, _disc_teff_func = _params.build_system(system, model, ntheta=ntheta, nphi=nphi)

    # recompute (cheap -- see save_temperature_maps' comment) rather than
    # persist: uses the loaded system/model's own angle_acc/theta_1/phi_1/
    # spot_acc/T_acc, so a --load-irradiation cache still shows/applies
    # the correct accretion spot.
    primary_spot_mask, accretion_footpoint_dirs, accretion_field_lines = _accretion_spot_data(
        lobe, system.R_1, system.T_1, system.theta_1_rad, system.phi_1_rad,
        model.angle_acc, model.spot_acc_rad, model.T_acc, primary_normals,
        stream_angle_deg=stream_angle_deg)

    temp_maps = TemperatureMaps(Tsec=Tsec, sec_areas=sec_areas, sec_pts=sec_pts, sec_normals=sec_normals,
                                 primary_normals=primary_normals, primary_base_areas=primary_base_areas,
                                 primary_spot_mask=primary_spot_mask,
                                 accretion_footpoint_dirs=accretion_footpoint_dirs,
                                 accretion_field_lines=accretion_field_lines,
                                 disc_pts=disc_pts, disc_normals=disc_normals, disc_T=disc_T,
                                 disc_areas=disc_areas, disc_cell_size=disc_cell_size,
                                 stream_pts=stream_pts, stream_areas=stream_areas, stream_T=stream_T)

    return lobe, disc, temp_maps, system, model, irradiate


def _magnetic_stream_ribbon(line, R1, center1, T_acc, stream_T, n_strands=5):
    """
    A small ribbon of points along one accretion field line (an (n,3)
    curve, primary surface to its stream connection point -- see
    accretion_connection_line), for treating it as emitting material:
    n_strands parallel strands offset sideways (the local radial
    direction from `center1` crossed with the line's own tangent, since a
    field line isn't confined to any one plane like the ballistic stream
    is) by up to R1/2 -- same reasoning as the ballistic stream's own
    ribbon (a literal zero-width line is nearly invisible/carries no
    area) -- and a LOGARITHMIC temperature grade along the line -- log(T)
    linear in position, i.e. T_acc*(stream_T/T_acc)**s -- from T_acc at
    the surface end (line[0]) to stream_T at the connection-point end
    (line[-1]): see render_system_image's own docstring for why log, not
    linear.

    Returns (pts, areas, T): pts is (n*n_strands,3), areas/T are
    (n*n_strands,) -- shared by render_system_image (which projects and
    occlusion-tests these before plotting) and radial_velocity_curve
    (which weights them by intensity instead; each ribbon point's own
    slightly-offset position, not just the field line's own centerline,
    is what actually goes into that weighting's rigid-rotation velocity,
    which only makes it more correct, not less -- see that function's
    own docstring).
    """
    n_pts = line.shape[0]
    tangent = np.gradient(line, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=-1, keepdims=True), 1e-300)
    radial = line - center1
    radial /= np.maximum(np.linalg.norm(radial, axis=-1, keepdims=True), 1e-300)
    perp = np.cross(tangent, radial)
    perp /= np.maximum(np.linalg.norm(perp, axis=-1, keepdims=True), 1e-300)

    width = R1
    offsets = np.linspace(-0.5, 0.5, n_strands) * width
    pts = (line[:, None, :] + offsets[None, :, None] * perp[:, None, :]).reshape(-1, 3)

    seg_len = np.linalg.norm(np.diff(line, axis=0), axis=-1)
    avg_ds = seg_len.sum() / max(n_pts - 1, 1)
    areas = np.full(n_pts * n_strands, (width / n_strands) * avg_ds)

    s = np.arange(n_pts) / max(n_pts - 1, 1)
    T_line = T_acc * (stream_T / T_acc) ** s
    T = np.repeat(T_line, n_strands)
    return pts, areas, T


def _magnetic_stream_velocity(line, lobe, center1, traj):
    """
    The relative (corotating-frame) velocity of material threading down
    one accretion field line (line[0]=primary surface, line[-1]=stream
    connection point -- see accretion_connection_line), at each of its
    points: a "bead on a wire" constrained to follow the line itself,
    starting from the ballistic stream's own velocity at the connection
    point (already known -- traj is stream.integrate_stream's result),
    projected onto the line's own tangent there, then evolved along the
    line by energy conservation in the SAME rotating-frame potential
    (roche.potential) that governs the ballistic stream's own motion
    (stream._eom).

    This is exactly right, not an approximation: stream._eom's
    acceleration is -grad(Phi) - 2*Omega x v (gravity+centrifugal, then
    Coriolis), and the Jacobi integral 0.5*|v|^2 + Phi(r) is conserved
    along any such trajectory because the Coriolis term is always
    perpendicular to v and so does no work -- true whether or not
    anything else (here, magnetic tension) also constrains the path,
    since a constraint force normal to the motion does no work either.
    So the along-line SPEED only cares about Phi, even though the actual
    3D path (and the Coriolis force needed to bend it) is set by the
    field line's own shape instead of by gravity+centrifugal alone.

    Returns (n,3): each point's relative velocity vector, tangent to the
    line, pointing toward the surface (line[0]) -- i.e. the direction of
    actual motion, infall.
    """
    from roche import potential

    tangent = np.gradient(line, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=-1, keepdims=True), 1e-300)
    down = -tangent  # points from the connection end toward the surface

    Phi = potential(line[:, 0], line[:, 1], line[:, 2], lobe.q)
    Phi_conn = Phi[-1]

    # line[-1] IS the connection point angle_acc_index originally found in
    # traj (see accretion_connection_line) -- recover that same index by
    # nearest (x,y) match rather than re-deriving a radius or angle from
    # it: an angle recovered via arctan2 would wrap (mod 360), silently
    # finding the WRONG, much-earlier point whenever angle_acc's own
    # value exceeds 360 (the trajectory loops around more than once).
    if traj is not None:
        dist_to_conn = np.hypot(traj["x"] - line[-1, 0], traj["y"] - line[-1, 1])
        i_acc = int(np.argmin(dist_to_conn))
    else:
        i_acc = None
    if i_acc is None:
        v_conn_vec = np.zeros(3)
    else:
        v_conn_vec = np.array([traj["vx"][i_acc], traj["vy"][i_acc], 0.0])
    # the ballistic velocity's component already heading down the line;
    # floored at 0 rather than let a badly-aligned connection (should not
    # happen for a field line that genuinely connects to the stream) give
    # a negative "initial speed" energy conservation can't make sense of
    v0 = max(float(np.dot(v_conn_vec, down[-1])), 0.0)

    speed = np.sqrt(np.maximum(v0 ** 2 + 2.0 * (Phi_conn - Phi), 0.0))
    return speed[:, None] * down


def render_system_image(lobe, disc, T_eff1, R1, phase, incl_deg, temp_maps,
                         quantity="temperature", wavelength_m=V_WAVELENGTH_M,
                         image_size=500, extent=None,
                         pixel_mapping="direct",
                         u_primary=0.0, u_secondary=0.0, u_disc=0.0, T_acc=100000.0,
                         u_acc=0.0):
    """
    Render the system (secondary + primary + disc + accretion stream) at
    one orbital phase from a precomputed TemperatureMaps
    (build_temperature_maps) -- for a single still image just do

        maps = build_temperature_maps(lobe, disc, T_eff1, T_eff2, R1, ...)
        img, xedges, yedges = render_system_image(lobe, disc, T_eff1, R1, phase, incl_deg, maps)

    and for a movie, build `maps` once and call this once per frame: the
    expensive irradiation/disc-temperature calculation in
    build_temperature_maps is not repeated, only occlusion and the final
    projection/binning (both cheap and genuinely phase-dependent) are.
    The default extent (see `extent`'s own entry below, and auto_extent)
    is already phase-independent -- every frame gets the same field of
    view without passing anything explicitly -- but it re-derives that
    same answer via a 73-phase sweep on every call; for many frames,
    compute it once with auto_extent(...) and pass the result as `extent`
    to skip repeating that sweep.

    `quantity` picks what ends up in the returned image:

      "temperature" -- temperature directly (default); u_primary/
        u_secondary/u_disc have no effect here, since limb darkening is a
        brightness effect, not a temperature one.
      "v_band"      -- band_intensity(T, wavelength_m) * (1-u+u*mu), each
        point's own limb-darkening factor (see _phase_chunk_flux's
        docstring for the same (1-u+u*mu) law used in the light curve),
        mu = cos(angle between the point's local outward normal and the
        line of sight -- the same foreshortening cosine already used for
        that body's occlusion/visibility test, so no extra geometry is
        computed. This is the quantity to render when you want to *see*
        what a nonzero limb-darkening coefficient does to a body's disc,
        rather than just its integrated effect on the light curve.

    Using the same pipeline for both means the two representations can
    never drift out of sync with each other's occlusion/geometry -- the
    only difference is the very last step, bin T directly or convert each
    point's T through the Planck function (and, for "v_band", the limb
    factor) first.

    T_eff1: primary effective temperature [K]. R1: primary radius, units of a.
    T_acc: accretion spot temperature [K], applied only where
    temp_maps.primary_spot_mask is set (see build_temperature_maps'
    angle_acc/spot_acc) -- live here (like T_eff1) rather than baked into
    temp_maps, so it stays a cheap parameter to vary without rebuilding.

    The accretion stream (temp_maps.stream_pts, L1 to the disc-impact
    point) is rendered at the single constant temp_maps.stream_T -- "the
    temperature of the L1 point" -- since the stream isn't otherwise
    given its own thermal model (see build_temperature_maps). Each active
    --angle_acc entry's magnetically-channeled continuation
    (temp_maps.accretion_field_lines, the primary's surface out to that
    entry's stream connection point) is rendered too, logarithmically
    graded (log(T) linear along the line) from T_acc at the surface end
    to temp_maps.stream_T at the connection-point end -- the same live
    T_acc as the primary's own spot, so this segment always matches
    whatever's currently heating it.

    If disc.opening_angle>0, the disc is rendered as the full flared solid
    (four surfaces, see disc.all_surfaces_grid): its own self-occlusion is
    included (disc.visible -> disc.occluded_by_disc). If opening_angle==0,
    everything uses the flat, zero-thickness treatment (disc.visible_from).

    wavelength_m: observing wavelength [m] used for quantity="v_band"
    (default V_WAVELENGTH_M, ~5500 Angstrom); has no effect on
    quantity="temperature".

    pixel_mapping: how surface samples become image pixels.
      "direct" (default) -- each sample's own projected (foreshortened)
        area is compared against the pixel area and splatted across
        every pixel it actually covers, with a depth buffer resolving
        overlaps between bodies (_bin_image_direct); no holes, at
        significantly higher cost.
      "indirect" -- each sample scatters into whichever single pixel it
        projects onto (_bin_image_indirect); cheap, but can leave "holes"
        where the sample grid is coarser than the pixel grid.

    extent: None (default) -- tightly fit the pixel grid to auto_extent's
    phase-independent envelope of the system's full geometry (not just
    this call's own visible content, which would make the field of view
    zoom/pan between phases), plus 5% margin on each side; a scalar --
    the old fixed square [-extent,extent] x [-extent,extent] behavior,
    used exactly as given (no extra margin); or a (xmin,xmax,ymin,ymax)
    4-tuple for an explicit rectangular field of view (also used exactly
    as given), e.g. auto_extent's own result precomputed once and reused
    across many frames (see this function's docstring above).

    Pixels are always kept square (same physical size in X and Y, so
    shapes aren't stretched) -- see _pixel_edges. `image_size` sets the
    pixel count along the longer of the (possibly unequal) X/Y spans; the
    shorter axis gets proportionally fewer pixels at the same size,
    rather than a fixed image_size x image_size grid wasting pixels on
    empty space whenever a system's sky-projected X and Y extents differ
    a lot (typical for a near-edge-on system).

    Returns (img, xedges, yedges): img is (len(yedges)-1, len(xedges)-1)
    indexed [iy,ix]; xedges/yedges are the pixel-edge coordinates, for
    imshow(img, extent=(xedges[0], xedges[-1], yedges[0], yedges[-1])).
    """
    if quantity not in ("temperature", "v_band"):
        raise ValueError("quantity must be 'temperature' or 'v_band'")

    incl = np.radians(incl_deg)
    n, eX, eY = observer_frame(phase, incl)
    x1 = lobe.x1
    center1 = np.array([x1, 0.0, 0.0])
    flared = disc.opening_angle > 0.0

    Xs_list, Ys_list, Ts_list, Ds_list, Hx_list, Hy_list, Lm_list = [], [], [], [], [], [], []

    # --- secondary: near hemisphere, occulted by the disc and the primary ---
    pts = temp_maps.sec_pts
    normals = temp_maps.sec_normals
    mu_sec = np.einsum("...i,i->...", normals, n)
    visible = ((mu_sec > 0.0) & disc.visible(pts, phase, incl)
               & sphere_visible_mask(pts, phase, incl, center1, R1))
    Xp, Yp, Dp = project(pts[visible], phase, incl)
    Ap = temp_maps.sec_areas[visible] * mu_sec[visible]
    Hp = 0.5 * np.sqrt(np.maximum(Ap, 0.0))
    Lp = 1.0 - u_secondary + u_secondary * mu_sec[visible]
    Xs_list.append(Xp); Ys_list.append(Yp); Ts_list.append(temp_maps.Tsec[visible])
    Ds_list.append(Dp); Hx_list.append(Hp); Hy_list.append(Hp); Lm_list.append(Lp)

    # --- primary: uniform T_eff1 except the accretion spot (T_acc, see
    #     temp_maps.primary_spot_mask/build_temperature_maps' angle_acc),
    #     occulted by secondary lobe + disc -- fixed direction pattern
    #     (temp_maps.primary_normals), same backface-cull + occlusion
    #     pattern as the secondary above ---
    normals1 = temp_maps.primary_normals
    pts1 = center1 + R1 * normals1
    area1 = temp_maps.primary_base_areas * R1 ** 2
    mu1 = np.einsum("ij,j->i", normals1, n)
    vis1 = (mu1 > 0.0) & visible_mask_bulk(pts1, phase, incl, lobe) & disc.visible(pts1, phase, incl)
    X1, Y1, D1 = project(pts1[vis1], phase, incl)
    T1 = np.where(temp_maps.primary_spot_mask[vis1], T_acc, T_eff1)
    Ap1 = area1[vis1] * mu1[vis1]
    H1 = 0.5 * np.sqrt(np.maximum(Ap1, 0.0))
    u1_live = np.where(temp_maps.primary_spot_mask[vis1], u_acc, u_primary)
    L1 = 1.0 - u1_live + u1_live * mu1[vis1]
    Xs_list.append(X1); Ys_list.append(Y1); Ts_list.append(T1)
    Ds_list.append(D1); Hx_list.append(H1); Hy_list.append(H1); Lm_list.append(L1)

    # --- disc: occulted by secondary lobe, and (if flared) by itself ---
    pts_d, ns, Td = temp_maps.disc_pts, temp_maps.disc_normals, temp_maps.disc_T
    if flared:
        # nudge each point just outside the solid along its own normal so
        # the self-occlusion ray march doesn't immediately register the
        # point (which sits exactly on the boundary) as "inside" at t=0
        pts_d_test = pts_d + 1e-6 * ns
        # Each of the four surfaces bounds real solid material and is
        # only visible from its own outward side -- the ray march above
        # catches most back-facing points too (moving toward the observer
        # from a back-facing point re-enters the solid almost
        # immediately), but that re-entry can be a sliver thinner than
        # the march's t-sampling for a thin/grazing disc, so check the
        # exact analytic condition explicitly rather than rely on that:
        cos_obs_d = np.einsum("ij,j->i", ns, n)
        mu_disc = cos_obs_d > 0.0
    else:
        pts_d_test = pts_d
        mu_disc = True  # the flat disc's two faces are both legitimately visible
        cos_obs_d = np.cos(incl)  # matches physical_light_curve's flat-disc convention
    vis_d = (visible_mask_bulk(pts_d, phase, incl, lobe)
             & disc.visible(pts_d_test, phase, incl) & mu_disc
             & sphere_visible_mask(pts_d, phase, incl, center1, R1))
    Xd, Yd, Dd = project(pts_d[vis_d], phase, incl)
    cos_obs_d_vis = np.broadcast_to(cos_obs_d, temp_maps.disc_areas.shape)[vis_d]
    area_d = temp_maps.disc_areas[vis_d] * cos_obs_d_vis
    Ld = 1.0 - u_disc + u_disc * cos_obs_d_vis
    if flared:
        # isotropic sqrt(area) badly overstates the splat's extent along
        # whichever screen axis the cell's azimuthal (locally "narrow at
        # grazing angles") direction happens to foreshorten onto -- see
        # _disc_splat_halfwidths. The flat disc doesn't need this: its
        # area_d already goes to exactly 0 at i=90 (cos_obs_d=cos(incl)),
        # so it naturally vanishes instead of overshooting.
        Hxd, Hyd = _disc_splat_halfwidths(pts_d[vis_d], ns[vis_d], area_d, x1, eX, eY)
    else:
        Hxd = Hyd = 0.5 * np.sqrt(np.maximum(area_d, 0.0))
    Xs_list.append(Xd); Ys_list.append(Yd); Ts_list.append(Td[vis_d])
    Ds_list.append(Dd); Hx_list.append(Hxd); Hy_list.append(Hyd); Lm_list.append(Ld)

    # --- accretion stream: L1 to the disc impact point, at the single
    #     constant temp_maps.stream_T (sampled as a small-width ribbon,
    #     see build_temperature_maps, so it stays visible rather than
    #     rendering as a literal single-pixel-wide line) ---
    pts_s = temp_maps.stream_pts
    vis_s = visible_mask(pts_s, phase, incl, lobe) & disc.visible(pts_s, phase, incl)
    Xst, Yst, Dst = project(pts_s[vis_s], phase, incl)
    Hst = 0.5 * np.sqrt(np.maximum(temp_maps.stream_areas[vis_s], 0.0))
    Xs_list.append(Xst); Ys_list.append(Yst)
    Ts_list.append(np.full(vis_s.sum(), temp_maps.stream_T))
    Ds_list.append(Dst); Hx_list.append(Hst); Hy_list.append(Hst)
    Lm_list.append(np.ones(vis_s.sum()))  # no limb-darkening coefficient for the stream

    # --- magnetic accretion stream: one ribbon per active --angle_acc entry
    #     (temp_maps.accretion_field_lines, primary surface to that
    #     entry's stream connection point -- see build_temperature_maps'
    #     angle_acc/_magnetic_stream_ribbon), painted with a LOGARITHMIC
    #     temperature transition along the line -- log(T) linear in
    #     position, i.e. T_acc*(stream_T/T_acc)**s -- from T_acc at the
    #     surface end (points[0]) down to temp_maps.stream_T -- the
    #     free-falling stream's own characteristic temperature -- at the
    #     connection-point end (points[-1]). T_acc/stream_T typically
    #     differ by 1-2 orders of magnitude, so a plain linear ramp spends
    #     nearly its whole length within a few percent of T_acc and then
    #     drops the rest of the way in a short, visually abrupt stretch
    #     near the connection point; grading log(T) linearly instead
    #     spreads that same drop evenly (in relative, not absolute,
    #     terms) along the whole line. Rather than the invisible
    #     (unrendered) segment it was before this feature existed.
    for line in temp_maps.accretion_field_lines:
        pts_m, areas_m, T_m = _magnetic_stream_ribbon(line, R1, center1, T_acc, temp_maps.stream_T)

        vis_m = (visible_mask(pts_m, phase, incl, lobe) & disc.visible(pts_m, phase, incl)
                 & sphere_visible_mask(pts_m, phase, incl, center1, R1))
        Xm, Ym, Dm = project(pts_m[vis_m], phase, incl)
        Hm = 0.5 * np.sqrt(np.maximum(areas_m[vis_m], 0.0))
        Xs_list.append(Xm); Ys_list.append(Ym)
        Ts_list.append(T_m[vis_m])
        Ds_list.append(Dm); Hx_list.append(Hm); Hy_list.append(Hm)
        Lm_list.append(np.ones(vis_m.sum()))  # no limb-darkening coefficient, same as the stream above

    X_all = np.concatenate(Xs_list)
    Y_all = np.concatenate(Ys_list)
    T_all = np.concatenate(Ts_list)
    D_all = np.concatenate(Ds_list)
    Hx_all = np.concatenate(Hx_list)
    Hy_all = np.concatenate(Hy_list)
    Lm_all = np.concatenate(Lm_list)
    values = band_intensity(T_all, wavelength_m) * Lm_all if quantity == "v_band" else T_all

    if extent is None:
        # phase-independent: the envelope of everywhere the system could
        # ever project to (auto_extent), not just what happens to be
        # visible at this specific phase -- so a sequence of frames at
        # different phases (a movie) shares one fixed field of view
        # instead of zooming/panning as occlusion changes what's visible.
        # See _pixel_edges for the padding/pixel-count sizing.
        xmin, xmax, ymin, ymax = auto_extent(lobe, disc, R1, incl_deg, temp_maps)
        pad_frac = 0.05
    elif np.isscalar(extent):
        xmin, xmax, ymin, ymax = -extent, extent, -extent, extent
        pad_frac = 0.0
    else:
        xmin, xmax, ymin, ymax = extent
        pad_frac = 0.0
    xedges, yedges = _pixel_edges(xmin, xmax, ymin, ymax, image_size, pad_frac=pad_frac)

    img = _bin_image(X_all, Y_all, values, xedges, yedges,
                      mode=pixel_mapping, depths=D_all, half_x=Hx_all, half_y=Hy_all)
    return img, xedges, yedges


def _weighted_median(values, weights):
    """
    The value v such that half the total weight lies at or below it --
    unlike a weighted mean, robust to a small number of very bright
    (heavily weighted) outlier points dominating the result. NaN if
    `values` is empty or every weight is exactly 0.
    """
    if len(values) == 0:
        return np.nan
    order = np.argsort(values)
    v, w = values[order], weights[order]
    total = w.sum()
    if total <= 0.0:
        return np.nan
    cw = np.cumsum(w) / total
    return float(v[min(np.searchsorted(cw, 0.5), len(v) - 1)])


def _rigid_velocity(pts):
    """
    The velocity (...,3) a point (...,3), rigidly co-rotating with the
    binary (no motion of its own within the corotating frame), has in
    the frame where the observer direction n_hat(phase,incl) is what
    moves (see eclipse.observer_frame) -- i.e. an inertial-frame
    observer's view of a point fixed in these dimensionless a=1,
    Omega=1 units: the standard rigid-body rotation formula Omega_vec x
    p, with Omega_vec = z_hat (rotation about the orbital axis, rate 1).
    """
    pts = np.asarray(pts, dtype=float)
    return np.stack([-pts[..., 1], pts[..., 0], np.zeros_like(pts[..., 0])], axis=-1)


def radial_velocity_curve(lobe, disc, T_eff1, R1, phases, incl_deg, a_meters, P_orb_s, temp_maps,
                           wavelength_m=V_WAVELENGTH_M, u_primary=0.0, u_secondary=0.0,
                           T_acc=100000.0, u_acc=0.0, stream_angle_deg=None):
    """
    Intensity-weighted (flux-weighted: intensity * projected area *
    limb-darkening, same formula _phase_chunk_flux sums for the light
    curve) MEDIAN line-of-sight velocity of the system's visible emitting
    material at each phase, in km/s -- positive means RECEDING from the
    observer (standard spectroscopic convention), i.e. the negative of
    eclipse.project's own "depth increases toward the observer"
    convention: RV = -d(depth)/dt.

    Computed SEPARATELY for each of four families of emitting elements --
    primary, secondary, the ballistic accretion stream, and its
    magnetically-channeled continuation (see build_temperature_maps'
    angle_acc) -- rather than pooled into one combined median, so each
    component's own kinematics (e.g. the stream's free fall, distinct
    from the two stars' rigid orbital motion) stays visible instead of
    being washed out by whichever component happens to be brightest. The
    disc is deliberately excluded. The stream/magnetic-stream get no
    limb-darkening or projected-area foreshortening in their own weight
    (intensity * area only), matching how they're rendered elsewhere in
    this package (see render_system_image): treated as isotropically-
    emitting thin ribbons, not oriented surfaces with their own normal.

    Velocity model: every element's velocity is the RIGID co-rotation
    term (_rigid_velocity, the standard Omega_vec x p rigid-rotation
    formula) plus, for the two components with material actually moving
    relative to the corotating frame, that relative velocity on top. The
    ballistic stream's is stream.integrate_stream's own vx/vy. The
    magnetically-channeled segment's is _magnetic_stream_velocity: a
    "bead on a wire" constrained to the field line, starting from the
    ballistic stream's own velocity at the connection point (projected
    onto the line's tangent there) and evolved along the line by energy
    conservation in the SAME potential (roche.potential) that governs
    the ballistic stream's own motion -- see that function's own
    docstring for why this is exact, not an approximation, despite the
    magnetic tension's own (unmodeled) force on the material.

    a_meters/P_orb_s: convert the dimensionless (a=1, Omega=1) internal
    velocity unit to km/s -- v_scale = a_meters*2*pi/P_orb_s, the
    standard orbital-velocity normalization (e.g. SystemParams.a/P_orb_s).

    stream_angle_deg: forwarded to this function's own (fresh, see below)
    stream.integrate_stream call -- see its own docstring. Only affects
    rv_stream (the ballistic stream's own contribution): rv_magnetic's
    field-line geometry always comes from temp_maps.accretion_field_lines
    (built wherever temp_maps itself was, with whatever stream_angle_deg
    build_temperature_maps was given), and its connection-point velocity
    lookup only ever needs the trajectory up to that connection point --
    already covered by the default stopping condition -- so extending it
    further here has no effect on rv_magnetic.

    Returns (rv_primary, rv_secondary, rv_stream, rv_magnetic, rv1, rv2),
    each an array over `phases`, km/s:
      rv_primary/rv_secondary/rv_stream/rv_magnetic: that component's own
        weighted-median curve. NaN at any phase where the component has
        no visible weight at all (e.g. rv_magnetic when --angle_acc is
        unset, or a component fully eclipsed).
      rv1/rv2: the primary/secondary Roche mass-points' OWN circular
        velocity (not weighted by anything -- just those two point
        masses' own orbital motion, the classical single-line
        spectroscopic-binary sinusoid) over the same phases -- for
        plotting as a reference, not one of the four components above.
    """
    from stream import integrate_stream, disc_impact_index, closest_approach_index

    incl = np.radians(incl_deg)
    x1 = lobe.x1
    center1 = np.array([x1, 0.0, 0.0])
    v_scale = a_meters * 2.0 * np.pi / P_orb_s / 1000.0  # km/s per internal velocity unit

    # --- reference curves: the two Roche mass-points' own circular
    #     velocity, a closed form over the whole `phases` array at once ---
    phases_arr = np.asarray(phases, dtype=float)
    n_hat_all, _, _ = observer_frame(phases_arr, incl)
    p1 = np.array([lobe.x1, 0.0, 0.0])
    p2 = np.array([lobe.x2, 0.0, 0.0])
    rv1 = -np.einsum("i,ti->t", _rigid_velocity(p1), n_hat_all) * v_scale
    rv2 = -np.einsum("i,ti->t", _rigid_velocity(p2), n_hat_all) * v_scale

    # --- phase-independent per-body data (positions/areas/temperatures) ---
    sec_pts, sec_normals, sec_areas = temp_maps.sec_pts, temp_maps.sec_normals, temp_maps.sec_areas
    Isec = band_intensity(temp_maps.Tsec, wavelength_m)

    primary_normals = temp_maps.primary_normals
    primary_pts = center1 + R1 * primary_normals
    primary_areas = temp_maps.primary_base_areas * R1 ** 2
    primary_T_live = np.where(temp_maps.primary_spot_mask, T_acc, T_eff1)
    Iprim = band_intensity(primary_T_live, wavelength_m)
    u1_live = np.where(temp_maps.primary_spot_mask, u_acc, u_primary)

    # the ballistic stream needs its own fresh sampling (not
    # temp_maps.stream_pts, whose ribbon layout doesn't retain each
    # point's arclength, so temp_maps alone can't recover vx/vy at each
    # point) -- same truncation/resolution convention as
    # build_temperature_maps' own stream section, minus the ribbon
    # spread (irrelevant here: every strand at a given arclength shares
    # one velocity, so only the centerline needs sampling for this).
    if lobe.fill_factor >= 1.0:
        traj = integrate_stream(lobe, stream_angle_deg=stream_angle_deg)
        # always reported (see disc_impact_index's own report= option),
        # even when its result isn't used for truncation below
        disc_idx = disc_impact_index(traj, disc.rim, lobe.x1, report=True)
        if stream_angle_deg is not None:
            # see build_temperature_maps' matching explicit-stream_angle_deg
            # override -- an explicit request for how far the stream
            # extends shouldn't be second-guessed by the disc-impact/
            # closest-approach heuristics below.
            idx = len(traj["x"]) - 1
        else:
            idx = disc_idx
            if idx is None:
                idx = closest_approach_index(traj, lobe.x1)
            if idx is None:
                idx = len(traj["x"]) - 1
        n_along = 300
        s_max = traj["s"][idx]
        s_vals = np.linspace(0.0, s_max, n_along)
        xs_s = np.interp(s_vals, traj["s"], traj["x"])
        ys_s = np.interp(s_vals, traj["s"], traj["y"])
        stream_pts = np.stack([xs_s, ys_s, np.zeros(n_along)], axis=-1)
        stream_vel_rel = np.stack([np.interp(s_vals, traj["s"], traj["vx"]),
                                    np.interp(s_vals, traj["s"], traj["vy"]),
                                    np.zeros(n_along)], axis=-1)
        stream_areas = np.full(n_along, R1 * (s_max / n_along))
        Istream = band_intensity(np.full(n_along, temp_maps.stream_T), wavelength_m)
    else:
        traj = None
        stream_pts = np.zeros((0, 3))
        stream_vel_rel = np.zeros((0, 3))
        stream_areas = np.zeros(0)
        Istream = np.zeros(0)

    # magnetically-channeled continuation, one ribbon per active --angle_acc
    # entry (see _magnetic_stream_ribbon) -- position/area/temperature
    # and now velocity (_magnetic_stream_velocity) are all phase-
    # independent, so build every ribbon once here rather than once per
    # phase inside the loop below.
    magnetic_ribbons = []
    for line in temp_maps.accretion_field_lines:
        pts_m, areas_m, T_m = _magnetic_stream_ribbon(line, R1, center1, T_acc, temp_maps.stream_T)
        n_strands = pts_m.shape[0] // line.shape[0]
        v_rel_line = _magnetic_stream_velocity(line, lobe, center1, traj)
        v_rel_m = np.repeat(v_rel_line, n_strands, axis=0)
        magnetic_ribbons.append((pts_m, areas_m, band_intensity(T_m, wavelength_m), v_rel_m))

    rv_primary = np.empty(len(phases_arr))
    rv_secondary = np.empty(len(phases_arr))
    rv_stream = np.empty(len(phases_arr))
    rv_magnetic = np.empty(len(phases_arr))
    for i, ph in enumerate(phases_arr):
        frame = observer_frame(ph, incl)
        n_hat = frame[0]

        # primary
        mu1 = np.einsum("ij,j->i", primary_normals, n_hat)
        limb1 = 1.0 - u1_live + u1_live * mu1
        vis1 = ((mu1 > 0.0) & visible_mask_bulk(primary_pts, ph, incl, lobe)
                & disc.visible(primary_pts, ph, incl))
        v1 = -np.dot(_rigid_velocity(primary_pts[vis1]), n_hat)
        w1 = (Iprim * primary_areas * mu1 * limb1)[vis1]
        rv_primary[i] = _weighted_median(v1, w1) * v_scale

        # secondary
        mu_sec = np.einsum("ij,j->i", sec_normals, n_hat)
        limb_sec = 1.0 - u_secondary + u_secondary * mu_sec
        vis_sec = ((mu_sec > 0.0) & disc.visible(sec_pts, ph, incl)
                   & sphere_visible_mask(sec_pts, ph, incl, center1, R1))
        v_sec = -np.dot(_rigid_velocity(sec_pts[vis_sec]), n_hat)
        w_sec = (Isec * sec_areas * mu_sec * limb_sec)[vis_sec]
        rv_secondary[i] = _weighted_median(v_sec, w_sec) * v_scale

        # ballistic accretion stream (no limb-darkening/foreshortening,
        # same isotropic-ribbon treatment as render_system_image)
        if stream_pts.shape[0] > 0:
            vis_s = visible_mask(stream_pts, ph, incl, lobe) & disc.visible(stream_pts, ph, incl)
            v_s = -np.dot(_rigid_velocity(stream_pts[vis_s]) + stream_vel_rel[vis_s], n_hat)
            w_s = (Istream * stream_areas)[vis_s]
        else:
            v_s, w_s = np.zeros(0), np.zeros(0)
        rv_stream[i] = _weighted_median(v_s, w_s) * v_scale

        # magnetically-channeled continuation, one ribbon per active
        # --angle_acc entry (see magnetic_ribbons above); every entry pooled
        # into ONE median, since they're all the same physical component
        # (just at different radii). v_rel_m is the along-line free-fall
        # velocity from _magnetic_stream_velocity, on top of the same
        # rigid co-rotation every other component gets.
        v_m_list, w_m_list = [], []
        for pts_m, areas_m, Im, v_rel_m in magnetic_ribbons:
            vis_m = (visible_mask(pts_m, ph, incl, lobe) & disc.visible(pts_m, ph, incl)
                     & sphere_visible_mask(pts_m, ph, incl, center1, R1))
            v_m_list.append(-np.dot(_rigid_velocity(pts_m[vis_m]) + v_rel_m[vis_m], n_hat))
            w_m_list.append((Im * areas_m)[vis_m])
        v_m = np.concatenate(v_m_list) if v_m_list else np.zeros(0)
        w_m = np.concatenate(w_m_list) if w_m_list else np.zeros(0)
        rv_magnetic[i] = _weighted_median(v_m, w_m) * v_scale

    return rv_primary, rv_secondary, rv_stream, rv_magnetic, rv1, rv2


def _phase_chunk_flux(lobe, disc, temp_maps, T_eff1, R1, incl_deg, wavelength_m, phases,
                       progress_label=None, u_primary=0.0, u_secondary=0.0, u_disc=0.0,
                       T_acc=100000.0, u_acc=0.0):
    """
    The per-phase flux loop from physical_light_curve, factored out as a
    standalone (picklable) function so it can run in a worker process --
    see physical_light_curve's n_workers. Returns (star, disc, secondary)
    arrays over `phases`.

    progress_label: if given, report "label: NN%" as phases are actually
    computed (see utils.with_progress) -- this is the real, potentially
    slow work (each phase costs a visible_fraction/disc.visible
    evaluation), so it must wrap this loop specifically, not some cheaper
    proxy like materializing the phases list. Left None (no reporting)
    for worker-process calls, where the caller already reports progress
    over completed chunks instead (interleaved per-phase prints from
    multiple processes would just be noise).

    u_primary/u_secondary/u_disc: linear limb-darkening coefficients
    (I=I0*(1-u+u*mu), mu=cos of the angle to the observer), 0=disabled --
    an extra multiplicative factor on top of each body's existing mu
    (projected-area) foreshortening, independent per body since a white
    dwarf, an irradiated secondary, and an accretion disc have quite
    different limb-darkening behavior in general. Like the secondary and
    disc, the primary's total flux genuinely scales with (1-u_primary/3)
    when eclipse-free -- no compensating renormalization -- consistent
    across all three bodies.
    """
    incl = np.radians(incl_deg)
    x1 = lobe.x1

    sec_pts = temp_maps.sec_pts
    sec_normals = temp_maps.sec_normals
    sec_areas = temp_maps.sec_areas
    Isec = band_intensity(temp_maps.Tsec, wavelength_m)

    disc_pts, disc_normals, dareas = temp_maps.disc_pts, temp_maps.disc_normals, temp_maps.disc_areas
    Idisc = band_intensity(temp_maps.disc_T, wavelength_m)
    cell_disc = temp_maps.disc_cell_size

    star_center = np.array([x1, 0.0, 0.0])
    # primary_normals is a fixed unit-sphere DIRECTION pattern (not scaled
    # by R1, see build_temperature_maps' n_primary docstring) -- points and
    # areas are rescaled by the live R1 here so a "cheap" fit trial with a
    # different R1 is honored without rebuilding temp_maps.
    primary_normals = temp_maps.primary_normals
    primary_areas = temp_maps.primary_base_areas * R1 ** 2
    # T_acc applied live (like T_eff1), not baked into temp_maps -- see
    # TemperatureMaps.primary_spot_mask's docstring for why, and
    # render_system_image's matching T_acc parameter.
    primary_T_live = np.where(temp_maps.primary_spot_mask, T_acc, T_eff1)
    Iprim = band_intensity(primary_T_live, wavelength_m)
    cell_primary = float(np.sqrt(np.mean(primary_areas)))

    star_arr = np.empty(len(phases))
    disc_arr = np.empty(len(phases))
    sec_arr = np.empty(len(phases))

    phase_iter = phases if progress_label is None \
        else with_progress(phases, len(phases), progress_label)
    for i, ph in enumerate(phase_iter):
        # computed once and threaded through every occlusion test below
        # (visible_fraction_bulk, sphere_visible_mask) instead
        # of each recomputing its own observer_frame(ph, incl) -- cheap
        # per call, but this loop calls into it half a dozen times a
        # phase, and that overhead stopped being negligible once the
        # Roche-lobe occlusion tests themselves got fast (see
        # eclipse.sphere_visible_mask/visible_fraction_bulk docstrings)
        frame = observer_frame(ph, incl)
        n_hat = frame[0]
        primary_pts = star_center + R1 * primary_normals
        mu1 = np.einsum("ij,j->i", primary_normals, n_hat)
        # u_acc applied live (like T_acc/u_primary), not baked into
        # temp_maps -- see TemperatureMaps.primary_spot_mask's docstring.
        u1_live = np.where(temp_maps.primary_spot_mask, u_acc, u_primary)
        limb1 = 1.0 - u1_live + u1_live * mu1
        # visible_fraction_bulk is a smooth 0..1 fraction, not a boolean
        # mask, so it combines with the other two (boolean) tests by
        # multiplication, not bitwise-and -- same convention as
        # lightcurve.star_flux/disc_flux.
        vis1 = ((mu1 > 0.0)
                * visible_fraction_bulk(primary_pts, ph, incl, lobe, cell_primary, frame=frame)
                * disc.visible(primary_pts, ph, incl))
        star_arr[i] = np.sum(Iprim * primary_areas * mu1 * limb1 * vis1)

        if disc_normals is not None:
            cos_obs = np.clip(np.einsum("ij,j->i", disc_normals, n_hat), 0.0, None)
            f_disc_weighted = Idisc * dareas * cos_obs * (1.0 - u_disc + u_disc * cos_obs)
            pts_test = disc_pts + 1e-6 * disc_normals
            # disc points never enter the L1 funnel, so the fast hull-based
            # shortcut is exact here -- see eclipse.visible_fraction_bulk.
            # use_hull=False: this is the disc's OWN points self-occlusion
            # test, always inside its own hull by construction, so the
            # hull pre-filter (see disc.occluded_by_disc) would only add
            # overhead here, never reject anything.
            vis = (visible_fraction_bulk(disc_pts, ph, incl, lobe, cell_disc, frame=frame)
                   * disc.visible(pts_test, ph, incl, use_hull=False))
        else:
            cos_obs = np.cos(incl)
            f_disc_weighted = Idisc * dareas * cos_obs * (1.0 - u_disc + u_disc * cos_obs)
            vis = visible_fraction_bulk(disc_pts, ph, incl, lobe, cell_disc, frame=frame)
        vis = vis * sphere_visible_mask(disc_pts, ph, incl, star_center, R1, n_hat=n_hat)
        disc_arr[i] = np.sum(f_disc_weighted * vis)

        mu_sec = np.einsum("...i,i->...", sec_normals, n_hat)
        limb_sec = 1.0 - u_secondary + u_secondary * mu_sec
        vis_sec = ((mu_sec > 0.0) & disc.visible(sec_pts, ph, incl)
                   & sphere_visible_mask(sec_pts, ph, incl, star_center, R1, n_hat=n_hat))
        sec_arr[i] = np.sum((Isec * sec_areas * mu_sec * limb_sec)[vis_sec])

    return star_arr, disc_arr, sec_arr


def physical_light_curve(lobe, disc, T_eff1, T_eff2, R1, phases, incl_deg, a_meters,
                          disc_teff_func=None, beta_grav=0.08, irradiate=True,
                          hotspot_T_h=None, hotspot_L_h_deg=5.0,
                          wavelength_m=V_WAVELENGTH_M,
                          n_disc=(40, 80), n_disc_z=15,
                          n_disc_irrad=(30, 60), irrad_chunk=150,
                          temp_maps=None, n_workers=1, distance_pc=10.0,
                          u_primary=0.0, u_secondary=0.0, u_disc=0.0,
                          n_primary=200, n_primary_irrad=200,
                          theta_1=0.0, phi_1=0.0, angle_acc=None, spot_acc=np.radians(5.0),
                          T_acc=100000.0, u_acc=0.0, stream_angle_deg=None):
    """
    Physically self-consistent eclipse light curve (at `wavelength_m`,
    default V-band), built from the exact same per-point temperature maps
    as render_system_image (gravity darkening + full-thermalization
    irradiation for the secondary; disc_teff_func plus the stream-impact
    hot spot for the disc) rather than the flux-shape-only model in
    lightcurve.System -- so the hot spot (hotspot_T_h, hotspot_L_h_deg)
    and gravity darkening (beta_grav) modulate the light curve exactly as
    they modulate the rendered image.

    At each phase, flux = sum(band_intensity(T, wavelength_m) * dA *
    cos(angle to observer)) over each component's visible surface (the
    disc dispatches flat vs. flared exactly as lightcurve.disc_flux
    does); there is no separate "stream" term since the stream's energy
    shows up only via the disc-rim hot spot, same convention as
    render_system_image (which does not render the stream trajectory
    itself as a luminous body).

    temp_maps: pass a precomputed build_temperature_maps(...) result to
    reuse it here too (e.g. the same TemperatureMaps already built for a
    render_system_image movie) instead of recomputing irradiation; if
    omitted (the common case), it's built internally from
    disc_teff_func/irradiate/beta_grav/hotspot_T_h/hotspot_L_h_deg/
    n_disc/n_disc_z/n_disc_irrad/irrad_chunk/stream_angle_deg -- see
    build_temperature_maps for what each of those means (stream_angle_deg
    has no effect at all if temp_maps is given directly instead, same as
    every other build_temperature_maps-only argument above). Either way
    it's built once (not per-phase); only occultation and observer-angle
    foreshortening are recomputed inside the phase loop below.

    Units: the surface elements (sec_areas/disc_areas, from
    RocheLobe/Disc) are dimensionless multiples of a^2 -- the geometry is
    built with the orbital separation as the length unit -- so
    band_intensity(T, wavelength_m) * dA * cos(theta), summed, is only
    proportional to a real flux. a_meters (the orbital separation in
    meters, i.e. SystemParams.a) converts that to an actual observed
    spectral flux density at `distance_pc` parsecs (10 pc by default):

        F_lambda(d) = (relative flux) * a_meters^2 / (distance_pc * PARSEC_M)^2   [W/m^2/m]

    which is then itself converted, via blackbody.flux_lambda_to_mjy, to
    F_nu in milliJansky (the standard output convention -- see
    blackbody.mjy_to_ab_mag for the AB-magnitude equivalent). There's no
    way to opt back into the old undistanced/pre-conversion relative
    units short of passing distance_pc=None (returns the relative flux,
    same convention as render_system_image's pixel values, with no mJy
    conversion applied either).

    n_workers: split `phases` into this many chunks and evaluate them in
    separate processes (each phase's flux is independent given
    temp_maps). Worth it once the phase loop itself takes more than
    about a second -- e.g. the default resolution costs several hundred
    ms/phase, so it pays off past a handful of phases, but a
    low-resolution/no-irradiation preview with few phases can finish
    before a process pool even starts up (~1s on macOS, dominated by
    re-importing numpy in each worker), making n_workers>1 a net loss
    there. Default 1 (serial, no subprocesses) so behavior is unchanged
    unless you opt in.

    u_primary/u_secondary/u_disc: see _phase_chunk_flux's docstring --
    linear limb-darkening coefficients (0=disabled). u_primary/u_disc
    also feed into build_temperature_maps' irradiation calculation (only
    used when temp_maps isn't already supplied) -- see n_primary_irrad
    below and build_temperature_maps' own docstring.

    n_primary: number of roughly equal-area points sampled over the
    primary's WHOLE surface (only used when temp_maps is built internally,
    i.e. temp_maps=None -- see build_temperature_maps' own n_primary,
    which also backs render_system_image's primary rendering, so one CLI
    value controls both).

    n_primary_irrad: only used when temp_maps is built internally (i.e.
    temp_maps=None) -- the primary's own (coarser, cheaper) surface
    resolution for its irradiation-of-the-secondary integral, a separate
    concern from n_primary (the observer-facing flux resolution); see
    build_temperature_maps/irradiation.star_irradiation_flux.

    theta_1/phi_1/angle_acc/spot_acc: only used when temp_maps is built
    internally -- the primary's accretion spot (see
    build_temperature_maps' own docstring). T_acc/u_acc, in contrast, are
    live parameters every call actually uses (forwarded to
    _phase_chunk_flux regardless of temp_maps), since the primary's own
    flux reconstructs the spot's temperature and limb law live -- see
    TemperatureMaps.primary_spot_mask.

    Returns (star, disc, secondary, total), each an array over `phases`.
    """
    if temp_maps is None:
        temp_maps = build_temperature_maps(
            lobe, disc, T_eff1, T_eff2, R1, disc_teff_func=disc_teff_func,
            irradiate=irradiate, beta_grav=beta_grav,
            hotspot_T_h=hotspot_T_h, hotspot_L_h_deg=hotspot_L_h_deg,
            n_disc=n_disc, n_disc_z=n_disc_z,
            n_disc_irrad=n_disc_irrad, irrad_chunk=irrad_chunk, u_disc=u_disc,
            u_primary=u_primary, n_primary_irrad=n_primary_irrad, n_primary=n_primary,
            theta_1=theta_1, phi_1=phi_1, angle_acc=angle_acc, spot_acc=spot_acc, T_acc=T_acc,
            u_acc=u_acc, incl_deg=incl_deg, stream_angle_deg=stream_angle_deg)

    if n_workers is None or n_workers <= 1:
        star_arr, disc_arr, sec_arr = _phase_chunk_flux(
            lobe, disc, temp_maps, T_eff1, R1, incl_deg, wavelength_m, phases,
            progress_label="light curve", u_primary=u_primary, u_secondary=u_secondary,
            u_disc=u_disc, T_acc=T_acc, u_acc=u_acc)
    else:
        from concurrent.futures import ProcessPoolExecutor
        from functools import partial

        chunks = [c for c in np.array_split(np.asarray(phases), n_workers) if len(c)]
        # ex.map only forwards positional args, and the keyword-only extras
        # (u_primary/u_secondary/u_disc/T_acc/u_acc) come after
        # progress_label -- bind them with partial rather than resorting
        # to passing progress_label positionally too.
        chunk_flux = partial(_phase_chunk_flux, u_primary=u_primary, u_secondary=u_secondary,
                              u_disc=u_disc, T_acc=T_acc, u_acc=u_acc)
        with ProcessPoolExecutor(max_workers=len(chunks)) as ex:
            results = list(with_progress(
                ex.map(chunk_flux,
                       [lobe] * len(chunks), [disc] * len(chunks), [temp_maps] * len(chunks),
                       [T_eff1] * len(chunks), [R1] * len(chunks), [incl_deg] * len(chunks),
                       [wavelength_m] * len(chunks), chunks),
                len(chunks), "light curve (chunks)"))
        star_arr = np.concatenate([r[0] for r in results])
        disc_arr = np.concatenate([r[1] for r in results])
        sec_arr = np.concatenate([r[2] for r in results])

    if distance_pc is not None:
        scale = (a_meters / (distance_pc * PARSEC_M)) ** 2
        star_arr = flux_lambda_to_mjy(star_arr * scale, wavelength_m)
        disc_arr = flux_lambda_to_mjy(disc_arr * scale, wavelength_m)
        sec_arr = flux_lambda_to_mjy(sec_arr * scale, wavelength_m)

    return star_arr, disc_arr, sec_arr, star_arr + disc_arr + sec_arr
