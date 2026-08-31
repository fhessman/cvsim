# blackbody.py
"""
Blackbody specific intensity, used as a monochromatic (single effective
wavelength) proxy for photometric-band surface brightness -- e.g. the
Johnson V band, approximated here by its effective wavelength
(~5500 Angstrom) rather than a full bandpass-integrated synthetic
spectrum. This is a standard simplification for illustrative rendering;
a proper synthetic photometry calculation would convolve a stellar/disc
spectrum model with the real V-band transmission curve.
"""

import numpy as np

H = 6.62607015e-34     # J s
C = 2.99792458e8       # m/s
KB = 1.380649e-23      # J/K
PARSEC_M = 3.0856775814913673e16  # m

V_WAVELENGTH_M = 5500e-10   # Johnson V effective wavelength, ~5500 Angstrom


def planck_lambda(wavelength_m, T):
    """
    Planck spectral radiance B_lambda(T) [W / (m^2 sr m)], vectorized over T
    (or wavelength). T in Kelvin, wavelength in meters.
    """
    T = np.asarray(T, dtype=float)
    x = H * C / (wavelength_m * KB * np.maximum(T, 1.0))
    # avoid overflow for very small T / large x
    with np.errstate(over="ignore"):
        denom = np.expm1(np.clip(x, None, 700.0))
    return (2.0 * H * C ** 2 / wavelength_m ** 5) / denom


def v_band_intensity(T):
    """Blackbody specific intensity at the V-band effective wavelength."""
    return planck_lambda(V_WAVELENGTH_M, T)


def band_intensity(T, wavelength_m=V_WAVELENGTH_M):
    """
    Blackbody specific intensity at an arbitrary effective wavelength
    (the general form of v_band_intensity, which is just this with
    wavelength_m=V_WAVELENGTH_M) -- lets the observing wavelength be a
    genuine system parameter instead of a hardcoded V-band assumption.
    """
    return planck_lambda(wavelength_m, T)


AB_ZEROPOINT_JY = 3631.0  # Oke & Gunn 1983 AB zero point, in Jansky


def flux_lambda_to_mjy(flux_lambda, wavelength_m):
    """
    Convert a spectral flux density F_lambda [W/m^2/m] to F_nu in
    milliJansky. F_nu = F_lambda * wavelength_m^2 / c [W/m^2/Hz] (both
    sides express the same power: F_nu |dnu| = F_lambda |dlambda|, and
    |dnu/dlambda| = c/wavelength_m^2), then 1 Jy = 1e-26 W/m^2/Hz, and
    1 mJy = 1e-3 Jy.
    """
    flux_lambda = np.asarray(flux_lambda, dtype=float)
    return flux_lambda * wavelength_m ** 2 / C * 1e29  # combines both unit conversions


def mjy_to_ab_mag(flux_mjy):
    """
    Convert a milliJansky flux density (e.g. flux_lambda_to_mjy's output,
    or render.physical_light_curve's) to an AB magnitude: m_AB =
    -2.5*log10(F_nu / 3631 Jy) (Oke & Gunn 1983).

    Zero or negative flux (e.g. a fully eclipsed component) gives +inf
    (no light -> infinitely faint), not an error.
    """
    flux_mjy = np.asarray(flux_mjy, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -2.5 * np.log10(flux_mjy / 1e3 / AB_ZEROPOINT_JY)


def flux_to_ab_mag(flux_lambda, wavelength_m):
    """Convenience wrapper: flux_lambda_to_mjy then mjy_to_ab_mag, for a
    raw F_lambda [W/m^2/m] value rather than an already-mJy one."""
    return mjy_to_ab_mag(flux_lambda_to_mjy(flux_lambda, wavelength_m))


def ab_mag_to_mjy(mag):
    """
    Inverse of mjy_to_ab_mag: AB magnitude to milliJansky flux density,
    F_nu = 3631 Jy * 10^(-mag/2.5) (Oke & Gunn 1983). Used to convert
    observed magnitude data (e.g. --data-mag-col) onto the same flux
    scale as the model light curve.
    """
    mag = np.asarray(mag, dtype=float)
    return AB_ZEROPOINT_JY * 1e3 * 10.0 ** (-mag / 2.5)
