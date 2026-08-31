# fit.py
"""
Least-squares fitting of the lightcurve.System forward model to observed
photometry (e.g. a continuous TESS light curve folded/restricted to one or
more eclipse windows). The "given" parameters (mass ratio q, inclination)
are fixed by constructing the roche.RocheLobe/lightcurve.System once;
everything else (central-star size/brightness, disc flux distribution and
shape, stream/hot-spot brightness) is fit via scipy.optimize.least_squares.
"""

import numpy as np
from scipy.optimize import least_squares


def _assemble(x, param_names, fixed):
    p = dict(fixed)
    for name, val in zip(param_names, x):
        p[name] = val
    return p


def _residuals(x, param_names, fixed, system, phases, flux, err):
    p = _assemble(x, param_names, fixed)
    _, _, _, model = system.light_curve(phases, p)
    return (model - flux) / err


def fit_light_curve(system, phases, flux, err, initial, bounds=None, fixed=None,
                     **least_squares_kwargs):
    """
    system   : lightcurve.System (fixes q, inclination via its RocheLobe)
    phases   : orbital phases of the data
    flux,err : observed flux and its 1-sigma uncertainty, same shape as phases
    initial  : dict {param_name: guess} -- the parameters to fit
    bounds   : optional dict {param_name: (lo,hi)}; unlisted params are
               unbounded
    fixed    : optional dict of additional model parameters held fixed
               during the fit (merged into every model evaluation)

    Returns (result, best_fit, uncertainties) where `result` is the raw
    scipy.optimize.OptimizeResult, `best_fit` is the full parameter dict
    (fixed + best-fit values), and `uncertainties` is a dict of 1-sigma
    parameter uncertainties from the linearized covariance
    (J^T J)^-1 * chi2_reduced (valid near the optimum / for reasonably
    well-behaved residuals -- treat as approximate).
    """
    param_names = list(initial.keys())
    x0 = np.array([initial[k] for k in param_names], dtype=float)
    fixed = dict(fixed) if fixed else {}

    if bounds:
        lo = np.array([bounds.get(k, (-np.inf, np.inf))[0] for k in param_names])
        hi = np.array([bounds.get(k, (-np.inf, np.inf))[1] for k in param_names])
        bnds = (lo, hi)
    else:
        bnds = (-np.inf, np.inf)

    result = least_squares(
        _residuals, x0, bounds=bnds,
        args=(param_names, fixed, system, phases, flux, err),
        **least_squares_kwargs,
    )

    best_fit = _assemble(result.x, param_names, fixed)

    uncertainties = {}
    dof = max(len(phases) - len(param_names), 1)
    chi2_reduced = np.sum(result.fun ** 2) / dof
    try:
        J = result.jac
        cov = np.linalg.inv(J.T @ J) * chi2_reduced
        sigmas = np.sqrt(np.diag(cov))
        uncertainties = dict(zip(param_names, sigmas))
    except np.linalg.LinAlgError:
        uncertainties = {k: np.nan for k in param_names}

    return result, best_fit, uncertainties
