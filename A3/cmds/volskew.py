import numpy as np
from scipy.optimize import brentq


def sabrATM(F, T, alpha, beta, rho, nu):
    """SABR ATM (K=F) Black implied volatility — Hagan (2002)."""
    omB = 1.0 - beta
    Fb = F ** omB
    term1 = omB**2 / 24.0 * alpha**2 / Fb**2
    term2 = 0.25 * rho * beta * nu * alpha / Fb
    term3 = (2.0 - 3.0 * rho**2) / 24.0 * nu**2
    return alpha / Fb * (1.0 + (term1 + term2 + term3) * T)


def sabr(F, K, T, alpha, beta, rho, nu):
    """
    Hagan (2002) SABR Black implied volatility approximation.
    Vectorized over K.
    """
    K = np.asarray(K, dtype=float)
    scalar = K.ndim == 0
    K = np.atleast_1d(K)

    eps = 1e-10
    omB = 1.0 - beta

    FK = F * K
    logFK = np.log(F / K)
    FK_mid = FK ** (omB / 2.0)

    atm = np.abs(logFK) < eps

    # --- General (off-ATM) ---
    A = 1.0 + omB**2 / 24.0 * logFK**2 + omB**4 / 1920.0 * logFK**4

    z = np.where(atm, 0.0, nu / alpha * FK_mid * logFK)
    disc = np.maximum(1.0 - 2.0 * rho * z + z**2, 0.0)
    sqrt_disc = np.sqrt(disc)

    num = sqrt_disc + z - rho
    den = 1.0 - rho
    arg = np.maximum(num / den, eps)
    chi = np.log(arg)

    z_over_chi = np.where(
        np.abs(z) < eps, 1.0,
        z / np.where(np.abs(chi) < eps, 1.0, chi)
    )

    term1 = omB**2 / 24.0 * alpha**2 / FK ** omB
    term2 = 0.25 * rho * beta * nu * alpha / FK_mid
    term3 = (2.0 - 3.0 * rho**2) / 24.0 * nu**2
    correction = 1.0 + (term1 + term2 + term3) * T

    vol = alpha / (FK_mid * A) * z_over_chi * correction

    # --- ATM override ---
    vol_atm = sabrATM(F, T, alpha, beta, rho, nu)
    vol = np.where(atm, vol_atm, vol)

    return float(vol[0]) if scalar else vol


def solve_alpha(F, T, sigma_atm, beta, rho, nu):
    """
    Solve for SABR alpha given ATM Black vol constraint.
    ATM formula: sigma = (alpha / F^(1-beta)) * [1 + (...) * T]
    leads to a cubic in alpha.
    """
    omB = 1.0 - beta
    Fb = F ** omB

    c1 = omB**2 / (24.0 * Fb**2) * T
    c2 = 0.25 * rho * beta * nu / Fb * T
    c3 = 1.0 + (2.0 - 3.0 * rho**2) / 24.0 * nu**2 * T
    target = sigma_atm * Fb

    def f(a):
        return c1 * a**3 + c2 * a**2 + c3 * a - target

    a_max = max(10.0 * abs(target) / max(abs(c3), 1e-6), 1000.0)

    try:
        return brentq(f, 1e-12, a_max)
    except ValueError:
        return target / c3 if c3 > 0 else abs(target)
