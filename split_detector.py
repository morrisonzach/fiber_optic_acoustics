"""
Exact (symbolic) differential power for a Hermite-Gaussian beam on a split
D-fiber detector, using SymPy.

Collecting regions: y in [g/2, r] (top), y in [-r, -g/2] (bottom), x in [-r, r].
The beam is displaced by mu along y (and offset by mu_x along x).

The 1D HG intensity has a closed-form antiderivative F_k(u) (erf plus
polynomial * Gaussian terms), so every quantity below is an exact expression.
Because |E_y|^2 is even, the difference is written in the exact strip form

    P_top - P_bot = P * eta_x * [ (F(g/2+mu) - F(g/2-mu)) - (F(r+mu) - F(r-mu)) ]

which is algebraically identical to (integral over top) - (integral over bottom);
the __main__ block checks this against the direct two-integral form.

Inputs are converted to exact Rationals. Numbers are produced only at the very
end, with evalf at a precision you choose (default 50 digits).

Units: lengths in m, P in mW -> result in mW. Multiply by G [V/mW] for volts.
"""

from functools import lru_cache

import mpmath as mp
import sympy as sp

u = sp.Symbol("u", real=True)
mu, mux = sp.symbols("mu mu_x", real=True)
g, r, w, P = sp.symbols("g r w P", positive=True)


def exact(v):
    """Convert a number to an exact SymPy value (floats via their decimal string)."""
    if isinstance(v, sp.Basic):
        return v
    if isinstance(v, int):
        return sp.Integer(v)
    return sp.Rational(repr(float(v)))


def intensity_1d(k, var=u):
    """Normalized 1D HG_k intensity |E_k(var)|^2 [1/m], integrates to 1."""
    s = sp.sqrt(2) * var / w
    norm = sp.sqrt(2) / (sp.sqrt(sp.pi) * w * 2**k * sp.factorial(k))
    return norm * sp.hermite(k, s) ** 2 * sp.exp(-(s**2))


@lru_cache(maxsize=None)
def cdf(k):
    """Exact antiderivative F_k(u) of intensity_1d, as a SymPy Lambda."""
    F = sp.integrate(sp.expand(intensity_1d(k)), u)
    return sp.Lambda(u, sp.simplify(F))


@lru_cache(maxsize=None)
def DP_expr(mode):
    """Exact symbolic P_top - P_bot in terms of mu, mu_x, g, r, w, P."""
    n, m = mode
    Fn, Fm = cdf(n), cdf(m)
    eta_x = Fn(r - mux) - Fn(-r - mux)
    inner = Fm(g / 2 + mu) - Fm(g / 2 - mu)
    outer = Fm(r + mu) - Fm(r - mu)
    return P * eta_x * (inner - outer)


@lru_cache(maxsize=None)
def slope_expr(mode):
    """Exact linear sensitivity d(DP)/d(mu) at mu = 0 [mW/m]."""
    return sp.simplify(sp.diff(DP_expr(mode), mu).subs(mu, 0))


def beam_radius(w0, z, wavelength):
    """Exact Gaussian beam radius w(z) = w0 sqrt(1 + (z/zR)^2)."""
    w0, z, lam = exact(w0), exact(z), exact(wavelength)
    zR = sp.pi * w0**2 / lam
    return w0 * sp.sqrt(1 + (z / zR) ** 2)


def _params(mu_val, gap, r_val, w_val, P_val, mux_val):
    return {mu: exact(mu_val), g: exact(gap), r: exact(r_val),
            w: exact(w_val), P: exact(P_val), mux: exact(mux_val)}


def DP(mu_val, mode, w_val, P_val, gap, r_val, mux_val=0, dps=50):
    """Evaluate the exact DP at one displacement to `dps` significant digits."""
    if exact(gap) / 2 >= exact(r_val):
        return sp.Integer(0)
    subs = _params(mu_val, gap, r_val, w_val, P_val, mux_val)
    return DP_expr(tuple(mode)).evalf(dps, subs=subs)


def DP_slope(mode, w_val, P_val, gap, r_val, mux_val=0, dps=50):
    """Evaluate the exact linear sensitivity [mW/m]."""
    subs = _params(0, gap, r_val, w_val, P_val, mux_val)
    return slope_expr(tuple(mode)).evalf(dps, subs=subs)


def DP_series(mu_vals, mode, w_val, P_val, gap, r_val, mux_val=0, dps=50):
    """
    Exact DP for many displacements (e.g. a time series), evaluated with
    mpmath at `dps` digits. Returns a list of mpmath numbers; convert with
    [float(v) for v in out] when passing to numpy/scipy.
    """
    if exact(gap) / 2 >= exact(r_val):
        return [mp.mpf(0)] * len(mu_vals)
    fixed = {g: exact(gap), r: exact(r_val), w: exact(w_val),
             P: exact(P_val), mux: exact(mux_val)}
    f = sp.lambdify(mu, DP_expr(tuple(mode)).subs(fixed), modules="mpmath")
    with mp.workdps(dps):
        return [f(mp.mpf(repr(float(m_)))) for m_ in mu_vals]


if __name__ == "__main__":
    mode = (0, 0)
    w_val = beam_radius(sp.Rational(48, 100000), sp.Rational(1, 10),
                        sp.Rational(657, 10**9))
    r_val = 3 * w_val
    gap = sp.Rational(3, 10000)                       # 0.3 mm
    P_val = 53 * sp.Rational(2, 10)                   # mW
    d = sp.Rational(125, 10**11) / 2                  # mirror amplitude
    mu_val = 2 * d * sp.cos(7 * sp.pi / 180)          # beam displacement

    print("Exact symbolic DP for", mode, ":\n ", DP_expr(mode), "\n")
    print("DP          =", DP(mu_val, mode, w_val, P_val, gap, r_val), "mW")
    print("slope * mu  =", (DP_slope(mode, w_val, P_val, gap, r_val)
                            * mu_val).evalf(50), "mW")

    # Check against the direct (top integral) - (bottom integral) form
    y = sp.Symbol("y", real=True)
    Iy = intensity_1d(mode[1], y - mu)
    top = sp.integrate(Iy, (y, g / 2, r))
    bot = sp.integrate(Iy, (y, -r, -g / 2))
    direct = P * (cdf(mode[0])(r - mux) - cdf(mode[0])(-r - mux)) * (top - bot)
    subs = _params(mu_val, gap, r_val, w_val, P_val, 0)
    print("direct form =", direct.evalf(50, subs=subs), "mW")

    # HG01 with a 0.70 mm mask, as a second example
    print("\nHG01, 0.70 mm mask:",
          DP(mu_val, (0, 1), w_val, P_val, sp.Rational(7, 10000), r_val), "mW")