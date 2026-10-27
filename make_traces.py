"""
Noiseless 109 kHz balanced-detector voltage traces (HG00) for each mask,
scaled by the MEASURED fiber output powers.
 
Model: each D-fiber transmits a fraction T_i of the light landing on its face.
At the balanced operating point,
    V(t) = G * (P_out_top + P_out_bot) / eta_open * DP_unit(mu(t))
where DP_unit is the exact top-minus-bottom power difference for a beam of
unit power and eta_open is the fraction of the beam landing on the two
open fiber faces. (Unequal T_top, T_bot plus the small beam offset used to
balance the detector change this by about 1% at most.)
 
Output: traces_109kHz.npz with t and V_<dataset>_<mask> arrays
"""
 
import mpmath as mp
import numpy as np
import sympy as sp
 
from split_detector import DP_expr, cdf, g, mu, mux, P, r, w
 
N, FS, F0 = 65535, 1_000_000, 109_000
G = sp.Integer(133)                         # V/mW
W = sp.Rational(48, 100000)                 # 0.48 mm radius at the detector face
R = sp.Rational(15, 10000)                  # 1.5 mm POF radius
D_PP = sp.Rational(125, 10**11)             # 1.25 nm p-p mirror displacement
AMP = 2 * (D_PP / 2) * sp.cos(7 * sp.pi / 180)   # beam displacement amplitude
DPS = 40
 
GAPS = {"0.30mm": sp.Rational(30, 100000), "0.38mm": sp.Rational(38, 100000),
        "0.54mm": sp.Rational(54, 100000), "0.70mm": sp.Rational(70, 100000)}
 
# Measured total fiber output (bottom + top) in mW, at the balanced point
P_OUT = {
    "unnormalized": {"0.30mm": sp.Rational("3.69") + sp.Rational("2.74"),
                     "0.38mm": sp.Rational("1.77") + sp.Rational("1.41"),
                     "0.54mm": sp.Rational("1.61") + sp.Rational("1.35"),
                     "0.70mm": sp.Rational("1.07") + sp.Rational("1.02")},
    "normalized":   {"0.30mm": sp.Rational("1.08") + sp.Rational("1.03"),
                     "0.38mm": sp.Rational("1.06") + sp.Rational("1.02"),
                     "0.54mm": sp.Rational("1.08") + sp.Rational("1.02"),
                     "0.70mm": sp.Rational("1.07") + sp.Rational("1.02")},
}
 
num, den = (sp.Rational(F0, FS)).as_numer_denom()
num, den = int(num), int(den)
k = (num * np.arange(N)) % den             # exact phase index; period = 1000 samples
 
F = cdf(0)
expr = DP_expr((0, 0))
out = {"t": np.arange(N) / FS}
 
with mp.workdps(DPS):
    amp = mp.mpf(sp.N(AMP, DPS + 10))
    for label, gap in GAPS.items():
        eta_x = F(R) - F(-R)
        eta_open = (eta_x * 2 * (F(R) - F(gap / 2))).subs(w, W)
        unit = expr.subs({g: gap, r: R, w: W, P: 1, mux: 0})
        f = sp.lambdify(mu, unit / eta_open, modules="mpmath")
        table = np.array([float(f(amp * mp.sin(2 * mp.pi * j / den)))
                          for j in range(den)])          # = DP per mW detected
        for ds, pouts in P_OUT.items():
            pout = pouts[label]
            V = float(G * pout) * table
            out[f"V_{ds}_{label}"] = V[k]
            print(f"{ds:>12} {label}: P_out = {float(pout):.2f} mW, "
                  f"amplitude = {V.max()*1e3:.4f} mV")
 
np.savez("traces_109kHz.npz", **out)
print(f"beam displacement amplitude = {float(AMP)*1e9:.4f} nm")
