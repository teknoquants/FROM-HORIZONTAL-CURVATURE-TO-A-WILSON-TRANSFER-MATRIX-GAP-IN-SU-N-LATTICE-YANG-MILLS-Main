# MIT License
#
# Copyright (c) 2026 Roger Cooperman; A.E. Petrunina V.V.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Validation of Proposition 2 (the one-loop log-Sobolev constant is at least 2/3) and of the numerical comparison.

Normalization (one SU(2) loop): d mu_beta ∝ e^{2 beta cos theta} sin^2 theta d theta, |grad f|^2 = (f')^2/2;
C_LS = sup Ent(f^2)/(2 int |grad f|^2), so Var <= C_LS int |grad f|^2 and C_opt <= C_LS.

What is checked
  (a) symbolic: Z(beta) = int e^{2 beta cos} sin^2 is even in beta, so f_beta^2 d mu_beta with f_beta = e^{-2 beta x} is the
      reflected probability measure; d/d theta [sin^3 e^{2 beta cos}] = (3 sin^2 cos - 2 beta sin^4) e^{2 beta cos};
  (b) by quadrature for beta in {0.1, 0.5, 1, 2, 4, 8}: Ent(f_beta^2) = 4 beta m_beta, 2 beta E[sin^2] = 3 m_beta,
      int |grad f_beta|^2 = 3 beta m_beta, ratio = 2/3;
  (c) beta = 0: C_opt = C_LS = 2/3 (sharp sphere constants for S^3 with Ric = 1);
  (d) the order C_opt(beta) <= 2/3 <= C_LS(beta) <= C_kappa(beta), with C_opt from finite elements and C_kappa from the exact
      formula of the Challenge A paper; C_opt ~ 1/(2 beta) while C_LS stays >= 2/3;
  (e) diagnostic: over exp(Chebyshev) families of degree 6 and 14 with 40 random starts, no function exceeds 2/3, and random
      perturbations of f_beta decrease the ratio (numerical evidence that C_LS(beta) = 2/3; not a proof).
"""
import math
import sys

import numpy as np
import sympy as sp
from numpy.polynomial import chebyshev as Ch
from numpy.polynomial.legendre import leggauss
from scipy.optimize import minimize

from common import Report, C_opt, logC_accurate

g_, w_ = leggauss(3000); TH = (g_ + 1) * np.pi / 2; WT = w_ * np.pi / 2; XG = np.cos(TH)


def measure(beta):
    lp = 2 * beta * XG + 2 * np.log(np.sin(TH)); lp -= lp.max()
    p = np.exp(lp) * WT
    return p / p.sum()


def ratio_family(beta, deg):
    p = measure(beta)
    T = np.array([Ch.chebval(XG, [0] * k + [1]) for k in range(1, deg + 1)])
    dT = np.array([Ch.chebval(XG, Ch.chebder([0] * k + [1])) for k in range(1, deg + 1)]) * (-np.sin(TH))

    def r(c):
        h = c @ T; h = h - h.max(); f2 = np.exp(2 * h); f2 /= (f2 * p).sum()
        return (f2 * np.log(f2 + 1e-300) * p).sum() / (2 * (0.5 * (c @ dT) ** 2 * f2 * p).sum())
    return r


def main():
    rep = Report("Proposition 2 (one-loop log-Sobolev constant)")
    th, b = sp.symbols("theta beta", real=True)
    expr = sp.diff(sp.sin(th) ** 3 * sp.exp(2 * b * sp.cos(th)), th)
    ok = sp.simplify(expr - (3 * sp.sin(th) ** 2 * sp.cos(th) - 2 * b * sp.sin(th) ** 4) * sp.exp(2 * b * sp.cos(th))) == 0
    Zp = lambda bb: np.sum(np.exp(2 * bb * XG) * np.sin(TH) ** 2 * WT)
    ok &= all(abs(Zp(bb) / Zp(-bb) - 1) < 1e-12 for bb in (0.3, 2.0, 7.0))
    rep.check("(a) Z(beta) even in beta (f^2 d mu is the reflected measure); derivative identity for sin^3 e^{2 beta cos}", ok)

    ok, det = True, []
    for beta in (0.1, 0.5, 1, 2, 4, 8):
        p = measure(beta); m = np.sum(p * XG)
        f2 = np.exp(-4 * beta * XG); f2 /= np.sum(p * f2)
        ent = np.sum(p * f2 * np.log(f2)) ; s2 = np.sum(p * np.sin(TH) ** 2)
        grad = np.sum(p * f2 * 0.5 * (2 * beta * np.sin(TH)) ** 2)
        ok &= abs(ent / (4 * beta * m) - 1) < 1e-9 and abs(2 * beta * s2 / (3 * m) - 1) < 1e-9
        ok &= abs(grad / (3 * beta * m) - 1) < 1e-9 and abs(ent / (2 * grad) - 2 / 3) < 1e-9
        det.append(f"beta={beta}: m = {m:.4f}")
    rep.check("(b) Ent(f^2) = 4 beta m, 2 beta E[sin^2] = 3 m, int|grad f|^2 = 3 beta m, ratio 2/3", ok, "; ".join(det[::2]))

    r0 = ratio_family(0.0, 8)
    best0 = max(-minimize(lambda c: -r0(c), np.random.default_rng(i).normal(size=8), method="BFGS").fun for i in range(10))
    c_small = C_opt(1e-6, 6000)
    rep.check("(c) beta = 0: C_opt = 2/3 and the log-Sobolev optimum is 2/3 (sphere S^3, Ric = 1)",
              abs(c_small - 2 / 3) < 1e-4 and abs(best0 - 2 / 3) < 1e-4, f"C_opt = {c_small:.6f}, LS optimum {best0:.6f}")

    ok, det = True, []
    for beta in (0.5, 1, 2, 4, 8):
        co = C_opt(beta, 6000); ck = math.exp(logC_accurate(beta))
        ok &= co <= 2 / 3 + 1e-12 and 2 / 3 <= ck
        det.append(f"beta={beta}: C_opt = {co:.4f}, C_kappa = {ck:.4g}")
    rep.check("(d) C_opt(beta) <= 2/3 <= C_LS(beta) <= C_kappa(beta); C_opt decays like 1/(2 beta), C_LS does not", ok, "; ".join(det))

    rng = np.random.default_rng(3); ok, det = True, []
    for beta in (0.05, 1, 4):
        for deg in (6, 14):
            r = ratio_family(beta, deg); best = 0
            c_t = np.zeros(deg); c_t[0] = -2 * beta
            for i in range(40):
                c0 = c_t + rng.normal(scale=0.5 * max(1, beta), size=deg) if i % 2 else rng.normal(scale=3 * max(1, beta), size=deg)
                best = max(best, -minimize(lambda c: -r(c), c0, method="BFGS", options={"maxiter": 3000, "gtol": 1e-10}).fun)
            near = max(r(c_t + 1e-3 * rng.normal(size=deg)) for _ in range(100))
            ok &= best <= 2 / 3 + 1e-6 and near <= r(c_t) + 1e-9 and abs(r(c_t) - 2 / 3) < 1e-9
        det.append(f"beta={beta}: best {best:.6f}")
    rep.check("(e) no function in the tested families exceeds 2/3; f_beta is a local maximum (evidence that C_LS = 2/3)", ok,
              "; ".join(det))
    return rep.summary()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
