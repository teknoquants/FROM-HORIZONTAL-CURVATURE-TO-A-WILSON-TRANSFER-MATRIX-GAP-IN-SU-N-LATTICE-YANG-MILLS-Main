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
"""Validation of Theorem 1 (log-Sobolev inequality with constant 2 C_kappa) on one SU(2) loop.

The proof runs along the flow u_t = P_t(g e^W)/P_t(e^W), d mu_t = Z^{-1} e^{-V_t} dU. On one loop every step can be computed
exactly in the character basis (charflow.py), so each identity of the proof is checked numerically, not only the conclusion.

What is checked (beta in {0.5, 1, 2}; test functions g = e^{-2 beta x}, 1 + 0.6 x^2, e^{x^3}):
  (a) the total mass int u_t d mu_t is constant in t;
  (b) d/dt Ent_{mu_t}(u_t) = -I(t), I(t) = int u_t |grad log u_t|^2 d mu_t (central differences);
  (c) I'(t) = -2 int u_t [ ||Hess log u_t||^2 + (Ric + 2 Hess V_t)(grad log u_t, grad log u_t) ] d mu_t;
  (d) I(t) <= e^{-2 Lambda(t)} I(0), Lambda = int kappa, kappa(t) = 1 - max_theta q''_t (Ric = 1 for -Tr on SU(2));
  (e) Ent_mu(g) = int_0^infty I(t) dt (terminal entropy zero) and Ent_mu(f^2) <= 2 C_kappa int |grad f|^2 d mu with f^2 = g;
  (f) beta = 0 normalization: Lambda = t, C_kappa = 2/N = 1 for N = 2, and Ent(f^2) <= (4/N) int |grad f|^2 for random f.
"""
import math
import sys

import numpy as np

from charflow import TH, X, WQ, heat, d1, d2, end_values
from common import Report

BETAS = (0.5, 1.0, 2.0)


def flow_quantities(beta, g, t):
    B = heat(np.exp(2 * beta * X), t)
    A = heat(g * np.exp(2 * beta * X), t)
    u = A / B
    w = WQ * B / np.sum(WQ * B)                       # mu_t on the quadrature grid
    lu = np.log(u)
    lu1, lu2 = d1(lu), d2(lu)
    mass = np.sum(w * u)
    ent = np.sum(w * u * np.log(u / mass))
    I = np.sum(w * u * 0.5 * lu1 ** 2)                  # |grad phi|^2 = phi'^2 / 2
    V = -np.log(B)
    hess2 = (lu2 / 2) ** 2 + 2 * (np.cos(TH) / np.sin(TH) * lu1 / 2) ** 2   # radial^2 + 2 tangential^2
    curv = (1 + d2(V)) * 0.5 * lu1 ** 2                 # (Ric + 2 Hess V)(grad phi, grad phi), grad phi radial
    Iprime = -2 * np.sum(w * u * (hess2 + curv))
    q2 = d2(np.log(B)); _, _, z0pp, zppp = end_values(B); z0, zp = end_values(B)[:2]
    kap = 1 - max(q2.max(), z0pp / z0, zppp / zp)
    return mass, ent, I, Iprime, kap


def main():
    rep = Report("Theorem 1 (log-Sobolev inequality), one SU(2) loop")
    tests = {"e^{-2 beta x}": lambda b: np.exp(-2 * b * X), "1 + 0.6 x^2": lambda b: 1 + 0.6 * X ** 2,
             "e^{x^3}": lambda b: np.exp(X ** 3)}
    ts = np.concatenate([np.linspace(0, 0.1, 201)[:-1], np.linspace(0.1, 12, 2381)])
    okm = okb = okc = okd = oke = True
    det_b, det_e = [], []
    for beta in BETAS:
        for name, gf in tests.items():
            g = gf(beta)
            Q = np.array([flow_quantities(beta, g, t) for t in ts])
            mass, ent, I, Ip, kap = Q.T
            okm &= np.max(np.abs(mass / mass[0] - 1)) < 1e-9
            dE = np.gradient(ent, ts); dI = np.gradient(I, ts)
            sl = slice(5, -5)
            eb = np.max(np.abs(dE[sl] + I[sl])) / I[0]
            ec = np.max(np.abs(dI[sl] - Ip[sl])) / abs(Ip[0])
            okb &= eb < 2e-3; okc &= ec < 2e-3
            Lam = np.concatenate([[0], np.cumsum(0.5 * (kap[1:] + kap[:-1]) * np.diff(ts))])
            okd &= np.all(I <= np.exp(-2 * Lam) * I[0] * (1 + 1e-6))
            intI = np.trapezoid(I, ts)
            Ck = 2 * np.trapezoid(np.exp(-2 * Lam), ts) + np.exp(-2 * Lam[-1]) / kap[-1]
            f2 = g / np.sum(WQ * np.exp(2 * beta * X) * g) * np.sum(WQ * np.exp(2 * beta * X))
            w0 = WQ * np.exp(2 * beta * X); w0 /= w0.sum()
            ent0 = np.sum(w0 * g * np.log(g / np.sum(w0 * g)))
            gradf2 = np.sum(w0 * 0.5 * d1(np.sqrt(g)) ** 2)
            oke &= abs(intI / ent0 - 1) < 2e-3 and ent0 <= 2 * Ck * gradf2
            if name == "e^{-2 beta x}":
                det_b.append(f"beta={beta}: rel. err. {eb:.1e}, {ec:.1e}")
                det_e.append(f"beta={beta}: Ent = {ent0:.4f} <= 2 C_k |grad f|^2 = {2 * Ck * gradf2:.4f} (C_k = {Ck:.3f})")
    rep.check("(a) the mass int u_t d mu_t is conserved along the flow", okm)
    rep.check("(b) d/dt Ent_{mu_t}(u_t) = -I(t)", okb, "; ".join(det_b))
    rep.check("(c) I'(t) = -2 int u [ ||Hess log u||^2 + (Ric + 2 Hess V)(grad log u, grad log u) ] d mu_t", okc)
    rep.check("(d) I(t) <= e^{-2 Lambda(t)} I(0) with the curvature profile kappa = 1 - max q''", okd)
    rep.check("(e) Ent_mu(g) = int_0^infty I dt, and Ent_mu(f^2) <= 2 C_kappa int |grad f|^2 d mu", oke, "; ".join(det_e))

    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(300):
        c = rng.normal(size=6)
        f = np.exp(sum(c[k] * X ** (k + 1) for k in range(6)) / 2)
        w0 = WQ / WQ.sum(); f2 = f ** 2
        ent = np.sum(w0 * f2 * np.log(f2 / np.sum(w0 * f2)))
        worst = max(worst, ent / (np.sum(w0 * 0.5 * d1(f) ** 2)))
    kap0 = flow_quantities(0.0, np.ones_like(X), 0.3)[4]
    ok = abs(kap0 - 1) < 1e-9 and worst <= 2.0
    rep.check("(f) beta = 0: kappa = Ric = 1, C_kappa = 2/N = 1; Ent(f^2) <= (4/N) int |grad f|^2 for random f", ok,
              f"largest ratio Ent/int|grad f|^2 = {worst:.4f} (sharp sphere value 4/3 = 1.3333)")
    return rep.summary()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
