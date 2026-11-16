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
"""Shared tools for the validation suite of "The Weak-Coupling Cost of Uniform Heat Flow on One SU(2) Loop"
(R. Cooperman and A.E. Petrunina V.V., XISOSIS, Vienna, 15 November 2026).

The heat flow of the Wilson weight on one SU(2) loop is evaluated from a positive integral (no alternating series):

F_t(theta) = int_0^pi rho_t(phi) exp(2 beta cos(theta) cos(phi)) S(2 beta sin(theta) sin(phi)) dphi,
rho_t(phi) = (2/pi) sin^2(phi) p_t(phi), S(a) = sinh(a)/a, p_t the SU(2) heat kernel (normalized Haar, metric -Tr).
"""
import math
import numpy as np
from numpy.polynomial.legendre import leggauss


def log_pt(phi, t):
    """log of the SU(2) heat kernel p_t(phi) w.r.t. normalized Haar measure, eigenvalues n(n+2)/2."""
    phi = np.asarray(phi, float)
    if t > 1.0:
        n = np.arange(0, 80)[:, None]
        s = ((n + 1) * np.exp(-n * (n + 2) * t / 2) * np.sin((n + 1) * phi)).sum(0) / np.sin(phi)
        return np.log(s)
    tau = t / 2
    u = np.pi - phi
    z = np.pi * u / (2 * tau)
    # images k = 0 and k = -1 combined (they cancel at the antipode); k = 1 and k = -2 as a relative correction
    Lnum = -(np.pi ** 2 + u ** 2) / (4 * tau) + z + np.log(-2 * u - (np.pi + u) * np.expm1(-2 * z))
    r = ((phi + 2 * np.pi) * np.exp(-(phi + 2 * np.pi) ** 2 / (4 * tau) - Lnum)
         + (phi - 4 * np.pi) * np.exp(-(phi - 4 * np.pi) ** 2 / (4 * tau) - Lnum))
    pre = math.log(2 * math.pi ** 2) - 1.5 * math.log(4 * math.pi * tau) + tau
    return pre + Lnum + np.log1p(r) - np.log(np.sin(phi))


def L1(a):
    a = np.asarray(a, float)
    out = np.empty_like(a)
    s = a < 0.05
    x = a[s]
    out[s] = x / 3 - x ** 3 / 45 + 2 * x ** 5 / 945
    x = a[~s]
    out[~s] = 1 / np.tanh(x) - 1 / x
    return out


def L2(a):
    a = np.asarray(a, float)
    out = np.empty_like(a)
    s = a < 0.05
    x = a[s]
    out[s] = 1 / 3 - x ** 2 / 15 + 2 * x ** 4 / 189
    x = a[~s]
    with np.errstate(over="ignore"):
        out[~s] = 1 / x ** 2 - 1 / np.sinh(x) ** 2
    return out


def logS(a):
    a = np.asarray(a, float)
    out = np.empty_like(a)
    s = a < 1e-3
    out[s] = a[s] ** 2 / 6
    x = a[~s]
    out[~s] = x + np.log1p(-np.exp(-2 * x)) - np.log(2 * x)
    return out


class Loop:
    def __init__(self, beta, nphi=3000):
        self.beta = beta
        g, w = leggauss(nphi)
        self.phi = (g + 1) * np.pi / 2
        self.wphi = w * np.pi / 2

    def derivs(self, theta, t):
        """log F, q' and q'' at the angles theta (array), at heat time t > 0."""
        b = self.beta
        phi = self.phi[None, :]
        th = np.asarray(theta, float)[:, None]
        logrho = math.log(2 / math.pi) + 2 * np.log(np.sin(phi)) + log_pt(self.phi, t)[None, :] + np.log(self.wphi)[None, :]
        A, Bv = 2 * b * np.cos(phi), 2 * b * np.sin(phi)
        a = Bv * np.abs(np.sin(th))
        sgn = np.sign(np.sin(th)) + (np.sin(th) == 0)
        h = A * np.cos(th) + logS(a)
        l1, l2 = L1(a) * sgn, L2(a)
        h1 = -A * np.sin(th) + Bv * np.cos(th) * l1
        h2 = -A * np.cos(th) - Bv * np.abs(np.sin(th)) * L1(a) + Bv ** 2 * np.cos(th) ** 2 * l2
        lw = logrho + h
        m = lw.max(axis=1, keepdims=True)
        e = np.exp(lw - m)
        F0 = e.sum(1)
        F1 = (e * h1).sum(1) / F0
        F2 = (e * (h2 + h1 ** 2)).sum(1) / F0
        return m[:, 0] + np.log(F0), F1, F2 - F1 ** 2


# ---------------------------------------------------------------- constants (Lemma 4 with D = 0, Theorem 3)
from scipy.special import ive
from scipy.integrate import quad


TS = np.concatenate([np.geomspace(1e-6, 1.0, 500), np.linspace(1.0, 40.0, 400)[1:]])

def antipode(beta, ts=TS, nphi=3000):
    L = Loop(beta, nphi)
    lf, q2 = [], []
    for t in ts:
        a, _, c = L.derivs(np.array([math.pi]), t)
        lf.append(a[0]); q2.append(c[0])
    return np.array(lf), np.array(q2)

def trap(y, x):
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))

def constants(beta):
    lf, q2 = antipode(beta)
    t = np.concatenate([[0.0], TS]); lf = np.concatenate([[-2 * beta], lf]); q2 = np.concatenate([[2 * beta], q2])
    kap = 1 - q2
    Lam = np.concatenate([[0], np.cumsum(0.5 * (kap[1:] + kap[:-1]) * np.diff(t))])
    g = np.exp(-2 * Lam)
    C_profile = 2 * trap(g, t) + g[-1] / kap[-1]
    h = np.exp(-2 * t + (4 / 3) * (lf + 2 * beta))
    C_formula = 2 * trap(h, t) + h[-1] / 1.0          # tail: e^{-2t} times a constant
    # identity d/dt log F(pi) = (3/2) q''(pi)
    dlf = np.gradient(lf, t)
    sel = (t > 1e-3) & (t < 5)
    ident = np.max(np.abs(dlf[sel] - 1.5 * q2[sel]) / np.maximum(1, np.abs(q2[sel])))
    return dict(C_profile=C_profile, C_formula=C_formula, ident=ident, t=t, lf=lf, q2=q2, kap=kap)

def Z(beta):        # int e^{2 beta x} dHaar = I_1(2 beta)/beta
    return ive(1, 2 * beta) * math.exp(2 * beta) / beta

def logZ(beta):     # log Z_beta, overflow-free
    return math.log(ive(1, 2 * beta)) + 2 * beta - math.log(beta)


def p_antipode(t):
    n = np.arange(0, 400)
    return float(np.sum((n + 1) ** 2 * (-1.0) ** n * np.exp(-n * (n + 2) * t / 2)))

def K0():
    f = lambda t: math.exp(-2 * t) * max(p_antipode(t), 0.0) ** (4 / 3)
    return 2 * (quad(f, 0, 0.05, limit=200)[0] + quad(f, 0.05, 1, limit=200)[0] + quad(f, 1, 60, limit=200)[0])


# ---------------------------------------------------------------- optimal constant and the Bessel lower bound
from scipy.linalg import eigh_tridiagonal


def C_opt(beta, n=6000):
    """1/lambda_1 for (1/2) int f'^2 p / int f^2 p, p = e^{2 beta cos} sin^2 on (0, pi); finite elements."""
    th = np.linspace(0, math.pi, n + 1)
    tm = 0.5 * (th[1:] + th[:-1]); d = th[1] - th[0]
    wm = np.exp(2 * beta * (np.cos(tm) - 1) + 2 * np.log(np.sin(tm)))
    mass = np.zeros(n + 1); mass[:-1] += 0.5 * wm * d; mass[1:] += 0.5 * wm * d
    diag = np.zeros(n + 1); c = 0.5 * wm / d
    diag[:-1] += c; diag[1:] += c
    idx = np.flatnonzero(mass > 1e-300); i0, i1 = idx[0], idx[-1]
    D = diag[i0:i1 + 1]; O = -c[i0:i1]; M = mass[i0:i1 + 1]; s = 1 / np.sqrt(M)
    ev = eigh_tridiagonal(D * s * s, O * s[:-1] * s[1:], select="i", select_range=(0, 1), eigvals_only=True)
    return 1 / ev[1]


def R_lower(beta):
    """Rayleigh quotient of cos(theta): Var(cos)/E[sin^2/2] under p_beta, from Bessel functions."""
    z = 2 * beta
    I0, I1, I2, I3 = (ive(k, z) for k in range(4))
    Ec = I2 / I1
    I1p = 0.5 * (I0 + I2)
    I1pp = 0.25 * (3 * I1 + I3)
    Zpp_over_Z = (I1pp * z * z - 2 * z * I1p + 2 * I1) / (z * z * I1)
    return (Zpp_over_Z - Ec ** 2) / (0.5 * (1 - Zpp_over_Z))


def h_fn(s, beta):
    return 2 * beta * np.sin(s) - 2 / np.tan(s)


# ---------------------------------------------------------------- high-precision character series (Part 4)
def F_series_mp(beta, t, th, nmax=260, dps=80):
    import mpmath as mp
    mp.mp.dps = dps
    b, t, th = mp.mpf(beta), mp.mpf(t), mp.mpf(th)
    tot = mp.mpf(0)
    for n in range(nmax):
        tot += (mp.besseli(n, 2 * b) - mp.besseli(n + 2, 2 * b)) * mp.e ** (-n * (n + 2) * t / 2) * mp.sin((n + 1) * th) / mp.sin(th)
    return tot


# ---------------------------------------------------------------- reporting
class Report:
    def __init__(self, name):
        self.name = name
        self.rows = []

    def check(self, label, ok, detail=""):
        self.rows.append((label, bool(ok), detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""))
        return ok

    def summary(self):
        p = sum(r[1] for r in self.rows)
        print(f"  -> {self.name}: {p}/{len(self.rows)} checks passed")
        return p == len(self.rows)


# ---------------------------------------------------------------- accurate C_kappa (Lemma 4 with D = 0), adaptive quadrature
def logC_accurate(beta, T=40.0):
    """log C_kappa = log 2 int_0^infty e^{-2t} (F_t(-1) e^{2 beta})^{4/3} dt, adaptive Gauss-Kronrod on fixed sub-intervals."""
    from scipy.integrate import quad as _quad
    L = Loop(beta)
    f = lambda t: math.exp(-2 * t + (4 / 3) * (L.derivs(np.array([math.pi]), t)[0][0] + 2 * beta))
    edges = [0, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, T]
    s = sum(_quad(f, a, c, limit=200, epsabs=0, epsrel=1e-11)[0] for a, c in zip(edges, edges[1:]))
    return math.log(2 * s + f(T))          # tail: (F e^{2 beta})^{4/3} is constant to 1e-16 beyond T, 2 int_T e^{-2t} = e^{-2T}
