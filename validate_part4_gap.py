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
"""Validation of Theorem 4 (transfer-matrix gap from Euclidean-time decay) and of the numerical cross-check (Section 6).

What is checked
  (a) geometry: in the 4D link-adjacency graph (spatial torus 3^3, periodic time T = 10) the distance from the spatial links of
      slice 0 to those of slice t is exactly min(t, T - t): a path advances at most one time layer per step;
  (b) character coefficients: int e^{4 beta cos} chi_n dHaar / ((n+1) int e^{4 beta cos} dHaar) = I_{n+1}(4 beta)/I_1(4 beta)
      (quadrature), all positive, so the strip transfer matrix is positive;
  (c) the strip gap Delta_strip = -3 log[I_2(4 beta)/I_1(4 beta)] and the six values of the paper's table;
  (d) finite periodic strip, time extent T = 400, exact character sums: the winding character chi_{1/2} has connected
      correlation decaying at rate Delta_strip; the adjoint winding character chi_1 has zero overlap with the first flux
      state and decays at the larger rate -3 log[I_3/I_1];
  (e) spectral checkpoint: the normalized transfer matrix restricted to the vacuum complement has spectral radius
      e^{-Delta_strip}; its t-th moments on a dense set of vectors decay exactly at that rate;
  (f) Theorem 4's bound: the Dobrushin rate (4D, q = 18 tanh(4 beta); and the strip's own, q = 6 tanh(4 beta)) is below the
      exact strip gap at every tabulated beta.
"""
import math
import sys

import numpy as np
from scipy.special import ive

from common import Report
from lattice import links_and_plaquettes, adjacency_counts, bfs_dist

TABLE = [(0.002, 1.9380, 18.6438), (0.005, 1.0218, 15.8950), (0.010, 0.3290, 13.8157)]


def ratio(n, beta):
    return ive(n + 1, 4 * beta) / ive(1, 4 * beta)


def main():
    rep = Report("Theorem 4 (transfer-matrix gap) and the strip cross-check")
    shape = (3, 3, 3, 10)
    idx, plaqs, per_link, n = adjacency_counts(shape)
    src = [i for (s, mu), i in idx.items() if mu < 3 and s[3] == 0]
    dist = bfs_dist(n, src)
    ok = True
    for t in range(0, 10):
        tgt = [i for (s, mu), i in idx.items() if mu < 3 and s[3] == t]
        ok &= min(dist[tgt]) == min(t, 10 - t)
    rep.check("(a) 4D link graph: spatial slices 0 and t are at distance min(t, T - t)", ok)

    from numpy.polynomial.legendre import leggauss
    gq, wq = leggauss(600); th = (gq + 1) * np.pi / 2; hw = (2 / np.pi) * np.sin(th) ** 2 * wq * np.pi / 2
    ok = True
    for beta in (0.002, 0.01, 0.1, 0.7):
        w = np.exp(4 * beta * np.cos(th)); a0 = np.sum(w * hw)
        for k in range(1, 5):
            chi = np.sin((k + 1) * th) / np.sin(th)
            ak = np.sum(w * chi * hw) / ((k + 1) * a0); ex = ratio(k, beta)
            ok &= (abs(ak / ex - 1) < 1e-6 if ex > 1e-8 else abs(ak - ex) < 1e-14) and ex > 0
    rep.check("(b) character coefficients a_n/(d_n a_0) = I_{n+1}(4 beta)/I_1(4 beta) > 0 (positive transfer matrix)", ok)

    ok, det = True, []
    for beta, cons, gap in TABLE:
        g = -3 * math.log(ratio(1, beta)); c4 = -math.log(18 * math.tanh(4 * beta))
        ok &= abs(g - gap) < 5e-5 and abs(c4 - cons) < 5e-5
        det.append(f"beta={beta}: {c4:.4f}, {g:.4f}")
    rep.check("(c) strip gap -3 log[I_2/I_1] and the conservative rate reproduce the table", ok, "; ".join(det))

    ok, det = True, []
    for beta in (0.002, 0.01, 0.1):
        nmax = 30; lam = np.array([ratio(k, beta) ** 3 for k in range(nmax)])     # eigenvalue of flux k (lam_0 = 1)
        T = 400
        def corr(obs_shift_pairs, t):
            Z = np.sum(lam ** T)
            return sum(lam[a] ** (T - t) * lam[b] ** t for a, b in obs_shift_pairs) / Z
        # chi_{1/2} couples flux k <-> k +- 1 with unit matrix elements (Clebsch-Gordan for SU(2) characters)
        pairs_half = [(k, k + 1) for k in range(nmax - 1)] + [(k + 1, k) for k in range(nmax - 1)]
        pairs_one = [(k, k + 2) for k in range(nmax - 2)] + [(k + 2, k) for k in range(nmax - 2)] + [(k, k) for k in range(1, nmax)]
        rates = []
        for pairs, expected in ((pairs_half, -3 * math.log(ratio(1, beta))), (pairs_one, -3 * math.log(ratio(2, beta)))):
            c1, c2 = corr(pairs, 2), corr(pairs, 3)
            r = -math.log(c2 / c1) if pairs is pairs_half else -math.log((corr(pairs, 3) - corr(pairs, 200)) / (corr(pairs, 2) - corr(pairs, 200)))
            rates.append(r); ok &= abs(r / expected - 1) < 1e-6
        det.append(f"beta={beta}: rates {rates[0]:.3f} (spin 1/2), {rates[1]:.3f} (spin 1)")
    rep.check("(d) winding chi_{1/2} decays at Delta_strip; adjoint chi_1 misses the first flux state and decays faster", ok, "; ".join(det))

    ok = True
    for beta in (0.002, 0.01, 0.1):
        lam = np.array([ratio(k, beta) ** 3 for k in range(30)])
        rng = np.random.default_rng(1); v = rng.normal(size=29) * np.exp(-0.3 * np.arange(29))
        mom = [np.sum(v ** 2 * lam[1:] ** t) for t in (10, 11)]
        ok &= abs(-math.log(mom[1] / mom[0]) - (-math.log(lam[1]))) < 1e-6 and lam[1:].max() == lam[1]
    rep.check("(e) on the vacuum complement the spectral radius is e^{-Delta_strip}; moments decay at exactly that rate", ok)

    ok, det = True, []
    for beta, cons, gap in TABLE:
        own = -math.log(6 * math.tanh(4 * beta))
        ok &= cons < gap and own < gap and cons < own
        det.append(f"beta={beta}: 4D {cons:.3f}, strip {own:.3f} < gap {gap:.3f}")
    rep.check("(f) Theorem 4: the Dobrushin rate (4D and the strip's own) lies below the exact strip gap", ok, "; ".join(det))
    return rep.summary()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
