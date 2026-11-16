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
"""Validation of Lemma 3 and Theorem 3 (Dobrushin route to spatial decay at strong coupling) and of the stated constants.

What is checked
  (a) Lemma 3(i): two probability densities whose log-ratio has oscillation a are at total variation <= tanh(a/4)
      (random densities, and equality for two-point laws);
  (b) lattice counting: in d = 4 each link lies in 2(d-1) = 6 plaquettes, each with 3 other links; n_ej in {0, 1};
      sum_j n_ej = 18 (d = 2: 2 plaquettes, sum 6), on periodic tori of side 3 and 4;
  (c) Lemma 3(ii): for SU(2), the total-variation change of the conditional law of U_e when one neighbouring link changes is at
      most tanh(N^2 beta n_ej) (quasi-Monte Carlo over Haar, random and adversarial staple configurations, beta up to 0.05);
      the observed worst case is about 0.42 of the bound, which is therefore conservative;
  (d) the Dobrushin coefficient q = 18 tanh(N^2 beta) and the threshold beta_D(N) = N^{-2} artanh(1/18); comparison of
      beta_D(2) with the curvature range beta < 1/192 and the benchmark 1/48;
  (e) Theorem 3 against an exact covariance: the 2D periodic SU(2) strip (circumference 3), where the winding-loop covariance
      is (1/4)(I_2(4 beta)/I_1(4 beta))^{3t}; the bound (4|S|/(1-q)) q^t with the strip's own q = 6 tanh(4 beta) holds.
"""
import math
import sys

import numpy as np
from scipy.special import ive
from scipy.stats import qmc

from common import Report
from lattice import adjacency_counts


def haar_su2(n, seed):
    s = qmc.Sobol(d=4, scramble=True, seed=seed).random(n)
    from scipy.stats import norm
    q = norm.ppf(np.clip(s, 1e-12, 1 - 1e-12))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def qmul(a, b):
    a0, a1, a2, a3 = a.T if a.ndim > 1 else a
    b0, b1, b2, b3 = b.T if b.ndim > 1 else b
    return np.array([a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3, a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2,
                     a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1, a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0]).T


def tv_conditional(beta, staples, j_old, j_new, U):
    """TV between conditional laws of U_e with staple sums differing in one staple; ReTr(U S) = 2 Re(U S) (quaternions)."""
    N = 2
    def logdens(st):
        a = st.sum(axis=0)
        return N * beta * 2 * (qmul(U, np.tile(a, (len(U), 1)))[:, 0])
    s1 = staples.copy(); s1[0] = j_old
    s2 = staples.copy(); s2[0] = j_new
    l1, l2 = logdens(s1), logdens(s2)
    p1 = np.exp(l1 - l1.max()); p1 /= p1.mean()
    p2 = np.exp(l2 - l2.max()); p2 /= p2.mean()
    return 0.5 * np.mean(np.abs(p1 - p2))


def main():
    rep = Report("Lemma 3 and Theorem 3 (Dobrushin decay)")
    rng = np.random.default_rng(11)
    ok, worst = True, 0.0
    for _ in range(3000):
        k = rng.integers(2, 40); p = rng.dirichlet(np.ones(k)); h = rng.normal(size=k) * rng.uniform(0.01, 3)
        q = p * np.exp(h); q /= q.sum(); a = h.max() - h.min()
        tv = 0.5 * np.abs(p - q).sum(); ok &= tv <= math.tanh(a / 4) + 1e-12; worst = max(worst, tv / math.tanh(a / 4))
    a = 1.3; p = np.array([1, math.exp(a / 2)]); p /= p.sum(); q = p[::-1]
    p2 = np.array([0.5, 0.5]) * np.exp(np.array([-a / 4, a / 4])); p2 /= p2.sum(); q2 = p2[::-1]
    ok &= abs(0.5 * np.abs(p2 - q2).sum() - math.tanh(a / 4)) < 1e-12
    rep.check("(a) TV <= tanh(a/4) for log-ratio oscillation a; equality for symmetric two-point laws", ok,
              f"largest TV/tanh(a/4) over random laws {worst:.3f}")

    ok, det = True, []
    for shape, (pl, nb, tot) in (((3, 3, 3, 3), (6, 18, 18)), ((4, 4, 4, 4), (6, 18, 18)), ((4, 4), (2, 6, 6)), ((5, 3), (2, 6, 6))):
        idx, plaqs, per_link, n = adjacency_counts(shape)
        ok &= np.all(per_link == pl) and np.all((n > 0).sum(1) == nb) and np.all(n.sum(1) == tot) and n.max() == 1
        det.append(f"{len(shape)}D side {shape[0]}: {pl} plaquettes, {nb} neighbours")
    rep.check("(b) each link: 2(d-1) plaquettes x 3 other links; n_ej <= 1; sum_j n_ej = 18 in 4D, 6 in 2D", ok, "; ".join(det[::2]))

    U = haar_su2(2 ** 19, 5)
    ok, det = True, []
    for beta in (0.005, 0.01, 0.02, 0.05):
        bound = math.tanh(4 * beta)
        tvs = []
        for trial in range(30):
            st = haar_su2(8, 100 + trial)[:6] * (1 if trial % 3 else 0)        # random, or other staples switched off
            if trial % 5 == 0:
                st = np.tile(np.array([1.0, 0, 0, 0]), (6, 1))
            jo = haar_su2(8, 300 + trial)[0]; jn = -jo if trial % 2 else haar_su2(8, 500 + trial)[0]
            tvs.append(tv_conditional(beta, st, jo, jn, U))
        mx = max(tvs); ok &= mx <= bound * (1 + 2e-3)
        det.append(f"beta={beta}: max TV {mx:.5f} vs tanh(4 beta) {bound:.5f}")
    rep.check("(c) SU(2): one changed neighbour moves the conditional law by at most tanh(N^2 beta n_ej)", ok, "; ".join(det))

    bD = {N: math.atanh(1 / 18) / N ** 2 for N in (2, 3)}
    q = lambda N, b: 18 * math.tanh(N * N * b)
    ok = abs(q(2, bD[2]) - 1) < 1e-12 and q(2, 0.9 * bD[2]) < 1 and q(2, 1.1 * bD[2]) > 1
    ok &= 1 / 192 < bD[2] < 1 / 48
    rep.check("(d) q = 18 tanh(N^2 beta) < 1 iff beta < N^{-2} artanh(1/18); for SU(2) this lies between 1/192 and 1/48", ok,
              f"beta_D(2) = {bD[2]:.5f}, beta_D(3) = {bD[3]:.5f}; 1/192 = {1 / 192:.5f}, 1/48 = {1 / 48:.5f}")

    ok, det = True, []
    for beta in (0.002, 0.005, 0.01, 0.03):
        r = ive(2, 4 * beta) / ive(1, 4 * beta); qs = 6 * math.tanh(4 * beta)
        for t in range(1, 8):
            cov = 0.25 * r ** (3 * t)
            ok &= cov <= 4 * 3 / (1 - qs) * qs ** t
        det.append(f"beta={beta}: exact rate {-3 * math.log(r):.3f} vs strip q-rate {-math.log(qs):.3f}")
    rep.check("(e) exact strip covariance obeys the Dobrushin bound with the strip's q = 6 tanh(4 beta)", ok, "; ".join(det))
    return rep.summary()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
