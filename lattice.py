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
"""Link-adjacency graphs of periodic hypercubic lattices (links adjacent when they share a plaquette)."""
import itertools
from collections import deque

import numpy as np


def links_and_plaquettes(shape):
    d = len(shape)
    sites = list(itertools.product(*[range(n) for n in shape]))
    idx = {}
    for s in sites:
        for mu in range(d):
            idx[(s, mu)] = len(idx)

    def shift(s, mu):
        s = list(s); s[mu] = (s[mu] + 1) % shape[mu]; return tuple(s)
    plaqs = []
    for s in sites:
        for mu in range(d):
            for nu in range(mu + 1, d):
                plaqs.append((idx[(s, mu)], idx[(shift(s, mu), nu)], idx[(shift(s, nu), mu)], idx[(s, nu)]))
    return idx, plaqs


def adjacency_counts(shape):
    idx, plaqs = links_and_plaquettes(shape)
    n = np.zeros((len(idx), len(idx)), dtype=int)
    per_link = np.zeros(len(idx), dtype=int)
    for p in plaqs:
        for e in p:
            per_link[e] += 1
            for j in p:
                if j != e:
                    n[e, j] += 1
    return idx, plaqs, per_link, n


def bfs_dist(n, sources):
    dist = np.full(n.shape[0], -1); q = deque()
    for s in sources:
        dist[s] = 0; q.append(s)
    nbrs = [np.nonzero(n[i])[0] for i in range(n.shape[0])]
    while q:
        i = q.popleft()
        for j in nbrs[i]:
            if dist[j] < 0:
                dist[j] = dist[i] + 1; q.append(j)
    return dist
