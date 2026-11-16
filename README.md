# Validation suite: From Horizontal Curvature to a Wilson Transfer-Matrix Gap in SU(N) Lattice Yang–Mills

Companion code to the white paper *From Horizontal Curvature to a Wilson Transfer-Matrix Gap in SU(N) Lattice Yang–Mills*
(R. Cooperman and A.E. Petrunina V.V., XISOSIS, Vienna, 16 November 2026).

- Repository: `[GitHub link placeholder]`
- DOI: `[Zenodo DOI placeholder]`
- Licence: MIT (see `LICENSE`; every Python file carries the licence header)

## Running

```
python -m pip install -r requirements.txt
python run_all.py          # about 2 minutes; exit code 0 only if every check passes
```

`run_log.txt` is the output of a complete run (22/22 checks pass).

## Coverage

| Script | Paper result | Checks |
|---|---|---|
| `validate_part1_lsi.py` | Theorem 1: every identity of the proof along the exact one-loop flow (mass conservation, dEnt/dt = −I, the formula for I′, I(t) ≤ e^{−2Λ}I(0), Ent = ∫I), the inequality with 2C_κ, and the β = 0 normalization | 6 |
| `validate_part2_oneloop.py` | Proposition 2: reflection, Ent(f²) = 4βm, 3m = 2βE[sin²θ], ∫\|∇f\|² = 3βm, ratio 2/3; C_opt ≤ 2/3 ≤ C_LS ≤ C_κ; numerical optimum of C_LS | 5 |
| `validate_part3_decay.py` | Lemma 3 and Theorem 3: TV ≤ tanh(a/4); lattice counts (6 plaquettes, 18 neighbours, n_ej ≤ 1); the SU(2) single-link influence bound by quasi-Monte Carlo; the threshold N⁻²artanh(1/18); the covariance bound against the exact strip | 5 |
| `validate_part4_gap.py` | Theorem 4 and Section 6: time-layer geometry of the 4D link graph; character coefficients and positivity; the strip gap and every table entry; winding versus adjoint characters; the spectral checkpoint; rates below the exact gap | 6 |

Shared code:
- `common.py` (Report class, finite-element C_opt, exact C_κ for one loop) and `charflow.py` (character-basis heat flow on
  SU(2) class functions) are reused from the earlier suites of this programme.
- `lattice.py` builds link-adjacency graphs of periodic hypercubic lattices.

## What is not validated

- Theorem 1 is checked on one SU(2) loop, where the flow is exact; the four-dimensional lattice statement is proved in the
  paper, not computed.
- C_LS(β) = 2/3 on one loop is a numerical observation over finite-dimensional families of test functions, not a theorem.
- Theorems 3 and 4 use the Dobrushin–Föllmer covariance estimate and reflection positivity of the Wilson action as cited
  results; the suite checks their inputs (influence coefficients, geometry) and their conclusions on an exactly solvable strip.
- Problem B-1 (boundary control from the curvature profile alone) is open and has no check.
- The comparison of the Wilson transfer matrix with the Kogut–Susskind Hamiltonian is not addressed.
