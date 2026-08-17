# SchurVINS: what we implemented, and can we do better?

Paper: Fan et al., *SchurVINS: Schur Complement-Based Lightweight Visual
Inertial Navigation System*, CVPR 2024. Reference code: `Jas0nG/ov_SchurVINS`
(read as reference, never vendored -- GPL).

**Short answers.**
1. We implemented the ego-motion Schur reduction (§3.3) data-oriented, plus the
   noise model from their eq. 28 which their own released code abandons.
2. Yes, we do better. On an equal-footing comparison inside this pipeline the
   Schur reduction buys **0.19 ms/frame (4.6% of total) for a 63% worse ATE**.

---

## 1. What is implemented, by paper section

| § | Component | Status |
|---|---|---|
| 3.2 | Propagation & augmentation (RK4, Phi/Q) | **Already had it** -- this is standard MSCKF, not SchurVINS-specific |
| 3.3 | Residual model projected to Jacobian space, eq. 20-23 | **Implemented** (`msckf/updater_schur.cpp`) |
| 3.3 | Schur complement of the landmark block, eq. 24-31 | **Implemented** |
| 3.3 | `R'' = (C1 - C2 C3^-1 C2^T) u^2`, eq. 28 | **Implemented** -- see §2, their code does not |
| 3.4 | EKF-based landmark solver, eq. 38-40 (independent 3x3 per landmark) | **Not implemented** -- see §4 |
| 3.5 | SVO2.0 frontend: local map, depth filters, feature alignment | **Not implemented**, and not comparable to ours |
| 3.6 | Keyframe selection | **Not implemented** |

Faithful to the reference where it differs from us: no chi2 gate (Huber weight
only), `obs_invdev = 0.25`, GLOBAL_3D representation only, no FEJ, no camera
calibration.

## 2. Where the paper and its code disagree -- and the paper is right

Paper eq. 23 and 28 derive the equivalent observation covariance as the Hessian
scaled by the pixel variance:

```
R'  = J^T R J           = [Jx^T Jx ...] u^2         (eq. 23)
R'' = (C1 - C2 C3^-1 C2^T) u^2                      (eq. 28)
```

The released code ships

```cpp
//  Eigen::MatrixXd R_big = _options.sigma_pix_sq * Amtx;     // <- eq. 28, commented out
Eigen::MatrixXd R_big = _options.sigma_pix_sq * Eigen::MatrixXd::Identity(...);
```

We derived `cov(b) = sigma^2 A` independently (from `b = J^T n`, the Schur
cross-terms cancel exactly), hit the numerical wall that presumably caused them
to abandon it -- `S = A P A + sigma^2 A` scales as `J^4`, covariance diagonal at
-1.6e9 within three updates -- and resolved it by whitening instead of
discarding: with `A = L L^T`,

```
L^-1 b = L^T x~ + L^-1 n,      cov(L^-1 n) = sigma^2 I
```

so `H = L^T`, `res = L^-1 b`, `R = sigma^2 I`. This keeps eq. 28's covariance
and is numerically stable. Ported literally instead, the shipped rule diverges
here (1.4e5 m ATE, `|dx|` of 0.6-2.4 m on the first updates, at every damping
scale from 1e2 to 1e8).

## 3. Equal-footing comparison, EuRoC MH_01

Both arms identical except the update rule (FEJ and intrinsic calibration off on
both, since the reference supports neither):

| | baseline MSCKF | Schur reduction |
|---|---|---|
| ATE | **0.1129 m** | 0.1837 m (+63%) |
| MSCKF update stage | 0.540 ms | **0.351 ms** |
| share of a 4.15 ms frame | 13% | 8.5% |

**The trade is bad**: 0.19 ms/frame, 4.6% of total frame time, for 63% more
error. And the advantage is shrinking as the baseline improves -- it was
0.786 vs 0.476 ms (1.65x) before the optimisation work in Benchmarks 12-15, and
is 0.540 vs 0.351 (1.54x) now, because the costs Schur avoids (nullspace
projection, QR compression) are exactly the ones we made cheap.

Why the accuracy loss is structural, not a tuning miss:

- **Their `obs_invdev = 0.25` is load-bearing damping.** It scales residuals and
  Jacobians as if the measurement sigma were 4 px against our 1 px. Setting the
  statistically correct 1.0 diverges (9.1e3 m); 0.5 diverges (96 m). Stability
  requires assuming 16x more measurement variance than the sensor has.
- **Adding outlier rejection does not recover it.** A chi2 gate is free in this
  formulation -- `r^T r - gv^T V^-1 gv` is exactly the nullspace-projected
  residual energy with 2m-3 dof, two dot products on quantities the reduction
  already computed. Adding it moved MH_01 from 0.1837 to 0.1861, i.e. nothing,
  because the damping above makes the statistic ~16x too small to fire.

## 4. The one idea we have not taken, and why

§3.4 is the part we have *not* implemented, and it is where their real speed
comes from: **landmarks are not in the global covariance at all**. Each keeps
its own independent 3x3 (Fig. 3b), so the landmark update decomposes into `m`
tiny independent EKFs (eq. 40) instead of carrying 150 extra covariance
dimensions the way our SLAM path does.

That would attack our largest remaining stage (SLAM, 0.90-1.04 ms/frame). It is
also a deliberate approximation -- it discards pose-landmark cross-covariance --
and this pipeline's brief is speed *with accuracy intact*. It is the obvious
next experiment, and it should be judged on accuracy, not assumed.

## 5. So, better than Schur?

Within this pipeline, yes, and the numbers are above. Against the paper's own
results the comparison is not meaningful and should not be claimed: SchurVINS
reports 0.075 m mean on EuRoC against our 0.178, but it runs on an SVO2.0
frontend with a local map, depth filters and keyframe selection (§3.5-3.6) --
Table 1 lists SVO2.0 alone at 0.109, i.e. better than our whole system. Their
frontend is doing work ours does not. The estimator-level comparison in §3 is
the one that holds the frontend fixed, and it favours ours.

Where we ended up relative to the paper's actual goal -- a lightweight,
fast filter -- is: **4.15 ms/frame, 1.94x faster than official OpenVINS
end-to-end**, reached by making the conventional path cheap rather than by
changing the estimator.

---

## 6. Head-to-head against the authors' own implementation (2026-08-17)

We built `ov_SchurVINS` (their released code, a fork of OpenVINS) in
`ros_container_v2` and ran all three systems on the same bag, same machine, same
evaluator. SchurVINS does not support dynamic initialisation, so all three use
its intended protocol: `bag_start 40`, static init -- the same 40 s skip their
own launch file defaults to for MH_01.

### Wall clock, same transport (hyperfine, 5 runs)

| | mean | sigma | |
|---|---|---|---|
| **DOD (ours)** | **10.752 s** | 0.013 | |
| ov_SchurVINS | 21.146 s | 0.193 | we are **1.97x faster** |
| OpenVINS | 23.247 s | 0.083 | we are **2.16x faster** |

Whole process, identical bag through each system's own bag reader.

### Per-frame, internal timers (same segment)

| | ATE | tracking | estimator | total |
|---|---|---|---|---|
| **DOD (ours)** | 0.1431 | **1.546** | **2.089** | **3.635 ms** |
| ov_SchurVINS | **0.1365** | 2.113 | 4.399 | 6.512 ms |
| OpenVINS | 0.1476 | 2.116 | 5.136 | 7.252 ms |

**The estimator -- the part SchurVINS is about -- is 2.11x faster in ours**
(2.089 vs 4.399 ms), reached without their Schur reduction, by making the
conventional path cheap.

### On "running what they already have in the frontend"

Their frontend *is* OpenVINS's TrackKLT: 2.113 ms/frame, against 1.546 ms for
ours doing the same job (§Benchmark 14 measured our tracker's quality as equal
or better on lifetime, parallax and epipolar residual). Adopting their frontend
verbatim and keeping our back end would give 2.113 + 2.089 = **4.202 ms**, still
**1.55x faster** than their 6.512 -- so the result is not a frontend artefact.

### The honest caveat

**They are more accurate on this segment: 0.1365 vs our 0.1431, a 4.8% gap.**
Both beat OpenVINS (0.1476). So the trade is real and it is theirs to claim on
accuracy; ours is the speed claim. Note also that this is the ov_SchurVINS port
on an OpenVINS frontend, not the paper's SVO2.0 system, whose 0.075 m mean is
not comparable to any number here.

### Conclusion

Yes -- faster than SchurVINS, by ~2x end to end and 2.1x in the estimator, at
4.8% more error, on their own protocol and their own implementation.
