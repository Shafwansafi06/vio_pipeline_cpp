# `vio_pipeline` (v1) vs `vio_pipeline_v2` (v2) — What Actually Differs

This is a byte-level diff, not a guess. Only **8 files** differ between the
two packages (everything else — `type/`, `core/`, `msckf/State.py`,
`StateOptions.py`, `NoiseManager.py`, `UpdaterSLAM.py`, `updater_helper.py`,
`update_zero_velocity.py`, `initialize/`, all YAML configs except
`Custom_v1/estimator_config.yaml` — is **byte-identical** between v1 and
v2). If you're editing one of the identical files, the fix almost
certainly needs to go into both packages.

```
diff -rq vio_pipeline vio_pipeline_v2   (excluding __pycache__/, egg-info/)

  config/config_loader.py                 differs
  config/Custom_v1/estimator_config.yaml  differs
  main.py                                  differs
  msckf/Propagator.py                      differs
  msckf/state_helper.py                    differs
  msckf/UpdaterMSCKF.py                    differs
  msckf/UpdaterOptions.py                  differs
  msckf/VioManagerOptions.py               differs
  msckf/VioManager.py                      differs   (846-line diff, the big one)
  + v2-only: docs/, "Detecting and Dealing with Hovering Maneuvers...pdf"
```

## The short version

v2 = v1 **+ three independent additions**, all confined to the 8 files
above:

1. **FIFO/LIFO hovering detection** (the headline feature) — new state
   machine in `VioManager.py`, a frozen-covariance `propagate_only` path in
   `Propagator.py`, and ~60 new tunables in `VioManagerOptions.py` +
   `config_loader.py`.
2. **Numerical robustness guards** on the MSCKF update path —
   `UpdaterOptions.py`, `UpdaterMSCKF.py`, `state_helper.py` — reject or
   gracefully degrade on ill-conditioned/singular systems instead of
   silently producing garbage or hard-crashing.
3. **Fuller YAML config loading** — v1's `config_loader.py` only reads a
   subset of `estimator_config.yaml` keys; v2 reads the rest (FEJ toggle,
   integration method, calibration toggles, IMU model, hovering params,
   MSCKF robustness params).

None of these three are entangled with each other — you could in
principle port (2) or (3) into v1 without touching hovering at all.

## 1. FIFO/LIFO hovering detection — `VioManager.py`, `Propagator.py`, `VioManagerOptions.py`

### Methods that exist **only in v2** `VioManager`

v1's `VioManager` has 20 methods. v2 has 45. Everything below is new in
v2 — grep for these names if you're checking "is this a v1 or v2 concept":

```
extract_and_track_FORWARD          _hungarian_pairs
extract_and_track_BACKWARD         _update_chain_quality
_build_lifo_observations           _prune_backward_chains
_compute_orb_descriptor            _update_backward_feature_chains
_get_chain_last_obs                _backward_chains_as_features
_append_obs_to_chain               detect_hovering
_rotation_prior                    apply_temporal_smoothing
_predict_uv_norm                   switch_to_LIFO
_reprojection_residual             switch_to_FIFO
_descriptor_distance               msckf_state_only_update
_build_cost_and_meta               perform_covariance_update
                                    _snapshot_state_values
                                    _restore_state_values
                                    _collect_hovering_measurements
                                    fifo_update
                                    lifo_update
                                    _pose_for_time
```

### The key structural difference: `do_feature_propagate_update`

**v1**: `do_feature_propagate_update` is one monolithic method that always
does the same thing — propagate with cloning, then run the MSCKF/SLAM
update inline. There is no mode switching.

**v2**: `do_feature_propagate_update` became a **dispatcher**:

```
v1:                                    v2:
do_feature_propagate_update            do_feature_propagate_update
  propagate_and_clone()                  propagate_and_clone()  [FIFO]
  (inline MSCKF/SLAM update,               or propagate_only()    [LIFO]
   ~280 lines, unconditional)            detect_hovering() + apply_temporal_smoothing()
                                         switch_to_LIFO() / switch_to_FIFO() if triggered
                                         extract_and_track_BACKWARD()  [if LIFO]
                                         fifo_update()  or  lifo_update()
```

v2's `fifo_update` is functionally v1's old inline update logic, extracted
into its own method — **if you're porting a v1 bugfix into v2's normal
(non-hovering) path, look in `fifo_update`, not
`do_feature_propagate_update`.**

### `Propagator.py`: `propagate_only`

v2 adds one new method, `propagate_only(state, timestamp)` — identical
propagation math to `propagate_and_clone`, but **skips
`StateHelper.augment_clone`** (no new pose clone added). Used exclusively
while in LIFO/hovering mode, where the clone window is intentionally
frozen. v1 has no equivalent — it always clones.

### `VioManagerOptions.py`: ~30 new fields

All new, all v2-only, all under a `HOVERING MODE SWITCH (FIFO <-> LIFO)`
block:

```
sliding_window_size, hovering_threshold, hover_smoothing_required,
hover_baseline_threshold, hover_epipolar_inlier_threshold,
hover_bearing_inlier_threshold, hover_min_inliers,
hover_entry_max_baseline, hover_entry_max_velocity,
hover_exit_velocity_threshold, hover_max_duration_sec,
lifo_backward_match_threshold, lifo_backward_min_chain_length,
lifo_backward_reproj_threshold, lifo_backward_mutual_consistency,
lifo_backward_max_age_steps, lifo_orb_descriptor_fallback,
lifo_orb_hamming_threshold, lifo_quality_init, lifo_quality_inc_match,
lifo_quality_inc_orb, lifo_quality_dec_miss, lifo_quality_min_keep,
lifo_quality_min_output, lifo_max_misses
```

Full semantics of each: [vio-manager.md](vio-manager.md#hovering-detection-fifolifo-state-machine).

### `main.py`

Trivial: v2 adds one `rospy.loginfo` line at startup logging which
`VioManager` module got imported (a debugging aid for when both packages
are on the Python path simultaneously — makes it obvious at runtime which
version is actually running). No behavioral difference.

## 2. Numerical robustness guards — `UpdaterOptions.py`, `UpdaterMSCKF.py`, `state_helper.py`

These are **unrelated to hovering** — they harden the ordinary MSCKF update
path against degenerate feature geometry (near-parallel rays, singular
innovation covariance, etc.) regardless of FIFO/LIFO mode.

### `UpdaterOptions.py` — 4 new fields (v2 only)

```python
min_features_for_update = 3       # skip an update batch with too few surviving features
min_update_rows = 6                # skip if the compressed system has too few rows
max_innovation_condition = 1e12    # reject a feature if S's condition number exceeds this
ekf_innovation_jitter = 1e-9       # diagonal jitter added to S for invertibility
```

v1 has none of these — `UpdaterOptions` is just `chi2_multipler` and
`sigma_pix`.

### `UpdaterMSCKF.py` — new rejection paths using those fields

v2 adds, in `update()`:
- Early-out if `len(feature_vec) < min_features_for_update` after
  triangulation (marks everything `to_delete`, logs `[MSCKF-UP][SKIP]
  too_few_features`, returns).
- Per-feature: reject if the nullspace-projected `H_x`/`res` is empty or
  contains non-finite values.
- Per-feature: symmetrize `S`, compute its condition number, reject if
  non-finite or `> max_innovation_condition`.
- Batch-level: reject if the compressed system has fewer than
  `min_update_rows`, or contains non-finite values.

v1's `UpdaterMSCKF.update()` has none of these checks — a degenerate
feature or ill-conditioned batch flows straight into the Kalman update.

### `state_helper.py` — `EKFUpdate` no longer crashes the process

Two changes, both in `EKFUpdate`:

| | v1 | v2 |
|---|---|---|
| Innovation covariance `S` | no jitter | `+= 1e-9` on the diagonal before Cholesky |
| Cholesky failure fallback | `np.linalg.inv(S)` | `np.linalg.pinv(S)` (safer for near-singular `S`) |
| Negative covariance diagonal after update | **`sys.exit(1)`** — kills the whole process | Floors negative/tiny variances to `1e-12` and continues |

**This is the single most consequential line-level difference.** In v1, a
single numerically bad update anywhere in a long-running ROS node takes
the entire process down. In v2, the filter degrades gracefully and keeps
running. If you're debugging "why did v1 crash but v2 didn't" (or vice
versa, if you're deliberately trying to catch a bug loudly), this is why.
See [estimator-core.md](estimator-core.md#ekfupdatestate-h_order-h-res-r)
for the full update-equation context.

## 3. Fuller config loading — `config_loader.py`, `Custom_v1/estimator_config.yaml`

v1's `_load_estimator_config` only reads: `max_cameras`, `use_stereo`,
`max_clones`, `max_slam`, tracker params, ZUPT params, `gravity_mag`,
`up_msckf_sigma_px`, `up_msckf_chi2_multipler`. Everything else falls back
to whatever `VioManagerOptions.__init__`/`StateOptions.__init__` hardcode.

v2's loader additionally reads (and v1 silently ignores if present in the
YAML):

```
use_fej, integration, calib_cam_extrinsics, calib_cam_intrinsics,
calib_cam_timeoffset, calib_imu_intrinsics, calib_imu_g_sensitivity,
use_mask, hovering_threshold and all hover_*/lifo_* keys,
msckf_min_features_for_update, msckf_min_update_rows,
msckf_max_innovation_condition, msckf_ekf_innovation_jitter,
imu0.model (kalibr/calibrated/rpng IMU intrinsic parameterization)
```

**Practical consequence**: if you hand v1's `estimator_config.yaml` to v2
(or vice versa), it will load without error either way — but any of the
v2-only keys present in a YAML will be silently ignored by v1's loader,
and any v2-only feature (hovering, MSCKF robustness) will silently use
its hardcoded default rather than your YAML value if you're running it
through v1's loader (which is a no-op there anyway, since v1's
`VioManager`/`UpdaterMSCKF` don't read those fields at all).

`Custom_v1/estimator_config.yaml` itself differs between the two packages
— the v2 copy has the hovering/robustness keys added; the v1 copy doesn't.
`kaist_vio/estimator_config.yaml` and both `kalibr_*_chain.yaml` files are
identical in both packages.

## Practical checklist when working across both packages

- **Fixing a bug in tracking, cameras, triangulation, SLAM updates, ZUPT,
  or initialization?** Those files are identical — fix it in both
  `vio_pipeline/` and `vio_pipeline_v2/` (there's no automated sync between
  them; copy the change manually).
- **Fixing a bug in the MSCKF update numerics, propagation, state
  bookkeeping, or config loading?** Check this doc's diff first — the
  fix may only be relevant to v2 (if it's about a robustness guard or
  hovering) or may need porting to both (if it's a genuine bug in shared
  logic that happens to sit in one of the 8 differing files, e.g. a typo
  in `predict_and_compute`'s Xi_sum math that both versions share
  line-for-line except for the `propagate_only` addition).
- **Adding a new estimator feature?** Decide up front whether it belongs
  in both packages or is v2-specific (hovering-adjacent). If it's
  general-purpose (like the robustness guards), consider whether v1 should
  get it too rather than letting the packages drift further apart.
- **Confused about whether a symbol exists?** `grep -rn "<symbol>"
  vio_pipeline/ vio_pipeline_v2/` — if it only hits one, that tells you
  immediately which package the code you're reading actually belongs to.
