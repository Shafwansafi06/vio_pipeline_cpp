# Which spine: systems, not methodology

Decided 2026-08-30. The paper currently straddles two contributions. This
records which one it commits to and what that costs.

## The choice

**Systems paper: the fastest and most predictable MSCKF implementation,
validated on the hardware where memory layout actually matters.**

The methodology material — transport confound, toolchain confound, the
control-tree comparison method — stays in the paper as a *methods* section that
earns the main result its credibility. It does not become the thesis.

## Why systems

**The thesis is already systems.** The abstract's own sentence is "what differs
is memory layout, and the paper reports what that layout change costs and
buys". Every asset built over the last two weeks serves that: the per-optimization
ablation with bit-identical trajectories, the DHAT allocation baseline, the
bit-exact accuracy gate, the per-frame timing CSVs. A methodology spine would
discard most of it.

**Accuracy parity is a feature here, not a weakness.** It reads as a weakness
only if the axis is accuracy. For a systems paper the axis is cost at fixed
accuracy, and parity is the control that makes the speed number mean something.
"1.78-2.10x at ATE parity, every trajectory bit-identical across the
optimizations" is a clean claim. "We are also slightly more accurate" would
actually muddy it.

**The two weakest sections are both fixable on the systems axis and neither is
fixable on the methodology axis.** Predictability (p99, jitter, RSS) is the
claim's own metric and is missing; the data for it is already on disk.
Embedded is the thesis's proof and is currently feasibility-only. Both are
systems work.

**The embedded platform is where the argument is strongest.** 2x on a 6-core
A78AE with a small cache is a different and better claim than 2x on a 7950X,
because it is the platform where layout dominates. That asymmetry is the
paper's best available "so what", and it is currently unexploited.

## Why not methodology

A fair-benchmarking paper needs the full 10 x 2 x 2 transport matrix plus the
toolchain matrix, and its central move is arguing that other people's published
numbers are confounded. That is a large matrix and a reviewer-hostile framing
without one. It also throws away the Jetson, the SchurVINS comparison, the
ablation and the allocation profile — the parts nobody else can produce.

The confounds are real and worth publishing. They are stronger as the reason
this paper's numbers can be trusted than as the paper's point.

## What this means for the draft

- **Keep and lead with:** layout thesis, ablation, allocation profile, wall-clock
  at parity, silent-failure-path fixes.
- **Promote:** predictability numbers into the abstract if p99 beats mean
  (P-01), embedded from "preliminary" to a real section (P-02).
- **Demote to methods:** transport confound, toolchain confound, control-tree
  comparison. One subsection, tightly argued, cited as why the numbers hold.
- **Cut or defer:** anything that argues about other papers' rigour at length.
- **Add if it lands:** SchurVINS head-to-head (P-03) as the second baseline.

## The honest risk

If the Orin Nano stays unreachable, P-02 cannot land, and the paper keeps its
weakest section. In that case the fallback is RA-L rather than ICRA: the
existing result is a strong RA-L paper today, and RA-L has no deadline
pressure. That decision point arrives when the device does.
