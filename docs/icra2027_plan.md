# ICRA 2027 — status pointer

**This file is a pointer, not a plan.** The working state lives in two
places that are updated continuously:

- `docs/paper_spine.md` — which paper the repo is building (systems spine,
  decided 2026-08-30) and what that choice costs.
- `docs/tickets/BOARD.md` — the ticket board; P-01..P-07 are the paper
  tickets, T-00x the closed research tickets.

## Where things stand (2026-08-30, end of day)

- **Speed/systems paper is the submission.** 1.78–2.10x wall-clock at ATE
  parity on x86; p99 latency advantage exceeds the mean advantage; embedded
  validation complete on the Orin Nano (all ten sequences, same toolchain
  container, honest narrower margin); toolchain confound measured (P-05);
  transport confound re-measured properly (P-04: 0–27% per-sequence,
  previously misattributed 15% now understood as the toolchain confound);
  ov_SchurVINS row dropped after the fork proved unreproducible (P-03).
- **Everything in the paper's tables is reproduced from the current tree
  in the pinned container** as of 2026-08-30, except the OpenVINS V1 rows
  and the RPE table, which predate the re-verification pass — see the
  board before quoting them.
- Earlier sections of this file (the altitude-covariance plan, the stale
  recorded baseline, the "build is broken" notes) described a state that
  no longer exists. They are superseded by the board, not updated here.
- Deadline: check the ICRA 2027 CFP (historically mid-September).
