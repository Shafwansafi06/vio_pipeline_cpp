# Control-tree comparison: does a change move the trajectories?

The method used by T-003 and every accepted optimisation since: build the
unchanged base commit and the change side by side in the SAME environment,
run both on the SAME data, and compare estimate CSVs byte-for-byte. Exact,
immune to environment drift, and it has resolution (distinct md5 per
sequence proves the comparison can tell two things apart).

Usage (on the lab box, moonlab (lab x86 box), in ros_container_v2):

    BASE=<base-commit-or-ref> NAME=<label> DATA=<euroc-asl-dir> bash tools/control_tree.sh

What it does:
  1. `git archive` HEAD and BASE into two trees under the workspace
  2. builds both with identical flags in the same container
  3. runs both on $DATA
  4. `cmp` the two estimate CSVs per sequence and reports identical/differ
     plus md5 of each (so a null result can be checked for resolution)

Nothing in this script touches the estimator; it is glue around
git-archive, cmake, and cmp.
