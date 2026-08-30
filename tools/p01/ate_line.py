import json
import sys

seq = sys.argv[1]
recorded = sys.argv[2] if len(sys.argv) > 2 else ""
d = json.load(sys.stdin)
print("%-9s %-10.4f %s" % (seq, d["ate_rmse_m"], recorded))
