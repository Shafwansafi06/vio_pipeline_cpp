#!/usr/bin/env python3
"""Write a groundtruth.csv from the V1_* vicon TransformStamped topic.

The EuRoC machine-hall bags carry /leica/position (PointStamped) which the
rosbag runner already dumps; the vicon-room bags carry
/vicon/firefly_sbx/firefly_sbx (TransformStamped) instead, so the runner writes
an empty ground truth for V1_*. Same CSV columns, so the scorer is unchanged.
"""
import sys
import rosbag

bag_path, out_path = sys.argv[1], sys.argv[2]
TOPIC = "/vicon/firefly_sbx/firefly_sbx"
n = 0
with rosbag.Bag(bag_path) as bag, open(out_path, "w") as out:
    out.write("timestamp,px,py,pz,qx,qy,qz,qw\n")
    for _, msg, _ in bag.read_messages(topics=[TOPIC]):
        t = msg.transform.translation
        r = msg.transform.rotation
        out.write("%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f\n" % (
            msg.header.stamp.to_sec(), t.x, t.y, t.z, r.x, r.y, r.z, r.w))
        n += 1
print(f"wrote {n} ground-truth poses to {out_path}")
