#!/usr/bin/env python3
"""Extract a EuRoC ROS1 bag into ASL folder layout: images, IMU, ground truth.

Images are written as binary PGM, which is byte-for-byte the mono8 payload plus
a header, so the pixels the trackers see are exactly the bag's. cv::imread reads
PGM as grayscale, same as PNG, with no encoder in the path to alter a single
pixel. IMU and Leica ground truth are written as the dataset's own CSV schemas,
so tools/dod_asl_runner.cpp sees exactly what the ROS runner saw.

Needs `rosbags` (pure python, no ROS install).

    python3 tools/bag_to_asl.py MH_01_easy.bag out_dir/mav0
"""

import sys
from pathlib import Path

from rosbags.highlevel import AnyReader

TOPICS = {"/cam0/image_raw": "cam0", "/cam1/image_raw": "cam1"}
IMU_TOPIC = "/imu0"
TRUTH_TOPIC = "/leica/position"


def stamp_ns(msg):
    return msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    bag, out = Path(sys.argv[1]), Path(sys.argv[2])

    index = {name: [] for name in TOPICS.values()}
    for name in TOPICS.values():
        (out / name / "data").mkdir(parents=True, exist_ok=True)
    (out / "imu0").mkdir(parents=True, exist_ok=True)
    (out / "leica0").mkdir(parents=True, exist_ok=True)
    imu_rows, truth_rows = [], []

    with AnyReader([bag]) as reader:
        wanted = set(TOPICS) | {IMU_TOPIC, TRUTH_TOPIC}
        conns = [c for c in reader.connections if c.topic in wanted]
        if not any(c.topic in TOPICS for c in conns):
            print(f"no image topics in {[c.topic for c in reader.connections]}")
            return 3
        for conn, _, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            if conn.topic == IMU_TOPIC:
                w, a = msg.angular_velocity, msg.linear_acceleration
                imu_rows.append((stamp_ns(msg), w.x, w.y, w.z, a.x, a.y, a.z))
                continue
            if conn.topic == TRUTH_TOPIC:
                p = msg.point
                truth_rows.append((stamp_ns(msg), p.x, p.y, p.z))
                continue
            if msg.encoding != "mono8":
                print(f"unexpected encoding {msg.encoding}")
                return 4
            cam = TOPICS[conn.topic]
            ns = stamp_ns(msg)
            data = bytes(msg.data)
            if msg.step != msg.width:  # de-pad rows so the PGM payload is dense
                data = b"".join(data[r * msg.step:r * msg.step + msg.width]
                                for r in range(msg.height))
            name = f"{ns}.pgm"
            with open(out / cam / "data" / name, "wb") as handle:
                handle.write(b"P5\n%d %d\n255\n" % (msg.width, msg.height))
                handle.write(data)
            index[cam].append((ns, name))

    for cam, rows in index.items():
        rows.sort()
        with open(out / cam / "data.csv", "w") as handle:
            handle.write("#timestamp [ns],filename\n")
            for ns, name in rows:
                handle.write(f"{ns},{name}\n")
        print(f"{cam}: {len(rows)} images")

    imu_rows.sort()
    with open(out / "imu0" / "data.csv", "w") as handle:
        handle.write("#timestamp [ns],w_RS_S_x,w_RS_S_y,w_RS_S_z,a_RS_S_x,a_RS_S_y,a_RS_S_z\n")
        for row in imu_rows:
            handle.write("%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n" % row)
    print(f"imu0: {len(imu_rows)} samples")

    truth_rows.sort()
    with open(out / "leica0" / "data.csv", "w") as handle:
        handle.write("#timestamp [ns],p_RS_R_x,p_RS_R_y,p_RS_R_z\n")
        for row in truth_rows:
            handle.write("%d,%.17g,%.17g,%.17g\n" % row)
    print(f"leica0: {len(truth_rows)} samples")
    return 0


if __name__ == "__main__":
    sys.exit(main())
