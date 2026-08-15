#!/usr/bin/env python3
"""Write a EuRoC ASL folder back out as a ROS1 bag, so official OpenVINS can run
on sequences that were only ever distributed in ASL form.

The EuRoC downloads that are still reachable are mostly ASL zips, but official
reads bags only. This produces the same topics its euroc_mav config expects:
/cam0/image_raw, /cam1/image_raw (mono8) and /imu0. Ground truth is not written
-- both pipelines are scored against the ASL CSV directly.

Needs `rosbags` (pure python, no ROS install).

    python3 tools/asl_to_bag.py path/to/mav0 out.bag
"""

import sys
from pathlib import Path

import numpy as np
from rosbags.rosbag1 import Writer
from rosbags.typesys import Stores, get_typestore


def read_csv(path):
    rows = []
    with open(path) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            parts = line.strip().split(",")
            if len(parts) >= 2:
                rows.append(parts)
    return rows


def read_image(path):
    """Minimal binary PGM / PNG loader -> (height, width, bytes)."""
    if path.suffix.lower() == ".pgm":
        data = path.read_bytes()
        fields, offset = [], 0
        while len(fields) < 4:
            end = data.index(b"\n", offset)
            token = data[offset:end]
            offset = end + 1
            if not token.startswith(b"#"):
                fields.extend(token.split())
        width, height = int(fields[1]), int(fields[2])
        return height, width, data[offset:offset + width * height]
    import cv2
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return image.shape[0], image.shape[1], image.tobytes()


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    mav0, out = Path(sys.argv[1]), Path(sys.argv[2])
    store = get_typestore(Stores.ROS1_NOETIC)
    Image = store.types["sensor_msgs/msg/Image"]
    Imu = store.types["sensor_msgs/msg/Imu"]
    Header = store.types["std_msgs/msg/Header"]
    Time = store.types["builtin_interfaces/msg/Time"]
    Quaternion = store.types["geometry_msgs/msg/Quaternion"]
    Vector3 = store.types["geometry_msgs/msg/Vector3"]

    def stamp(ns):
        return Time(sec=int(ns // 10**9), nanosec=int(ns % 10**9))

    if out.exists():
        out.unlink()
    with Writer(out) as writer:
        conns = {}
        for topic, msgtype in (("/cam0/image_raw", "sensor_msgs/msg/Image"),
                               ("/cam1/image_raw", "sensor_msgs/msg/Image"),
                               ("/imu0", "sensor_msgs/msg/Imu")):
            conns[topic] = writer.add_connection(topic, msgtype, typestore=store)

        # Interleave by timestamp so bag order matches capture order.
        events = []
        for cam, topic in (("cam0", "/cam0/image_raw"), ("cam1", "/cam1/image_raw")):
            for row in read_csv(mav0 / cam / "data.csv"):
                events.append((int(row[0]), topic, mav0 / cam / "data" / row[1]))
        for row in read_csv(mav0 / "imu0" / "data.csv"):
            events.append((int(row[0]), "/imu0", [float(x) for x in row[1:7]]))
        events.sort(key=lambda e: (e[0], e[1]))

        zero = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        cov = np.zeros(9, dtype=np.float64)
        for ns, topic, payload in events:
            header = Header(stamp=stamp(ns), frame_id=topic.strip("/").split("/")[0])
            if topic == "/imu0":
                message = Imu(header=header, orientation=zero, orientation_covariance=cov,
                              angular_velocity=Vector3(x=payload[0], y=payload[1], z=payload[2]),
                              angular_velocity_covariance=cov,
                              linear_acceleration=Vector3(x=payload[3], y=payload[4], z=payload[5]),
                              linear_acceleration_covariance=cov)
            else:
                height, width, data = read_image(payload)
                message = Image(header=header, height=height, width=width, encoding="mono8",
                                is_bigendian=0, step=width,
                                data=np.frombuffer(data, dtype=np.uint8))
            writer.write(conns[topic], ns, store.serialize_ros1(message, message.__msgtype__))
    print(f"wrote {out} ({len(events)} messages)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
