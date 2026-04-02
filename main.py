# add tracking + in/out counter with fixed door line

import time
import cv2

from config import DetectionConfig
from detector import PersonDetector
from tracker import PersonTracker


def scale_dets(dets, scale):
    out = []
    inv = 1.0 / scale
    for d in dets:
        x1, y1, x2, y2 = d["box"]
        x1, y1, x2, y2 = int(x1 * inv), int(y1 * inv), int(x2 * inv), int(y2 * inv)
        out.append({
            "box": (x1, y1, x2, y2),
            "confidence": d["confidence"],
            "center": ((x1 + x2) // 2, (y1 + y2) // 2),
        })
    return out


def main():
    cfg = DetectionConfig()

    cam_index_raw = input("Webcam index (default 0): ").strip()
    cam_index = int(cam_index_raw) if cam_index_raw else 0

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print("could not open camera")
        return

    ok, warm = cap.read()
    if not ok:
        print("camera read failed")
        return

    h, w = warm.shape[:2]
    line_y = h // 2

    detector = PersonDetector(model_size=cfg.model_size)
    tracker = PersonTracker(door_line_y=line_y)

    fps_tick = time.time()
    fps_cnt = 0
    fps = 0

    print("tracking enabled. press ESC to quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        fps_cnt += 1

        small = cv2.resize(frame, (0, 0), fx=cfg.resize_scale, fy=cfg.resize_scale)
        dets_small = detector.detect(small, conf=cfg.confidence_threshold)
        dets = scale_dets(dets_small, cfg.resize_scale)

        events = tracker.update(dets)
        for e in events:
            print(f"person {e['id']} -> {e['direction']}")

        cv2.line(frame, (0, line_y), (w, line_y), (0, 255, 255), 2)

        for d in dets:
            x1, y1, x2, y2 = d["box"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (30, 220, 30), 2)

        if time.time() - fps_tick >= 1:
            fps = fps_cnt
            fps_cnt = 0
            fps_tick = time.time()

        cv2.putText(frame, f"FPS: {fps}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"IN: {tracker.people_in}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"OUT: {tracker.people_out}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("People Counter - Tracking", frame)

        if (cv2.waitKey(1) & 0xFF) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
