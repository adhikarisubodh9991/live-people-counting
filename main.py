# interactive setup + counting app

import cv2
import time

from config import DetectionConfig
from detector import PersonDetector
from tracker import PersonTracker
from setup_flow import CameraSetup, DoorLineSetup


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


def draw(frame, dets, tracker, p1, p2, fps):
    cv2.line(frame, p1, p2, (0, 255, 0), 3)

    for d in dets:
        x1, y1, x2, y2 = d["box"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (20, 220, 20), 2)

    cv2.rectangle(frame, (0, 0), (320, 160), (0, 0, 0), -1)
    cv2.putText(frame, f"FPS: {fps}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"IN: {tracker.people_in}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"OUT: {tracker.people_out}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"ACTIVE: {len(tracker.get_active_persons())}", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def main():
    setup = CameraSetup()
    while not setup.show_menu():
        pass

    if not setup.test_camera():
        print("camera test failed")
        return

    line_setup = DoorLineSetup(setup.config)
    if not line_setup.show_line_selector():
        print("line not set")
        return

    cfg = DetectionConfig()
    detector = PersonDetector(model_size=cfg.model_size)
    tracker = PersonTracker(
        door_line_y=setup.config.door_line_y,
        door_line_start=setup.config.door_line_start,
        door_line_end=setup.config.door_line_end,
    )

    if setup.config.camera_type == "webcam":
        cap = cv2.VideoCapture(setup.config.camera_index)
    else:
        cap = cv2.VideoCapture(setup.config.camera_url)

    if not cap.isOpened():
        print("failed to open camera")
        return

    fps_tick = time.time()
    fps_cnt = 0
    fps = 0

    cv2.namedWindow("People Counter", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("People Counter", 1000, 650)

    p1 = setup.config.door_line_start
    p2 = setup.config.door_line_end

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
            now = time.strftime("%H:%M:%S")
            print(f"[{now}] person {e['id']} -> {e['direction']}")

        if time.time() - fps_tick >= 1:
            fps = fps_cnt
            fps_cnt = 0
            fps_tick = time.time()

        draw(frame, dets, tracker, p1, p2, fps)
        cv2.imshow("People Counter", frame)

        if (cv2.waitKey(1) & 0xFF) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
