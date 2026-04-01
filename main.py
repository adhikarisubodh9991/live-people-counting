# camera + person detection boxes

import cv2
from config import DetectionConfig
from detector import PersonDetector


def main():
    cfg = DetectionConfig()

    cam_index_raw = input("Webcam index (default 0): ").strip()
    cam_index = int(cam_index_raw) if cam_index_raw else 0

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print("could not open camera")
        return

    detector = PersonDetector(model_size=cfg.model_size)

    print("running detection... press ESC to quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("camera read failed")
            break

        dets = detector.detect(frame, conf=cfg.confidence_threshold)

        for det in dets:
            x1, y1, x2, y2 = det["box"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (30, 220, 30), 2)
            cv2.putText(
                frame,
                f"person {det['confidence']:.2f}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (30, 220, 30),
                1,
            )

        cv2.imshow("People Counter - Detection", frame)

        if (cv2.waitKey(1) & 0xFF) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
