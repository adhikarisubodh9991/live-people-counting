#yolo detector wrapper

from ultralytics import YOLO


class PersonDetector:
    def __init__(self, model_size="n"):
        self.model = YOLO(f"yolov8{model_size}.pt")

    def detect(self, frame, conf=0.45):
        out = []
        results = self.model(frame, conf=conf, verbose=False)

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id != 0:  # person class in coco
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                out.append({
                    "box": (x1, y1, x2, y2),
                    "confidence": float(box.conf[0])
                })

        return out
