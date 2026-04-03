# YOLO person detector wrapper

import cv2
import numpy as np
from ultralytics import YOLO


class PersonDetector:
    # thin wrapper around ultralytics output format
    
    def __init__(self, model_size='s'):
        # model_size: n, s, m, l, x
        # loads once, then reused every frame
        self.model = YOLO(f'yolov8{model_size}.pt')
        self.last_results = None
        
    def detect(self, frame, conf=0.5):
        # returns list of dicts: box/confidence/center
        # run model on current frame
        results = self.model(frame, conf=conf, verbose=False)
        detections = []
        
        # Extract detections from results
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                # class id 0 => person (coco)
                if cls_id == 0:
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    detections.append({
                        'box': (x1, y1, x2, y2),
                        'confidence': confidence,
                        'center': (center_x, center_y)
                    })
        
        return detections
