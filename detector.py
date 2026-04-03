# YOLO person detector wrapper

import cv2
import numpy as np
import os
from pathlib import Path
import sys
from ultralytics import YOLO

try:
    import torch
except Exception:
    torch = None


class PersonDetector:
    # thin wrapper around ultralytics output format
    
    def __init__(self, model_size='n', prefer_custom_yolov11=True, custom_repo_path=None, custom_weights=None):
        # model_size: n, s, m, l, x
        # loads once, then reused every frame
        self.model = None
        self.custom_predictor = None
        self.last_results = None
        self.device = 'cpu'
        self.backend_name = 'ultralytics-yolo11'
        self.backend_detail = ''
        self.fallback_reason = None

        if torch is not None and torch.cuda.is_available():
            self.device = 'cuda:0'

        if prefer_custom_yolov11:
            ok, reason = self._try_init_custom_yolov11(model_size, custom_repo_path, custom_weights)
            if not ok:
                self.fallback_reason = reason
        else:
            self.fallback_reason = 'custom YOLOv11 backend disabled in config'

        if self.custom_predictor is None:
            # fallback path: use official Ultralytics YOLO11 weights
            self.model = YOLO(f'yolo11{model_size}.pt')
            self.model.to(self.device)
            self.backend_detail = f'fallback model=yolo11{model_size}.pt device={self.device}'

            # Fuse layers when available for slightly faster inference.
            try:
                self.model.fuse()
            except Exception:
                pass

    def _try_init_custom_yolov11(self, model_size, custom_repo_path, custom_weights):
        # Optional integration with B4rtekk1/YOLO if repo + weights are available locally.
        repo_hint = custom_repo_path or os.getenv('YOLOV11_CUSTOM_REPO')
        candidates = []
        if repo_hint:
            candidates.append(Path(repo_hint))
        script_dir = Path(__file__).resolve().parent
        candidates.extend([
            script_dir / 'YOLO',
            script_dir.parent / 'YOLO',
            script_dir.parent / 'B4rtekk1-YOLO',
        ])

        repo_path = None
        for c in candidates:
            if (c / 'inference.py').exists():
                repo_path = c
                break

        if repo_path is None:
            return False, 'custom repo not found (set custom_yolov11_repo_path or YOLOV11_CUSTOM_REPO)'

        try:
            if str(repo_path) not in sys.path:
                sys.path.insert(0, str(repo_path))
            from inference import YOLOv11Predictor  # type: ignore
        except Exception as exc:
            return False, f'failed importing YOLOv11Predictor from {repo_path}: {exc}'

        weights_path = custom_weights
        if not weights_path:
            default_weights = repo_path / 'weights' / f'yolov11{model_size}.pt'
            if default_weights.exists():
                weights_path = str(default_weights)

        if not weights_path or not Path(weights_path).exists():
            return False, f'custom weights not found: {weights_path}'

        try:
            device = '0' if self.device.startswith('cuda') else 'cpu'
            self.custom_predictor = YOLOv11Predictor(
                weights=weights_path,
                task='detect',
                device=device,
                conf_thres=0.25,
                iou_thres=0.45,
                img_size=640,
            )
            self.backend_name = 'yolov11-custom'
            self.backend_detail = f'custom repo={repo_path} weights={weights_path} device={device}'
            return True, None
        except Exception as exc:
            self.custom_predictor = None
            return False, f'YOLOv11Predictor init failed: {exc}'
        
    def detect(self, frame, conf=0.5, iou=0.45, imgsz=640, max_det=100, use_half=False):
        # returns list of dicts: box/confidence/center
        # run model on current frame
        if self.custom_predictor is not None:
            return self._detect_with_custom_predictor(frame, conf, iou)

        half = bool(use_half and self.device.startswith('cuda'))
        results = self.model(
            frame,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            max_det=max_det,
            classes=[0],  # person class only
            half=half,
            verbose=False
        )
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

    def _detect_with_custom_predictor(self, frame, conf, iou):
        try:
            self.custom_predictor.conf_thres = conf
            self.custom_predictor.iou_thres = iou
            out = self.custom_predictor.predict(frame)
        except Exception:
            return []

        boxes = out.get('boxes', np.zeros((0, 4), dtype=np.float32))
        scores = out.get('scores', np.zeros(0, dtype=np.float32))
        labels = out.get('labels', np.zeros(0, dtype=np.int32))

        detections = []
        for i in range(len(boxes)):
            cls_id = int(labels[i]) if i < len(labels) else -1
            if cls_id != 0:
                continue

            x1, y1, x2, y2 = map(int, boxes[i])
            conf_val = float(scores[i]) if i < len(scores) else 0.0
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            detections.append({
                'box': (x1, y1, x2, y2),
                'confidence': conf_val,
                'center': (center_x, center_y)
            })

        return detections
