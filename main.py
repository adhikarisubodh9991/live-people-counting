# main app loop for live people counting

import cv2
import time
import logging
import numpy as np
from collections import deque
from config import CameraConfig, DetectionConfig
from detector import PersonDetector
from tracker import PersonTracker
from setup_flow import CameraSetup, DoorLineSetup

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PeopleCounter:
    # setup + detector + tracking + display
    
    def __init__(self):
        self.cam_cfg = None
        self.det_cfg = DetectionConfig()
        self.detector = None
        self.tracker = None
        self.cap = None
        self.running = False
        self._last_dets = []
        self._detect_interval = self.det_cfg.frame_skip
        self._fps_samples = deque(maxlen=30)
        self._fps_smooth = 0.0
        self._low_light_boost = True
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=22, detectShadows=True)
        
    def setup(self):
        # ask user for camera + door line
        # Get camera source from user
        setup_screen = CameraSetup()
        
        while not setup_screen.show_menu():
            pass
        
        # Test that the camera works
        if not setup_screen.test_camera():
            return False
        
        self.cam_cfg = setup_screen.config
        self._detect_interval = self.det_cfg.frame_skip
        
        # Get the door line where people will be counted
        line_setup = DoorLineSetup(self.cam_cfg)
        if not line_setup.show_line_selector():
            logger.error("Door line not configured")
            return False
        
        # Log the configured door line
        if self.cam_cfg.door_line_start and self.cam_cfg.door_line_end:
            logger.info(f"Setup done. Door: {self.cam_cfg.door_line_start} -> {self.cam_cfg.door_line_end}")
        else:
            logger.info(f"Door line Y={self.cam_cfg.door_line_y}")
        logger.info(f"Detection profile: {self.det_cfg.performance_mode} (fixed)")
        return True
    
    def initialize(self):
        # init detector/tracker and open camera
        # Enable OpenCV optimizations for better FPS.
        cv2.setUseOptimized(True)
        try:
            cv2.setNumThreads(0)
        except Exception:
            pass

        # Initialize the person detector
        self.detector = PersonDetector(
            model_size=self.det_cfg.model_size,
            prefer_custom_yolov11=self.det_cfg.prefer_custom_yolov11,
            custom_repo_path=self.det_cfg.custom_yolov11_repo_path,
            custom_weights=self.det_cfg.custom_yolov11_weights,
        )
        logger.info(f"Detector backend: {self.detector.backend_name}")
        if self.detector.backend_detail:
            logger.info(f"Backend detail: {self.detector.backend_detail}")
        if self.detector.fallback_reason:
            logger.warning(f"Custom backend unavailable: {self.detector.fallback_reason}")
        
        # Initialize the person tracker
        self.tracker = PersonTracker(
            door_line_y=self.cam_cfg.door_line_y,
            zone_top=150,
            zone_bottom=150,
            door_line_start=self.cam_cfg.door_line_start,
            door_line_end=self.cam_cfg.door_line_end,
            in_side_sign=self.cam_cfg.in_side_sign
        )
        
        # Open the camera
        if self.cam_cfg.camera_type == 'webcam':
            self.cap = cv2.VideoCapture(self.cam_cfg.camera_index)
        elif self.cam_cfg.camera_type == 'video':
            self.cap = cv2.VideoCapture(self.cam_cfg.video_path)
        else:
            self.cap = cv2.VideoCapture(self.cam_cfg.camera_url)
        
        if not self.cap.isOpened():
            logger.error("Camera failed to open")
            return False
        
        # Optimize for low latency streaming
        self.cap.set(cv2.CAP_PROP_FPS, 20)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Try to keep decode cost lower on local webcams.
        if self.cam_cfg.camera_type == 'webcam':
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        
        return True
    
    
    def run(self):
        # run until ESC / window close
        # Perform initial setup
        if not self.setup():
            return
        if not self.initialize():
            return
        
        self.running = True
        frame_index = 0
        adapt_counter = 0
        prev_time = time.perf_counter()
        
        # Create display window
        cv2.namedWindow('People Counter', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('People Counter', 1000, 650)
        logger.info("Starting... Press ESC to exit")
        
        # Main loop
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Camera lost")
                break
            
            frame_index += 1
            adapt_counter += 1

            run_detector = (frame_index % max(1, self._detect_interval) == 0)
            if run_detector:
                detect_frame = self._prepare_detection_frame(frame)
                # Resize frame for faster detection
                small = cv2.resize(detect_frame, (0, 0), fx=self.det_cfg.resize_scale, fy=self.det_cfg.resize_scale)

                # Detect people in the small frame
                dets_small = self.detector.detect(
                    small,
                    conf=self.det_cfg.confidence_threshold,
                    iou=self.det_cfg.iou_threshold,
                    imgsz=self.det_cfg.imgsz,
                    max_det=self.det_cfg.max_det,
                    use_half=self.det_cfg.use_half
                )

                dets_small = self._filter_detections(small, dets_small)

                # Scale detections back to original frame size
                self._last_dets = self._scale_detections(dets_small)

            dets = self._last_dets
            
            # Track people and check for door crossings
            moves = self.tracker.update(dets)
            tracks = self.tracker.get_display_tracks()
            
            # Log any crossings
            for move in moves:
                logger.info(f"ID {move['id']} went {move['direction']}")
                print(f">>> Person {move['id']} {move['direction'].upper()} <<<")
            
            # Draw detections and door line
            self._draw_overlays(frame, tracks)

            # FPS smoothing (moving average)
            now = time.perf_counter()
            dt = max(1e-6, now - prev_time)
            prev_time = now
            self._fps_samples.append(1.0 / dt)
            self._fps_smooth = sum(self._fps_samples) / len(self._fps_samples)

            # Adaptive detector interval to stay near target FPS (20-24+ smooth range)
            if adapt_counter >= 15:
                if self._fps_smooth < (self.det_cfg.target_fps - 2) and self._detect_interval < self.det_cfg.max_frame_skip:
                    self._detect_interval += 1
                elif self._fps_smooth > (self.det_cfg.target_fps + 3) and self._detect_interval > self.det_cfg.min_frame_skip:
                    self._detect_interval -= 1
                adapt_counter = 0
            
            # Draw statistics
            self._draw_stats(frame, self._fps_smooth)
            
            # Display the frame
            cv2.imshow('People Counter', frame)
            
            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC - exit
                self.running = False
            
            # Check if window was closed
            try:
                if cv2.getWindowProperty('People Counter', cv2.WND_PROP_VISIBLE) < 1:
                    self.running = False
            except:
                pass
        
        self.cleanup()
    
    def _draw_overlays(self, frame, tracks):
        # draw line + boxes
        h, w = frame.shape[:2]

        # Draw the door line
        if self.cam_cfg.door_line_start and self.cam_cfg.door_line_end:
            p1, p2 = self.cam_cfg.door_line_start, self.cam_cfg.door_line_end
        else:
            # Horizontal line if no specific start/end points
            p1 = (0, self.cam_cfg.door_line_y)
            p2 = (w, self.cam_cfg.door_line_y)

        # Draw line in two colors for better visibility
        cv2.line(frame, p1, p2, (0, 255, 0), 4, cv2.LINE_AA)  # Green thick line
        cv2.line(frame, p1, p2, (255, 255, 0), 2, cv2.LINE_AA)  # Yellow thin line
        cv2.circle(frame, p1, 6, (0, 255, 0), -1)
        cv2.circle(frame, p2, 6, (0, 255, 0), -1)

        # Add "DOOR" label at the midpoint
        mx = (p1[0] + p2[0]) // 2
        my = (p1[1] + p2[1]) // 2
        self._draw_text(frame, "DOOR", (max(10, mx - 70), max(30, my - 15)),
                color=(0, 255, 0), scale=0.7, thickness=2)

        # Draw stable tracked boxes (persists through short miss windows)
        for tr in tracks:
            x1, y1, x2, y2 = tr['box']
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            stale = tr.get('stale', False)
            
            # Draw rectangle around person
            color = (0, 220, 0) if not stale else (0, 170, 170)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            
            # Draw center point
            cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
            
            # Draw ID + confidence
            self._draw_text(frame, f"ID {tr['id']} {tr['confidence']:.2f}", (x1, y1 - 10),
                            color=color, scale=0.5, thickness=1)

    def _prepare_detection_frame(self, frame):
        if not self._low_light_boost:
            return frame

        # Boost visibility only when scene is dark to reduce dim-light misses.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_luma = float(gray.mean())
        if mean_luma >= 85.0:
            return frame

        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        y_eq = clahe.apply(y)
        enhanced = cv2.cvtColor(cv2.merge((y_eq, cr, cb)), cv2.COLOR_YCrCb2BGR)

        gamma = 1.18
        table = (255.0 * ((np.arange(256) / 255.0) ** (1.0 / gamma))).astype('uint8')
        enhanced = cv2.LUT(enhanced, table)
        return enhanced

    def _scale_detections(self, detections):
        # map boxes from small frame back to original frame
        scale = 1.0 / self.det_cfg.resize_scale
        scaled = []
        
        for det in detections:
            x1, y1, x2, y2 = det['box']
            # Scale coordinates back up
            x1, y1, x2, y2 = int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale)
            
            scaled.append({
                'box': (x1, y1, x2, y2),
                'confidence': det['confidence'],
                'center': ((x1 + x2) // 2, (y1 + y2) // 2)
            })
        
        return scaled

    def _filter_detections(self, frame, detections):
        # Drop likely false positives from dark noise/shadows while keeping real people.
        h, w = frame.shape[:2]
        keep = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        fgmask = self._bg_subtractor.apply(frame)

        for det in detections:
            x1, y1, x2, y2 = det['box']
            conf = float(det.get('confidence', 0.0))

            x1 = max(0, min(w - 1, int(x1)))
            y1 = max(0, min(h - 1, int(y1)))
            x2 = max(0, min(w, int(x2)))
            y2 = max(0, min(h, int(y2)))
            if x2 <= x1 or y2 <= y1:
                continue

            bw = x2 - x1
            bh = y2 - y1
            area = bw * bh
            aspect = bh / float(max(1, bw))

            # Basic person-like geometry checks.
            if bw < 14 or bh < 24:
                continue
            if area < 520:
                continue
            if aspect < 0.95 or aspect > 5.5:
                continue

            roi = gray[y1:y2, x1:x2]
            roi_hsv = hsv[y1:y2, x1:x2]
            roi_fg = fgmask[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            mean_luma = float(roi.mean())
            std_luma = float(roi.std())
            mean_sat = float(roi_hsv[..., 1].mean()) if roi_hsv.size else 0.0
            lap_var = float(cv2.Laplacian(roi, cv2.CV_64F).var())
            fg_ratio = float((roi_fg == 255).sum()) / float(max(1, roi_fg.size))

            # Dark flat regions tend to create ghost/shadow boxes.
            if mean_luma < 28 and std_luma < 13 and conf < 0.78:
                continue
            if mean_luma < 22 and conf < 0.84:
                continue
            if mean_luma < 42 and mean_sat < 24 and lap_var < 42 and conf < 0.88:
                continue
            if fg_ratio < 0.12 and conf < 0.86:
                continue
            if fg_ratio < 0.08 and conf < 0.92:
                continue

            keep.append({
                'box': (x1, y1, x2, y2),
                'confidence': conf,
                'center': ((x1 + x2) // 2, (y1 + y2) // 2)
            })

        # Second pass: suppress likely shadow duplicates close to a stronger person box.
        keep.sort(key=lambda d: d['confidence'], reverse=True)
        final = []

        for cand in keep:
            cx, cy = cand['center']
            x1, y1, x2, y2 = cand['box']
            c_area = max(1, (x2 - x1) * (y2 - y1))
            blocked = False

            for ref in final:
                rx1, ry1, rx2, ry2 = ref['box']
                r_area = max(1, (rx2 - rx1) * (ry2 - ry1))
                rcx, rcy = ref['center']

                dx = abs(cx - rcx)
                dy = cy - rcy
                overlap_x = max(0, min(x2, rx2) - max(x1, rx1))
                min_w = max(1, min((x2 - x1), (rx2 - rx1)))
                overlap_ratio_x = overlap_x / float(min_w)
                area_ratio = c_area / float(max(1, r_area))

                # Strict shadow duplicate rule: lower-confidence box below/near a strong box.
                if (
                    cand['confidence'] <= ref['confidence'] and
                    dx <= 80 and
                    -20 <= dy <= 135 and
                    overlap_ratio_x >= 0.30 and
                    area_ratio <= 1.15
                ):
                    blocked = True
                    break

            if not blocked:
                final.append(cand)

        return final
    
    def _draw_stats(self, frame, fps):
        # compact v07-style stats panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (330, 170), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

        stats = [
            f"FPS: {fps:.1f}",
            f"IN: {self.tracker.people_in}",
            f"OUT: {self.tracker.people_out}",
            f"ACTIVE: {len(self.tracker.get_active_persons())}",
            f"MODEL: {self.detector.backend_name}",
        ]

        y = 26
        for stat in stats:
            self._draw_text(frame, stat, (10, y), color=(245, 245, 245), scale=0.68, thickness=2)
            y += 30

    def _draw_text(self, frame, text, org, color=(255, 255, 255), scale=0.6, thickness=2, font=cv2.FONT_HERSHEY_SIMPLEX):
        # Clean text with subtle shadow for readability.
        x, y = org
        cv2.putText(frame, text, (x + 1, y + 1), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
    
    def cleanup(self):
        # release cam/window resources before exit
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        logger.info(f"Done - IN: {self.tracker.people_in}, OUT: {self.tracker.people_out}")


def main():
    # app entry
    counter = PeopleCounter()
    counter.run()


if __name__ == "__main__":
    main()
