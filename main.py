# main app loop for live people counting

import cv2
import numpy as np
import time
import logging
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
        return True
    
    def initialize(self):
        # init detector/tracker and open camera
        # Initialize the person detector
        self.detector = PersonDetector(model_size=self.det_cfg.model_size)
        
        # Initialize the person tracker
        self.tracker = PersonTracker(
            door_line_y=self.cam_cfg.door_line_y,
            zone_top=150,
            zone_bottom=150,
            door_line_start=self.cam_cfg.door_line_start,
            door_line_end=self.cam_cfg.door_line_end
        )
        
        # Open the camera
        if self.cam_cfg.camera_type == 'webcam':
            self.cap = cv2.VideoCapture(self.cam_cfg.camera_index)
        else:
            self.cap = cv2.VideoCapture(self.cam_cfg.camera_url)
        
        if not self.cap.isOpened():
            logger.error("Camera failed to open")
            return False
        
        # Optimize for low latency streaming
        self.cap.set(cv2.CAP_PROP_FPS, 20)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        return True
    
    
    def run(self):
        # run until ESC / window close
        # Perform initial setup
        if not self.setup():
            return
        if not self.initialize():
            return
        
        self.running = True
        frame_count = 0
        fps_time = time.time()
        fps = 0
        
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
            
            frame_count += 1
            
            # Resize frame for faster detection
            small = cv2.resize(frame, (0, 0), fx=self.det_cfg.resize_scale, fy=self.det_cfg.resize_scale)
            
            # Detect people in the small frame
            dets_small = self.detector.detect(small, conf=self.det_cfg.confidence_threshold)
            
            # Scale detections back to original frame size
            dets = self._scale_detections(dets_small)
            
            # Track people and check for door crossings
            moves = self.tracker.update(dets)
            
            # Log any crossings
            for move in moves:
                logger.info(f"ID {move['id']} went {move['direction']}")
                print(f">>> Person {move['id']} {move['direction'].upper()} <<<")
            
            # Draw detections and door line
            self._draw_overlays(frame, dets)
            
            # FPS calculation (update every second)
            if time.time() - fps_time >= 1.0:
                fps = frame_count
                frame_count = 0
                fps_time = time.time()
            
            # Draw statistics
            self._draw_stats(frame, fps)
            
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
    
    def _draw_overlays(self, frame, detections):
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
        cv2.line(frame, p1, p2, (0, 255, 0), 4)  # Green thick line
        cv2.line(frame, p1, p2, (255, 255, 0), 2)  # Yellow thin line
        cv2.circle(frame, p1, 6, (0, 255, 0), -1)
        cv2.circle(frame, p2, 6, (0, 255, 0), -1)

        # Add "DOOR" label at the midpoint
        mx = (p1[0] + p2[0]) // 2
        my = (p1[1] + p2[1]) // 2
        cv2.putText(frame, "DOOR", (max(10, mx - 70), max(30, my - 15)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw bounding boxes for detections
        for det in detections:
            x1, y1, x2, y2 = det['box']
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            # Draw rectangle around person
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw center point
            cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
            
            # Draw confidence score
            cv2.putText(frame, f"{det['confidence']:.2f}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

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
    
    def _draw_stats(self, frame, fps):
        # simple stats panel
        h, w = frame.shape[:2]
        
        # Create semi-transparent overlay for stats panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (300, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Build stats list
        stats = [
            f"FPS: {fps}",
            f"IN:  {self.tracker.people_in}",
            f"OUT: {self.tracker.people_out}",
            f"Active: {len(self.tracker.get_active_persons())}"
        ]
        
        # Draw each stat on the frame
        y = 25
        for stat in stats:
            cv2.putText(frame, stat, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y += 35
    
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
