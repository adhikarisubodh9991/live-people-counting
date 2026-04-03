# config values live here
from pathlib import Path

class CameraConfig:
    # camera source and line config
    
    def __init__(self):
        self.camera_type = None  # webcam, ip, or video
        self.camera_index = 0  # Webcam index (0 = default camera)
        self.camera_url = None  # URL for IP cameras
        self.video_path = None  # Local test video file
        self.door_line_y = None  # Y coordinate of the door line
        self.door_line_start = None  # Start point of door line (x, y)
        self.door_line_end = None  # End point of door line (x, y)
        self.in_side_sign = 1  # Which side of the line means IN (+1 or -1)
        self.performance_mode = 'accurate'  # fixed: accurate only
        
    def set_webcam(self, index):
        # pick local cam by index
        self.camera_type = 'webcam'
        self.camera_index = index
        self.camera_url = None
        self.video_path = None
        
    def set_ip_camera(self, url):
        # use rtsp/ip stream
        self.camera_type = 'ip'
        self.camera_url = url
        self.video_path = None

    def set_video_file(self, path):
        # use local video file as source for testing
        self.camera_type = 'video'
        self.video_path = path
        self.camera_url = None
        
    def set_door_line(self, y=None, start_point=None, end_point=None):
        # accepts either (start,end) or just y
        if start_point is not None and end_point is not None:
            # full custom line
            self.door_line_start = (int(start_point[0]), int(start_point[1]))
            self.door_line_end = (int(end_point[0]), int(end_point[1]))
            # Calculate midpoint Y
            self.door_line_y = (self.door_line_start[1] + self.door_line_end[1]) // 2
        elif y is not None:
            # fallback horizontal line
            self.door_line_y = int(y)
            self.door_line_start = None
            self.door_line_end = None

    def set_in_side_sign(self, sign):
        # sign should be +1 or -1
        self.in_side_sign = 1 if sign >= 0 else -1

    def set_performance_mode(self, mode):
        # Kept for compatibility with older flow; app is now accurate-only.
        self.performance_mode = 'accurate'


class DetectionConfig:
    # detector tuning params
    
    def __init__(self):
        # Accurate-only profile.
        self.apply_performance_mode('accurate')

    def apply_performance_mode(self, mode):
        base_dir = Path(__file__).resolve().parent

        # Accurate-only profile (mode input ignored intentionally).
        self.confidence_threshold = 0.30
        self.iou_threshold = 0.45
        self.frame_skip = 1
        self.target_fps = 16
        self.min_frame_skip = 1
        self.max_frame_skip = 1
        self.resize_scale = 0.60
        self.model_size = 'n'
        self.imgsz = 640
        self.max_det = 100
        self.use_half = False
        self.performance_mode = 'accurate'
        self.prefer_custom_yolov11 = True
        self.custom_yolov11_repo_path = str((base_dir / 'YOLO').resolve())
        self.custom_yolov11_weights = str((base_dir / 'YOLO' / 'weights' / f'yolov11{self.model_size}.pt').resolve())
