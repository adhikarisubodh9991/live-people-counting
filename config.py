# config values live here

class CameraConfig:
    # camera source and line config
    
    def __init__(self):
        self.camera_type = None  # webcam or ip
        self.camera_index = 0  # Webcam index (0 = default camera)
        self.camera_url = None  # URL for IP cameras
        self.door_line_y = None  # Y coordinate of the door line
        self.door_line_start = None  # Start point of door line (x, y)
        self.door_line_end = None  # End point of door line (x, y)
        
    def set_webcam(self, index):
        # pick local cam by index
        self.camera_type = 'webcam'
        self.camera_index = index
        
    def set_ip_camera(self, url):
        # use rtsp/ip stream
        self.camera_type = 'ip'
        self.camera_url = url
        
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


class DetectionConfig:
    # detector tuning params
    
    def __init__(self):
        self.confidence_threshold = 0.45  # Minimum confidence for detection
        self.frame_skip = 1  # Process every frame (1 = process all frames)
        self.target_fps = 30  # Target frames per second
        self.resize_scale = 0.5  # Resize frames to 50% for faster processing
        self.model_size = 'n'  # 'n' = nano (fastest), 's' = small, 'm' = medium, etc.
