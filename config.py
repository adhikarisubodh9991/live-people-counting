# config values


class CameraConfig:
    def __init__(self):
        self.camera_type = None
        self.camera_index = 0
        self.camera_url = None
        self.door_line_y = None
        self.door_line_start = None
        self.door_line_end = None

    def set_webcam(self, index):
        self.camera_type = "webcam"
        self.camera_index = index

    def set_ip_camera(self, url):
        self.camera_type = "ip"
        self.camera_url = url

    def set_door_line(self, y=None, start_point=None, end_point=None):
        if start_point is not None and end_point is not None:
            self.door_line_start = (int(start_point[0]), int(start_point[1]))
            self.door_line_end = (int(end_point[0]), int(end_point[1]))
            self.door_line_y = (self.door_line_start[1] + self.door_line_end[1]) // 2
        elif y is not None:
            self.door_line_y = int(y)
            self.door_line_start = None
            self.door_line_end = None


class DetectionConfig:
    def __init__(self):
        self.confidence_threshold = 0.45
        self.resize_scale = 0.5
        self.model_size = "n"
        self.frame_skip = 2
        self.target_fps = 20
