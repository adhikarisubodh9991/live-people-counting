# setup flow

import cv2
from config import CameraConfig


class CameraSetup:
    def __init__(self):
        self.config = CameraConfig()
        self.cam_list = self._find_webcams()

    def _find_webcams(self):
        cams = []
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cams.append(i)
                cap.release()
        return cams

    def show_menu(self):
        print("\n" + "=" * 50)
        print("  LIVE PEOPLE COUNTER - SETUP")
        print("=" * 50)
        print("\nSelect Camera Source:")
        print("-" * 50)

        for idx, cam_i in enumerate(self.cam_list, 1):
            print(f"  {idx}. Webcam {cam_i}")

        ip_choice = len(self.cam_list) + 1
        print(f"  {ip_choice}. IP Camera (RTSP)")
        print("-" * 50)

        raw = input(f"\nEnter selection (1-{ip_choice}): ")

        try:
            sel = int(raw)
            if 1 <= sel <= len(self.cam_list):
                self.config.set_webcam(self.cam_list[sel - 1])
                return True

            if sel == ip_choice:
                rtsp = input("Enter RTSP URL: ")
                if rtsp.strip():
                    self.config.set_ip_camera(rtsp)
                    return True
        except ValueError:
            pass

        print("Invalid selection")
        return False

    def test_camera(self):
        if self.config.camera_type == "webcam":
            cap = cv2.VideoCapture(self.config.camera_index)
        else:
            cap = cv2.VideoCapture(self.config.camera_url)

        if not cap.isOpened():
            print("Failed to open camera")
            return False

        ok, _ = cap.read()
        cap.release()
        return ok


class DoorLineSetup:
    def __init__(self, config):
        self.config = config
        self.cap = None

    def open_camera(self):
        if self.config.camera_type == "webcam":
            self.cap = cv2.VideoCapture(self.config.camera_index)
        else:
            self.cap = cv2.VideoCapture(self.config.camera_url)

        self.cap.set(cv2.CAP_PROP_FPS, 20)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return self.cap.isOpened()

    def show_line_selector(self):
        if not self.open_camera():
            print("Failed to open camera")
            return False

        cv2.namedWindow("Door Line Selection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Door Line Selection", 1200, 700)

        drawing = False
        p1 = None
        p2 = None

        def on_mouse(event, x, y, flags, param):
            nonlocal drawing, p1, p2
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                p1 = (x, y)
                p2 = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE and drawing:
                p2 = (x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False

        cv2.setMouseCallback("Door Line Selection", on_mouse)

        while True:
            ok, frame = self.cap.read()
            if not ok:
                cv2.destroyAllWindows()
                return False

            cv2.putText(frame, "DRAG: draw line | SPACE: confirm | ESC: cancel",
                        (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)

            if p1 and p2:
                cv2.line(frame, p1, p2, (0, 255, 0), 3)

            cv2.imshow("Door Line Selection", frame)
            key = cv2.waitKey(30) & 0xFF

            if key == 32:
                if p1 and p2 and p1 != p2:
                    self.config.set_door_line(start_point=p1, end_point=p2)
                    cv2.destroyAllWindows()
                    return True
                print("draw a line first")

            elif key == 27:
                cv2.destroyAllWindows()
                return False
