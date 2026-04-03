# setup steps: camera pick and line drawing

import cv2
from config import CameraConfig


class CameraSetup:
    def __init__(self):
        self.config = CameraConfig()
        self.cam_list = self._find_webcams()

    def _find_webcams(self):
        # quick scan of common cam indexes
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
                picked = self.cam_list[sel - 1]
                self.config.set_webcam(picked)
                return True

            if sel == ip_choice:
                rtsp = input("Enter RTSP URL (e.g. rtsp://user:pass@192.168.1.100:554/stream): ")
                if rtsp.strip():
                    self.config.set_ip_camera(rtsp)
                    return True
        except ValueError:
            pass

        print("Invalid selection!")
        return False

    def test_camera(self):
        print("\nTesting camera connection...")

        # quick open test. could be improved later with retry.
        if self.config.camera_type == "webcam":
            cap = cv2.VideoCapture(self.config.camera_index)
        else:
            cap = cv2.VideoCapture(self.config.camera_url)

        if not cap.isOpened():
            print("Failed to open camera!")
            return False

        ok, _frame = cap.read()
        cap.release()

        if ok:
            print(" Camera connected successfully")
            return True

        print(" Could not read from camera")
        return False


class DoorLineSetup:
    def __init__(self, config):
        self.config = config
        self.cap = None

    def open_camera(self):
        if self.config.camera_type == 'webcam':
            self.cap = cv2.VideoCapture(self.config.camera_index)
        else:
            self.cap = cv2.VideoCapture(self.config.camera_url)

        # keep stream low-latency
        self.cap.set(cv2.CAP_PROP_FPS, 20)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        return self.cap.isOpened()

    def show_line_selector(self):
        print("\n" + "="*60)
        print("  SELECT DOOR LINE")
        print("="*60)
        print("\nHow to select the door line:")
        print("  1. Click and DRAG to draw a line where people cross")
        print("  2. Press SPACE to confirm")
        print("  3. Press ESC to cancel/redraw")
        print("="*60 + "\n")

        if not self.open_camera():
            print("Failed to open camera!")
            return False

        cv2.namedWindow('Door Line Selection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Door Line Selection', 1200, 700)

        drawing = False
        p1 = None
        p2 = None
        confirmed = None

        def on_mouse(event, x, y, flags, param):
            nonlocal drawing, p1, p2

            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                p1 = (int(x), int(y))
                p2 = (int(x), int(y))
            elif event == cv2.EVENT_MOUSEMOVE and drawing:
                p2 = (int(x), int(y))
            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False

        cv2.setMouseCallback('Door Line Selection', on_mouse)

        print("Draw the line with mouse drag...\n")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Lost camera connection!")
                cv2.destroyAllWindows()
                return False

            cv2.putText(frame, "DRAG: draw line | SPACE: confirm | ESC: cancel",
                        (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)

            # preview while drawing
            if p1 and p2:
                cv2.line(frame, p1, p2, (0, 255, 0), 3)
                cv2.circle(frame, p1, 8, (255, 0, 0), -1)
                cv2.circle(frame, p2, 8, (255, 0, 0), -1)

            if confirmed:
                c1, c2 = confirmed
                cx = (c1[0] + c2[0]) // 2
                cy = (c1[1] + c2[1]) // 2

                cv2.line(frame, c1, c2, (0, 255, 255), 4)
                cv2.circle(frame, c1, 8, (0, 255, 255), -1)
                cv2.circle(frame, c2, 8, (0, 255, 255), -1)
                cv2.circle(frame, (cx, cy), 10, (0, 255, 255), -1)

                cv2.putText(frame, "CONFIRMED - This is your door line!",
                            (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

            cv2.imshow('Door Line Selection', frame)

            key = cv2.waitKey(30) & 0xFF

            if key == 32:  # SPACE
                if p1 and p2 and p1 != p2:
                    confirmed = (p1, p2)
                    center_y = (p1[1] + p2[1]) // 2
                    self.config.set_door_line(start_point=p1, end_point=p2)
                    print(f"[debug] line: {p1} -> {p2}")
                    # print("door line saved")
                    print(f" Door line CONFIRMED (center Y={center_y})")
                    cv2.destroyAllWindows()
                    return True
                else:
                    print("Draw a line first!")

            elif key == 27:  # ESC
                if confirmed:
                    # reset and redraw
                    confirmed = None
                    p1 = None
                    p2 = None
                    print("Line cleared. Draw again...")
                else:
                    print("Cancelled.")
                    cv2.destroyAllWindows()
                    return False

        cv2.destroyAllWindows()
        return False
