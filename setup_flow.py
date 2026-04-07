# setup steps: camera pick and line drawing

import cv2
from pathlib import Path
from config import CameraConfig


class CameraSetup:
    def __init__(self):
        self.config = CameraConfig()
        self.video_files = self._find_local_videos()

    def _find_local_videos(self):
        # Find local videos near this script (final_project folder)
        base = Path(__file__).resolve().parent
        exts = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        return sorted([p for p in base.iterdir() if p.is_file() and p.suffix.lower() in exts])

    def show_menu(self):
        print("\n" + "=" * 50)
        print("  LIVE PEOPLE COUNTER - SETUP")
        print("=" * 50)
        print("\nSelect Camera Source:")
        print("-" * 50)

        webcam_choice = 1
        ip_choice = 2
        video_choice = 3
        print(f"  {webcam_choice}. Webcam (choose index)")
        print(f"  {ip_choice}. IP Camera (RTSP)")
        print(f"  {video_choice}. Local Video File")
        print("-" * 50)

        raw = input(f"\nEnter selection (1-{video_choice}): ")

        try:
            sel = int(raw)
            if sel == webcam_choice:
                idx_raw = input("Enter webcam index (default 0): ").strip()
                try:
                    picked = int(idx_raw) if idx_raw else 0
                except ValueError:
                    print("Invalid webcam index!")
                    return False
                self.config.set_webcam(picked)
                print("✓ Detection profile: Accurate (fixed)")
                return True

            if sel == ip_choice:
                rtsp = input("Enter RTSP URL (e.g. rtsp://user:pass@192.168.1.100:554/stream): ")
                if rtsp.strip():
                    self.config.set_ip_camera(rtsp)
                    print("✓ Detection profile: Accurate (fixed)")
                    return True

            if sel == video_choice:
                if not self.video_files:
                    print("No local videos found in final_project folder.")
                    return False

                print("\nAvailable videos:")
                for idx, vid in enumerate(self.video_files, 1):
                    print(f"  {idx}. {vid.name}")

                pick = input(f"Choose video (1-{len(self.video_files)}): ").strip()
                try:
                    pick_idx = int(pick)
                    if 1 <= pick_idx <= len(self.video_files):
                        chosen = str(self.video_files[pick_idx - 1])
                        self.config.set_video_file(chosen)
                        print(f"✓ Using video: {self.video_files[pick_idx - 1].name}")
                        print("✓ Detection profile: Accurate (fixed)")
                        return True
                except ValueError:
                    pass

                print("Invalid video selection!")
                return False
        except ValueError:
            pass

        print("Invalid selection!")
        return False

    def test_camera(self):
        print("\nTesting camera connection...")

        # quick open test. could be improved later with retry.
        if self.config.camera_type == "webcam":
            cap = cv2.VideoCapture(self.config.camera_index)
        elif self.config.camera_type == "video":
            cap = cv2.VideoCapture(self.config.video_path)
        else:
            cap = cv2.VideoCapture(self.config.camera_url)

        if self.config.camera_type == 'webcam':
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

        if not cap.isOpened():
            print("Failed to open camera!")
            return False

        ok, _frame = cap.read()
        cap.release()

        if ok:
            print("✓ Camera connected successfully")
            return True

        print("✗ Could not read from camera")
        return False


class DoorLineSetup:
    def __init__(self, config):
        self.config = config
        self.cap = None

    def open_camera(self):
        if self.config.camera_type == 'webcam':
            self.cap = cv2.VideoCapture(self.config.camera_index)
        elif self.config.camera_type == 'video':
            self.cap = cv2.VideoCapture(self.config.video_path)
        else:
            self.cap = cv2.VideoCapture(self.config.camera_url)

        if self.config.camera_type == 'webcam':
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

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
        print("  2. Press SPACE to lock the line")
        print("  3. Click on the side that should count as IN")
        print("  4. Press SPACE again to confirm")
        print("  5. Press ESC to cancel/redraw")
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
        selecting_in_side = False
        in_side_sign = None
        in_click = None

        def point_side(line_p1, line_p2, x, y):
            x1, y1 = line_p1
            x2, y2 = line_p2
            return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)

        def on_mouse(event, x, y, flags, param):
            nonlocal drawing, p1, p2, selecting_in_side, in_side_sign, in_click

            if selecting_in_side and event == cv2.EVENT_LBUTTONDOWN and p1 and p2:
                side_val = point_side(p1, p2, int(x), int(y))
                in_side_sign = 1 if side_val >= 0 else -1
                in_click = (int(x), int(y))
                print(f"✓ IN side selected at {in_click}")
                return

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
                # Local videos naturally end; rewind so user can keep calibrating.
                if self.config.camera_type == 'video':
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                    if not ret:
                        print("Could not read from local video. Please choose another file.")
                        cv2.destroyAllWindows()
                        return False
                else:
                    print("Lost camera connection!")
                    cv2.destroyAllWindows()
                    return False

            if selecting_in_side:
                cv2.putText(frame, "CLICK side that means IN | SPACE: confirm | ESC: redraw",
                            (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 220), 2)
            else:
                cv2.putText(frame, "DRAG: draw line | SPACE: lock line | ESC: cancel",
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

            if in_click is not None:
                cv2.circle(frame, in_click, 10, (255, 255, 0), -1)
                cv2.putText(frame, "IN", (in_click[0] + 12, in_click[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            cv2.imshow('Door Line Selection', frame)

            key = cv2.waitKey(30) & 0xFF

            if key == 32:  # SPACE
                if selecting_in_side:
                    if in_side_sign is None:
                        print("Click the IN side first!")
                        continue

                    confirmed = (p1, p2)
                    center_y = (p1[1] + p2[1]) // 2
                    fh, fw = frame.shape[:2]
                    self.config.set_door_line(start_point=p1, end_point=p2, frame_size=(fw, fh))
                    self.config.set_in_side_sign(in_side_sign)
                    print(f"[debug] line: {p1} -> {p2}")
                    print(f"[debug] line reference size: {fw}x{fh}")
                    print(f"[debug] in_side_sign: {in_side_sign}")
                    print(f"✓ Door line CONFIRMED (center Y={center_y})")
                    cv2.destroyAllWindows()
                    return True

                if p1 and p2 and p1 != p2:
                    confirmed = (p1, p2)
                    selecting_in_side = True
                    in_side_sign = None
                    in_click = None
                    print("Line locked. Now click the side that should count as IN, then press SPACE.")
                else:
                    print("Draw a line first!")

            elif key == 27:  # ESC
                if selecting_in_side or confirmed:
                    # reset and redraw
                    confirmed = None
                    p1 = None
                    p2 = None
                    selecting_in_side = False
                    in_side_sign = None
                    in_click = None
                    print("Line cleared. Draw again...")
                else:
                    print("Cancelled.")
                    cv2.destroyAllWindows()
                    return False

        cv2.destroyAllWindows()
        return False
