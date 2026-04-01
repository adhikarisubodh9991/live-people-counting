#just camera preview from webcam

import cv2


def main():
    cam_index_raw = input("Webcam index (default 0): ").strip()
    cam_index = int(cam_index_raw) if cam_index_raw else 0

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print("could not open camera")
        return

    print("camera opened. press ESC to quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("camera read failed")
            break

        cv2.imshow("People Counter - Base", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
