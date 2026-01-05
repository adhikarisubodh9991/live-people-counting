import numpy as np
import cv2 as cv
import time
from ultralytics import YOLO
import Person

# Load the YOLOv8 model
model = YOLO('yolov8x.pt')

# Log setup
try:
    log = open('log.txt', "w")
except IOError:
    print("Cannot open the log file")

cnt_up = 0
cnt_down = 0
total_inside = 0  # Tracks the total people inside the room

# Video capture
cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open the camera")
    exit()

# Frame dimensions
h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frameArea = h * w
areaTH = frameArea / 250
print('Area Threshold:', areaTH)

# Tracking and detection parameters
line_up = int(2 * (h / 5))
line_down = int(3 * (h / 5))
up_limit = int(1 * (h / 5))
down_limit = int(4 * (h / 5))

line_down_color = (255, 0, 0)
line_up_color = (0, 0, 255)

# Polylines
pts_L1 = np.array([[0, line_down], [w, line_down]], np.int32).reshape((-1, 1, 2))
pts_L2 = np.array([[0, line_up], [w, line_up]], np.int32).reshape((-1, 1, 2))
pts_L3 = np.array([[0, up_limit], [w, up_limit]], np.int32).reshape((-1, 1, 2))
pts_L4 = np.array([[0, down_limit], [w, down_limit]], np.int32).reshape((-1, 1, 2))

font = cv.FONT_HERSHEY_SIMPLEX
persons = []
max_p_age = 5
pid = 1

# Resizable display
cv.namedWindow("Frame", cv.WINDOW_NORMAL)

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Could not capture video")
        break

    # YOLO detection
    results = model(frame, conf=0.5)  # Confidence threshold
    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            confidence = box.conf[0]

            if class_id == 0 and confidence > 0.5:  # Person class ID
                detections.append((x1, y1, x2, y2, confidence))

    # Update tracking
    for x1, y1, x2, y2, conf in detections:
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        if cy in range(up_limit, down_limit):
            new = True
            for i in persons:
                if abs(cx - i.getX()) <= (x2 - x1) and abs(cy - i.getY()) <= (y2 - y1):
                    new = False
                    i.updateCoords(cx, cy)
                    if i.going_UP(line_down, line_up):
                        cnt_up += 1
                        if total_inside > 0:
                            total_inside -= 1
                        log.write(f"ID: {i.getId()} crossed going up at {time.strftime('%c')}. Total inside: {total_inside}\n")
                    elif i.going_DOWN(line_down, line_up):
                        cnt_down += 1
                        total_inside += 1
                        log.write(f"ID: {i.getId()} crossed going down at {time.strftime('%c')}. Total inside: {total_inside}\n")
                    break
            if new:
                p = Person.MyPerson(pid, cx, cy, max_p_age)
                persons.append(p)
                pid += 1

        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv.putText(frame, f"ID:{pid} {conf:.2f}", (x1, y1 - 10), font, 0.5, (0, 255, 0), 1, cv.LINE_AA)
        cv.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    # Polylines
    frame = cv.polylines(frame, [pts_L1], False, line_down_color, thickness=2)
    frame = cv.polylines(frame, [pts_L2], False, line_up_color, thickness=2)
    frame = cv.polylines(frame, [pts_L3], False, (255, 255, 255), thickness=1)
    frame = cv.polylines(frame, [pts_L4], False, (255, 255, 255), thickness=1)

    # Display counts
    cv.putText(frame, f'Out: {cnt_up}', (10, 40), font, 0.5, (0,0,0), 2, cv.LINE_AA)
    cv.putText(frame, f'In: {cnt_down}', (10, 90), font, 0.5, (0,0,0), 2, cv.LINE_AA)
    cv.putText(frame, f'Total Inside: {total_inside}', (10, 140), font, 0.5, (0,0,0), 2, cv.LINE_AA)

    # Show frame
    cv.imshow('Frame', frame)

    if cv.waitKey(30) & 0xff == 27:  # Press 'Esc' to exit
        break

# Cleanup
log.flush()
log.close()
cap.release()
cv.destroyAllWindows()
