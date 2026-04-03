# basic multi-person tracker and line crossing counter

import time
from collections import deque


class Person:
    # per-person tracking state
    
    def __init__(self, person_id, x, y, box=None):
        # tracked person state
        self.id = person_id
        self.x = x
        self.y = y
        self.prev_x = x  # Previous frame position for movement detection
        self.prev_y = y
        self.positions = [(x, y)]  # History of all positions
        self.counted = False  # Whether they've crossed the door line
        self.direction = None  # 'in' or 'out' - which way did they go?
        self.last_seen = time.time()
        self.frames_without_detection = 0  # How many frames since we last saw them
        self.hits = 1  # number of matched frames
        self.confirmed = False
        self.vx = 0.0
        self.vy = 0.0
        self.width = 0
        self.height = 0
        self.last_box = None
        self.last_confidence = 0.0
        self.side_history = deque(maxlen=8)
        self.last_side_state = 'unknown'
        self.stable_side = 'unknown'
        self.stable_side_frames = 0
        self.cross_cooldown_until = 0.0
        self.armed_side = None
        if box is not None:
            self.width = max(1, int(box[2] - box[0]))
            self.height = max(1, int(box[3] - box[1]))
            self.last_box = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
        
    def update(self, x, y, box=None, confidence=0.0):
        # save last point before moving
        self.prev_x = self.x
        self.prev_y = self.y

        # Use EMA to reduce jitter and prevent false line-cross flips.
        alpha = 0.65
        self.x = int(alpha * x + (1.0 - alpha) * self.x)
        self.y = int(alpha * y + (1.0 - alpha) * self.y)
        self.vx = float(self.x - self.prev_x)
        self.vy = float(self.y - self.prev_y)

        if box is not None:
            bw = max(1, int(box[2] - box[0]))
            bh = max(1, int(box[3] - box[1]))
            self.width = int(alpha * bw + (1.0 - alpha) * max(1, self.width or bw))
            self.height = int(alpha * bh + (1.0 - alpha) * max(1, self.height or bh))

        self.positions.append((x, y))
        self.frames_without_detection = 0
        self.last_seen = time.time()
        self.hits += 1
        self.last_confidence = float(confidence)

        if self.width > 0 and self.height > 0:
            x1 = int(self.x - self.width // 2)
            y1 = int(self.y - self.height // 2)
            x2 = int(self.x + self.width // 2)
            y2 = int(self.y + self.height // 2)
            self.last_box = (x1, y1, x2, y2)
        
    def age(self):
        # not seen in this frame
        self.frames_without_detection += 1
        # Keep a short prediction trail so the green box does not vanish instantly.
        self.x = int(self.x + self.vx * 0.35)
        self.y = int(self.y + self.vy * 0.35)
        self.vx *= 0.75
        self.vy *= 0.75

        if self.width > 0 and self.height > 0:
            x1 = int(self.x - self.width // 2)
            y1 = int(self.y - self.height // 2)
            x2 = int(self.x + self.width // 2)
            y2 = int(self.y + self.height // 2)
            self.last_box = (x1, y1, x2, y2)
        
    def is_old(self, max_age=10):
        # stale track check
        return self.frames_without_detection > max_age


class PersonTracker:
    # matches detections to tracks + counts in/out
    
    def __init__(self, door_line_y, zone_top=150, zone_bottom=150, door_line_start=None, door_line_end=None, in_side_sign=1):
        # zone_top / zone_bottom kept for compatibility
        self.door_line_y = door_line_y
        self.door_line_start = door_line_start
        self.door_line_end = door_line_end
        self.in_side_sign = 1 if in_side_sign >= 0 else -1

        # If no specific line points provided, assume a horizontal line
        if self.door_line_start is None or self.door_line_end is None:
            self.door_line_start = (0, int(door_line_y))
            self.door_line_end = (1, int(door_line_y))

        self.zone_top = door_line_y - zone_top
        self.zone_bottom = door_line_y + zone_bottom
        self.persons = {}  # Dictionary of Person objects, keyed by ID
        self.next_id = 1
        self.people_in = 0  # Total count of people entering
        self.people_out = 0  # Total count of people exiting
        self.match_max_distance = 120
        self.min_confirm_hits = 3
        self.display_hold_frames = 10
        self.min_match_iou = 0.18
        self.crossing_threshold = 8
        self.min_cross_travel_px = 10
        self.min_stable_side_frames = 1
        self.crossing_cooldown_sec = 0.55

    def _box_iou(self, b1, b2):
        if b1 is None or b2 is None:
            return 0.0

        ax1, ay1, ax2, ay2 = b1
        bx1, by1, bx2, by2 = b2

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0

        a = max(1, (ax2 - ax1) * (ay2 - ay1))
        b = max(1, (bx2 - bx1) * (by2 - by1))
        return inter / float(a + b - inter)

    def _point_side(self, x, y):
        # cross-product sign tells which side of the line this point is on
        x1, y1 = self.door_line_start
        x2, y2 = self.door_line_end
        return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)

    def _person_ref_point(self, person):
        # Use bottom-center of box (feet area) for faster and more stable crossing events.
        if person.last_box is not None:
            x1, y1, x2, y2 = person.last_box
            return ((x1 + x2) // 2, y2)
        return (person.x, person.y)
        
    def update(self, detections):
        # one frame update: match, count crossings, cleanup
        movements = []
        matched_ids = set()  # avoid double-match in same frame
        
        # Try to match each detection with an existing tracked person
        for det in detections:
            cx, cy = det['center']
            
            best_match = None
            best_distance = float('inf')
            
            det_box = det.get('box')

            # Find the closest tracked person to this detection
            for pid, person in self.persons.items():
                if pid in matched_ids:
                    # Already matched this person in this frame
                    continue

                # Predict short-term motion to avoid ID swap on fast movement.
                px = person.x + int(person.vx * min(2, person.frames_without_detection + 1))
                py = person.y + int(person.vy * min(2, person.frames_without_detection + 1))
                dist = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5
                iou = self._box_iou(person.last_box, det_box)

                gate = self.match_max_distance + 25 * person.frames_without_detection
                if person.height > 0 and det_box is not None:
                    det_h = max(1, int(det_box[3] - det_box[1]))
                    det_w = max(1, int(det_box[2] - det_box[0]))
                    size_delta = abs(det_h - person.height)
                    width_delta = abs(det_w - max(1, person.width))
                    # Reject impossible scale jumps to reduce duplicate IDs.
                    if iou < 0.05 and (size_delta > max(120, int(person.height * 0.9)) or width_delta > max(140, int(max(1, person.width) * 1.1))):
                        continue

                # If overlap is decent, prefer this match even with arm swing / shape change.
                if iou >= self.min_match_iou:
                    overlap_score = dist * 0.35
                    if overlap_score < best_distance:
                        best_match = pid
                        best_distance = overlap_score
                    continue

                if dist < gate and dist < best_distance:
                    best_match = pid
                    best_distance = dist
            
            if best_match is not None:
                # Update existing tracked person
                person = self.persons[best_match]
                person.update(cx, cy, box=det_box, confidence=det.get('confidence', 0.0))
                if person.hits >= self.min_confirm_hits:
                    person.confirmed = True
                matched_ids.add(best_match)
            else:
                # Suppress likely shadow duplicates near an already tracked person.
                if det_box is not None:
                    is_shadow_like = False
                    det_h = max(1, int(det_box[3] - det_box[1]))
                    det_w = max(1, int(det_box[2] - det_box[0]))
                    det_area = det_h * det_w
                    det_conf = float(det.get('confidence', 0.0))
                    for person in self.persons.values():
                        if not person.confirmed:
                            continue
                        if person.last_box is None:
                            continue
                        px = person.x
                        py = person.y
                        if abs(px - cx) > 110 or abs(py - cy) > 120:
                            continue
                        p_area = max(1, person.width * person.height)
                        iou = self._box_iou(person.last_box, det_box)
                        mostly_below = cy >= (py - 10)
                        area_ratio = det_area / float(p_area)

                        # Stricter rejection for shadow-like duplicate near same person.
                        if mostly_below and det_conf < 0.68 and (
                            area_ratio < 0.9 or iou > 0.12 or (det_h < int(max(1, person.height) * 0.9))
                        ):
                            is_shadow_like = True
                            break

                    if is_shadow_like:
                        continue

                # Create a new track for this person
                person = Person(self.next_id, cx, cy, box=det_box)
                person.last_confidence = float(det.get('confidence', 0.0))
                self.persons[self.next_id] = person
                matched_ids.add(self.next_id)
                self.next_id += 1
        
        # Check if any tracked people crossed the door line with side-stability checks.
        now = time.time()

        for pid, person in self.persons.items():
            if not person.confirmed:
                continue

            rx, ry = self._person_ref_point(person)
            side_val = self._point_side(rx, ry) * self.in_side_sign
            if side_val >= self.crossing_threshold:
                side_state = 'in'
            elif side_val <= -self.crossing_threshold:
                side_state = 'out'
            else:
                side_state = 'near'

            person.side_history.append(side_state)

            if side_state == person.last_side_state:
                person.stable_side_frames += 1
            else:
                person.last_side_state = side_state
                person.stable_side_frames = 1

            # Arm track when person is stably on one side.
            if side_state in {'in', 'out'} and person.stable_side_frames >= self.min_stable_side_frames:
                if person.armed_side is None:
                    person.armed_side = side_state

            if person.armed_side is None:
                continue
            if side_state == 'near':
                continue
            if side_state == person.armed_side:
                continue
            if person.stable_side_frames < self.min_stable_side_frames:
                continue
            if now < person.cross_cooldown_until:
                continue
            if len(person.positions) < 2:
                continue

            # Require real travel to avoid jitter-triggered crossings.
            dx = person.x - person.prev_x
            dy = person.y - person.prev_y
            step_move = (dx * dx + dy * dy) ** 0.5
            if step_move < self.min_cross_travel_px:
                continue

            if person.armed_side == 'out' and side_state == 'in':
                self.people_in += 1
                direction = 'IN'
            elif person.armed_side == 'in' and side_state == 'out':
                self.people_out += 1
                direction = 'OUT'
            else:
                continue

            person.direction = direction.lower()
            person.cross_cooldown_until = now + self.crossing_cooldown_sec
            person.armed_side = side_state
            movements.append({
                'id': pid,
                'direction': direction,
                'time': time.strftime('%H:%M:%S')
            })
        
        # drop stale tracks
        to_remove = []
        for pid, person in self.persons.items():
            if pid not in matched_ids:
                person.age()
                if person.is_old(max_age=30):  # Keep tracks longer to survive brief detector misses
                    to_remove.append(pid)
        
        for pid in to_remove:
            del self.persons[pid]
        
        return movements
    
    def get_active_persons(self):
        # used by stats panel
        return {pid: p for pid, p in self.persons.items() if p.confirmed}

    def get_display_tracks(self):
        # Draw stable boxes from confirmed tracks, including short missed windows.
        tracks = []
        for pid, person in self.persons.items():
            if not person.confirmed:
                continue
            if person.last_box is None:
                continue
            if person.frames_without_detection > self.display_hold_frames:
                continue
            tracks.append({
                'id': pid,
                'box': person.last_box,
                'center': (person.x, person.y),
                'confidence': person.last_confidence,
                'stale': person.frames_without_detection > 0,
            })
        return tracks
    
    def reset_counts(self):
        # handy while testing
        self.people_in = 0
        self.people_out = 0
