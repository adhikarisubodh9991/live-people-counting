# tracker

import time


class Person:
    def __init__(self, pid, x, y):
        self.id = pid
        self.x = x
        self.y = y
        self.prev_x = x
        self.prev_y = y
        self.counted = False
        self.missed = 0
        self.last_seen = time.time()

    def update(self, x, y):
        self.prev_x = self.x
        self.prev_y = self.y
        self.x = x
        self.y = y
        self.missed = 0
        self.last_seen = time.time()

    def age(self):
        self.missed += 1

    def is_old(self, max_age=20):
        return self.missed > max_age


class PersonTracker:
    def __init__(self, door_line_y, door_line_start=None, door_line_end=None):
        self.door_line_y = int(door_line_y)
        self.door_line_start = door_line_start
        self.door_line_end = door_line_end

        if self.door_line_start is None or self.door_line_end is None:
            self.door_line_start = (0, self.door_line_y)
            self.door_line_end = (1, self.door_line_y)

        self.people = {}
        self.next_id = 1
        self.people_in = 0
        self.people_out = 0

    def _side(self, x, y):
        x1, y1 = self.door_line_start
        x2, y2 = self.door_line_end
        return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)

    def update(self, detections):
        events = []
        used = set()
        threshold = 5

        for d in detections:
            cx, cy = d["center"]

            best_id = None
            best_dist = 10 ** 9
            for pid, p in self.people.items():
                if pid in used:
                    continue

                dist = ((p.x - cx) ** 2 + (p.y - cy) ** 2) ** 0.5
                if dist < 200 and dist < best_dist:
                    best_dist = dist
                    best_id = pid

            if best_id is None:
                p = Person(self.next_id, cx, cy)
                self.people[self.next_id] = p
                used.add(self.next_id)
                self.next_id += 1
            else:
                p = self.people[best_id]
                p.update(cx, cy)
                used.add(best_id)

        for pid, p in list(self.people.items()):
            if not p.counted:
                prev_side = self._side(p.prev_x, p.prev_y)
                cur_side = self._side(p.x, p.y)

                if abs(prev_side) <= threshold and abs(cur_side) <= threshold:
                    continue

                if prev_side < -threshold and cur_side >= threshold:
                    p.counted = True
                    self.people_in += 1
                    events.append({"id": pid, "direction": "IN"})
                elif prev_side > threshold and cur_side <= -threshold:
                    p.counted = True
                    self.people_out += 1
                    events.append({"id": pid, "direction": "OUT"})

            if pid not in used:
                p.age()
                if p.is_old():
                    del self.people[pid]

        return events
