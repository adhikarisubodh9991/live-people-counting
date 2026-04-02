# simple nearest-neighbor tracker and line crossing counter

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


class PersonTracker:
    def __init__(self, door_line_y):
        self.door_line_y = int(door_line_y)
        self.people = {}
        self.next_id = 1
        self.people_in = 0
        self.people_out = 0

    def update(self, detections):
        events = []
        used = set()

        for det in detections:
            cx, cy = det["center"]

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
                if p.prev_y < self.door_line_y <= p.y:
                    p.counted = True
                    self.people_in += 1
                    events.append({"id": pid, "direction": "IN"})
                elif p.prev_y > self.door_line_y >= p.y:
                    p.counted = True
                    self.people_out += 1
                    events.append({"id": pid, "direction": "OUT"})

            if pid not in used:
                p.age()
                if p.missed > 20:
                    del self.people[pid]

        return events
