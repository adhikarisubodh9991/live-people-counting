# basic multi-person tracker and line crossing counter

import time


class Person:
    # per-person tracking state
    
    def __init__(self, person_id, x, y):
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
        
    def update(self, x, y):
        # save last point before moving
        self.prev_x = self.x
        self.prev_y = self.y
        self.x = x
        self.y = y
        self.positions.append((x, y))
        self.frames_without_detection = 0
        self.last_seen = time.time()
        
    def age(self):
        # not seen in this frame
        self.frames_without_detection += 1
        
    def is_old(self, max_age=10):
        # stale track check
        return self.frames_without_detection > max_age


class PersonTracker:
    # matches detections to tracks + counts in/out
    
    def __init__(self, door_line_y, zone_top=150, zone_bottom=150, door_line_start=None, door_line_end=None):
        # zone_top / zone_bottom kept for compatibility
        self.door_line_y = door_line_y
        self.door_line_start = door_line_start
        self.door_line_end = door_line_end

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

    def _point_side(self, x, y):
        # cross-product sign tells which side of the line this point is on
        x1, y1 = self.door_line_start
        x2, y2 = self.door_line_end
        return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
        
    def update(self, detections):
        # one frame update: match, count crossings, cleanup
        movements = []
        matched_ids = set()  # avoid double-match in same frame
        
        # Try to match each detection with an existing tracked person
        for det in detections:
            cx, cy = det['center']
            
            best_match = None
            best_distance = float('inf')
            
            # Find the closest tracked person to this detection
            for pid, person in self.persons.items():
                if pid in matched_ids:
                    # Already matched this person in this frame
                    continue
                    
                # basic nearest-neighbor match
                dist = ((person.x - cx) ** 2 + (person.y - cy) ** 2) ** 0.5
                
                # 200px gate keeps random jumps low
                if dist < 200 and dist < best_distance:
                    best_match = pid
                    best_distance = dist
            
            if best_match is not None:
                # Update existing tracked person
                person = self.persons[best_match]
                person.update(cx, cy)
                matched_ids.add(best_match)
            else:
                # Create a new track for this person
                person = Person(self.next_id, cx, cy)
                self.persons[self.next_id] = person
                matched_ids.add(self.next_id)
                self.next_id += 1
        
        # Check if any tracked people crossed the door line
        crossing_threshold = 5  # tiny buffer so jitter doesn't count
        
        for pid, person in self.persons.items():
            if not person.counted:
                prev_side = self._point_side(person.prev_x, person.prev_y)
                curr_side = self._point_side(person.x, person.y)

                # Ignore if they're just vibrating near the line (noise)
                if abs(prev_side) <= crossing_threshold and abs(curr_side) <= crossing_threshold:
                    continue

                # side flip == crossed line
                if prev_side < -crossing_threshold and curr_side >= crossing_threshold:
                    # Crossed from top to bottom (entering)
                    person.counted = True
                    person.direction = 'in'
                    self.people_in += 1
                    movements.append({
                        'id': pid,
                        'direction': 'IN',
                        'time': time.strftime('%H:%M:%S')
                    })
                elif prev_side > crossing_threshold and curr_side <= -crossing_threshold:
                    # Crossed from bottom to top (exiting)
                    person.counted = True
                    person.direction = 'out'
                    self.people_out += 1
                    movements.append({
                        'id': pid,
                        'direction': 'OUT',
                        'time': time.strftime('%H:%M:%S')
                    })
        
        # drop stale tracks
        to_remove = []
        for pid, person in self.persons.items():
            if pid not in matched_ids:
                person.age()
                if person.is_old(max_age=20):  # Allow up to 20 frames of no detection
                    to_remove.append(pid)
        
        for pid in to_remove:
            del self.persons[pid]
        
        return movements
    
    def get_active_persons(self):
        # used by stats panel
        return self.persons
    
    def reset_counts(self):
        # handy while testing
        self.people_in = 0
        self.people_out = 0
