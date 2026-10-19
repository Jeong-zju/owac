"""Development grasp acceptance, separate from the unchanged benchmark flag.

For the Aloha grab_roller task: both hands contact the roller, no table contact,
roller centre >= 0.83 m (0.85 m demonstration goal, 2 cm tolerance), continuously
for one simulation second. Hidden state is used for scoring only.
"""


class GraspHold:
    def __init__(self, *, minimum_z=0.83, seconds=1.0):
        self.minimum_z, self.seconds = minimum_z, seconds
        self.start = None
        self.last_time = None
        self.hold_seconds = 0.0
        self.max_hold_seconds = 0.0
        self.valid = False

    def update(self, row):
        now = row["physics_tick"] * row["dt"]
        if self.last_time is not None and now <= self.last_time:
            raise ValueError("physics samples must be strictly increasing")
        names = set(row["gripper_link_names"])
        left = {n for n in names if n.startswith("fl_")}
        right = {n for n in names if n.startswith("fr_")}
        if not left or not right:
            raise ValueError("gripper link identity is missing for Aloha acceptance")
        touched = {name for c in row["roller_contacts"] if c["points"] for name in c["bodies"]}
        valid = (
            row["roller_p"][2] >= self.minimum_z
            and bool(touched & left)
            and bool(touched & right)
            and "table" not in touched
            and all(row["closed_flags"])
        )
        if not valid or (self.last_time is not None and now - self.last_time > 0.1):
            self.start = None
        if valid and self.start is None:
            self.start = now
        self.hold_seconds = now - self.start if self.start is not None else 0.0
        self.max_hold_seconds = max(self.max_hold_seconds, self.hold_seconds)
        self.last_time, self.valid = now, valid
        return self.hold_seconds >= self.seconds
