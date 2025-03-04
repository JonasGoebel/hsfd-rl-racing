from enum import IntEnum


class Action(IntEnum):
    LEFT = 0  # turn more left
    FORWARD = 1  # no corrections
    RIGHT = 2  # turn more right
