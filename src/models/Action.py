from enum import IntEnum


class Action(IntEnum):
    LEFT = -1  # turn more left
    FORWARD = 0  # no corrections
    RIGHT = 1  # turn more right
