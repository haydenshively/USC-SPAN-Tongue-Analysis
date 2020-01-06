import numpy as np


class History:
    DEFAULT_MEMORY_LENGTH = 20

    def __init__(self, length=DEFAULT_MEMORY_LENGTH):
        self.length = length
        self.time = 0
        self.data = None

    def __iadd__(self, frame):
        if self.time is 0: self.data = np.zeros((self.length, frame.shape[0], frame.shape[1]), dtype=frame.dtype)
        self.data[self.time % self.length] = frame
        self.time += 1
        return self
