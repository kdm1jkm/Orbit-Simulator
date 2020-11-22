import math

import numpy as np

# 중력상수
G = 6.67384 * 10 ** (-11)


class Object:
    def __init__(self, mass: float, coord, velocity) -> None:
        self.mass: float = mass
        self.coord: np.ndarray[float, float] = coord
        self.velocity: np.ndarray[float, float] = velocity

    def move(self, dt: float):
        self.coord = self.coord + self.velocity * dt

    def calc_acc(self, subject):
        subject: Object = subject
        delta_coord: np.ndarray[float, float] = self.coord - subject.coord
        r = math.sqrt(np.sum(delta_coord ** 2))
        if r == 0:
            return delta_coord
        a = G * self.mass / (r ** 2)
        vector_a: np.ndarray[float, float] = delta_coord / r * a
        return vector_a

    def gravitate(self, subject, dt: float):
        vector_a = self.calc_acc(subject)
        subject.velocity = subject.velocity + vector_a * dt
        return vector_a
