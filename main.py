import pygame
from pygame.locals import *
from typing import *
import numpy as np
import math
import sys

G = 6.67384 * 10 ** (-11)

EARTH_MASS = 5.972 * (10 ** 24)

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

SIZE = (1280, 720)
CENTER = (SIZE[0] / 2, SIZE[1] / 2)
CENTER_VECTOR = np.array(CENTER)


class Object:
    def __init__(self, mass: float, coord, velocity) -> None:
        self.mass: float = mass
        self.coord: np.ndarray[float, float] = coord
        self.velocity: np.ndarray[float, float] = velocity

    def move(self, dt: float):
        self.coord = self.coord + self.velocity * dt

    def gravitate(self, subject, dt: float):
        subject: Object = subject
        delta_coord: np.ndarray[float, float] = self.coord - subject.coord
        r = math.sqrt(np.sum(delta_coord ** 2))
        if r == 0:
            return delta_coord
        a = G * self.mass / (r ** 2)
        vector_a: np.ndarray[float, float] = delta_coord / r * a
        subject.velocity = subject.velocity + vector_a * dt
        return vector_a


def main():
    dt = 0.01
    fps = int(1 / dt)
    # radius = 637
    earth: Object = Object(10 ** 17, np.array([0, 0]), np.array([0, 0]))
    # obj: Object = Object(1, np.array([0, 100]), np.array([0, 0]))

    objects: Optional[List[Object]] = [Object(1, np.array([0, 100]), np.array([0, 0]))]
    objects.append(earth)

    pygame.init()
    screen = pygame.display.set_mode(SIZE)
    clock = pygame.time.Clock()
    prepare_object = False
    pos = np.array([0, 0])

    while True:
        clock.tick(fps)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = np.array(pygame.mouse.get_pos()) - CENTER_VECTOR
                prepare_object = True
            elif event.type == pygame.MOUSEBUTTONUP:
                if prepare_object:
                    prepare_object = False
                    objects.append(Object(10 ** 15, pos, np.array(pygame.mouse.get_pos()) - CENTER_VECTOR - pos))

        if prepare_object:
            pygame.draw.circle(screen, BLACK, pos + CENTER_VECTOR, 10)
            pygame.draw.line(screen, RED, pos + CENTER_VECTOR, pygame.mouse.get_pos(), 3)

        screen.fill(WHITE)
        for o in objects:
            pygame.draw.circle(screen, BLACK, o.coord + CENTER_VECTOR, o.mass ** (1 / 15))
            pygame.draw.line(screen, RED, o.coord + CENTER_VECTOR, o.coord + CENTER_VECTOR + o.velocity, 3)

        for oo in objects:
            for o in objects:
                if o == oo:
                    continue
                a = oo.gravitate(o, dt)
                pygame.draw.line(screen, BLUE, o.coord + CENTER_VECTOR, o.coord + CENTER_VECTOR + a, 3)

        for o in objects:
            o.move(dt)

        new_object: List[Object] = []
        del_list: List[Object] = []
        for i in range(len(objects)):
            if objects[i] is None:
                continue
            for j in range(i + 1, len(objects)):
                if objects[i] is None or objects[j] is None:
                    continue
                o1 = objects[i]
                o2 = objects[j]
                delta_coord = o1.coord - o2.coord
                r = math.sqrt(sum(delta_coord ** 2))
                if r < 10:
                    new_object.append(Object(o1.mass + o2.mass, (o1.coord + o2.coord) / 2,
                                             (o1.velocity * o1.mass + o2.velocity * o2.mass) / (o1.mass + o2.mass)))
                    objects[i] = None
                    objects[j] = None
            if objects[i] is not None:
                new_object.append(objects[i])

        objects = new_object

        pygame.display.flip()


if __name__ == '__main__':
    main()
