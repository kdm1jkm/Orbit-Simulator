import pygame
from pygame.locals import *
from typing import *
import numpy as np
import math
import sys

ZOOM_RATE = 1.5

LIGHTER_MASS = 10 ** 5
LIGHT_MASS = 10 ** 10
HEAVY_MASS = 10 ** 18
HEAVIER_MASS = 10 ** 22

G = 6.67384 * 10 ** (-11)

SUN_MASS = 1.989 * (10 ** 30)
EARTH_MASS = 5.972 * (10 ** 24)
MOON_MASS = 7.347673 * (10 ** 22)

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

SIZE = (1280, 720)
# SIZE = (1920, 1080)
CENTER = (SIZE[0] / 2, SIZE[1] / 2)
CENTER_VECTOR = np.array(CENTER)


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


def to_screen_coord(coord: np.ndarray, screen_center: np.ndarray, zoom: float):
    return (coord + screen_center - CENTER_VECTOR) * zoom + CENTER_VECTOR


def to_original_coord(coord: np.ndarray, screen_center: np.ndarray, zoom: float):
    return (coord - CENTER_VECTOR) / zoom - screen_center + CENTER_VECTOR


def main():
    dt = 0.01
    fps = int(1 / dt)

    v = (G * HEAVY_MASS / 1000) ** 0.5

    objects: Optional[List[Object]] = [
        # Object(HEAVY_MASS, np.array([0, 0]), np.array([v, 0])),
        # Object(HEAVY_MASS, np.array([500, 500 * (3 ** 0.5)]), np.array([-v / 2, v / 2 * (3 ** 0.5)])),
        # Object(HEAVY_MASS, np.array([-500, 500 * (3 ** 0.5)]), np.array([-v / 2, -v / 2 * (3 ** 0.5)])),

        # Object(HEAVIER_MASS, np.array([0, 0]), np.array([100, 0])),
        # Object(HEAVY_MASS, np.array([10000, 0]), np.array([100, (G * HEAVIER_MASS / 10000) ** 0.5])),
        # Object(LIGHT_MASS, np.array([10100, 0]),
        #        np.array([100, (G * HEAVY_MASS / 100) ** 0.5 + (G * HEAVIER_MASS / 10000) ** 0.5])),

        # Object(SUN_MASS, np.array([0, 0]), np.array([0, 0])),
        # Object(EARTH_MASS, np.array([1000, 0]), np.array([0, (G * SUN_MASS / 1000) ** 0.5])),
        # Object(MOON_MASS, np.array([100100, 0]), np.array([0, (G * EARTH_MASS / 100) ** 0.5])),

        # Object(HEAVY_MASS, np.array([500, 0]), np.array([0, 100])),
        # Object(HEAVY_MASS, np.array([-500, 0]), np.array([0, -100])),
    ]

    print("1. Triangle")
    print("2. Simplified Solar System")
    print("3. Two planet gravitating each other.")
    choice = input("Enter>>")

    if choice == "1":
        objects.append(Object(HEAVY_MASS, np.array([0, 0]), np.array([v, 0])))
        objects.append(Object(HEAVY_MASS, np.array([500, 500 * (3 ** 0.5)]), np.array([-v / 2, v / 2 * (3 ** 0.5)])))
        objects.append(Object(HEAVY_MASS, np.array([-500, 500 * (3 ** 0.5)]), np.array([-v / 2, -v / 2 * (3 ** 0.5)])))
    elif choice == "2":
        objects.append(Object(HEAVIER_MASS, np.array([0, 0]), np.array([100, 0])))
        objects.append(Object(HEAVY_MASS, np.array([10000, 0]), np.array([100, (G * HEAVIER_MASS / 10000) ** 0.5])))
        objects.append(Object(LIGHT_MASS, np.array([10100, 0]),
                              np.array([100, (G * HEAVY_MASS / 100) ** 0.5 + (G * HEAVIER_MASS / 10000) ** 0.5])))
    elif choice == "3":
        objects.append(Object(HEAVY_MASS, np.array([500, 0]), np.array([0, 100])))
        objects.append(Object(HEAVY_MASS, np.array([-500, 0]), np.array([0, -100])))

    pygame.init()
    screen = pygame.display.set_mode(SIZE)
    tracing = False
    clock = pygame.time.Clock()
    prepare_object = False
    pos = np.array([0, 0])

    screen_center = CENTER_VECTOR
    mouse_start = np.array([0, 0])
    moving_screen = False

    next_mass = 0

    show_physical_quantity = True

    zoom = 1
    screen.fill(WHITE)

    while True:
        clock.tick(fps)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == pygame.BUTTON_LEFT:
                    pos = to_original_coord(np.array(pygame.mouse.get_pos()), screen_center, zoom)
                    prepare_object = True
                    next_mass = LIGHT_MASS
                elif event.button == pygame.BUTTON_RIGHT:
                    mouse_start = to_original_coord(np.array(pygame.mouse.get_pos()), screen_center, zoom)
                    moving_screen = True
                elif event.button == pygame.BUTTON_WHEELDOWN:
                    zoom /= ZOOM_RATE
                elif event.button == pygame.BUTTON_WHEELUP:
                    zoom *= ZOOM_RATE
                elif event.button == pygame.BUTTON_MIDDLE:
                    pos = to_original_coord(np.array(pygame.mouse.get_pos()), screen_center, zoom)
                    prepare_object = True
                    next_mass = HEAVY_MASS
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_t:
                    tracing = not tracing
                elif event.key == pygame.K_p:
                    show_physical_quantity = not show_physical_quantity
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == pygame.BUTTON_LEFT or event.button == pygame.BUTTON_MIDDLE:
                    if prepare_object:
                        prepare_object = False
                        objects.append(Object(next_mass, pos,
                                              to_original_coord(np.array(pygame.mouse.get_pos()), screen_center,
                                                                zoom) - pos))
                elif event.button == pygame.BUTTON_RIGHT:
                    moving_screen = False

        if not tracing:
            screen.fill(WHITE)

        if moving_screen:
            screen_center = screen_center + to_original_coord(np.array(pygame.mouse.get_pos()), screen_center,
                                                              zoom) - mouse_start
            mouse_start = to_original_coord(np.array(pygame.mouse.get_pos()), screen_center, zoom)

        if prepare_object and len(objects) > 0:
            pygame.draw.circle(screen, BLACK, to_screen_coord(pos, screen_center, zoom), int(10 * zoom))
            pygame.draw.line(screen, RED, to_screen_coord(pos, screen_center, zoom),
                             pygame.mouse.get_pos(), int(3 * zoom) + 1)
            temp_o = Object(next_mass, pos, np.array([0, 0]))
            acc_sum = np.array([0, 0])
            mass_sum = (next_mass, pos * next_mass)
            for o in objects:
                acc_sum = acc_sum + o.calc_acc(temp_o)
                mass_sum = (mass_sum[0] + o.mass, mass_sum[1] + o.coord * o.mass)
            r = (sum(((mass_sum[1] / mass_sum[0]) - pos) ** 2) ** 0.5)
            a = sum(acc_sum ** 2) ** 0.5
            v = (a * r) ** 0.5
            pygame.draw.circle(screen, BLUE, to_screen_coord(pos, screen_center, zoom), int(v * zoom), 3)
            pygame.draw.circle(screen, RED, to_screen_coord(pos, screen_center, zoom),
                               int((2 * G * mass_sum[0] / r) ** 0.5 * zoom), 3)
            pygame.draw.line(screen, BLUE, to_screen_coord(mass_sum[1] / mass_sum[0], screen_center, zoom),
                             to_screen_coord(pos, screen_center, zoom), 3)

            guide1 = np.array([pos[0] + (G * mass_sum[0] / r) ** 0.5 / r * (mass_sum[1][1] / mass_sum[0] - pos[1]),
                               pos[1] - (G * mass_sum[0] / r) ** 0.5 / r * (mass_sum[1][0] / mass_sum[0] - pos[0])])
            guide2 = np.array([pos[0] - (G * mass_sum[0] / r) ** 0.5 / r * (mass_sum[1][1] / mass_sum[0] - pos[1]),
                               pos[1] + (G * mass_sum[0] / r) ** 0.5 / r * (mass_sum[1][0] / mass_sum[0] - pos[0])])
            pygame.draw.line(screen, BLUE, to_screen_coord(guide1, screen_center, zoom),
                             to_screen_coord(guide2, screen_center, zoom))

        for o in objects:
            pygame.draw.circle(screen, BLACK, to_screen_coord(o.coord, screen_center, zoom),
                               int(o.mass ** (1 / 14) * zoom) + 2)
            if show_physical_quantity:
                pygame.draw.line(screen, RED, to_screen_coord(o.coord, screen_center, zoom),
                                 to_screen_coord(o.coord + o.velocity, screen_center, zoom), int(3 * zoom) + 1)

        for oo in objects:
            for o in objects:
                if o == oo:
                    continue
                a = oo.gravitate(o, dt)
                if show_physical_quantity:
                    pygame.draw.line(screen, BLUE, to_screen_coord(o.coord, screen_center, zoom),
                                     to_screen_coord(o.coord + a, screen_center, zoom), int(3 * zoom) + 1)

        for o in objects:
            o.move(dt)

        new_object: List[Object] = []
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
                if r < o1.mass ** (1 / 15) + o2.mass ** (1 / 15):
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
