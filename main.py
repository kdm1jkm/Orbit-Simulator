import math
import sys
from typing import *
from enum import Enum

import numpy as np
import pygame

from Object import Object

# 각종 상수들

# 1회당 확대율
ZOOM_RATE = 1.5

# 질량들
LIGHTER_MASS = 10 ** 5
LIGHT_MASS = 10 ** 10
HEAVY_MASS = 10 ** 18
HEAVIER_MASS = 10 ** 22

SUN_MASS = 1.989 * (10 ** 30)
EARTH_MASS = 5.972 * (10 ** 24)
MOON_MASS = 7.347673 * (10 ** 22)

MASS_RATE = 10

# 중력상수
G = 6.67384 * 10 ** (-11)

# 색
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

# 화면 크기, 중심 위치
SIZE = (1280, 720)
# SIZE = (1920, 1080)
CENTER = (SIZE[0] / 2, SIZE[1] / 2)
CENTER_VECTOR = np.array(CENTER)


class ObjectExample(Enum):
    TRIANGLE = 1
    SIMPLIFIED_SOLAR_SYSTEM = 2
    TWO_OBJECT_GRAVITATING = 3
    PERFECT_CIRCLE = 4


TRIANGLE_V = (G * HEAVY_MASS / 1000) ** 0.5
OBJECT_EXAMPLES: Dict[ObjectExample, List[Object]] = {
    ObjectExample.TRIANGLE: [
        Object(HEAVY_MASS, np.array([0, 0]), np.array([TRIANGLE_V, 0])),
        Object(HEAVY_MASS, np.array([500, 500 * (3 ** 0.5)]), np.array([-TRIANGLE_V / 2, TRIANGLE_V / 2 * (3 ** 0.5)])),
        Object(HEAVY_MASS, np.array([-500, 500 * (3 ** 0.5)]),
               np.array([-TRIANGLE_V / 2, -TRIANGLE_V / 2 * (3 ** 0.5)])),
    ],
    ObjectExample.SIMPLIFIED_SOLAR_SYSTEM: [
        Object(HEAVIER_MASS, np.array([0, 0]), np.array([100, 0])),
        Object(HEAVY_MASS, np.array([10000, 0]), np.array([100, (G * HEAVIER_MASS / 10000) ** 0.5])),
        Object(LIGHT_MASS, np.array([10100, 0]),
               np.array([100, (G * HEAVY_MASS / 100) ** 0.5 + (G * HEAVIER_MASS / 10000) ** 0.5])),
    ],
    ObjectExample.TWO_OBJECT_GRAVITATING: [
        Object(HEAVY_MASS, np.array([500, 0]), np.array([0, 100])),
        Object(HEAVY_MASS, np.array([-500, 0]), np.array([0, -100])),
    ],
    ObjectExample.PERFECT_CIRCLE: [
        Object(HEAVY_MASS, np.array([0, 0]), np.array([0, 0])),
        Object(LIGHT_MASS, np.array([500, 0]), np.array([0, (G * HEAVY_MASS / 500) ** 0.5]))
    ]
}


def to_screen_coord(coord: np.ndarray, screen_center: np.ndarray, zoom: float):
    return (coord + screen_center - CENTER_VECTOR) * zoom + CENTER_VECTOR


def to_original_coord(coord: np.ndarray, screen_center: np.ndarray, zoom: float):
    return (coord - CENTER_VECTOR) / zoom - screen_center + CENTER_VECTOR


def main():
    dt = 0.01
    fps = int(1 / dt)

    objects: List[Optional[Object]] = []

    print("Choose sample(Enter nothing to start with blank screen)")
    print("1. Triangle")
    print("2. Simplified Solar System")
    print("3. Two planet gravitating each other.")
    print("4. Perfect Circle")
    choice = input("Enter>>")

    try:
        n = int(choice)
        for sort in ObjectExample:
            if sort.value == n:
                for o in OBJECT_EXAMPLES[sort]:
                    objects.append(o)
    except:
        pass

    pygame.init()
    screen = pygame.display.set_mode(SIZE)
    clock = pygame.time.Clock()

    pos = np.array([0, 0])
    mouse_start = np.array([0, 0])
    screen_center = CENTER_VECTOR

    next_mass = LIGHT_MASS
    follow_num = 0
    zoom = 1

    tracing = False
    prepare_object = False
    moving_screen = False
    show_physical_quantity = True
    follow_object = False
    make_object = False

    screen.fill(WHITE)

    while True:
        clock.tick(fps)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

            elif event.type == pygame.MOUSEBUTTONDOWN:

                if event.button == pygame.BUTTON_LEFT:
                    if make_object:
                        pos = to_original_coord(np.array(pygame.mouse.get_pos()), screen_center, zoom)
                        prepare_object = True

                elif event.button == pygame.BUTTON_RIGHT:
                    mouse_start = to_original_coord(np.array(pygame.mouse.get_pos()), screen_center, zoom)
                    moving_screen = True

                elif event.button == pygame.BUTTON_WHEELDOWN:
                    zoom /= ZOOM_RATE

                elif event.button == pygame.BUTTON_WHEELUP:
                    zoom *= ZOOM_RATE

            elif event.type == pygame.KEYDOWN:

                if event.key == pygame.K_t:
                    tracing = not tracing

                elif event.key == pygame.K_p:
                    show_physical_quantity = not show_physical_quantity

                elif event.key == pygame.K_f:
                    follow_object = not follow_object

                elif event.key == pygame.K_g:
                    follow_num -= 1

                elif event.key == pygame.K_h:
                    follow_num += 1

                elif event.key == pygame.K_d:
                    if follow_object and len(objects) > 0:
                        objects.pop(follow_num)

                elif event.key == pygame.K_q:
                    make_object = not make_object
                    if not make_object:
                        prepare_object = False

                elif event.key == pygame.K_w:
                    next_mass /= MASS_RATE

                elif event.key == pygame.K_e:
                    next_mass *= MASS_RATE

                elif event.key == pygame.K_r:
                    next_mass = LIGHT_MASS

            elif event.type == pygame.MOUSEBUTTONUP:

                if event.button == pygame.BUTTON_LEFT:
                    if make_object and prepare_object:
                        prepare_object = make_object = False
                        velocity = to_original_coord(np.array(pygame.mouse.get_pos()), screen_center, zoom) - pos
                        objects.append(Object(next_mass, pos, velocity))

                elif event.button == pygame.BUTTON_RIGHT:
                    moving_screen = False

        if not tracing:
            screen.fill(WHITE)

        if follow_num >= len(objects):
            follow_num = 0
        elif follow_num < 0:
            follow_num = len(objects) - 1

        if follow_object and len(objects) > 0:
            screen_center = -objects[follow_num].coord + CENTER_VECTOR

        if moving_screen:
            screen_center = screen_center + (
                    to_original_coord(np.array(pygame.mouse.get_pos()), screen_center, zoom) - mouse_start)
            mouse_start = to_original_coord(np.array(pygame.mouse.get_pos()), screen_center, zoom)

        if not prepare_object:
            pos = to_original_coord(np.array(pygame.mouse.get_pos()), screen_center, zoom)

        screen_pos = to_screen_coord(pos, screen_center, zoom)
        if make_object:
            pygame.draw.circle(screen, BLACK, screen_pos, int(next_mass ** (1 / 14) * zoom) + 2)
            pygame.draw.line(screen, RED, screen_pos, pygame.mouse.get_pos(), int(3 * zoom) + 1)
        if make_object and len(objects) > 0:
            temp_o = Object(next_mass, pos, np.array([0, 0]))
            acc_sum = np.array([0, 0])
            mass_sum = (0, np.array([0, 0]))

            for sort in objects:
                acc_sum = acc_sum + sort.calc_acc(temp_o)
                r = sum((temp_o.coord - sort.coord) ** 2)
                mass_sum = (mass_sum[0] + sort.mass / r, mass_sum[1] + sort.coord * sort.mass / r)

            mass_center = mass_sum[1] / mass_sum[0]
            r = sum((mass_center - pos) ** 2) ** 0.5
            a = sum(acc_sum ** 2) ** 0.5
            v = (a * r) ** 0.5

            pygame.draw.circle(screen, BLUE, screen_pos, int(v * zoom), 3)
            pygame.draw.circle(screen, RED, screen_pos, int((2 * G * mass_sum[0] * r) ** 0.5 * zoom), 3)
            pygame.draw.line(screen, BLUE, to_screen_coord(mass_center, screen_center, zoom), screen_pos, 3)

            guide1 = np.array([pos[0] + (G * mass_sum[0] * r) ** 0.5 / r * (mass_sum[1][1] / mass_sum[0] - pos[1]),
                               pos[1] - (G * mass_sum[0] * r) ** 0.5 / r * (mass_sum[1][0] / mass_sum[0] - pos[0])])
            guide2 = np.array([pos[0] - (G * mass_sum[0] * r) ** 0.5 / r * (mass_sum[1][1] / mass_sum[0] - pos[1]),
                               pos[1] + (G * mass_sum[0] * r) ** 0.5 / r * (mass_sum[1][0] / mass_sum[0] - pos[0])])
            pygame.draw.line(screen, BLUE, to_screen_coord(guide1, screen_center, zoom),
                             to_screen_coord(guide2, screen_center, zoom))

        for sort in objects:
            if sort == objects[follow_num] and follow_object:
                color = BLUE
            else:
                color = BLACK
            pygame.draw.circle(screen, color, to_screen_coord(sort.coord, screen_center, zoom),
                               int(sort.mass ** (1 / 14) * zoom) + 2)
            if show_physical_quantity:
                pygame.draw.line(screen, RED, to_screen_coord(sort.coord, screen_center, zoom),
                                 to_screen_coord(sort.coord + sort.velocity, screen_center, zoom), int(3 * zoom) + 1)

        for oo in objects:
            for sort in objects:
                if sort == oo:
                    continue
                a = oo.gravitate(sort, dt)
                if show_physical_quantity:
                    pygame.draw.line(screen, BLUE, to_screen_coord(sort.coord, screen_center, zoom),
                                     to_screen_coord(sort.coord + a, screen_center, zoom), int(3 * zoom) + 1)

        for sort in objects:
            sort.move(dt)

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
