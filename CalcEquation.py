import numpy as np
from typing import *


def calc(coords: List[np.ndarray]):
    f = 1

    l: List[np.ndarray] = []
    for i in range(5):
        x = coords[i][0]
        y = coords[i][1]
        l.append(np.array([x ** 2, x * y, x, y ** 2, y]))
    a = np.array(l)
    inv_a = np.linalg.inv(a)
    ans = inv_a.dot(np.array([f, f, f, f, f]).T)

    print(
        "%.20lfx^2  %+.20lfxy  %+.20lfx  %+.20lfy^2  %+.20lfy = %f" %
        (ans[0], ans[1] * -1, ans[2], ans[3], ans[4] * -1, f)
    )
