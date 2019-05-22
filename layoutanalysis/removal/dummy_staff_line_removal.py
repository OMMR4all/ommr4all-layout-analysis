import numpy as np
from scipy.interpolate import interpolate
import math
from typing import List


def staff_removal(staffs_lines: List[List[List[int]]], img: np.ndarray, line_height: int):
    nimg = np.copy(img)
    h = nimg.shape[0]
    l2 = math.ceil(line_height / 2)
    l2 = max(l2, 2)
    for system in staffs_lines:
        for ind, staff in enumerate(system):
            y, x = zip(*staff)
            f = interpolate.interp1d(x, y)
            x_start, x_end = int(min(x)), int(max(x))
            for i in range(x_start, x_end):
                count = []
                st_point = int(f(i))
                if nimg[st_point][i] != 0:
                    for z in range(1, l2 + 1):
                        if nimg[st_point - z][i] == 0:
                            st_point = st_point-z
                            break
                        if nimg[st_point + z][i] == 0:
                            st_point = st_point+z
                            break
                yt = st_point
                yb = st_point
                if nimg[yt][i] == 0:
                    count.append(yt)
                    while yt < h - 1:
                        yt += 1
                        if nimg[yt][i] == 0:
                            count.append(yt)
                        else:
                            break
                    while yb > 0:
                        yb -= 1
                        if nimg[yb][i] == 0:
                            count.append(yb)
                        else:
                            break
                if len(count) <= line_height:
                    for it in count:
                        nimg[it][i] = 1
    return nimg
