import numpy as np
from itertools import tee
import math
from typing import List
from shapely.geometry import Polygon


class MusicRegion:
    def __init__(self, regions: List[Polygon],  staffs: List[List[List[int]]]):
        self.regions = sorted(regions, key=lambda x: x.centroid.x)
        self.avg_height = np.mean([y.centroid.y for y in regions])
        self.staffs = staffs

    def get_horizontal_gaps(self):
        if len(self.regions) == 1:
            return 0
        i1, i2 = tee(self.regions, 2)
        next(i2)
        combinations = list(zip(i1, i2))
        gaps = []
        for p1, p2 in combinations:
            p1_bounds = p1.bounds
            p2_bounds = p2.bounds
            gaps.append([p1_bounds[2], p2_bounds[0]])

        return gaps

    def get_region_borders(self):
        if len(self.regions) == 1:
            return self.regions[0].bounds
        else:
            return [x.bounds for x in self.regions]

    def get_height_difference(self, height: int):
        return abs(self.avg_height - height)

    def get_miny_at_x(self, x_arr: List[int]):
        y, x = zip(*self.staffs[0][0])
        return np.interp(x_arr, x, y)

    def get_maxy_at_x(self, x_arr: List[int]):
        y, x = zip(*self.staffs[0][-1])
        return np.interp(x_arr, x, y)


class MusicRegions:
    def __init__(self, music_regions: List[MusicRegion], staffs: List[List[List[int]]]):
        self.music_regions = music_regions
        self.avg_height = [x.avg_height for x in music_regions]
        self.staffs = staffs

    def get_upper_region(self, height: int):
        prev_value = -math.inf
        region = None
        for x in self.music_regions:
            difference = x.avg_height - height
            if 0 > difference > prev_value:
                prev_value = difference
                region = x
        return region

    def get_lower_region(self, height: int):
        prev_value = math.inf
        region = None
        for x in self.music_regions:
            difference = x.avg_height - height
            if 0 < difference < prev_value:
                prev_value = difference
                region = x
        return region




