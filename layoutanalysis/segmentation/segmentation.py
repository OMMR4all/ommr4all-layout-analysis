from layoutanalysis.pixelclassifier.predictor import PCPredictor
from pagesegmentation.lib.predictor import PredictSettings
from layoutanalysis.removal.dummy_staff_line_removal import staff_removal
from layoutanalysis.preprocessing.util import extract_connected_components, vertical_runs
from layoutanalysis.segmentation.music_region import MusicRegion, MusicRegions
from layoutanalysis.datatypes.datatypes import ImageData
from layoutanalysis.segmentation.util import alpha_shape, cc_cover
from layoutanalysis.segmentation.callback import SegmentationCallback, SegmentationDummyCallback
from layoutanalysis.preprocessing.binarization.ocropus_binarizer import binarize

from PIL import Image
import matplotlib.pyplot as plt
from itertools import chain
import numpy as np
from scipy.ndimage.morphology import binary_erosion, binary_dilation
from dataclasses import dataclass
from collections import defaultdict
from skimage.draw import polygon
import multiprocessing
import tqdm
from functools import partial
from shapely.geometry import Polygon
from scipy.interpolate import interpolate
import math
from scipy.ndimage import gaussian_filter
import cv2

from typing import List


@dataclass
class SegmentationSettings:
    erode: bool = False
    debug: bool = False
    capitals: bool = True
    weight_threshold: float = 0.5
    lineSpaceHeight: int = 20
    targetLineSpaceHeight: int = 10
    model: [str] = None
    cover: float = 0.1
    processes: int = 12


class Segmentator:
    def __init__(self, settings: SegmentationSettings, callback: SegmentationCallback = None):
        self.predictor = None
        self.settings = settings
        if self.settings.model:
            import os
            pcsettings = PredictSettings(
                mode='meta',
                network=os.path.abspath(self.settings.model),
                output=None,
                high_res_output=False
            )
            self.predictor = PCPredictor(pcsettings, self.settings.targetLineSpaceHeight)
        if callback is None:
            self.callback: SegmentationCallback = SegmentationCallback()
        else:
            self.callback: SegmentationCallback = callback

    def segment(self, staffs, img_paths):
        create_data_partial = partial(create_data, line_space_height=self.settings.lineSpaceHeight)
        with multiprocessing.Pool(processes=self.settings.processes) as p:
            data = [v for v in tqdm.tqdm(p.imap(create_data_partial, img_paths), total=len(img_paths))]
        for i_ind, i in enumerate(zip(staffs, data)):
            self.callback.reset_page_state()
            yield self.segment_with_weight_image(i[0], i[1])
            self.callback.update_total_state()

    def segment_with_weight_image(self, staffs: List[List[List[int]]], img_data: np.ndarray):
        poly_dict = defaultdict(list)
        staffs.sort(key=lambda staff: staff[0][0][0])

        img = np.array(Image.open(img_data.path)) / 255
        img_data.image = binarize(img)
        self.callback.update_current_page_state()
        if self.settings.erode:
            img_data.image = binary_dilation(img_data.image, structure=np.full((1, 3), 1))

        staff_image = np.zeros(img_data.image.shape)
        staff_polygons = [generate_polygon_from_staff(staff) for staff in staffs]
        staff_img = draw_polygons(staff_polygons, staff_image)
        self.callback.update_current_page_state()

        distance = vertical_runs(staff_img)[1]
        self.callback.update_current_page_state()

        music_regions = generate_music_region(staff_polygons, staffs, distance)
        self.callback.update_current_page_state()

        # generate weight image
        weight = gaussian_filter(staff_img, sigma=(distance * 2 / 4, distance * 2 / 4))
        weight[staff_img == 1] = 1
        self.callback.update_current_page_state()

        # cut out peripheries
        rmin, rmax, cmin, cmax = bbox2(staff_img)
        rmin = rmin - rmin // 10
        rmax = rmax + (staff_img.shape[0] - rmax) // 5
        cmin = cmin - cmin // 10
        cmax = cmax + (staff_img.shape[1] - cmax) // 5

        bbox = img_data.image[rmin:rmax, cmin:cmax]
        processed_image = np.ones(img_data.image.shape)
        processed_image[rmin:rmax, cmin:cmax] = bbox

        # remove staff lines
        processed_image = staff_removal(staffs, processed_image, 3)
        self.callback.update_current_page_state()

        # extract remaining CCs
        cc_list_with_stats = extract_connected_components(((1 - processed_image) * 255).astype(np.uint8))

        # separate CCs in music and text
        def segment_cc(cc_list_with_stats: List[List[List[int]]], weight_matrix: np.ndarray,
                       avg_distance_between_systems: float, segment_capitals: bool = True,
                       threshold: float = 0.5):
            __cc_list = cc_list_with_stats[0]
            __cc_list_stats = cc_list_with_stats[1]
            __initials = []
            __cc_list_new = []
            for cc_ind, cc in enumerate(__cc_list):
                y, x = zip(*cc)
                min_max_difference = np.max(weight_matrix[y, x]) - np.min(weight_matrix[y, x])

                if segment_capitals and min_max_difference > 0.5 and __cc_list_stats[cc_ind, 3]\
                        > avg_distance_between_systems:
                    __initials.append(cc)
                    continue
                if np.mean(weight_matrix[y, x]) > threshold:
                    __cc_list_new.append(cc)
                    continue
            return __cc_list_new, __initials

        # Divide spaces into groups that reflect systems
        def group_ccs(cc_list: List[List[int]], music_regions : MusicRegion):
            d = defaultdict(list)
            for cc_ind, cc in enumerate(cc_list):
                avg_cc_height = np.mean([cc[-1][0], cc[0][0]])
                avg_cc_x_pos = np.mean([cc[-1][1], cc[0][1]])
                r = math.inf
                r_ind = - 1
                index = 0
                for x in music_regions.music_regions:
                    changed = False
                    if abs(avg_cc_height - x.avg_height) < r:
                        r = abs(avg_cc_height - x.avg_height)
                        r_ind = index
                        changed = True
                    if x.get_horizontal_gaps() != 0:
                        for ind, border in enumerate(x.get_region_borders()):

                            if changed and border[0] < avg_cc_x_pos < border[2]:
                                r_ind = index + ind
                                break
                    index += len(x.regions)
                d[r_ind].append(cc_list[cc_ind])

            # add staff lines to groups
            for r_ind, staff in enumerate(music_regions.staffs):
                for cc in staff:
                    y_list, x_list = zip(*cc)
                    func = interpolate.interp1d(x_list, y_list)
                    start = min(x_list)
                    end = max(x_list)
                    values = list(range(int(start), int(end)))
                    fp = [func(x) for x in values]
                    d[r_ind].append(zip(fp, values))
            return d.values()

        # separate CCs in music and text
        system_ccs, initials = segment_cc(cc_list_with_stats, weight, distance,
                                          segment_capitals=self.settings.capitals)

        self.callback.update_current_page_state()

        # Generate polygons enclosing the initials/ capitals
        initials_polygons = []
        if len(initials) > 0:
            initials_polygons = list(chain.from_iterable([generate_polygons_from__ccs([x], alpha=distance / 2)
                                                          for x in initials]))
            initials_polygons = remove_polys_within_polys(initials_polygons)

        for poly in initials_polygons:
            poly_dict['initials'].append(poly)

        cc_list = group_ccs(system_ccs, music_regions)
        self.callback.update_current_page_state()

        generate_polygons_from__ccs_partial = partial(generate_polygons_from__ccs, alpha=distance,
                                                      buffer_size=2.0)
        self.callback.update_current_page_state()

        # Generate polygons enclosing the systems
        with multiprocessing.Pool(processes=self.settings.processes) as p:
            data = [v for v in tqdm.tqdm(p.imap(generate_polygons_from__ccs_partial, cc_list), total=len(cc_list))]
        system_polygons = [poly for p_data in data for poly in p_data]
        system_polygons = remove_polys_within_polys(system_polygons)
        for poly in system_polygons:
            poly_dict['system'].append(poly)
        self.callback.update_current_page_state()

        # Remove capital and system CCs
        text_image = np.zeros(img_data.image.shape)
        text_image = draw_polygons(poly_dict['system'], text_image)
        text_image = draw_polygons(poly_dict['initials'], text_image)
        processed_image = np.clip(processed_image + text_image, 0, 1)
        self.callback.update_current_page_state()

        # extract remaining CCs
        processed_image_cc = extract_connected_components(((1 - processed_image) * 255).astype(np.uint8))

        # divide remaining CCs in lyric/text
        def divide_cc_in_lyric_and_text(ccs: List[List[int]], distance_between_staffs, music_regions: music_regions):

            __ccs_stats = ccs[1]
            __lyric_cc = []
            __text_cc = []

            for ind, cc in enumerate(ccs[0]):
                area = __ccs_stats[ind, cv2.CC_STAT_AREA]
                if area > 10:  # skip small ccs
                    width = __ccs_stats[ind, cv2.CC_STAT_WIDTH]
                    height = __ccs_stats[ind, cv2.CC_STAT_HEIGHT]
                    top = __ccs_stats[ind, cv2.CC_STAT_TOP]

                    left = __ccs_stats[ind, cv2.CC_STAT_LEFT]
                    avg_y = top + height // 2
                    avg_x = left + width // 2
                    top_poly = music_regions.get_upper_region(avg_y)
                    bot_poly = music_regions.get_lower_region(avg_y)

                    if top_poly is None:
                        __text_cc.append(cc)
                        continue

                    top_poly_bounds = top_poly.regions[0].bounds
                    top_border = top_poly.get_maxy_at_x(avg_x)
                    if bot_poly:
                        bot_border = bot_poly.get_miny_at_x(avg_x)
                    else:
                        bot_border = top_border + distance_between_staffs
                    if top_border < avg_y < bot_border:
                        if top_poly.get_horizontal_gaps() == 0:
                            if top_poly_bounds[2] > left + width // 2 > top_poly_bounds[0]:
                                __lyric_cc.append(cc)
                            else:
                                __text_cc.append(cc)
                        else:
                            lies_between_staffs = False
                            for x in top_poly.get_horizontal_gaps():
                                if left > x[0] and left + width < x[1]:
                                    lies_between_staffs = True
                                    break
                            if lies_between_staffs:
                                __text_cc.append(cc)
                            else:
                                __lyric_cc.append(cc)

                    else:
                        __text_cc.append(cc)
            return __lyric_cc, __text_cc

        lyric_cc, text_cc = divide_cc_in_lyric_and_text(processed_image_cc, distance, music_regions)
        self.callback.update_current_page_state()

        data = generate_polygons_from__ccs(text_cc, alpha=distance / 2.3)
        text_polygons = [poly for poly in data]
        text_polygons = remove_polys_within_polys(text_polygons)
        text_polygons = remove_polys_smaller_than_threshold(text_polygons, 0.1)
        self.callback.update_current_page_state()

        data = generate_polygons_from__ccs(lyric_cc, alpha=distance / 2.3)
        lyric_polygons = [poly for poly in data]
        lyric_polygons = remove_polys_within_polys(lyric_polygons)
        lyric_polygons = remove_polys_smaller_than_threshold(lyric_polygons, 0.1)
        self.callback.update_current_page_state()

        for poly in text_polygons:
            poly_dict['text'].append(poly)
        for poly in lyric_polygons:
            poly_dict['lyrics'].append(poly)
        self.callback.update_current_page_state()

        if self.settings.debug:
            print('Generating debug image')
            lyric_image = np.zeros(img_data.image.shape)
            text_image = np.zeros(img_data.image.shape)
            initials_image = np.zeros(img_data.image.shape)
            system_image = np.zeros(img_data.image.shape)
            lyric_image = draw_polygons(poly_dict['lyrics'], lyric_image) * 255
            text_image = draw_polygons(poly_dict['text'], text_image) * 255
            initials_image = draw_polygons(poly_dict['initials'], initials_image) * 255
            system_image = draw_polygons(poly_dict['system'], system_image) * 255

            og_image = np.squeeze(np.stack((img_data.image,) * 3, -1)) * 255
            og_image[np.where(lyric_image == 255)] = og_image[np.where(lyric_image == 255)] * [0.8, 0.2, 0]
            og_image[np.where(text_image == 255)] = og_image[np.where(text_image == 255)] * [0.6, 0.6, 1]
            og_image[np.where(initials_image == 255)] = og_image[np.where(initials_image == 255)] * [0.8, 0.4, 1]
            og_image[np.where(system_image == 255)] = og_image[np.where(system_image == 255)] * [0.3, 1, 0.3]

            plt.imshow(og_image)
            plt.show()
        return poly_dict

    # alternative algorithm, outdated
    def segment_image(self, staffs: List[List[List[int]]], img_data: np.ndarray, region_prediction: np.ndarray):

        img = np.array(Image.open(img_data.path)) / 255
        img_data.image = binarize(img)
        t_region = np.clip(region_prediction, 0, 1) * 255
        region_prediction[region_prediction < 255] = 0
        binarized = 1 - img_data.image
        if self.settings.erode:
            binarized = binary_erosion(binarized, structure=np.full((1, 3), 1))

        staff_image = np.zeros(img_data.image.shape)
        staff_polygons = [generate_polygon_from_staff(staff) for staff in staffs]

        staff_img = draw_polygons(staff_polygons, staff_image)
        img_with_staffs_removed = staff_removal(staffs, 1 - binarized, 3)

        processed_img = np.clip(img_with_staffs_removed + staff_img, 0, 1).astype(np.uint8)
        cc_list = extract_connected_components((1 - processed_img) * 255)[0]
        cc_list_prediction = extract_connected_components(t_region - region_prediction)[0]
        cc_list_prediction = [x for x in cc_list_prediction if len(x) > 100]
        cc_list = cc_cover(np.array(cc_list), np.array(cc_list_prediction), self.settings.cover)
        poly_dict = defaultdict(list)

        with multiprocessing.Pool(processes=self.settings.processes) as p:
            data = [v for v in tqdm.tqdm(p.imap(generate_polygons_from__ccs, cc_list), total=len(cc_list))]

        text_polygons = [poly for p_data in data for poly in p_data]

        polys_to_remove = []
        for ind1, poly in enumerate(text_polygons):
            for ind2, poly2 in enumerate(text_polygons):
                if ind1 != ind2:
                    if poly.contains(poly2):
                        polys_to_remove.append(ind2)

        for ind in reversed(polys_to_remove):
            del text_polygons[ind]
        for poly in text_polygons:
            poly_dict['lyrics'].append(poly)

        text_image = np.zeros(img_data.image.shape)
        text_image = draw_polygons(poly_dict['lyrics'], text_image)

        image_with_text_removed = np.clip(img_data.image + text_image, 0, 1)
        cc_list1 = extract_connected_components(((1 - image_with_text_removed) * 255).astype(np.uint8))[0]
        cc_list1 = [cc for cc in cc_list1 if len(cc) > 10]
        cc_list2 = extract_connected_components((staff_img * 255).astype(np.uint8))[0]
        cc_list_cover = cc_cover(np.array(cc_list1), np.array(cc_list2), self.settings.cover, use_pred_as_start=True)

        generate_polygons_from__ccs_partial = partial(generate_polygons_from__ccs)
        with multiprocessing.Pool(processes=self.settings.processes) as p:
            data = [v for v in tqdm.tqdm(p.imap(generate_polygons_from__ccs_partial, cc_list_cover),
                                         total=len(cc_list_cover))]
        system_polygons = [poly for p_data in data for poly in p_data]

        polys_to_remove = []
        for ind1, poly in enumerate(system_polygons):
            for ind2, poly2 in enumerate(system_polygons):
                if ind1 != ind2:
                    if poly.contains(poly2):
                        polys_to_remove.append(ind2)

        for ind in reversed(polys_to_remove):
            del system_polygons[ind]

        for poly in system_polygons:
            poly_dict['system'].append(poly)

        if self.settings.debug:
            print('Generating debug image')
            c, ax = plt.subplots(1, 3, True, True)
            ax[0].imshow(img_data.image)
            for _poly in poly_dict['lyricssdf']:
                x, y = _poly.exterior.xy
                ax[0].plot(x, y)
            for _poly in poly_dict['system']:
                x, y = _poly.exterior.xy
                ax[0].plot(x, y)
            ax[1].imshow(t_region - region_prediction)
            ax[2].imshow(text_image)
            plt.show()
        return poly_dict


def generate_music_region(staff_polygons: List[Polygon], staffs: List[List[List[int]]], distance: float):
    poly_avg_height = [x.centroid.y for x in staff_polygons]

    def cluster(data, max_gap):
        # Arrange data into groups where successive elements
        # differ by no more than *max_gap*

        data.sort()
        groups = [[data[0]]]
        for x in data[1:]:
            if abs(x - groups[-1][-1]) <= max_gap:
                groups[-1].append(x)
            else:
                groups.append([x])
        return groups

    music_regions = []
    staff_regions = []
    for x in cluster(poly_avg_height, distance):
        region = []
        line_regions = []
        for y in x:
            region.append(staff_polygons[poly_avg_height.index(y)])
            line_regions.append(staffs[poly_avg_height.index(y)])
        music_regions.append(region)
        if len(line_regions) > 1:
            staff_regions.append(list(zip(*line_regions)))
        else:
            staff_regions.append(line_regions)
    music_region = [MusicRegion(x, y) for x, y in zip(music_regions, staff_regions)]
    music_regions = MusicRegions(music_region, staffs)
    return music_regions


def remove_polys_within_polys(polygons: List[Polygon]):
    polys_to_remove = []
    for ind1, poly in enumerate(polygons):
        for ind2, poly2 in enumerate(polygons):
            if ind1 != ind2:
                if poly2.contains(poly):
                    polys_to_remove.append(ind1)

    for ind in reversed(polys_to_remove):
        del polygons[ind]
    return polygons


def remove_polys_smaller_than_threshold(__polygons: List[Polygon], threshold: float):
    average_text_area = np.mean([x.area for x in __polygons])
    __polygons = [x for x in __polygons if x.area / average_text_area > threshold]
    return __polygons


def bbox1(img: np.ndarray):
    a = np.where(img != 0)
    bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
    return bbox


def bbox2(img: np.ndarray):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax


def create_data(path: str, line_space_height: int):
    space_height = line_space_height
    if line_space_height == 0:
        space_height = vertical_runs(binarize(np.array(Image.open(path)) / 255))[0]
    image_data = ImageData(path=path, height=space_height)
    return image_data


def generate_polygons_from__ccs(cc, alpha: int = 15, buffer_size: float = 1.0):
    points = np.array(list(chain.from_iterable(cc)))
    if len(points) < 4:
        return []
    edges = alpha_shape(points, alpha)

    # edges2, polys = alpha_shape_numpy(points, 0.05)
    polys = polygons(edges)
    
    polys = [np.flip(points[poly], axis=1) for poly in polys]
    polygons_paths = []
    for poly in polys:
        poly = Polygon(poly)
        poly_buffered = poly.buffer(buffer_size)
        poly_buffered = poly_buffered.simplify(1, preserve_topology=False)
        if poly_buffered.geom_type == 'MultiPolygon':
            poly_buffered = poly
        polygons_paths.append(poly_buffered)

    return polygons_paths


def generate_polygon_from_staff(staff: List[List[int]]):
    first_line = list(map(list, staff[0]))

    first_line[0][1] = first_line[0][1] + -5

    last_line = list(map(list, staff[-1]))
    last_line[0][1] = last_line[0][1] + -5
    last_line.reverse()
    _polygon = first_line + last_line
    y, x = zip(*_polygon)

    _polygon = list(zip(x, y))
    return Polygon(_polygon)


def draw_polygons(_polygons: List[Polygon], polygon_img: np.ndarray):
    for _poly in _polygons:
        x, y = _poly.exterior.xy
        rr, cc = polygon(y, x)
        polygon_img[rr, cc] = 1
    return polygon_img


def polygons(edges: List[int]):
    # Generates polygons from Delaunay edges
    if len(edges) == 0:
        return []

    edges = list(edges.copy())

    shapes = []

    initial = edges[0][0]
    current = edges[0][1]
    points = [initial]
    del edges[0]
    while len(edges) > 0:
        found = False
        for idx, (i, j) in enumerate(edges):
            if i == current:
                points.append(i)
                current = j
                del edges[idx]
                found = True
                break
            if j == current:
                points.append(j)
                current = i
                del edges[idx]
                found = True
                break

        if not found:
            shapes.append(points)
            initial = edges[0][0]
            current = edges[0][1]
            points = [initial]
            del edges[0]

    if len(points) > 1:
        shapes.append(points)

    return shapes


if __name__ == "__main__":
    import pickle
    import os
    from layoutanalysis.preprocessing.util import vertical_runs
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_dir, 'demo/models/model')
    page_path = os.path.join(project_dir, 'demo/images/Graduel_de_leglise_de_Nevers-509.nrm.png')
    staff_path = os.path.join(project_dir, 'demo/staffs/Graduel_de_leglise_de_Nevers-509.staffs')
    text_extractor_settings = SegmentationSettings(debug=True)
    with open(staff_path, 'rb') as f:
        _staffs = pickle.load(f)
    t_callback = SegmentationDummyCallback()
    text_extractor = Segmentator(text_extractor_settings, t_callback)
    for _ in text_extractor.segment([_staffs], [page_path]):
        pass
