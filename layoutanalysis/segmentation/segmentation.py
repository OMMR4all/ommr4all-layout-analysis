from layoutanalysis.pixelclassifier.predictor import PCPredictor
from pagesegmentation.lib.predictor import PredictSettings
from layoutanalysis.removal.dummy_staff_line_removal import staff_removal
from layoutanalysis.preprocessing.preprocessingUtil import extract_connected_components, convert_2darray_to_1darray
from PIL import Image
import matplotlib.pyplot as plt
from itertools import chain
from scipy.spatial import Delaunay
import numpy as np
from scipy.ndimage.morphology import binary_erosion, binary_dilation
from layoutanalysis.datatypes.datatypes import ImageData
from dataclasses import dataclass
from collections import defaultdict
from skimage.draw import polygon
import multiprocessing
import tqdm
from functools import partial
from layoutanalysis.preprocessing.binarization.ocropus_binarizer import binarize
from shapely.geometry import Polygon
from shapely import affinity
from scipy.ndimage.filters import convolve1d
from scipy.interpolate import interpolate
import math


@dataclass
class SegmentationSettings:
    erode: bool = False
    debug: bool = False
    lineSpaceHeight: int = 20
    targetLineSpaceHeight: int = 10
    model: [str] = None
    cover: float = 0.1
    processes: int = 12


class Segmentator:
    def __init__(self, settings: SegmentationSettings):
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

    def segmentate(self, staffs, img_paths):
        create_data_partital = partial(create_data, line_space_height=self.settings.lineSpaceHeight)
        with multiprocessing.Pool(processes=self.settings.processes) as p:
            data = [v for v in tqdm.tqdm(p.imap(create_data_partital, img_paths), total=len(img_paths))]
        if self.settings.model:
            for i, pred in enumerate(self.predictor.predict(data)):
                yield self.segmentate_image(staffs[i], data[i], pred)
        else:
            for i_ind, i in enumerate(zip(staffs, data)):
                yield self.segmentate_with_weight_image(i[0], i[1])

    def segmentate_basic(self, staffs, img_data):

        poly_dict = defaultdict(list)

        img = np.array(Image.open(img_data.path)) / 255
        img_data.image = binarize(img)
        binarized = 1 - img_data.image
        if self.settings.erode:
            img_data.image = binary_dilation(img_data.image, structure=np.full((1, 3), 1))
        staff_image = np.zeros(img_data.image.shape)
        staff_polygons = [generate_polygon_from_staff(staff) for staff in staffs]
        staff_img = draw_polygons(staff_polygons, staff_image)
        staff_cc = extract_connected_components((staff_img * 255).astype(np.uint8))[0]

        rmin, rmax, cmin, cmax = bbox2(staff_img)
        rmin = rmin - rmin // 10
        rmax = rmax + (staff_img.shape[0] - rmax) // 5
        cmin = cmin - cmin // 10
        cmax = cmax + (staff_img.shape[1] - cmax) // 5

        bbox = img_data.image[rmin:rmax, cmin:cmax]
        processed_image = np.ones(img_data.image.shape)
        processed_image[rmin:rmax, cmin:cmax] = bbox
        processed_image = staff_removal(staffs, processed_image, 3)

        cc_list = extract_connected_components(((1 - processed_image) * 255).astype(np.uint8))[0]
        cc_list_cover = cc_cover(np.array(cc_list), np.array(staff_cc), self.settings.cover, use_pred_as_start=True)
        generate_polygons_from__ccs_partial = partial(generate_polygons_from__ccs, yscale=1.03)
        with multiprocessing.Pool(processes=self.settings.processes) as p:
            data = [v for v in tqdm.tqdm(p.imap(generate_polygons_from__ccs_partial, cc_list_cover), total=len(cc_list_cover))]
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

        text_image = np.zeros(img_data.image.shape)
        text_image = draw_polygons(poly_dict['system'], text_image)

        processed_image = np.clip(processed_image + text_image, 0, 1)
        processed_image_cc = extract_connected_components(((1 - processed_image) * 255).astype(np.uint8))[0]
        processed_image_cc = [cc for cc in processed_image_cc if len(cc) > 10]
        data = generate_polygons_from__ccs(processed_image_cc)
        text_polygons = [poly for poly in data]

        polys_to_remove = []
        for ind1, poly in enumerate(text_polygons):
            for ind2, poly2 in enumerate(text_polygons):
                if ind1 != ind2:
                    if poly.contains(poly2):
                        polys_to_remove.append(ind2)

        for ind in reversed(polys_to_remove):
            del text_polygons[ind]
        for poly in text_polygons:
            poly_dict['text'].append(poly)

        if self.settings.debug:
            print('Generating debug image')
            c, ax = plt.subplots(1, 3, True, True)
            ax[0].imshow(img_data.image)
            for _poly in poly_dict['text']:
                x, y = _poly.exterior.xy
                ax[0].plot(x, y)
            for _poly in poly_dict['system']:
                x, y = _poly.exterior.xy
                ax[0].plot(x, y)
            ax[1].imshow(staff_img)
            ax[2].imshow(text_image)
            plt.show()
        return poly_dict

    def segmentate_with_weight_image(self, staffs, img_data):
        from scipy.ndimage import gaussian_filter
        poly_dict = defaultdict(list)

        img = np.array(Image.open(img_data.path)) / 255
        img_data.image = binarize(img)
        binarized = 1 - img_data.image
        if self.settings.erode:
            img_data.image = binary_dilation(img_data.image, structure=np.full((1, 3), 1))
        staff_image = np.zeros(img_data.image.shape)

        staff_polygons = [generate_polygon_from_staff(staff) for staff in staffs]
        distance = []
        for x in range(len(staff_polygons)-1):
            distance.append(staff_polygons[x].distance(staff_polygons[x+1]))

        staff_img = draw_polygons(staff_polygons, staff_image)
        weight = gaussian_filter(staff_img, sigma=(np.average(distance) / 4, np.average(distance) * 3 / 4))
        #weight = box_blur(staff_img, 130, 48)
        weight = np.clip(weight * 2, 0, 1) * 255 - 125

        rmin, rmax, cmin, cmax = bbox2(staff_img)
        rmin = rmin - rmin // 10
        rmax = rmax + (staff_img.shape[0] - rmax) // 5
        cmin = cmin - cmin // 10
        cmax = cmax + (staff_img.shape[1] - cmax) // 5

        bbox = img_data.image[rmin:rmax, cmin:cmax]
        processed_image = np.ones(img_data.image.shape)
        processed_image[rmin:rmax, cmin:cmax] = bbox
        processed_image = staff_removal(staffs, processed_image, 3)

        cc_list_with_stats = extract_connected_components(((1 - processed_image) * 255).astype(np.uint8))

        def get_cc(cc_list, cc_list_stats, cc_list_centroids, weight_matrix, avg_distance):
            initials = []
            cc_list_new = []
            for cc_ind, cc in enumerate(cc_list):
                y, x = zip(*cc)
                if cc_list_stats[cc_ind, 3] > avg_distance * 1.5:
                    initials.append(cc)
                    continue
                if np.sum(weight_matrix[y, x]) > 0:
                    #print(cc_list_stats[cc_ind, 3])

                    cc_list_new.append(cc_ind)
            return [cc_list[i] for i in cc_list_new], [cc_list_stats[i] for i in cc_list_new], [cc_list_centroids[i] for i in cc_list_new], initials

        cc_list_with_stats = get_cc(cc_list_with_stats[0], cc_list_with_stats[1], cc_list_with_stats[2], weight, np.average(distance))
        initials = cc_list_with_stats[3]
        initials_polygons = []
        if len(initials) > 0:
            initials_polygons = generate_polygons_from__ccs(initials)
        for poly in initials_polygons:
            poly_dict['initials'].append(poly)
        if self.settings.debug:
            print('Generating debug image')

            def visulize_cc_list(cc_list, img_shape):
                cc_image = np.ones(img_shape)
                for cc in cc_list:
                    y, x = zip(*cc)
                    cc_image[y, x] = 0
                return cc_image

            z, ax = plt.subplots(1, 4, True, True)
            ax[0].imshow(weight)
            ax[1].imshow(staff_img)
            ax[2].imshow(visulize_cc_list(cc_list_with_stats[0], img_data.image.shape))
            ax[3].imshow(img_data.image)
            plt.show()
        generate_polygons_from__ccs_partial = partial(generate_polygons_from__ccs, yscale=1.03)

        def divide_ccs_into_groups(cc_list, staffs):
            cc_list_height = [cc[-1][0]+ cc[0][0] for cc in cc_list]
            staffs_height = [staff[0][0][0] + staff[-1][0][0] for staff in staffs]
            d = defaultdict(list)
            for cc_ind, cc in enumerate(cc_list_height):
                r = math.inf
                r_ind = - 1
                for index, staff in enumerate(staffs_height):
                    if abs(cc - staff) < r:
                        r = abs(cc - staff)
                        r_ind = index
                d[r_ind].append(cc_list[cc_ind])

            # add staff lines to groups
            for r_ind, staff in enumerate(staffs):
                for cc in staff:
                    y_list, x_list = zip(*cc)
                    func = interpolate.interp1d(x_list, y_list)
                    start = min(x_list)
                    end = max(x_list)
                    values = list(range(start, end))
                    fp = [func(x) for x in values]
                    d[r_ind].append(zip(fp, values))
            return d.values()

        cc_list = divide_ccs_into_groups(cc_list_with_stats[0], staffs)
        with multiprocessing.Pool(processes=self.settings.processes) as p:
            data = [v for v in tqdm.tqdm(p.imap(generate_polygons_from__ccs_partial, cc_list), total=len(cc_list))]
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

        text_image = np.zeros(img_data.image.shape)
        text_image = draw_polygons(poly_dict['system'], text_image)
        text_image = draw_polygons(poly_dict['initials'], text_image)

        processed_image = np.clip(processed_image + text_image, 0, 1)
        processed_image_cc = extract_connected_components(((1 - processed_image) * 255).astype(np.uint8))[0]
        processed_image_cc = [cc for cc in processed_image_cc if len(cc) > 10]
        data = generate_polygons_from__ccs(processed_image_cc)
        text_polygons = [poly for poly in data]

        polys_to_remove = []
        for ind1, poly in enumerate(text_polygons):
            for ind2, poly2 in enumerate(text_polygons):
                if ind1 != ind2:
                    if poly.contains(poly2):
                        polys_to_remove.append(ind2)

        for ind in reversed(polys_to_remove):
            del text_polygons[ind]
        for poly in text_polygons:
            poly_dict['text'].append(poly)
        if self.settings.debug:
            print('Generating debug image')
            c, ax = plt.subplots(1, 3, True, True)
            ax[0].imshow(img_data.image)
            for _poly in poly_dict['text']:
                x, y = _poly.exterior.xy
                ax[0].plot(x, y)
            for _poly in poly_dict['system']:
                x, y = _poly.exterior.xy
                ax[0].plot(x, y)
            for _poly in poly_dict['initials']:
                x, y = _poly.exterior.xy
                ax[0].plot(x, y)
            ax[1].imshow(staff_img)
            ax[2].imshow(text_image)
            plt.show()
        return poly_dict

    def segmentate_image(self, staffs, img_data, region_prediction):

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
        # charheight, systemheight = vertical_runs((t_region - region_prediction) //255)

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
            poly_dict['text'].append(poly)

        text_image = np.zeros(img_data.image.shape)
        text_image = draw_polygons(poly_dict['text'], text_image)

        image_with_text_removed = np.clip(img_data.image + text_image, 0, 1)
        cc_list1 = extract_connected_components(((1 - image_with_text_removed) * 255).astype(np.uint8))[0]
        cc_list1 = [cc for cc in cc_list1 if len(cc) > 10]
        cc_list2 = extract_connected_components((staff_img * 255).astype(np.uint8))[0]
        cc_list_cover = cc_cover(np.array(cc_list1), np.array(cc_list2), self.settings.cover, use_pred_as_start=True)

        generate_polygons_from__ccs_partial = partial(generate_polygons_from__ccs, yscale=1.03)
        with multiprocessing.Pool(processes=self.settings.processes) as p:
            data = [v for v in tqdm.tqdm(p.imap(generate_polygons_from__ccs_partial, cc_list_cover), total=len(cc_list_cover))]
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
            for _poly in poly_dict['text']:
                x, y = _poly.exterior.xy
                ax[0].plot(x, y)
            for _poly in poly_dict['system']:
                x, y = _poly.exterior.xy
                ax[0].plot(x, y)
            ax[1].imshow(t_region - region_prediction)
            ax[2].imshow(text_image)
            plt.show()
        return poly_dict


def cc_cover(cc_list, pred_list, cover=0.9, img_width=10000, use_pred_as_start=False):
    point_list = []
    ccs_medium_height = [np.mean(cc, axis=0)[0] for cc in cc_list]
    pred_cc_medium_heights = [np.mean(cc, axis=0)[0] for cc in pred_list]
    cc_list_array = [np.array(cc) for cc in cc_list]
    pred_list_array = [np.array(cc) for cc in pred_list]
    cc_1d_list = [convert_2darray_to_1darray(cc, img_width) for cc in cc_list_array]
    pred_1d_list = [convert_2darray_to_1darray(cc, img_width) for cc in pred_list_array]
    for pred_cc_indc, pred_cc in enumerate(pred_1d_list):
        pred_cc_medium_height = pred_cc_medium_heights[pred_cc_indc]
        pp_list = []
        if use_pred_as_start:
            pp_list.append(pred_list_array[pred_cc_indc])
        for cc_indc, cc in enumerate(cc_1d_list):
            cc_medium_height = ccs_medium_height[cc_indc]
            if abs(cc_medium_height - pred_cc_medium_height) > 50:
                continue
            C, pred_ind, cc_ind = np.intersect1d(pred_cc, cc, return_indices=True)
            if cc_ind.size != 0:
                if float(len(cc_ind)) / float(len(cc)) > cover:
                    pp_list.append(cc_list_array[cc_indc])
                else:
                    pred_cc_arr = pred_list_array[pred_cc_indc]
                    pp_list.append(pred_cc_arr[pred_ind])
        if len(pp_list) > 0:
            point_list.append(pp_list)
    return point_list


def bbox1(img):
    a = np.where(img != 0)
    bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
    return bbox


def bbox2(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax


def create_data(path, line_space_height):
    space_height = line_space_height
    if line_space_height == 0:
        space_height = vertical_runs(binarize(np.array(Image.open(path)) / 255))[0]
    image_data = ImageData(path=path, height=space_height)
    return image_data


def generate_polygons_from__ccs(cc, alpha=15, xscale=1.001, yscale=1.1):
    points = np.array(list(chain.from_iterable(cc)))
    edges = alpha_shape(points, alpha)
    polys = polygons(edges)
    polys = [np.flip(points[poly], axis=1) for poly in polys]
    polygons_paths = []
    for poly in polys:
        poly = Polygon(poly)
        poly = poly.simplify(0.8)
        poly = affinity.scale(poly, xscale, yscale, origin='centroid')
        polygons_paths.append(poly)
    return polygons_paths


def generate_polygon_from_staff(staff):
    first_line = staff[0]
    first_line[0][1] = first_line[0][1] + -5

    last_line = staff[-1]
    last_line[0][1] = last_line[0][1] + -5
    last_line.reverse()
    _polygon = first_line + last_line
    y, x = zip(*_polygon)

    _polygon = list(zip(x, y))
    return Polygon(_polygon)


def draw_polygons(_polygons, polygon_img):
    for _poly in _polygons:
        x, y = _poly.exterior.xy
        rr, cc = polygon(y, x)
        polygon_img[rr, cc] = 1
    return polygon_img


def polygons(edges):
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


def alpha_shape(points, alpha, only_outer=True):
    """
    Compute the alpha shape (concave hull) of a set of points.
    :param points: np.array of shape (n,2) points.
    :param alpha: alpha value.
    :param only_outer: boolean value to specify if we keep only the outer border
    or also inner edges.
    :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
    the indices in the points array.
    """
    assert points.shape[0] > 3, "Need at least four points"

    def add_edge(edges, i, j):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            assert (j, i) in edges, "Can't go twice over same directed edge right?"
            if only_outer:
                # if both neighboring triangles are in shape, it's not a boundary edge
                edges.remove((j, i))
            return
        edges.add((i, j))

    tri = Delaunay(points)
    edges = set()
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.vertices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)

        if circum_r < alpha:
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)
    return edges


def check_polygon_within_polygon(poly1, poly2):
    import matplotlib.path as mpltPath
    def inside_polygon(x, y, points):
        """
        Return True if a coordinate (x, y) is inside a polygon defined by
        a list of verticies [(x1, y1), (x2, x2), ... , (xN, yN)].

        Reference: http://www.ariel.com.au/a/python-point-int-poly.html
        """
        n = len(points)
        inside = False
        p1x, p1y = points[0]
        for i in range(1, n + 1):
            p2x, p2y = points[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    t = check_polygon_intersection(poly1, poly2)
    if not t:
        path = mpltPath.Path(poly2)
        inside = path.contains_point([poly1[0][0], poly1[0][1]])
        if inside:
            return True
        #if inside_polygon(poly1[0][0], poly1[0][1], poly2):
        #    return True
    return False


def check_polygon_intersection(p1, p2):

    def check_line_intersects_polygon(l1, p2):
        for t in range(len(p2)-1):
            if check_for_intersection(l1, [p2[t], p2[t+1]]):
                return True

        return False
    intersection = False
    for i in range(len(p1)-1):
        if check_line_intersects_polygon([p1[i], p1[i+1]], p2):
            intersection = True
            break

    return intersection


def check_for_intersection(l1, l2):
    p1 = l1[0]
    p2 = l1[1]
    p3 = l2[0]
    p4 = l2[1]
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]
    x3, y3 = p3[0], p3[1]
    x4, y4 = p4[0], p4[1]

    denominator = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
    if denominator == 0:
        return False
    t = ((x1 - x3)*(y3 - y4) - (y1 - y3)*(x3 - x4)) / denominator
    u = -(((x1-x2)*(y1 - y3) - (y1-y2)*(x1-x3)) / denominator)
    if 0 <= u <= 1 and 0 <= t <= 1:
        return True

    #px = x1 + t*(x2 - x1)
    #py = y1 + t*(y2 - y1)
    return False


def box_blur(img, radiusc, radiusr):
    filterr = np.ones(radiusr * 1) / radiusr
    filterc = np.ones(radiusc * 1) / radiusc
    image = convolve1d(img, filterr, axis = 0)
    image = convolve1d(image, filterc, axis = 1)
    return image

if __name__ == "__main__":
    import pickle
    import os
    from layoutanalysis.preprocessing.preprocessingUtil import vertical_runs

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_dir, 'demo/models/model')
    page_path = os.path.join(project_dir, 'demo/images/Graduel_de_leglise_de_Nevers-509.nrm.png')
    staff_path = os.path.join(project_dir, 'demo/staffs/Graduel_de_leglise_de_Nevers-509.staffs')
    text_extractor_settings = SegmentationSettings(debug=True, cover=0.3, erode=False)\
     #   ,                                                     model=model_path)
    with open(staff_path, 'rb') as f:
        _staffs = pickle.load(f)
    text_extractor = Segmentator(text_extractor_settings)
    for _ in text_extractor.segmentate([_staffs], [page_path]):
        pass
