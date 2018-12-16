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
from skimage.measure import approximate_polygon
from collections import defaultdict
from skimage.draw import polygon
import multiprocessing
import tqdm
from functools import partial
import itertools as IT

@dataclass
class TextExtractionSettings:
    erode: bool = False
    debug: bool = False
    lineSpaceHeight: int = 20
    targetLineSpaceHeight: int = 10
    model: [str] = None
    cover: float = 0.1
    processes: int = 12


class TextExtractor:
    def __init__(self, settings: TextExtractionSettings):
        self.predictor = None
        self.settings = settings
        if self.settings.model:
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
        for i, pred in enumerate(self.predictor.predict(data)):
            yield self.segmentate_image(staffs[i], data[i], pred)

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

        processed_img = np.clip(img_with_staffs_removed + staff_img, 0, 1).astype(np.uint8)
        cc_list = extract_connected_components((1 - processed_img) * 255)
        cc_list_prediction = extract_connected_components(t_region - region_prediction)
        cc_list_prediction = [x for x in cc_list_prediction if len(x) > 100]
        cc_list = cc_cover(np.array(cc_list), np.array(cc_list_prediction), self.settings.cover)
        poly_dict = defaultdict(list)
        for ccs in cc_list:
            polys = generate_polygons_from__ccs(ccs, 2)
            poly_dict['text'].append(polys)

        text_image = np.zeros(img_data.image.shape)
        for _polys in poly_dict['text']:
            text_image = draw_polygons(_polys, text_image)
        image_with_text_removed = np.clip(img_data.image + text_image, 0, 1)
        cc_list1 = extract_connected_components(((1 - image_with_text_removed) * 255).astype(np.uint8))
        cc_list1 = [cc for cc in cc_list1 if len(cc) > 10]
        cc_list2 = extract_connected_components((staff_img * 255).astype(np.uint8))
        cc_list_cover = cc_cover(np.array(cc_list1), np.array(cc_list2), self.settings.cover, use_pred_as_start=True)
        for ccs in cc_list_cover:
            polys = generate_polygons_from__ccs(ccs, 2)
            poly_dict['system'].append(polys)
        if self.settings.debug:
            print('Generating debug image')
            c, ax = plt.subplots(1, 3, True, True)
            ax[0].imshow(img_data.image)
            for _poly in poly_dict['text']:
                for z in _poly:
                    ax[0].plot(z[:, 0], z[:, 1])
            for _poly in poly_dict['system']:
                for z in _poly:
                    ax[0].plot(z[:, 0], z[:, 1])
            ax[1].imshow(staff_img)
            ax[2].imshow(text_image)
            plt.show()
        return poly_dict


def cc_cover(cc_list, pred_list, cover=0.1, img_width=10000, use_pred_as_start=False):
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
            if abs(cc_medium_height - pred_cc_medium_height) > 30:
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


def create_data(path, line_space_height):
    space_height = line_space_height
    if line_space_height == 0:
        space_height = vertical_runs(binarize(np.array(Image.open(path)) / 255))[0]
    image_data = ImageData(path=path, height=space_height)
    return image_data


def generate_polygons_from__ccs(cc, offset=1, alpha=15):
    points = np.array(list(chain.from_iterable(cc)))
    edges = alpha_shape(points, alpha)
    polys = polygons(edges)
    polys = [np.flip(points[poly], axis=1) for poly in polys]
    polys = [scale_polygon(approximate_polygon(p, 1), offset) for p in polys]
    return polys


def generate_polygon_from_staff(staff):
    first_line = staff[0]
    last_line = staff[-1]
    last_line.reverse()
    _polygon = first_line + last_line
    y, x = zip(*_polygon)
    _polygon = list(zip(x, y))
    return _polygon


def draw_polygons(_polygons, polygon_img):
    for _poly in _polygons:
        x, y = zip(*_poly)
        rr, cc = polygon(y, x)
        polygon_img[rr, cc] = 1
    return polygon_img


def scale_polygon(p_path,offset):
    center = centroid_of_polygon(p_path)
    for i in p_path:
        if i[0] > center[0]:
            i[0] += offset
        else:
            i[0] -= offset
        if i[1] > center[1]:
            i[1] += offset
        else:
            i[1] -= offset
    return p_path


def area_of_polygon(x, y):
    """Calculates the signed area of an arbitrary polygon given its verticies
    http://stackoverflow.com/a/4682656/190597 (Joe Kington)
    http://softsurfer.com/Archive/algorithm_0101/algorithm_0101.htm#2D%20Polygons
    """
    area = 0.0
    for i in range(-1, len(x) - 1):
        area += x[i] * (y[i + 1] - y[i - 1])
    return area / 2.0


def centroid_of_polygon(points):
    """
    http://stackoverflow.com/a/14115494/190597 (mgamba)
    """
    area = area_of_polygon(*zip(*points))
    result_x = 0
    result_y = 0
    N = len(points)
    points = IT.cycle(points)
    x1, y1 = next(points)
    for i in range(N):
        x0, y0 = x1, y1
        x1, y1 = next(points)
        cross = (x0 * y1) - (x1 * y0)
        result_x += (x0 + x1) * cross
        result_y += (y0 + y1) * cross
    result_x /= (area * 6.0)
    result_y /= (area * 6.0)
    return (result_x, result_y)


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


if __name__ == "__main__":
    import os
    import pickle
    from layoutanalysis.preprocessing.binarization.ocropus_binarizer import binarize
    from layoutanalysis.preprocessing.preprocessingUtil import vertical_runs

    text_extractor_settings = TextExtractionSettings(debug=True, cover=0.01, erode=True,
                                                     model='/home/alexanderh/Schreibtisch/git/data/models/' \
                                                       'textprediction_interesting/model2')
    _path = '/home/alexanderh/Schreibtisch/masterarbeit/OMR/Graduel_de_leglise_de_Nevers/interesting/' \
           'part2/bin/Graduel_de_leglise_de_Nevers-509.nrm.png'
    with open('/home/alexanderh/Schreibtisch/git/data/staffs/staffs_data509.pickle', 'rb') as f:
        _staffs = pickle.load(f)
    text_extractor = TextExtractor(text_extractor_settings)
    for _ in text_extractor.segmentate([_staffs], [_path]):
        pass

