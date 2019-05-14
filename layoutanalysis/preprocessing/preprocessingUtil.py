import cv2
from collections import defaultdict
import numpy as np
#import peakutils
from scipy.signal import medfilt2d
from itertools import tee
from scipy.ndimage.filters import convolve1d


def extract_connected_components(image):
    connectivity = 8
    output = cv2.connectedComponentsWithStats(image, connectivity)
    ccdict = defaultdict(list)
    indexdim0, indexdim1 = np.array(output[1]).nonzero()
    points = list(zip(indexdim0, indexdim1))
    for p in points:
        y_coord, x_coord = p[0], p[1]
        k = output[1][y_coord][x_coord]
        ccdict[k].append([y_coord, x_coord])
    cc_list = [ccdict[k] for k in sorted(ccdict.keys())]
    # skip first element of centroid and stats, since it is not a cc, but information related to the image
    labels = output[2][1:]
    centroids = output[3][1:]
    return cc_list, labels, centroids


def normalize_connected_components(cc_list):
    # Normalize the CCs (line segments), so that the height of each cc is normalized to one pixel
    def normalize(point_list):
        normalized_cc_list = []
        for cc in point_list:
            cc_dict = defaultdict(list)
            for y, x in cc:
                cc_dict[x].append(y)
            normalized_cc = []
            for key, value in cc_dict.items():
                normalized_cc.append([int(np.floor(np.mean(value) + 0.5)), key])
            normalized_cc_list.append(normalized_cc)
        return normalized_cc_list
    return normalize(cc_list)


def convert_2dpoint_to_1did(list, width):
    point_to_id = list[1] * width + list[0]
    return point_to_id


def convert_2darray_to_1darray(array, width):
    return array[:, 0] * width + array[:, 1]


def get_text_borders(image, preprocess=False, min_dist=30, thres=0.3):
    med = image.copy()
    if preprocess:
        med = medfilt2d(image, 9)
    histogram = np.sum(med == 255, axis=1)
    text_borders = peakutils.indexes(histogram, thres=thres, min_dist=min_dist)
    return text_borders


def vertical_runs(img: np.array):
    img = np.transpose(img)
    h = img.shape[0]
    w = img.shape[1]
    transitions = np.transpose(np.nonzero(np.diff(img)))
    white_runs = [0] * (w + 1)
    black_runs = [0] * (w + 1)
    a, b = tee(transitions)
    next(b, [])
    for f, g in zip(a, b):
        if f[0] != g[0]:
            continue
        tlen = g[1] - f[1]
        if img[f[0], f[1] + 1] == 1:
            white_runs[tlen] += 1
        else:
            black_runs[tlen] += 1

    for y in range(h):
        x = 1
        col = img[y, 0]
        while x < w and img[y, x] == col:
            x += 1
        if col == 1:
            white_runs[x] += 1
        else:
            black_runs[x] += 1

        x = w - 2
        col = img[y, w - 1]
        while x >= 0 and img[y, x] == col:
            x -= 1
        if col == 1:
            white_runs[w - 1 - x] += 1
        else:
            black_runs[w - 1 - x] += 1
    black_r = np.argmax(black_runs) + 1
    # on pages with a lot of text the staffspaceheight can be falsified.
    # --> skip the first elements of the array
    white_r = np.argmax(white_runs) + 1

    img = np.transpose(img)
    return white_r, black_r


def calculate_horizontal_runs(img: np.array, min_length: int):
    h = img.shape[0]
    w = img.shape[1]
    np_matrix = np.zeros([h, w], dtype=np.uint8)
    t = np.transpose(np.nonzero(np.diff(img) == -1))
    for trans in t:
        y, x = trans[0], trans[1] + 1
        xo = x
        # rl = 0
        while x < w and img[y, x] == 0:
            x += 1
        rl = x - xo
        if rl >= min_length:
            for x in range(xo, xo + rl):
                np_matrix[y, x] = 255
    return np_matrix

def box_blur(img, radiusc, radiusr):
    filterr = np.ones(radiusr * 1) / radiusr
    filterc = np.ones(radiusc * 1) / radiusc
    image = convolve1d(img, filterr, axis = 0)
    image = convolve1d(image, filterc, axis = 1)
    return image

if __name__ == "__main__":
    l = np.array([[1,2],[2,4], [2,1]])
    print(convert_2darray_to_1darray(l))
