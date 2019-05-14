from shapely.geometry import Polygon
from scipy.spatial import Delaunay
import numpy as np


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


def alpha_shape_numpy(points, alpha):
    from shapely import geometry
    from shapely.ops import cascaded_union, polygonize
    """
    Compute the alpha shape (concave hull) of a set
    of points.
    @param points: Iterable container of points.
    @param alpha: alpha value to influence the
        gooeyness of the border. Smaller numbers
        don't fall inward as much as larger numbers.
        Too large, and you lose everything!
    """
    if len(points) < 4:
        # When you have a triangle, there is no sense
        # in computing an alpha shape.
        return geometry.MultiPoint(list(points)).convex_hull

    coords = np.array(points)
    tri = Delaunay(coords)
    triangles = coords[tri.vertices]
    a = ((triangles[:, 0, 0] - triangles[:, 1, 0]) ** 2 + (triangles[:, 0, 1] - triangles[:,1,1]) ** 2) ** 0.5
    b = ((triangles[:, 1, 0] - triangles[:, 2, 0]) ** 2 + (triangles[:, 1, 1] - triangles[:,2,1]) ** 2) ** 0.5
    c = ((triangles[:, 2, 0] - triangles[:, 0, 0]) ** 2 + (triangles[:, 2, 1] - triangles[:,0,1]) ** 2) ** 0.5
    s = ( a + b + c ) / 2.0
    areas = (s*(s-a)*(s-b)*(s-c)) ** 0.5
    circums = a * b * c / (4.0 * areas)
    filtered = triangles[circums < (1.0 / alpha)]
    edge1 = filtered[:, (0, 1)]
    edge2 = filtered[:, (1, 2)]
    edge3 = filtered[:, (2, 0)]
    edge_points = np.unique(np.concatenate((edge1,edge2,edge3)), axis = 0).tolist()

    m = geometry.MultiLineString(edge_points)

    triangles = list(polygonize(m))

    return edge_points, triangles


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

    return False