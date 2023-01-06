import copy
import os
import pickle
import random as random
import time
from math import *

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gnutools.fs import name, parent
from scipy import ndimage

x, y, z = 0, 1, 2
r, g, b = 0, 1, 2
xyz = [x, y, z]
rgb = [r, g, b]



def dist(v1, v2):
    """
    Euclidian distance between two vectors

    :param v1:
    :param v2:
    :return:
    """
    return np.abs(v1 - v2)


def bounding_box(vertices, r):
    """
    Bounding box of a list of vertices

    :param vertices:
    :param r:
    :return:
    """
    args = np.argwhere(((np.prod(vertices <= r[1], axis=1)) & (
        np.prod(vertices >= r[0], axis=1)))).reshape(-1, )
    return vertices[args]


def ranges(vertices, scale=np.ones(3)):
    """
    Get the  bounding box of vertices

    :param vertices:
    :param scale:
    :return:
    """
    minimums = np.min(vertices, axis=0)
    maximums = np.max(vertices, axis=0)
    lengths = (maximums - minimums)

    r = np.array([minimums - ((scale - 1) * lengths),
                 maximums + ((scale - 1) * lengths)])
    return r


def length(vertices):
    """
    Get the length of vertices

    :param vertices:
    :return:
    """
    minimums = np.min(vertices, axis=0)
    maximums = np.max(vertices, axis=0)
    return (maximums - minimums)


def center(vertices):
    """
    Translate the vertices to the origin

    :param vertices:
    :return:
    """
    T = -np.mean(ranges(vertices), axis=0)
    return [vertices + T, T]


def center_mean(vertices):
    """
    Translate by the center of mass of the vertices

    :param vertices:
    :return:
    """
    trans = -np.mean(vertices, axis=0)
    vertices += trans
    return [vertices, trans]


def translate(vertices, translation):
    """
    Translate vertices

    :param vertices:
    :param translation:
    :return:
    """
    if vertices is None:
        return None
    return vertices + translation


def crop_bounding_box(vertices, r):
    """
    Reduce a list of vertices to a specific region in the space

    :param r: region of reference to crop the mesh
    :return:
    """
    vinds = np.argwhere((np.min(vertices > r[0], axis=1).flatten()) & (
        np.min(vertices < r[1], axis=1).flatten())).flatten()
    return vertices[vinds]


def crop(vertices, axis=z, borne_inf=0, borne_sup=1, t_min=None, t_max=None, return_inds=False):
    """
    Reduce vertices to specific region

    :param vertices:
    :param axis:
    :param borne_inf:
    :param borne_sup:
    :param t_min:
    :param t_max:
    :param return_inds:
    :return:
    """
    # [x, y, z] = [0, 1, 2]
    # Restriction to the top of the base
    [minimums, maximums] = ranges(vertices)[:, axis]
    if t_min is None:
        t_min = (minimums + ((maximums - minimums) * borne_inf))
    if t_max is None:
        t_max = (minimums + ((maximums - minimums) * borne_sup))
    inds = np.where((vertices[:, axis] >= t_min) &
                    (vertices[:, axis] <= t_max))
    if return_inds:
        return vertices[inds], inds[0]
    else:
        return vertices[inds]


def rotateMatrix(vertices, M):
    """
    Rotate vertices by a given matrix

    :param vertices:
    :param M:
    :return:
    """
    return np.dot(M, vertices.transpose()).transpose()


def rotate(vertices, theta=0, deg=None, axis_rotation=z, origin=np.zeros(3), degs=None):
    """
    Rotate vertices

    :param vertices:
    :param theta:
    :param deg:
    :param axis_rotation:
    :param origin:
    :param degs:
    :return:
    """
    centroid = np.mean(ranges(vertices), axis=0)
    vertices = translate(vertices, -origin)
    # start_time = time.time()
    if degs is None:
        if deg is not None:
            theta = (deg / 180) * pi
        if axis_rotation == 0:
            T = np.array([
                [1, 0, 0],
                [0, cos(theta), sin(theta)],
                [0, -sin(theta), cos(theta)]
            ])
        elif axis_rotation == 1:
            T = np.array([
                [cos(theta), 0, -sin(theta)],
                [0, 1, 0],
                [sin(theta), 0, cos(theta)]
            ])
        else:
            T = np.array([
                [cos(theta), sin(theta), 0],
                [-sin(theta), cos(theta), 0],
                [0, 0, 1]
            ])
        vertices = np.dot(vertices, T)
    else:
        T = []
        for deg in degs:
            theta = (deg / 180) * pi
            if axis_rotation == 0:
                T.append(np.array([
                    [1, 0, 0],
                    [0, cos(theta), sin(theta)],
                    [0, -sin(theta), cos(theta)]
                ]))
            elif axis_rotation == 1:
                T.append(np.array([
                    [cos(theta), 0, -sin(theta)],
                    [0, 1, 0],
                    [sin(theta), 0, cos(theta)]
                ]))
            else:
                T.append(np.array([
                    [cos(theta), sin(theta), 0],
                    [-sin(theta), cos(theta), 0],
                    [0, 0, 1]
                ]))
        T = np.array(T)
        vertices = np.array([vertices] * len(degs))
        vertices = np.matmul(vertices, T)
    # print("[rotate]\t--- %s seconds ---" % (time.time() - start_time))
    if origin is not None:
        vertices = translate(vertices, origin)
    return vertices


def display_image(im, wait=1, factor=5, title=None):
    """
    Display an image
    :param im:
    :param wait:
    :param factor:
    :param title:
    :return:
    """
    if title is None:
        title = "display_image"
    if len(im.shape) == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)

    cv2.imshow(title, cv2.resize(im, None, fx=factor,
               fy=factor, interpolation=cv2.INTER_CUBIC))
    cv2.waitKey(wait)
    return


def rotate_image(im=None, deg=0):
    """

    :param im:
    :param deg:
    :return:
    """
    if im is None:
        return im
    else:
        M = cv2.getRotationMatrix2D((im.shape[1] / 2, im.shape[0] / 2), deg, 1)
        return cv2.warpAffine(im, M, (max(im.shape), max(im.shape)))


def export_pickle(file, filename):
    """

    :param file:
    :param filename:
    :return:
    """
    with open(filename, "wb") as f:
        pickle.dump(file, f)


def saveAll(fv_tooth=None, fv_base=None, fv_upper=None, fv_lower=None, fv_left=None, fv_front=None, fv_right=None,
            fv_tmp=None, dir="/tmp/", ext=""):
    """

    :param fv_tooth:
    :param fv_base:
    :param fv_upper:
    :param fv_lower:
    :param fv_left:
    :param fv_front:
    :param fv_right:
    :param fv_tmp:
    :param dir:
    :param ext:
    :return:
    """
    if fv_tmp is not None:
        if not ext == "ply":
            export_pickle(fv_tmp, dir + "/tmp.pickle")
        fv_tmp.export_mesh(
            filename=dir + "/tmp.ply")
    if fv_tooth is not None:
        fv_tooth.mlx(name="taubin")
        if not ext == "ply":
            export_pickle(fv_tooth, dir + "/tooth.pickle")
        fv_tooth.export_mesh(
            filename=dir + "/tooth.ply")
    if fv_base is not None:
        if not ext == "ply":
            export_pickle(fv_base, dir + "/base.pickle")
        fv_base.export_mesh(
            filename=dir + "/base.ply")
    if fv_lower is not None:
        if not ext == "ply":
            export_pickle(fv_lower, dir + "/lower.pickle")
        fv_lower.export_mesh(
            filename=dir + "/lower.ply")
    if fv_upper is not None:
        if not ext == "ply":
            export_pickle(fv_upper, dir + "/upper.pickle")
        fv_upper.export_mesh(
            filename=dir + "/upper.ply")
    if fv_left is not None:
        if not ext == "ply":
            export_pickle(fv_left, dir + "/left.pickle")
        fv_left.export_mesh(
            filename=dir + "/left.ply")
    if fv_right is not None:
        if not ext == "ply":
            export_pickle(fv_right, dir + "/right.pickle")
        fv_right.export_mesh(
            filename=dir + "/right.ply")
    if fv_front is not None:
        if not ext == "ply":
            export_pickle(fv_front, dir + "/front.pickle")
        fv_front.export_mesh(
            filename=dir + "/front.ply")


def execute_ops(vertices, operations):
    """

    :param vertices:
    :param operations:
    :return:
    """
    if vertices is None:
        return None
    for operation in operations:
        if list(operation.keys())[0] == "rotation":
            rotation = operation["rotation"]
            axis = rotation["axis"]
            theta = rotation["theta"]
            vertices = rotate(vertices=vertices,
                              axis_rotation=axis, theta=theta)
        elif list(operation.keys())[0] == "rotation_matrix":
            M = operation["rotation_matrix"]
            vertices = rotateMatrix(vertices=vertices, M=M)
        elif list(operation.keys())[0] == "translation":
            trans = operation["translation"]
            vertices = translate(vertices=vertices, translation=trans)
    return vertices


def split(v_tooth, v_left=None, v_right=None, axis=y):
    """

    :param v_tooth:
    :param v_left:
    :param v_right:
    :param axis:
    :return:
    """
    r_tooth = ranges(v_tooth)
    limit_axis = (r_tooth[0][axis] + r_tooth[1][axis]) / 2
    if v_left is not None:
        v_left = v_left[np.where(v_left[:, axis] < limit_axis)]
    if v_right is not None:
        v_right = v_right[np.where(v_right[:, axis] >= limit_axis)]
    return [v_left, v_right]


def detect_flip(heap_id):
    """

    :param heap_id:
    :return:
    """
    ops = []
    # variables extraction
    if int(heap_id / 10) in [1, 2]:
        # Flip the base
        flip = [{"rotation": {"axis": x, "theta": pi}}]
        ops.extend(flip)
    return ops


def scale(vertices, rate, axis, positive=False, negative=False, t_max=None, t_min=None):
    """

    :param vertices:
    :param rate:
    :param axis:
    :param positive:
    :param negative:
    :param t_max:
    :param t_min:
    :return:
    """
    vertices = copy.deepcopy(vertices)
    if positive:
        [vertices, trans] = center(vertices=vertices)
        vertices[np.where(vertices[:, axis] >= 0), axis] *= rate
        vertices = translate(vertices, -trans)
    elif negative:
        [vertices, trans] = center(vertices=vertices)
        vertices[np.where(vertices[:, axis] < 0), axis] *= rate
        vertices = translate(vertices, -trans)
    elif t_max is not None:
        [vertices, trans_center] = center(vertices=vertices)
        t_max = t_max + trans_center[axis]
        if t_min is not None:
            t_min = t_min + trans_center[axis]
        else:
            t_min = min(vertices[:, axis]) + trans_center[axis]
        t_middle = t_min + (t_max - t_min) / 2
        inds_positives = np.where(vertices[:, axis] >= t_middle)[0]
        inds_negatives = np.where(vertices[:, axis] < t_middle)[0]
        vertices_positives = vertices[inds_positives, :]
        vertices_negatives = vertices[inds_negatives, :]

        trans = np.array([0, 0, -t_max])
        vertices_positives = translate(vertices_positives, trans)
        vertices_positives[np.where(vertices_positives[:, axis] < 0)[
            0], axis] *= rate
        trans = np.zeros(3)
        trans[axis] = min(vertices_positives[:, axis])
        vertices_positives = translate(vertices_positives, -trans)

        if t_min is not None:
            trans = np.array([0, 0, t_min])
            vertices_negatives = translate(vertices_negatives, trans)
            vertices_negatives[np.where(vertices_negatives[:, axis] > 0)[
                0], axis] *= rate
            trans = np.zeros(3)
            trans[axis] = max(vertices_negatives[:, axis])
            vertices_negatives = translate(vertices_negatives, -trans)

        vertices[inds_positives, :] = vertices_positives
        vertices[inds_negatives, :] = vertices_negatives

        vertices = translate(vertices, -trans_center)
    elif t_min is not None:
        trans = np.array([0, 0, t_min])
        vertices = translate(vertices, trans)
        vertices[np.where(vertices[:, axis] > 0), axis] *= rate
        vertices = translate(vertices, -trans)
    else:
        trans = np.mean(ranges(vertices), axis=0)
        vertices = translate(vertices, trans)
        vertices[:, axis] *= rate
        vertices = translate(vertices, -trans)

    return vertices


def rgb2int(face_colors):
    """

    :param face_colors:
    :return:
    """
    face_colors = face_colors.astype(str)
    f01 = list(np.core.defchararray.add(
        list(face_colors[:, 0]), list(face_colors[:, 1])))
    f12 = np.core.defchararray.add(f01, list(face_colors[:, 2]))
    f23 = np.core.defchararray.add(f12, list(face_colors[:, 3]))
    return np.array(f23, dtype=np.longlong)


def slicing(vertices, maj_axis=y, min_axis=x, N=100):
    """

    :param vertices:
    :param maj_axis:
    :param min_axis:
    :param N:
    :return:
    """
    slices = []
    r_major_axis = ranges(vertices=vertices, axis=maj_axis)
    pad_major_axis = (max(r_major_axis) - min(r_major_axis)) / N
    inds = np.arange(min(r_major_axis), max(r_major_axis), pad_major_axis)
    for t_min in inds:
        try:
            slice = ranges(vertices=crop(vertices=vertices, axis=maj_axis, t_min=t_min, t_max=t_min + pad_major_axis),
                           axis=min_axis)
            slices.append(slice)
        except:
            slices.append(slice)
    slices = np.array(slices)

    error_mean = np.mean(np.abs(np.sum(slices, axis=1)))
    amplitude = np.mean([np.mean(np.abs(slices[:, 0])),
                        np.mean(np.abs(slices[:, 1]))])
    q = np.clip(1 - (error_mean / amplitude), 0, 1)

    return inds, slices, q


def eops(meshes, operations):
    """

    :param meshes:
    :param operations:
    :return:
    """
    for m in list(set(meshes)):
        m.eops(operations)


def iops(meshes, operations):
    """

    :param meshes:
    :param operations:
    :return:
    """
    for m in list(set(meshes)):
        m.iops(operations)


def image(vertices=None, display=False, title=None, RGB=False, factor=None, wait=1):
    """
    Convert a mesh into a 2d image

    :param vertices:
    :param ptop:
    :param pbottom:
    :param display:
    :param title:
    :param RGB:
    :param factor:
    :param wait:
    :return:
    """

    def max_filtering(im, pad=10):
        return ndimage.maximum_filter(im, size=pad)

    start_time = time.time()
    [x, y, z] = [0, 1, 2]
    # Preprocess
    vertices_rect = vertices
    correction = 10000
    vertices_rect[:, 0:2] = np.array(
        np.floor(vertices_rect[:, 0:2] * correction) / correction)

    correction = [min(vertices[:, 0]), min(
        vertices[:, 1]), min(vertices[:, 2])]
    vertices_rect = vertices - correction

    correction = max(vertices_rect[:, 2])
    vertices_rect[:, 2] = vertices_rect[:, 2] / correction

    correction = 10
    vertices_rect[:, 0:2] = vertices_rect[:, 0:2] * correction + 1

    # start_time1 = time.time()
    i_max = int(max(vertices_rect[:, 0]))
    j_max = int(max(vertices_rect[:, 1]))
    M = np.zeros([i_max + 1, j_max + 1])
    M_inds = np.zeros([i_max + 1, j_max + 1], dtype=int)
    vertices_rect[:, 0:2] = np.array(vertices_rect[:, 0:2], dtype=int)
    df = pd.DataFrame(data=vertices_rect)
    df = df.drop_duplicates(subset=z, keep='last')
    X = list(np.array(df.values[:, x], dtype=int))
    Y = list(np.array(df.values[:, y], dtype=int))
    Z = df.values[:, z]
    INDS = list(np.array(df.index.values, dtype=int))
    M[X, Y] = Z
    M_inds[X, Y] = INDS
    # print("[M_inds]\t--- %s seconds ---" % (time.time() - start_time1))

    start_time1 = time.time()
    # Remove empty rows, cols
    x = np.arange(0, len(M))
    mask_x = np.ones(len(x), dtype=bool)
    inds_x = np.where(np.max(M, axis=1) == 0)
    mask_x[inds_x] = False  # Set unwanted elements to False
    M = M[x[mask_x], :]
    M_inds = M_inds[x[mask_x], :]

    y = np.arange(0, len(M[0]))
    mask_y = np.ones(len(y), dtype=bool)
    inds_y = np.where(np.max(M, axis=0) == 0)
    mask_y[inds_y] = False  # Set unwanted elements to False
    M = M[:, y[mask_y]]
    M_inds = M_inds[:, y[mask_y]]
    # print("[remoev]\t--- %s seconds ---" % (time.time() - start_time1))

    # start_time1 = time.time()
    # Maxfiltering
    im = np.array(M * 255, dtype=np.uint8)
    im = max_filtering(im)
    M_inds = max_filtering(M_inds)
    # print("[max_filtering]\t--- %s seconds ---" % (time.time() - start_time1))

    if factor is None:
        factor = 1
    else:
        im = cv2.resize(im, None, fx=factor, fy=factor,
                        interpolation=cv2.INTER_CUBIC)

    if RGB:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)

    if display:
        display_image(im=im, title=title, factor=factor, wait=wait)

    return im


def find_orientation(vertices=None, im=None, borne_inf=0, borne_sup=1, display=False, wait=0, radian=False,
                     return_image=False):
    """

    :param vertices:
    :param im:
    :param borne_inf:
    :param borne_sup:
    :param display:
    :param wait:
    :param radian:
    :param return_image:
    :return:
    """

    # Tooth orientation
    if im is None:
        im = image(crop(axis=z, vertices=vertices,
                   borne_inf=borne_inf, borne_sup=borne_sup))

    # Get the contours
    ret, thresh = cv2.threshold(im, 0, 255, 0)
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]

    # Get the the image
    ellipse = cv2.fitEllipse(cnt)
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    cv2.ellipse(im, ellipse, (0, 255, 0), 2)
    theta_opt_tooth = ellipse[2]

    # Display
    if display:
        cv2.imshow("im", im)
        cv2.waitKey(wait)

    # Reformat the value to output
    theta_opt_tooth = (theta_opt_tooth / 180) * \
        pi if radian else theta_opt_tooth
    # Return the result
    return theta_opt_tooth if return_image is False else theta_opt_tooth, im


def is_contained_in(inside_vertices, outside_vertices, grid=100, display=False, return_inter=False):
    """

    :param inside_vertices:
    :param outside_vertices:
    :param grid:
    :param display:
    :param return_inter:
    :return:
    """
    # print("outside_vertices" + str(ranges(outside_vertices)))

    # inside_vertices, trans = center(inside_vertices)
    # outside_vertices = translate(outside_vertices, trans)

    # r_inside = np.array(ranges(inside_vertices) * 10, dtype=int)
    # r_outside = np.array(ranges(outside_vertices) * 10, dtype=int)
    # r = r_outside - r_inside
    # if np.min(r)==0:
    #     return True
    # return False

    # print("inside_vertices" + str(ranges(inside_vertices)))
    inside_vertices = np.unique(
        np.array(inside_vertices * grid, dtype=int), axis=0) / grid
    # print("inside_vertices" + str(ranges(inside_vertices)))
    outside_vertices = np.unique(
        np.array(outside_vertices * grid, dtype=int), axis=0) / grid

    rz = ranges(inside_vertices, axis=z)
    pad = 1 / grid
    K = []
    for z_min in np.arange(rz[0], rz[1] - pad, pad):
        outside_vertices_croped = crop(
            outside_vertices, axis=z, t_min=z_min, t_max=z_min + pad)
        inside_vertices_croped = crop(
            inside_vertices, axis=z, t_min=z_min, t_max=z_min + pad)
        K.append([z_min, z_min + pad, len(bounding_box(outside_vertices_croped,
                 r=ranges(inside_vertices_croped)))])
    K = np.array(K)
    if display:
        plt.plot(K[:, 0], K[:, 2])
        plt.show()
    if max(K[:, 2]) <= 1:
        if return_inter:
            return True, max(K[:, 2])
        else:
            return True
    else:
        if return_inter:
            return False, max(K[:, 2])
        else:
            return False


def random_color():
    """

    :return:
    """
    color = [random.randint(0, 255) for _ in range(3)]
    color.append(255)
    return np.array(color, dtype=np.uint8)


def xyz(x, y, res=np.ones(3) * 64):
    """

    :param x:
    :param y:
    :param res:
    :return:
    """
    # all = np.concatenate((x,y), axis=0)
    y -= np.min(x, axis=0)
    x -= np.min(x, axis=0)
    L = length(x)
    y /= L  # np.max(x, axis=0)
    x /= L  # np.max(x, axis=0)

    all = np.concatenate((x, y), axis=0)
    assert np.max(length(all)) < 1.1

    x = bounding_box(x, [[0, 0, 0], [1, 1, 1]])
    y = bounding_box(y, [[0, 0, 0], [1, 1, 1]])

    x *= (res - 1)
    y *= (res - 1)
    x = np.unique(np.array(x, dtype=int), axis=0)
    y = np.unique(np.array(y, dtype=int), axis=0)

    return x, y


def rgb2flaot(crgb):
    """

    :param crgb:
    :return:
    """
    if crgb[0] > 0:
        value = 0.5 - (crgb[0] / 255) / 2
    else:
        value = (crgb[2] / 255) / 2 + 0.5
    return value


def meshlab(script_name, filename, ext_out="ply"):
    """
    Apply a filter to the mesh

    :param name:
    :return:
    """

    from .nmesh import NMesh
    file_out = "{}/{}.{}".format(parent(filename), name(filename), ext_out)
    script = f"__data__/mlx/{script_name}.mlx".format
    command = "xvfb-run -a -s \"-screen 0 800x600x24\" meshlabserver -i \"{}\" -o \"{}\" -s {} -om vc".format(
        filename,
        file_out,
        script)
    os.system(command)
    print(">> meshlab : {}".format(command))
    mesh = NMesh(file_out)
    os.system("rm {}".format(file_out))
    return mesh
