'''
Title: HW-4, CS-532 Spring 2022, helper file
Author: Agamdeep S. Chopra
Date: 04/28/2022
'''

import numpy as np


def grey(img, mode='average'):
    if mode == 'average':
        output = (1/3) * (img[:, :, 0] + img[:, :, 1] + img[:, :, 2])
    elif mode == 'BT.601':
        output = 0.299 * img[:, :, 0] + 0.587 * \
            img[:, :, 1] + 0.114 * img[:, :, 2]
    elif mode == 'BT.709':
        output = 0.2126 * img[:, :, 0] + 0.7152 * \
            img[:, :, 1] + 0.0722 * img[:, :, 2]
    return output


def conv2D(img, kernel, padding=0, strides=1):  # input img -> 2d (x,y). Greyscale imgs only!
    kernel = np.flipud(np.fliplr(kernel))
    output = np.zeros((int(((img.shape[0] - kernel.shape[0] + 2 * padding) / strides) + 1), int(
        ((img.shape[1] - kernel.shape[1] + 2 * padding) / strides) + 1)))
    if padding != 0:
        img_ = np.zeros((img.shape[0] + padding*2, img.shape[1] + padding*2))
        img_[int(padding):int(-1 * padding),
             int(padding):int(-1 * padding)] = img
    else:
        img_ = img
    for y in range(img.shape[1]):
        if y > img.shape[1] - kernel.shape[1]:
            break
        if y % strides == 0:
            for x in range(img.shape[0]):
                if x > img.shape[0] - kernel.shape[0]:
                    break
                try:
                    if x % strides == 0:
                        output[x, y] = (
                            kernel * img_[x: x + kernel.shape[0], y: y + kernel.shape[1]]).sum()
                except:
                    break
    return output


def non_max_suppress(R, window_size=3, padding=1, stride=1):  # odd windows only
    R_supp = np.zeros(R.shape)
    R_ = np.ones((R.shape[0] + padding*2, R.shape[1] +
                 padding*2)) * np.min(R) * 1E-1
    R_[padding:-padding, padding:-padding] = R
    for i in range(padding, R.shape[0]+padding):
        for j in range(padding, R.shape[1]+padding):
            if np.max(R_[i-int(window_size/2):i+int(window_size/2)+1, j-int(window_size/2):j+int(window_size/2)+1]) == R_[i, j]:
                R_supp[i-padding, j-padding] = R_[i, j]
    return R_supp


def rank_transform(img, window):
    if len(img.shape) > 2:
        img = grey(img, 'BT.709')
    padding = int(window/2)
    img_ = np.zeros((img.shape[0] + padding*2, img.shape[1] + padding*2))
    img_[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = img
    img_rank = np.zeros(img.shape)
    for i in range(padding, img.shape[0]+padding):
        for j in range(padding, img.shape[1]+padding):
            rank = 0
            for u in range(-padding, padding+1):
                for v in range(-padding, padding+1):
                    if img_[i, j] > img_[i+u, j+v]:
                        rank += 1
            img_rank[i-padding, j-padding] = rank
    return img_rank


def SAD_dist(img_1, img_2, corner_1, corner_2):
    if len(img_1.shape) > 2:
        img_1 = grey(img_1, 'BT.709')
    if len(img_2.shape) > 2:
        img_2 = grey(img_2, 'BT.709')
    dist_list = []
    for c1 in corner_1:
        dist_list.append([])
        for c2 in corner_2:
            sad = 0
            for i in range(-5, 6):
                for j in range(-5, 6):
                    f1 = 0 <= c1[0] + i < img_1.shape[0]
                    f2 = 0 <= c2[0] + i < img_1.shape[0]
                    f3 = 0 <= c1[1] + j < img_1.shape[1]
                    f4 = 0 <= c2[1] + j < img_1.shape[1]
                    if f1 and f2 and f3 and f4:
                        sad += np.abs(np.round(img_1[c1[0] + i, c1[1] + j]) -
                                      np.round(img_2[c2[0] + i, c2[1] + j]))
            dist_list[-1].append(sad)
    return np.asarray(dist_list)


def RANSAC(coords1, coords2, pt_idx, alpha=2, beta=0.8):  # 3d coords and corrosponding pt idx
    l = len(pt_idx)
    idx = np.arange(l)
    flag = True
    while flag:
        np.random.shuffle(idx)
        P1 = [coords1[pt_idx[idx[0], 0]],
              coords1[pt_idx[idx[1], 0]], coords1[pt_idx[idx[2], 0]]]
        P2 = [coords2[pt_idx[idx[0], 1]],
              coords2[pt_idx[idx[1], 1]], coords2[pt_idx[idx[2], 1]]]
        v11 = P1[0] - P1[1]
        v12 = P1[1] - P1[2]
        v21 = P2[0] - P2[1]
        v22 = P2[1] - P2[2]
        R_ = np.asarray([v21, v22, np.cross(
            v21, v22)]) @ np.linalg.inv(np.asarray([v11, v12, np.cross(v11, v12)]))
        U, _, V_T = np.linalg.svd(R_)
        R = U @ V_T
        T = P2[0] - R @ P1[0]
        P2_ = []
        P2_real = []
        for i in range(l):
            pt = R @ coords1[pt_idx[idx[i], 0]] + T
            pt = pt[:2]/pt[2]
            P2_.append(pt)
            pt = coords2[pt_idx[idx[i], 1]]
            pt = pt[:2]/pt[2]
            P2_real.append(pt)
        P2_ = np.asarray(P2_)
        P2_real = np.asarray(P2_real)
        score = 0
        for i in range(l):
            SSD = np.sum((P2_real[i]-P2_[i])**2)
            if SSD < alpha:
                score += 1
        if score >= l*beta:
            flag = False
    return R, T


def save_point_cloud(points, color, path, file_name):
    fout = open(path+'\\'+file_name+'.ply', 'w')
    fout.write("ply\n")
    fout.write("format ascii 1.0\n")
    fout.write("element vertex %d\n" % (points.shape[0]))
    fout.write("property float x\n")
    fout.write("property float y\n")
    fout.write("property float z\n")
    fout.write("property uchar red\n")
    fout.write("property uchar green\n")
    fout.write("property uchar blue\n")
    fout.write("element face 0\n")
    fout.write("end_header\n")
    for i in range(points.shape[0]):
        fout.write('%f %f %f %d %d %d\n' % (
            points[i, 0], points[i, 1], -points[i, 2], color[i, 0], color[i, 1], color[i, 2]))
    fout.close()


def surface_normal(points3d, mask, i, j):
    pt1 = points3d[int(mask[i, j])]
    pt2 = None
    pt3 = None
    i_, j_ = i-3, j-3
    for k in range(7):
        for l in range(7):
            if mask[i_+k, j_+l] > -1 and i_+k != i and j_+l != j:
                if pt2 is None:
                    pt2 = points3d[int(mask[i_+k, j_+l])]
                elif pt3 is None:
                    pt3 = points3d[int(mask[i_+k, j_+l])]
                else:
                    break
        if pt2 is not None and pt3 is not None:
            break
    a = pt2 - pt1
    b = pt3 - pt1
    normal = np.cross(a, b)
    return normal


# return list of surface normals same size as point3d
def FastNormalApproximation(points3d, mask):
    pad_mask = np.ones((mask.shape[0]+6, mask.shape[1]+6)) * -1
    pad_mask[3:-3, 3:-3] = mask
    output = []
    for i in range(pad_mask.shape[0]):
        for j in range(pad_mask.shape[1]):
            if pad_mask[i, j] > -1:
                output.append(surface_normal(points3d, pad_mask, i, j))
    return output


def FastNormalApproximationSimpleSmooth(points3d, mask):
    pad_mask = np.ones((mask.shape[0]+6, mask.shape[1]+6)) * -1
    pad_mask[3:-3, 3:-3] = mask
    output = []
    for i in range(pad_mask.shape[0]):
        for j in range(pad_mask.shape[1]):
            if pad_mask[i, j] > -1:
                normal = surface_normal(points3d, pad_mask, i, j)
                k = -np.sum(normal * points3d[int(pad_mask[i, j])])
                if (normal[0] > 0 and normal[1] > 0 and normal[2] > 0 and k < 0) or (normal[0] < 0 and normal[1] < 0 and normal[2] < 0 and k > 0):
                    normal = -normal
                output.append(normal)
    return output


def normal2img(normals, mask):
    idx = 0
    output = np.zeros((mask.shape[0], mask.shape[1], 3))
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] > -1:
                n = normals[idx]
                n_norm = np.linalg.norm(n)
                output[i, j, :] = 255 * \
                    ((n/(2*n_norm)) + np.asarray([0.5, 0.5, 0.5]))
                idx += 1
    return output.astype(int)
