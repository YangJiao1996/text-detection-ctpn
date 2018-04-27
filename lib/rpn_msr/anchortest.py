# -*- coding:utf-8 -*-
import numpy as np
from .utils.bbox import bbox_overlaps, bbox_intersections
DEBUG = True

def generate_basic_anchors(sizes, base_size=16):
    base_anchor = np.array([0, 0, base_size - 1, base_size - 1], np.int32)
    anchors = np.zeros((len(sizes), 4), np.int32)
    index = 0
    for h, w in sizes:
        anchors[index] = scale_anchor(base_anchor, h, w)
        index += 1
    return anchors


def scale_anchor(anchor, h, w):
    x_ctr = (anchor[0] + anchor[2]) * 0.5
    y_ctr = (anchor[1] + anchor[3]) * 0.5
    scaled_anchor = anchor.copy()
    scaled_anchor[0] = x_ctr - w / 2  # xmin
    scaled_anchor[2] = x_ctr + w / 2  # xmax
    scaled_anchor[1] = y_ctr - h / 2  # ymin
    scaled_anchor[3] = y_ctr + h / 2  # ymax
    return scaled_anchor


def generate_anchors(base_size=16, ratios=[0.5, 1, 2],
                     scales=2**np.arange(3, 6)):
    heights = [11, 16, 23, 33, 48, 68, 97, 139, 198, 283]
    # Modified according to CTPN
    widths = [16]
    sizes = []
    for h in heights:
        for w in widths:
            sizes.append((h, w))
    return generate_basic_anchors(sizes)

# if __name__ == '__main__':
#     import time
#     t = time.time()
#     a = generate_anchors()
#     print(time.time() - t)
#     print(a)
#     from IPython import embed; embed()


# heights = [11, 16, 23, 33, 48, 68, 97, 139, 198, 283]
# widths = [16]
# sizes = []
# for h in heights:
#     for w in widths:
#         sizes.append((h, w))
# base_anchor = np.array([0, 0, 16 - 1, 16 - 1], np.int32)
# anchors = np.zeros((len(sizes), 4), np.int32)

HEIGHTS = 50
WIDTHS = 37
gt_1 = np.array([210, 115, 310, 225])
gt_2 = np.array([316, 275, 420, 336])
gt_3 = np.array([410, 135, 510, 265])
gt_4 = np.array([290, 110, 370, 240])
gt_5 = np.array([500, 299, 610, 350])


rpn_cls_score = np.random.rand(1, HEIGHTS, WIDTHS, 4) # (1, H, W, Ax2)
_feat_stride = [16, ]
im_info = [800, 600, 3]
_allowed_border =  0
_anchors = generate_anchors(scales = 16)
_num_anchors = _anchors.shape[0]
gt_boxes = np.vstack((gt_1, gt_2, gt_3, gt_4, gt_5))







if DEBUG:
    #anchors = anchors.reshape((K * A, 4))
    print("_anchors: ")
    print(_anchors)
    print("anchor shapes:")
    print(np.hstack((
         _anchors[:, 2::4] - _anchors[:, 0::4],
         _anchors[:, 3::4] - _anchors[:, 1::4],
         )))
    #print("shifts: ")
    #print(shifts)
    #print("anchors: ")
    #print(anchors)

assert rpn_cls_score.shape[0] == 1, "Only single item batches are supported"

height, width = rpn_cls_score.shape[1:3] # Height: 4, Width: 6


shift_x = np.arange(0, height) * _feat_stride
shift_y = np.arange(0, width) * _feat_stride
shift_x, shift_y = np.meshgrid(shift_x, shift_y)
shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), 
                    shift_x.ravel(), shift_y.ravel())).transpose()

A = _num_anchors 
K = shifts.shape[0] #50x37 here according to ctpn

# anchors from Ax4 to 1xAx4
# shifts from Kx4 to 1xKx4 to Kx1x4
all_anchors = _anchors.reshape(1, A, 4) + \
         shifts.reshape(1, K, 4).transpose((1, 0, 2)) # KxAx4
all_anchors = all_anchors.reshape(K*A, 4)
total_anchors = int(K * A)

if DEBUG:
    print("shift_x: ")
    print(shift_x)  
    print("shift_y: ")
    print(shift_y)
    print("shifts:")
    print(shifts)
    print("number of anchors: ", A)
    print("shifts shape:", K)
    print("all anchors:")
    print(all_anchors)


# only keep anchors inside the image
#仅保留那些还在图像内部的anchor，超出图像的都删掉
inds_inside = np.where(
    (all_anchors[:, 0] >= -_allowed_border) &
    (all_anchors[:, 1] >= -_allowed_border) &
    (all_anchors[:, 2] < im_info[1] + _allowed_border) &  # width
    (all_anchors[:, 3] < im_info[0] + _allowed_border)    # height
)[0]

anchors = all_anchors[inds_inside, :]

if DEBUG:
    print('total_anchors', total_anchors)
    print('inds_inside', len(inds_inside))
    print("anchors.shape", anchors.shape)


labels = np.empty((len(inds_inside), ), dtype=np.float32)
labels.fill(-1)

if DEBUG:
    print ("labels shape", labels.shape)


overlaps = bbox_overlaps(
    np.ascontiguousarray(anchors, dtype=np.float),
    np.ascontiguousarray(gt_boxes, dtype=np.float))