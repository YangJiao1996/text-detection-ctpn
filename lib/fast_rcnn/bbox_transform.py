import numpy as np

def offset_intergration(offsets_left, offsets_right):
    """
    Integrating two offset arrays in a specific manner.

    Parameters
    ------------
    offsets_left, offsets_right: n * 1 numpy arrays
    ------------
    Return
    ------------
    offsets: n * 1 arrays, integrated offsets
    """
    assert offsets_left.shape == offsets_right.shape, \
            "The two offsets have different shapes: {}, {}". \
            format(offsets_left.shape, offsets_right.shape)
    zero = np.zeros(offsets_left.shape)
    offsets = np.zeros(offsets_right.shape)

    all_zero = np.logical_and(offsets_left == zero, offsets_right == zero)
    index_bothzero = np.where(all_zero)
    left_lesser = np.logical_and(np.logical_not(all_zero), (abs(offsets_left) < abs(offsets_right)))
    right_lesser = np.logical_and(np.logical_not(all_zero), (abs(offsets_left) >= abs(offsets_right)))
    left_zero = np.logical_and(np.logical_not(all_zero), offsets_left == zero)
    right_zero = np.logical_and(np.logical_not(all_zero), offsets_right == zero)

    offsets[index_bothzero] = 0
    offsets[left_lesser] = offsets_left[left_lesser]
    offsets[right_lesser] = offsets_right[right_lesser]
    offsets[left_zero] = offsets_right[left_zero]
    offsets[right_zero] = offsets_left[right_zero]


    return offsets



def bbox_transform(ex_rois, gt_rois):
    """
    computes the distance from ground-truth boxes to the given boxes, normed by their size
    :param ex_rois: n * 4 numpy array, given boxes
    :param gt_rois: n * 4 numpy array, ground-truth boxes
    :return: deltas: n * 4 numpy array, ground-truth boxes
    """

    DEBUG = False

    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    assert np.min(ex_widths) > 0.1 and np.min(ex_heights) > 0.1, \
        'Invalid boxes found: {} {}'. \
            format(ex_rois[np.argmin(ex_widths), :], ex_rois[np.argmin(ex_heights), :])

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights
    gt_left_sides = gt_rois[: ,0]
    gt_right_sides = gt_rois[:, 2]

    # Finding the side-anchors
    horizon_left_dists = gt_left_sides - ex_ctr_x
    horizon_right_dists = gt_right_sides - ex_ctr_x
    side_left = np.where(abs(horizon_left_dists) <= 32)
    side_right = np.where(abs(horizon_right_dists) <= 32)

    # Calculating offsets
    offsets_left = np.zeros((len(gt_ctr_x)))
    offsets_left[side_left] = horizon_left_dists[side_left]
    offsets_right = np.zeros((len(gt_ctr_x)))
    offsets_right[side_right] = horizon_right_dists[side_right]
    offsets = offset_intergration(offsets_left, offsets_right) / ex_widths

    # warnings.catch_warnings()
    # warnings.filterwarnings('error')
    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    if DEBUG:
        print("gt_ctr_x.shape: ", gt_ctr_x.shape)
        print("ex_ctr_x.shape: ", ex_ctr_x.shape)
        print("ex_widths: ")
        print(ex_widths)
        print("ex_widths.shape: ", ex_widths.shape)
        print("targets_dx.shape: ", targets_dx.shape)
        print("horizon_left_dists: ")
        print(horizon_left_dists)
        print("horizon_right_dists: ")
        print(horizon_right_dists)
        print("side_left: ")
        print(side_left)
        print("side_right: ")
        print(side_right)
        print("offsets: ")
        print(np.where(abs(offsets != 0)))

    targets = np.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()

    return targets

def bbox_transform_inv(boxes, deltas):

    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes

def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """

    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes

