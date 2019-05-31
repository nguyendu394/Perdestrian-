import numpy as np
import torch

def getGTboxesPadding(gt_boxes,MAX_GTS):
    gt_boxes_padding = np.zeros((MAX_GTS, 5),dtype=np.float)
    if gt_boxes:
        num_gts = len(gt_boxes)
        gt_boxes = np.asarray(gt_boxes,dtype=np.float)

        # gt_boxes_padding = torch.FloatTensor(self.MAX_GTS, 5).zero_()
        gt_boxes_padding[:num_gts,:] = gt_boxes[:num_gts]
    return gt_boxes_padding

def getAllrois(all_rois,gt_boxes):
    ''' get all bbox include predicted bbox, gt_boxes and jitter gt bboxes
        input: all_rois(numpy)(N,4)[l,t,r,b]
               gt_boxes(numpy)(K,5)[l,t,r,b,cls]
        output: numpy (M,4) M = N+K*2
    '''

    if gt_boxes:
        gt_boxes = np.asarray(gt_boxes)
        jit_gt_boxes = _jitter_gt_boxes(gt_boxes)
        idx = np.ones((gt_boxes.shape[0] * 2, 1), dtype=gt_boxes.dtype)*all_rois[0,0]
        all_rois = np.vstack((all_rois, np.hstack((idx,np.vstack((gt_boxes[:, :-1], jit_gt_boxes[:, :-1]))))))
        # print(all_rois)
        # print('END')
        return all_rois
    else:
        return all_rois

def _jitter_gt_boxes(gt_boxes, jitter=0.05):
    """ jitter the gtboxes, before adding them into rois, to be more robust for cls and rgs
    gt_boxes: (G, 5) [x1 ,y1 ,x2, y2, class] int
    """
    jittered_boxes = gt_boxes.copy()
    ws = jittered_boxes[:, 2] - jittered_boxes[:, 0] + 1.0
    hs = jittered_boxes[:, 3] - jittered_boxes[:, 1] + 1.0
    width_offset = (np.random.rand(jittered_boxes.shape[0]) - 0.5) * jitter * ws
    height_offset = (np.random.rand(jittered_boxes.shape[0]) - 0.5) * jitter * hs
    jittered_boxes[:, 0] += width_offset
    jittered_boxes[:, 2] += width_offset
    jittered_boxes[:, 1] += height_offset
    jittered_boxes[:, 3] += height_offset

    return jittered_boxes

def bbox_overlaps(anchors, gt_boxes):
    """
    anchors: (N, 4) ndarray of float
    gt_boxes: (K, 4) ndarray of float
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = anchors.size(0)
    K = gt_boxes.size(0)

    gt_boxes_area = ((gt_boxes[:,2] - gt_boxes[:,0] + 1) *
                (gt_boxes[:,3] - gt_boxes[:,1] + 1)).view(1, K)

    anchors_area = ((anchors[:,2] - anchors[:,0] + 1) *
                (anchors[:,3] - anchors[:,1] + 1)).view(N, 1)

    boxes = anchors.view(N, 1, 4).expand(N, K, 4)
    query_boxes = gt_boxes.view(1, K, 4).expand(N, K, 4)

    iw = (torch.min(boxes[:,:,2], query_boxes[:,:,2]) -
        torch.max(boxes[:,:,0], query_boxes[:,:,0]) + 1)
    iw[iw < 0] = 0

    ih = (torch.min(boxes[:,:,3], query_boxes[:,:,3]) -
        torch.max(boxes[:,:,1], query_boxes[:,:,1]) + 1)
    ih[ih < 0] = 0

    ua = anchors_area + gt_boxes_area - (iw * ih)
    overlaps = iw * ih / ua

    return overlaps

def bbox_transform(ex_rois, gt_rois):
    """
    computes the distance from ground-truth boxes to the given boxes, normed by their size
    :param ex_rois: n * 4 numpy array, given boxes
    :param gt_rois: n * 4 numpy array, ground-truth boxes
    :return: deltas: n * 4 numpy array, ground-truth boxes
    """
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    # assert np.min(ex_widths) > 0.1 and np.min(ex_heights) > 0.1, \
    #     'Invalid boxes found: {} {}'. \
    #         format(ex_rois[np.argmin(ex_widths), :], ex_rois[np.argmin(ex_heights), :])

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
    return targets

def _compute_targets(ex_rois, gt_rois, labels):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    targets = bbox_transform(ex_rois, gt_rois)
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
        targets = ((targets - np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
                   / np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS))
    return np.hstack(
        (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)


def sample_rois(all_rois, gt_boxes, fg_rois_per_image, rois_per_image, num_classes):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # overlaps: R x G
    overlaps = bbox_overlaps(all_rois[:,1:], gt_boxes[:,:-1])
    print(overlaps)
    overlaps = overlaps.numpy()
    gt_assignment = overlaps.argmax(axis=1)  # R
    max_overlaps = overlaps.max(axis=1)  # R

    labels = gt_boxes[gt_assignment, 4]
    print('labels')
    print(labels)
    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= 0.5)[0]
    # fg_inds = np.setdiff1d(fg_inds, ignore_inds)
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = min(fg_rois_per_image, fg_inds.size)
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        # fg_inds = np.random.choice(fg_inds, size=fg_rois_per_this_image, replace=False)
        fg_inds = fg_inds[:fg_rois_per_this_image]
    print('fg_inds')
    print(fg_inds)
    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((max_overlaps < 0.5) &
                       (max_overlaps >= 0))[0]
    # bg_inds = np.setdiff1d(bg_inds, ignore_inds)
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)
    # Sample background regions without replacement
    if bg_inds.size > 0:
        # bg_inds = np.random.choice(bg_inds, size=bg_rois_per_this_image, replace=False)
        bg_inds = bg_inds[:bg_rois_per_this_image]

    print('bg_inds')
    print(bg_inds)
    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[fg_rois_per_this_image:] = 0
    rois = all_rois[keep_inds]

    return labels, rois


def sample_rois_tensor(all_rois, gt_boxes, fg_rois_per_image, rois_per_image, num_classes):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # overlaps: R x G

    overlaps = bbox_overlaps(all_rois[:,1:], gt_boxes[:,:-1])
    print(overlaps)
    # overlaps = overlaps.numpy()
    # gt_assignment = overlaps.argmax(axis=1)  # R
    # max_overlaps = overlaps.max(axis=1)  # R
    max_overlaps, gt_assignment = torch.max(overlaps, 1)
    labels = gt_boxes[gt_assignment, 4]
    print('labels')
    print(labels)
    # preclude hard samples
    # ignore_inds = torch.empty(shape=(0), dtype=int)

    # Select foreground RoIs as those with >= FG_THRESH overlap
    # fg_inds = np.where(max_overlaps >= 0.5)[0]
    fg_inds = (max_overlaps >= 0.5).nonzero().view(-1)
    print('fg_inds')
    print(fg_inds)
    #Return the sorted, unique values in ar1 that are not in ar2.
    # fg_inds = torch.setdiff1d(fg_inds, ignore_inds)
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs

    fg_rois_per_this_image = min(fg_rois_per_image, fg_inds.size(0))
    # print(fg_inds)
    # Sample foreground regions without replacement
    if fg_inds.size(0) > 0:
        # fg_inds = np.random.choice(fg_inds, size=fg_rois_per_this_image, replace=False)
        fg_inds = fg_inds[:fg_rois_per_this_image]

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    # bg_inds = np.where((max_overlaps < 0.5) &
    #                    (max_overlaps >= 0))[0]
    bg_inds = (max_overlaps < 0.5).nonzero().view(-1)


    # bg_inds = np.setdiff1d(bg_inds, ignore_inds)
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size(0))
    # Sample background regions without replacement
    if bg_inds.size(0) > 0:
        # bg_inds = np.random.choice(bg_inds, size=bg_rois_per_this_image, replace=False)
        bg_inds = bg_inds[:bg_rois_per_this_image]

    print('bg_inds')
    print(bg_inds)
    # The indices that we're selecting (both fg and bg)
    # keep_inds = np.append(fg_inds, bg_inds)
    keep_inds = torch.cat((fg_inds, bg_inds))
    # Select sampled values from various arrays:

    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[fg_rois_per_this_image:] = 0
    rois = all_rois[keep_inds]

    return labels,rois
    # bbox_target_data = _compute_targets(
        # rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4], labels)

    # bbox_target_data (1 x H x W x A, 5)
    # bbox_targets <- (1 x H x W x A, K x 4)
    # bbox_inside_weights <- (1 x H x W x A, K x 4)
    # bbox_targets, bbox_inside_weights = \
        # _get_bbox_regression_labels(bbox_target_data, num_classes)

    # return labels, rois, bbox_targets, bbox_inside_weights
