from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import torch
from config import cfg

def getGTboxesPadding(gt_boxes,MAX_GTS):
    gt_boxes_padding = np.zeros((MAX_GTS, 5),dtype=np.float)
    if gt_boxes:
        num_gts = len(gt_boxes)
        # print('num',num_gts)
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

def bbox_transform_batch(ex_rois, gt_rois):

    if ex_rois.dim() == 2:
        ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
        ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
        ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
        ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

        gt_widths = gt_rois[:, :, 2] - gt_rois[:, :, 0] + 1.0
        gt_heights = gt_rois[:, :, 3] - gt_rois[:, :, 1] + 1.0
        gt_ctr_x = gt_rois[:, :, 0] + 0.5 * gt_widths
        gt_ctr_y = gt_rois[:, :, 1] + 0.5 * gt_heights

        targets_dx = (gt_ctr_x - ex_ctr_x.view(1,-1).expand_as(gt_ctr_x)) / ex_widths
        targets_dy = (gt_ctr_y - ex_ctr_y.view(1,-1).expand_as(gt_ctr_y)) / ex_heights
        targets_dw = torch.log(gt_widths / ex_widths.view(1,-1).expand_as(gt_widths))
        targets_dh = torch.log(gt_heights / ex_heights.view(1,-1).expand_as(gt_heights))

    elif ex_rois.dim() == 3:
        ex_widths = ex_rois[:, :, 2] - ex_rois[:, :, 0] + 1.0
        ex_heights = ex_rois[:,:, 3] - ex_rois[:,:, 1] + 1.0
        ex_ctr_x = ex_rois[:, :, 0] + 0.5 * ex_widths
        ex_ctr_y = ex_rois[:, :, 1] + 0.5 * ex_heights

        gt_widths = gt_rois[:, :, 2] - gt_rois[:, :, 0] + 1.0
        gt_heights = gt_rois[:, :, 3] - gt_rois[:, :, 1] + 1.0
        gt_ctr_x = gt_rois[:, :, 0] + 0.5 * gt_widths
        gt_ctr_y = gt_rois[:, :, 1] + 0.5 * gt_heights

        targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw = torch.log(gt_widths / ex_widths)
        targets_dh = torch.log(gt_heights / ex_heights)
    else:
        raise ValueError('ex_roi input dimension is not correct.')

    targets = torch.stack(
        (targets_dx, targets_dy, targets_dw, targets_dh),2)

    return targets

# def compute_targets(ex_rois, gt_rois, labels):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    BBOX_NORMALIZE_TARGETS_PRECOMPUTED = True
    BBOX_NORMALIZE_MEANS = (0.0, 0.0, 0.0, 0.0)
    BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)

    targets = bbox_transform(ex_rois, gt_rois)
    if BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
        targets = ((targets - np.array(BBOX_NORMALIZE_MEANS)) / np.array(BBOX_NORMALIZE_STDS))
    return np.hstack((labels[:, np.newaxis], targets)).astype(np.float32, copy=False)

def compute_targets_pytorch(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.size(1) == gt_rois.size(1)
    assert ex_rois.size(2) == 4
    assert gt_rois.size(2) == 4

    BBOX_NORMALIZE_TARGETS_PRECOMPUTED = cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED
    BBOX_NORMALIZE_MEANS = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
    BBOX_NORMALIZE_STDS = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS)

    batch_size = ex_rois.size(0)
    rois_per_image = ex_rois.size(1)

    targets = bbox_transform_batch(ex_rois, gt_rois)

    if BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
        targets = ((targets - BBOX_NORMALIZE_MEANS.expand_as(targets)) / BBOX_NORMALIZE_STDS.expand_as(targets))

    return targets

def get_bbox_regression_labels_pytorch(bbox_target_data, labels_batch, num_classes=2):
    """Bounding-box regression targets (bbox_target_data) are stored in a
    compact form b x N x (class, tx, ty, tw, th)
    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets).
    Returns:
        bbox_target (ndarray): b x N x 4K blob of regression targets
        bbox_inside_weights (ndarray): b x N x 4K blob of loss weights
    """

    BBOX_INSIDE_WEIGHTS = torch.FloatTensor(cfg.TRAIN.BBOX_INSIDE_WEIGHTS)
    batch_size = labels_batch.size(0)
    rois_per_image = labels_batch.size(1)
    clss = labels_batch
    bbox_targets = bbox_target_data.new(batch_size, rois_per_image, 4).zero_()
    bbox_inside_weights = bbox_target_data.new(bbox_targets.size()).zero_()

    for b in range(batch_size):
        # assert clss[b].sum() > 0
        if clss[b].sum() == 0:
            continue
        inds = torch.nonzero(clss[b] > 0).view(-1)
        for i in range(inds.numel()):
            ind = inds[i]
            bbox_targets[b, ind, :] = bbox_target_data[b, ind, :]
            bbox_inside_weights[b, ind, :] = BBOX_INSIDE_WEIGHTS

    return bbox_targets, bbox_inside_weights

def sample_rois_tensor(all_rois, gt_boxes, fg_rois_per_image, rois_per_image, num_classes):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
        input: all_rois(tensor)(N,5)[id,l,t,r,b]
               gt_boxes(tensor)(K,5)[l,t,r,b,cls]
               fg_rois_per_image: the number of foreground rois per image
               rois_per_image: the number of rois per image
               num_classes(scalar): the number of classes (include background)
        output:(tuple)
                labels:
                rois:
    """
    # overlaps: R x G

    overlaps = bbox_overlaps(all_rois[:,1:], gt_boxes[:,:-1])
    # print(overlaps)
    # overlaps = overlaps.numpy()
    # gt_assignment = overlaps.argmax(axis=1)  # R
    # max_overlaps = overlaps.max(axis=1)  # R
    max_overlaps, gt_assignment = torch.max(overlaps, 1)
    labels = gt_boxes[gt_assignment, 4]
    # print('labels')
    # print(labels)
    # preclude hard samples
    # ignore_inds = torch.empty(shape=(0), dtype=int)

    # Select foreground RoIs as those with >= FG_THRESH overlap
    # fg_inds = np.where(max_overlaps >= 0.5)[0]
    fg_inds = (max_overlaps >= cfg.TRAIN.FG_THRESH).nonzero().view(-1)
    # print('fg_inds')
    # print(fg_inds)
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
    bg_inds = (max_overlaps < cfg.TRAIN.BG_THRESH_HI).nonzero().view(-1)


    # bg_inds = np.setdiff1d(bg_inds, ignore_inds)
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size(0))
    # Sample background regions without replacement
    if bg_inds.size(0) > 0:
        # bg_inds = np.random.choice(bg_inds, size=bg_rois_per_this_image, replace=False)
        bg_inds = bg_inds[:bg_rois_per_this_image]

    # The indices that we're selecting (both fg and bg)
    # keep_inds = np.append(fg_inds, bg_inds)
    keep_inds = torch.cat((fg_inds, bg_inds))
    # Select sampled values from various arrays:

    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[fg_rois_per_this_image:] = 0
    rois = all_rois[keep_inds]
    gt_rois = gt_boxes[gt_assignment[keep_inds]]
    # bbox_target_data = compute_targets_pytorch(rois[:, 1:5], gt_rois[:,:4])
    #
    # bbox_targets, bbox_inside_weights = _get_bbox_regression_labels_pytorch(bbox_target_data, labels, num_classes=2)

    return labels,rois,gt_rois

def smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):

    sigma_2 = sigma ** 2
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = torch.abs(in_box_diff)
    smoothL1_sign = (abs_in_box_diff < 1. / sigma_2).detach().float()
    in_loss_box = torch.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                  + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = out_loss_box
    for i in sorted(dim, reverse=True):
      loss_box = loss_box.sum(i)
    loss_box = loss_box.mean()
    return loss_box

def bbox_transform_inv(boxes, deltas, batch_size):
    widths = boxes[:, :, 2] - boxes[:, :, 0] + 1.0
    heights = boxes[:, :, 3] - boxes[:, :, 1] + 1.0
    ctr_x = boxes[:, :, 0] + 0.5 * widths
    ctr_y = boxes[:, :, 1] + 0.5 * heights

    dx = deltas[:, :, 0::4]
    dy = deltas[:, :, 1::4]
    dw = deltas[:, :, 2::4]
    dh = deltas[:, :, 3::4]

    pred_ctr_x = dx * widths.unsqueeze(2) + ctr_x.unsqueeze(2)
    pred_ctr_y = dy * heights.unsqueeze(2) + ctr_y.unsqueeze(2)
    pred_w = torch.exp(dw) * widths.unsqueeze(2)
    pred_h = torch.exp(dh) * heights.unsqueeze(2)

    pred_boxes = deltas.clone()
    # x1
    pred_boxes[:, :, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, :, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, :, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, :, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes

def clip_boxes_batch(boxes, im_shape, batch_size):
    """
    Clip boxes to image boundaries.
    """
    num_rois = boxes.size(1)

    boxes[boxes < 0] = 0
    # batch_x = (im_shape[:,0]-1).view(batch_size, 1).expand(batch_size, num_rois)
    # batch_y = (im_shape[:,1]-1).view(batch_size, 1).expand(batch_size, num_rois)

    batch_x = im_shape[:, 1] - 1
    batch_y = im_shape[:, 0] - 1

    boxes[:,:,0][boxes[:,:,0] > batch_x] = batch_x
    boxes[:,:,1][boxes[:,:,1] > batch_y] = batch_y
    boxes[:,:,2][boxes[:,:,2] > batch_x] = batch_x
    boxes[:,:,3][boxes[:,:,3] > batch_y] = batch_y

    return boxes

def clip_boxes(boxes, batch_size):

    for i in range(batch_size):
        boxes[i,:,0::4].clamp_(0, cfg.IMAGE_WIDTH - 1) #left
        boxes[i,:,1::4].clamp_(0, cfg.IMAGE_HEIGHT - 1) #top
        boxes[i,:,2::4].clamp_(0, cfg.IMAGE_WIDTH - 1) #right
        boxes[i,:,3::4].clamp_(0, cfg.IMAGE_HEIGHT - 1) #bottom

    return boxes
