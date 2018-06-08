import numpy as np
import tensorflow as tf

def bbox_transform_inv(anchor, rpn_bbox):
    w = tf.substract(rpn_bbox[:,2], rpn_bbox[:,0]) + 1.0
    h = tf.substract(rpn_bbox[:,3], rpn_bbox[:,1]) + 1.0
    center_x = tf.add(rpn_bbox[:,0], w*0.5)
    center_y = tf.add(rpn_bbox[:,1], h*0.5)

    dx = anchor[:,0]
    dy = anchor[:,1]
    dw = anchor[:,2]
    dh = anchor[:,3]

    pred_center_x = tf.add(center_x, tf.multiply(dx,w))
    pred_center_y = tf.add(center_y, tf.multiply(dy,h))
    pred_w = tf.multiply(w,dw)
    pred_h = tf.multiply(h,dh)

    pred_boxes0 = tf.substract(pred_center_x, pred_w*0.5)
    pred_boxes1 = tf.substract(pred_center_y, pred_h*0.5)
    pred_boxes2 = tf.add(pred_center_x,pred_w * 0.5)
    pred_boxes3 = tf.add(pred_center_y, pred_h * 0.5)

    return tf.stack([pred_boxes0,pred_boxes1,pred_boxes2, pred_boxes3],axis= 1)

def clip_boxes(boxes, im):
    b0 = tf.minimum(boxes[:,0], im[0] -1)
    b1 = tf.minimum(boxes[:,1], im[1] -1)
    b2 = tf.minimum(boxes[:,2], im[0] -1)
    b3 = tf.minimum(boxes[:,3], im[1] -1)
    return tf.stack([b0,b1,b2,b3], axis= 1)

def bbox_transform(rois, gt_rois):
    rois_width  = rois[:,2] - rois[:,0]
    rois_height = rois[:,3] - rois[:,1]
    rois_ctr_x  = rois[:,0] + rois_width/2
    rois_ctr_y  = rois[:,1] + rois_height/2

    gt_width  = gt_rois[:,2] - gt_rois[:,0]
    gt_height = gt_rois[:,3] - gt_rois[:,1]
    gt_ctr_x  = gt_rois[:,0] + gt_width/2
    gt_ctr_y  = gt_rois[:,1] + gt_height/2

    targets_dx = (gt_ctr_x - rois_ctr_x)/rois_width
    targets_dy = (gt_ctr_y - rois_ctr_y)/rois_height
    targets_dw = np.log(gt_width/rois_width)
    targets_dh = np.log(gt_height/rois_height)

    targets = np.vstack((targets_dx,targets_dy,targets_dw,targets_dh)).transpose()
    return targets





def bbox_overlaps(boxes, gt_boxes):
    N = boxes.shape[0]
    K = gt_boxes.shape[0]
    overlaps = np.zeros((N,K),dtype=np.float)
    for k in range(K):
        gt_boxes_area = (gt_boxes[k, 2] - gt_boxes[k, 0]) * (gt_boxes[k, 3] - gt_boxes[k, 1])
        for n in range(N):
            iw = (min(boxes[n,2],gt_boxes[k,2]) - max(boxes[n,0],gt_boxes[k,0]) + 1)
            ih = (min(boxes[n,3],gt_boxes[k,3]) - max(boxes[n,1],gt_boxes[k,1]) + 1)
            intersection = (iw*ih)
            boxes_area = (boxes[n, 2] - boxes[n, 0] + 1) * (boxes[n, 3] - boxes[n, 1] + 1)
            overlaps[n,k] = intersection/(gt_boxes_area+boxes_area - intersection)
    return overlaps


def _sample_rois(all_rois,all_scores, gt_boxes,num_class):
    overlaps = bbox_overlaps(all_rois[:,1:5],gt_boxes[:,1:])
    gt_assignmet = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis= 1)

    labels = gt_boxes[gt_assignmet,0]

    fg_inds = np.where(max_overlaps >= 0.8)[0]
    bg_inds = np.where(max_overlaps < 0.6 and max_overlaps >= 0.0)

    keep_inds = np.append(fg_inds,bg_inds)
    labels = labels[keep_inds]
    rois = all_rois[keep_inds]
    roi_scores = all_scores[keep_inds]
    gt_boxes = gt_boxes[gt_assignment[keep_inds], 1:5]

    targets = bbox_transform(rois[:,1:5],gt_boxes)

    box_target_data = np.hstack((labels[:,np.newaxis],target)).astype(np.float32,copy=False)





