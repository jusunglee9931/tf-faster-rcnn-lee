import numpy as np
import tensorflow as tf


def bbox_transform_inv_tf(boxes, delta):
    w = tf.subtract(boxes[:,2], boxes[:,0]) + 1.0
    h = tf.subtract(boxes[:,3], boxes[:,1]) + 1.0
    center_x = tf.add(boxes[:,0], w*0.5)
    center_y = tf.add(boxes[:,1], h*0.5)

    dx = delta[:,0]
    dy = delta[:,1]
    dw = delta[:,2]
    dh = delta[:,3]

    pred_center_x = tf.add(center_x, tf.multiply(w,dx))
    pred_center_y = tf.add(center_y, tf.multiply(h,dy))
    pred_w = tf.multiply(tf.exp(dw),w)
    pred_h = tf.multiply(tf.exp(dh),h)

    pred_boxes0 = tf.subtract(pred_center_x, pred_w*0.5)
    pred_boxes1 = tf.subtract(pred_center_y, pred_h*0.5)
    pred_boxes2 = tf.add(pred_center_x, pred_w * 0.5)
    pred_boxes3 = tf.add(pred_center_y, pred_h * 0.5)

    return tf.stack([pred_boxes0,pred_boxes1,pred_boxes2, pred_boxes3],axis= 1)


def bbox_transform_inv(boxes, delta):
    w = boxes[:,2] - boxes[:,0] + 1.0
    h = boxes[:,3] - boxes[:,1] + 1.0
    center_x = boxes[:,0] + w*0.5
    center_y = boxes[:,1] + h*0.5

    dx = delta[:,0]
    dy = delta[:,1]
    dw = delta[:,2]
    dh = delta[:,3]

    pred_center_x = center_x + w * dx
    pred_center_y = center_y + h * dy
    pred_w = np.exp(dw) * w
    pred_h = np.exp(dh) * h

    pred_boxes0 = pred_center_x - pred_w* 0.5
    pred_boxes1 = pred_center_y - pred_h* 0.5
    pred_boxes2 = pred_center_x + pred_w * 0.5
    pred_boxes3 = pred_center_y + pred_h * 0.5

    return np.hstack([pred_boxes0.reshape(-1,1),pred_boxes1.reshape(-1,1),pred_boxes2.reshape(-1,1), pred_boxes3.reshape(-1,1)])
def clip_boxes_tf(boxes, im):
    b0 = tf.maximum(tf.minimum(boxes[:, 0], im[1] - 1), 0)
    b1 = tf.maximum(tf.minimum(boxes[:, 1], im[0] - 1), 0)
    b2 = tf.maximum(tf.minimum(boxes[:, 2], im[1] - 1), 0)
    b3 = tf.maximum(tf.minimum(boxes[:, 3], im[0] - 1), 0)
    return tf.stack([b0,b1,b2,b3], axis= 1)

def clip_boxes(boxes, im):
    b0 = np.maximum(np.minimum(boxes[:, 0], im[1] - 1), 0)
    b1 = np.maximum(np.minimum(boxes[:, 1], im[0] - 1), 0)
    b2 = np.maximum(np.minimum(boxes[:, 2], im[1] - 1), 0)
    b3 = np.maximum(np.minimum(boxes[:, 3], im[0] - 1), 0)
    return np.stack([b0,b1,b2,b3], axis= 1)

def bbox_transform_tf(rois, gt_rois):
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
    targets_dw = tf.log(gt_width/rois_width)
    targets_dh = tf.log(gt_height/rois_height)

    targets = tf.stack((targets_dx,targets_dy,targets_dw,targets_dh),axis=1).transpose()
    return targets

def bbox_transform(rois, gt_rois):


    rois_width  = rois[:,2] - rois[:,0] + 1.0
    rois_height = rois[:,3] - rois[:,1] + 1.0
    rois_ctr_x  = rois[:,0] + rois_width/2
    rois_ctr_y  = rois[:,1] + rois_height/2

    gt_width  = gt_rois[:,2] - gt_rois[:,0] + 1.0
    gt_height = gt_rois[:,3] - gt_rois[:,1] + 1.0
    gt_ctr_x  = gt_rois[:,0] + gt_width/2
    gt_ctr_y  = gt_rois[:,1] + gt_height/2

    targets_dx = (gt_ctr_x - rois_ctr_x)/rois_width
    targets_dy = (gt_ctr_y - rois_ctr_y)/rois_height
    targets_dw = np.log(gt_width/rois_width)
    targets_dh = np.log(gt_height/rois_height)
    #print(targets_dx.shape)
    #print(gt_ctr_x.shape)
    #print(rois_ctr_x.shape)

    targets = np.stack((targets_dx,targets_dy,targets_dw,targets_dh),axis=1).reshape(-1,4)
    #print(targets.shape)
    return targets



def bbox_overlaps(boxes, gt_boxes):
    #print("bbox_boxes_shape")
    #print(boxes.shape)

    gt_boxes = gt_boxes[:,0:4]
    boxes    = boxes.reshape((-1,4))

    #N = boxes.shape[0]
    #K = gt_boxes.shape[0]

    #print(gt_boxes.shape)
    #print(boxes.shape)

    x11,y11,x12,y12 = np.hsplit(boxes,4)
    x21,y21,x22,y22 = np.hsplit(gt_boxes,4)

    #print(boxes)
    #print(x22)

    #print(x21.reshape((1,-1)).shape)


    xI1 = np.maximum(x11, x21.reshape((1,-1)))
    yI1 = np.maximum(y11, y21.reshape((1,-1)))

    xI2 = np.minimum(x12, x22.reshape((1,-1)))
    yI2 = np.minimum(y12, y22.reshape((1,-1)))

    intersection = np.maximum(0,(xI2 - xI1 + 1)) * np.maximum(0,(yI2 - yI1 + 1))
    box_area = (x12 - x11 + 1) * (y12 - y11 + 1)
    gt_box_area = (x22 - x21 + 1) * (y22 - y21 + 1)


    union = (box_area + gt_box_area.reshape((1,-1))) - intersection
    union[np.where(union <= 0)] = -1

    overlaps = intersection / union
    #print("bbox_overlaps debug")
    #print(np.where(overlaps > 0.8))
    #print(overlaps[np.where(overlaps>0.4)])


    return np.maximum(overlaps,0)


def _sample_rois(all_rois,all_scores, gt_boxes,num_class,threshold_fg, threshold_bg,im):
    #find overlaps -> cal iou..
    #print(gt_boxes)
    #print(all_rois)
    overlaps = bbox_overlaps(all_rois[:,1:5],gt_boxes)
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis= 1)

    labels = gt_boxes[gt_assignment,-1]

    #get index of fg,bg based on argmax
    fg_index = np.where(max_overlaps >= threshold_fg)[0]
    bg_index = np.where(max_overlaps < threshold_bg)[0]

   # print(fg_index)
   # print(bg_index)

    fg_index_len = len(fg_index)
    bg_index_len = len(bg_index)

    if fg_index_len > bg_index_len:
        fg_index = np.random.choice(fg_index, size=(bg_index_len), replace=False)
    else:
        if fg_index_len == 0:
            bg_index = np.random.choice(bg_index, size=(1), replace=False)
        else:
            bg_index = np.random.choice(bg_index, size=(fg_index_len ), replace=False)




    keep_inds = np.append(fg_index,bg_index)
    #print(keep_inds)
    #print(np.where(labels == 0))
    labels = labels[keep_inds]
    labels[fg_index_len:] = 0
    rois = all_rois[keep_inds]
    print(labels)

    '''
    for debug only..
    '''
    
    #print("sample rois")
    #det_roi = all_rois[fg_inds]
    #print(det_roi)
    '''
    debug_idx = np.where(det_roi[:,3] - det_roi[:,1] < 0)
    debug_idx2= np.where(det_roi[:,4] - det_roi[:,2] < 0)
    print(debug_idx)
    print(debug_idx2)
    print(det_roi)
    #print("overlaps max...")
    #print(overlaps[debug_idx])
    '''
    roi_scores = all_scores[keep_inds]
    gt_boxes = gt_boxes[gt_assignment[keep_inds], 0:4]


    targets = bbox_transform(rois[:,1:5],gt_boxes)
    clip_boxes(targets,im)

    '''
    box_target_data = N x (class, tx, ty, tw, th)
    '''
    bbox_target_data = np.hstack((labels[:,np.newaxis],targets)).astype(np.float32,copy=False)




    clss = bbox_target_data[:,0]
    bbox_targets = np.zeros((clss.shape[0],4*num_class), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape,dtype= np.float32)
    inds = np.where(clss >0)[0]


    for ind in inds:
        cls = clss[ind]
        start = int(4*cls)
        end = start+4

        bbox_inside_weights[ind, start:end] = [1.0, 1.0, 1.0, 1.0]
        bbox_targets[ind, start:end] = bbox_target_data[ind ,  1: ]


    return labels, rois, roi_scores, bbox_targets,bbox_inside_weights




def _unmap(data, count, inds, fill=0):
  """ Unmap a subset of item (data) back to the original set of items (of
  size count) """
  if len(data.shape) == 1:
    ret = np.empty((count,), dtype=np.float32)
    ret.fill(fill)
    ret[inds] = data
  else:
    ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
    ret.fill(fill)
    ret[inds, :] = data
  return ret