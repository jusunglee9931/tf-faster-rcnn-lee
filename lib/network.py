import tensorflow as tf
from utils import bbox_transform_inv, clip_boxes

class network(object):
    def __init__(self):
        self._predictions = {}
        self._num_anchors
        self._RPN_CHANNELS




    def __build__network(self):
        net_conv = __image_to_head(training)
        rpn = tf.layers.conv2d(inputs = net_conv,filters = self._RPN_CHANNELS,kernel_size = [3,3],padding='valid')
        rpn_cls_score = tf.layers.conv2d(inputs = rpn, filters = self._num_anchors*2,kernel_size = [1,1], padding='valid')
        rpn_bbox_score = tf.layers.conv2d(inputs = rpn, filters= self._num_anchors*4,kernel_size = [1,1], padding='valid')
        rpn_cls_pred = tf.argmax(tf.reshape(rpn_cls_score,[-1,2]), axis= 1, name= 'rpn_cls_pred')
        rpn_cls_prob = tf.nn.softmax(tf.reshape(rpn_cls_socre,[-1,2], axis= 1, name= 'rpn_cls_prob'))

        self._predictions["rpn_cls_core"]  = rpn_cls_score
        self._predictions["rpn_bbox_score"]= rpn_bbox_score
        self._predictions["rpn_cls_pred"]  = rpn_cls_pred
        self._predictions["rpn_cls_prob"]  = rpn_cls_prob




    def __image_to_head(self , is_training, reuse=None):
        raise NotImplementedError
    def __head_to_tail(self, pool_last, is_training, reuse=None):
        raise NotImplementedError

    def __proposal_layer(self,rpn_cls_prob, rpn_bbox_pred, anchors, im, name):
        scores = rpn_cls_prob[:,:,:,num_anchors:]
        scores = tf.reshape(scores, (-1,))
        rpn_bbox_pred = tf.reshape(rpn_bbox_pred,(-1,4))

        proposals = bbox_transform_inv(anchors, rpn_bbox_pred)
        proposals = clip_boxes(proposals,im)

        indices = tf.image.non_max_suppression(proposals, scores, max_output_size = 300, iou_threshold = 0.5)
        boxes = tf.gather(proposals,indices)
        boxes = tf.to_float(boxes)
        scores = tf.gather(scores,indices)
        scores = tf.reshape(scores,shape=(-1,1))

        batch_inds = tf.zeros((tf.shape(indices)[0],1),dtype= tf.float32)
        blob = tf.concat([batch_inds,boxes],1)
        return blob, scores

