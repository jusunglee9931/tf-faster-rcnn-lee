
import tensorflow as tf
import numpy as np
from utils import bbox_transform_inv_tf, bbox_transform, clip_boxes_tf, _unmap,_sample_rois,bbox_overlaps
from generate_anchors import generate_anchors


class network(object):
    def __init__(self):
        self._predictions = {}
        self._anchor_targets = {}
        self._proposal_targets={}
        self._debug = {}
        self._losses = {}
        self._layers = {}
        self._RPN_CHANNELS= 512
        self._anchor_ratio = [0.5,1,2,4,8]
        self._anchor_scale  = [4,8,16,32,48,64]
        self._num_anchors = len(self._anchor_scale)*len(self._anchor_ratio)
        self._threshold_for_label_zero = 0.4
        self._threshold_for_label_one = 0.6
        self._threshold_for_rois_fg    = 0.6
        self._threshold_for_rois_bg    = 0.4
        self._pool_size = 7



    def create_arch(self,is_training,num_class):
        self._image = tf.placeholder(tf.float32, shape=[1, None, None, 3])
        self._im_info = tf.placeholder(tf.float32, shape=[3])
        self._gt_boxes = tf.placeholder(tf.float32, shape=[None, 5])
        self._num_class = num_class + 1
        initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
        self.__build__network(initializer,is_training)

    def _reshape_layer(self, bottom, num_dim, name):
        input_shape = tf.shape(bottom)
        with tf.variable_scope(name) as scope:
            # change the channel to the caffe format
            to_caffe = tf.transpose(bottom, [0, 3, 1, 2])
            # then force it to have channel 2
            reshaped = tf.reshape(to_caffe,
                                  tf.concat(axis=0, values=[[1, num_dim, -1], [input_shape[2]]]))
            # then swap the channel back
            to_tf = tf.transpose(reshaped, [0, 2, 3, 1])
            return to_tf

    def _softmax_layer(self, bottom, name):
        if name.startswith('rpn_cls_prob_reshape'):
            input_shape = tf.shape(bottom)
            bottom_reshaped = tf.reshape(bottom, [-1, input_shape[-1]])
            reshaped_score = tf.nn.softmax(bottom_reshaped, name=name)
            return tf.reshape(reshaped_score, input_shape)
        return tf.nn.softmax(bottom, name=name)


    def __build__network(self,initializer,is_training):
        '''regino proposal block'''
        if is_training == True:
            self.post_nms_topN = 1200
            self.nms_thresh    = 0.7
        else:
            self.post_nms_topN = 300
            self.nms_thresh    = 0.7

        height = tf.to_int32(tf.ceil(self._im_info[0] / np.float32(self._feat_stride[0])))
        width = tf.to_int32(tf.ceil(self._im_info[1] / np.float32(self._feat_stride[0])))

        self.__generate_anchor(width,height,self._feat_stride)

        net_conv = self._image_to_head(is_training)

        rpn = tf.layers.conv2d(inputs = net_conv,filters = self._RPN_CHANNELS,kernel_size = [3,3],padding='same', kernel_initializer=initializer,name = 'rpn')
        rpn_cls_score = tf.layers.conv2d(inputs = rpn, filters = self._num_anchors*2,kernel_size = [1,1], padding='valid', kernel_initializer=initializer,name='rpn_cls_score')
        #rpn_cls_pred = tf.argmax(tf.reshape(rpn_cls_score,[-1,2]), axis= 1, name= 'rpn_cls_pred')
        #rpn_cls_prob = tf.nn.softmax(tf.reshape(rpn_cls_score,[-1,2], name= 'rpn_cls_prob'))
        rpn_cls_score_reshape = self._reshape_layer(rpn_cls_score, 2, 'rpn_cls_score_reshape')
        rpn_cls_prob_reshape = self._softmax_layer(rpn_cls_score_reshape, "rpn_cls_prob_reshape")
        rpn_cls_pred = tf.argmax(tf.reshape(rpn_cls_score_reshape, [-1, 2]), axis=1, name="rpn_cls_pred")
        rpn_cls_prob = self._reshape_layer(rpn_cls_prob_reshape, self._num_anchors * 2, "rpn_cls_prob")

        rpn_bbox_pred = tf.layers.conv2d(inputs=rpn, filters=self._num_anchors * 4, kernel_size=[1, 1], padding='valid', kernel_initializer=initializer,name='rpn_bbox_pred')

        self._predictions["rpn_cls_score"]  = rpn_cls_score
        self._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape
        self._predictions["rpn_cls_pred"]  = rpn_cls_pred
        self._predictions["rpn_cls_prob"]  = rpn_cls_prob
        self._predictions["rpn_bbox_pred"] = rpn_bbox_pred

        rois, scores = self.__proposal_layer(rpn_cls_prob, rpn_bbox_pred, self._anchors,self._im_info)

        if is_training:
            rpn_labels = self._anchor_target_layer(rpn_cls_score)
            with tf.control_dependencies([rpn_labels]):
                rois, _ = self._proposal_target_layer(rois, scores)
        self._predictions["rois"] = rois
        self._predictions["rois_score"] = scores

        pool = self._crop_pool_layer(net_conv,rois,'crop_layer')
        fc = self._head_to_tail(pool, is_training)
        cls_prob,bbox_pred =  self._region_classification_layer(fc,is_training,initializer)


        return rois , cls_prob, bbox_pred

    def _image_to_head(self , is_training, reuse=None):
        raise NotImplementedError
    def _head_to_tail(self, pool_last, is_training, reuse=None):
        raise NotImplementedError

    def __proposal_layer(self,rpn_cls_prob, rpn_bbox_pred, anchors,im):
        post_nms_topN = self.post_nms_topN
        nms_thresh = self.nms_thresh
        #print("rpn_cls_prob_shape")
        #print(rpn_cls_prob.shape)
        self._debug["rpn_cls_prob"] =tf.shape(rpn_cls_prob)

        scores = rpn_cls_prob[:, :, :, self._num_anchors:]
        self._debug["scores"] = tf.shape(scores)
        #print("scores")
        #print(scores.shape)

        scores = tf.reshape(scores, (-1,))
        rpn_bbox_pred = tf.reshape(rpn_bbox_pred,(-1,4))

        proposals = bbox_transform_inv_tf(anchors, rpn_bbox_pred)
        #proposals = bbox_transform_inv_tf(anchors)
        self._predictions["proposals"] = proposals
        proposals = clip_boxes_tf(proposals,im)


        indices = tf.image.non_max_suppression(proposals, scores, max_output_size = post_nms_topN, iou_threshold = nms_thresh)
        boxes = tf.gather(proposals,indices)
        boxes = tf.to_float(boxes)
        scores = tf.gather(scores,indices)
        scores = tf.reshape(scores,shape=(-1,1))

        batch_inds = tf.zeros((tf.shape(indices)[0],1),dtype= tf.float32)
        blob = tf.concat([batch_inds,boxes],1)
        '''
        batch_inds = tf.zeros((tf.shape(proposals)[0], 1), dtype=tf.float32)
        blob = tf.concat([batch_inds, proposals], 1)
        '''

        return blob, scores



    def _anchor_target_layer(self, rpn_cls_score):
        rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = tf.py_func(
            self.__anchor_target_layer,
            [rpn_cls_score, self._gt_boxes, self._im_info, self._feat_stride, self._anchors,self._num_anchors],
            [tf.float32, tf.float32, tf.float32, tf.float32],
            name="anchor_target")

        rpn_labels.set_shape([1, 1, None, 1])
        rpn_bbox_targets.set_shape([1, None, None, self._num_anchors * 4])
        rpn_bbox_inside_weights.set_shape([1, None, None, self._num_anchors * 4])
        rpn_bbox_outside_weights.set_shape([1, None, None, self._num_anchors * 4])

        rpn_labels = tf.to_int32(rpn_labels, name="to_int32")
        self._anchor_targets['rpn_labels'] = rpn_labels
        self._anchor_targets['rpn_bbox_targets'] = rpn_bbox_targets
        self._anchor_targets['rpn_bbox_inside_weights'] = rpn_bbox_inside_weights
        self._anchor_targets['rpn_bbox_outside_weights'] = rpn_bbox_outside_weights

        return rpn_labels


    def __anchor_target_layer(self,rpn_cls_score,gt_boxes,im_info,feat_stride,anchor,A):
        allowed_border = 0
        total_anchors = anchor.shape[0]
        height, width = rpn_cls_score.shape[1:3]

        inds_inside = np.where( (anchor[:,0] >= allowed_border) &
                                (anchor[:,1] >= allowed_border) &
                                (anchor[:,2] < im_info[1] + allowed_border) &
                                (anchor[:,3] < im_info[0] + allowed_border)
                               )[0]


        anchors = anchor[inds_inside,:]
        labels = np.empty((len(inds_inside),), dtype=np.float32)
        labels.fill(-1)
        #print("anchor detail..")
        #print(anchors)
        #print("gt_boxes detail..")
        #print(gt_boxes)

        overlaps = bbox_overlaps(anchors,gt_boxes)
        '''anchor class output..'''
        argmax_overlap = overlaps.argmax(axis= 1)
        max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlap]
        '''gt class output'''
        gt_argmax_overlaps = overlaps.argmax(axis=0)
        gt_max_overlaps    = overlaps[gt_argmax_overlaps,np.arange(overlaps.shape[1])]
        gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]
        #print("argmax_overlap")
        #print(np.where(argmax_overlap > 0))

        #print("gt_argmax_overlaps")
        #print(gt_argmax_overlaps.shape)
        #print(gt_argmax_overlaps)
        #print(anchors[gt_argmax_overlaps])


        labels[max_overlaps < self._threshold_for_label_zero] = 0
        labels[gt_argmax_overlaps] = 1
        #print(gt_argmax_overlaps.shape)
        labels[max_overlaps > self._threshold_for_label_one] = 1
        #print("label_one")
        #print(np.where(max_overlaps > self._threshold_for_label_one)[0].shape)



        fg_index = np.where(labels == 1)[0]
        bg_index = np.where(labels == 0)[0]
        fg_index_len =len(fg_index)
        bg_index_len =len(bg_index)
        '''always same ratio'''
        if 3* fg_index_len > bg_index_len:
            disable_inds = np.random.choice(fg_index,size = (3* fg_index_len - bg_index_len) ,replace = False)
        else:
            disable_inds = np.random.choice(bg_index,size = (bg_index_len - 3* fg_index_len) , replace = False)
        labels[disable_inds] = -1

        #print(np.where(labels == 0)[0].shape)
        #print(np.where(labels == 1)[0].shape)
        #print("gt_boxes[argmax_overlap,:]")

        #print(gt_boxes[argmax_overlap, :])
        #print(gt_boxes[argmax_overlap,:].shape)
        #print(gt_boxes[argmax_overlap,:][np.where(argmax_overlap > 0)])
        #print(argmax_overlap.shape)

        bbox_targets = bbox_transform(anchors,gt_boxes[argmax_overlap,:])
        bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
        bbox_inside_weights[labels == 1, :] = np.array([1.0,1.0,1.0,1.0])

        bbox_outside_weights = np.zeros((len(inds_inside),4), dtype=np.float32)
        num_example = np.sum(labels >= 0)
        positive_weight = np.ones((1,4)) * 1.0 / num_example
        negative_weight = np.ones((1,4)) * 1.0 / num_example

        bbox_outside_weights[labels == 1, :] = positive_weight
        bbox_outside_weights[labels == 0, :] = negative_weight


        labels          = _unmap(labels, total_anchors, inds_inside,fill = -1)
        bbox_targets    = _unmap(bbox_targets,total_anchors,inds_inside,fill = 0)
        bbox_inside_weights = _unmap(bbox_inside_weights,total_anchors,inds_inside,fill = 0)
        bbox_outside_weights= _unmap(bbox_outside_weights,total_anchors,inds_inside,fill = 0)


        # labels
        labels = labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2)
        labels = labels.reshape((1, 1, A * height, width))
        #print("label")
        #print(np.where(labels == 1)[0].shape)
        #print(np.where(labels == 0)[0].shape)

        # bbox_targets
        bbox_targets = bbox_targets.reshape((1, height, width, A * 4))

        # bbox_inside_weights
        bbox_inside_weights = bbox_inside_weights.reshape((1, height, width, A * 4))

        # bbox_outside_weights
        bbox_outside_weights = bbox_outside_weights.reshape((1, height, width, A * 4))


        return labels, bbox_targets, bbox_inside_weights, bbox_outside_weights


    def _proposal_target_layer(self,rpn_rois,rpn_scores):
        rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = tf.py_func(
            self.__proposal_target_layer,
            [rpn_rois, rpn_scores, self._gt_boxes, self._num_class,self._im_info],
            [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32],
            name="proposal_target")

        rois.set_shape([None, 5])
        roi_scores.set_shape([None])
        labels.set_shape([None, 1])
        bbox_targets.set_shape([None, self._num_class * 4])
        bbox_inside_weights.set_shape([None, self._num_class * 4])
        bbox_outside_weights.set_shape([None, self._num_class * 4])

        self._proposal_targets['rois'] = rois
        self._proposal_targets['labels'] = tf.to_int32(labels, name="to_int32")
        self._proposal_targets['bbox_targets'] = bbox_targets
        self._proposal_targets['bbox_inside_weights'] = bbox_inside_weights
        self._proposal_targets['bbox_outside_weights'] = bbox_outside_weights


        return rois, roi_scores


    def __proposal_target_layer(self,rpn_rois,rpn_scores,gt_boxes,num_class,im):
        #zeros = np.zeros((gt_boxes.shape[0],1), dtype=gt_boxes.dtype)
        #rpn_rois  = np.vstack((rpn_rois,np.hstack((zeros,gt_boxes[:,:-1]))))
        #rpn_scores= np.vstack((rpn_scores,zeros))

        #print("rpn_rois_debug")
        #debug_idx = np.where(rpn_rois[:,3] - rpn_rois[:,1] < 0)
        #debug_idx2= np.where(rpn_rois[:,4] - rpn_rois[:,2] < 0)
        #print(debug_idx)
        #print(debug_idx2)#debug_idx = np.where(rpn_rois[:,3] - rpn_rois[:,1] < 0)
        #debug_idx2= np.where(rpn_rois[:,4] - rpn_rois[:,2] < 0)
        #print(debug_idx)
        #print(debug_idx2)
        labels, rois, roi_scores,bbox_targets, bbox_inside_weights = _sample_rois(
            rpn_rois,rpn_scores,gt_boxes,num_class,self._threshold_for_rois_fg, self._threshold_for_rois_bg,im
        )

        rois = rois.reshape(-1, 5)
        roi_scores = roi_scores.reshape(-1)
        labels = labels.reshape(-1, 1)
        bbox_targets = bbox_targets.reshape(-1, num_class * 4)
        bbox_inside_weights = bbox_inside_weights.reshape(-1, num_class * 4)
        bbox_outside_weights = np.array(bbox_inside_weights > 0).astype(np.float32)

        return rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights




    def _region_classification_layer(self,fc7,is_training, initializer):
        cls_score = tf.layers.dense(fc7, self._num_class,
                                         kernel_initializer=initializer,
                                         trainable=is_training,
                                         name='cls_score')

        cls_prob = self._softmax_layer(cls_score, "cls_prob")
        cls_pred = tf.argmax(cls_score, axis=1, name="cls_pred")
        bbox_pred = tf.layers.dense(fc7, self._num_class * 4,
                                         kernel_initializer=initializer,
                                         trainable=is_training,
                                         name='bbox_pred')

        self._predictions["cls_score"] = cls_score
        self._predictions["cls_pred"] = cls_pred
        self._predictions["cls_prob"] = cls_prob
        self._predictions["bbox_pred"] = bbox_pred

        return cls_prob, bbox_pred

    def _crop_pool_layer(self, bottom, rois, name):
        with tf.variable_scope(name) as scope:
            batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
            # Get the normalized coordinates of bboxes
            bottom_shape = tf.shape(bottom)
            height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(self._feat_stride[0])
            width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(self._feat_stride[0])
            x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
            y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
            x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
            y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height
            # Won't be back-propagated to rois anyway, but to save time
            bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], 1))
            crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [self._pool_size, self._pool_size],
                                             name="crops")
        return crops

    def __generate_anchor(self,width,height,feat_stride):

        anchors = generate_anchors(ratios =np.array(self._anchor_ratio), scales = np.array(self._anchor_scale))
        num_anchor = anchors.shape[0]

        shift_x    = tf.range(width)*feat_stride[0]
        shift_y    = tf.range(height)*feat_stride[0]
        shift_x, shift_y = tf.meshgrid(shift_x, shift_y)
        sx = tf.reshape(shift_x,shape=(-1,))
        sy = tf.reshape(shift_y,shape=(-1,))
        shifts = tf.transpose(tf.stack([sx,sy,sx,sy]))
        rect = tf.multiply(width, height)
        shifts = tf.transpose(tf.reshape(shifts,shape=[1,rect,4]), perm=(1,0,2))
        anchor_constant = tf.constant(anchors.reshape((1,num_anchor,4)), dtype= tf.int32)

        length = num_anchor*rect
        anchors_tf = tf.cast(tf.reshape(tf.add(anchor_constant,shifts),shape=(length,4)),dtype = tf.float32)
        anchors_tf.set_shape([None,4])
        length.set_shape([])
        self._anchors = anchors_tf

    def _add_losses(self, sigma_rpn=3.0):
        with tf.variable_scope('loss') as scope:
            # RPN, class loss
            rpn_cls_score = tf.reshape(self._predictions['rpn_cls_score_reshape'], [-1, 2])
            rpn_label = tf.reshape(self._anchor_targets['rpn_labels'], [-1])
            rpn_select = tf.where(tf.not_equal(rpn_label, -1))
            rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_select), [-1, 2])
            rpn_label = tf.reshape(tf.gather(rpn_label, rpn_select), [-1])
            rpn_cross_entropy = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label))

            # RPN, bbox loss
            rpn_bbox_pred = self._predictions['rpn_bbox_pred']
            rpn_bbox_targets = self._anchor_targets['rpn_bbox_targets']
            rpn_bbox_inside_weights = self._anchor_targets['rpn_bbox_inside_weights']
            rpn_bbox_outside_weights = self._anchor_targets['rpn_bbox_outside_weights']
            rpn_loss_box = self._smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                                rpn_bbox_outside_weights, sigma=sigma_rpn, dim=[1, 2, 3])

            self._losses["rpn_label"] = rpn_label
            self._losses["rpn_cls_score"] = rpn_cls_score


            self._losses['rpn_cross_entropy'] = rpn_cross_entropy
            self._losses['rpn_loss_box'] = rpn_loss_box

            cls_score = self._predictions["cls_score"]
            label = tf.reshape(self._proposal_targets["labels"], [-1])
            cross_entropy = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score, labels=label))

            bbox_pred = self._predictions['bbox_pred']
            bbox_targets = self._proposal_targets['bbox_targets']
            bbox_inside_weights = self._proposal_targets['bbox_inside_weights']
            bbox_outside_weights = self._proposal_targets['bbox_outside_weights']
            loss_box = self._smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)

            self._losses['cross_entropy'] = cross_entropy
            self._losses['loss_box'] = loss_box




            loss =  rpn_cross_entropy + rpn_loss_box + cross_entropy + loss_box
            #regularization_loss = tf.add_n(tf.losses.get_regularization_losses(), 'regu')
            self._losses['total_loss'] = loss #+ regularization_loss






            tf.summary.scalar('total_loss',loss)
            tf.summary.scalar('rpn_cross_entropy', rpn_cross_entropy)
            tf.summary.scalar('rpn_loss_box', rpn_loss_box)
            self.summary = tf.summary.merge_all()



        return loss


    def _smooth_l1_loss(self, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
        sigma_2 = sigma ** 2
        box_diff = bbox_pred - bbox_targets
        in_box_diff = bbox_inside_weights * box_diff
        abs_in_box_diff = tf.abs(in_box_diff)
        smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_box_diff, 1. / sigma_2)))
        in_loss_box = tf.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                      + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
        out_loss_box = bbox_outside_weights * in_loss_box
        loss_box = tf.reduce_mean(tf.reduce_sum(
            out_loss_box,
            axis=dim
        ))
        return loss_box







    def train_step(self, sess, blobs ,train_op):
        feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                 self._gt_boxes: blobs['gt_boxes']}
        rpn_loss_cls, rpn_loss_box, loss,loss_cls,loss_box,opt = sess.run([self._losses["rpn_cross_entropy"],
                                                                        self._losses['rpn_loss_box'],
                                                                        self._losses['total_loss'],
                                                                        self._losses['cross_entropy'],
                                                                        self._losses['loss_box'],
                                                                        train_op
                                                                        ],
                                                                       feed_dict=feed_dict)
        return loss,rpn_loss_box,rpn_loss_cls,loss_box,loss_cls
    def train_step_summary(self, sess, blobs ,train_op):
        feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                 self._gt_boxes: blobs['gt_boxes']}
        rpn_loss_cls, rpn_loss_box, loss, loss_cls, loss_box, summary,opt= sess.run([self._losses["rpn_cross_entropy"],
                                                                        self._losses['rpn_loss_box'],
                                                                        self._losses['total_loss'],
                                                                        self._losses['cross_entropy'],
                                                                        self._losses['loss_box'],
                                                                        self.summary,
                                                                        train_op,
                                                                        ],
                                                                       feed_dict=feed_dict)
        return loss,rpn_loss_box,rpn_loss_cls,loss_box,loss_cls,summary

    def test_image(self, sess, image, im_info):
        feed_dict = {self._image: image,
                     self._im_info: im_info}
        rois_score, rois,rpn_cls_prob ,cls_pred, bbox_pred= sess.run([self._predictions["rois_score"],
                           self._predictions["rois"],
                           self._predictions["rpn_cls_prob"],
                           self._predictions["cls_pred"],
                           self._predictions["bbox_pred"]
                        ],
                        feed_dict=feed_dict)

        return rois_score, rois, rpn_cls_prob,cls_pred,bbox_pred
    def test_image_train(self, sess, image, im_info,gt_boxes):
        feed_dict = {self._image: image,
                     self._im_info: im_info,
                 self._gt_boxes: gt_boxes}
        rois_score, rois,rpn_cls_prob = sess.run([self._predictions["rois_score"],
                           self._predictions["rois"],
                           self._predictions["rpn_cls_prob"],
                        ],
                        feed_dict=feed_dict)

        return rois_score, rois, rpn_cls_prob