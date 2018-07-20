from vgg16 import vgg16
from dataloader import dataloader
from utils import bbox_overlaps,bbox_transform_inv
import tensorflow as tf
import numpy as np
import os,random
from PIL import Image,ImageDraw,ImageFont
from resnet_v1 import resnetv1
from tensorflow.python import pywrap_tensorflow
from line_recog import line_recog

colorlist = ['white', 'red', 'blue', 'green', 'yellow', 'brown', 'purple', 'orange']

#os.environ['CUDA_VISIBLE_DEVICES'] = ''

threshold_overlaps = 0.5
threshold_score    = 0.5
MODEL_CKPT ="vgg16.ckpt"

def get_variables_in_checkpoint_file(file_name):
    try:
      reader = pywrap_tensorflow.NewCheckpointReader(file_name)
      var_to_shape_map = reader.get_variable_to_shape_map()
      return var_to_shape_map
    except Exception as e:  # pylint: disable=broad-except
      print(str(e))
      if "corrupted compressed block contents" in str(e):
        print("It's likely that your checkpoint file has been compressed "
              "with SNAPPY.")
def printmap(var_to_shape_map):
    for v in var_to_shape_map:
        print(v)

def construct_graph(net,sess):
    net.create_arch(True,1)
    loss = net._add_losses()
    lr = tf.Variable(0.001,  trainable=False)
    opt = tf.train.MomentumOptimizer(lr,0.09)
    graident = opt.compute_gradients(loss)
    train_opt= opt.apply_gradients(graident)

    return lr, train_opt

if __name__ == '__main__':
    loader = dataloader("../data/Challenge2_Training_Task12_Images", "../data/Challenge2_Training_Task1_GT")

    with tf.Session() as sess:
        #net = resnetv1(101)
        net = vgg16()
        net.create_arch(False, 1)
        #variables = tf.global_variables()
        #var_keep_dic = get_variables_in_checkpoint_file("./checkpoint.ckpt")
        #printmap(var_keep_dic)
        #variables_to_restore = net.get_variables_to_restore(variables, var_keep_dic)
        #restorer = tf.train.Saver(variables_to_restore)
        #restorer.restore(sess, MODEL_CKPT)

        #lr, train_opt = construct_graph(net, sess)
        saver =tf.train.Saver()
        saver.restore(sess,"./checkpoint.ckpt")
        #init = tf.global_variables_initializer()
        #sess.run(init)
        for i in range(0,150):
            blob = loader.fetch()
            roi_score, rois,rpn_cls_prob,cls_pred,bbox_pred = net.test_image(sess,blob["data"],blob["im_info"])
            #roi_score, rois, rpn_cls_prob = net.test_image_train(sess,blob["data"],blob["im_info"],blob['gt_boxes'])
            index = np.where(cls_pred == 1)[0]
            print("roi_score_num : "+str(bbox_pred.shape[0])+" roi_index_num : "+str(index.shape[0]) )
            #print(rois[index][:,1:5])
            bbox = bbox_transform_inv(rois[:,1:5],bbox_pred)
            print(bbox)
            print(bbox.shape)

            bbox = bbox[index]
            print(bbox.shape)

            #overlaps = bbox_overlaps(rois[index][:,1:5], blob["gt_boxes"])
            print("bbox_overlaps debug")
            #print(overlaps[np.where(overlaps>threshold_overlaps)])

            #print(rois[np.where(overlaps>threshold_overlaps)[0]])
            #high_prob_roi = rois[index][np.where(overlaps>threshold_overlaps)[0]]

            img = blob["pil_im"]
            brush = ImageDraw.Draw(img)
            for row in range(bbox.shape[0]):
                    box= [bbox[row,0], bbox[row,1], bbox[row,2], bbox[row,3] ]
                    c_idx = random.randrange(0, len(colorlist))
                    brush.rectangle([(box[0], box[1]), (box[2], box[3])], outline=colorlist[c_idx])

            img.save(str(i)+".jpg")


            #loss, rpn_loss_box, rpn_loss_cls = net.train_step(sess, blob, train_opt)
            #print("loss : %.6f" % loss)
            #print("rpn_loss_box : %.6f" % rpn_loss_box)
            #print("rpn_loss_cls : %.6f\n" % rpn_loss_cls)




