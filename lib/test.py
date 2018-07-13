from vgg16 import vgg16
from dataloader import dataloader
from utils import bbox_overlaps
import tensorflow as tf
import numpy as np
import os,random
from PIL import Image,ImageDraw,ImageFont

colorlist = ['white', 'red', 'blue', 'green', 'yellow', 'brown', 'purple', 'orange']

from line_recog import line_recog
os.environ['CUDA_VISIBLE_DEVICES'] = ''

threshold_overlaps = 0.8
threshold_score    = 0.55
def construct_graph(net,sess):
    net.create_arch(True,1)
    loss = net._add_losses()
    lr = tf.Variable(0.001,  trainable=False)
    opt = tf.train.MomentumOptimizer(lr,0.9)
    graident = opt.compute_gradients(loss)
    train_opt= opt.apply_gradients(graident)

    return lr, train_opt

if __name__ == '__main__':
    loader = dataloader("/data/Challenge2_Training_Task12_Images", "/data/Challenge2_Training_Task1_GT")

    with tf.Session() as sess:
        net = vgg16()
        net.create_arch(False, 1)
        #lr, train_opt = construct_graph(net, sess)
        saver =tf.train.Saver()
        saver.restore(sess,"./checkpoint.ckpt")
        #init = tf.global_variables_initializer()
        #sess.run(init)
        for i in range(0,50):
            blob = loader.fetch()
            roi_score, rois,rpn_cls_prob = net.test_image(sess,blob["data"],blob["im_info"])
            #roi_score, rois, rpn_cls_prob = net.test_image_train(sess,blob["data"],blob["im_info"],blob['gt_boxes'])
            index = np.where(roi_score>threshold_score)[0]
            print("roi_score_num : "+str(roi_score.shape[0])+" roi_index_num : "+str(index.shape[0]) )
            print(rois[index][:,1:5])
            overlaps = bbox_overlaps(rois[index][:,1:5], blob["gt_boxes"])
            print("bbox_overlaps debug")
            print(overlaps[np.where(overlaps>threshold_overlaps)])
            print(rois[np.where(overlaps>threshold_overlaps)[0]])
            high_prob_roi = rois[index]#[np.where(overlaps>0.4)[0]]

            img = blob["pil_im"]
            brush = ImageDraw.Draw(img)
            for row in range(high_prob_roi.shape[0]):
                    box= [high_prob_roi[row,1],high_prob_roi[row,2],high_prob_roi[row,3],high_prob_roi[row,4]]
                    c_idx = random.randrange(0, len(colorlist))
                    brush.rectangle([(box[0], box[1]), (box[2], box[3])], outline=colorlist[c_idx])

            img.save(str(i)+".jpg")


            #loss, rpn_loss_box, rpn_loss_cls = net.train_step(sess, blob, train_opt)
            #print("loss : %.6f" % loss)
            #print("rpn_loss_box : %.6f" % rpn_loss_box)
            #print("rpn_loss_cls : %.6f\n" % rpn_loss_cls)




