from vgg16 import vgg16
from dataloader import dataloader
import tensorflow as tf
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

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
        #saver =tf.train.Saver()
        #saver.restore(sess,"./checkpoint.ckpt")
        init = tf.global_variables_initializer()
        sess.run(init)
        for i in range(0,350):
            blob = loader.fetch()
            roi_score, rois = net.test_image(sess,blob["data"],blob["im_info"])
            index = np.where(rois[:,0] == 1)
            print(rois[:,0])
            #loss, rpn_loss_box, rpn_loss_cls = net.train_step(sess, blob, train_opt)
            #print("loss : %.6f" % loss)
            #print("rpn_loss_box : %.6f" % rpn_loss_box)
            #print("rpn_loss_cls : %.6f\n" % rpn_loss_cls)




