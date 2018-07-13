from vgg16 import vgg16
from dataloader import dataloader
import tensorflow as tf
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
    MAX_ITERATION = 100000


    with tf.Session() as sess:
        net = vgg16()
        lr, train_opt  = construct_graph(net,sess)
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()

        writer = tf.summary.FileWriter("./tensorboard",sess.graph)
        iteration = 0
        while iteration < MAX_ITERATION:
            blob = loader.fetch()
            loss,rpn_loss_box,rpn_loss_cls = net.train_step(sess,blob,train_opt)
            print("loss : %.6f" %loss)
            print("rpn_loss_box : %.6f" % rpn_loss_box)
            print("rpn_loss_cls : %.6f\n" % rpn_loss_cls)

            if iteration % 1000 == 0:
                saver.save(sess,"./checkpoint.ckpt")
            iteration += 1






