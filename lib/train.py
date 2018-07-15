from vgg16 import vgg16
from dataloader import dataloader
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = ''

def construct_graph(net,sess,initial_lr = 0.001):
    net.create_arch(True,1)
    loss = net._add_losses()
    lr = tf.Variable(initial_lr,  trainable=False)
    opt = tf.train.MomentumOptimizer(lr,0.9)
    graident = opt.compute_gradients(loss)
    train_opt= opt.apply_gradients(graident)

    return lr, train_opt

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
if __name__ == '__main__':
    loader = dataloader("../data/Challenge2_Training_Task12_Images", "../data/Challenge2_Training_Task1_GT")
    MAX_ITERATION = 1000000
    LR_DECAY_ITERATION = 10000
    GAMMA = 0.8
    INITIAL_LR = 0.0001
    DISPLAY_ITERATION = 20


    with tf.Session() as sess:
        net = vgg16()
        lr, train_opt  = construct_graph(net,sess,INITIAL_LR)
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()
        if os.path.isfile("./checkpoint.ckpt.index"):
            ck_restore = tf.train.Saver()
            ck_restore.restore(sess, "./checkpoint.ckpt")
            print("checkpoint restored")
        else:
            variables = tf.global_variables()
            var_keep_dic = get_variables_in_checkpoint_file("vgg_16.ckpt")
            variables_to_restore = net.get_variables_to_restore(variables, var_keep_dic)
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, "vgg_16.ckpt")

        writer = tf.summary.FileWriter("./tensorboard",sess.graph)
        iteration = 1

        while iteration < MAX_ITERATION:
            blob = loader.fetch()
            if iteration % DISPLAY_ITERATION == 0:
                loss, rpn_loss_box, rpn_loss_cls,  rpn_cls_score,summary,_,_,_ = net.train_step_summary(sess, blob, train_opt)
                writer.add_summary(summary,float(iteration))
                print("iteration:%d" % iteration)
                print("loss : %.6f" % loss)
                print("rpn_loss_box : %.6f" % rpn_loss_box)
                print("rpn_loss_cls : %.6f\n" % rpn_loss_cls)
                #print(rois)

            else:
                loss,rpn_loss_box,rpn_loss_cls,rpn_cls_score,_,_,_ = net.train_step(sess,blob,train_opt)


            #print(rpn_cls_score)

            if iteration % 100 == 0:
                saver.save(sess,"./checkpoint.ckpt")


            if iteration % LR_DECAY_ITERATION == 0:
                INITIAL_LR *= GAMMA
                sess.run(tf.assign(lr,INITIAL_LR))

            iteration += 1






