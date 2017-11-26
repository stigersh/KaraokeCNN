import tensorflow as tf
import numpy as np
import pickle
from Params import options as opt
import os


from DataClass import DataClass

def model(x, x_size):
    # define variables scope!!! reuse = True stuff
    # with tf.device("/device:GPU:1"):
    #   v = tf.get_variable("v", [1])
    # layer 1
    with tf.variable_scope("my_net", reuse=tf.AUTO_REUSE):
        W1 = tf.get_variable('w1', [x_size, x_size], initializer=tf.random_normal_initializer())
        b1 = tf.get_variable('b1', [x_size], initializer=tf.random_normal_initializer())
        # y1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
        y1 = tf.nn.relu(tf.matmul(x, W1) + b1,'y1')
        Weights = [W1,b1]

        # # layer 2
        # W2 = tf.get_variable('w2', [x_size, x_size], initializer=tf.random_normal_initializer())
        # b2 = tf.get_variable('b2', [x_size], initializer=tf.random_normal_initializer())
        # y2 = tf.nn.relu(tf.matmul(y1, W2) + b2,'y2')
        # Weights.append([W2,b2])
        #
        # # # layer 3
        # W3 = tf.get_variable('w3', [x_size, x_size], initializer=tf.random_normal_initializer())
        # b3 = tf.get_variable('b3', [x_size], initializer=tf.random_normal_initializer())
        # Weights.append([W3, b3])
        # net_y = tf.nn.relu(tf.matmul(y2, W3) + b3)

        net_y = y1
        reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l1_regularizer(1.0),Weights) #can be l2 also

    return net_y,reg

#----------------------------------
if __name__ == "__main__":
    options = opt

    n_iters = options.n_iters#3
    x_size = options.L*options.N_BINS #4
    print(x_size)
    lr = options.lr
    batch_size = options.batch_size

    # build net
    # create placeholders for input X (stft part) and mask y  #prob
    x = tf.placeholder(tf.float32, name='x', shape=[None, x_size])
    y = tf.placeholder(tf.float32, name='y', shape=[None, x_size])
    net_y,reg = model(x,x_size)

    #calculate loss
    # Training computation: logits + cross-entropy loss.
    # probs = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=net_y)
    # cross_entropy = tf.reduce_mean(probs)
    loss = tf.nn.l2_loss(y - net_y)
    objective = loss + options.reg*reg
    # global_step = tf.Variable(0, trainable=False)
    # tf.train.exponential_decay(starter_learning_rate, global_step,
    # 100000, 0.96, staircase=True)

    train_step = tf.train.GradientDescentOptimizer(lr).minimize(objective)
    #----------------------------------
    data_class = DataClass(options.train_mixes_filename, options.train_masks_filename, batch_size)

    loss_vec = np.zeros([n_iters, 1])

    #saving the model
    saver = tf.train.Saver()

    if not os.path.exists(options.model_dir):
        os.makedirs(options.model_dir)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(0, n_iters):
            print(i)
            batch = data_class.get_batch()
            if len(batch) == 0:
                print('batches are finished')
                break
            _, loss_val, probs_val = sess.run([train_step, loss, net_y],
                                              feed_dict={x: batch[0], y: batch[1]})
            print(loss_val)
            loss_vec[i] = loss_val
            if i % 20 == 0:
                print('step %d, loss val %g' % (i, loss_val))
                save_path = saver.save(sess, "save_model/model_iter_"+str(i)+".ckpt")
                print("Model saved in file: %s" % save_path)
            if i== n_iters-1 :
                save_path = saver.save(sess, "/save_model/model_final.ckpt")


    print("Model saved in file: %s" % save_path)

    lossfilename = 'loss.pckl'
    f = open(lossfilename, 'wb')
    pickle.dump(loss_vec, f)
    f.close()


# eval on test - currently train
# _, loss_val, probs_val = sess.run([loss, net_y],
#          feed_dict={x: test[0], y: test[1]})

