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

def evaluate_epoch_error(sess,loss,mixes_filename,masks_filename,x,y,batch_size):
    data_class = DataClass(mixes_filename, masks_filename, batch_size)
    bnewEpoc = False
    tot_loss = 0
    counter = 0
    while bnewEpoc==False :
        batch, bnewEpoc = data_class.get_batch()
        if bnewEpoc:
            break
        counter+=1
        tot_loss += sess.run(loss,feed_dict={x: batch[0], y: batch[1]})
    tot_loss /= (counter*batch_size)
    return tot_loss

def evaluate_over_n_first_batches(sess,loss,mixes_filename,masks_filename,x,y,batch_size,n):
    data_class = DataClass(mixes_filename, masks_filename, batch_size)
    bnewEpoc = False
    tot_loss = 0
    counter = 0
    while counter < n :
        batch, bnewEpoc = data_class.get_batch()
        # print(batch[0][:10])
        if bnewEpoc:
            break
        counter+=1
        l = sess.run(loss,feed_dict={x: batch[0], y: batch[1]})
        tot_loss +=l
    tot_loss /= (counter*batch_size)
    return tot_loss
#----------------------------------
if __name__ == "__main__":
    options = opt

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

    loss_vec = []
    valid_vec = []
    #saving the model
    saver = tf.train.Saver()

    if not os.path.exists(options.model_dir):
        os.makedirs(options.model_dir)

    n_epochs = 0
    i = 0
    loss_test = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        while n_epochs < options.n_epochs :
            print(i)
            batch,bnewEpoc = data_class.get_batch()
            # print(batch[0][:10])
            if bnewEpoc:
                n_epochs+=1
            _, loss_val, probs_val = sess.run([train_step, loss, net_y],
                                              feed_dict={x: batch[0], y: batch[1]})
            loss_val2 = sess.run(loss,feed_dict={x: batch[0], y: batch[1]})
            #BUG!!! why loss_val2!=loss_Val??????
            print(loss_val)

            if i % options.save_iters == 0:
                loss_train = evaluate_over_n_first_batches(sess, loss, options.train_mixes_filename,
                                                           options.train_masks_filename, x, y, batch_size, i + 1)
                loss_vec.append(loss_train)

                loss_valid = evaluate_epoch_error(sess, loss, options.valid_mixes_filename,
                                                  options.valid_masks_filename, x, y, batch_size)
                valid_vec.append(loss_valid)
                print('step %d, loss train %g' % (i, loss_train))
                print('step %d, loss valid %g' % (i, loss_valid))

                save_path = saver.save(sess, options.model_dir+"/model_iter_"+str(i)+".ckpt")
                print("Model saved in file: %s" % save_path)
            i += 1

        save_path = saver.save(sess, options.model_dir+"/model_final.ckpt")
        print("Model saved in file: %s" % save_path)
        loss_test = evaluate_epoch_error(sess, options.test_mixes_filename, options.test_masks_filename, x, y,
                                         batch_size)

    lossfilename = options.model_dir+'/paramsAndloss.pckl'
    f = open(lossfilename, 'wb')
    pickle.dump(loss_vec, f)
    pickle.dump(valid_vec, f)
    pickle.dump(options, f)
    pickle.dump(loss_test, f)
    f.close()


# eval on test - currently train
# _, loss_val, probs_val = sess.run([loss, net_y],
#          feed_dict={x: test[0], y: test[1]})

