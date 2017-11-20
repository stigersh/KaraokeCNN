import tensorflow as tf
import numpy as np

from Params import options as opt

options = opt

# hello = tf.constant('Hello, TensorFlow!')
# sess = tf.Session()
# print(sess.run(hello))
n_iters = 10000


x_size = options.L*options.N_BINS
print(x_size)
lr = 0.2
batch_size = 100

from DataClass import DataClass
train_mixes_filename = 'train_mixes.csv'
train_masks_filename = 'train_masks.csv'


def build_net(x,y,x_size):

  # define variables scope!!! reuse = True stuff
  # with tf.device("/device:GPU:1"):
  #   v = tf.get_variable("v", [1])

  # layer 1

  W1 = tf.get_variable('w1', [x_size, x_size], initializer=tf.random_normal_initializer())
  b1 = tf.get_variable('b1', [x_size], initializer=tf.random_normal_initializer())
  y1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)

  #layer 2
  W2 = tf.get_variable('w2',[x_size,x_size], initializer=
  tf.random_normal_initializer())
  b2 = tf.get_variable('b2',[x_size], initializer=tf.random_normal_initializer())
  y2 = tf.nn.sigmoid(tf.matmul(y1, W2) + b2)

  #layer 3
  W3 = tf.get_variable('w3',[x_size,x_size], initializer=
  tf.random_normal_initializer())
  b3 = tf.get_variable('b3',[x_size], initializer=tf.random_normal_initializer())
  net_y = tf.matmul(y2, W3) + b3

  probs = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=net_y)
  cross_entropy = tf.reduce_mean(probs)

  train_step = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)
  return train_step,cross_entropy,probs

data_class = DataClass(train_mixes_filename,train_masks_filename,batch_size)
# create placeholders for input X (stft part) and mask y  #prob
x = tf.placeholder(tf.float32, name='X', shape=[None, x_size])
y = tf.placeholder(tf.float32, name='Y', shape=[None, x_size])

train_step,cross_entropy,probs = build_net(x,y,x_size)
loss_vec = np.zeros([n_iters,1])

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(n_iters):
    batch = data_class.get_batch()
    if len(batch)==0 :
      break #reset DataClass
    _, loss_val = sess.run([train_step, cross_entropy],
                             feed_dict={x: batch[0], y: batch[1]})
    loss_vec[i] = loss_val
    if i % 100 == 0:
      print('step %d, loss val %g' % (i, loss_val))

