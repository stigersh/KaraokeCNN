import tensorflow as tf
# hello = tf.constant('Hello, TensorFlow!')
# sess = tf.Session()
# print(sess.run(hello))

x_size = 20500
lr = 0.2

# Step 2: create placeholders for input X (number of fire) and mask P  #prob
x = tf.placeholder(tf.float32, name='X')
P = tf.placeholder(tf.float32, name='P')


#define variables scope!!! reuse = True stuff
# with tf.device("/device:GPU:1"):
#   v = tf.get_variable("v", [1])
# Step 3: create weight and bias, initialized to 0
#layer 1
W1 = tf.get_variable('w1', [x_size, 100], initializer=tf.random_normal_initializer())
b1 = tf.get_variable('b1', [1,], initializer=tf.random_normal_initializer())
y1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)

#layer 2
W2 = tf.get_variable('w2',[x_size,10], initializer=
tf.random_normal_initializer())
b2 = tf.get_variable('b2',[1,], initializer=tf.random_normal_initializer())
y2 = tf.nn.softmax(tf.matmul(y1, W2) + b2)

#layer 3
W3 = tf.get_variable('w3',[x_size,10], initializer=
tf.random_normal_initializer())
b3 = tf.get_variable('b3',[1,], initializer=tf.random_normal_initializer())
y3 = tf.nn.softmax(tf.matmul(y2, W3) + b3)

#output
y = y3
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),
reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for _ in range(10000):
 #get batch func
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


    # Step 9: output the values of w and b
   # w, b = sess.run([w, b])

    # logits = tf.matmul(inputs, tf.transpose(weights))
# logits = tf.nn.bias_add(logits, biases)
# labels_one_hot = tf.one_hot(labels, n_classes)
# loss = tf.nn.sigmoid_cross_entropy_with_logits(
#     labels=labels_one_hot,
#     logits=logits)
# loss = tf.reduce_sum(loss, axis=1)