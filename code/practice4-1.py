import os
import numpy as np
import tensorflow as tf

# to deactivate warnings (if you didn't installed TF by source)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.set_random_seed(777)  # for reproducibility

x_data = [[0, 0],
          [0, 1],
          [1, 0],
          [1, 1]]
y_data = [[0],
          [1],
          [1],
          [0]]

x_data = np.array(x_data, dtype=np.float32)
y_data = np.array(y_data, dtype=np.float32)

X = tf.placeholder(tf.float32, [None, 2])
Y = tf.placeholder(tf.float32, [None, 1])

W1 = tf.Variable(tf.random_normal([2, 10]), name='weight1')
b1 = tf.Variable(tf.random_normal([10]), name='bias1')
layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([10, 10]), name='weight2')
b2 = tf.Variable(tf.random_normal([10]), name='bias2')
layer2 = tf.sigmoid(tf.matmul(layer1, W2) + b2)

W3 = tf.Variable(tf.random_normal([10, 10]), name='weight3')
b3 = tf.Variable(tf.random_normal([10]), name='bias3')
layer3 = tf.sigmoid(tf.matmul(layer2, W3) + b3)

W4 = tf.Variable(tf.random_normal([10, 10]), name='weight4')
b4 = tf.Variable(tf.random_normal([10]), name='bias4')
layer4 = tf.sigmoid(tf.matmul(layer3, W4) + b4)

W5 = tf.Variable(tf.random_normal([10, 10]), name='weight5')
b5 = tf.Variable(tf.random_normal([10]), name='bias5')
layer5 = tf.sigmoid(tf.matmul(layer4, W5) + b5)

W6 = tf.Variable(tf.random_normal([10, 10]), name='weight6')
b6 = tf.Variable(tf.random_normal([10]), name='bias4')
layer6 = tf.sigmoid(tf.matmul(layer5, W6) + b6)

learning = layer6

# cost/loss function
cost = -tf.reduce_mean(Y * tf.log(learning) + (1 - Y) *
                       tf.log(1 - learning))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-2)
train = optimizer.minimize(cost)

# Accuracy computation
# True if hypothesis>0.5 else False
predicted = tf.cast(learning > 0.99, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# Launch graph
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        sess.run(train, feed_dict={X: x_data, Y: y_data})
        if step % 100 == 0:
            print(step, sess.run(cost, feed_dict={
                  X: x_data, Y: y_data}), sess.run([cost, learning, train], feed_dict={X: x_data, Y: y_data}))

    # Accuracy report
    h, c, a = sess.run([learning, predicted, accuracy],
                       feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis: ", h, "\nCorrect: ", c, "\nAccuracy: ", a)
