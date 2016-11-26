from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import tensorflow as tf

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

w = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

x = tf.placeholder("float", [None, 784])

y = tf.nn.softmax(tf.matmul(x,w) + b)

y_ = tf.placeholder("float",[None,10])

cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

sess = tf.Session()

sess.run(tf.initialize_all_variables())

accuracy_history = []

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x:batch_xs,y_:batch_ys})

    cross_prediction =tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(cross_prediction,"float"))
    accuracy_temp = sess.run(accuracy, feed_dict={x:mnist.test.images,y_:mnist.test.labels})
    print accuracy_temp
    accuracy_history.append(accuracy_temp)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(accuracy_history)
plt.legend()
fig.savefig('accuracy.png')

