from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import tensorflow as tf

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import time
from tqdm import tqdm

######### setting ################ 
bLoad = False
bSave = False

trainEpoch = 500
evalEpoch = 5000
#################################

w = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

x = tf.placeholder("float", [None, 784])

y = tf.nn.softmax(tf.matmul(x,w) + b)

y_ = tf.placeholder("float",[None,10])

cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

cross_prediction =tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(cross_prediction,"float"))

sess = tf.Session()
saver = tf.train.Saver()

sess.run(tf.initialize_all_variables())

if bLoad == True:
	saver.restore(sess,'checkpoint/tmp1.ckpt')

############### accuracy check ####################
print "Trainig Start !!!"
accuracy_history = []
for i in tqdm(range(trainEpoch)):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x:batch_xs,y_:batch_ys})
	
    current_accuracy = sess.run(accuracy, feed_dict={x:mnist.test.images,y_:mnist.test.labels})
    accuracy_history.append(current_accuracy)
	
print "%3dth epoch %.2f accuracy\n" % (i+1, current_accuracy )
	
if bSave == True:
	save_path = saver.save(sess, 'checkpoint/tmp1.ckpt')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(accuracy_history)
fig.savefig('accuracy.png')



################ execution time check ################
print "Evalution Start !!!"
start_time = time.time()
for i in tqdm(range(evalEpoch)):
    current_accuracy = sess.run(accuracy, feed_dict={x:mnist.test.images,y_:mnist.test.labels})
processing_time = float(time.time() - start_time)
print "average processing time : %f second" % (processing_time / (evalEpoch+1))

	



