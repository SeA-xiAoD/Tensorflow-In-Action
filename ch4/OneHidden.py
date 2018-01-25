import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
sess = tf.InteractiveSession()

W1 = tf.Variable(tf.truncated_normal([784, 300], stddev=0.1))
b1 = tf.Variable(tf.zeros([300]))
W3 = tf.Variable(tf.truncated_normal([300, 10], stddev=0.1))
b3 = tf.Variable(tf.zeros([10]))

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

hidden1 = tf.nn.relu(tf.matmul(X, W1) + b1)
hidden_drop1 = tf.nn.dropout(hidden1, keep_prob)
output = tf.nn.softmax(tf.matmul(hidden_drop1, W3) + b3)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(output), reduction_indices=[1]))
train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)

sess.run(tf.global_variables_initializer())
for i in range(50000):
    mini_batch_x, mini_batch_y = mnist.train.next_batch(1000)
    train_step.run({X:mini_batch_x/255.0, Y:mini_batch_y, keep_prob:0.75})

correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval({X:mnist.test.images/255.0, Y:mnist.test.labels, keep_prob:1.0}))
