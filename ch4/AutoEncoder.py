import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def xavier_init(fan_in, fan_out, constant = 1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

class AdditiveGaussionNoiseAutoEncoder(object):
    '''Only 1 hidden layer and the number of unit of input and output is equal
    and is more than the hidden layer.'''

    def __init__(self, n_input, n_hidden, activation=tf.nn.softmax,
                optimizer=tf.train.AdamOptimizer(learning_rate=0.001), scale=0.1):
        # Initialize all parameters
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.activation = activation
        self.scale = scale
        self.weights = self._initialize_weights()
        self.sess = tf.Session()

        # Define reconstruction process
        self.x = tf.placeholder(tf.float32, [None, n_input])
        x_noised = self.x + self.scale * tf.random_normal((self.n_input, ))
        self.hidden = tf.matmul(x_noised, self.weights['W1']) + self.weights['b1']
        self.reconstruction = tf.matmul(self.hidden, self.weights['W2']) + self.weights['b2']
        self.cost = tf.reduce_sum(tf.pow(self.reconstruction - self.x, 2.0)) / 2
        self.optimizer = optimizer.minimize(self.cost)
        self.sess.run(tf.global_variables_initializer())

    def _initialize_weights(self):
        weights = {}
        weights['W1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden), tf.float32)
        weights['b1'] = tf.Variable(tf.zeros(self.n_hidden), tf.float32)
        weights['W2'] = tf.Variable(xavier_init(self.n_hidden, self.n_input), tf.float32)
        weights['b2'] = tf.Variable(tf.zeros(self.n_input), tf.float32)
        return weights

    def partial_fit(self, X):
        cost, _ = self.sess.run([self.cost, self.optimizer], feed_dict={self.x:X})
        return cost

    def total_cost(self, X):
        '''To calculate the total cost.'''
        return self.sess.run(self.cost, feed_dict={self.x:X})

    def transform(self, X):
        return self.sess.run(self.hidden, feed_dict={self.x:X})

    def generate(self, hidden=None):
        if hidden == None:
            hidden = np.random.normal(size = self.weights['b1'])
        return self.sess.run(self.reconstruction, feed_dict={self.hidden:hidden})

    def reconstruct(self, X):
        return self.sess.run(self.reconstruction, feed_dict={self.x:X})

    def getWeights(self):
        return self.weights

def standard_scale(X_train, X_test):
    '''A function to standardize the input.
    Note: we need to use same standard preprocessor'''
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test

def get_random_batch_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    np.random.shuffle(data)
    return data[start_index:(start_index+batch_size)]

if __name__=='__main__':
    AGNAE = AdditiveGaussionNoiseAutoEncoder(784, 200, scale=0.01)
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)
    n_samples = int(mnist.train.num_examples)
    training_epoch = 20
    batch_size = 128
    print('Start training!')
    for epoch in range(0, training_epoch):
        avg_cost = 0
        total_batch = int(n_samples / batch_size)
        for i in range(total_batch):
            mini_batch_x = get_random_batch_from_data(X_train, batch_size)
            cost = AGNAE.partial_fit(X_train)
            avg_cost += cost / n_samples * batch_size
        print("Epoch %d:" % (epoch + 1), 'cost = ', '{:.9f}'.format(avg_cost))
    print('Training finish!')
    print('Total cost = ', str(AGNAE.total_cost(X)))
