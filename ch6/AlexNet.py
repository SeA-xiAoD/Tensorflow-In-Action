'''
    This is a AlexNet program.
    I use random images as input not ImageNet's images to calculate the time
    cost when runing forward and forward-backward propagation.
'''


from datetime import datetime
import math
import time
import tensorflow as tf

batch_size = 32
num_batches = 100

# using to print the structure of tensor
def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())

def inference(images):
    parameters = []

    # Layer 1 [224, 224, 3] -> [27, 27, 96]
    with tf.name_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([11,11,3,96],
                                dtype=tf.float32, stddev=0.1), name='weights')
        biases = tf.Variable(tf.constant(0, shape=[96], dtype=tf.float32),
                                                trainable=True, name='biases')
        conv = tf.nn.conv2d(images, kernel, [1,4,4,1], 'SAME')
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope)
        print_activations(conv1)
        parameters += [kernel, biases]

    lrn1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001/9, beta=0.75, name='lrn1')
    pool1 = tf.nn.max_pool(lrn1, [1,3,3,1], [1,2,2,1], 'VALID')
    print_activations(pool1)

    # Layer 2 [27, 27, 96] -> [13, 13, 256]
    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5,5,96,256],
                                dtype=tf.float32, stddev=0.1), name='weights')
        biases = tf.Variable(tf.constant(0, shape=[256], dtype=tf.float32),
                                                trainable=True, name='biases')
        conv = tf.nn.conv2d(pool1, kernel, [1,1,1,1], 'SAME')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope)
        print_activations(conv2)
        parameters += [kernel, biases]

    lrn2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001/9, beta=0.75, name='lrn2')
    pool2 = tf.nn.max_pool(lrn2, [1,3,3,1], [1,2,2,1], 'VALID')
    print_activations(pool2)

    # Layer 3 [13, 13, 256] -> [13, 13, 384]
    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3,3,256,384],
                                dtype=tf.float32, stddev=0.1), name='weights')
        biases = tf.Variable(tf.constant(0, shape=[384], dtype=tf.float32),
                                                trainable=True, name='biases')
        conv = tf.nn.conv2d(pool2, kernel, [1,1,1,1], 'SAME')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope)
        print_activations(conv3)
        parameters += [kernel, biases]

    # Layer 4 [13, 13, 384] -> [13, 13, 384]
    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(tf.truncated_normal([3,3,384,384],
                                dtype=tf.float32, stddev=0.1), name='weights')
        biases = tf.Variable(tf.constant(0, shape=[384], dtype=tf.float32),
                                                trainable=True, name='biases')
        conv = tf.nn.conv2d(conv3, kernel, [1,1,1,1], 'SAME')
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope)
        print_activations(conv4)
        parameters += [kernel, biases]

    # Layer 5 [13, 13, 384] -> [6, 6, 256]
    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(tf.truncated_normal([3,3,384,256],
                                dtype=tf.float32, stddev=0.1), name='weights')
        biases = tf.Variable(tf.constant(0, shape=[256], dtype=tf.float32),
                                                trainable=True, name='biases')
        conv = tf.nn.conv2d(conv4, kernel, [1,1,1,1], 'SAME')
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope)
        print_activations(conv5)
        parameters += [kernel, biases]

    pool5 = tf.nn.max_pool(conv5, [1,3,3,1], [1,2,2,1], 'VALID')
    print_activations(pool5)

    # Layer 6 FC  [6, 6, 256] -> [9216] -> [4096]
    fc1 = tf.layers.flatten(pool5)
    W6 = tf.Variable(tf.truncated_normal([fc1.get_shape().as_list()[1],4096],
                                dtype=tf.float32, stddev=0.1), name='W6')
    b6  = tf.Variable(tf.constant(0, shape=[4096], dtype=tf.float32),
                                            trainable=True, name='b7')
    fc2 = tf.nn.relu(tf.matmul(fc1, W6) + b6)

    # Layer 7 FC [4096] -> [4096]
    fc1 = tf.layers.flatten(pool5)
    W7 = tf.Variable(tf.truncated_normal([4096,4096],
                                dtype=tf.float32, stddev=0.1), name='W7')
    b7  = tf.Variable(tf.constant(0, shape=[4096], dtype=tf.float32),
                                            trainable=True, name='b7')
    fc3 = tf.nn.relu(tf.matmul(fc2, W7) + b7)

    # Layer 8 FC [4096] -> [1000]
    fc1 = tf.layers.flatten(pool5)
    W8 = tf.Variable(tf.truncated_normal([4096, 1000],
                                dtype=tf.float32, stddev=0.1), name='W8')
    b8  = tf.Variable(tf.constant(0, shape=[1000], dtype=tf.float32),
                                            trainable=True, name='b8')
    output = tf.nn.softmax(tf.matmul(fc2, W8) + b8)

    return output, parameters

def time_tensorflow_run(session, target, info_string):
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0

    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target)
        duration = time.time() - start_time
        if i >= num_steps_burn_in:
            if i % 10 == 0:
                print('%s: step %d, duration = %.3f' %
                    (datetime.now(), i - num_steps_burn_in, duration))
            total_duration += duration
            total_duration_squared += duration * duration
    mn = total_duration / num_batches
    vr = total_duration_squared / num_batches - mn * mn
    sd = math.sqrt(vr)
    print('%s: %s acorss %d steps, %.3f +/- %.3f sec / batch' %
            (datetime.now(), info_string, num_batches, mn, sd))

def run_benchmark():
    with tf.Graph().as_default():
        image_size = 224
        images = tf.Variable(tf.random_normal([batch_size, image_size, image_size, 3]
                                            , dtype=tf.float32, stddev=0.1))

        output, parameters = inference(images)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        time_tensorflow_run(sess, output, 'Forward')

        objective = tf.nn.l2_loss(output)
        grad = tf.gradients(objective, parameters)
        time_tensorflow_run(sess, grad, 'Forward-Backward')

if __name__=='__main__':
    run_benchmark()
