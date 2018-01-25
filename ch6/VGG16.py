from datetime import datetime
import math
import time
import tensorflow as tf

def conv_op(input_tensor, name, kh, kw, d_out, sh, sw, parameters):
    '''A function using to create convolution layer.
    input_tensor: a tensor of input
    name: the layer's name
    kh: conv kernel height
    kw: conv kernel width
    d_out: demention of output layer
    sh: the height of stride
    sw: the width of stride
    parameters: a list of parameters'''

    # get the demention of input layer
    d_in = input_tensor.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+'W',
                    shape=[kh, kw, d_in, d_out], dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases = tf.Variable(tf.constant(0, shape=[d_out], dtype=tf.float32),
                                                    trainable=True, name='b')
        conv = tf.nn.conv2d(input_tensor, kernel, [1,sh,sw,1], 'SAME')
        z = tf.nn.bias_add(conv, biases)
        a = tf.nn.relu(z, name=scope)
        parameters += [kernel, biases]
        return a


def fc_op(input_tensor, name, d_out, parameters):
    '''A function using to create fully connected layer.
    input_tensor: a tensor of input
    name: the layer's name
    d_out: the demention of output
    parameters: a list of parameters'''

    # get the demention of input layer
    d_in = input_tensor.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+'W',shape=[d_in,d_out], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.Variable(tf.constant(0.1, shape=[d_out], dtype=tf.float32),
                                                    trainable=True, name='b')
        a = tf.nn.relu_layer(input_tensor, kernel, biases, scope)
        parameters += [kernel, biases]
        return a


def mp_op(input_tensor, name, kh, kw, sh, sw):
    '''A function using to create max pool layer.'''

    return tf.nn.max_pool(input_tensor, [1,kh,kw,1], [1,sh,sw,1],
                                    padding='SAME', name=name)


def inference_op(input_data, keep_prob):
    '''A function to build VGG16.'''

    # Initialize parameters
    parameters = []

    # Block 1
    conv1_1 = conv_op(input_data, 'Conv1_1', 3, 3, 64, 1, 1, parameters)
    conv1_2 = conv_op(conv1_1, 'Conv1_2', 3, 3, 64, 1, 1, parameters)

    # Max Pooling 1
    mp_1 = mp_op(conv1_2, 'MaxPooling1', 2, 2, 2, 2)

    # Block 2
    conv2_1 = conv_op(mp_1, 'Conv2_1', 3, 3, 128, 1, 1, parameters)
    conv2_2 = conv_op(conv2_1, 'Conv2_2', 3, 3, 128, 1, 1, parameters)

    # Max Pooling 2
    mp_2 = mp_op(conv2_2, 'MaxPooling2', 2, 2, 2, 2)

    # Block 3
    conv3_1 = conv_op(mp_2, 'Conv3_1', 3, 3, 256, 1, 1, parameters)
    conv3_2 = conv_op(conv3_1, 'Conv3_2', 3, 3, 256, 1, 1, parameters)
    conv3_3 = conv_op(conv3_2, 'Conv3_3', 3, 3, 256, 1, 1, parameters)

    # Max Pooling 3
    mp_3 = mp_op(conv3_3, 'MaxPooling3', 2, 2, 2, 2)

    # Block 4
    conv4_1 = conv_op(mp_3, 'Conv4_1', 3, 3, 512, 1, 1, parameters)
    conv4_2 = conv_op(conv4_1, 'Conv4_2', 3, 3, 512, 1, 1, parameters)
    conv4_3 = conv_op(conv4_2, 'Conv4_3', 3 ,3 , 512, 1, 1, parameters)

    # Max Pooling 4
    mp_4 = mp_op(conv4_3, 'MaxPooling4', 2, 2, 2, 2)

    # Block 5
    conv5_1 = conv_op(mp_4, 'Conv5_1', 3, 3, 512, 1, 1, parameters)
    conv5_2 = conv_op(conv5_1, 'Conv5_2', 3, 3, 512, 1, 1, parameters)
    conv5_3 = conv_op(conv5_2, 'Conv5_3', 3, 3, 512, 1, 1, parameters)

    # Max Pooling 5
    mp_5 = mp_op(conv5_3, 'MaxPooling5', 2, 2, 2, 2)

    mp_5_shape = mp_5.get_shape().as_list()
    flatten = tf.reshape(mp_5, [-1, mp_5_shape[1]*mp_5_shape[2]*mp_5_shape[3]],
                                                                name='Flatten')

    # FC Layer 6
    fc_6 = fc_op(flatten, 'FC_6', 4096, parameters)
    fc_6_drop = tf.nn.dropout(fc_6, keep_prob, name='FC_6_Dropdout')

    # FC Layer 7
    fc_7 = fc_op(fc_6_drop, 'FC_7', 4096, parameters)
    fc_7_drop = tf.nn.dropout(fc_7, keep_prob, name='FC_7_Dropout')

    # FC Layer 8
    fc_8 = fc_op(fc_7_drop, 'FC_8', 1000, parameters)
    softmax = tf.nn.softmax(fc_8)
    predictions = tf.argmax(softmax, 1)

    return predictions, softmax, fc_8, parameters


def time_tensorflow_run(session, target, feed, info_string):
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0
    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target, feed_dict=feed)
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
        images = tf.Variable(tf.random_normal([batch_size,image_size,image_size,3]
                                            , dtype=tf.float32, stddev=0.1))

        keep_prob = tf.placeholder(tf.float32)
        predictions, softmax, fc_8, parameters = inference_op(images, keep_prob)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        time_tensorflow_run(sess, predictions, {keep_prob:1.0}, 'Forward')
        objective = tf.nn.l2_loss(fc_8)
        grad = tf.gradients(objective, parameters)
        time_tensorflow_run(sess, grad, {keep_prob:0.5}, 'Forward-Backward')

if __name__=='__main__':
    batch_size = 32
    num_batches = 100
    run_benchmark()
