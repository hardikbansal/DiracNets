import tensorflow as tf
import numpy as np

def dirac_initializer(filter_height, filter_width, input_dim, output_dim):

	return 1

def ncrelu(x, name="crelu"):

	with tf.variable_scope(name) as scope:
		return tf.concat([tf.nn.relu(x), -tf.nn.relu(-x)], 3)



def dirac_conv2d(inputconv, output_dim=64, filter_height=5, filter_width=5, stride_height=1, stride_width=1, stddev=0.02, padding="SAME", name="dirac_conv2d"):

	with tf.variable_scope(name) as scope:

		input_dim = inputconv.get_shape()[-1]

		weight = tf.get_variable("weight",[filter_height, filter_width, input_dim, output_dim], initializer=tf.truncated_normal_initializer(stddev=stddev))		
		bias = tf.get_variable("bias",[output_dim], dtype=np.float32, initializer=tf.constant_initializer(0.0))

		dirac_weight = tf.get_variable("dirac_weight",[filter_height, filter_width, input_dim, output_dim], initializer=tf.constant_initializer(1.0), trainable=False)

		alpha = tf.get_variable("alpha", 1, initializer=tf.constant_initializer(5.0))
		beta = tf.get_variable("beta", 1, initializer=tf.constant_initializer(1e-5))

		output_conv = tf.nn.conv2d(inputconv, alpha*dirac_weight + beta*weight, [1, stride_height, stride_width, 1], padding="SAME")

		return output_conv
