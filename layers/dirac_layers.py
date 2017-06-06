import tensorflow as tf
import numpy as np

def dirac_initializer(f_h, f_w, inputdim, outputdim):

	return 1

def crelu(x, name="crelu"):

	with tf.variable_scope(name) as scope:
		return tf.concat([tf.nn.relu(x), -tf.nn.relu(-x)], 3)



def dirac_conv2d(inputconv, o_d=64, f_h=5, f_w=5, s_h=1, s_w=1, stddev=0.02, padding="SAME", name="dirac_conv2d", do_norm=True, norm_type='batch_norm', do_relu=True, relufactor=0):

	with tf.variable_scope(name) as scope:

        weight = tf.get_variable("weight",[f_h, f_w, inputdim, outputdim], initializer=tf.truncated_normal_initializer(stddev=stddev))
		
		bias = tf.get_variable("bias",[outputdim], dtype=np.float32, initializer=tf.constant_initializer(0.0))

		dirac_weight = tf.get_variable("dirac_weight",[f_h, f_w, inputdim, outputdim],
		initializer=tf.constant_initializer(dirac_initializer(f_h, f_w, i`nputdim, outputdim)), trainable=False)

		alpha = tf.get_variable("alpha", 1, initializer=tf.constant_initializer(5.0))
		beta = tf.get_variable("beta", 1, initializer=tf.constant_initializer(1e-5))

		output_conv = tf.nn.conv2d(inputconv, alpha*dirac_weight + beta*weight, [1, s_h, s_w, 1])
