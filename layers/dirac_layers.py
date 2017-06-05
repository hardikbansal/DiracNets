import tensorflow as tf
import numpy as np

def dirac_conv2d(inputconv, o_d=64, f_h=5, f_w=5, s_h=1, s_w=1, stddev=0.02, padding="SAME", name="dirac_conv2d", do_norm=True, norm_type='batch_norm', do_relu=True, relufactor=0):

	with tf.variable_scope(name) as scope:

        weight = tf.get_variable("weight",[f_h, f_w, inputdim, outputdim])
