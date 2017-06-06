import tensorflow as tf
import numpy as np
import optparse
import os
import shutil
import time
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.insert(0, os.path.abspath('..'))

from layers.basic_layers import *
from layers.dirac_layers import *

from tensorflow.examples.tutorials.mnist import input_data
from scipy.misc import imsave
from PIL import Image
from tqdm import tqdm


class Dirac():


	def run_parser(self):

		self.parser = optparse.OptionParser()

		self.parser.add_option('--num_iter', type='int', default=1000, dest='num_iter')
		self.parser.add_option('--batch_size', type='int', default=100, dest='batch_size')
		self.parser.add_option('--img_width', type='int', default=28, dest='img_width')
		self.parser.add_option('--img_height', type='int', default=28, dest='img_height')
		self.parser.add_option('--img_depth', type='int', default=1, dest='img_depth')
		self.parser.add_option('--num_groups', type='int', default=4, dest='num_groups')
		self.parser.add_option('--num_blocks', type='int', default=4, dest='num_blocks')
		self.parser.add_option('--max_epoch', type='int', default=20, dest='max_epoch')
		self.parser.add_option('--n_samples', type='int', default=50000, dest='n_samples')
		self.parser.add_option('--test', action="store_true", default=False, dest="test")
		self.parser.add_option('--steps', type='int', default=10, dest='steps')
		self.parser.add_option('--enc_size', type='int', default=256, dest='enc_size')
		self.parser.add_option('--dec_size', type='int', default=256, dest='dec_size')
		self.parser.add_option('--model', type='string', default="draw_attn", dest='model_type')
		self.parser.add_option('--dataset', type='string', default="mnist", dest='dataset')

	def initialize(self):

		self.run_parser()

		opt = self.parser.parse_args()[0]

		self.batch_size = opt.batch_size
		self.img_width = opt.img_width
		self.img_height = opt.img_height
		self.img_depth = opt.img_depth
		self.dataset = opt.dataset
		self.num_groups = opt.num_groups
		self.num_blocks = opt.num_blocks
		self.model = "dirac"
		self.to_test = opt.test

		self.tensorboard_dir = "./output/" + self.model + "/" + self.dataset + "/tensorboard"
		self.check_dir = "./output/"+ self.model + "/" + self.dataset +"/checkpoints"
		self.images_dir = "./output/" + self.model + "/" + self.dataset + "/imgs"



	def model_setup(self):


		with tf.variable_scope("Model") as scope:

			input_x = tf.placeholder(tf.float32, [self.batch_size, self.img_height, self.img_width, self.img_depth])
			input_pad = tf.pad(input_x, [[0, 0], [1, 3], [2, 3], [3, 0]])
			o_c1 = general_conv2d(input_pad, 48, 7, 7, 2, 2, name="conv_top")

			print(o_c1.shape)

			for group in range(0, self.num_groups):

				if(group == 0):
					o_loop = tf.nn.pool(o_c1, [3, 3], "MAX", "SAME", [1, 1], [2, 2], name="maxpool_"+str(group))
				else :
					o_loop = tf.nn.pool(o_loop, [3, 3], "MAX", "SAME", [1, 1], [2, 2], name="maxpool_"+str(group))

				for block in range(0, self.num_blocks):

					o_loop = ncrelu(o_loop, name="crelu_"+str(group)+"_"+str(block))
					print("In the group "+str(group)+ " and in the block "+ str(block) + " with dimension of o_loop as "+ str(o_loop.shape))
					# o_loop = general_conv2d(o_loop, 48, 3, 3, 1, 1, padding="SAME", name="conv_"+str(group)+"_"+str(block))
					o_loop = dirac_conv2d(o_loop, 48, 3, 3, 1, 1, name="conv_"+str(group)+"_"+str(block))



		# Printing the model variables

		self.model_vars = tf.trainable_variables()
		for var in self.model_vars: print(var.name, var.get_shape())




	def train(self):

		self.model_setup()

		sys.exit()

		if self.dataset == 'mnist':
			self.n_samples = self.mnist.train.num_examples
			self.mnist = input_data.read_data_sets('MNIST_data', one_hot=True)



	def test():

		return 1


def main():

	model = Dirac()
	model.initialize()

	if(model.to_test):
		model.test()
	else:
		model.train()

if __name__ == "__main__":
	main()
