import tensorflow as tf
import numpy as np
import os
import shutil
import time
import random
import sys

from layers import *
from ops import *

from tensorflow.examples.tutorials.mnist import input_data
from scipy.misc import imsave
from PIL import Image
from options import trainOptions
from tqdm import tqdm


class Dirac():


	def initialize(self):
		opt = trainOptions().parse()[0]
		self.batch_size = opt.batch_size
		self.img_width = opt.img_width
		self.img_height = opt.img_height
		self.img_depth = opt.img_depth
		self.dataset = opt.dataset
		self.model = "dirac"

		self.tensorboard_dir = "./output/" + self.model + "/" + self.dataset + "/tensorboard"
		self.check_dir = "./output/"+ self.model + "/" + self.dataset +"/checkpoints"
		self.images_dir = "./output/" + self.model + "/" + self.dataset + "/imgs"



	def model_setup(self):


		with tf.variable_scope("Model") as scope:

			self.input_x = tf.placeholder(tf.float32, [self.batch_size, self.img_size])


	def train():

		if self.dataset == 'mnist':
			self.n_samples = self.mnist.train.num_examples
			self.mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

	def test():

		



def main():

	model = Dirac()
	model.initialize()

	if(model.to_test):
		model.test()
	else:
		model.train()

if __name__ == "__main__":
	main()
