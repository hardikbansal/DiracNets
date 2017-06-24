import tensorflow as tf
import numpy as np
import optparse
import os
import shutil
import time
import random
import sys
import pickle
import wget
import tarfile


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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
		self.parser.add_option('--img_width', type='int', default=32, dest='img_width')
		self.parser.add_option('--img_height', type='int', default=32, dest='img_height')
		self.parser.add_option('--img_depth', type='int', default=3, dest='img_depth')
		self.parser.add_option('--num_groups', type='int', default=3, dest='num_groups')
		self.parser.add_option('--num_blocks', type='int', default=4, dest='num_blocks')
		self.parser.add_option('--num_test_images', type='int', default=100, dest='num_test_images')
		self.parser.add_option('--max_epoch', type='int', default=20, dest='max_epoch')
		self.parser.add_option('--n_samples', type='int', default=50000, dest='n_samples')
		self.parser.add_option('--test', action="store_true", default=False, dest="test")
		self.parser.add_option('--steps', type='int', default=10, dest='steps')
		self.parser.add_option('--enc_size', type='int', default=256, dest='enc_size')
		self.parser.add_option('--dec_size', type='int', default=256, dest='dec_size')
		self.parser.add_option('--model', type='string', default="draw_attn", dest='model_type')
		self.parser.add_option('--dataset', type='string', default="cifar-10", dest='dataset')
		self.parser.add_option('--dataset_folder', type='string', default="../../../datasets", dest='dataset_folder')

	def initialize(self):

		self.run_parser()

		opt = self.parser.parse_args()[0]

		self.max_epoch = opt.max_epoch
		self.batch_size = opt.batch_size
		self.dataset = opt.dataset

		if(self.dataset == 'cifar-10'):
			self.img_width = 32
			self.img_height = 32
			self.img_depth = 3
		elif(self.dataset == 'Imagenet'):
			self.img_width = 256
			self.img_height = 256
			self.img_depth = 3
		else :
			self.img_width = opt.img_width
			self.img_height = opt.img_height
			self.img_depth = opt.img_depth

		self.img_size = self.img_width*self.img_height*self.img_depth
		self.num_groups = opt.num_groups
		self.num_blocks = opt.num_blocks
		self.num_images_per_file = 10000
		self.num_files = 5
		self.num_images = self.num_images_per_file*self.num_files
		self.num_test_images = opt.num_test_images
		self.dataset_folder = opt.dataset_folder
		self.model = "dirac"
		self.to_test = opt.test
		self.load_checkpoint = False
		self.do_setup = True

		self.tensorboard_dir = "./output/" + self.model + "/" + self.dataset + "/tensorboard"
		self.check_dir = "./output/"+ self.model + "/" + self.dataset +"/checkpoints"
		self.images_dir = "./output/" + self.model + "/" + self.dataset + "/imgs"


	def load_cifar_dataset(self, mode='train'):


		if self.dataset=='cifar-10':

			if not os.path.isdir(os.path.join(os.path.dirname(__file__),self.dataset_folder+"/cifar-10-batches-py")):
				if not os.path.isdir(os.path.join(os.path.dirname(__file__),self.dataset_folder)):
					os.makedirs(os.path.join(os.path.dirname(__file__),self.dataset_folder))
				wget.download("https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",out=os.path.join(os.path.dirname(__file__),self.dataset_folder))
				tar = tarfile.open(os.path.join(os.path.dirname(__file__),self.dataset_folder+ "/cifar-10-python.tar.gz"))
				tar.extractall()
				tar.close()
				shutil.move("./cifar-10-batches-py",os.path.join(os.path.dirname(__file__),self.dataset_folder))



			if (mode == 'train'):

				self.train_images = np.zeros([self.num_images,self.img_size], dtype=np.float32)
				self.train_labels = np.zeros([self.num_images], dtype=np.int32)

				for i in range(0, 5):
					file_path = os.path.join(os.path.dirname(__file__), self.dataset_folder + "/cifar-10-batches-py/data_batch_" + str(i+1))
					print(file_path)
					with open(file_path, mode='rb') as file:
						data = pickle.load(file, encoding='bytes')
						temp_images = np.array(data[b'data'])
						temp_labels = np.array(data[b'labels']).astype(np.int32)
						self.train_images[i*self.num_images_per_file:(i+1)*self.num_images_per_file,:] = temp_images
						self.train_labels[i*self.num_images_per_file:(i+1)*self.num_images_per_file] = temp_labels
				
				self.train_images = np.reshape(self.train_images,[self.num_images, self.img_height, self.img_width, self.img_depth])

			elif (mode == 'test'):

				self.test_images = np.zeros([self.num_images_per_file,3072], dtype=np.float32)
				self.test_labels = np.zeros([self.num_images_per_file], dtype=np.int32)

				for i in range(0, 1):
					file_path = os.path.join(os.path.dirname(__file__), self.dataset_folder + "/cifar-10-batches-py/data_batch_" + str(i+1))
					print(file_path)
					with open(file_path, mode='rb') as file:
						data = pickle.load(file, encoding='bytes')
						temp_images = np.array(data[b'data'])
						temp_labels = np.array(data[b'labels']).astype(np.int32)
						self.test_images[i*self.num_images_per_file:(i+1)*self.num_images_per_file,:] = temp_images
						self.test_labels[i*self.num_images_per_file:(i+1)*self.num_images_per_file] = temp_labels

				self.test_images = np.reshape(self.test_images,[self.num_images, self.img_height, self.img_width, self.img_depth])
		else:
			print("Model not supported for this dataset")
			sys.exit()


	def normalize_input(self, imgs):

		return imgs/127.5-1.0

	def cifar_model_setup(self):

		self.input_imgs = tf.placeholder(tf.float32, [self.batch_size, self.img_height, self.img_width, self.img_depth])
		self.input_labels = tf.placeholder(tf.int32, [self.batch_size])

		input_pad = tf.pad(self.input_imgs, [[0, 0], [1, 1], [1, 1], [0, 0]])
		o_c1 = general_conv2d(input_pad, 16, 3, 3, 1, 1, name="conv_top")
		o_loop = tf.nn.relu(o_c1, name="relu_1")

		outdim = 16

		for group in range(0, self.num_groups):
			for block in range(0, self.num_blocks):

				o_loop = ncrelu(o_loop, name="crelu_"+str(group)+"_"+str(block))
				o_loop = dirac_conv2d(o_loop, outdim, 3, 3, 1, 1, name="conv_"+str(group)+"_"+str(block))
			
			if(group != self.num_groups-1):
				o_loop = tf.nn.pool(o_loop, [2, 2], "MAX", "VALID", None, [2, 2], name="maxpool_"+str(group))
			
			outdim = outdim*2

		temp_shape = o_loop.get_shape().as_list()
		o_avgpool = tf.nn.avg_pool(o_loop, [1, temp_shape[1], temp_shape[2], 1], [1, temp_shape[1], temp_shape[2], 1], "VALID", name="avgpool")
		temp_depth = o_avgpool.get_shape().as_list()[-1]
		self.final_output = linear1d(tf.reshape(o_avgpool, [self.batch_size, temp_depth]), temp_depth, 10)

	def inet_model_setup(self):

		self.input_imgs = tf.placeholder(tf.float32, [self.batch_size, self.img_height, self.img_width, self.img_depth])
		self.input_labels = tf.placeholder(tf.int32, [self.batch_size])

		input_conv = general_conv2d(self.input_imgs, 48, 7, 7, 2, 2, padding="SAME", name="conv_top")

		outdim = 48

		for group in range(0, self.num_groups):

			if(group != 0):
				o_loop = tf.nn.pool(o_loop, [2, 2], "MAX", "VALID", None, [2, 2], name="maxpool_"+str(group))
			else : 
				input_pad = tf.pad(input_conv, [[0, 0], [1, 1], [1, 1], [0, 0]])
				o_loop = tf.nn.pool(o_loop, [3, 3], "MAX", "VALID", None, [2, 2], name="maxpool_"+str(group))

			for block in range(0, self.num_blocks):
				o_loop = ncrelu(o_loop, name="crelu_"+str(group)+"_"+str(block))
				o_loop = dirac_conv2d(o_loop, outdim, 3, 3, 1, 1, name="conv_"+str(group)+"_"+str(block))

			outdim = outdim*2

		o_relu = tf.nn.relu(o_loop)
		temp_shape = o_relu.get_shape().as_list()
		o_avgpool = tf.nn.avg_pool(o_relu, [1, temp_shape[1], temp_shape[2], 1], [1, temp_shape[1], temp_shape[2], 1], "VALID", name="avgpool")
		temp_depth = o_avgpool.get_shape().as_list()[-1]
		self.final_output = linear1d(tf.reshape(o_avgpool, [self.batch_size, temp_depth]), temp_depth, 100)

	def model_setup(self):


		with tf.variable_scope("Model") as scope:

			self.input_imgs = tf.placeholder(tf.float32, [self.batch_size, self.img_height, self.img_width, self.img_depth])
			self.input_labels = tf.placeholder(tf.int32, [self.batch_size])

			if (self.dataset == 'cifar-10'):
				self.cifar_model_setup()
			elif (self.dataset == 'Imagenet'):
				self.inet_model_setup()
			else :
				print("No such dataset exist. Exiting the program")
				sys.exit()

		self.model_vars = tf.trainable_variables()
		for var in self.model_vars: print(var.name, var.get_shape())

		self.do_setup = False

	def loss_setup(self):

		self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_labels, logits=self.final_output, name="Error_loss")
		self.loss = tf.reduce_mean(self.loss)

		optimizer = tf.train.AdamOptimizer(0.001, beta1=0.5)
		self.loss_optimizer = optimizer.minimize(self.loss)

		# Defining the summary ops
		self.cl_loss_summ = tf.summary.scalar("cl_loss", self.loss)
		# print(self.loss.shape)


	def train(self):

		self.model_setup()

		self.loss_setup()


		if self.dataset == 'cifar-10':
			self.load_cifar_dataset('train')
			self.normalize_input(self.input_imgs)
		else :
			print('No such dataset exist')
			sys.exit()

		init = tf.global_variables_initializer()
		saver = tf.train.Saver()

		if not os.path.exists(self.images_dir+"/train/"):
			os.makedirs(self.images_dir+"/train/")
		if not os.path.exists(self.check_dir):
			os.makedirs(self.check_dir)


		with tf.Session() as sess:

			sess.run(init)
			writer = tf.summary.FileWriter(self.tensorboard_dir)
			writer.add_graph(sess.graph)

			if self.load_checkpoint:
				chkpt_fname = tf.train.latest_checkpoint(self.check_dir)
				saver.restore(sess,chkpt_fname)

			for epoch in range(0, self.max_epoch):

				for itr in range(0, int(self.num_images/self.batch_size)):

					imgs = self.train_images[itr*self.batch_size:(itr+1)*(self.batch_size)]
					labels = self.train_labels[itr*self.batch_size:(itr+1)*(self.batch_size)]

					_, summary_str, cl_loss_temp = sess.run([self.loss_optimizer, self.cl_loss_summ, self.loss],feed_dict={self.input_imgs:imgs, self.input_labels:labels})

					print("In the iteration "+str(itr)+" of epoch "+str(epoch)+" with classification loss of " + str(cl_loss_temp))

					writer.add_summary(summary_str,epoch*int(self.num_images/self.batch_size) + itr)


				saver.save(sess,os.path.join(self.check_dir,"dirac"),global_step=epoch)


	def test():

		if(do_setup):
			self.model_setup()

		if self.dataset == 'cifar-10':
			self.load_cifar_dataset('test')
			self.normalize_input(self.input_imgs)
		else :
			print('No such dataset exist')
			sys.exit()


		init = tf.global_variables_initializer()

		if not os.path.exists(self.images_dir+"/test/"):
			os.makedirs(self.images_dir+"/test/")
		if not os.path.exists(self.check_dir):
			print("No checkpoint directory exist.")
			sys.exit()



		with tf.Session() as sess:

			sess.run(init)

			if self.load_checkpoint:
				chkpt_fname = tf.train.latest_checkpoint(self.check_dir)
				saver.restore(sess,chkpt_fname)

			for itr in range(0, int(self.num_test_images/self.batch_size)):

				imgs = self.train_images[itr*self.batch_size:(itr+1)*(self.batch_size)]
				labels = self.train_labels[itr*self.batch_size:(itr+1)*(self.batch_size)]

				test_output = sess.run([self.final_output],feed_dict={self.input_imgs:imgs, self.input_labels:labels})

				print(test_output)
				print(labels)


def main():

	model = Dirac()
	model.initialize()

	# model.load_dataset()

	# sys.exit()

	if(model.to_test):
		model.test()
	else:
		model.train()

if __name__ == "__main__":
	main()
