import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.misc
import numpy as np

class Model:
	#build icnet model class
	def resize_nn(self, x, ratio):
		"""
	    Resizing image mianly for upsample
	    :param x: Input TF Tensor
	    :param ratio: Ratio for resizing, must be int
	    :return: Output for resized image(layer)
	    """
		s = tf.shape(x)
		h = s[1]
		w = s[2]
		return tf.image.resize_nearest_neighbor(x, [h * ratio, w * ratio])


	def conv(self, x, num_out_layers, kernel_size = 3, stride = 1, activation_fn=tf.nn.elu,
			 normalizer_fn=None):
		"""
	    Convolution layer
	    :param x: Input TF Tensor
		:param num_out_layers: Number of output filters
		:param kernel_size: Convolution kernel size
	    :param stride: Stride for convolution
	    :return: Convolution layer
	    """
		return slim.conv2d(x, num_out_layers, kernel_size, stride, 'SAME',
			activation_fn=activation_fn, normalizer_fn=normalizer_fn)

	def max_pool(self, x):
		"""
	    Max Pooling layer
	    :param x: Input TF Tensor
	    :return: Pooling layer
	    """
		return  slim.max_pool2d(x, [2, 2])

	def conv_block(self, x, num_out):
		"""
	    Convolution block
	    :param x: Input TF Tensor
		:param num_out_layers: Number of output filters
	    :return: Output of TF Tensor
	    """
		with tf.variable_scope("block1"):
			conv1 = self.conv(x, num_out)
			pool1 = self.max_pool(conv1)
		with tf.variable_scope("block2"):
			conv2 = self.conv(pool1, num_out*2)
			pool2 = self.max_pool(conv2)
		with tf.variable_scope("block3"):
			conv3 = self.conv(pool2, num_out*4)
			pool3 = self.max_pool(conv3)
		return pool3

	def dilate_conv (self, x, ratio, num_in, num_out ):
		"""
	    Dilate Convolution
	    :param x: Input TF Tensor
		:param ratio: Number of dilation
	 	:param num_in_layers: Number of input filters
		:param num_out_layers: Number of output filters
	    :return: Dilated layer
	    """
		dilate_filter =  tf.Variable(tf.truncated_normal([3, 3, num_in, num_out], stddev=0.01))
		conv1 = tf.nn.atrous_conv2d(x, filters = dilate_filter, rate=ratio, padding = 'SAME')
		return conv1

	def cff_block(self, small, big, num_in_layers):
		"""
	    Cascade Feature Fusion Block
	    :param small: Input Smaller TF Tensor
		:param big: Input Bigger TF Tensor
	 	:param num_in_layers: Number of input filters
	    :return: Fused Layer and Classfier Layer
	    """
		#both small and big has to have same depth layers
		upsample1 = self.resize_nn(small, 2)
		upsample2 = self.conv(upsample1, num_in_layers*2, normalizer_fn = slim.batch_norm)

		#projection conv 1x1
		projec = self.conv(big, num_in_layers*2, kernel_size = 1, normalizer_fn = slim.batch_norm)

		elementwise_sum1 = tf.add(upsample2, projec)
		elementwise_sum2 = tf.nn.relu(elementwise_sum1)

		classifier = self.conv(upsample1, 2, kernel_size = 1)
		softmax = tf.nn.softmax(classifier)
		return elementwise_sum2, classifier


	def build_model(self, low, middle, high):
		"""
	    Build ICNet
	    :param low: Low Resolution Input TF Tensor
		:param middle: Middle Resolution Input TF Tensor
	 	:param high: High Resolution Input TF Tensor
	    :return: classifier Layer * 4
	    """

		with tf.variable_scope("shared") as scope:
			low = self.conv_block(low, 32)
			scope.reuse_variables()
			middle = self.conv_block(middle, 32)
		with tf.variable_scope("dilated"):
			low = self.dilate_conv(low, 3, 32*4, 32*4*2)
			#reduce conv
			low = self.conv(low, 32*4)
		with tf.variable_scope("cff1"):
			integrate1, classifier1 = self.cff_block(low, middle, 32*4)

		with tf.variable_scope("high_res"):
			high = self.conv_block(high, 32*2)

		with tf.variable_scope("cff2"):
			integrate2, classifier2 = self.cff_block(integrate1, high, 32*4*2)

		with tf.variable_scope("decode"):
			upsample1 = self.resize_nn(integrate2,2)
			upsample2 = self.resize_nn(upsample1,2)
			pre_cls3 = self.conv(upsample2, 2, kernel_size = 1)
			classifier3 = tf.nn.softmax(pre_cls3)
			projec = self.conv(upsample2, 32*4*2, kernel_size = 1)
			upsample3 = self.resize_nn(upsample2,2)
			pre_cls4 = self.conv(upsample3, 2, kernel_size = 1)
			classifier4 = tf.nn.softmax(pre_cls4)
		return classifier1, classifier2, classifier3, classifier4

if __name__ == '__main__':
	#overfitting one image

	#loading image
	image_path = "./ex_data/um_000000.png"
	gt_image_file = "./ex_data/um_lane_000000.png"
	image = scipy.misc.imread(image_path)
	gt = scipy.misc.imread(gt_image_file)


	#input shape
	image_shape = (320, 1152)
	middle_shape = (int(image_shape[0]/2), int(image_shape[1]/2))
	low_shape = (int(image_shape[0]/4), int(image_shape[1]/4))
	gt2_shape = (int(image_shape[0]/8), int(image_shape[1]/8))
	gt1_shape = (int(image_shape[0]/16), int(image_shape[1]/16))


	#resize images
	image = scipy.misc.imresize(image, image_shape)
	middle_img = scipy.misc.imresize(image, middle_shape)
	low_img = scipy.misc.imresize(image, low_shape)
	gt4 = scipy.misc.imresize(gt, image_shape)
	gt3 = scipy.misc.imresize(gt, middle_shape)
	gt2 = scipy.misc.imresize(gt, gt2_shape)
	gt1 = scipy.misc.imresize(gt, gt1_shape)


	#gt processing
	def gt_process(gt_image):
		foreground_color = np.array([255, 0, 255])
		gt_fg = np.all(gt_image == foreground_color, axis=2)
		gt_fg = gt_fg.reshape(*gt_fg.shape, 1)
		gt_image = np.concatenate((np.invert(gt_fg), gt_fg), axis=2)
		return gt_image

	# Making 1 batch
	image = [image]
	middle_img = [middle_img]
	low_img = [low_img]
	gt4 = [gt_process(gt4)]
	gt3 = [gt_process(gt3)]
	gt2 = [gt_process(gt2)]
	gt1 = [gt_process(gt1)]

	# place holder
	num_classes = 2
	image_input = tf.placeholder(tf.float32, shape=( None, None, None, 3), name="img")
	middle_input = tf.placeholder(tf.float32, shape=( None, None, None, 3), name="middle_img")
	low_input = tf.placeholder(tf.float32, shape=( None, None, None, 3), name="low_img")
	correct_label1 = tf.placeholder(tf.float32, shape=( None, None, None, num_classes), name="label1")
	correct_label2 = tf.placeholder(tf.float32, shape=( None, None, None, num_classes), name="label2")
	correct_label3 = tf.placeholder(tf.float32, shape=( None, None, None, num_classes), name="label3")
	correct_label4 = tf.placeholder(tf.float32, shape=( None, None, None, num_classes), name="label4")

	#model
	model = Model()
	cls1, cls2, cls3, cls4 = model.build_model(low_input, middle_input, image_input)

	#loss
	learning_rate = 0.001
	loss1 = tf.reduce_mean(
		tf.nn.softmax_cross_entropy_with_logits(logits = cls1, labels = correct_label1))
	loss2 = tf.reduce_mean(
		tf.nn.softmax_cross_entropy_with_logits(logits = cls2, labels = correct_label2))
	loss3 = tf.reduce_mean(
		tf.nn.softmax_cross_entropy_with_logits(logits = cls3, labels = correct_label3))
	loss4 = tf.reduce_mean(
		tf.nn.softmax_cross_entropy_with_logits(logits = cls4, labels = correct_label4))
	cross_entropy_loss = loss1+loss2+loss3+loss4
	train_op = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cross_entropy_loss)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		#training one image
		for i in range(200):
			# print(i)
			train_loss , _ = sess.run([cross_entropy_loss, train_op],feed_dict={low_input: low_img,
													middle_input: middle_img,
													image_input: image,
													correct_label1: gt1,
													correct_label2: gt2,
													correct_label3: gt3,
													correct_label4: gt4})
			if(i % 20 == 0):
				print("epoch: %d, loss: %e" %(i, train_loss))
		#testing one image
		im_softmax = sess.run(
		            [cls4],
		            {low_input: low_img,
					middle_input: middle_img,
					image_input: image})

	#process output
	im_softmax = im_softmax[0][0][:, :,1].reshape(image_shape[0], image_shape[1])
	segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
	mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
	mask = scipy.misc.toimage(mask, mode="RGBA")

	#load(reload) test image
	image_path = "./ex_data/um_000000.png"
	image = scipy.misc.imread(image_path)
	image_shape = (320, 1152)
	image = scipy.misc.imresize(image, image_shape)

	#paste mask
	street_im = scipy.misc.toimage(image)
	street_im.paste(mask, box=None, mask=mask)
	scipy.misc.imsave('./output/output.png', street_im)
