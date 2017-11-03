import re
import random
import os.path
from glob import glob
import tensorflow as tf
from model import Model
import tensorflow.contrib.slim as slim
import scipy.misc
import numpy as np


def gt_process(gt_image):
    """
    Processing ground truth image for training data
    :param gt_image: Original ground truth image
    :return: ground truth data with shape(*gt_fg.shape, 2)
    """
    foreground_color = np.array([255, 0, 255])
    gt_fg = np.all(gt_image == foreground_color, axis=2)
    gt_fg = gt_fg.reshape(*gt_fg.shape, 1)
    gt_image = np.concatenate((np.invert(gt_fg), gt_fg), axis=2)
    return gt_image



def gen_batch_function(data_folder, image_shape):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    middle_shape = (int(image_shape[0]/2), int(image_shape[1]/2))
    low_shape = (int(image_shape[0]/4), int(image_shape[1]/4))
    gt2_shape = (int(image_shape[0]/8), int(image_shape[1]/8))
    gt1_shape = (int(image_shape[0]/16), int(image_shape[1]/16))

    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
        label_paths = {
            re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
            for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}

        random.shuffle(image_paths)
        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            gt_images = []
            for image_file in image_paths[batch_i:batch_i+batch_size]:
                #reading images
                gt_image_file = label_paths[os.path.basename(image_file)]
                image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
                gt=  scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)
                gt_image = gt_process(gt)

                #add to batch list
                images.append(image)
                gt_images.append(gt_image)

            yield np.array(images), np.array(gt_images)
    return get_batches_fn


def pre_process(input, gt, image_shape):
    """
    Transforming the size of input image and ground truth image
    :param input: High-resolution input image
    :param gt: Ground truth label
    :return: Scaled input values.
    """
    middle = tf.image.resize_images(input, (int(image_shape[0]/2), int(image_shape[1]/2)), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    low = tf.image.resize_images(input, (int(image_shape[0]/4), int(image_shape[1]/4)), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    gt3 = tf.image.resize_images(gt, (int(image_shape[0]/2), int(image_shape[1]/2)), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    gt2 = tf.image.resize_images(gt, (int(image_shape[0]/8), int(image_shape[1]/8)), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    gt1 = tf.image.resize_images(gt, (int(image_shape[0]/16), int(image_shape[1]/16)), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return middle, low, gt3, gt2, gt1

def train_nn():
    """
    Training ICNet
    :return:
    """
    #define variables
    num_classes = 2
    learning_rate = 0.001
    epochs = 300
    batch_size = 1
    image_shape = (320, 1152)


    # input and gt placeholder
    image_input = tf.placeholder(tf.float32, shape=( None, None, None, 3), name="img")
    correct_label4 = tf.placeholder(tf.float32, shape=( None, None, None, num_classes), name="label4")

    #model
    model = Model()
    middle_input, low_input, correct_label3, correct_label2, correct_label1 = pre_process(image_input, correct_label4, image_shape)
    cls1, cls2, cls3, cls4 = model.build_model(low_input, middle_input, image_input)

    #loss and train_op
    loss1 = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits = cls1, labels = correct_label1))
    loss2 = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits = cls2, labels = correct_label2))
    loss3 = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits = cls3, labels = correct_label3))
    loss4 = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits = cls4, labels = correct_label4))
    # cross_entropy_loss = loss1+loss2+loss3+loss4
    cross_entropy_loss = loss2+loss3+loss4
    train_op = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cross_entropy_loss)

    #get data function
    get_batches_fn = gen_batch_function('./data/road',image_shape )

    #train
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            print(epoch)
            for image, gt4 in get_batches_fn(batch_size):
                l1, l2, l3, l4, _ = sess.run([loss1, loss2, loss3, loss4, train_op],
                                        feed_dict={image_input: image,
                                                    correct_label4: gt4})
            if(epoch % 20 == 1):
                print("epoch: %d, loss1: %e, loss2: %e, loss3: %e, loss4: %e" %(epoch, l1, l2, l3, l4))

        #save trained model
        saver = tf.train.Saver()
        saver.save(sess, "./model/test_model")

        #prepare for result image
        im_softmax = sess.run(
            [cls4],
            {image_input: image})
        im_softmax = im_softmax[0][0][:, :,1].reshape(image_shape[0], image_shape[1])
        segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")

        #test image
        image_path = "./ex_data/um_000000.png"
        image = scipy.misc.imread(image_path)
        image_shape = (320, 1152)
        image = scipy.misc.imresize(image, image_shape)
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)
        scipy.misc.imsave('./output/post_train_output.png', street_im)


if __name__ == '__main__':
    train_nn()
