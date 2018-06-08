from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
sys.path.insert(0,'../slim/')
import os
import json
import math
import time
import numpy as np
import tensorflow as tf
from nets import nets_factory
from preprocessing import preprocessing_factory
from PIL import Image
import cv2
import pandas as pd
slim = tf.contrib.slim

tf.logging.set_verbosity(tf.logging.INFO)

filenames = pd.read_csv(sys.argv[1],header=None).iloc[:,0].tolist()

for filename in filenames:
    image=cv2.imread(filename)
    res=cv2.resize(image,(299,299),interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(filename, res)
    
result = {}
with tf.Graph().as_default():
        tf_global_step = slim.get_or_create_global_step()
        network_fn = nets_factory.get_network_fn(
        'inception_resnet_v2',
        num_classes=(2),
        is_training=False)
        preprocessing_name = 'inception_resnet_v2'
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name,
            is_training=False)
        test_image_size = network_fn.default_image_size
        checkpoint_path = tf.train.latest_checkpoint('/home/paperspace/stanford/train_logs/')
        batch_size = 1
        tensor_input = tf.placeholder(tf.float32, [None, test_image_size, test_image_size, 3])
        logits, _ = network_fn(tensor_input)
        logits = tf.nn.top_k(logits, 2)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, checkpoint_path)
            for filename in filenames:
                image = tf.read_file(filename)
                image = tf.image.decode_jpeg(image, channels=0)
                processed_image = image_preprocessing_fn(image, test_image_size, test_image_size)
                processed_image = sess.run(processed_image)
                images = np.array([processed_image])
                predictions = sess.run(logits, feed_dict = {tensor_input : images})
                pre = predictions.values.argmax()
                result[filename] = pre
with open(sys.argv[2],'w') as file:
    for i in list(result.keys()):
        file.write(i+',')
        file.write(str(result[i]))
        file.write('\n')