#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 20:02:27 2018

@author: JieweiLu

Function:
    This script is used to turn the image dataset into TFrecord
    
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import random

from dataDecoder import dataset_utils

dataName = 'clean'

# Seed for repeatability
_RANDOM_SEED = 0

class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg,
                     feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image




def main(_):
    print('Start:Using the %s dataset' % (dataName))
    
    # file_tfRecord is the file created by Yuming
    if dataName == 'clean':
        dataset_crop_path = '/home/JieweiLu/jackie/Program/ButterflyProgram/ButterflyProgram4/saveDataset/datasetKuo2Aug/image/'
        output_file = '/home/JieweiLu/jackie/Program/ButterflyProgram/ButterflyProgram4/saveDataset/datasetKuo2Aug/tfrecord/butterfly_kuo2.tfrecord'
    else:
        assert 1>2, 'I can not find the train or test set'
        print('Error')
    
    # initialization
    photo_filenames = []
    filenameAll = os.listdir(dataset_crop_path)
    for filename in filenameAll:
        path = os.path.join(dataset_crop_path, filename)
        photo_filenames.append(path)
    num_image = len(filenameAll)
    
    # shuffle the dataset
    random.seed(_RANDOM_SEED)
    random.shuffle(photo_filenames)
    
    
    with tf.Graph().as_default():
        image_reader = ImageReader()
        with tf.Session('') as sess:
            with tf.python_io.TFRecordWriter(output_file) as tfrecord_writer:
                
                # read an image each time
                for i in range(num_image):
                    print(i)
                    # Read the filename:
                    image_data = tf.gfile.FastGFile(photo_filenames[i], 'rb').read()
                    height, width = image_reader.read_image_dims(sess, image_data)
                    
                    # get the label make the label from 0 to 93
                    result = photo_filenames[i].split('_')
                    label = int(result[2])-1
                    
                    example = dataset_utils.image_to_tfexample(
                              image_data, b'jpg', height, width, label)
                    tfrecord_writer.write(example.SerializeToString())
                
    print('OK')
    
    
    

if __name__ == '__main__':
    tf.app.run()

