#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 14:27:38 2018

@author: JieweiLu
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def preprocess_image(image, output_height, output_width):
  """Preprocesses the given image.

  Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    is_training: `True` if we're preprocessing the image for training and
      `False` otherwise.

  Returns:
    A preprocessed image.
  """
  image = tf.to_float(image)
  #image = tf.expand_dims(image, 0) # this operation is needed to confirm
  image = tf.image.resize_images(image,(output_height, output_width))
  image = tf.subtract(image, 128.0)
  image = tf.div(image, 128.0)
  return image
