# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A factory-pattern class which returns classification image/label pairs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim

_NUM_CLASSES = 108

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying size.',
    'label': 'A single integer between 0 and 107',
}


def get_dataset(dataset_dir,split_name_own):
  """Given a dataset name and a split_name_own returns a Dataset.

  Args:
    dataset_dir: The directory where the dataset files are stored.

  Returns:
    A `Dataset` class.

  Raises:
    ValueError: If the dataset `name` is unknown.
  """
  reader = tf.TFRecordReader
  
  keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
      'image/class/label': tf.FixedLenFeature(
          [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
  }

  items_to_handlers = {
      'image': slim.tfexample_decoder.Image(),
      'label': slim.tfexample_decoder.Tensor('image/class/label'),
  }
  
  decoder = slim.tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)
  labels_to_names = None
  
  if split_name_own == 'train':
      # include augmented or not
      num_samples_define = 12160
  elif split_name_own == 'test':
      num_samples_define = 699
  else:
      print('error')
  
  
  return slim.dataset.Dataset(
      data_sources=dataset_dir,
      reader=reader,
      decoder=decoder,
      num_samples=num_samples_define,
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      num_classes=_NUM_CLASSES,
      labels_to_names=labels_to_names)
