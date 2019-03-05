#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 15:06:50 2018

@author: YumingWu
"""

#import os
from PIL import Image
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
from object_detection.data_decoders import tf_example_decoder
import tensorflow as tf  

def draw_box(image, box, filename, label):
#    image = np.asarray(image)
    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(image)
    for bbox in box:
        xmin = bbox[1] * image.shape[1]
        xmax = bbox[3] * image.shape[1]
        ymin = bbox[0] * image.shape[0]
        ymax = bbox[2] * image.shape[0]
        ax.add_patch(
            plt.Rectangle((xmin, ymin),
                      xmax - xmin,
                      ymax - ymin, fill=False,
                      edgecolor='red', linewidth=3.5))
    plt.axis('off')
    plt.tight_layout()
    plt.title([filename, label])
    plt.draw()

file_tfRecord = '/home/YumingWu/Datasets/ButterflyAugDataset/butterfly_aug.record'

queue = tf.train.string_input_producer([file_tfRecord])
reader = tf.TFRecordReader()
_,serialized_example = reader.read(queue)
decoder = tf_example_decoder.TfExampleDecoder()    
tensor_dict = decoder.decode(serialized_example)
#print(tensor_dict)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

filename = []
label_list = {}
diction = sess.run(tensor_dict)
name = diction['filename']
label = diction['groundtruth_classes'][0]
while name not in filename:
    filename.append(name)
    if label not in label_list:
        label_list[label] = 1
    else:
        label_list[label] += 1
    if np.random.random_sample() < 0.001:
        image_ = diction['image']
        bbox = diction['groundtruth_boxes']    
        xmin = bbox[0,1] * image_.shape[1]
        xmax = bbox[0,3] * image_.shape[1]
        ymin = bbox[0,0] * image_.shape[0]
        ymax = bbox[0,2] * image_.shape[0]
        draw_box(image_, bbox, name, label)
    diction = sess.run(tensor_dict)
    name = diction['filename']
    label = diction['groundtruth_classes'][0]
print(len(filename))
