#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 15:46:43 2018

@author: YumingWu
"""

import hashlib
import os
import io

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
from IPython.display import display
from lxml import etree

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', '/home/YumingWu/Datasets/Butterfly_English', 'Root directory to raw Caltech-UCSD Birds dataset.')
flags.DEFINE_string('output_path', '/home/YumingWu/Datasets/Butterfly_English/butterfly', 'Path to output TFRecord')
FLAGS = flags.FLAGS

def get_label_dict(class_map):
    label_dict = {}
    with open(class_map, 'r') as file:
        while True:
            line = file.readline()
            if not line:
                break
                pass
            ID_tmp, name_tmp = [str(i) for i in line.split()]
            label_dict[name_tmp] = int(ID_tmp)
            pass
    return label_dict

def get_label_num(class_map):
    label_dict = {}
    with open(class_map, 'r') as file:
        while True:
            line = file.readline()
            if not line:
                break
                pass
            name_tmp, num_tmp = [str(i) for i in line.split()]
            label_dict[name_tmp] = int(num_tmp)
            pass
    return label_dict   

def generate_tf_example(image, data, dic_list):
    
    width = int(data['size']['width'])
    height = int(data['size']['height'])
    key = hashlib.sha256(image).hexdigest()
    
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes_text = []
    classes = []
    
    for obj in data['object']:

        xmin.append(float(obj['bndbox']['xmin']) / width)
        ymin.append(float(obj['bndbox']['ymin']) / height)
        xmax.append(float(obj['bndbox']['xmax']) / width)
        ymax.append(float(obj['bndbox']['ymax']) / height)
        classes_text.append(str(dic_list[obj['name']]).encode('utf8'))
        classes.append(dic_list[obj['name']])   
        
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(
            data['filename'].encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(
            data['filename'].encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(image),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return example

def conver_butterfly_to_tfrecords(data_dir, output_path):
    
    # path    
    image_path = os.path.join(data_dir, 'dataOneBBCleanTest')
    label_path = os.path.join(data_dir, 'Annotations')
    image_list = os.listdir(image_path)
    
    # read image and sort them into dir
    image_list = np.array(image_list)
    np.random.seed(42)
    permutation = np.random.permutation(image_list.shape[0])
    image_list_shuffled = image_list[permutation].tolist()
    assert len(image_list_shuffled) <= 680, "num of samples is wrong"
    dic_cn = get_label_dict('%s_label_cn.txt'%output_path)
    test_writer = tf.python_io.TFRecordWriter('%s_OneBBCleanTest.record'%output_path)
    for num in range(len(image_list_shuffled)):
        file = image_list_shuffled[num]
        image = os.path.join(image_path, file)
        label = os.path.join(label_path, file[:-4] + '.xml')
        with tf.gfile.GFile(image, 'rb') as fid:
            encoded_image = fid.read()

        with tf.gfile.GFile(label, 'r') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
        assert data['size']['width'] != 0, "Error in width of image %s"%data['filename']
        assert data['size']['height'] != 0, "Error in height of image %s"%data['filename']
        
        tf_example = generate_tf_example(encoded_image, data, dic_cn)
        test_writer.write(tf_example.SerializeToString())    
          
    print('num of samples: %d'%len(image_list_shuffled))              
    test_writer.close() 
    
def main(_):
    conver_butterfly_to_tfrecords(
            data_dir=FLAGS.data_dir, 
            output_path=FLAGS.output_path)

if __name__ == '__main__':
    tf.app.run()
