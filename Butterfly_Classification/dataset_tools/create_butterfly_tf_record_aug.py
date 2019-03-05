#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 10:41:04 2018

@author: YumingWu
"""
import hashlib
import os

import tensorflow as tf
import numpy as np
from lxml import etree
import matplotlib.pyplot as plt
from PIL import Image

from object_detection.utils import dataset_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', '/home/YumingWu/Project/butterfly/experiment/dataButterflyKuo2', 'Root directory to raw butterfly dataset.')
flags.DEFINE_string('output_path', '/home/YumingWu/Project/butterfly/experiment/dataButterflyKuo2/butterfly', 'Path to output TFRecord')
FLAGS = flags.FLAGS

def get_label_list(class_map):
    label_list = []
    name_list = []
    with open(class_map, 'r') as file:
        while True:
            line = file.readline()
            if not line:
                break
                pass
            ID_tmp, name_tmp = [str(i) for i in line.split()]
            label_list.append(ID_tmp + '.' + name_tmp)
            name_list.append(name_tmp)
            pass
    return label_list, name_list

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

def draw_box(image, box, filename, label):
    image = Image.open(image)
    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(image)
    for bbox in box:
        xmin = bbox[0]
        xmax = bbox[2]
        ymin = bbox[1]
        ymax = bbox[3]
        ax.add_patch(
            plt.Rectangle((xmin, ymin),
                      xmax - xmin,
                      ymax - ymin, fill=False,
                      edgecolor='red', linewidth=3.5))
    plt.axis('off')
    plt.tight_layout()
    plt.title([filename, str(label)])
    plt.draw()
    plt.savefig('/home/YumingWu/Datasets/ButterflyAugDataset/bbox_display/%s'%(str(label) + '_' + filename))
    plt.close()

def generate_tf_example(image, data, class_id, filename, class_name):
    with tf.gfile.GFile(image, 'rb') as fid:
        encoded_image = fid.read()
    width = int(data['size']['width'])
    height = int(data['size']['height'])
    if width == 0 or height == 0:
        image_data = Image.open(image)
        width = image_data.width
        height = image_data.height
    key = hashlib.sha256(encoded_image).hexdigest()
    
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
        classes_text.append(str(class_id).encode('utf8'))
        classes.append(class_id)
#        print(class_id)
        
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(
            filename.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(
            filename.encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_image),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
#    print(example.features.feature['image/object/class/text'].bytes_list.value)
    return example

def conver_butterfly_to_tfrecords(data_dir, output_path):
    
    # path    
    class_path = os.path.join(data_dir, 'images')
    label_class_path = os.path.join(data_dir, 'annotations')
    class_list, class_name_list = get_label_list(
            '/home/YumingWu/Project/butterfly/experiment/dataButterflyKuo2/butterfly_kuo_dict.txt')
    class_dict = get_label_dict('/home/YumingWu/Datasets/Butterfly_English/butterfly_label_cn.txt')
    image_list = []
    label_list = []
    class_id_list = []
    image_name_list = []
    match_name_list = []
    for i in range(len(class_list)):
        assert str(i+1) in class_list[i], "label list is wrong"
        image_path = os.path.join(class_path, class_list[i])
        label_path = os.path.join(label_class_path, class_list[i])
        name_list = os.listdir(image_path)
        label_name_list = os.listdir(label_path)
        for name in name_list:
            image_list.append(os.path.join(image_path, name))
            assert name[-3:] == 'jpg' or name[-3:] == 'JPG', "the format of %s is wrong"%(
                    os.path.join(image_path, name))
            img_temp = Image.open(os.path.join(image_path, name))
            assert img_temp.mode == 'RGB', "the mode of %s is wrong"%(
                    os.path.join(image_path, name))
            assert len(img_temp.split()) == 3, "the number of channels of %s is wrong"%(
                    os.path.join(image_path, name))
            label_name = name[:-4] + '.xml'
            assert label_name in label_name_list, "%s label_name not found"
            label_list.append(os.path.join(label_path, label_name))
            class_id_list.append(i+1)
            image_name_list.append(name)
            match_name_list.append(class_dict[class_name_list[i]])
#        ch_name = class_list[i][len(str(i+1))+1:]
#        idx = class_dict[ch_name]
#        with open('%s_kuo_label_map.pbtxt'%output_path, 'a') as f:
#            f.write("item {\n")
#            f.write("  id: %d\n"%(i+1))
#            f.write("  name: '%s'\n"%str(idx))
#            f.write("}\n")
#            f.write("\n")
#    total = 0
#    for name in class_dict:
#        with open('%s_classes_count_aug.txt'%output_path, 'a') as f:
#            f.write("%s  %d\n"%(name, class_dict[name]))
#        total += class_dict[name]
#    assert total == len(image_list), "total number is wrong"
#    with open('%s_classes_count_aug.txt'%output_path, 'a') as f:
#        f.write("total  %d\n"%total)    
    image_list = np.array(image_list)
    label_list = np.array(label_list)
    class_id_list = np.array(class_id_list)
    image_name_list = np.array(image_name_list)
    match_name_list = np.array(match_name_list)
    np.random.seed(42)
    permutation = np.random.permutation(image_list.shape[0])
    image_list_shuffled = image_list[permutation].tolist()
    label_list_shuffled = label_list[permutation].tolist()
    class_id_list_shuffled = class_id_list[permutation].tolist()
    image_name_list_shuffled = image_name_list[permutation].tolist()
    match_name_list_shuffled = match_name_list[permutation].tolist()
    
    train_writer = tf.python_io.TFRecordWriter('%s_kuo2.record'%output_path)
    for num in range(len(image_list_shuffled)):
        image = image_list_shuffled[num]
        label = label_list_shuffled[num]
        class_id = class_id_list_shuffled[num]
        image_name = image_name_list_shuffled[num]
        class_name = match_name_list_shuffled[num]
        with tf.gfile.GFile(label, 'r') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
        filename = str(class_id) + '_' + image_name
        tf_example = generate_tf_example(image, data, class_id, filename, class_name)
        train_writer.write(tf_example.SerializeToString()) 
    train_writer.close()
    print("total: %d"%len(image_list_shuffled))

def main(_):
    conver_butterfly_to_tfrecords(
            data_dir=FLAGS.data_dir, 
            output_path=FLAGS.output_path)

if __name__ == '__main__':
    tf.app.run()