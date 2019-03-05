#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 10:12:06 2018

@author: YumingWu
"""
import os
import sys

import tensorflow as tf
import numpy as np
from PIL import Image
from lxml import etree
import time

sys.path.append("..")
from utils import label_map_util
from utils import dataset_util
from utils import np_box_list
from utils import np_box_list_ops
from utils import visualization_utils as vis_util

PATH_TO_MODEL = '/home/YumingWu/Project/butterfly/models/save_model/inception_v2/frozen_inference_graph.pb'
PATH_TO_LABELS = '/home/YumingWu/Project/butterfly/data/butterfly_save/butterfly_label_map.pbtxt'
export_dir = '/home/YumingWu/Project/butterfly/data/results'
groundtruth_dir = '/home/YumingWu/Datasets/Butterfly_English/Annotations'
PATH_TO_CH = '/home/YumingWu/Project/butterfly/data/butterfly_save/butterfly_label_cn.txt'

PATH_TO_TEST_IMAGES_DIR = '/home/YumingWu/Datasets/Butterfly_English/dataOneBBCleanTest_402/'
TEST_IMAGE_PATHS = os.listdir(PATH_TO_TEST_IMAGES_DIR)

visual = False

NUM_CLASSES = 94
matching_iou_threshold = 0.5

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.5

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

dic_cn = get_label_dict(PATH_TO_CH)
match_dict = {95:5, 96:21, 97:23, 98:25, 99:61, 100:62, 101:64, 102:70, 103:74, 104:77, 105:79, 106:83, 107:87, 108:88}

def load_groundtrouth(path):
    groundtruth_dict = {}
    classes = []
    boxes = []
    with tf.gfile.GFile(path, 'r') as fid:
        xml_str = fid.read()
    xml = etree.fromstring(xml_str)
    data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
    width = int(data['size']['width'])
    height = int(data['size']['height'])
    for obj in data['object']:
        classes.append(dic_cn[obj['name']])
        xmin = (float(obj['bndbox']['xmin']) / width)
        ymin = (float(obj['bndbox']['ymin']) / height)
        xmax = (float(obj['bndbox']['xmax']) / width)
        ymax = (float(obj['bndbox']['ymax']) / height)
        boxes.append(np.array([ymin,xmin,ymax,xmax]))
    groundtruth_dict['classes'] = np.array(classes)
    groundtruth_dict['boxes'] = np.array(boxes)
    return groundtruth_dict

def evaluate(detected_boxes, detected_class_labels,
            groundtruth_boxes, groundtruth_class_labels):
    TP = np.zeros(NUM_CLASSES)
    FP = np.zeros(NUM_CLASSES)
    GT = np.zeros(NUM_CLASSES)
    GT[groundtruth_class_labels - 1] = 1
    assert detected_class_labels.size == 1, "Only one box is considered"
    detected_boxlist = np_box_list.BoxList(detected_boxes)
    gt_boxlist = np_box_list.BoxList(groundtruth_boxes)
    iou_all = np_box_list_ops.iou(detected_boxlist, gt_boxlist)
    iou = np.max(iou_all)
#    if iou >= matching_iou_threshold:
    if detected_class_labels[0] == groundtruth_class_labels[0]:
        error = 0
        TP[detected_class_labels - 1] = 1
    else:
        error = 1
        FP[detected_class_labels - 1] = 1
#    else:
#        error = 1
#        FP[detected_class_labels - 1] = 1
    return error, iou, TP, FP, GT

def load_graph(path):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(path, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in ['detection_boxes', 'detection_scores', 'detection_classes']:
          tensor_name = key + ':0'
          if tensor_name in all_tensor_names:
                tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                  tensor_name)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
    return detection_graph, tensor_dict, image_tensor

def load_image_list(TEST_IMAGE_PATHS):
    num = 0
    idx = 0
    image_path_list = []
    image_name_list = []
    while num < len(TEST_IMAGE_PATHS):
        image_filename = 'IMG_' + str(idx).zfill(6) + '.jpg'
        idx += 1
        if image_filename in TEST_IMAGE_PATHS:
          num += 1
          image_path = os.path.join(PATH_TO_TEST_IMAGES_DIR, image_filename)
          image_path_list.append(image_path)
          image_name_list.append(image_filename)
    return image_path_list, image_name_list

def inference_on_single_image(image_np, sess, tensor_dict, image_tensor):
    output_dict = sess.run(tensor_dict,
                         feed_dict={image_tensor: np.expand_dims(
                                 image_np, 0)})

    detected_class_labels = output_dict[
          'detection_classes'][0].astype(np.uint8)
    detected_boxes = output_dict['detection_boxes'][0]
    detected_scores = output_dict['detection_scores'][0]

    detected_class_labels = detected_class_labels[np.where(np.max(
          detected_scores))]
    detected_boxes = detected_boxes[np.where(np.max(detected_scores))]
    detected_scores = detected_scores[np.where(np.max(detected_scores))]
    return detected_class_labels, detected_boxes, detected_scores

def run_butterfly_inference():
    start_time = time.time()
    
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
            label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    detection_graph, tensor_dict, image_tensor = load_graph(PATH_TO_MODEL)
    
    sess = tf.Session(config=config, graph=detection_graph)
    image_path_list, image_name_list = load_image_list(TEST_IMAGE_PATHS)
    num_loc_error = 0
    iou = 0
    TP = np.zeros(NUM_CLASSES)
    FP = np.zeros(NUM_CLASSES)
    GT = np.zeros(NUM_CLASSES)    
    for i in range(len(image_path_list)):
        image_path = image_path_list[i]
        image_filename = image_name_list[i]
        groundtruth_path = os.path.join(groundtruth_dir, (
              image_filename[:-4] + '.xml'))
        export_path = os.path.join(export_dir, (
                  image_filename[:-4] + '.png'))
        groundtruth_dict = load_groundtrouth(groundtruth_path)
        
        image = Image.open(image_path)
        image_np = np.array(image)

        [detected_class_labels, detected_boxes, detected_scores] = inference_on_single_image(image_np, sess, tensor_dict, image_tensor)
        if detected_class_labels[0] > 94:
            detected_class_labels[0] = match_dict[detected_class_labels[0]]

        loc_error, iou_per, TP_per, FP_per, GT_per = evaluate(
                    detected_boxes=detected_boxes,
                    detected_class_labels=detected_class_labels,
                    groundtruth_boxes=groundtruth_dict['boxes'],
                    groundtruth_class_labels=groundtruth_dict['classes'])
        num_loc_error += loc_error
        iou += iou_per
        TP += TP_per
        FP += FP_per
        GT += GT_per
        if loc_error:
            # Visualization of the results of a detection.
            if visual:
                vis_util.visualize_boxes_and_labels_on_image_array(
                      image_np,
                      detected_boxes,
                      detected_class_labels,
                      detected_scores,
                      category_index,
                      use_normalized_coordinates=True,
                      min_score_thresh=0,
                      line_thickness=8)
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    boxes=groundtruth_dict['boxes'],
                    classes=groundtruth_dict['classes'],
                    scores=None,
                    category_index=category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=None,
                    groundtruth_box_visualization_color='white')
                vis_util.save_image_array_as_png(image_np, export_path)
                print("successfully save %s"%(image_filename[:-4] + '.png'))
        print("Finish inferencing %s"%image_filename)
    precision_all = TP / (TP + FP)
    recall_all = TP / GT
    F1_all = 2 * precision_all * recall_all / (precision_all + recall_all)
    pr = np.nanmean(precision_all)
    re = np.nanmean(recall_all)
    F1 = np.nanmean(F1_all)
    print("LOC error: %0.4f"%(num_loc_error / len(image_path_list)))
    print("IOU: %.4f"%(iou / len(image_path_list)))
    print("Precision: %.4f"%pr)
    print("Recall: %.4f"%re)
    print("F1: %.4f"%F1)
    print("Finish testing, average time on per image: %.3f"%(
            (time.time() - start_time) / len(image_path_list)))
    
if __name__ == '__main__':
    run_butterfly_inference()