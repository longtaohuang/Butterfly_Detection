#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 18:58:32 2018

@author: YumingWu
"""
import logging
import pdb
import numpy as np

from object_detection.utils import object_detection_evaluation
from object_detection.utils import label_map_util
from object_detection.utils import per_image_evaluation

from object_detection.utils import np_box_list
from object_detection.utils import np_box_list_ops
from object_detection.utils import np_box_mask_list
from object_detection.utils import np_box_mask_list_ops

class ButterflyEvaluator(object_detection_evaluation.ObjectDetectionEvaluator):
    
  def __init__(self,
               categories,
               matching_iou_threshold=0.5,
               score_thresh=0.5):
    """Constructor.

    Args:
      categories: A list of dicts, each of which has the following keys -
        'id': (required) an integer id uniquely identifying this category.
        'name': (required) string representing category name e.g., 'cat', 'dog'.
      matching_iou_threshold: IOU threshold to use for matching groundtruth
        boxes to detection boxes.
      score_thresh: score threshold to use for determining valid detection.
    """
    super(ButterflyEvaluator, self).__init__(
        categories,
        matching_iou_threshold,
        metric_prefix='Butterfly')
    
    self.score_thresh = score_thresh
    self._evaluation = ObjectDetectionEvaluation(
        num_groundtruth_classes=self._num_classes,
        matching_iou_threshold=self._matching_iou_threshold,
        use_weighted_mean_ap=self._use_weighted_mean_ap,
        label_id_offset=self._label_id_offset,
        score_thresh=self.score_thresh)    

  def evaluate(self):
    """Compute evaluation result.

    Returns:
      A dictionary of metrics with the following fields -

      1. LOC + CLS:
        'LOC_error@<matching_iou_threshold>IOU': localization error considering 
        classification and localization at the specified IOU threshold.
        
        'meanPrecision', 'meanRecall', 'meanF1': mean precision, mean recall
        and mean F1 at the specified IOU threshold.
        
        'PerformanceByCategory_Pr', 'PerformanceByCategory_Re', 
        'PerformanceByCategory_F1': category specific results with keys.
        
      2. CLS:
        'CLS_error': classification error for requiring at least one
        correct classification.

      3. LOC
        'meanIOU': mean IOU for all images.
    """    
    (loc_error, cls_error, mean_iou, pr_per_class, re_per_class, 
     F1_per_class) = (self._evaluation.evaluate())
    mean_pr = np.nanmean(pr_per_class)
    mean_re = np.nanmean(re_per_class)
    mean_F1 = np.nanmean(F1_per_class)
    butterfly_metrics = {}
    butterfly_metrics[self._metric_prefix + 'meanPrecision'] = mean_pr
    butterfly_metrics[self._metric_prefix + 'meanRecall'] = mean_re
    butterfly_metrics[self._metric_prefix + 'meanF1'] = mean_F1
    category_index = label_map_util.create_category_index(self._categories)
    for idx in range(F1_per_class.size):
      if idx + self._label_id_offset in category_index:
        
        display_pr = (
              self._metric_prefix + 'PerformanceByCategory_Pr/Precision@{}IOU/{}'
              .format(self._matching_iou_threshold,
                      category_index[idx + self._label_id_offset]['name']))
        butterfly_metrics[display_pr] = pr_per_class[idx]
        
        display_re = (
              self._metric_prefix + 'PerformanceByCategory_Re/Recall@{}IOU/{}'
              .format(self._matching_iou_threshold,
                      category_index[idx + self._label_id_offset]['name']))
        butterfly_metrics[display_re] = re_per_class[idx]
        
        display_F1 = (
              self._metric_prefix + 'PerformanceByCategory_F1/F1@{}IOU/{}'
              .format(self._matching_iou_threshold,
                      category_index[idx + self._label_id_offset]['name']))
        butterfly_metrics[display_F1] = F1_per_class[idx]        
    butterfly_metrics[self._metric_prefix + 
                'LOC_error@{}IOU'.format(self._matching_iou_threshold)] = loc_error
    butterfly_metrics[self._metric_prefix + 'CLS_error'] = cls_error
    butterfly_metrics[self._metric_prefix + 'meanIOU'] = mean_iou     
    
    return butterfly_metrics

  def clear(self):
    """Clears the state to prepare for a fresh evaluation."""
    self._evaluation = ObjectDetectionEvaluation(
        num_groundtruth_classes=self._num_classes,
        matching_iou_threshold=self._matching_iou_threshold,
        use_weighted_mean_ap=self._use_weighted_mean_ap,
        label_id_offset=self._label_id_offset)
    self._image_ids.clear()

class ObjectDetectionEvaluation(object_detection_evaluation.ObjectDetectionEvaluation):
  """Implementation of butterfly metrics"""
    
  def __init__(self,
               num_groundtruth_classes,
               matching_iou_threshold=0.5,
               nms_iou_threshold=1.0,
               nms_max_output_boxes=10000,
               use_weighted_mean_ap=False,
               label_id_offset=0,
               score_thresh=0.5): 
    super(ObjectDetectionEvaluation, self).__init__(
        num_groundtruth_classes,
        matching_iou_threshold,
        use_weighted_mean_ap,
        label_id_offset)
    
    self.per_image_eval = PerImageEvaluation(
        num_groundtruth_classes=num_groundtruth_classes,
        matching_iou_threshold=matching_iou_threshold,
        nms_iou_threshold=nms_iou_threshold,
        nms_max_output_boxes=nms_max_output_boxes)
    self.num_image_loc_error = 0
    self.num_image_cls_error_least = 0
    self.current_loc_error = 1
    self.iou = []
    self.precisions_per_class = np.empty(self.num_class, dtype=float)
    self.recalls_per_class = np.empty(self.num_class, dtype=float)
    self.F1_per_class = np.empty(self.num_class, dtype=float)
    self.precisions_per_class.fill(np.nan)
    self.recalls_per_class.fill(np.nan)
    self.F1_per_class.fill(np.nan)
    self.score_thresh = score_thresh

  def add_single_detected_image_info(self, image_key, detected_boxes,
                                     detected_scores, detected_class_labels,
                                     detected_masks=None):
    """Adds detections for a single image to be used for evaluation.

    Args:
      image_key: A unique string/integer identifier for the image.
      detected_boxes: float32 numpy array of shape [num_boxes, 4]
        containing `num_boxes` detection boxes of the format
        [ymin, xmin, ymax, xmax] in absolute image coordinates.
      detected_scores: float32 numpy array of shape [num_boxes] containing
        detection scores for the boxes.
      detected_class_labels: integer numpy array of shape [num_boxes] containing
        0-indexed detection classes for the boxes.
      detected_masks: np.uint8 numpy array of shape [num_boxes, height, width]
        containing `num_boxes` detection masks with values ranging
        between 0 and 1.

    Raises:
      ValueError: if the number of boxes, scores and class labels differ in
        length.
    """
    detected_boxes = detected_boxes[np.where(np.max(detected_scores))]
    detected_class_labels = detected_class_labels[np.where(np.max(detected_scores))]
    detected_scores = detected_scores[np.where(np.max(detected_scores))]
    if (len(detected_boxes) != len(detected_scores) or
        len(detected_boxes) != len(detected_class_labels)):
      raise ValueError('detected_boxes, detected_scores and '
                       'detected_class_labels should all have same lengths. Got'
                       '[%d, %d, %d]' % len(detected_boxes),
                       len(detected_scores), len(detected_class_labels))

    if image_key in self.detection_keys:
      logging.warn(
          'image %s has already been added to the detection result database',
          image_key)
      return

    self.detection_keys.add(image_key)
    if image_key in self.groundtruth_boxes:
      groundtruth_boxes = self.groundtruth_boxes[image_key]
      groundtruth_class_labels = self.groundtruth_class_labels[image_key]
      # Masks are popped instead of look up. The reason is that we do not want
      # to keep all masks in memory which can cause memory overflow.
      groundtruth_masks = self.groundtruth_masks.pop(
          image_key)
      groundtruth_is_difficult_list = self.groundtruth_is_difficult_list[
          image_key]
      groundtruth_is_group_of_list = self.groundtruth_is_group_of_list[
          image_key]
    else:
      groundtruth_boxes = np.empty(shape=[0, 4], dtype=float)
      groundtruth_class_labels = np.array([], dtype=int)
      if detected_masks is None:
        groundtruth_masks = None
      else:
        groundtruth_masks = np.empty(shape=[0, 1, 1], dtype=float)
      groundtruth_is_difficult_list = np.array([], dtype=bool)
      groundtruth_is_group_of_list = np.array([], dtype=bool)
    scores, tp_fp_labels, is_class_correctly_detected_in_image = (
        self.per_image_eval.compute_object_detection_metrics(
            detected_boxes=detected_boxes,
            detected_scores=detected_scores,
            detected_class_labels=detected_class_labels,
            groundtruth_boxes=groundtruth_boxes,
            groundtruth_class_labels=groundtruth_class_labels,
            groundtruth_is_difficult_list=groundtruth_is_difficult_list,
            groundtruth_is_group_of_list=groundtruth_is_group_of_list,
            detected_masks=detected_masks,
            groundtruth_masks=groundtruth_masks))
    iou_per_image = self.per_image_eval.compute_loc_metrics(
            detected_boxes=detected_boxes, groundtruth_boxes=groundtruth_boxes)
    cls_error_per_image = self.per_image_eval.compute_cls_metrics(
            detected_class_labels=detected_class_labels,
            groundtruth_class_labels=groundtruth_class_labels)
    self.num_image_cls_error_least += cls_error_per_image
    self.iou += iou_per_image
    if ~is_class_correctly_detected_in_image.any():
      self.current_loc_error = 1
      print(image_key)
    else:
      self.current_loc_error = 0
    self.num_image_loc_error += int(~is_class_correctly_detected_in_image.any())
    for i in range(self.num_class):
      if scores[i].shape[0] > 0:
        self.scores_per_class[i].append(scores[i])
        self.tp_fp_labels_per_class[i].append(tp_fp_labels[i])
    (self.num_images_correctly_detected_per_class
    ) += is_class_correctly_detected_in_image
    
  def evaluate(self):
    mean_iou = np.nanmean(self.iou)
    if len(self.detection_keys) > 0:
        loc_error = self.num_image_loc_error / len(self.detection_keys)
        cls_error = self.num_image_cls_error_least / len(self.detection_keys)
    else:
        loc_error = 0
        cls_error = 0
    for class_index in range(self.num_class):
      if self.num_gt_instances_per_class[class_index] == 0:
        continue
      if not self.tp_fp_labels_per_class[class_index]:
        tp_fp_labels = np.array([], dtype=bool)
      else:
        tp_fp_labels = np.concatenate(self.tp_fp_labels_per_class[class_index])
      tp = np.nansum(tp_fp_labels)
      fp = len(tp_fp_labels) - tp
      precision = tp / (tp + fp)
      recall = tp / self.num_gt_instances_per_class[class_index]
      F1 = 2 * precision * recall / (precision + recall)
      self.precisions_per_class[class_index] = precision
      self.recalls_per_class[class_index] = recall
      self.F1_per_class[class_index] = F1
    assert (loc_error <= 1 and cls_error <= 1), "num_image_loc_error is wrong"
    return (loc_error, cls_error, mean_iou, self.precisions_per_class,
            self.recalls_per_class, self.F1_per_class)

class PerImageEvaluation(per_image_evaluation.PerImageEvaluation):
  """Evaluate detection result of a single image."""
    
  def __init__(self,
               num_groundtruth_classes,
               matching_iou_threshold=0.5,
               nms_iou_threshold=0.3,
               nms_max_output_boxes=50):
      
    super(PerImageEvaluation, self).__init__(
               num_groundtruth_classes,
               matching_iou_threshold,
               nms_iou_threshold,
               nms_max_output_boxes)

  def _compute_is_class_correctly_detected_in_image(
      self, detected_boxes, detected_scores, groundtruth_boxes,
      detected_masks=None, groundtruth_masks=None):
    """Compute CorLoc score for a single class.

    Args:
      detected_boxes: A numpy array of shape [N, 4] representing detected box
          coordinates
      detected_scores: A 1-d numpy array of length N representing classification
          score
      groundtruth_boxes: A numpy array of shape [M, 4] representing ground truth
          box coordinates
      detected_masks: (optional) A np.uint8 numpy array of shape
        [N, height, width]. If not None, the scores will be computed based
        on masks.
      groundtruth_masks: (optional) A np.uint8 numpy array of shape
        [M, height, width].

    Returns:
      is_class_correctly_detected_in_image: An integer 1 or 0 denoting whether a
          class is correctly detected in the image or not
    """
    if detected_boxes.size > 0:
      if groundtruth_boxes.size > 0:
          for idx in range(len(detected_scores)):
            mask_mode = False
            if detected_masks is not None and groundtruth_masks is not None:
              mask_mode = True
            if mask_mode:
              detected_boxlist = np_box_mask_list.BoxMaskList(
                  box_data=np.expand_dims(detected_boxes[idx], axis=0),
                  mask_data=np.expand_dims(detected_masks[idx], axis=0))
              gt_boxlist = np_box_mask_list.BoxMaskList(
                  box_data=groundtruth_boxes, mask_data=groundtruth_masks)
              iou = np_box_mask_list_ops.iou(detected_boxlist, gt_boxlist)
            else:
              detected_boxlist = np_box_list.BoxList(
                  np.expand_dims(detected_boxes[idx, :], axis=0))
              gt_boxlist = np_box_list.BoxList(groundtruth_boxes)
              iou = np_box_list_ops.iou(detected_boxlist, gt_boxlist)
            if np.max(iou) >= self.matching_iou_threshold:
              return 1
    return 0

  def compute_loc_metrics(self, detected_boxes, groundtruth_boxes):
    """compute localization results for a single image.
    
    Args:
      detected_boxes: A float numpy array of shape [N, 4], representing N
          regions of detected object regions.
          Each row is of the format [y_min, x_min, y_max, x_max]      
      groundtruth_boxes: A float numpy array of shape [M, 4], representing M
          regions of object instances in ground truth
    
    Returns:
      loc_error_loc: An integer 1 or 0 denoting whether at least one box is 
          correctly localized in the image or not
      iou_max: A list of float representing the max iou of each detected box.
          If no detected boxes in the image, it will return a list with a single
          float 0.0
      tp_box: An integer denoting number of TP boxes
      fp_box: An integer denoting number of FP boxes. If two or more detected
          boxes correspond to one groundtruth box, then one is denoted as TP, 
          others are denoted as FP.
    """
    iou_max = []
    if detected_boxes.size > 0:
      detected_boxlist = np_box_list.BoxList(detected_boxes)
      gt_boxlist = np_box_list.BoxList(groundtruth_boxes)
      iou = np.max(np_box_list_ops.iou(detected_boxlist, gt_boxlist))
      iou_max.append(iou)
    else:
      iou_max.append(0)
    return iou_max
    
  def compute_cls_metrics(self, detected_class_labels, groundtruth_class_labels):
    """compute classification results for a single image.
    
    Args:
      detected_class_labels: A integer numpy array of shape [N, 1], repreneting
          the class labels of the detected N object instances.     
      groundtruth_class_labels: An integer numpy array of shape [M, 1],
          representing M class labels of object instances in ground truth
    
    Returns:
      cls_error_least: An integer 1 or 0 denoting whether at least one class 
          label is correctly detected in the image or not
      cls_error_all: An integer 1 or 0 denoting whether all class labels are
          correctly detected in the image or not
    """    
    cls_error_least = 1
    if detected_class_labels.size > 0:
        detected = set(detected_class_labels)
        groundtruth = set(groundtruth_class_labels)
        if len(detected & groundtruth) > 0:
            cls_error_least = 0
    return cls_error_least