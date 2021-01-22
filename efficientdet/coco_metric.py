# Lint as: python3
# Copyright 2020 Google Research. All Rights Reserved.
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
"""COCO-style evaluation metrics.

Implements the interface of COCO API and metric_fn in tf.TPUEstimator.

COCO API: github.com/cocodataset/cocoapi/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
from absl import flags
from absl import logging

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import tensorflow as tf
import utils

FLAGS = flags.FLAGS


class EvaluationMetric():
  """COCO evaluation metric class.

  This class cannot inherit from tf.keras.metrics.Metric due to numpy.
  """

  def __init__(self, filename=None, testdev_dir=None):
    """Constructs COCO evaluation class.

    The class provides the interface to metrics_fn in TPUEstimator. The
    _update_op() takes detections from each image and push them to
    self.detections. The _evaluate() loads a JSON file in COCO annotation format
    as the groundtruth and runs COCO evaluation.

    Args:
      filename: Ground truth JSON file name. If filename is None, use
        groundtruth data passed from the dataloader for evaluation. filename is
        ignored if testdev_dir is not None.
      testdev_dir: folder name for testdev data. If None, run eval without
        groundtruth, and filename will be ignored.
    """
    self.filename = filename
    self.testdev_dir = testdev_dir
    self.metric_names = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'ARmax1',
                         'ARmax10', 'ARmax100', 'ARs', 'ARm', 'ARl']
    self.reset_states()

  def reset_states(self):
    """Reset COCO API object."""
    self.detections = []
    self.dataset = {
        'images': [],
        'annotations': [],
        'categories': []
    }
    self.image_id = 1
    self.annotation_id = 1
    self.category_ids = []
    self.metric_values = None

  def evaluate(self):
    """Evaluates with detections from all images with COCO API.

    Returns:
      coco_metric: float numpy array with shape [12] representing the
        coco-style evaluation metrics.
    """
    if self.filename:
      coco_gt = COCO(self.filename)
    else:
      coco_gt = COCO()
      coco_gt.dataset = self.dataset
      coco_gt.createIndex()

    if self.testdev_dir:
      # Run on test-dev dataset.
      box_result_list = []
      for det in self.detections:
        box_result_list.append({
            'image_id': int(det[0]),
            'category_id': int(det[6]),
            'bbox': np.around(
                det[1:5].astype(np.float64), decimals=2).tolist(),
            'score': float(np.around(det[5], decimals=3)),
        })
      json.encoder.FLOAT_REPR = lambda o: format(o, '.3f')
      # Must be in the formst of 'detections_test-dev2017_xxx_results'.
      fname = 'detections_test-dev2017_test_results'
      output_path = os.path.join(self.testdev_dir, fname + '.json')
      logging.info('Writing output json file to: %s', output_path)
      with tf.io.gfile.GFile(output_path, 'w') as fid:
        json.dump(box_result_list, fid)
      return np.array([0.], dtype=np.float32)
    else:
      # Run on validation dataset.
      detections = np.array(self.detections)
      image_ids = list(set(detections[:, 0]))
      coco_dt = coco_gt.loadRes(detections)
      coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
      coco_eval.params.imgIds = image_ids
      coco_eval.evaluate()
      coco_eval.accumulate()
      coco_eval.summarize()
      coco_metrics = coco_eval.stats
      return np.array(coco_metrics, dtype=np.float32)

  def result(self):
    """Return the metric values (and compute it if needed)."""
    if not self.metric_values:
      self.metric_values = self.evaluate()
    return self.metric_values

  def update_state(self, groundtruth_data, detections):
    """Update detection results and groundtruth data.

    Append detection results to self.detections to aggregate results from
    all validation set. The groundtruth_data is parsed and added into a
    dictionary with the same format as COCO dataset, which can be used for
    evaluation.

    Args:
      groundtruth_data: Groundtruth annotations in a tensor with each row
        representing [y1, x1, y2, x2, is_crowd, area, class].
      detections: Detection results in a tensor with each row representing
        [image_id, x, y, width, height, score, class].
    """
    for i, det in enumerate(detections):
      # Filter out detections with predicted class label = -1.
      indices = np.where(det[:, -1] > -1)[0]
      det = det[indices]
      if det.shape[0] == 0:
        continue
      # Append groundtruth annotations to create COCO dataset object.
      # Add images.
      image_id = det[0, 0]
      if image_id == -1:
        image_id = self.image_id
      det[:, 0] = image_id
      self.detections.extend(det)

      if not self.filename and not self.testdev_dir:
        # process groudtruth data only if filename is empty and no test_dev.
        self.dataset['images'].append({
            'id': int(image_id),
        })

        # Add annotations.
        indices = np.where(groundtruth_data[i, :, -1] > -1)[0]
        for data in groundtruth_data[i, indices]:
          box = data[0:4]
          is_crowd = data[4]
          area = (box[3] - box[1]) * (box[2] - box[0])
          category_id = data[6]
          if category_id < 0:
            break
          self.dataset['annotations'].append({
              'id': int(self.annotation_id),
              'image_id': int(image_id),
              'category_id': int(category_id),
              'bbox': [box[1], box[0], box[3] - box[1], box[2] - box[0]],
              'area': area,
              'iscrowd': int(is_crowd)
          })
          self.annotation_id += 1
          self.category_ids.append(category_id)

      self.image_id += 1

    if not self.filename:
      self.category_ids = list(set(self.category_ids))
      self.dataset['categories'] = [
          {'id': int(category_id)} for category_id in self.category_ids
      ]


  def get_leaf_to_parent_path(self, leaf2root, leaf):
    parents = []
    while (True):
      if leaf2root.get(leaf):
        parent = leaf2root[leaf]
        parents.append(parent)
        leaf = parent
      else:
        break
    return parents

  def get_max_iou(self, pred_boxes, gt_box):
    """
    calculate the iou multiple pred_boxes and 1 gt_box (the same one)
    pred_boxes: multiple predict  boxes coordinate
    gt_box: ground truth bounding  box coordinate
    return: the max overlaps about pred_boxes and gt_box
    """
    # 1. calculate the inters coordinate
    if pred_boxes.shape[0] > 0:
      ixmin = np.maximum(pred_boxes[:, 1], gt_box[1])
      ixmax = np.minimum(pred_boxes[:, 3], gt_box[3])
      iymin = np.maximum(pred_boxes[:, 0], gt_box[0])
      iymax = np.minimum(pred_boxes[:, 2], gt_box[2])

      iw = np.maximum(ixmax - ixmin + 1., 0.)
      ih = np.maximum(iymax - iymin + 1., 0.)

      # 2.calculate the area of inters
      inters = iw * ih

      # 3.calculate the area of union
      uni = ((pred_boxes[:, 2] - pred_boxes[:, 0] + 1.) *
             (pred_boxes[:, 3] - pred_boxes[:, 1] + 1.) + (gt_box[2] - gt_box[0] + 1.) *
             (gt_box[3] - gt_box[1] + 1.) - inters)

      # 4.calculate the overlaps and find the max overlap ,the max overlaps index for pred_box
      iou = inters / uni
      iou_max = np.max(iou)
      nmax = np.argmax(iou)
      return iou, iou_max, nmax

  def update_state_hierarchical(self, groundtruth_data, detections, hierarchical_scores,
                                hierarchical_classes, tree_filename):
    """Update detection results and groundtruth data.

    Similar to update_state(), difference in this function is that it walks up the hierarchy and
    resets the predicted class, in order to match with the groundtruth.
    For example, given gt as 'Person', if the nearest bbox prediction is 'Girl',
    we walk up the tree to find if 'Person' exists in parents. If it exists, set the nearest
    predicted bbox class as 'Person', as for the score, sum up corresponding leaf predicted score
    of that bbox. In this case, Person has leaves as ['Woman', 'Man', 'Boy', 'Girl']. We sum up
    scores of the four classes, take min(new_score, 1.0), set the bbox score as that value.
    Args:
      groundtruth_data: Groundtruth annotations in a tensor with each row
        representing [y1, x1, y2, x2, class].
      detections: Detection results in a tensor with each row representing
        [image_id, x, y, width, height, score, class].
      hierarchical_scores: [batch_size, num_boxes, num_classes], note:num_boxes is remaining bboxes after nms
      hierarchical_classes: [batch_size, num_boxes, num_classes].
      tree_filename: string file name.
    """
    tree_leaf2root, sumrule = utils.parse_tree(tree_filename)

    batch_parent_scores = []
    batch_parent_classes = []

    new_bbox_scores = hierarchical_scores[:, :, 0]
    new_bbox_classes = hierarchical_classes[:, :, 0]

    for i, det in enumerate(detections):
      # Filter out detections with the top1 predicted class label = -1.
      indices = np.where(hierarchical_classes[i, :, 0] > -1)[0]
      hierarchical_classes[i][indices].astype(int)
      det = det[indices]
      if det.shape[0] == 0:
        continue
      # Append groundtruth annotations to create COCO dataset object.
      # Add images.
      image_id = det[0, 0]
      if image_id == -1:
        image_id = self.image_id
      det[:, 0] = image_id

      max_levels = 3
      per_bbox_parent_scores = []  # each leaf score corresponds to maximum of max_level parents
      per_bbox_parent_classes = []

      # find parent scores and classes for each bbox
      for _scores, _classes in zip(hierarchical_scores[i], hierarchical_classes[i]):
        per_class_parent_scores = []
        per_class_parent_classes = []
        for _s, _c in zip(_scores, _classes):
          parent_scores = [-1] * max_levels
          parent_classes = [-1] * max_levels
          parents = self.get_leaf_to_parent_path(tree_leaf2root, _c)[:max_levels]
          np_classes = np.asarray(_classes)
          np_scores = np.asarray(_scores)
          for _ii, _p in enumerate(parents):
            np_leaves = np.asarray(sumrule[_p])
            overlap_leaves, indices1, indices2 = np.intersect1d(
                np_leaves, np_classes, return_indices=True)
            _p_score = min(np_scores[indices2].sum(), 1.0)
            parent_scores[_ii] = _p_score
            parent_classes[_ii] = _p
          per_class_parent_scores.append(parent_scores)
          per_class_parent_classes.append(parent_classes)

        per_bbox_parent_scores.append(per_class_parent_scores)
        per_bbox_parent_classes.append(per_class_parent_classes)

      batch_parent_scores.append(per_bbox_parent_scores)
      batch_parent_classes.append(per_bbox_parent_classes)

      if not self.filename and not self.testdev_dir:
        # process groudtruth data only if filename is empty and no test_dev.
        self.dataset['images'].append({
            'id': int(image_id),
        })

        # Add annotations.
        indices = np.where(groundtruth_data[i, :, -1] > -1)[0]
        for data in groundtruth_data[i, indices]:
          box = data[0:4]
          category_id = data[4]
          area = (box[3] - box[1]) * (box[2] - box[0])
          if category_id < 0:
            break
          # find predicted bbox that has the largest IOU with gt, reset its' prediction
          _iou, _iou_max, _idx = self.get_max_iou(detections[i][:, 1:5], box)
          matched_bbox_parent_classes = per_bbox_parent_classes[_idx]  #shape: (k, max_levels)
          matched_bbox_parent_classes = np.asarray(matched_bbox_parent_classes)

          matched_bbox_parent_scores = per_bbox_parent_scores[_idx]
          matched_bbox_parent_scores = np.asarray(matched_bbox_parent_scores)

          parent_match_bbox_idx, parent_match_class_idx = np.where(
              matched_bbox_parent_classes == category_id)
          # get the top1 score for a matching parent class
          if len(parent_match_bbox_idx) > 0 and len(parent_match_class_idx) > 0:
            parent_match_bbox_idx = min(parent_match_bbox_idx)
            parent_match_class_idx = min(parent_match_class_idx)
            new_bbox_classes[i][_idx] = matched_bbox_parent_classes[parent_match_bbox_idx][
                parent_match_class_idx]
            new_bbox_scores[i][_idx] = matched_bbox_parent_scores[parent_match_bbox_idx][
                parent_match_class_idx]

          self.dataset['annotations'].append({
              'id': int(self.annotation_id),
              'image_id': int(image_id),
              'iscrowd': False,
              'category_id': int(category_id),
              'bbox': [box[1], box[0], box[3] - box[1], box[2] - box[0]],
              'area': area,
          })
          self.annotation_id += 1
          self.category_ids.append(category_id)

      det[:, 5] = new_bbox_scores[i]
      det[:, 6] = new_bbox_classes[i]

      self.detections.extend(det)
      self.image_id += 1

    if not self.filename:
      self.category_ids = list(set(self.category_ids))
      self.dataset['categories'] = [{'id': int(category_id)} for category_id in self.category_ids]


  def estimator_metric_fn(self, detections, groundtruth_data):
    """Constructs the metric function for tf.TPUEstimator.

    For each metric, we return the evaluation op and an update op; the update op
    is shared across all metrics and simply appends the set of detections to the
    `self.detections` list. The metric op is invoked after all examples have
    been seen and computes the aggregate COCO metrics. Please find details API
    in: https://www.tensorflow.org/api_docs/python/tf/contrib/learn/MetricSpec
    Args:
      detections: Detection results in a tensor with each row representing
        [image_id, x, y, width, height, score, class]
      groundtruth_data: Groundtruth annotations in a tensor with each row
        representing [y1, x1, y2, x2, is_crowd, area, class].
    Returns:
      metrics_dict: A dictionary mapping from evaluation name to a tuple of
        operations (`metric_op`, `update_op`). `update_op` appends the
        detections for the metric to the `self.detections` list.
    """
    with tf.name_scope('coco_metric'):
      if self.testdev_dir:
        if type(hierarchical_scores) == list and len(hierarchical_scores) == 0:
          update_op = tf.numpy_function(self.update_state, [groundtruth_data, detections], [])
        else:
          update_op = tf.numpy_function(self.update_state_hierarchical, [
              groundtruth_data, detections, hierarchical_scores, hierarchical_classes,
              tree_filename
          ], [])

        metrics = tf.numpy_function(self.result, [], tf.float32)
        metrics_dict = {'AP': (metrics, update_op)}
        return metrics_dict
      else:
        if type(hierarchical_scores) == list and len(hierarchical_scores) == 0:
          update_op = tf.numpy_function(self.update_state, [groundtruth_data, detections], [])
        else:
          update_op = tf.numpy_function(self.update_state_hierarchical, [
              groundtruth_data, detections, hierarchical_scores, hierarchical_classes,
              tree_filename
          ], [])
        metrics = tf.numpy_function(self.result, [], tf.float32)
        metrics_dict = {}
        for i, name in enumerate(self.metric_names):
          metrics_dict[name] = (metrics[i], update_op)
        return metrics_dict
