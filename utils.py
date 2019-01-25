#!/usr/bin/python
#
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Util functions for representation learning.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import re

import numpy as np
import tensorflow as tf


INPUT_DATA_STR = "input_data"
IS_TRAINING_STR = "is_training"
REPR_PREFIX_STR = "representation_"
TAGS_IS_TRAINING = ["is_training"]


def adaptive_pool(inp, num_target_dimensions=9000, mode="adaptive_max"):
  """Adaptive pooling layer.

     This layer performs adaptive pooling, such that the total
     dimensionality of output is not bigger than num_target_dimension

  Args:
     inp: input tensor
     num_target_dimensions: maximum number of output dimensions
     mode: one of {"adaptive_max", "adaptive_avg", "max", "avg"}

  Returns:
    Result of the pooling operation

  Raises:
    ValueError: mode is unexpected.
  """

  size, _, k = inp.get_shape().as_list()[1:]
  if mode in ["adaptive_max", "adaptive_avg"]:
    if mode == "adaptive_max":
      pool_fn = tf.nn.fractional_max_pool
    else:
      pool_fn = tf.nn.fractional_avg_pool

    # Find the optimal target output tensor size
    target_size = (num_target_dimensions / float(k)) ** 0.5
    if (abs(num_target_dimensions - k * np.floor(target_size) ** 2) <
        abs(num_target_dimensions - k * np.ceil(target_size) ** 2)):
      target_size = max(np.floor(target_size), 1.0)
    else:
      target_size = max(np.ceil(target_size), 1.0)

    # Get optimal stride. Subtract epsilon to ensure correct rounding in
    # pool_fn.
    stride = size / target_size - 1.0e-5

    # Make sure that the stride is valid
    stride = max(stride, 1)
    stride = min(stride, size)

    result = pool_fn(inp, [1, stride, stride, 1])[0]
  elif mode in ["max", "avg"]:
    if mode == "max":
      pool_fn = tf.contrib.layers.max_pool2d
    else:
      pool_fn = tf.contrib.layers.avg_pool2d
    total_size = float(np.prod(inp.get_shape()[1:].as_list()))
    stride = int(np.ceil(np.sqrt(total_size / num_target_dimensions)))
    stride = min(max(1, stride), size)

    result = pool_fn(inp, kernel_size=stride, stride=stride)
  else:
    raise ValueError("Not supported %s pool." % mode)

  return result


def append_multiple_rows_to_csv(dictionaries, csv_path):
  """Writes multiples rows to csv file from a list of dictionaries.

  Args:
    dictionaries: a list of dictionaries, mapping from csv header to value.
    csv_path: path to the result csv file.
  """

  keys = set([])
  for d in dictionaries:
    keys.update(d.keys())

  if not tf.gfile.Exists(csv_path):
    with tf.gfile.Open(csv_path, "w") as f:
      writer = csv.DictWriter(f, sorted(keys))
      writer.writeheader()
      f.flush()

  with tf.gfile.Open(csv_path, "a") as f:
    writer = csv.DictWriter(f, sorted(keys))
    writer.writerows(dictionaries)
    f.flush()


def concat_dicts(dict_list):
  """Given a list of dicts merges them into a single dict.

  This function takes a list of dictionaries as an input and then merges all
  these dictionaries into a single dictionary by concatenating the values
  (along the first axis) that correspond to the same key.

  Args:
    dict_list: list of dictionaries

  Returns:
    d: merged dictionary
  """
  d = collections.defaultdict(list)
  for e in dict_list:
    for k, v in e.items():
      d[k].append(v)
  for k in d:
    d[k] = tf.concat(d[k], axis=0)
  return d


def str2intlist(s, repeats_if_single=None):
  """Parse a config's "1,2,3"-style string into a list of ints.

  Args:
    s: The string to be parsed, or possibly already an int.
    repeats_if_single: If s is already an int or is a single element list,
                       repeat it this many times to create the list.

  Returns:
    A list of integers based on `s`.
  """
  if isinstance(s, int):
    result = [s]
  else:
    result = [int(i.strip()) if i != "None" else None
              for i in s.split(",")]
  if repeats_if_single is not None and len(result) == 1:
    result *= repeats_if_single
  return result


def tf_apply_to_image_or_images(fn, image_or_images):
  """Applies a function to a single image or each image in a batch of them.

  Args:
    fn: the function to apply, receives an image, returns an image.
    image_or_images: Either a single image, or a batch of images.

  Returns:
    The result of applying the function to the image or batch of images.

  Raises:
    ValueError: if the input is not of rank 3 or 4.
  """
  static_rank = len(image_or_images.get_shape().as_list())
  if static_rank == 3:  # A single image: HWC
    return fn(image_or_images)
  elif static_rank == 4:  # A batch of images: BHWC
    return tf.map_fn(fn, image_or_images)
  elif static_rank > 4:  # A batch of images: ...HWC
    input_shape = tf.shape(image_or_images)
    h, w, c = image_or_images.get_shape().as_list()[-3:]
    image_or_images = tf.reshape(image_or_images, [-1, h, w, c])
    image_or_images = tf.map_fn(fn, image_or_images)
    return tf.reshape(image_or_images, input_shape)
  else:
    raise ValueError("Unsupported image rank: %d" % static_rank)


def tf_apply_with_probability(p, fn, x):
  """Apply function `fn` to input `x` randomly `p` percent of the time."""
  return tf.cond(
      tf.less(tf.random_uniform([], minval=0, maxval=1, dtype=tf.float32), p),
      lambda: fn(x),
      lambda: x)


def expand_glob(glob_patterns):
  checkpoints = []
  for pattern in glob_patterns:
    checkpoints.extend(tf.gfile.Glob(pattern))
  assert checkpoints, "There are no checkpoints in " + str(glob_patterns)
  return checkpoints


def get_latest_hub_per_task(hub_module_paths):
  """Get latest hub module for each task.

  The hub module path should match format ".*/hub/[0-9]*/module/.*".
  Example usage:
  get_latest_hub_per_task(expand_glob(["/cns/el-d/home/dune/representation/"
                                       "xzhai/1899361/*/export/hub/*/module/"]))
  returns 4 latest hub module from 4 tasks respectivley.

  Args:
    hub_module_paths: a list of hub module paths.

  Returns:
    A list of latest hub modules for each task.

  """
  task_to_path = {}
  for path in hub_module_paths:
    task_name, module_name = path.split("/hub/")
    timestamp = int(re.findall(r"([0-9]*)/module", module_name)[0])
    current_path = task_to_path.get(task_name, "0/module")
    current_timestamp = int(re.findall(r"([0-9]*)/module", current_path)[0])
    if current_timestamp < timestamp:
      task_to_path[task_name] = path
  return sorted(task_to_path.values())


def get_classification_metrics(tensor_names):
  """Gets classification eval metric on input logits and labels.

  Args:
    tensor_names: a list of tensor names for _metrics input tensors.

  Returns:
    A function computes the metric result, from input logits and labels.
  """

  def _top_k_accuracy(k, labels, logits):
    in_top_k = tf.nn.in_top_k(predictions=logits, targets=labels, k=k)
    return tf.metrics.mean(tf.cast(in_top_k, tf.float32))

  def _metrics(labels, *tensors):
    """Computes the metric from logits and labels.

    Args:
      labels: ground truth labels.
      *tensors: tensors to be evaluated.

    Returns:
      Result dict mapping from the metric name to the list of result tensor and
      update_op used by tf.metrics.
    """
    metrics = {}
    assert len(tensor_names) == len(tensors), "Names must match tensors."
    for i in range(len(tensors)):
      tensor = tensors[i]
      name = tensor_names[i]
      for k in (1, 5):
        metrics["top%d_accuracy_%s" % (k, name)] = _top_k_accuracy(
            k, labels, tensor)

    return metrics

  return _metrics
