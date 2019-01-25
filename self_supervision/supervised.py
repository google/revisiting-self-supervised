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

"""Implements fully-supervised model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import tensorflow as tf
import tensorflow_hub as hub

import datasets
from models.utils import get_net
import trainer
import utils

FLAGS = tf.flags.FLAGS


def apply_model(image_fn,  # pylint: disable=missing-docstring
                is_training,
                num_outputs,
                make_signature=False):

  # Image tensor needs to be created lazily in order to satisfy tf-hub
  # restriction: all tensors should be created inside tf-hub helper function.
  images = image_fn()

  net = get_net(num_classes=num_outputs)

  output, end_points = net(images, is_training)

  if make_signature:
    hub.add_signature(inputs={'image': images}, outputs=output)
    hub.add_signature(inputs={'image': images}, outputs=end_points,
                      name='representation')
  return output


def model_fn(data, mode):
  """Produces a loss for the fully-supervised task.

  Args:
    data: Dict of inputs containing, among others, "image" and "label."
    mode: model's mode: training, eval or prediction

  Returns:
    EstimatorSpec
  """
  images = data['image']

  # In predict mode (called once at the end of training), we only instantiate
  # the model in order to export a tf.hub module for it.
  # This will then make saving and loading much easier down the line.
  if mode == tf.estimator.ModeKeys.PREDICT:
    input_shape = utils.str2intlist(
        FLAGS.get_flag_value('serving_input_shape', 'None,None,None,3'))
    apply_model_function = functools.partial(
        apply_model,
        image_fn=lambda: tf.placeholder(shape=input_shape, dtype=tf.float32),  # pylint: disable=g-long-lambda
        num_outputs=datasets.get_num_classes(),
        make_signature=True)
    tf_hub_module_spec = hub.create_module_spec(
        apply_model_function,
        [(utils.TAGS_IS_TRAINING, {'is_training': True}),
         (set(), {'is_training': False})])
    tf_hub_module = hub.Module(tf_hub_module_spec, trainable=False, tags=set())
    hub.register_module_for_export(tf_hub_module, export_name='module')
    logits = tf_hub_module(images)

    # There is no training happening anymore, only prediciton and model export.
    return trainer.make_estimator(mode, predictions=logits)

  # From here on, we are either in train or eval modes.
  # Create the model in the 'module' name scope so it matches nicely with
  # tf.hub's requirements for import/export later.
  with tf.variable_scope('module'):
    logits = apply_model(
        image_fn=lambda: images,
        is_training=(mode == tf.estimator.ModeKeys.TRAIN),
        num_outputs=datasets.get_num_classes(),
        make_signature=False)

  labels = data['label']

  # build loss and accuracy
  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits)
  loss = tf.reduce_mean(loss)

  # Gets a metric_fn which evaluates the "top1_accuracy" and "top5_accuracy".
  # The resulting metrics are named "top1_accuracy_{tensor_name}",
  # "top5_accuracy_{tensor_name}".
  metrics_fn = utils.get_classification_metrics(['logits'])
  # A tuple of metric_fn and a list of tensors to be evaluated by TPUEstimator.
  eval_metrics_tuple = (metrics_fn, [labels, logits])

  return trainer.make_estimator(mode, loss, eval_metrics_tuple)
