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

import collections
import functools
import os

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

import datasets
from self_supervision.patch_utils import get_patch_representation
from trainer import make_estimator
import utils

FLAGS = tf.flags.FLAGS


def apply_fractional_pooling(taps, target_features=9000, mode='adaptive_max'):
  """Applies fractional pooling to each of `taps`.

  Args:
    taps: A dict of names:tensors to which to attach the head.
    target_features: If the input tensor has more than this number of features,
                     perform fractional pooling to reduce it to this amount.
    mode: one of {'adaptive_max', 'adaptive_avg', 'max', 'avg'}

  Returns:
    tensors: An ordered dict with target_features dimension tensors.

  Raises:
    ValueError: mode is unexpected.
  """
  out_tensors = collections.OrderedDict()
  for k, t in sorted(taps.items()):
    if len(t.get_shape().as_list()) == 2:
      t = t[:, None, None, :]
    _, h, w, num_channels = t.get_shape().as_list()
    if h * w * num_channels > target_features:
      t = utils.adaptive_pool(t, target_features, mode)
      _, h, w, num_channels = t.get_shape().as_list()
    out_tensors[k] = t

  return out_tensors


def add_linear_heads(rep_tensors, n_out):
  """Adds a linear head to each of rep_tensors.

  Args:
    rep_tensors: A dict of names:tensors to which to attach the head.
    n_out: The number of features the head should map to.

  Returns:
    tensors: An ordered dict like `taps` but with the head's output as value.
  """
  for k in list(rep_tensors.keys()):
    t = rep_tensors[k]
    t = tf.reshape(
        t, tf.stack([-1, 1, 1, tf.reduce_prod(t.shape[1:])]))
    t = tf.layers.conv2d(
        t,
        filters=n_out,
        kernel_size=1,
        padding='valid',
        activation=None,
        kernel_regularizer=tf.contrib.layers.l2_regularizer(
            scale=FLAGS.get_flag_value('weight_decay', 0.)))
    rep_tensors[k] = tf.squeeze(t, [1, 2])

  return rep_tensors


def add_mlp_heads(rep_tensors, n_out, is_training):
  """Adds a mlp head to each of rep_tensors.

  Args:
    rep_tensors: A dict of names:tensors to which to attach the head.
    n_out: The number of features the head should map to.
    is_training: whether in training mode.

  Returns:
    tensors: An ordered dict like `taps` but with the head's output as value.
  """
  kernel_regularizer = tf.contrib.layers.l2_regularizer(
      scale=FLAGS.get_flag_value('weight_decay', 0.))
  channels_hidden = FLAGS.get_flag_value('channels_hidden', 1000)
  for k, t in rep_tensors.iteritems():
    t = tf.reshape(t, [-1, 1, 1, np.prod(t.shape[1:])])
    t = tf.layers.conv2d(
        t, channels_hidden, kernel_size=1, padding='VALID',
        activation=tf.nn.relu, kernel_regularizer=kernel_regularizer)
    t = tf.layers.dropout(t, rate=0.5, training=is_training)
    t = tf.layers.conv2d(
        t, n_out, kernel_size=1, padding='VALID',
        kernel_regularizer=kernel_regularizer)
    rep_tensors[k] = tf.squeeze(t, [1, 2])

  return rep_tensors


def model_fn(data, mode):
  """Produces a loss for the fully-supervised task.

  Args:
    data: Dict of inputs containing, among others, "image" and "label."
    mode: model's mode: training, eval or prediction

  Returns:
    EstimatorSpec

  Raises:
    ValueError: Unexpected FLAGS.eval_model
  """
  images = data['image']
  tf.logging.info('model_fn(): features=%s, mode=%s)', images, mode)

  input_rank = len(images.get_shape().as_list())
  image_input_rank = 4  # NHWC
  patch_input_rank = 5  # NPHWC
  assert input_rank in [image_input_rank, patch_input_rank], (
      'Unsupported input rank: %d' % input_rank)

  module = hub.Module(os.path.expanduser(str(FLAGS.hub_module)))

  if mode == tf.estimator.ModeKeys.PREDICT:
    return make_estimator(
        mode,
        predictions=module(
            images,
            signature=FLAGS.get_flag_value('signature', 'representation'),
            as_dict=True))

  is_training = (mode == tf.estimator.ModeKeys.TRAIN)
  pooling_fn = functools.partial(
      apply_fractional_pooling,
      mode=FLAGS.get_flag_value('pool_mode', 'adaptive_max'))
  target_features = 9000
  if input_rank == patch_input_rank:
    out_tensors = get_patch_representation(
        images,
        module,
        patch_preprocess=None,
        is_training=is_training,
        target_features=target_features,
        combine_patches=FLAGS.get_flag_value('combine_patches', 'avg_pool'),
        signature='representation',
        pooling_fn=pooling_fn)
  else:
    out_tensors = module(
        images,
        signature=FLAGS.get_flag_value('signature', 'representation'),
        as_dict=True)
    out_tensors = pooling_fn(out_tensors, target_features=target_features)

  eval_model = FLAGS.get_flag_value('eval_model', 'linear')
  if eval_model == 'linear':
    out_logits = add_linear_heads(out_tensors, datasets.get_num_classes())
  elif eval_model == 'mlp':
    out_logits = add_mlp_heads(out_tensors, datasets.get_num_classes(),
                               is_training=is_training)
  else:
    raise ValueError('Unsupported eval %s model.' % eval_model)

  # build loss and accuracy
  labels = data['label']
  losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                           logits=logits)
            for logits in out_logits.values()]
  loss = tf.add_n([tf.reduce_mean(loss) for loss in losses])

  metrics_fn = utils.get_classification_metrics(
      tensor_names=out_logits.keys())
  # A tuple of metric_fn and a list of tensors to be evaluated by TPUEstimator.
  eval_metrics_tuple = (metrics_fn, [labels] + list(out_logits.values()))

  return make_estimator(mode, loss, eval_metrics_tuple)
