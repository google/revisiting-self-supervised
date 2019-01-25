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

"""Produces ratations for input images.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import tensorflow as tf
import tensorflow_hub as hub

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
    hub.add_signature(
        name='representation',
        inputs={'image': images},
        outputs=end_points)
  return output


def model_fn(data, mode):
  """Produces a loss for the rotation task.

  Args:
    data: Dict of inputs containing, among others, "image" and "label."
    mode: model's mode: training, eval or prediction

  Returns:
    EstimatorSpec
  """
  num_angles = 4
  images = data['image']

  if mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
    images = tf.reshape(images, [-1] + images.get_shape().as_list()[-3:])
    with tf.variable_scope('module'):
      image_fn = lambda: images
      logits = apply_model(
          image_fn=image_fn,
          is_training=(mode == tf.estimator.ModeKeys.TRAIN),
          num_outputs=num_angles,
          make_signature=False)
  else:
    input_shape = utils.str2intlist(
        FLAGS.get_flag_value('serving_input_shape', 'None,None,None,3'))
    image_fn = lambda: tf.placeholder(shape=input_shape,  # pylint: disable=g-long-lambda
                                      dtype=tf.float32)
    apply_model_function = functools.partial(
        apply_model,
        image_fn=image_fn,
        num_outputs=num_angles,
        make_signature=True)
    tf_hub_module_spec = hub.create_module_spec(apply_model_function,
                                                [(utils.TAGS_IS_TRAINING, {
                                                    'is_training': True
                                                }),
                                                 (set(), {
                                                     'is_training': False
                                                 })])
    tf_hub_module = hub.Module(tf_hub_module_spec, trainable=False, tags=set())
    hub.register_module_for_export(tf_hub_module, export_name='module')
    logits = tf_hub_module(images)

    return trainer.make_estimator(mode, predictions=logits)

  labels = tf.reshape(data['label'], [-1])

  # build loss and accuracy
  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits)
  loss = tf.reduce_mean(loss)

  eval_metrics = (
      lambda labels, logits: {  # pylint: disable=g-long-lambda
          'accuracy': tf.metrics.accuracy(
              labels=labels,
              predictions=tf.argmax(logits, axis=-1))},
      [labels, logits])
  return trainer.make_estimator(mode, loss, eval_metrics, logits)
