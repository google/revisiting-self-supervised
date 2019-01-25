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

"""Exemplar implementation with triplet semihard loss."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import tensorflow as tf
import tensorflow_hub as hub

import utils
from models.utils import get_net
from trainer import make_estimator

FLAGS = tf.flags.FLAGS


def apply_model(image_fn,
                is_training,
                num_outputs,
                make_signature=False):
  """Creates the patch based model output from patches representations.

  Args:
    image_fn: function returns image tensor.
    is_training: is training flag used for batch norm and drop out.
    num_outputs: number of output classes.
    make_signature: whether to create signature for hub module.


  Returns:
    out: output tensor with shape [n*m, 1, 1, num_outputs].

  Raises:
    ValueError: An error occurred when the architecture is unknown.
  """
  # Image tensor needs to be created lazily in order to satisfy tf-hub
  # restriction: all tensors should be created inside tf-hub helper function.
  images = image_fn()

  net = get_net(num_classes=num_outputs)
  out, end_points = net(images, is_training,
                        weight_decay=FLAGS.get_flag_value('weight_decay', 1e-4))

  print(end_points)

  if len(out.get_shape().as_list()) == 4:
    out = tf.squeeze(out, [1, 2])

  if make_signature:
    hub.add_signature(inputs={'image': images}, outputs=out)
    hub.add_signature(
        name='representation',
        inputs={'image': images},
        outputs=end_points)
  return out


def repeat(x, times):
  """Exactly like np.repeat."""
  return tf.reshape(tf.tile(tf.expand_dims(x, -1), [1, times]), [-1])


def model_fn(data, mode):
  """Produces a loss for the exemplar task supervision.

  Args:
    data: Dict of inputs containing, among others, "image" and "label."
    mode: model's mode: training, eval or prediction

  Returns:
    EstimatorSpec
  """
  images = data['image']
  batch_size = tf.shape(images)[0]
  print('   +++ Mode: %s, data: %s' % (mode, data))

  embed_dim = FLAGS.embed_dim
  patch_count = images.get_shape().as_list()[1]

  images = tf.reshape(
      images, shape=[-1] + images.get_shape().as_list()[-3:])

  if mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
    images = tf.reshape(images, [-1] + images.get_shape().as_list()[-3:])
    with tf.variable_scope('module'):
      image_fn = lambda: images
      logits = apply_model(
          image_fn=image_fn,
          is_training=(mode == tf.estimator.ModeKeys.TRAIN),
          num_outputs=embed_dim,
          make_signature=False)
  else:
    input_shape = utils.str2intlist(
        FLAGS.get_flag_value('serving_input_shape', 'None,None,None,3'))
    image_fn = lambda: tf.placeholder(shape=input_shape,  # pylint: disable=g-long-lambda
                                      dtype=tf.float32)
    apply_model_function = functools.partial(
        apply_model,
        image_fn=image_fn,
        num_outputs=embed_dim,
        make_signature=True)

    tf_hub_module_spec = hub.create_module_spec(apply_model_function,
                                                [(utils.TAGS_IS_TRAINING, {
                                                    'is_training': True
                                                }),
                                                 (set(), {
                                                     'is_training': False
                                                 })],
                                                drop_collections=['summaries'])
    tf_hub_module = hub.Module(tf_hub_module_spec, trainable=False, tags=set())
    hub.register_module_for_export(tf_hub_module, export_name='module')
    logits = tf_hub_module(images)
    return make_estimator(mode, predictions=logits)

  labels = repeat(tf.range(batch_size), patch_count)
  norm_logits = tf.nn.l2_normalize(logits, axis=0)
  loss = tf.contrib.losses.metric_learning.triplet_semihard_loss(
      labels, norm_logits, margin=FLAGS.margin)

  return make_estimator(mode, loss, predictions=logits)
