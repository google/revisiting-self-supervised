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

"""Implements Resnet model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import tensorflow as tf


def get_shape_as_list(x):
  return x.get_shape().as_list()


def fixed_padding(x, kernel_size):
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg

  x = tf.pad(x, [[0, 0],
                 [pad_beg, pad_end], [pad_beg, pad_end],
                 [0, 0]])
  return x


def batch_norm(x, training):
  return tf.layers.batch_normalization(x, fused=True, training=training)


def identity_norm(x, training):
  del training
  return x


def bottleneck_v1(x, filters, training,  # pylint: disable=missing-docstring
                  strides=1,
                  activation_fn=tf.nn.relu,
                  normalization_fn=batch_norm,
                  kernel_regularizer=None):

  # Record input tensor, such that it can be used later in as skip-connection
  x_shortcut = x

  # Project input if necessary
  if (strides > 1) or (filters != x.shape[-1]):
    x_shortcut = tf.layers.conv2d(x_shortcut, filters=filters, kernel_size=1,
                                  strides=strides,
                                  kernel_regularizer=kernel_regularizer,
                                  use_bias=False,
                                  padding='SAME')
    x_shortcut = normalization_fn(x_shortcut, training=training)

  # First convolution
  # Note, that unlike original Resnet paper we never use stride in the first
  # convolution. Instead, we apply stride in the second convolution. The reason
  # is that the first convolution has kernel of size 1x1, which results in
  # information loss when combined with stride bigger than one.
  x = tf.layers.conv2d(x, filters=filters // 4,
                       kernel_size=1,
                       kernel_regularizer=kernel_regularizer,
                       use_bias=False,
                       padding='SAME')
  x = normalization_fn(x, training=training)
  x = activation_fn(x)

  # Second convolution
  x = fixed_padding(x, kernel_size=3)
  x = tf.layers.conv2d(x, filters=filters // 4,
                       strides=strides,
                       kernel_size=3,
                       kernel_regularizer=kernel_regularizer,
                       use_bias=False,
                       padding='VALID')
  x = normalization_fn(x, training=training)
  x = activation_fn(x)

  # Third convolution
  x = tf.layers.conv2d(x, filters=filters,
                       kernel_size=1,
                       kernel_regularizer=kernel_regularizer,
                       use_bias=False,
                       padding='SAME')
  x = normalization_fn(x, training=training)

  # Skip connection
  x = x_shortcut + x
  x = activation_fn(x)

  return x


def bottleneck_v2(x, filters, training,  # pylint: disable=missing-docstring
                  strides=1,
                  activation_fn=tf.nn.relu,
                  normalization_fn=batch_norm,
                  kernel_regularizer=None,
                  no_shortcut=False):

  # Record input tensor, such that it can be used later in as skip-connection
  x_shortcut = x

  x = normalization_fn(x, training=training)
  x = activation_fn(x)

  # Project input if necessary
  if (strides > 1) or (filters != x.shape[-1]):
    x_shortcut = tf.layers.conv2d(x, filters=filters, kernel_size=1,
                                  strides=strides,
                                  kernel_regularizer=kernel_regularizer,
                                  use_bias=False,
                                  padding='VALID')

  # First convolution
  # Note, that unlike original Resnet paper we never use stride in the first
  # convolution. Instead, we apply stride in the second convolution. The reason
  # is that the first convolution has kernel of size 1x1, which results in
  # information loss when combined with stride bigger than one.
  x = tf.layers.conv2d(x, filters=filters // 4,
                       kernel_size=1,
                       kernel_regularizer=kernel_regularizer,
                       use_bias=False,
                       padding='SAME')

  # Second convolution
  x = normalization_fn(x, training=training)
  x = activation_fn(x)
  # Note, that padding depends on the dilation rate.
  x = fixed_padding(x, kernel_size=3)
  x = tf.layers.conv2d(x, filters=filters // 4,
                       strides=strides,
                       kernel_size=3,
                       kernel_regularizer=kernel_regularizer,
                       use_bias=False,
                       padding='VALID')

  # Third convolution
  x = normalization_fn(x, training=training)
  x = activation_fn(x)
  x = tf.layers.conv2d(x, filters=filters,
                       kernel_size=1,
                       kernel_regularizer=kernel_regularizer,
                       use_bias=False,
                       padding='SAME')

  if no_shortcut:
    return x
  else:
    return x + x_shortcut


def resnet(x,  # pylint: disable=missing-docstring
           is_training,
           num_layers,
           strides=(2, 2, 2),
           num_classes=1000,
           filters_factor=4,
           weight_decay=1e-4,
           include_root_block=True,
           root_conv_size=7, root_conv_stride=2,
           root_pool_size=3, root_pool_stride=2,
           activation_fn=tf.nn.relu,
           last_relu=True,
           normalization_fn=batch_norm,
           global_pool=True,
           mode='v2'):

  assert mode in ['v1', 'v2'], 'Unknown Resnet mode: {}'.format(mode)
  unit = bottleneck_v2 if mode == 'v2' else bottleneck_v1

  end_points = {}

  filters = 16 * filters_factor

  kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=weight_decay)

  if include_root_block:
    x = fixed_padding(x, kernel_size=root_conv_size)
    x = tf.layers.conv2d(x, filters=filters,
                         kernel_size=root_conv_size,
                         strides=root_conv_stride,
                         padding='VALID', use_bias=False,
                         kernel_regularizer=kernel_regularizer)

    if mode == 'v1':
      x = normalization_fn(x, training=is_training)
      x = activation_fn(x)

    x = fixed_padding(x, kernel_size=root_pool_size)
    x = tf.layers.max_pooling2d(x, pool_size=root_pool_size,
                                strides=root_pool_stride, padding='VALID')
    end_points['after_root'] = x

  params = {'activation_fn': activation_fn,
            'normalization_fn': normalization_fn,
            'training': is_training,
            'kernel_regularizer': kernel_regularizer,
           }

  strides = list(strides)[::-1]
  num_layers = list(num_layers)[::-1]

  filters *= 4
  for _ in range(num_layers.pop()):
    x = unit(x, filters, strides=1, **params)
  end_points['block1'] = x

  filters *= 2
  x = unit(x, filters, strides=strides.pop(), **params)
  for _ in range(num_layers.pop() - 1):
    x = unit(x, filters, strides=1, **params)
  end_points['block2'] = x

  filters *= 2
  x = unit(x, filters, strides=strides.pop(), **params)
  for _ in range(num_layers.pop() - 1):
    x = unit(x, filters, strides=1, **params)
  end_points['block3'] = x

  filters *= 2
  x = unit(x, filters, strides=strides.pop(), **params)
  for _ in range(num_layers.pop() - 1):
    x = unit(x, filters, strides=1, **params)
  end_points['block4'] = x

  if (mode == 'v1') and (not last_relu):
    raise ValueError('last_relu is always True (implicitly) in the v1 mode.')

  if mode == 'v2':
    x = normalization_fn(x, training=is_training)
    if last_relu:
      x = activation_fn(x)

  if global_pool:
    x = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    end_points['pre_logits'] = tf.squeeze(x, [1, 2])
  else:
    end_points['pre_logits'] = x

  if num_classes:
    logits = tf.layers.conv2d(x, filters=num_classes,
                              kernel_size=1,
                              kernel_regularizer=kernel_regularizer)
    if global_pool:
      logits = tf.squeeze(logits, [1, 2])
    end_points['logits'] = logits
    return logits, end_points
  else:
    return end_points['pre_logits'], end_points

resnet50 = functools.partial(resnet, num_layers=(3, 4, 6, 3))

# Experimental code ########################################
# "Reversible" resnet ######################################


# Invertible residual block as outlined in https://arxiv.org/abs/1707.04585
def bottleneck_rev(x, training,  # pylint: disable=missing-docstring
                   activation_fn=tf.nn.relu,
                   normalization_fn=batch_norm,
                   kernel_regularizer=None):

  unit = bottleneck_v2

  x1, x2 = tf.split(x, 2, 3)

  y1 = x1 + unit(x2, x2.shape[-1], training,
                 strides=1,
                 activation_fn=activation_fn,
                 normalization_fn=normalization_fn,
                 kernel_regularizer=kernel_regularizer,
                 no_shortcut=True)
  y2 = x2

  return tf.concat([y2, y1], axis=3)


def pool_and_double_channels(x, pool_stride):
  if pool_stride > 1:
    x = tf.layers.average_pooling2d(x, pool_size=pool_stride,
                                    strides=pool_stride,
                                    padding='SAME')
  return tf.pad(x, [[0, 0], [0, 0], [0, 0],
                    [x.shape[3] // 2, x.shape[3] // 2]])


def revnet(x,  # pylint: disable=missing-docstring
           is_training,
           num_layers,
           strides=(2, 2, 2),
           num_classes=1000,
           filters_factor=4,
           weight_decay=1e-4,
           include_root_block=True,
           root_conv_size=7, root_conv_stride=2,
           root_pool_size=3, root_pool_stride=2,
           global_pool=True,
           activation_fn=tf.nn.relu,
           normalization_fn=batch_norm,
           last_relu=False,
           mode='v2'):

  del mode  # unused parameter, exists for compatibility with resnet function

  unit = bottleneck_rev

  end_points = {}

  filters = 16 * filters_factor

  kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=weight_decay)

  # First convolution serves as random projection in order to increase number
  # of channels. It is not possible to skip it.
  x = fixed_padding(x, kernel_size=root_conv_size)
  x = tf.layers.conv2d(x, filters=4 * filters,
                       kernel_size=root_conv_size,
                       strides=root_conv_stride,
                       padding='VALID', use_bias=False,
                       kernel_regularizer=None)

  if include_root_block:
    x = fixed_padding(x, kernel_size=root_pool_size)
    x = tf.layers.max_pooling2d(
        x, pool_size=root_pool_size, strides=root_pool_stride, padding='VALID')

  end_points['after_root'] = x

  params = {'activation_fn': activation_fn,
            'normalization_fn': normalization_fn,
            'training': is_training,
            'kernel_regularizer': kernel_regularizer,
           }

  num_layers = list(num_layers)[::-1]
  strides = list(strides)[::-1]

  for _ in range(num_layers.pop()):
    x = unit(x, **params)
  end_points['block1'] = x
  x = pool_and_double_channels(x, strides.pop())

  for _ in range(num_layers.pop()):
    x = unit(x, **params)
  end_points['block2'] = x
  x = pool_and_double_channels(x, strides.pop())

  for _ in range(num_layers.pop()):
    x = unit(x, **params)
  end_points['block3'] = x
  x = pool_and_double_channels(x, strides.pop())

  for _ in range(num_layers.pop()):
    x = unit(x, **params)
  end_points['block4'] = x

  x = normalization_fn(x, training=is_training)

  if last_relu:
    x = activation_fn(x)

  if global_pool:
    x = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    end_points['pre_logits'] = tf.squeeze(x, [1, 2])
  else:
    end_points['pre_logits'] = x

  if num_classes:
    logits = tf.layers.conv2d(x, filters=num_classes,
                              kernel_size=1,
                              kernel_regularizer=kernel_regularizer)
    if global_pool:
      logits = tf.squeeze(logits, [1, 2])
    end_points['logits'] = logits
    return logits, end_points
  else:
    return end_points['pre_logits'], end_points


revnet50 = functools.partial(revnet, num_layers=(3, 4, 6, 3))
