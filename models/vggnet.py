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

"""Implements VGG model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import tensorflow as tf


def convbnrelu(x, filters, size, training, **convkw):
  x = tf.layers.conv2d(x, kernel_size=size, filters=filters,
                       use_bias=False, **convkw)
  x = tf.layers.batch_normalization(x, fused=True, training=training)
  x = tf.nn.relu(x)
  return x


def vgg19(x, is_training, num_classes=1000,  # pylint: disable=missing-docstring
          filters_factor=8,
          weight_decay=5e-4):
  # NOTE: default weight_decay here is as in the VGGNet paper, which is
  # different from the ResNet and RevNet models.
  # NOTE: Another difference is that we are using BatchNorm, and because of
  # that, we are not using Dropout in the final FC layers.

  regularizer = tf.contrib.layers.l2_regularizer(scale=weight_decay)
  conv3 = functools.partial(convbnrelu, size=3, training=is_training,
                            kernel_regularizer=regularizer, padding='same')
  fc = functools.partial(convbnrelu, training=is_training,
                         kernel_regularizer=regularizer, padding='valid')

  end_points = {}

  # After long discussion, we settled on filters_factor=8 being the default and
  # thus needing to match the vanilla VGGNet, which starts with 64.
  w = 8 * filters_factor  # w stands for width (a la wide-resnet)
  x = conv3(x, w)
  x = conv3(x, w)
  end_points['block1'] = x
  x = tf.layers.max_pooling2d(x, pool_size=2, strides=2)  # 112x112
  x = conv3(x, 2*w)
  x = conv3(x, 2*w)
  end_points['block2'] = x
  x = tf.layers.max_pooling2d(x, pool_size=2, strides=2)  # 56x56
  x = conv3(x, 4*w)
  x = conv3(x, 4*w)
  x = conv3(x, 4*w)
  x = conv3(x, 4*w)
  end_points['block3'] = x
  x = tf.layers.max_pooling2d(x, pool_size=2, strides=2)  # 28x28
  x = conv3(x, 8*w)
  x = conv3(x, 8*w)
  x = conv3(x, 8*w)
  x = conv3(x, 8*w)
  end_points['block4'] = x
  x = tf.layers.max_pooling2d(x, pool_size=2, strides=2)  # 14x14
  x = conv3(x, 8*w)
  x = conv3(x, 8*w)
  x = conv3(x, 8*w)
  x = conv3(x, 8*w)
  end_points['block5'] = x

  x = tf.layers.max_pooling2d(x, pool_size=2, strides=2)  # 7x7
  x = fc(x, 512*filters_factor, x.get_shape().as_list()[-3:-1])
  end_points['fc6'] = x
  x = fc(x, 512*filters_factor, (1, 1))

  end_points['pre_logits'] = x
  x = tf.layers.conv2d(x, kernel_size=1, filters=num_classes,
                       use_bias=True, kernel_regularizer=regularizer)
  x = tf.squeeze(x, [1, 2])
  end_points['logits'] = x

  return x, end_points
