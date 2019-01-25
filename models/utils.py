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

"""Helper functions for NN models.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import absl.flags as flags

import models.resnet
import models.vggnet

FLAGS = flags.FLAGS


def get_net(num_classes=None):  # pylint: disable=missing-docstring
  architecture = FLAGS.architecture

  if 'vgg19' in architecture:
    net = functools.partial(
        models.vggnet.vgg19,
        filters_factor=FLAGS.get_flag_value('filters_factor', 8))
  else:
    if 'resnet50' in architecture:
      net = models.resnet.resnet50
    elif 'revnet50' in architecture:
      net = models.resnet.revnet50
    else:
      raise ValueError('Unsupported architecture: %s' % architecture)

    net = functools.partial(
        net,
        filters_factor=FLAGS.get_flag_value('filters_factor', 4),
        last_relu=FLAGS.get_flag_value('last_relu', True),
        mode=FLAGS.get_flag_value('mode', 'v2'))

    if FLAGS.task in ('jigsaw', 'relative_patch_location'):
      net = functools.partial(net, root_conv_stride=1, strides=(2, 2, 1))

  # Few things that are common across all models.
  net = functools.partial(
      net, num_classes=num_classes,
      weight_decay=FLAGS.get_flag_value('weight_decay', 1e-4))

  return net
