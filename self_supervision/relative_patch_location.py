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

import tensorflow as tf

from self_supervision import patch_utils


def model_fn(data, mode):
  """Produces a loss for the relative patch location task.

  Args:
    data: Dict of inputs ("image" being the image)
    mode: model's mode: training, eval or prediction

  Returns:
    EstimatorSpec
  """
  images = data['image']

  # Patch locations
  perms, num_classes = patch_utils.generate_patch_locations()
  labels = tf.tile(list(range(num_classes)), tf.shape(images)[:1])

  return patch_utils.creates_estimator_model(
      images, labels, perms, num_classes, mode)
