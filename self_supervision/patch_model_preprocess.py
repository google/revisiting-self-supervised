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

# pylint: disable=missing-docstring
"""Preprocessing methods for self supervised representation learning.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import tensorflow as tf

import utils as utils


def crop(image, is_training, crop_size):
  h, w, c = crop_size[0], crop_size[1], image.shape[-1]

  if is_training:
    return tf.random_crop(image, [h, w, c])
  else:
    # Central crop for now. (See Table 5 in Appendix of
    # https://arxiv.org/pdf/1703.07737.pdf for why)
    dy = (tf.shape(image)[0] - h)//2
    dx = (tf.shape(image)[1] - w)//2
    return tf.image.crop_to_bounding_box(image, dy, dx, h, w)


def image_to_patches(image, is_training, split_per_side, patch_jitter=0):
  """Crops split_per_side x split_per_side patches from input image.

  Args:
    image: input image tensor with shape [h, w, c].
    is_training: is training flag.
    split_per_side: split of patches per image side.
    patch_jitter: jitter of each patch from each grid.

  Returns:
    Patches tensor with shape [patch_count, hc, wc, c].
  """
  h, w, _ = image.get_shape().as_list()

  h_grid = h // split_per_side
  w_grid = w // split_per_side
  h_patch = h_grid - patch_jitter
  w_patch = w_grid - patch_jitter

  tf.logging.info(
      "Crop patches - image size: (%d, %d), split_per_side: %d, "
      "grid_size: (%d, %d), patch_size: (%d, %d), split_jitter: %d",
      h, w, split_per_side, h_grid, w_grid, h_patch, w_patch, patch_jitter)

  patches = []
  for i in range(split_per_side):
    for j in range(split_per_side):

      p = tf.image.crop_to_bounding_box(image, i * h_grid, j * w_grid, h_grid,
                                        w_grid)
      # Trick: crop a small tile from pixel cell, to avoid edge continuity.
      if h_patch < h_grid or w_patch < w_grid:
        p = crop(p, is_training, [h_patch, w_patch])

      patches.append(p)

  return tf.stack(patches)


def get_crop_patches_fn(is_training, split_per_side, patch_jitter=0):
  """Gets a function which crops split_per_side x split_per_side patches.

  Args:
    is_training: is training flag.
    split_per_side: split of patches per image side.
    patch_jitter: jitter of each patch from each grid. E.g. 255x255 input
      image with split_per_side=3 will be split into 3 85x85 grids, and
      patches are cropped from each grid with size (grid_size-patch_jitter,
      grid_size-patch_jitter).

  Returns:
    A function returns name to tensor dict. This function crops split_per_side x
    split_per_side patches from "image" tensor in input data dict.
  """

  def _crop_patches_pp(data):
    image = data["image"]

    image_to_patches_fn = functools.partial(
        image_to_patches,
        is_training=is_training,
        split_per_side=split_per_side,
        patch_jitter=patch_jitter)
    image = utils.tf_apply_to_image_or_images(image_to_patches_fn, image)

    data["image"] = image
    return data
  return _crop_patches_pp

