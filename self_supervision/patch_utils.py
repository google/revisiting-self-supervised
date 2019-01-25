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

"""Utils for patch based image processing."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import struct
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

import preprocess
import utils
from models.utils import get_net
from trainer import make_estimator

FLAGS = tf.flags.FLAGS

PATCH_H_COUNT = 3
PATCH_W_COUNT = 3
PATCH_COUNT = PATCH_H_COUNT * PATCH_W_COUNT


# It's supposed to be in the root folder, which is also pwd when running, if the
# instructions in the README are followed. Hence not a flag.
PERMUTATION_PATH = 'permutations_100_max.bin'


def apply_model(image_fn,
                is_training,
                num_outputs,
                perms,
                make_signature=False):
  """Creates the patch based model output from patches representations.

  Args:
    image_fn: function returns image tensor.
    is_training: is training flag used for batch norm and drop out.
    num_outputs: number of output classes.
    perms: numpy array with shape [m, k], element range [0, PATCH_COUNT). k
      stands for the patch numbers used in a permutation. m stands forthe number
      of permutations. Each permutation is used to concat the patch inputs
      [n*PATCH_COUNT, h, w, c] into tensor with shape [n*m, h, w, c*k].
    make_signature: whether to create signature for hub module.

  Returns:
    out: output tensor with shape [n*m, 1, 1, num_outputs].

  Raises:
    ValueError: An error occurred when the architecture is unknown.
  """
  images = image_fn()

  net = get_net(num_classes=FLAGS.get_flag_value('embed_dim', 1000))
  out, end_points = net(images, is_training,
                        weight_decay=FLAGS.get_flag_value('weight_decay', 1e-4))

  print(end_points)
  if not make_signature:
    out = permutate_and_concat_batch_patches(out, perms)
    out = fully_connected(out, num_outputs, is_training=is_training)

    out = tf.squeeze(out, [1, 2])

  if make_signature:
    hub.add_signature(inputs={'image': images}, outputs=out)
    hub.add_signature(
        name='representation',
        inputs={'image': images},
        outputs=end_points)
  return out


def image_grid(images, ny, nx, padding=0):
  """Create a batch of image grids from a batch of images.

  Args:
    images: A batch of patches (B,N,H,W,C)
    ny: vertical number of images
    nx: horizontal number of images
    padding: number of zeros between images, if any.

  Returns:
    A tensor batch of image grids shaped (B,H*ny,W*nx,C), although that is a
    simplifying lie: if padding is used h/w will be different.
  """
  with tf.name_scope('grid_image'):
    if padding:
      padding = [padding, padding]
      images = tf.pad(images, [[0, 0], [0, 0], padding, padding, [0, 0]])

    return tf.concat([
        tf.concat([images[:, y * nx + x] for x in range(nx)], axis=-2)
        for y in range(ny)], axis=-3)


def creates_estimator_model(images, labels, perms, num_classes, mode):
  """Creates EstimatorSpec for the patch based self supervised models.

  Args:
    images: images
    labels: self supervised labels (class indices)
    perms: patch permutations
    num_classes: number of different permutations
    mode: model's mode: training, eval or prediction

  Returns:
    EstimatorSpec
  """
  print('   +++ Mode: %s, images: %s, labels: %s' % (mode, images, labels))

  images = tf.reshape(images, shape=[-1] + images.get_shape().as_list()[-3:])
  if mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
    with tf.variable_scope('module'):
      image_fn = lambda: images
      logits = apply_model(
          image_fn=image_fn,
          is_training=(mode == tf.estimator.ModeKeys.TRAIN),
          num_outputs=num_classes,
          perms=perms,
          make_signature=False)
  else:
    input_shape = utils.str2intlist(
        FLAGS.get_flag_value('serving_input_shape', 'None,None,None,3'))
    image_fn = lambda: tf.placeholder(  # pylint: disable=g-long-lambda
        shape=input_shape,
        dtype=tf.float32)

    apply_model_function = functools.partial(
        apply_model,
        image_fn=image_fn,
        num_outputs=num_classes,
        perms=perms,
        make_signature=True)

    tf_hub_module_spec = hub.create_module_spec(
        apply_model_function, [(utils.TAGS_IS_TRAINING, {
            'is_training': True
        }), (set(), {
            'is_training': False
        })],
        drop_collections=['summaries'])
    tf_hub_module = hub.Module(tf_hub_module_spec, trainable=False, tags=set())
    hub.register_module_for_export(tf_hub_module, export_name='module')
    logits = tf_hub_module(images)
    return make_estimator(mode, predictions=logits)

  # build loss and accuracy
  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits)
  loss = tf.reduce_mean(loss)

  eval_metrics = (
      lambda labels, logits: {  # pylint: disable=g-long-lambda
          'accuracy': tf.metrics.accuracy(
              labels=labels, predictions=tf.argmax(logits, axis=-1))},
      [labels, logits])
  return make_estimator(mode, loss, eval_metrics, logits)


def fully_connected(inputs,
                    num_classes=100,
                    weight_decay=5e-4,
                    keep_prob=0.5,
                    is_training=True):
  """Two layers fully connected network copied from Alexnet fc7-fc8."""
  net = inputs
  _, _, w, _ = net.get_shape().as_list()
  kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=weight_decay)
  net = tf.layers.conv2d(
      net,
      filters=4096,
      kernel_size=w,
      padding='same',
      kernel_initializer=tf.truncated_normal_initializer(0.0, 0.005),
      bias_initializer=tf.constant_initializer(0.1),
      kernel_regularizer=kernel_regularizer)
  net = tf.layers.batch_normalization(
      net, momentum=0.997, epsilon=1e-5, fused=None, training=is_training)
  net = tf.nn.relu(net)
  if is_training:
    net = tf.nn.dropout(net, keep_prob=keep_prob)
  net = tf.layers.conv2d(
      net,
      filters=num_classes,
      kernel_size=1,
      padding='same',
      kernel_initializer=tf.truncated_normal_initializer(0.0, 0.005),
      bias_initializer=tf.zeros_initializer(),
      kernel_regularizer=kernel_regularizer)

  return net


def generate_patch_locations():
  """Generates relative patch locations."""
  perms = np.array([(i, 4) for i in range(9) if i != 4])
  return perms, len(perms)


def load_permutations():
  """Loads a set of pre-defined permutations."""
  with tf.gfile.Open(PERMUTATION_PATH, 'rb') as f:
    int32_size = 4
    s = f.read(int32_size * 2)
    [num_perms, c] = struct.unpack('<ll', s)
    perms = []
    for _ in range(num_perms * c):
      s = f.read(int32_size)
      x = struct.unpack('<l', s)
      perms.append(x[0])
    perms = np.reshape(perms, [num_perms, c])

  # The bin file used index [1,9] for permutation, updated to [0, 8] for index.
  perms = perms - 1
  assert np.min(perms) == 0 and np.max(perms) == PATCH_COUNT - 1
  return perms, num_perms


def permutate_and_concat_image_patches(patch_embeddings, perms):
  """Permutates patches from an image according to permutations.

  Args:
    patch_embeddings: input tensor with shape [PATCH_COUNT, h, w, c], where
      PATCH_COUNT is the patch number per image.
    perms: numpy array with shape [m, k], with element in range
      [0, PATCH_COUNT). Permutation is used to concat the patches.

  Returns:
    out: output tensor with shape [m, h, w, c*k].
  """

  _, h, w, c = patch_embeddings.get_shape().as_list()
  if isinstance(perms, np.ndarray):
    num_perms, perm_len = perms.shape
  else:
    num_perms, perm_len = perms.get_shape().as_list()

  def permutate_patch(perm):
    permed = tf.gather(patch_embeddings, perm, axis=0)
    concat_tensor = tf.transpose(permed, perm=[1, 2, 3, 0])
    concat_tensor = tf.reshape(
        concat_tensor, shape=[-1, h, w, perm_len * c])
    return concat_tensor

  permed_patches = tf.stack([
      permutate_patch(perms[i]) for i in range(num_perms)
  ])
  return permed_patches


def permutate_and_concat_batch_patches(batch_patch_embeddings, perms):
  """Permutates patches from a mini batch according to permutations.

  Args:
    batch_patch_embeddings: input tensor with shape [n*PATCH_COUNT, h, w, c] or
      [n*PATCH_COUNT, c], where PATCH_COUNT is the patch number per image
      and n is the number of images in this mini batch.
    perms: numpy array with shape [m, k], with element in range
      [0, PATCH_COUNT). Permutation is used to concat the patches.

  Returns:
    out: output tensor with shape [n*m, h, w, c*k].
  """

  print('   +++ permutate patches input: %s' % batch_patch_embeddings)
  if len(batch_patch_embeddings.get_shape().as_list()) == 4:
    _, h, w, c = batch_patch_embeddings.get_shape().as_list()
  elif len(batch_patch_embeddings.get_shape().as_list()) == 2:
    _, c = batch_patch_embeddings.get_shape().as_list()
    h, w = (1, 1)
  else:
    raise ValueError('Unexpected batch_patch_embeddings shape: %s' %
                     batch_patch_embeddings.get_shape().as_list())
  patches = tf.reshape(batch_patch_embeddings, shape=[-1, PATCH_COUNT, h, w, c])

  patches = tf.stack([
      permutate_and_concat_image_patches(patches[i], perms)
      for i in range(patches.get_shape().as_list()[0])
  ])

  patches = tf.reshape(patches, shape=[-1, h, w, perms.shape[1] * c])
  print('   +++ permutate patches output: %s' % batch_patch_embeddings)
  return patches


def get_patch_representation(
    images,
    hub_module,
    patch_preprocess='crop_patches,standardization',
    is_training=False,
    target_features=9000,
    pooling_fn=None,
    combine_patches='concat',
    signature='representation'):
  """Permutates patches from a mini batch according to permutations.

  Args:
    images: input images, can be full image (NHWC) or image patchs (NPHWC).
    hub_module: hub module.
    patch_preprocess: preprocess applied to the image. Note that preprocess may
      require setting parameters in the FLAGS.config file.
    is_training: is training mode.
    target_features: target feature dimension. Note that the features might
      exceed this number if there're too many channels.
    pooling_fn: pooling method applied to the features.
    combine_patches: one of {'concat', 'max_pool', 'avg_pool'}.
    signature: signature for the hub module.

  Returns:
    out: output representation tensors.

  Raises:
    ValueError: unsupported combine_patches.
  """

  if patch_preprocess:
    preprocess_fn = preprocess.get_preprocess_fn(patch_preprocess, is_training)
    images = preprocess_fn({'image': images})['image']

  assert len(images.get_shape().as_list()) == 5, 'Shape must match NPHWC.'
  _, num_of_patches, h, w, c = images.get_shape().as_list()
  images = tf.reshape(images, shape=[-1, h, w, c])

  out_tensors = hub_module(
      images,
      signature=signature,
      as_dict=True)

  if combine_patches == 'concat':
    target_features = target_features // num_of_patches
  if pooling_fn is not None:
    out_tensors = pooling_fn(out_tensors)

  for k, t in out_tensors.iteritems():
    if len(t.get_shape().as_list()) == 2:
      t = t[:, None, None, :]
    assert len(t.get_shape().as_list()) == 4, 'Unsupported rank %d' % len(
        t.get_shape().as_list())
    # Take patch-dimension out of batch-dimension: [NP]HWC -> NPHWC
    t = tf.reshape(t, [-1, num_of_patches] + t.get_shape().as_list()[-3:])
    if combine_patches == 'concat':
      # [N, P, H, W, C] -> [N, H, W, P*C]
      _, p, h, w, c = t.get_shape().as_list()
      out_tensors[k] = tf.reshape(
          tf.transpose(t, perm=[0, 2, 3, 4, 1]), tf.stack([-1, h, w, p * c]))
    elif combine_patches == 'max_pool':
      # Reduce max on P channel of NPHWC.
      out_tensors[k] = tf.reduce_max(t, axis=1)
    elif combine_patches == 'avg_pool':
      # Reduce mean on P channel of NPHWC.
      out_tensors[k] = tf.reduce_mean(t, axis=1)
    else:
      raise ValueError(
          'Unsupported combine patches method %s.' % combine_patches)

  return out_tensors
