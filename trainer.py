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

"""Base trainer class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import datasets
import utils

FLAGS = tf.flags.FLAGS


def get_lr(global_step, base_lr, steps_per_epoch,  # pylint: disable=missing-docstring
           decay_epochs, lr_decay_factor, warmup_epochs):

  warmup_lr = 0.0
  if warmup_epochs > 0:
    warmup_lr = (tf.cast(global_step, tf.float32) *
                 (base_lr / (warmup_epochs * steps_per_epoch)))

  normal_lr = tf.train.piecewise_constant(
      global_step,
      [e * steps_per_epoch for e in decay_epochs],
      [base_lr * (lr_decay_factor ** i) for i in range(len(decay_epochs) + 1)]
  )

  lr = tf.cond(tf.less(global_step, warmup_epochs * steps_per_epoch),
               lambda: warmup_lr,
               lambda: normal_lr)

  return lr


# TODO(akolesnikov): add more logging
class Trainer(object):
  """Base trainer class."""

  def __init__(self,
               update_batchnorm_params=True):
    self.update_batchnorm_params = update_batchnorm_params

    split = FLAGS.get_flag_value('train_split', 'train')
    num_samples = datasets.get_count(split)
    steps_per_epoch = num_samples // FLAGS.batch_size

    global_step = tf.train.get_or_create_global_step()
    self.global_step_inc = tf.assign_add(global_step, 1)

    # lr_scale_batch_size defines a canonical batch size that is coupled with
    # the initial learning rate. If actual batch size is not the same as
    # canonical than learning rate is linearly scaled. This is very convinient
    # as this allows to vary batch size without recomputing learning rate.
    lr_factor = 1.0
    if FLAGS.get_flag_value('lr_scale_batch_size', 0):
      lr_factor = FLAGS.batch_size / float(FLAGS.lr_scale_batch_size)

    deps = FLAGS.get_flag_value('decay_epochs', None)
    decay_epochs = utils.str2intlist(deps) if deps else [FLAGS.epochs]

    self.lr = get_lr(
        global_step,
        base_lr=FLAGS.lr * lr_factor,
        steps_per_epoch=steps_per_epoch,
        decay_epochs=decay_epochs,
        lr_decay_factor=FLAGS.get_flag_value('lr_decay_factor', 0.1),
        warmup_epochs=FLAGS.get_flag_value('warmup_epochs', 0))

    # TODO(marvinritter): Re-enable summaries with support for TPU training.
    # tf.summary.scalar('learning_rate', self.lr)

  def get_train_op(self, loss,  # pylint: disable=missing-docstring
                   var_list=None,
                   add_reg_loss=True,
                   use_tpu=False):

    if add_reg_loss:
      l2_loss = tf.reduce_sum(tf.losses.get_regularization_losses())
      loss += l2_loss

    optimizer = FLAGS.get_flag_value('optimizer', 'sgd')
    if optimizer == 'sgd':
      optimizer = tf.train.MomentumOptimizer(learning_rate=self.lr,
                                             momentum=0.9)
    elif optimizer == 'adam':
      optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
    else:
      raise ValueError('Unknown optimizer: %s' % optimizer)

    if use_tpu:
      # Wrap optimizer in CrossShardOptimizer which takes care of
      # synchronizing the weight updates between TPU cores.
      optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

    opt_step = optimizer.minimize(loss, var_list=var_list,
                                  colocate_gradients_with_ops=True)

    if self.update_batchnorm_params:
      opt_step = tf.group([opt_step] +
                          tf.get_collection(tf.GraphKeys.UPDATE_OPS))

    opt_step = tf.group([opt_step, self.global_step_inc])

    return opt_step


def make_estimator(mode, loss=None, eval_metrics=None, predictions=None):
  """Returns an EstimatorSpec (maybe TPU) for all modes."""

  # Always use TPUEstimator, even when not using TPU, then it's (almost) no-op.
  spec_type = tf.contrib.tpu.TPUEstimatorSpec

  if mode == tf.estimator.ModeKeys.PREDICT:
    assert predictions is not None, 'Need to pass `predict` arg.'
    return spec_type(mode=mode, predictions=predictions)

  if mode == tf.estimator.ModeKeys.EVAL:
    return spec_type(mode=mode, loss=loss, eval_metrics=eval_metrics)

  if mode == tf.estimator.ModeKeys.TRAIN:
    assert loss is not None, 'Need to pass `loss` arg.'
    trainer = Trainer(update_batchnorm_params=True)
    train_op = trainer.get_train_op(loss, use_tpu=FLAGS.use_tpu)
    return spec_type(mode=mode, loss=loss, train_op=train_op)

  raise ValueError('Unsupported mode %s' % mode)
