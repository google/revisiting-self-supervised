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

"""Datasets class to provide images and labels in tf batch.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import os

import tensorflow as tf

from preprocess import get_preprocess_fn

FLAGS = tf.flags.FLAGS


class AbstractDataset(object):
  """Base class for datasets using the simplied input pipeline."""

  def __init__(self,
               filenames,
               reader,
               num_epochs,
               shuffle,
               shuffle_buffer_size=10000,
               random_seed=None,
               num_reader_threads=64,
               drop_remainder=True):
    """Creates a new dataset. Sub-classes have to implement _parse_fn().

    Args:
      filenames: A list of filenames.
      reader: A dataset reader, e.g. `tf.data.TFRecordDataset`.
        `tf.data.TextLineDataset` and `tf.data.FixedLengthRecordDataset`.
      num_epochs: An int, defaults to `None`. Number of epochs to cycle
        through the dataset before stopping. If set to `None` this will read
        samples indefinitely.
      shuffle: A boolean, defaults to `False`. Whether output data are
        shuffled.
      shuffle_buffer_size: `int`, number of examples in the buffer for
        shuffling.
      random_seed: Optional int. Random seed for shuffle operation.
      num_reader_threads: An int, defaults to None. Number of threads reading
        from files. When `shuffle` is False, number of threads is set to 1. When
        using default value, there is one thread per filenames.
      drop_remainder: If true, then the last incomplete batch is dropped.
    """
    self.filenames = filenames
    self.reader = reader
    self.num_reader_threads = num_reader_threads
    self.num_epochs = num_epochs
    self.shuffle = shuffle
    self.shuffle_buffer_size = shuffle_buffer_size
    self.random_seed = random_seed
    self.drop_remainder = drop_remainder

    # Additional options for optimizing TPU input pipelines.
    self.num_parallel_batches = 8

  def _make_source_dataset(self):
    """Reads the files in self.filenames and returns a `tf.data.Dataset`.

    This does not parse the examples!

    Returns:
      `tf.data.Dataset` repeated for self.num_epochs and shuffled if
      self.shuffle is `True`. Files are always read in parallel and sloppy.
    """
    # Shuffle the filenames to ensure better randomization.
    dataset = tf.data.Dataset.list_files(self.filenames, shuffle=self.shuffle,
                                         seed=self.random_seed)

    dataset = dataset.repeat(self.num_epochs)

    def fetch_dataset(filename):
      buffer_size = 8 * 1024 * 1024  # 8 MiB per file
      dataset = tf.data.TFRecordDataset(filename, buffer_size=buffer_size)
      return dataset

    # Read the data from disk in parallel
    dataset = dataset.apply(
        tf.data.experimental.parallel_interleave(
            fetch_dataset,
            cycle_length=self.num_reader_threads,
            sloppy=self.shuffle and self.random_seed is None))

    if self.shuffle:
      dataset = dataset.shuffle(self.shuffle_buffer_size, seed=self.random_seed)
    return dataset

  @abc.abstractmethod
  def _parse_fn(self, value):
    """Parses an image and its label from a serialized TFExample.

    Args:
      value: serialized string containing an TFExample.

    Returns:
      Returns a tuple of (image, label) from the TFExample.
    """
    raise NotImplementedError

  def input_fn(self, params):
    """Input function which provides a single batch for train or eval.

    Args:
      params: `dict` of parameters passed from the `TPUEstimator`.
          `params['batch_size']` is provided and should be used as the effective
           batch size.

    Returns:
      A `tf.data.Dataset` object.
    """
    # Retrieves the batch size for the current shard. The # of shards is
    # computed according to the input pipeline deployment. See
    # tf.contrib.tpu.RunConfig for details.
    batch_size = params['batch_size']

    dataset = self._make_source_dataset()

    # Use the fused map-and-batch operation.
    #
    # For XLA, we must used fixed shapes. Because we repeat the source training
    # dataset indefinitely, we can use `drop_remainder=True` to get fixed-size
    # batches without dropping any training examples.
    #
    # When evaluating, `drop_remainder=True` prevents accidentally evaluating
    # the same image twice by dropping the final batch if it is less than a full
    # batch size. As long as this validation is done with consistent batch size,
    # exactly the same images will be used.
    dataset = dataset.apply(
        tf.data.experimental.map_and_batch(
            self._parse_fn,
            batch_size=batch_size,
            num_parallel_batches=self.num_parallel_batches,
            drop_remainder=self.drop_remainder))

    # Prefetch overlaps in-feed with training
    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
    return dataset


def generate_sharded_filenames(filename):
  base, count = filename.split('@')
  count = int(count)
  return ['{}-{:05d}-of-{:05d}'.format(base, i, count)
          for i in range(count)]


class DatasetImagenet(AbstractDataset):
  """Provides train/val/trainval/test splits for Imagenet data.

  -> trainval split represents official Imagenet train split.
  -> train split is derived by taking the first 984 of 1024 shards of
     the offcial training data.
  -> val split is derived by taking the last 40 shard of the official
     training data.
  -> test split represents official Imagenet test split.
  """

  COUNTS = {'train': 1231121,
            'val': 50046,
            'trainval': 1281167,
            'test': 50000}

  NUM_CLASSES = 1000
  IMAGE_KEY = 'image/encoded'
  LABEL_KEY = 'image/class/label'

  FEATURE_MAP = {
      IMAGE_KEY: tf.FixedLenFeature(shape=[], dtype=tf.string),
      LABEL_KEY: tf.FixedLenFeature(shape=[], dtype=tf.int64)
  }

  LABEL_OFFSET = 1

  def __init__(self,
               split_name,
               preprocess_fn,
               num_epochs,
               shuffle,
               random_seed=None,
               drop_remainder=True):
    """Initialize the dataset object.

    Args:
      split_name: A string split name, to load from the dataset.
      preprocess_fn: Preprocess a single example. The example is already
        parsed into a dictionary.
      num_epochs: An int, defaults to `None`. Number of epochs to cycle
        through the dataset before stopping. If set to `None` this will read
        samples indefinitely.
      shuffle: A boolean, defaults to `False`. Whether output data are
        shuffled.
      random_seed: Optional int. Random seed for shuffle operation.
      drop_remainder: If true, then the last incomplete batch is dropped.
    """
    # This is an instance-variable instead of a class-variable because it
    # depends on FLAGS, which is not parsed yet at class-parse-time.
    files = os.path.join(os.path.expanduser(FLAGS.dataset_dir),
                         '%s@%i')
    filenames = {
        'train': generate_sharded_filenames(files % ('train', 1024))[:-40],
        'val': generate_sharded_filenames(files % ('train', 1024))[-40:],
        'trainval': generate_sharded_filenames(files % ('train', 1024)),
        'test': generate_sharded_filenames(files % ('validation', 128))
    }

    super(DatasetImagenet, self).__init__(
        filenames=filenames[split_name],
        reader=tf.data.TFRecordDataset,
        num_epochs=num_epochs,
        shuffle=shuffle,
        random_seed=random_seed,
        drop_remainder=drop_remainder)
    self.split_name = split_name
    self.preprocess_fn = preprocess_fn

  def _parse_fn(self, value):
    """Parses an image and its label from a serialized TFExample.

    Args:
      value: serialized string containing an TFExample.

    Returns:
      Returns a tuple of (image, label) from the TFExample.
    """
    example = tf.parse_single_example(value, self.FEATURE_MAP)
    image = tf.image.decode_jpeg(example[self.IMAGE_KEY], channels=3)
    # Subtract LABEL_OFFSET so that labels are in [0, 1000).
    label = tf.cast(example[self.LABEL_KEY], tf.int32) - self.LABEL_OFFSET

    return self.preprocess_fn({'image': image, 'label': label})


DATASET_MAP = {
    'imagenet': DatasetImagenet,
}


def get_data(params,
             split_name,
             is_training,
             shuffle=True,
             num_epochs=None,
             drop_remainder=False):
  """Produces image/label tensors for a given dataset.

  Args:
    params: dictionary with `batch_size` entry (thanks TPU...).
    split_name: data split, e.g. train, val, test
    is_training: whether to run pre-processing in train or test mode.
    shuffle: if True, shuffles the data
    num_epochs: number of epochs. If None, proceeds indefenitely
    drop_remainder: Drop remaining examples in the last dataset batch. It is
      useful for third party checkpoints with fixed batch size.

  Returns:
    image, label, example counts
  """
  dataset = DATASET_MAP[FLAGS.dataset]
  preprocess_fn = get_preprocess_fn(FLAGS.preprocessing, is_training)

  return dataset(
      split_name=split_name,
      preprocess_fn=preprocess_fn,
      num_epochs=num_epochs,
      shuffle=shuffle,
      random_seed=FLAGS.get_flag_value('random_seed', None),
      drop_remainder=drop_remainder).input_fn(params)


def get_count(split_name):
  return DATASET_MAP[FLAGS.dataset].COUNTS[split_name]


def get_num_classes():
  return DATASET_MAP[FLAGS.dataset].NUM_CLASSES
