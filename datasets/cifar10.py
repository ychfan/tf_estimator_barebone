"""CIFAR-10 dataset
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from six.moves import cPickle
from six.moves import urllib
import tarfile
import numpy as np

import tensorflow as tf

import datasets

REMOTE_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
LOCAL_DIR = os.path.join('data/cifar10/')
ARCHIVE_NAME = 'cifar-10-python.tar.gz'
DATA_DIR = 'cifar-10-batches-py/'
TRAIN_BATCHES = ['data_batch_%d' % (i + 1) for i in range(5)]
TEST_BATCHES = ['test_batch']

IMAGE_SIZE = 32
NUM_CLASSES = 10


def update_argparser(parser):
  datasets.update_argparser(parser)
  parser.set_defaults(
      num_classes=NUM_CLASSES,
      train_batch_size=128,
      eval_batch_size=512,
      shuffle_buffer_size=50000,
  )


def _extract(mode, params):
  if not os.path.exists(LOCAL_DIR):
    os.makedirs(LOCAL_DIR)
  if not os.path.exists(LOCAL_DIR + ARCHIVE_NAME):
    print('Downloading...')
    urllib.request.urlretrieve(REMOTE_URL, LOCAL_DIR + ARCHIVE_NAME)
  if not os.path.exists(LOCAL_DIR + DATA_DIR):
    print('Extracting files...')
    tar = tarfile.open(LOCAL_DIR + ARCHIVE_NAME)
    tar.extractall(LOCAL_DIR)
    tar.close()

  batches = {
      tf.estimator.ModeKeys.TRAIN: TRAIN_BATCHES,
      tf.estimator.ModeKeys.EVAL: TEST_BATCHES,
  }[mode]

  all_images = []
  all_labels = []

  for batch in batches:
    with open('%s%s%s' % (LOCAL_DIR, DATA_DIR, batch), 'rb') as fo:
      if sys.version_info[0] < 3:
        entry = cPickle.load(fo)
      else:
        entry = cPickle.load(fo, encoding='latin1')
      images = np.array(entry['data'])
      labels = np.array(entry['labels'])

      num = images.shape[0]
      images = np.reshape(images, [num, 3, IMAGE_SIZE, IMAGE_SIZE])
      images = np.transpose(images, [0, 2, 3, 1])
      print('Loaded %d examples.' % num)

      all_images.append(images)
      all_labels.append(labels)

  all_images = np.concatenate(all_images)
  all_labels = np.concatenate(all_labels)

  def _generator(all_images, all_labels):
    for image, label in zip(all_images, all_labels):
      yield image, label

  dataset = tf.data.Dataset.from_generator(
      generator=lambda: _generator(all_images, all_labels),
      output_types=(tf.uint8, tf.int64),
  )
  dataset = dataset.cache()
  return dataset


def _transform(dataset, mode, params):
  if mode == tf.estimator.ModeKeys.TRAIN:
    dataset = dataset.shuffle(params.shuffle_buffer_size)
    dataset = dataset.repeat()

  def _preprocess(image, label):
    image = tf.to_float(image)
    image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, 3])

    if mode == tf.estimator.ModeKeys.TRAIN:
      padding = 4
      image = tf.pad(image, [[padding, padding], [padding, padding], [0, 0]])
      image = tf.random_crop(image, [IMAGE_SIZE, IMAGE_SIZE, 3])
      image = tf.image.random_flip_left_right(image)

    MEAN_IMAGE = tf.constant([0.4914, 0.4822, 0.4465], dtype=tf.float32)
    STD_IMAGE = tf.constant([0.2023, 0.1994, 0.2010], dtype=tf.float32)
    image = ((image / 255.0) - MEAN_IMAGE) / STD_IMAGE

    return {'image': image}, {'label': label}

  dataset = dataset.map(
      _preprocess,
      num_parallel_calls=params.num_data_threads,
  )
  batch_size = {
      tf.estimator.ModeKeys.TRAIN: params.train_batch_size,
      tf.estimator.ModeKeys.EVAL: params.eval_batch_size,
  }[mode]
  drop_remainder = {
      tf.estimator.ModeKeys.TRAIN: True,
      tf.estimator.ModeKeys.EVAL: False,
  }[mode]
  dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
  return dataset


input_fn = lambda mode, params: (
    datasets.input_fn_tplt(mode, params, extract=_extract, transform=_transform))
