"""DIV2K dataset
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random

import tensorflow as tf

import dataset

REMOTE_URL = 'http://data.vision.ee.ethz.ch/cvl/DIV2K/'
TRAIN_LR_ARCHIVE_NAME = lambda s: 'DIV2K_train_LR_bicubic_X{}.zip'.format(s)
TRAIN_HR_ARCHIVE_NAME = 'DIV2K_train_HR.zip'
EVAL_LR_ARCHIVE_NAME = lambda s: 'DIV2K_valid_LR_bicubic_X{}.zip'.format(s)
EVAL_HR_ARCHIVE_NAME = 'DIV2K_valid_HR.zip'
LOCAL_DIR = 'data/DIV2K/'
TRAIN_LR_DIR = lambda s: LOCAL_DIR + 'DIV2K_train_LR_bicubic/X{}/'.format(s)
TRAIN_HR_DIR = LOCAL_DIR + 'DIV2K_train_HR/'
EVAL_LR_DIR = lambda s: LOCAL_DIR + 'DIV2K_valid_LR_bicubic/X{}/'.format(s)
EVAL_HR_DIR = LOCAL_DIR + 'DIV2K_valid_HR/'


def update_argparser(parser):
  dataset.update_argparser(parser)
  parser.add_argument(
      '--scale', help='Scale for image super-resolution', default=2, type=int)
  parser.add_argument(
      '--lr-patch-size',
      help='Number of pixels in height or width of LR patches',
      default=48,
      type=int)
  parser.set_defaults(
      num_channels=3,
      train_batch_size=16,
      eval_batch_size=1,
      shuffle_buffer_size=800,
  )


def _extract(mode, params):
  lr_dir = {
      tf.estimator.ModeKeys.TRAIN: TRAIN_LR_DIR(params.scale),
      tf.estimator.ModeKeys.EVAL: EVAL_LR_DIR(params.scale),
  }[mode]
  #lr_dir = os.path.expanduser(lr_dir)
  hr_dir = {
      tf.estimator.ModeKeys.TRAIN: TRAIN_HR_DIR,
      tf.estimator.ModeKeys.EVAL: EVAL_HR_DIR,
  }[mode]

  def list_files(d):
    files = sorted(os.listdir(d))
    files = [os.path.join(d, f) for f in files]
    return files

  lr_files = list_files(lr_dir)
  hr_files = list_files(hr_dir)
  dataset = tf.data.Dataset.from_tensor_slices((lr_files, hr_files))

  def _read_image(lr_file, hr_file):
    lr_image = tf.image.decode_png(tf.read_file(lr_file), channels=3)
    hr_image = tf.image.decode_png(tf.read_file(hr_file), channels=3)
    return lr_image, hr_image

  dataset = dataset.map(
      _read_image,
      num_parallel_calls=params.num_data_threads,
  )
  dataset = dataset.cache()
  return dataset

  def __generator(lr_dir, hr_dir):
    for lr, hr in zip(sorted(os.listdir(lr_dir)), sorted(os.listdir(hr_dir))):
      lr = os.path.join(lr_dir, lr)
      hr = os.path.join(hr_dir, hr)
      yield lr, hr

  dataset = tf.data.Dataset.from_generator(
      generator=lambda: __generator(lr_dir, hr_dir),
      output_types=(tf.string, tf.string),
  )

  def __read_image(mode, lr, hr):
    lr = tf.image.decode_png(tf.read_file(lr), channels=3)
    hr = tf.image.decode_png(tf.read_file(hr), channels=3)
    return lr, hr

  dataset = dataset.map(
      lambda *args: __read_image(mode, *args),
      num_parallel_calls=params.num_data_threads,
  )
  dataset = dataset.cache()
  return dataset


def _transform(dataset, mode, params):
  if mode == tf.estimator.ModeKeys.TRAIN:
    dataset = dataset.shuffle(params.shuffle_buffer_size)
    dataset = dataset.repeat()

  def _preprocess(lr, hr):
    if mode == tf.estimator.ModeKeys.TRAIN:
      lr_shape = tf.shape(lr)
      lr_up = tf.random_uniform(
          shape=[],
          minval=0,
          maxval=lr_shape[0] - params.lr_patch_size,
          dtype=tf.int32)
      lr_left = tf.random_uniform(
          shape=[],
          minval=0,
          maxval=lr_shape[1] - params.lr_patch_size,
          dtype=tf.int32)
      lr = tf.slice(lr, [lr_up, lr_left, 0],
                    [params.lr_patch_size, params.lr_patch_size, 3])
      hr_up = lr_up * params.scale
      hr_left = lr_left * params.scale
      hr_patch_size = params.lr_patch_size * params.scale
      hr = tf.slice(hr, [hr_up, hr_left, 0], [hr_patch_size, hr_patch_size, 3])

      def _to_be_or_not_to_be(values, fn):

        def _to_be():
          return [fn(v) for v in values]

        def _not_to_be():
          return values

        pred = tf.less(
            tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32),
            0.5)
        values = tf.cond(pred, _to_be, _not_to_be)
        return values

      lr, hr = _to_be_or_not_to_be([lr, hr], tf.image.flip_left_right)
      lr, hr = _to_be_or_not_to_be([lr, hr], tf.image.flip_up_down)
      lr, hr = _to_be_or_not_to_be([lr, hr], tf.image.rot90)

    lr = tf.to_float(lr)
    hr = tf.to_float(hr)

    return {'source': lr}, {'target': hr}

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


def __transform(mode, lr, hr):
  if mode == tf.estimator.ModeKeys.TRAIN:
    # lr_shape = tf.shape(lr)
    # up_lr = tf.random_uniform(1, maxval=lr_shape[0] - PATCH_SIZE_LR, dtype=tf.int32)
    # left_lr = tf.random_uniform(1, maxval=lr_shape[1] - PATCH_SIZE_LR, dtype=tf.int32)
    # lr = tf.slice(lr, [up_lr, left_lr, 0], [PATCH_SIZE_LR, PATCH_SIZE_LR, 3])
    def extract_patches(lr, hr):
      up_lr = random.randrange(0, lr.shape[0] - PATCH_SIZE_LR + 1)
      left_lr = random.randrange(0, lr.shape[1] - PATCH_SIZE_LR + 1)
      up_hr = up_lr * SCALE
      left_hr = left_lr * SCALE
      lr = lr[up_lr:up_lr + PATCH_SIZE_LR, left_lr:left_lr + PATCH_SIZE_LR, :]
      hr = hr[up_hr:up_hr + PATCH_SIZE_HR, left_hr:left_hr + PATCH_SIZE_HR, :]
      if random.random() < 0.5:
        lr = lr[:, ::-1, :]
        hr = hr[:, ::-1, :]
      if random.random() < 0.5:
        lr = lr[::-1, :, :]
        hr = hr[::-1, :, :]
      if random.random() < 0.5:
        lr = lr.transpose(1, 0, 2)
        hr = hr.transpose(1, 0, 2)
      return lr, hr

    lr, hr = tf.py_func(
        extract_patches, [lr, hr], [tf.uint8, tf.uint8], stateful=False)
    lr = tf.reshape(lr, [PATCH_SIZE_LR, PATCH_SIZE_LR, 3])
    hr = tf.reshape(hr, [PATCH_SIZE_HR, PATCH_SIZE_HR, 3])

  lr = tf.to_float(lr)
  hr = tf.to_float(hr)

  return {"source": lr}, {"target": hr}


input_fn = lambda mode, params: (
    dataset.input_fn_tplt(mode, params, extract=_extract, transform=_transform))
