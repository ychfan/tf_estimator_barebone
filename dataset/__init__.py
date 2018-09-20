"""Dataset
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def update_argparser(parser):
  pass


def _extract(mode, params):
  raise NotImplementedError


def _transform(mode, image, label):
  raise NotImplementedError


def input_fn_tplt(mode, params, extract, transform):
  batch_size = {
      tf.estimator.ModeKeys.TRAIN: params.train_batch_size,
      tf.estimator.ModeKeys.EVAL: params.eval_batch_size,
  }[mode]
  dataset = extract(mode, params)
  if mode == tf.estimator.ModeKeys.TRAIN:
    dataset = dataset.shuffle(params.buffer_size)
    dataset = dataset.repeat(params.num_epochs)
  dataset = dataset.map(
      lambda *args: transform(mode, *args),
      num_parallel_calls=params.num_data_threads,
  )
  if mode == tf.estimator.ModeKeys.TRAIN:
    dataset = dataset.apply(
        tf.contrib.data.batch_and_drop_remainder(batch_size))
  else:
    dataset = dataset.batch(batch_size)
  dataset = dataset.prefetch(2)
  iterator = dataset.make_one_shot_iterator()
  features, labels = iterator.get_next()
  return features, labels

input_fn = lambda mode, params: (
    input_fn_tplt(mode, params, _extract, _transform))
