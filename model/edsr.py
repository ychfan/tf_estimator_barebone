"""EDSR model for DIV2K dataset
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import model


def update_argparser(parser):
  model.update_argparser(parser)
  parser.add_argument(
      '--num-blocks',
      help='Number of residual blocks in networks',
      default=16,
      type=int)
  parser.add_argument(
      '--num-residual-units',
      help='Number of residual units in networks',
      default=32,
      type=int)
  parser.set_defaults(
      train_steps=1500000,
      #learning_rate=((10000, 1000000, 1200000, 1400000), (1e-3, 1e-4, 5e-5,
      #                                                    3e-5, 1e-5)),
      learning_rate=((1000000, 1200000, 1300000, 1400000), (1e-3, 5e-4, 3e-4,
                                                            1e-4, 5e-5)),
      save_checkpoints_steps=50000,
  )


def model_fn(features, labels, mode, params, config):
  lr = features['source']
  if mode != tf.estimator.ModeKeys.PREDICT:
    hr = labels['target']

  def _edsr(x):

    def _residual_block(x, num_channels):
      skip = x
      x = tf.layers.conv2d(
          x,
          num_channels * 4,
          3,
          padding='same',
          name='conv0',
      )
      x = tf.nn.relu(x)
      x = tf.layers.conv2d(
          x,
          num_channels,
          3,
          padding='same',
          name='conv1',
      )
      return x + skip

    def _subpixel_block(x, kernel_size):
      x = tf.layers.conv2d_transpose(
          x,
          params.num_channels,
          kernel_size * params.scale,
          params.scale,
      )
      boundary_size = (kernel_size - 1) * params.scale
      x = x[:, boundary_size:-boundary_size, boundary_size:-boundary_size, :]
      return x

    MEAN = tf.constant([0.4488, 0.4371, 0.4040], dtype=tf.float32)
    x = (x / 255.0) - MEAN
    with tf.variable_scope('skip'):
      skip = _subpixel_block(x, 5)
    with tf.variable_scope('input'):
      x = tf.layers.conv2d(
          x,
          params.num_residual_units,
          3,
          padding='valid',
      )
    for i in range(params.num_blocks):
      with tf.variable_scope('layer{}'.format(i)):
        x = _residual_block(x, params.num_residual_units)
    with tf.variable_scope('output'):
      x = _subpixel_block(x, 3)
    x = x + skip
    x = (x + MEAN) * 255.0
    return x

  global_step = tf.train.get_or_create_global_step()

  boundary_size_lr = 2
  boundary_size_hr = boundary_size_lr * params.scale
  if mode == tf.estimator.ModeKeys.TRAIN:
    hr = hr[:, boundary_size_hr:-boundary_size_hr, boundary_size_hr:
            -boundary_size_hr, :]
  else:
    lr = tf.pad(lr, [[0, 0], [boundary_size_lr, boundary_size_lr],
                     [boundary_size_lr, boundary_size_lr], [0, 0]], 'SYMMETRIC')

  sr = _edsr(lr)

  predictions = tf.clip_by_value(tf.round(sr), 0.0, 255.0)

  if mode != tf.estimator.ModeKeys.PREDICT:
    loss = tf.losses.absolute_difference(labels=hr, predictions=sr)
  else:
    loss = None

  if mode == tf.estimator.ModeKeys.TRAIN:
    learning_rate = tf.train.piecewise_constant(
        global_step, params.learning_rate[0], params.learning_rate[1])
    opt = tf.train.AdamOptimizer(learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = opt.minimize(loss, global_step=global_step)
  else:
    train_op = None

  if mode == tf.estimator.ModeKeys.EVAL:
    boundary_size_mse = params.scale + 6
    psnr = tf.image.psnr(
        hr[:, boundary_size_mse:-boundary_size_mse, boundary_size_mse:
           -boundary_size_mse, :],
        predictions[:, boundary_size_mse:-boundary_size_mse, boundary_size_mse:
                    -boundary_size_mse, :],
        max_val=255.0,
    )
    eval_metrics = {
        'PSNR': tf.metrics.mean(psnr),
    }
  else:
    eval_metrics = None

  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=eval_metrics,
      export_outputs=None,
  )
