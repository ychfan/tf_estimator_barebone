"""WDSR model for DIV2K dataset
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from common.layers import conv2d_weight_norm
import model


def update_argparser(parser):
  model.update_argparser(parser)
  args, _ = parser.parse_known_args()
  if args.dataset == 'div2k':
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
        train_steps=2000000,
        learning_rate=((1500000, 1700000, 1800000, 1900000), (1e-3, 5e-4, 2e-4,
                                                              1e-4, 5e-5)),
        save_checkpoints_steps=50000,
        save_summary_steps=10000,
    )
  else:
    raise NotImplementedError('Needs to tune hyper parameters for new dataset.')


def model_fn(features, labels, mode, params, config):
  predictions = None
  loss = None
  train_op = None
  eval_metric_ops = None
  export_outputs = None

  lr = features['source']

  def _wdsr(x):

    def _residual_block(x, num_channels):
      skip = x
      x = conv2d_weight_norm(
          x,
          num_channels * 4,
          3,
          padding='same',
          name='conv0',
      )
      x = tf.nn.relu(x)
      x = conv2d_weight_norm(
          x,
          num_channels,
          3,
          padding='same',
          name='conv1',
      )
      return x + skip

    def _subpixel_block(x,
                        kernel_size,
                        num_channels=params.num_channels,
                        scale=params.scale):
      x = conv2d_weight_norm(
          x,
          num_channels * scale * scale,
          kernel_size,
          padding='same',
      )
      x = tf.depth_to_space(x, scale)
      return x

    MEAN = tf.constant([0.4488, 0.4371, 0.4040], dtype=tf.float32)
    x = x - MEAN
    with tf.variable_scope('skip'):
      skip = _subpixel_block(x, 5)
    with tf.variable_scope('input'):
      x = conv2d_weight_norm(
          x,
          params.num_residual_units,
          3,
          padding='same',
      )
    for i in range(params.num_blocks):
      with tf.variable_scope('layer{}'.format(i)):
        x = _residual_block(x, params.num_residual_units)
    with tf.variable_scope('output'):
      x = _subpixel_block(x, 3)
    x += skip
    x = x + MEAN
    return x

  sr = _wdsr(lr)

  predictions = tf.clip_by_value(sr, 0.0, 1.0)

  if mode == tf.estimator.ModeKeys.PREDICT:
    export_outputs = {
        tf.saved_model.signature_constants.PREDICT_METHOD_NAME:
            tf.estimator.export.PredictOutput(predictions)
    }
  else:
    hr = labels['target']
    loss = tf.losses.absolute_difference(labels=hr, predictions=sr)
    if mode == tf.estimator.ModeKeys.EVAL:

      def _ignore_boundary(images):
        boundary_size = params.scale + 6
        images = images[:, boundary_size:-boundary_size, boundary_size:
                        -boundary_size, :]
        return images

      def _float32_to_uint8(images):
        images = images * 255.0
        images = tf.round(images)
        images = tf.saturate_cast(images, tf.uint8)
        return images

      psnr = tf.image.psnr(
          _float32_to_uint8(_ignore_boundary(hr)),
          _float32_to_uint8(_ignore_boundary(predictions)),
          max_val=255,
      )
      eval_metric_ops = {
          'PSNR': tf.metrics.mean(psnr),
      }
    if mode == tf.estimator.ModeKeys.TRAIN:
      global_step = tf.train.get_or_create_global_step()
      learning_rate = tf.train.piecewise_constant(
          global_step, params.learning_rate[0], params.learning_rate[1])
      opt = tf.train.AdamOptimizer(learning_rate)
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        train_op = opt.minimize(loss, global_step=global_step)
      stats = tf.profiler.profile()
      print("Total parameters:", stats.total_parameters)

  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=eval_metric_ops,
      export_outputs=export_outputs,
  )
