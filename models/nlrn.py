"""NLRN model for denoise dataset
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import models


def update_argparser(parser):
  models.update_argparser(parser)
  args, _ = parser.parse_known_args()
  parser.add_argument(
      '--num-steps',
      help='Number of steps in recurrent networks',
      default=12,
      type=int)
  parser.add_argument(
      '--num-filters',
      help='Number of filters in networks',
      default=128,
      type=int)
  parser.add_argument(
      '--non-local-field-size',
      help='Size of receptive field in non-local blocks',
      default=35,
      type=int)
  parser.add_argument(
      '--init-ckpt',
      help='Checkpoint path to initialize',
      default=None,
      type=str,
  )
  parser.set_defaults(
      train_steps=500000,
      learning_rate=((100000, 200000, 300000, 400000, 450000),
                     (1e-3, 5e-4, 2.5e-4, 1.25e-4, 6.25e-5, 3.125e-5)),
      save_checkpoints_steps=20000,
      save_summary_steps=1000,
  )


def model_fn(features, labels, mode, params, config):
  predictions = None
  loss = None
  train_op = None
  eval_metric_ops = None
  export_outputs = None
  scaffold = None

  sources = features['source']

  net = _nlrn(sources, mode, params)

  predictions = tf.clip_by_value(net, 0.0, 1.0)

  if mode == tf.estimator.ModeKeys.PREDICT:
    export_outputs = {
        tf.saved_model.signature_constants.PREDICT_METHOD_NAME:
            tf.estimator.export.PredictOutput(predictions)
    }
  else:
    targets = labels['target']

    def central_crop(x, size=5):
      x_shape = tf.shape(x)
      x = tf.slice(
          x, [0, x_shape[1] // 2 - size // 2, x_shape[2] // 2 - size // 2, 0],
          [-1, size, size, -1])
      return x

    loss = tf.losses.mean_squared_error(
        labels=central_crop(targets), predictions=central_crop(net))
    if mode == tf.estimator.ModeKeys.EVAL:

      def _ignore_boundary(images):
        boundary_size = 16
        images = images[:, boundary_size:-boundary_size, boundary_size:
                        -boundary_size, :]
        return images

      def _float32_to_uint8(images):
        images = images * 255.0
        images = tf.round(images)
        images = tf.saturate_cast(images, tf.uint8)
        return images

      psnr = tf.image.psnr(
          _float32_to_uint8(_ignore_boundary(targets)),
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
        gvs = opt.compute_gradients(loss)
        capped_gvs = [(tf.clip_by_norm(grad, 2.5), var) for grad, var in gvs]
        train_op = opt.apply_gradients(capped_gvs, global_step=global_step)

      stats = tf.profiler.profile()
      print("Total parameters:", stats.total_parameters)

      if params.init_ckpt:
        init_fn = tf.contrib.framework.assign_from_checkpoint_fn(
            params.init_ckpt, tf.global_variables(), ignore_missing_vars=True)
        scaffold = tf.train.Scaffold(init_fn=init_fn)

  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=eval_metric_ops,
      export_outputs=export_outputs,
  )


def _nlrn(x, mode, params):
  training = mode == tf.estimator.ModeKeys.TRAIN
  skip = x
  x = tf.layers.batch_normalization(x, training=training)
  x = tf.layers.conv2d(
      x, params.num_filters, 3, padding='same', activation=None, name='conv1')
  y = x
  with tf.variable_scope("rnn"):
    for i in range(params.num_steps):
      if i == 0:
        x = _residual_block(
            x, y, params.num_filters, training, name='RB1', reuse=False)
      else:
        x = _residual_block(
            x, y, params.num_filters, training, name='RB1', reuse=True)

  x = tf.layers.batch_normalization(x, training=training)
  x = tf.nn.relu(x)
  x = tf.layers.conv2d(
      x,
      params.num_channels,
      3,
      padding='same',
      activation=None,
      name='conv_end')
  return x + skip


def _residual_block(x, y, filter_num, training, name, reuse):
  x = tf.layers.batch_normalization(x, training=training)
  x = tf.nn.relu(x)
  x = _non_local_block(x, 64, 128, training, 35, name='non_local', reuse=reuse)

  x = tf.layers.batch_normalization(x, training=training)
  x = tf.layers.conv2d(
      x,
      filter_num,
      3,
      padding='same',
      activation=None,
      name=name + '_a',
      reuse=reuse)
  x = tf.layers.batch_normalization(x, training=training)
  x = tf.nn.relu(x)
  x = tf.layers.conv2d(
      x,
      filter_num,
      3,
      padding='same',
      activation=None,
      name=name + '_b',
      reuse=reuse)

  x = tf.add(x, y)
  return x


def _non_local_block(x,
                     filter_num,
                     output_filter_num,
                     training,
                     field_size,
                     name,
                     reuse=False):
  x_theta = tf.layers.conv2d(
      x,
      filter_num,
      1,
      padding='same',
      activation=None,
      name=name + '_theta',
      reuse=reuse)
  x_phi = tf.layers.conv2d(
      x,
      filter_num,
      1,
      padding='same',
      activation=None,
      name=name + '_phi',
      reuse=reuse)
  x_g = tf.layers.conv2d(
      x,
      output_filter_num,
      1,
      padding='same',
      activation=None,
      name=name + '_g',
      reuse=reuse,
      kernel_initializer=tf.zeros_initializer())

  if True:
    x_theta_reshaped = tf.reshape(x_theta, [
        tf.shape(x_theta)[0],
        tf.shape(x_theta)[1] * tf.shape(x_theta)[2],
        tf.shape(x_theta)[3]
    ])
    x_phi_reshaped = tf.reshape(x_phi, [
        tf.shape(x_phi)[0],
        tf.shape(x_phi)[1] * tf.shape(x_phi)[2],
        tf.shape(x_phi)[3]
    ])
    x_phi_permuted = tf.transpose(x_phi_reshaped, perm=[0, 2, 1])
    x_mul1 = tf.matmul(x_theta_reshaped, x_phi_permuted)
    x_mul1_softmax = tf.nn.softmax(
        x_mul1, axis=-1)  # normalization for embedded Gaussian

    x_g_reshaped = tf.reshape(x_g, [
        tf.shape(x_g)[0],
        tf.shape(x_g)[1] * tf.shape(x_g)[2],
        tf.shape(x_g)[3]
    ])
    x_mul2 = tf.matmul(x_mul1_softmax, x_g_reshaped)
    x_mul2_reshaped = tf.reshape(x_mul2, [
        tf.shape(x_mul2)[0],
        tf.shape(x_phi)[1],
        tf.shape(x_phi)[2], output_filter_num
    ])
  else:
    x_theta = tf.expand_dims(x_theta, -2)
    x_phi_patches = tf.image.extract_image_patches(
        x_phi, [1, field_size, field_size, 1], [1, 1, 1, 1], [1, 1, 1, 1],
        padding='SAME')
    x_phi_patches = tf.reshape(x_phi_patches, [
        tf.shape(x_phi)[0],
        tf.shape(x_phi)[1],
        tf.shape(x_phi)[2],
        field_size * field_size,
        tf.shape(x_phi)[3],
    ])
    x_mul1 = tf.matmul(x_theta, x_phi_patches, transpose_b=True)
    x_mul1_softmax = tf.nn.softmax(x_mul1, axis=-1)
    x_g_patches = tf.image.extract_image_patches(
        x_g, [1, field_size, field_size, 1], [1, 1, 1, 1], [1, 1, 1, 1],
        padding='SAME')
    x_g_patches = tf.reshape(x_g_patches, [
        tf.shape(x_g)[0],
        tf.shape(x_g)[1],
        tf.shape(x_g)[2],
        field_size * field_size,
        tf.shape(x_g)[3],
    ])
    x_mul2 = tf.matmul(x_mul1_softmax, x_g_patches)
    x_mul2_reshaped = tf.reshape(
        x_mul2,
        [tf.shape(x)[0],
         tf.shape(x)[1],
         tf.shape(x)[2], output_filter_num])

  return tf.add(x, x_mul2_reshaped)
