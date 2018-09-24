"""ResNet model for CIFAR-10 dataset
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import model


def update_argparser(parser):
  model.update_argparser(parser)
  parser.add_argument(
      '--num-layers',
      help='Number of layers in networks',
      default=110,
      type=int)
  parser.add_argument(
      '--mixup',
      help='Hyper parameter for mixup training',
      default=0.0,
      type=float)
  parser.set_defaults(
      train_steps=150000,
      learning_rate=((32000, 48000, 120000), (0.1, 0.01, 0.001, 0.0002)),
      save_checkpoints_steps=5000,
  )


def model_fn(features, labels, mode, params, config):
  images = features['image']
  if mode != tf.estimator.ModeKeys.PREDICT:
    labels = labels['label']
  if params.mixup > 0.0 and mode == tf.estimator.ModeKeys.TRAIN:
    labels_onehot = tf.one_hot(labels, params.num_classes)
    images_aux = tf.reverse(images, [0])
    labels_onehot_aux = tf.reverse(labels_onehot, [0])
    beta = tf.distributions.Beta(params.mixup, params.mixup)
    lambdas = beta.sample(params.train_batch_size)
    labels_lambda = tf.expand_dims(lambdas, -1)
    images_lambda = tf.expand_dims(
        tf.expand_dims(tf.expand_dims(lambdas, -1), -1), -1)
    images = images_lambda * images + (1 - images_lambda) * images_aux
    labels_onehot = labels_lambda * labels_onehot + (
        1 - labels_lambda) * labels_onehot_aux

  def _resnet(x, training):
    kernel_init = tf.variance_scaling_initializer(
        scale=1.0 / 3, mode='fan_in', distribution='uniform')
    weight_decay = tf.train.exponential_decay(0.0002, global_step, 480000, 0.2,
                                              True)
    kernel_reg = tf.contrib.layers.l2_regularizer(weight_decay)

    def _bn_relu(x, training):
      x = tf.layers.batch_normalization(
          x, axis=1, training=training, fused=True)
      x = tf.nn.relu(x)
      return x

    def _residual_block(x, num_channels, pooling, training):
      skip = x
      if pooling:
        stride = 2
        num_channels *= 2
      else:
        stride = 1
      with tf.variable_scope('act0'):
        x = _bn_relu(x, training)
      x = tf.layers.conv2d(
          x,
          num_channels,
          3,
          strides=stride,
          padding='same',
          data_format='channels_first',
          kernel_initializer=kernel_init,
          kernel_regularizer=kernel_reg,
          name='conv0',
      )
      with tf.variable_scope('act1'):
        x = _bn_relu(x, training)
      x = tf.layers.conv2d(
          x,
          num_channels,
          3,
          padding='same',
          data_format='channels_first',
          kernel_initializer=kernel_init,
          kernel_regularizer=kernel_reg,
          name='conv1',
      )
      if pooling:
        skip = tf.layers.average_pooling2d(
            skip, 2, 2, padding='same', data_format='channels_first')
        skip = tf.pad(skip, [[0, 0], [0, num_channels // 2], [0, 0], [0, 0]])
      return x + skip

    x = tf.transpose(x, [0, 3, 1, 2])
    with tf.variable_scope('input'):
      x = tf.layers.conv2d(
          x,
          16,
          3,
          padding='same',
          data_format='channels_first',
          kernel_initializer=kernel_init,
          kernel_regularizer=kernel_reg,
      )
    assert ((params.num_layers - 2) % 6 == 0,
            'number of layers should be one of 20, 32, 44, 56, 110, 1202')
    n = (params.num_layers - 2) // 6
    with tf.variable_scope('stage1'):
      for i in range(n):
        with tf.variable_scope('layer{}'.format(i)):
          x = _residual_block(x, 16, i == n - 1, training)
    with tf.variable_scope('stage2'):
      for i in range(n):
        with tf.variable_scope('layer{}'.format(i)):
          x = _residual_block(x, 32, i == n - 1, training)
    with tf.variable_scope('stage3'):
      for i in range(n):
        with tf.variable_scope('layer{}'.format(i)):
          x = _residual_block(x, 64, False, training)
    with tf.variable_scope('fc'):
      x = _bn_relu(x, training)
      x = tf.reduce_mean(x, [-1, -2])
      x = tf.layers.dense(
          x,
          params.num_classes,
          kernel_initializer=tf.random_normal_initializer(stddev=1e-3),
          kernel_regularizer=kernel_reg,
      )
    return x

  global_step = tf.train.get_or_create_global_step()

  logits = _resnet(images, mode == tf.estimator.ModeKeys.TRAIN)

  predictions = tf.argmax(logits, axis=-1)

  if params.mixup > 0.0 and mode == tf.estimator.ModeKeys.TRAIN:
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=labels_onehot,
        logits=logits,
        reduction=tf.losses.Reduction.MEAN)
    accuracy = tf.reduce_mean(
        tf.cast(
            tf.equal(predictions, tf.argmax(labels_onehot, axis=-1)),
            tf.float32))
  else:
    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=labels, logits=logits, reduction=tf.losses.Reduction.MEAN)
    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(predictions, labels), tf.float32))
  tf.summary.scalar('cross_entropy', loss)
  tf.summary.scalar('accuracy', accuracy)

  reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
  loss = tf.add_n([loss] + reg_losses)

  learning_rate = tf.train.piecewise_constant(
      global_step, params.learning_rate[0], params.learning_rate[1])
  opt = tf.train.MomentumOptimizer(learning_rate, 0.9)
  if mode == tf.estimator.ModeKeys.TRAIN:
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = opt.minimize(loss, global_step=global_step)
  else:
    train_op = None

  eval_metrics = {
      'accuracy': tf.metrics.accuracy(labels, predictions),
      #'top_5_accuracy': tf.metrics.mean(tf.nn.in_top_k(logits, labels, 5)),
  }

  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=eval_metrics,
      export_outputs=None,
  )
