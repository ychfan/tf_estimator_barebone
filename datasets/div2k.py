"""DIV2K dataset
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
from PIL import Image
import numpy as np

import tensorflow as tf

import datasets

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
NUM_CHANNELS = 3


def update_argparser(parser):
  datasets.update_argparser(parser)
  parser.add_argument(
      '--scale', help='Scale for image super-resolution', default=2, type=int)
  parser.add_argument(
      '--lr-patch-size',
      help='Number of pixels in height or width of LR patches',
      default=48,
      type=int)
  parser.set_defaults(
      num_channels=NUM_CHANNELS,
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
    lr_image = tf.image.decode_png(tf.read_file(lr_file), channels=NUM_CHANNELS)
    hr_image = tf.image.decode_png(tf.read_file(hr_file), channels=NUM_CHANNELS)
    return lr_image, hr_image

  dataset = dataset.map(
      _read_image,
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
                    [params.lr_patch_size, params.lr_patch_size, -1])
      hr_up = lr_up * params.scale
      hr_left = lr_left * params.scale
      hr_patch_size = params.lr_patch_size * params.scale
      hr = tf.slice(hr, [hr_up, hr_left, 0], [hr_patch_size, hr_patch_size, -1])

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

    lr = tf.image.convert_image_dtype(lr, tf.float32)
    hr = tf.image.convert_image_dtype(hr, tf.float32)

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


input_fn = lambda mode, params: (
    datasets.input_fn_tplt(mode, params, extract=_extract, transform=_transform))


def predict_input_fn():
  input_tensor = tf.placeholder(
      dtype=tf.float32, shape=[None, None, None, 3], name='input_tensor')
  features = {'source': input_tensor}
  return tf.estimator.export.ServingInputReceiver(
      features=features,
      receiver_tensors={
          tf.saved_model.signature_constants.PREDICT_INPUTS: input_tensor
      })


def test_saved_model():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--model-dir', help='GCS location to load exported model', required=True)
  parser.add_argument(
      '--input-dir', help='GCS location to load input images', required=True)
  parser.add_argument(
      '--output-dir', help='GCS location to load output images', required=True)
  parser.add_argument(
      '--ensemble',
      help='Whether to ensemble with 8x rotation and flip',
      default=False,
      action='store_true')
  args = parser.parse_args()

  with tf.Session(graph=tf.Graph()) as sess:
    metagraph_def = tf.saved_model.loader.load(
        sess, [tf.saved_model.tag_constants.SERVING], args.model_dir)
    signature_def = metagraph_def.signature_def[
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    input_tensor = sess.graph.get_tensor_by_name(
        signature_def.inputs['inputs'].name)
    output_tensor = sess.graph.get_tensor_by_name(
        signature_def.outputs['output'].name)
    if not os.path.isdir(args.output_dir):
      os.mkdir(args.output_dir)
    for input_file in os.listdir(args.input_dir):
      print(input_file)
      output_file = os.path.join(args.output_dir, input_file)
      input_file = os.path.join(args.input_dir, input_file)
      input_image = np.asarray(Image.open(input_file))

      def forward_images(images):
        images = images.astype(np.float32) / 255.0
        images = output_tensor.eval(feed_dict={input_tensor: images})
        return images

      if args.ensemble:

        def flip(image):
          images = [image]
          images.append(image[::-1, :, :])
          images.append(image[:, ::-1, :])
          images.append(image[::-1, ::-1, :])
          images = np.stack(images)
          return images

        def mean_of_flipped(images):
          image = (images[0] + images[1, ::-1, :, :] + images[2, :, ::-1, :] +
                   images[3, ::-1, ::-1, :]) * 0.25
          return image

        rotate = lambda images: np.swapaxes(images, 1, 2)

        input_images = flip(input_image)
        output_image1 = mean_of_flipped(forward_images(input_images))
        output_image2 = mean_of_flipped(
            rotate(forward_images(rotate(input_images))))
        output_image = (output_image1 + output_image2) * 0.5
      else:
        input_images = np.expand_dims(input_image, axis=0)
        output_images = forward_images(input_images)
        output_image = output_images[0]
      output_image = np.around(output_image * 255.0).astype(np.uint8)
      output_image = Image.fromarray(output_image, 'RGB')
      output_image.save(output_file)


if __name__ == '__main__':
  test_saved_model()
