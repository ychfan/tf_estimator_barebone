"""DIV2K dataset
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
from hashlib import sha256
from PIL import Image
import numpy as np

import tensorflow as tf

import datasets

NUM_CHANNELS = 1


def update_argparser(parser):
  datasets.update_argparser(parser)
  parser.add_argument(
      '--noise-sigma',
      help='Scale for image super-resolution',
      default=25,
      type=float)
  parser.add_argument(
      '--train-patch-size',
      help='Number of pixels in height or width of patches',
      default=43,
      type=int)
  parser.add_argument(
      '--eval-patch-size',
      help='Number of pixels in height or width of patches',
      default=43,
      type=int)
  parser.add_argument(
      '--train-flist',
      help='GCS location to write checkpoints and export models',
      type=str,
      required=True)
  parser.add_argument(
      '--eval-flist',
      help='GCS location to write checkpoints and export models',
      type=str,
      required=True)
  parser.set_defaults(
      num_channels=NUM_CHANNELS,
      train_batch_size=16,
      eval_batch_size=1,
      shuffle_buffer_size=800,
  )


def _extract(mode, params):
  flist = {
      tf.estimator.ModeKeys.TRAIN: params.train_flist,
      tf.estimator.ModeKeys.EVAL: params.eval_flist,
  }[mode]
  with open(flist) as f:
    image_files = f.read().splitlines()
  dataset = tf.data.Dataset.from_tensor_slices((image_files,))

  dataset = dataset.map(
      tf.read_file,
      num_parallel_calls=params.num_data_threads,
  )
  dataset = dataset.cache()

  def _decode_image(image_file):
    image = tf.image.decode_png(image_file, channels=params.num_channels)
    return image

  dataset = dataset.map(
      _decode_image,
      num_parallel_calls=params.num_data_threads,
  )

  return dataset


def _transform(dataset, mode, params):
  if mode == tf.estimator.ModeKeys.TRAIN:
    dataset = dataset.shuffle(params.shuffle_buffer_size)
    dataset = dataset.repeat()

  def _preprocess(target):
    if mode == tf.estimator.ModeKeys.TRAIN:
      target = tf.image.random_crop(
          target, [params.train_patch_size, params.train_patch_size, 1])
      target = tf.image.random_flip_left_right(target)
      target = tf.image.random_flip_up_down(target)
      pred = tf.less(
          tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32),
          0.5)
      target = tf.cond(pred, lambda: tf.image.rot90(target), lambda: target)
    else:
      target = tf.image.resize_image_with_crop_or_pad(
          target, params.eval_patch_size, params.eval_patch_size)

    target = tf.image.convert_image_dtype(target, tf.float32)
    source = target + tf.random.normal(
        tf.shape(target),
        mean=0,
        stddev=params.noise_sigma / 255.0,
        seed=None if mode == tf.estimator.ModeKeys.TRAIN else 0)

    return {'source': source}, {'target': target}

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
      dtype=tf.float32,
      shape=[None, None, None, NUM_CHANNELS],
      name='input_tensor')
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
      '--noise-sigma',
      help='Scale for image super-resolution',
      default=25,
      type=float)
  parser.add_argument(
      '--patch-size',
      help='Number of pixels in height or width of patches',
      default=43,
      type=int)
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
      os.makedirs(args.output_dir)
    psnr_list = []
    for input_file in os.listdir(args.input_dir):
      print(input_file)
      sha = sha256(input_file.encode('utf-8'))
      seed = np.frombuffer(sha.digest(), dtype='uint32')
      rstate = np.random.RandomState(seed)

      output_file = os.path.join(args.output_dir, input_file)
      input_file = os.path.join(args.input_dir, input_file)
      input_image = np.asarray(Image.open(input_file))
      input_image = input_image.astype(np.float32) / 255.0

      def forward_images(images):
        images = output_tensor.eval(feed_dict={input_tensor: images})
        return images

      stride = 7
      h_idx_list = list(
          range(0, input_image.shape[0] - args.patch_size,
                stride)) + [input_image.shape[0] - args.patch_size]
      w_idx_list = list(
          range(0, input_image.shape[1] - args.patch_size,
                stride)) + [input_image.shape[1] - args.patch_size]
      output_image = np.zeros(input_image.shape)
      overlap = np.zeros(input_image.shape)
      noise_image = input_image + rstate.normal(0, args.noise_sigma / 255.0,
                                                input_image.shape)
      for h_idx in h_idx_list:
        for w_idx in w_idx_list:
          # print(h_idx, w_idx)
          input_patch = noise_image[h_idx:h_idx + args.patch_size, w_idx:
                                    w_idx + args.patch_size]
          input_patch = np.expand_dims(input_patch, axis=-1)
          input_patch = np.expand_dims(input_patch, axis=0)
          output_patch = forward_images(input_patch)
          output_patch = output_patch[0, :, :, 0]
          output_image[h_idx:h_idx + args.patch_size, w_idx:
                       w_idx + args.patch_size] += output_patch
          overlap[h_idx:h_idx + args.patch_size, w_idx:
                  w_idx + args.patch_size] += 1
      output_image /= overlap

      def psnr(im1, im2):
        im1_uint8 = np.rint(np.clip(im1 * 255, 0, 255))
        im2_uint8 = np.rint(np.clip(im2 * 255, 0, 255))
        diff = np.abs(im1_uint8 - im2_uint8).flatten()
        rmse = np.sqrt(np.mean(np.square(diff)))
        psnr = 20 * np.log10(255.0 / rmse)
        print(psnr)
        return psnr

      psnr_list.append(psnr(output_image, input_image))
      output_image = np.around(output_image * 255.0).astype(np.uint8)
      output_image = Image.fromarray(output_image)
      output_image.save(output_file)
    print('PSNR: ', np.average(np.array(psnr_list)))


if __name__ == '__main__':
  test_saved_model()
