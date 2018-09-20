"""Trainer
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import importlib

import tensorflow as tf

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--model',
      help='Model name',
      default='resnet',
      type=str,
  )
  # Input Arguments
  parser.add_argument(
      '--dataset',
      help='Dataset name',
      default='cifar10',
      type=str,
  )
  parser.add_argument(
      '--num-epochs',
      help="""\
      Maximum number of training data epochs on which to train.
      If both --max-steps and --num-epochs are specified,
      the training job will run for --max-steps or --num-epochs,
      whichever occurs first. If unspecified will run for --max-steps.\
      """,
      default=1,
      type=int,
  )
  parser.add_argument(
      '--buffer-size',
      help='Buffer size for training steps',
      type=int,
      default=10000)
  parser.add_argument(
      '--train-batch-size',
      help='Batch size for training steps',
      type=int,
      default=32)
  parser.add_argument(
      '--eval-batch-size',
      help='Batch size for evaluation steps',
      type=int,
      default=32)
  parser.add_argument(
      '--num-data-threads',
      help='Batch size for evaluation steps',
      type=int,
      default=8)
  # Training arguments
  parser.add_argument(
      '--job-dir',
      help='GCS location to write checkpoints and export models',
      required=True)

  # Argument to turn on all logging
  parser.add_argument(
      '--verbosity',
      choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
      default='INFO',
  )
  # Experiment arguments
  parser.add_argument(
      '--train-steps',
      help="""\
      Steps to run the training job for. If --num-epochs is not specified,
      this must be. Otherwise the training job will run indefinitely.\
      """,
      default=None,
      type=int)
  parser.add_argument(
      '--eval-steps',
      help='Number of steps to run evalution for at each checkpoint',
      default=None,
      type=int)
  parser.add_argument(
      '--eval-throttle-secs',
      help='Seconds between evalutions',
      default=1000000,
      type=int)
  parser.add_argument(
      '--random-seed',
      help='Random seed for TensorFlow',
      default=None,
      type=int)
  # Performance tuning parameters
  parser.add_argument(
      '--allow-growth',
      help='Whether to enable allow_growth in GPU_Options',
      default=False,
      type=bool)
  parser.add_argument(
      '--xla',
      help='Whether to enable XLA auto-jit compilation',
      default=False,
      type=bool)

  args, _ = parser.parse_known_args()

  # Set python level verbosity
  tf.logging.set_verbosity(args.verbosity)
  # Set C++ Graph Execution level verbosity
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(
      tf.logging.__dict__[args.verbosity] / 10)

  # Run the training job
  model_module = importlib.import_module('model.' + args.model)
  model_module.update_argparser(parser)
  dataset_module = importlib.import_module('dataset.' + args.dataset)
  dataset_module.update_argparser(parser)
  args = parser.parse_args()
  hparams = tf.contrib.training.HParams(**args.__dict__)
  print(hparams)

  model_fn = getattr(model_module, 'model_fn')
  input_fn = getattr(dataset_module, 'input_fn')
  train_input_fn = lambda: input_fn(
      mode=tf.estimator.ModeKeys.TRAIN,
      params=hparams,
  )
  eval_input_fn = lambda: input_fn(
      mode=tf.estimator.ModeKeys.EVAL,
      params=hparams,
  )

  session_config = tf.ConfigProto()
  session_config.gpu_options.allow_growth = args.allow_growth
  if args.xla:
    session_config.graph_options.optimizer_options.global_jit_level = (
        tf.OptimizerOptions.ON_1)
  run_config = tf.estimator.RunConfig(
      model_dir=hparams.job_dir,
      tf_random_seed=hparams.random_seed,
      session_config=session_config,
  )
  estimator = tf.estimator.Estimator(
      model_fn=model_fn,
      model_dir=None,
      config=run_config,
      params=hparams,
      # warm_start_from=None,
  )
  train_spec = tf.estimator.TrainSpec(
      input_fn=train_input_fn,
      max_steps=hparams.train_steps,
      hooks=[
          # tf.train.SummarySaverHook(
          #     save_secs=1000,
          #     output_dir=hparams.job_dir,
          #     scaffold=tf.train.Scaffold(summary_op=tf.summary.merge_all()),
          # ),
      ],
  )
  eval_spec = tf.estimator.EvalSpec(
      input_fn=eval_input_fn,
      steps=hparams.eval_steps,
      name=None,
      hooks=[
          # tf.train.SummarySaverHook(
          #     save_steps=1,
          #     output_dir=hparams.job_dir,
          #     scaffold=tf.train.Scaffold(summary_op=tf.summary.merge_all()),
          # ),
      ],
      exporters=None,
      throttle_secs=hparams.eval_throttle_secs,
  )
  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
