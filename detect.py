import os
import tensorflow as tf
import numpy as np

from model import DCGAN

flags = tf.app.flags
flags.DEFINE_float("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 108, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", None,
                     "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", None,
                     "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_boolean("crop", False, "True for training, False for testing [False]")
flags.DEFINE_integer("latent_dim", 100, "Number of images to generate during test. [100]")

# detect param
flags.DEFINE_float("lr", 0.01, "adam learning_rate")
flags.DEFINE_float("beta1", 0.9, "adam param beta1")
flags.DEFINE_float("beta2", 0.999, "adam param beta2")
flags.DEFINE_float("eps", 1e-8, "adam param eps")
flags.DEFINE_integer("Iter", 500, "iteration")
flags.DEFINE_string("outDir", "completions", "output Dir")

FLAGS = flags.FLAGS

# assert(os.path.exists(FLAGS.checkpoint_dir))

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    dcgan = DCGAN(
        sess,
        input_width=FLAGS.input_width,
        input_height=FLAGS.input_height,
        output_width=FLAGS.output_width,
        output_height=FLAGS.output_height,
        batch_size=FLAGS.batch_size,
        sample_num=FLAGS.batch_size,
        z_dim=FLAGS.latent_dim,
        dataset_name=FLAGS.dataset,
        input_fname_pattern=FLAGS.input_fname_pattern,
        crop=FLAGS.crop,
        checkpoint_dir=FLAGS.checkpoint_dir)

    dcgan.detect_anomaly(FLAGS)
