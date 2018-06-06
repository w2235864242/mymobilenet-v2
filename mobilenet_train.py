# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A binary to train CIFAR-10 using a single GPU.

Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
import math


import tensorflow.python.platform
from tensorflow.python.platform import gfile

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

#from tensorflow.models.image.cifar10 import cifar10
# import cifar10
import mobilenet_model as cifar10

slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', 'test_train/',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 80000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('num_examples', 219,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_string('log_dir', 'logger/',
                           """Directory where to write event logs """
                           """and checkpoint.""")

EVAL_ITER_NUM = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))

# def evaluation(logits, labels):
#   """Evaluate the quality of the logits at predicting the label.
#   Args:
#     logits: Logits tensor, float - [batch_size, NUM_CLASSES].
#     labels: Labels tensor, int32 - [batch_size], with values in the
#       range [0, NUM_CLASSES).
#   Returns:
#     A scalar int32 tensor with the number of examples (out of batch_size)
#     that were predicted correctly.
#   """
#   with tf.variable_scope('accuracy') as scope:
#       correct = tf.nn.in_top_k(logits, labels, 1)
#       correct = tf.cast(correct, tf.float16)
#       accuracy = tf.reduce_mean(correct)
#       tf.summary.scalar(scope.name+'/accuracy', accuracy)
#   return accuracy


def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)

    # Get images and labels for CIFAR-10.
    # images, labels = cifar10.distorted_inputs()
    images, labels = cifar10.read_and_decode("imagetrain.tfrecords")
    images_eval,labels_eval = cifar10.read_and_decode("imagetest.tfrecords")

    images, labels = cifar10.distort_inputs_train(images, labels)
    images_eval,labels_eval = cifar10.distort_inputs_train(images_eval,labels_eval)
    tf.summary.image('input_image', images, 20)

    # Build a Graph that computes the logits predictions from the
    # inference model.

    mobilenet_v2_arg_scope = cifar10.mobilenet_v2_arg_scope

    with slim.arg_scope(mobilenet_v2_arg_scope(is_training=True)):
        logits, end_points = cifar10.mobilenet_v2(images, num_classes=3, is_training=True)

    with tf.name_scope('train_cross_entropy'):
        loss = cifar10.loss(logits=logits,labels=labels)
        tf.summary.scalar('train_cross entropy', loss)

    with tf.name_scope('train_accuracy'):
        # p_label = np.argmax(batch_label, axis=0)
        # accuracy = np.mean(p_label == predict)
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float64)
        train__acc = tf.reduce_mean(correct)
        tf.summary.scalar('accuracy', train__acc)

    with slim.arg_scope(mobilenet_v2_arg_scope(is_training=False)):
        logits_eval, end_points_eval = cifar10.mobilenet_v2(images_eval, num_classes=3, is_training=False, reuse=True)

    with tf.name_scope('eval_cross_entropy'):
        loss_eval = cifar10.loss(logits=logits_eval,labels=labels_eval)
        tf.summary.scalar('eval_cross entropy', loss_eval)

    with tf.name_scope('eval_accuracy'):
        # e_label = np.argmax(batch_eval_label, axis=0)
        # e_accuracy = np.mean(e_label == predict_eval)
        correct = tf.nn.in_top_k(logits_eval, labels_eval, 1)
        correct = tf.cast(correct, tf.float64)
        e_accuracy = tf.reduce_mean(correct)
        tf.summary.scalar('eval_accuracy', e_accuracy)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = cifar10.train(loss, global_step)

    # Create a saver.
    saver = tf.train.Saver(tf.global_variables())

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    # Start running operations on the Graph.
    sess = tf.Session(config=tf.ConfigProto(
        log_device_placement=FLAGS.log_device_placement))

    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()
    sess.run(init)

    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      print ("restore from file")
    else:
      print('No checkpoint file found')

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.summary.FileWriter(FLAGS.log_dir,
                                            graph=sess.graph)
    filename1 = 'train_log.txt'
    filename2 = 'eval_log.txt'
    with open(filename1, 'w') as f1:
        f1.write("train_loss " + "train_accuracy\n" )
    with open(filename2, 'w') as f2:
        f2.write("eval_loss " + "eval_accuracy\n")

    for step in xrange(FLAGS.max_steps):
      start_time = time.time()
      _, loss_value, accuracy = sess.run([train_op, loss, train__acc])
      with open(filename1, 'a') as f1:
          f1.write(
              str(round(loss_value, 5)) + " " + str(round(accuracy, 5)) + "\n")

      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % 10 == 0:
        num_examples_per_step = FLAGS.batch_size
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = float(duration)

        format_str = ('%s: step %d, loss = %.5f (%.1f examples/sec; %.3f '
                      'sec/batch), train accuracy = %.2f%%')
        print (format_str % (datetime.now(), step, loss_value,
                             examples_per_sec, sec_per_batch, accuracy * 100))

      if step % 100 == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)


      if step % 500 == 0:
          dev_accuracy = []
          dev_cross_entropy = []
          for eval_idx in xrange(EVAL_ITER_NUM):
              # eval_loss_v, _trainY, _predict = sess.run([loss, trainY, predict], feed_dict ={train_status: False})
              eval_loss_v, eval_accuracy = sess.run([loss_eval, e_accuracy])
              dev_accuracy.append(eval_accuracy)
              dev_cross_entropy.append(eval_loss_v)

          format_str = "eval_step %d, eval_loss = %.5f, eval accuracy = %.2f%%"
          print(format_str % (step, np.mean(dev_cross_entropy), np.mean(dev_accuracy) * 100))
          with open(filename2, 'a') as f2:
              f2.write(str(round(np.mean(dev_cross_entropy), 5)) + " " + str(round(np.mean(dev_accuracy), 5)) + "\n")

      # plot.plot('dev accuracy', np.mean(dev_accuracy))
      # plot.plot('dev cross entropy', np.mean(dev_cross_entropy))

      # Save the model checkpoint periodically.
      if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)

    summary_writer.close()



def main(argv=None):  # pylint: disable=unused-argument

  if gfile.Exists(FLAGS.train_dir):
    # gfile.DeleteRecursively(FLAGS.train_dir)
    pass
  else:
    gfile.MakeDirs(FLAGS.train_dir)
  if gfile.Exists(FLAGS.log_dir):
      # gfile.DeleteRecursively(FLAGS.train_dir)
      pass
  else:
      gfile.MakeDirs(FLAGS.log_dir)
  train()


if __name__ == '__main__':
  tf.app.run()

