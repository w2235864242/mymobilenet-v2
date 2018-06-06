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

"""Builds the CIFAR-10 network.

Summary of available functions:

 # Compute input images and labels for training. If you would like to run
 # evaluations, use input() instead.
 inputs, labels = distorted_inputs()

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import namedtuple
import functools
import gzip
import os
import re
import sys
import tarfile

import tensorflow.python.platform
from six.moves import urllib
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

slim = tf.contrib.slim

#from tensorflow.models.image.cifar10 import cifar10_input

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 64,
                            """Number of images to process in a batch.""")
# tf.app.flags.DEFINE_string('data_dir', 'cifar10_data/',
#                            """Path to the CIFAR-10 data directory.""")


# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 224

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 3
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 1101
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 219

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = IMAGE_SIZE
NUM_CLASSES = NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 35.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.001       # Initial learning rate.

# If a model is trained with multiple GPU's prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


# _CONV_DEFS specifies the MobileNet body
Conv = namedtuple('Conv', ['kernel', 'stride', 'depth'])
InvertedResidual = namedtuple('InvertedResidual', ['kernel', 'stride', 'depth', 'num', 't']) # t is the expension factor
_CONV_DEFS = [
    Conv(kernel=[3, 3], stride=2, depth=32),
    InvertedResidual(kernel=[3, 3], stride=1, depth=16, num=1, t=1),
    InvertedResidual(kernel=[3, 3], stride=2, depth=24, num=2, t=6),
    InvertedResidual(kernel=[3, 3], stride=2, depth=32, num=3, t=6),
    InvertedResidual(kernel=[3, 3], stride=2, depth=64, num=4, t=6),
    InvertedResidual(kernel=[3, 3], stride=1, depth=96, num=3, t=6),
    InvertedResidual(kernel=[3, 3], stride=2, depth=160, num=3, t=6),
    InvertedResidual(kernel=[3, 3], stride=1, depth=320, num=1, t=6),
    Conv(kernel=[1, 1], stride=1, depth=1280)
]

@slim.add_arg_scope
def _inverted_residual_bottleneck(inputs, depth, stride, expand_ratio, scope=None):
  with tf.variable_scope(scope, 'InvertedResidual', [inputs]) as sc:
    depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
    output = slim.conv2d(inputs, expand_ratio*inputs.get_shape().as_list()[-1], 1, stride=1,
                              activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm, scope='conv')
    output = slim.separable_conv2d(output, None, 3, depth_multiplier=1, stride=stride,
                              activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm, scope='depthwise')
    output = slim.conv2d(output, depth, 1, stride=1,
                              activation_fn=None, normalizer_fn=slim.batch_norm, scope='pointwise')

    if stride==1 and depth==depth_in:
      shortcut = inputs
      output = shortcut + output

    return output



def mobilenet_v2_base(inputs,
                      final_endpoint='Conv2d_8',
                      min_depth=8,
                      depth_multiplier=1.0,
                      conv_defs=None,
                      output_stride=None,
                      scope=None):
  """Mobilenet v2.

  Constructs a Mobilenet v2 network from inputs to the given final endpoint.

  Args:
    inputs: a tensor of shape [batch_size, height, width, channels].
    final_endpoint: specifies the endpoint to construct the network up to. It
      can be one of ['Conv2d_0', 'InvertedResidual_16_0', 'InvertedResidual_24_0', 'InvertedResidual_24_1',
      'InvertedResidual_32_0', 'InvertedResidual_32_1', 'InvertedResidual_32_2',
      'InvertedResidual_64_0', 'InvertedResidual_64_1', 'InvertedResidual_64_2', 'InvertedResidual_64_3',
      'InvertedResidual_96_0', 'InvertedResidual_96_1', 'InvertedResidual_96_2',
      'InvertedResidual_160_0', 'InvertedResidual_160_1', 'InvertedResidual_160_2', 
      'InvertedResidual_320_0', 'Conv2d_8']
    min_depth: Minimum depth value (number of channels) for all convolution ops.
      Enforced when depth_multiplier < 1, and not an active constraint when
      depth_multiplier >= 1.
    depth_multiplier: Float multiplier for the depth (number of channels)
      for all convolution ops. The value must be greater than zero. Typical
      usage will be to set this value in (0, 1) to reduce the number of
      parameters or computation cost of the model.
    conv_defs: A list of ConvDef namedtuples specifying the net architecture.
    output_stride: An integer that specifies the requested ratio of input to
      output spatial resolution. If not None, then we invoke atrous convolution
      if necessary to prevent the network from reducing the spatial resolution
      of the activation maps. Allowed values are 8 (accurate fully convolutional
      mode), 16 (fast fully convolutional mode), 32 (classification mode).
    scope: Optional variable_scope.

  Returns:
    tensor_out: output tensor corresponding to the final_endpoint.
    end_points: a set of activations for external use, for example summaries or
                losses.

  Raises:
    ValueError: if final_endpoint is not set to one of the predefined values,
                or depth_multiplier <= 0, or the target output_stride is not
                allowed.
  """
  depth = lambda d: max(int(d * depth_multiplier), min_depth)
  end_points = {}

  # Used to find thinned depths for each layer.
  if depth_multiplier <= 0:
    raise ValueError('depth_multiplier is not greater than zero.')

  if conv_defs is None:
    conv_defs = _CONV_DEFS

  if output_stride is not None and output_stride not in [8, 16, 32]:
    raise ValueError('Only allowed output_stride values are 8, 16, 32.')

  with tf.variable_scope(scope, 'MobilenetV2', [inputs]):
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d], padding='SAME'):
      # The current_stride variable keeps track of the output stride of the
      # activations, i.e., the running product of convolution strides up to the
      # current network layer. This allows us to invoke atrous convolution
      # whenever applying the next convolution would result in the activations
      # having output stride larger than the target output_stride.
      current_stride = 1

      # The atrous convolution rate parameter.
      rate = 1

      net = inputs
      for i, conv_def in enumerate(conv_defs):
        if output_stride is not None and current_stride == output_stride:
          # If we have reached the target output_stride, then we need to employ
          # atrous convolution with stride=1 and multiply the atrous rate by the
          # current unit's stride for use in subsequent layers.
          layer_stride = 1
          layer_rate = rate
          rate *= conv_def.stride
        else:
          layer_stride = conv_def.stride
          layer_rate = 1
          current_stride *= conv_def.stride

        if isinstance(conv_def, Conv):
          end_point = 'Conv2d_%d' % i
          net = slim.conv2d(net, depth(conv_def.depth), conv_def.kernel,
                            stride=conv_def.stride,
                            normalizer_fn=slim.batch_norm,
                            scope=end_point)
          end_points[end_point] = net
          if end_point == final_endpoint:
            return net, end_points

        elif isinstance(conv_def, InvertedResidual):
          for n in range(conv_def.num):
            end_point = 'InvertedResidual_{}_{}'.format(conv_def.depth, n)
            stride = conv_def.stride if n == 0 else 1
            net = _inverted_residual_bottleneck(net, depth(conv_def.depth), stride, conv_def.t, scope=end_point)
            end_points[end_point] = net
            if end_point == final_endpoint:
              return net, end_points
        else:
          raise ValueError('Unknown convolution type %s for layer %d'
                           % (conv_def.ltype, i))
  raise ValueError('Unknown final endpoint %s' % final_endpoint)


def mobilenet_v2(inputs,
                 num_classes=1000,
                 dropout_keep_prob=0.999,
                 is_training=True,
                 min_depth=8,
                 depth_multiplier=1.0,
                 conv_defs=None,
                 prediction_fn=tf.contrib.layers.softmax,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='MobilenetV2',
                 global_pool=False):
  """Mobilenet v2 model for classification.

  Args:
    inputs: a tensor of shape [batch_size, height, width, channels].
    num_classes: number of predicted classes. If 0 or None, the logits layer
      is omitted and the input features to the logits layer (before dropout)
      are returned instead.
    dropout_keep_prob: the percentage of activation values that are retained.
    is_training: whether is training or not.
    min_depth: Minimum depth value (number of channels) for all convolution ops.
      Enforced when depth_multiplier < 1, and not an active constraint when
      depth_multiplier >= 1.
    depth_multiplier: Float multiplier for the depth (number of channels)
      for all convolution ops. The value must be greater than zero. Typical
      usage will be to set this value in (0, 1) to reduce the number of
      parameters or computation cost of the model.
    conv_defs: A list of ConvDef namedtuples specifying the net architecture.
    prediction_fn: a function to get predictions out of logits.
    spatial_squeeze: if True, logits is of shape is [B, C], if false logits is
        of shape [B, 1, 1, C], where B is batch_size and C is number of classes.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.
    global_pool: Optional boolean flag to control the avgpooling before the
      logits layer. If false or unset, pooling is done with a fixed window
      that reduces default-sized inputs to 1x1, while larger inputs lead to
      larger outputs. If true, any input size is pooled down to 1x1.

  Returns:
    net: a 2D Tensor with the logits (pre-softmax activations) if num_classes
      is a non-zero integer, or the non-dropped-out input to the logits layer
      if num_classes is 0 or None.
    end_points: a dictionary from components of the network to the corresponding
      activation.

  Raises:
    ValueError: Input rank is invalid.
  """
  input_shape = inputs.get_shape().as_list()
  if len(input_shape) != 4:
    raise ValueError('Invalid input tensor rank, expected 4, was: %d' %
                     len(input_shape))

  with tf.variable_scope(scope, 'MobilenetV2', [inputs], reuse=reuse) as scope:
    with slim.arg_scope([slim.batch_norm, slim.dropout],
                        is_training=is_training):
      net, end_points = mobilenet_v2_base(inputs, scope=scope,
                                          min_depth=min_depth,
                                          depth_multiplier=depth_multiplier,
                                          conv_defs=conv_defs)
      with tf.variable_scope('Logits'):
        if global_pool:
          # Global average pooling.
          net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
          end_points['global_pool'] = net
        else:
          # Pooling with a fixed kernel size.
          kernel_size = _reduced_kernel_size_for_small_input(net, [7, 7])
          net = slim.avg_pool2d(net, kernel_size, padding='VALID',
                                scope='AvgPool_1a')
          end_points['AvgPool_1a'] = net
        if not num_classes:
          return net, end_points
        # 1 x 1 x 3
        net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_1b')
        logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                             normalizer_fn=None, scope='Conv2d_1c_1x1')
        if spatial_squeeze:
          logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
      end_points['Logits'] = logits
      if prediction_fn:
        end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
  return logits, end_points



mobilenet_v2.default_image_size = 224



def wrapped_partial(func, *args, **kwargs):
  partial_func = functools.partial(func, *args, **kwargs)
  functools.update_wrapper(partial_func, func)
  return partial_func


mobilenet_v2_075 = wrapped_partial(mobilenet_v2, depth_multiplier=0.75)
mobilenet_v2_050 = wrapped_partial(mobilenet_v2, depth_multiplier=0.50)
mobilenet_v2_025 = wrapped_partial(mobilenet_v2, depth_multiplier=0.25)


def _reduced_kernel_size_for_small_input(input_tensor, kernel_size):
  """Define kernel size which is automatically reduced for small input.

  If the shape of the input images is unknown at graph construction time this
  function assumes that the input images are large enough.

  Args:
    input_tensor: input tensor of size [batch_size, height, width, channels].
    kernel_size: desired kernel size of length 2: [kernel_height, kernel_width]

  Returns:
    a tensor with the kernel size.
  """
  shape = input_tensor.get_shape().as_list()
  if shape[1] is None or shape[2] is None:
    kernel_size_out = kernel_size
  else:
    kernel_size_out = [min(shape[1], kernel_size[0]),
                       min(shape[2], kernel_size[1])]
  return kernel_size_out


def mobilenet_v2_arg_scope(is_training=True,
                           weight_decay=0.00004,
                           stddev=0.09,
                           regularize_depthwise=False):
  """Defines the default MobilenetV2 arg scope.

  Args:
    is_training: Whether or not we're training the model.
    weight_decay: The weight decay to use for regularizing the model.
    stddev: The standard deviation of the trunctated normal weight initializer.
    regularize_depthwise: Whether or not apply regularization on depthwise.

  Returns:
    An `arg_scope` to use for the mobilenet v2 model.
  """
  batch_norm_params = {
      'is_training': is_training,
      'center': True,
      'scale': True,
      'decay': 0.9997,
      'epsilon': 0.001,
  }

  # Set weight_decay for weights in Conv and DepthSepConv layers.
  weights_init = tf.truncated_normal_initializer(stddev=stddev)
  regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
  if regularize_depthwise:
    depthwise_regularizer = regularizer
  else:
    depthwise_regularizer = None
  with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                      weights_initializer=weights_init,
                      activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      with slim.arg_scope([slim.conv2d], weights_regularizer=regularizer):
        with slim.arg_scope([slim.separable_conv2d],
                            weights_regularizer=depthwise_regularizer) as sc:
          return sc


def loss(logits, labels):
  """Add L2Loss to all the trainable variables.

  Add summary for for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]

  Returns:
    Loss tensor of type float.
  """
  # Reshape the labels into a dense Tensor of
  # shape [batch_size, NUM_CLASSES].
  sparse_labels = tf.reshape(labels, [FLAGS.batch_size, 1])
  # indices = tf.reshape(tfrange(FLAGS.batch_size), [FLAGS.batch_size, 1])
  indices = tf.reshape(range(FLAGS.batch_size), [FLAGS.batch_size, 1])
  concated = tf.concat([indices, sparse_labels],1)
  dense_labels = tf.sparse_to_dense(concated,
                                    [FLAGS.batch_size, NUM_CLASSES],
                                    1.0, 0.0)

  # Calculate the average cross entropy loss across the batch.
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=dense_labels, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name +' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op


def train(total_loss, global_step):
  """Train CIFAR-10 model.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.summary.scalar('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
      if update_ops:
          updates = tf.group(*update_ops)
          total_loss = control_flow_ops.with_dependencies([updates], total_loss)

  optimizer = tf.train.GradientDescentOptimizer(lr)
  train_step = slim.learning.create_train_op(total_loss, optimizer, global_step=global_step)

  #      opt = tf.train.GradientDescentOptimizer(lr)
  #      grads = opt.compute_gradients(total_loss)
  # #
  # # Apply gradients.
  # apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
  #
  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)

  # Add histograms for gradients.
  # for grad, var in grads:
  #   if grad is not None:
  #     tf.summary.histogram(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([train_step, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op


def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [IMAGE_SIZE, IMAGE_SIZE, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)
    return img, label

def distort_inputs_train(img,label):
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)
    img = tf.image.per_image_standardization(img)
    images, labels = tf.train.shuffle_batch([img, label],
                                            batch_size=FLAGS.batch_size, capacity=min_queue_examples + 3 * FLAGS.batch_size,
                                            min_after_dequeue=min_queue_examples)
    return images, labels

def distort_inputs_train_deal(img,label):
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)
    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(img)

    # Because these operations are not commutative, consider randomizing
    # randomize the order their operation.
    distorted_image = tf.image.random_brightness(distorted_image,
                                                 max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image,
                                               lower=0.2, upper=1.8)

    # Subtract off the mean and divide by the variance of the pixels.
    img = tf.image.per_image_standardization(distorted_image)

    images, labels = tf.train.shuffle_batch([img, label],
                                            batch_size=FLAGS.batch_size,
                                            capacity=min_queue_examples + 3 * FLAGS.batch_size,
                                            min_after_dequeue=min_queue_examples)
    return images, labels




# def maybe_download_and_extract():
#   """Download and extract the tarball from Alex's website."""
#   dest_directory = FLAGS.data_dir
#   if not os.path.exists(dest_directory):
#     os.mkdir(dest_directory)
#   filename = DATA_URL.split('/')[-1]
#   filepath = os.path.join(dest_directory, filename)
#   if not os.path.exists(filepath):
#     def _progress(count, block_size, total_size):
#       sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
#           float(count * block_size) / float(total_size) * 100.0))
#       sys.stdout.flush()
#     filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath,
#                                              reporthook=_progress)
#     print()
#     statinfo = os.stat(filepath)
#     print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
#     tarfile.open(filepath, 'r:gz').extractall(dest_directory)
