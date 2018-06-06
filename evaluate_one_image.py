from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from PIL import Image
import os
import numpy as np
import tensorflow as tf
import bn_ince_resnet_model as cifar10
import string
import time
from datetime import datetime



slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('checkpoint_dir', 'test_train/',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_string('test_one_dir', 'test_data/test_others',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                         """Whether to run eval only once.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 5,
                            """How often to run the eval.""")

IMAGE_SIZE = 299

def get_files(file_dir):
    '''
    Args:
        file_dir: file directory
    Returns:
        list of images and labels
    '''
    bus= []
    bus_labels = []
    kongl = []
    kongl_labels = []
    elp = []
    elp_labels = []
    flower = []
    flower_labels = []
    horse = []
    horse_labels = []
    for file in os.listdir(file_dir):
        name = string.split(file, sep ='.')
        if 300 < int(name[0]) < 400:
            bus.append(file_dir + '/' + file)
            bus_labels.append(1)
        elif 400 < int(name[0]) < 500:
            kongl.append(file_dir+ '/' + file)
            kongl_labels.append(0)
        elif 500 < int(name[0]) < 600:
            elp.append(file_dir + '/' + file)
            elp_labels.append(4)
        elif 600 < int(name[0]) < 700:
            flower.append(file_dir + '/'+ file)
            flower_labels.append(3)
        else:
            horse.append(file_dir + '/' + file)
            horse_labels.append(2)
    image_list = np.hstack((bus, kongl, elp, flower, horse))
    label_list = np.hstack((bus_labels, kongl_labels, elp_labels, flower_labels, horse_labels))

    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(float(i)) for i in label_list]
    return image_list, label_list


def get_one_image(train, label):
   '''Randomly pick one image from training data
   Return: ndarray
   '''
   n = len(train)
   ind = np.random.randint(0, n)
   img_dir = train[ind]
   label = label[ind]
   print (img_dir)

   image = Image.open(img_dir)
   image.show()
   image = image.resize([299, 299])
   image = np.array(image)

   return image, label
#


def evaluate_once(saver,logit, x):
    test_one_dir = FLAGS.test_one_dir
    images, labels = get_files(test_one_dir)
    image_raw, label = get_one_image(images, labels)
    print(image_raw.shape)

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:

            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            print('Loading success, global_stp is %s' % global_step)

        else:
            print('No checkpoint file found')
            return

        coord = tf.train.Coordinator()

        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                 start=True))
            prediction = sess.run(logit, feed_dict={x: image_raw})
            # print(prediction)
            # while not coord.should_stop():
            #     prediction = sess.run(logit, feed_dict={x: image_raw})
            max_index = np.argmax(prediction)
            # a = max_index.eval()
            # print('predict label: %d , ture label: %d, possibility: %.3f' % (a, label, prediction[:, a]))
            # print (label)
            if max_index == 0:
                print('%s: kongl, possibility: %.6f%%' % (datetime.now(), prediction[:, 0] * 100))
            elif max_index == 1:
                print('%s: bus, possibility: %.6f%%' % (datetime.now(), prediction[:, 1] * 100))
            elif max_index == 2:
                print('%s: horse, possibility: %.6f%%' % (datetime.now(), prediction[:, 2] * 100))
            elif max_index == 3:
                print('%s: flower, possibility: %.6f%%' % (datetime.now(), prediction[:, 3] * 100))
            elif max_index == 4:
                print('%s: elp, possibility: %.6f%%' % (datetime.now(), prediction[:, 4] * 100))
                #
        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def evaluate_one_image():


    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32, shape=[299, 299, 3])
        img = tf.reshape(x, [IMAGE_SIZE, IMAGE_SIZE, 3])
        img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
        img = tf.image.per_image_standardization(img)
        image = tf.reshape(img, [1, 299, 299, 3])
        # image, _ = tf.train.batch([img, label],
        #                               batch_size=1, capacity=100)

        inception_resnet_v2_arg_scope = cifar10.inception_resnet_v2_arg_scope

        with slim.arg_scope(inception_resnet_v2_arg_scope(is_training=False)):
            logits, end_points = cifar10.inference(image, num_classes=5, is_training=False)

        logit = tf.nn.softmax(logits)




        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            cifar10.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        while True:
            evaluate_once(saver, logit, x)
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)



def main(argv=None):  # pylint: disable=unused-argument

  evaluate_one_image()



if __name__ == '__main__':
  tf.app.run()



