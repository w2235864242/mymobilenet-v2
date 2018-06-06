import os
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def creat_tfrecord(file_path):
    cwd = file_path
    classes = {'outdoor', 'guodu', 'indoor'}
    #writer = tf.python_io.TFRecordWriter("imagetrain.tfrecords")
    writer = tf.python_io.TFRecordWriter("imagetest.tfrecords")

    for index, name in enumerate(classes):
        class_path = cwd + name + '/'
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name

            img = Image.open(img_path)
            img = img.resize((224, 224))
            print img.size
            print img.mode
            img_raw = img.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
            writer.write(example.SerializeToString())

    writer.close()


def read_and_decode_image(filename):
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })

    image = tf.decode_raw(features['img_raw'], tf.uint8)
    image = tf.reshape(image, [299, 299, 3])
    # image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)

    with tf.Session() as sess:
        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(102):
            example, l = sess.run([image, label])
            img = Image.fromarray(example, 'RGB')
            img.save('/home/ning/new/programs/myprogram/myimageclass/test_data/data/tfrecord_test/' + str(i) + '_''Label_' + str(l) + '.jpg')
            print(example, l)
        coord.request_stop()
        coord.join(threads)


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
    img = tf.reshape(img, [320, 320, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)
    return img, label


if __name__ == '__main__':
    # img, label = read_and_decode("evironment_train.tfrecords")
    # print img.shape
    # print label.shape
    # img, label = read_and_decode("evironment_train.tfrecords")
    # print img.shape
    # print label.shape
    #creat_tfrecord('/home/ning/new/programs/myprogram/mymobilenet-v2/test_data/train/')
    creat_tfrecord('/home/ning/new/programs/myprogram/mymobilenet-v2/test_data/test/')


