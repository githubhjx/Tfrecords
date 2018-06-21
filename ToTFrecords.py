from skimage import io, transform
import glob
import os
import numpy as np
import tensorflow as tf


# decision picture parameter
w = 224
h = 224
c = 3


pic_path = 'This is your image path'

writer = tf.python_io.TFRecordWriter("train.tfrecords")


def read_img(path, writer):
    print('reading the image...')
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    cate.sort()
    # imgs = []
    # labels = []
    for idx, folder in enumerate(cate):
        # label = [0] * 101
        for im in glob.glob(folder + '/*' + '/*.jpg'):
            # print('reading the images:%s' % (im))
            img = io.imread(im)
            img = transform.resize(img, (w, h))
            img_raw = img.tobytes()  # 将图片转化为原生bytes

            # imgs.append(img)
            # label[idx] = 1
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[idx])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
            writer.write(example.SerializeToString())
    writer.close()
 

read_img(pic_path, writer)
print('read the image success!')
