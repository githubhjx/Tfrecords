from skimage import io, transform
from tqdm import tqdm
from time import sleep
import glob
import os
import gc
import numpy as np
import tensorflow as tf


# decision  parameter
N_LABEL = 1  # Number of classes
N_BATCH = 64  # Number of data points per mini-batch

# decision picture parameter
w = 224
h = 224
c = 3


# pic_path = '.../train/'

tr_path = '.../train/'
te_path = '.../test/'

trlb_path = '.../train/label.txt'
telb_path = '.../test/label.txt'


writer = tf.python_io.TFRecordWriter('/home/s2/data/Pain/leave_one/tfrecord/train.tfrecords')

# writer1 = tf.python_io.TFRecordWriter('/home/s2/data/CK_new/tfrecord/train.tfrecords')


def read_label(path):
    r = open(path)
    file = r.readlines()
    records = []
    for i in range(len(file)):
        f = file[i].split()
        ft = list(map(int, f))
        records.append(ft)
    return records


def make_tfrecord(path):
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    cate.sort()
    imgs0 = []
    imgs1 = []
    labels = []

    margin = []

    flag = []  # weakly_supervised flag

    record = read_label(trlb_path)

    for idx, folder in enumerate(cate):

        # L = len([name for name in os.listdir(folder) if os.path.isfile(os.path.join(folder, name))])
        temp = glob.glob(folder + '/*.png')
        temp.sort()

        id_temp = []
        # data = []
        # label = []

        for st, ed in zip(range(0, len(record[idx]), 5), range(5, len(record[idx])+5, 5)):
            if len(record[idx][st:ed]) >= 5:
                if record[idx][st:ed] == [record[idx][st]] * 5:
                    id_temp.append(st)
                else:
                    id_temp.extend(list(range(st, ed)))
            else:
                id_temp.extend(list(range(st, len(record[idx]))))

        L = len(id_temp)

        data = []
        data0_0 = []
        data1_1 = []
        label = []

        primary_label = []
        margin_temp = []
        flag_temp = []

        for im in id_temp:
            print('reading the images:%s' % (temp[im]))
            img = io.imread(temp[im])
            img = transform.resize(img, (w, h))
            data.append(img)
            primary_label.append([record[idx][im]])

        temp_label = [0] * L
        temp_label[0] = 1
        temp_label[-1] = 1

        # rank data
        label0 = [0]  # negative label
        label1 = [1]  # positive label
        for i in range(len(data)):
            for j in range(len(data)):
                if j > i:
                    data0_0.append(data[i])  # negative sample
                    data1_1.append(data[j])
                    # data0_0.append(data[j])
                    # data1_1.append(data[i])  # positive sample
                    label.append(primary_label[j])
                    # label.append(label0)
                    margin_temp.append(np.abs(i - j))
                    flag_temp.append(temp_label[j])

        data0_0.insert(0, data[0])
        data1_1.insert(0, data[0])
        label.insert(0, primary_label[0])
        margin_temp.insert(0, 0)
        flag_temp.insert(0, 1)

        # push data to img and label
        imgs0.extend(data0_0)
        imgs1.extend(data1_1)
        labels.extend(label)
        margin.extend(margin_temp)
        flag.extend(flag_temp)

    # print('Writing to tfrecord file ...')

    for ig in tqdm(range(len(imgs0))):

        X0 = imgs0[ig].tostring()
        X1 = imgs1[ig].tostring()
        Y = labels[ig][0]
        Mg = margin[ig]
        Fg = flag[ig]
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[Y])),
            'X0': tf.train.Feature(bytes_list=tf.train.BytesList(value=[X0])),
            'X1': tf.train.Feature(bytes_list=tf.train.BytesList(value=[X1])),
            'M': tf.train.Feature(int64_list=tf.train.Int64List(value=[Mg])),
            'F': tf.train.Feature(int64_list=tf.train.Int64List(value=[Fg])),
        }))
        writer.write(example.SerializeToString())
        # pass
    writer.close()

    for x in locals().keys():
        del locals()[x]
    gc.collect()

    # print('Process successfully!')


def make_prim_tfrecord(path):
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    cate.sort()
    imgs0 = []
    imgs1 = []
    labels = []

    imgs = []
    primary_label = []

    record = read_label(trlb_path)

    for idx, folder in enumerate(cate):

        # L = len([name for name in os.listdir(folder) if os.path.isfile(os.path.join(folder, name))])
        temp = glob.glob(folder + '/*.png')
        temp.sort()

        id_temp = []
        temp_labels = []
        # data = []
        # label = []

        for st, ed in zip(range(0, len(record[idx]), 5), range(5, len(record[idx])+5, 5)):
            if len(record[idx][st:ed]) >= 5:
                if record[idx][st:ed] == [record[idx][st]] * 5:
                    id_temp.append(st)
                else:
                    id_temp.extend(list(range(st, ed)))
            else:
                id_temp.extend(list(range(st, len(record[idx]))))

        L = len(id_temp)

        data = []
        data0_0 = []
        data1_1 = []
        label = []
        for im in id_temp:
            print('reading the images:%s' % (temp[im]))
            img = io.imread(temp[im])
            img = transform.resize(img, (w, h))
            data.append(img)
            temp_labels.append([record[idx][im]])

        imgs.extend(data)
        primary_label.extend(temp_labels)

    for ig in range(len(imgs)):
        X = imgs[ig].tostring()
        Y = primary_label[ig][0]
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[Y])),
            'X': tf.train.Feature(bytes_list=tf.train.BytesList(value=[X]))
        }))
        writer.write(example.SerializeToString())
    writer.close()

    for x in locals().keys():
        del locals()[x]
    gc.collect()


def make_ck_tfrecord(path):
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    cate.sort()
    imgs0 = []
    imgs1 = []
    labels = []
    margin = []
    slabels = []
    for idx, folder in enumerate(cate):

        L = len([name for name in os.listdir(folder) if os.path.isfile(os.path.join(folder, name))])
        temp = glob.glob(folder + '/*.png')
        temp.sort()

        for b in range(0, 3):
            data = []
            data0_0 = []
            data1_1 = []
            label = []
            data_label = []
            for im in range(b, L, 3):
                print('reading the images:%s' % (temp[im]))
                img = io.imread(temp[im])
                img = transform.resize(img, (w, h))
                data.append(img)
                # data_label.append(temp_label[im])

            # rank data
            label0 = [0]  # negative label
            label1 = [1]  # positive label
            for i in range(len(data)):
                for j in range(len(data)):
                    if j > i:
                        data0_0.append(data[i])  # negative sample
                        data1_1.append(data[j])
                        data0_0.append(data[j])
                        data1_1.append(data[i])  # positive sample
                        label.append(label1)
                        label.append(label0)

            # push data to img and label
            imgs0.extend(data0_0)
            imgs1.extend(data1_1)
            labels.extend(label)

    for ig in range(len(imgs0)):
        X0 = imgs0[ig].tostring()
        X1 = imgs1[ig].tostring()
        Y = labels[ig][0]
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[Y])),
            'X0': tf.train.Feature(bytes_list=tf.train.BytesList(value=[X0])),
            'X1': tf.train.Feature(bytes_list=tf.train.BytesList(value=[X1]))
        }))
        writer1.write(example.SerializeToString())
    writer1.close()

    for x in locals().keys():
        del locals()[x]
    gc.collect()


# make_prim_tfrecord(tr_path)

# make_ck_tfrecord(pic_path)
make_tfrecord(tr_path)
print('read the image success!')
