filename_queue = tf.train.string_input_producer([path], num_epochs=10000)
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(serialized_example,
                                    features={
                                        'label': tf.FixedLenFeature([], tf.int64),
                                        'X0': tf.FixedLenFeature([], tf.string),
                                        'X1': tf.FixedLenFeature([], tf.string),
                                        'M': tf.FixedLenFeature([], tf.int64),
                                        'F': tf.FixedLenFeature([], tf.int64),
                                    })
img0 = tf.reshape(tf.decode_raw(features['X0'], tf.float64), [224, 224, 3])
img1 = tf.reshape(tf.decode_raw(features['X1'], tf.float64), [224, 224, 3])
# img = tf.reshape(img, [224, 224, 3])
# img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
label = tf.cast(features['label'], tf.int64)
margin = tf.cast(features['M'], tf.int64)
flag = tf.cast(features['F'], tf.int64)

img0_batch, img1_batch, label_batch, margin_batch, flag_batch = tf.train.shuffle_batch([img0, img1, [label], margin, flag], batch_size=32, num_threads=4, capacity=1280, min_after_dequeue=10, allow_smaller_final_batch=True)

''''''
''''''
''''''

 trX0, trX1, trY, Margin, Flag = sess.run([img0_batch, img1_batch, label_batch, margin_batch, flag_batch])
