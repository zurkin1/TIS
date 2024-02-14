import tensorflow as tf


AUTO = tf.data.experimental.AUTOTUNE # used in tf.data.Dataset API.
CLASSES = [b'0', b'1', b'2', b'3', b'4', b'5', b'6', b'7', b'8', b'9']


def read_image_and_label(img_path):
    bits1 = tf.io.read_file(img_path)
    image1 = tf.image.decode_jpeg(bits1)
    label = tf.strings.split(img_path, sep='\\') # 9.882/u23253438U2OS054O22s1.jpeg
    label = tf.strings.split(label[-2], sep='.')
    return image1, label[0]


def recompress_image(image1, label):
  image1 = tf.cast(image1, tf.uint8)
  image1 = tf.image.encode_jpeg(image1, optimize_size=True, chroma_downsampling=False)
  return image1, label


def _bytestring_feature(list_of_bytestrings):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_of_bytestrings))

def _int_feature(list_of_ints): # int64
  return tf.train.Feature(int64_list=tf.train.Int64List(value=list_of_ints))

def _float_feature(list_of_floats): # float32
  return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))


def to_tfrecord(img_bytes1, label):
  #class_num = np.argmax(np.array(CLASSES)==label)
  class_num = CLASSES.index(label)
  feature = {
      "image1": _bytestring_feature([img_bytes1]), # one image in the list
      "class": _int_feature([class_num]),        # one class in the list
  }
  return tf.train.Example(features=tf.train.Features(feature=feature))


if __name__ == '__main__':
    #tf.enable_eager_execution()
    dataset = tf.data.Dataset.list_files('C:/Temp/' + '*.jpg', seed=10000)  # This also shuffles the images
    dataset = dataset.map(read_image_and_label)
    dataset = dataset.map(recompress_image, num_parallel_calls=AUTO)

    # Split to shards.
    shard_size = 1000 #5000
    print('Num samples per shard', shard_size)
    dataset = dataset.batch(shard_size)

    for shard, (image1, label) in enumerate(dataset):
        # shard_size = image.numpy().shape[0]
        filename = "C:/Temp/test_labels_" + "{:02d}.tfrec".format(shard)

        with tf.io.TFRecordWriter(filename) as out_file:
            for i in range(shard_size-10):
                example = to_tfrecord(image1.numpy()[i], label.numpy()[i])
                out_file.write(example.SerializeToString())
            print("Wrote file {} containing {} records".format(filename, shard_size))