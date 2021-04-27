import os
import tensorflow as tf


reader = tf.TFRecordReader()
_, serialized_example = reader.read("./data1/train.record")
print(_, serialized_example)
