import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 
import os
from PIL import Image
import random

objects = ['cat','dog']#'cat'0,'dog'1

filename_train="./data/train.tfrecords"
writer_train= tf.python_io.TFRecordWriter(filename_train)

tf.app.flags.DEFINE_string(
    'data', 'None', 'where the datas?.')
FLAGS = tf.app.flags.FLAGS

if(FLAGS.data == None):
    os._exit(0)

dim = (224,224)
object_path = FLAGS.data
total = os.listdir(object_path)
for index in total:
    img_path=os.path.join(object_path,index)
    img=Image.open(img_path)
    img=img.resize(dim)
    img_raw=img.tobytes()
    for i in range(len(objects)):
        if objects[i] in index:
            value = i
        else:
            continue
    example = tf.train.Example(features=tf.train.Features(feature={
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[value])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                }))
    print([index,value])
    writer_train.write(example.SerializeToString())  #序列化为字符串
writer_train.close()
