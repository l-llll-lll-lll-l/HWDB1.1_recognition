import os
import codecs
import random
import argparse
import numpy as np
import tensorflow as tf
from imageio.v2 import imread
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from the_model import model_8
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from tqdm import tqdm  

# 定义路径
WEIGHTS_PATH = os.path.join("src", "weights08.h5")
LABELS_PATH = os.path.join("src", "labels.txt")

IMG_SIZE = 96

num_classes = 3755

def split_list(data, num_chunks):
    """将列表分割为num_chunks个子列表"""
    avg = len(data) / float(num_chunks)
    chunks = []
    last = 0.0

    while last < len(data):
        chunks.append(data[int(last):int(last + avg)])
        last += avg

    return chunks

def load_image(file_info, PATH):
    k, v, png = file_info
    with open(os.path.join(PATH, v, png), 'rb') as fimg:
        img_data = resize_and_pad_channel(imread(fimg))
    return img_data, k

def process_images(image_list, start_idx, PATH):
    """处理一个图像列表，返回图像数据、标签和全局索引"""
    results = []
    for idx, file_info in enumerate(image_list):
        img_data, k = load_image(file_info, PATH)
        results.append((img_data, k, start_idx + idx))
    return results

def resize_and_pad_channel(image, target_size=(96, 96), channel_index=0):
    # 提取指定的通道
    channel = image[:, :, channel_index]

    current_height, current_width = channel.shape
    target_height, target_width = target_size

    # 初始化处理后的图像
    processed_image = channel

    # 处理高度
    if current_height > target_height:
        # 裁剪
        start_height = (current_height - target_height) // 2
        processed_image = processed_image[start_height:start_height + target_height, :]
    elif current_height < target_height:
        # 填充
        pad_height = (target_height - current_height) // 2
        processed_image = np.pad(processed_image,
                                 pad_width=((pad_height, target_height - current_height - pad_height),
                                            (0, 0)),
                                 mode='constant',
                                 constant_values=0)

    # 处理宽度
    current_height, current_width = processed_image.shape
    if current_width > target_width:
        # 裁剪
        start_width = (current_width - target_width) // 2
        processed_image = processed_image[:, start_width:start_width + target_width]
    elif current_width < target_width:
        # 填充
        pad_width = (target_width - current_width) // 2
        processed_image = np.pad(processed_image,
                                 pad_width=((0, 0),
                                            (pad_width, target_width - current_width - pad_width)),
                                 mode='constant',
                                 constant_values=0)
    # 反色处理，假设像素值的范围是0到255
    processed_image = 255 - processed_image
    return processed_image

def save_tfrecords(data, label, desfile):
    with tf.io.TFRecordWriter(desfile) as writer:
        for i in range(len(data)):
            features = tf.train.Features(
                feature={
                    "data": tf.train.Feature(bytes_list=tf.train.BytesList(value=[data[i].astype(np.float32).tobytes()])),
                    "label": tf.train.Feature(float_list=tf.train.FloatList(value=label[i].astype(np.float32)))
                }
            )
            example = tf.train.Example(features=features)
            serialized = example.SerializeToString()
            writer.write(serialized)

def _parse_function(example_proto):
    features = {
        "data": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([3755], tf.float32)  # 标签为one-hot编码
    }
    parsed_features = tf.io.parse_single_example(example_proto, features)
    data = tf.io.decode_raw(parsed_features['data'], tf.float32)
    data = tf.reshape(data, [IMG_SIZE, IMG_SIZE, 1])  # 调整为 (IMG_SIZE, IMG_SIZE, 1) 灰度图形状
    return data, parsed_features["label"]


def load_tfrecords(srcfile):
    dataset = tf.data.TFRecordDataset(srcfile)  
    dataset = dataset.map(_parse_function)  
    return dataset
    