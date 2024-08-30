# coding: utf-8
"""
    2013 CASIA 竞赛数据子集测试文件
"""

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
from utils import *

if __name__ == "__main__":
    
    TEST_PATH = r"./test"
    
    TRAIN_PATH = r"./train"
    
    # 获取args
    parser = argparse.ArgumentParser(description="CASIA 2013 数据集测试")
    
    # 一个参数 path 可以取两个值 test 和 train
    parser.add_argument("path", choices=["test", "train"], help="数据集路径")
    
    # 一个参数决定是否全部加载
    parser.add_argument("--all", action="store_true", help="是否全部加载")
    
    # 如果没选择全部加载，读取一个整数作为加载数量
    parser.add_argument("--num", type=int, help="加载数量")
    
    args = parser.parse_args()
    if args.path == "test":
        PATH = TEST_PATH
    else:
        PATH = TRAIN_PATH
    
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    
    print("数据集路径:")
    print(PATH)

    # 设置随机种子以保证可重复性
    random.seed(888)
    np.random.seed(888)
    tf.random.set_seed(888)

    IMG_SIZE = 96

    # 加载类别标签
    with codecs.open(LABELS_PATH, "r", "UTF-8") as label_file:
        klasses = [line.strip() for line in label_file.readlines()]

    label_pngs = []
    for k, v in enumerate(klasses):
        files = os.listdir(os.path.join(PATH, v))
        if len(files) >= 5:
            # 选取的文件
            # 如果 --all 参数为真，则选取全部文件
            if args.all:
                selected_files = files
            elif args.num:
                selected_files = files[:args.num]
            else:
                selected_files = files[:5] 
            for image_filename in selected_files:
                label_pngs.append((k, v, image_filename))
    print("测试样本总数:", len(label_pngs))

    test_data = np.ndarray([len(label_pngs), IMG_SIZE, IMG_SIZE], dtype=np.uint8)
    test_label = np.ndarray([len(label_pngs)], dtype=np.uint32)

    print("加载测试数据...")

    # 获取系统的 CPU 核心数
    num_cores = os.cpu_count()
    print("检测到CPU 核心数:", num_cores)

    # 使用tqdm来显示进度
    progress_bar = tqdm(total=len(label_pngs), desc="加载进度", unit="images")

    try:
        # 将标签列表划分为num_cores-2个子列表
        num_chunks = num_cores - 2
        split_label_pngs = split_list(label_pngs, num_chunks)

        with ProcessPoolExecutor(max_workers=num_chunks) as executor:
            futures = {executor.submit(process_images, chunk, sum(len(c) for c in split_label_pngs[:idx]),PATH): idx for idx, chunk in enumerate(split_label_pngs)}

            for future in as_completed(futures):
                result_list = future.result()
                for img_data, k, global_index in result_list:
                    test_data[global_index] = img_data
                    test_label[global_index] = k
                    progress_bar.update(1)

    except KeyboardInterrupt:
        print("\n检测到键盘中断，正在停止任务...")
    finally:
        progress_bar.close()
        print("已保存当前状态。")

    y_test = to_categorical(test_label)
    print("")
    print("测试数据加载完成！")

    print("测试数据形状:", test_data.shape)
    test_data = test_data.reshape(test_data.shape[0], IMG_SIZE, IMG_SIZE, 1).astype(np.float32)
    test_data /= 255.0
    print("数据处理完成")
    
    # 存储为 TFRecord 文件
    if args.path == "test":
        tf_filename = "test.tfrecords"
    else:
        tf_filename = "train.tfrecords"
    
    save_tfrecords(test_data, y_test, tf_filename)
    
    if args.path == "test":
        total_filename = "total_test_samples.txt"
    else:
        total_filename = "total_train_samples.txt"
    
    # 存储总样本数
    with open(total_filename, "w") as f:
        f.write(str(len(label_pngs)))
        
    print("已保存为 TFRecord 文件:", tf_filename)
    print("已保存总样本数:", total_filename)