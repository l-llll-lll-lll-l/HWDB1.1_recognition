import tensorflow as tf
from utils import *
from the_model import model_8
import argparse

# 设置一个启动该参数，表示要验证的模型的权重文件名称
argparser = argparse.ArgumentParser()
argparser.add_argument("--model", help="The name of the model weights file to verify", required=True)

# 从src中获得同名文件完整路径
WEIGHTS_PATH = os.path.join("src", argparser.parse_args().model)

random.seed(888)
np.random.seed(888)
tf.random.set_seed(888)

# 加载类别标签
with codecs.open(LABELS_PATH, "r", "UTF-8") as label_file:
    klasses = [line.strip() for line in label_file.readlines()]

# 加载模型和权重
model = model_8(IMG_SIZE, len(klasses))
model.load_weights(WEIGHTS_PATH)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
test_dataset = load_tfrecords('test.tfrecords') 
test_dataset = test_dataset.batch(64).prefetch(buffer_size=tf.data.AUTOTUNE)

# 使用test_dataset进行验证
results = model.evaluate(test_dataset)
print(f"Test Loss: {results[0]:.4f}")
print(f"Test Accuracy: {results[1]:.4%}")

# 保存模型的评估结果
filename = f"evaluate_results_{argparser.parse_args().model}.txt"
with open(os.path.join("report", filename), "w") as f:
    f.write(f"Test Loss: {results[0]:.4f}\n")
    f.write(f"Test Accuracy: {results[1]:.4%}\n")