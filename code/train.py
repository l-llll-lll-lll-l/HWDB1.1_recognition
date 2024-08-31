import tensorflow as tf
from utils import *
from the_model import model_8

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

print("开始训练...")
# 除最后四层外，其他层设置为不可训练
for layer in model.layers[:-4]:  
    layer.trainable = False

for layer in model.layers[-4:]:  
    layer.trainable = True
    
# 重新编译模型，使用较小的学习率进行微调
model.compile(optimizer=Adam(learning_rate=5e-5),  
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# 微调模型
train_dataset = load_tfrecords('train.tfrecords')
test_dataset = load_tfrecords('test.tfrecords')
train_dataset = train_dataset.shuffle(buffer_size=50000).batch(64).prefetch(buffer_size=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(64).prefetch(buffer_size=tf.data.AUTOTUNE)
history = model.fit(train_dataset, validation_data=test_dataset, epochs=5)

# 保存微调后的模型
weights_path = '.\\new_model.weights.h5'
model.save_weights(weights_path)
