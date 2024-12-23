import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
import numpy as np
from keras.layers import (Input, Conv1D, UpSampling1D, MaxPooling1D, Flatten, Add, Lambda,
                          BatchNormalization, concatenate, Dense, Dropout, LeakyReLU, Multiply, ReLU)
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import MeanSquaredError, SparseCategoricalCrossentropy
from data_loader import bjut_dataset
from sklearn.preprocessing import StandardScaler
from keras import backend as K
np.random.seed(1234)
tf.random.set_seed(1234)
from diffusion_model_pp import DiffusionModel
from FreqFusion import freq_fusion, FreqFusion0
import os
from scipy.io import savemat
import random as python_random
seed = 42
np.random.seed(seed)
python_random.seed(seed)
tf.random.set_seed(seed)
from tensorflow.keras.callbacks import LearningRateScheduler


def covariance_loss(x1, x2):
    # Perform covariance loss computation
    x1_centered = x1 - tf.reduce_mean(x1, axis=0)
    x2_centered = x2 - tf.reduce_mean(x2, axis=0)
    cov_matrix = tf.matmul(x1_centered, x2_centered, transpose_a=True)/ tf.cast(tf.shape(x1)[0], tf.float32)
    loss = tf.norm(cov_matrix, ord='fro', axis=[-2, -1])
    return loss

def cov_loss(x1, x2, x3, x12, x13, x23):
    loss = (covariance_loss(x1, x12) + covariance_loss(x1, x13) +
            covariance_loss(x2, x12) + covariance_loss(x2, x23) +
            covariance_loss(x3, x13) + covariance_loss(x3, x23))
    # covariance_loss(x12, x13) + covariance_loss(x12, x23) + covariance_loss(x13, x23))
    # covariance_loss(x1, x2) + covariance_loss(x1, x3) + covariance_loss(x2, x3) +
    # 计算平均损失
    total_loss  = tf.reduce_mean(loss)
    return total_loss

class FaultDiagnosisModel:
    def __init__(self, signal_length=4096, num_classes=5):
        self.signal_length = signal_length
        self.num_classes = num_classes
        self.conv1 = Conv1D(16, 33, activation='relu', padding='same')
        self.pool1 = MaxPooling1D(32)
        self.conv2 = Conv1D(32, 9, activation='relu', padding='same', name='x1_conv2')
        self.pool2 = MaxPooling1D(8)
        self.conv3_1 = Conv1D(64, 3, activation='relu', padding='same')
        self.conv3_2 = Conv1D(64, 3, activation='relu', padding='same')
        self.conv3_3 = Conv1D(64, 3, activation='relu', padding='same')
        self.pool3 = MaxPooling1D(2)

        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = Dense(128, name='fc1')
        self.dropout = Dropout(0.5)
        self.classifier = Dense(num_classes, activation='softmax')
        self.loss_fn = SparseCategoricalCrossentropy()
        self.optimizer = Adam(learning_rate=1e-4)
        self.model = self.build_model()

    def build_model(self):
        inputs = Input(shape=(self.signal_length, 1))
        x1 = self.conv1(inputs)
        x1 = self.pool1(x1)
        x1 = self.conv2(x1)
        x1 = self.pool2(x1)
        x1 = self.conv3_1(x1)
        x1 = self.pool3(x1)

        # Adaptive Feature Fusion with diffusion features
        diffusion_input_1 = Input(shape=(128, 32))
        x2 = diffusion_input_1
        x2 = self.pool2(x2)
        x2 = self.conv3_2(x2)
        x2 = self.pool3(x2)

        diffusion_input_2 = Input(shape=(128, 32))
        x3 = diffusion_input_2
        x3 = self.pool2(x3)
        x3 = self.conv3_3(x3)
        x3 = self.pool3(x3)

        # Frequency Fusion Layer
        x12 = FreqFusion0()([x1, x2])
        x13 = FreqFusion0()([x1, x3])
        x23 = FreqFusion0()([x2, x3])

        # Concatenate all features
        x = concatenate([x1, x2, x3, x12, x13, x23], axis=1, name='concat_features') # , x12, x13, x23

        # Fully connected layers
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)

        # Final classifier output
        outputs = self.classifier(x)
        covloss = cov_loss(x1, x2, x3, x12, x13, x23)
        # Define the model
        model = Model(inputs=[inputs, diffusion_input_1, diffusion_input_2], outputs=outputs)
        model.add_loss(0.5*covloss)
        return model

    def extract_encoder_features(self, x):
        # Access intermediate layers to extract features
        intermediate_model = Model(inputs=self.model.input, outputs=self.model.get_layer('x1_conv2').output)
        features = intermediate_model.predict(x, verbose=0)
        return features

    def extract_features_conca_fc(self, x):
        # Access intermediate layers to extract features
        intermediate_model_1 = Model(inputs=self.model.input, outputs=self.model.get_layer('concat_features').output)
        features_1 = intermediate_model_1.predict(x, verbose=0)
        intermediate_model_2 = Model(inputs=self.model.input, outputs=self.model.get_layer('fc1').output)
        features_2 = intermediate_model_2.predict(x, verbose=0)
        return features_1, features_2


# 加载数据 W20 W25 W30 W35 W40 W45 W50  W55
(data1, label1, data2, label2, data3, label3, data4, label4, data5, label5,
            data6, label6, data7, label7, data8, label8) = bjut_dataset()

train_data = data1
train_label = label1
test_data = data3
test_label = label3
# 数据标准化
scaler = StandardScaler()
x_train = scaler.fit_transform(train_data.reshape(-1, 4096)).reshape(-1, 4096, 1)
x_test = scaler.fit_transform(test_data.reshape(-1, 4096)).reshape(-1, 4096, 1)
# 实例化扩散模型并进行编译
diffusion_model = DiffusionModel(signal_length=4096, T=1000)
diffusion_model.compile(optimizer=Adam(learning_rate=3e-4), loss=MeanSquaredError())
# 实例化并编译故障诊断模型
fault_model_instance = FaultDiagnosisModel()
fault_diagnosis_model = fault_model_instance.build_model()
fault_diagnosis_model.compile(optimizer=Adam(learning_rate=3e-4), loss=SparseCategoricalCrossentropy(),
                              metrics=['accuracy'])

# Train fault diagnosis model with adaptive feature fusion
num_samples_0 = len(x_train)
for epoch in range(150):
    for _ in range(int(np.ceil(num_samples_0 / 256))):
        indices_0 = np.random.randint(0, num_samples_0, size=256)
        xx = x_train[indices_0]
        diffusion_model.train_on_batch(xx)
    if (epoch + 1) % 20 == 0:
        old_lr = K.get_value(diffusion_model.optimizer.learning_rate)
        new_lr = old_lr * 0.8
        K.set_value(diffusion_model.optimizer.learning_rate, new_lr)

for epoch in range(150):
    for _ in range(int(np.ceil(num_samples_0 / 256))):
        indices_0 = np.random.randint(0, num_samples_0, size=256)
        x = x_train[indices_0]
        y = train_label[indices_0]
        fea1, fea2 = diffusion_model.extract_encoder_features(x)
        fault_diagnosis_model.train_on_batch([x, fea1, fea2], y)

    if (epoch + 1) % 20 == 0:
        old_lr = K.get_value(fault_diagnosis_model.optimizer.learning_rate)
        new_lr = old_lr * 0.8
        K.set_value(fault_diagnosis_model.optimizer.learning_rate, new_lr)

fea1, fea2 = diffusion_model.extract_encoder_features(x_test)
eval = fault_diagnosis_model.evaluate([x_test, fea1, fea2], test_label)
print("Test loss: {:.4f}, Test Accuracy: {:.4f}".format(eval[0], eval[1]))
print(f"Finished Training and Testing Pair {1}\n")
# 保存扩散模型和故障诊断模型
path = '.\\pp\\'
experiment_id = f"case{1}"
fea1, fea2 = diffusion_model.extract_encoder_features(x_test)
features_con, features_fc = fault_model_instance.extract_features_conca_fc([x_test, fea1, fea2])
# 1. 获取预测结果
predictions = fault_diagnosis_model.predict([x_test, fea1, fea2])  # 获取分类输出结果
predicted_labels = predictions.argmax(axis=1)  # 预测类别标签
# 保存数据到.mat文件
filename = f"{path}exp{experiment_id}.mat"
savemat(filename, {
    'predictions': predicted_labels,  # 预测标签
    'true_labels': test_label,  # 真实标签
    'fc1_features': features_fc,  # fc1层特征
    'concatenated_features': features_con  # 拼接特征
})
print(f"Saved features and labels to {filename}")
# 清除模型和内存
del fault_model_instance
del diffusion_model
del fault_diagnosis_model
K.clear_session()
# Optional: 如果使用GPU，建议强制释放显存，确保没有残留
try:
    import gc
    gc.collect()  # 强制进行垃圾回收
except ImportError:
    pass