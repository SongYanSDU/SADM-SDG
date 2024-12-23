import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
import numpy as np
from keras.layers import (Input, Conv1D, UpSampling1D, MaxPooling1D, Flatten, Add, Lambda, Layer,
                          BatchNormalization, concatenate, Dense, Dropout, LeakyReLU, Multiply)
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from cov_loss import covariance_loss
import random as python_random
seed = 42
np.random.seed(seed)
python_random.seed(seed)
tf.random.set_seed(seed)


class SpectralAttention(Layer):
    def __init__(self, in_channels, **kwargs):
        super(SpectralAttention, self).__init__(**kwargs)
        self.in_channels = in_channels

    def build(self, input_shape):
        # 创建参数，这里使用Dense层的方式
        self.dense1 = Dense(self.in_channels // 16, use_bias=False)
        self.dense2 = Dense(self.in_channels, use_bias=False)
        self.relu = tf.keras.layers.ReLU()
        self.sigmoid = tf.keras.layers.Activation('sigmoid')
        super(SpectralAttention, self).build(input_shape)

    def call(self, x):
        # 对输入进行傅里叶变换
        freq_x = tf.signal.fft(tf.cast(x, tf.complex64))  # (N, T, C)
        freq_x = tf.abs(freq_x) ** 2 #    能量 (N, T, C)
        # 沿时间维度求平均
        freq_x_mean = tf.reduce_mean(freq_x, axis=1, keepdims=True)  # (N, 1, C)
        # 通道注意力权重计算
        y = tf.squeeze(freq_x_mean, axis=1)      # (N, C)
        y = self.dense1(y)                       # (N, C/4)
        y = self.relu(y)
        y = self.dense2(y)                       # (N, C)
        y = self.sigmoid(y)                      # (N, C)
        y = tf.expand_dims(y, axis=1)            # (N, 1, C)
        y = tf.tile(y, [1, tf.shape(x)[1], 1])   # (N, T, C)
        opt = x * y
        return opt

    def compute_output_shape(self, input_shape):
        return input_shape


def UNet1D(input_shape):
    inputs = Input(shape=input_shape)
    # Encoder
    c1 = Conv1D(16, 33, activation='relu', padding='same')(inputs)
    sa1 = SpectralAttention(in_channels=16)(c1)  # 使用自定义层
    p1 = MaxPooling1D(32)(sa1)

    c2 = Conv1D(32, 9, activation='relu', padding='same', name='opt1')(p1)
    sa2 = SpectralAttention(in_channels=32)(c2)  # 使用自定义层
    p2 = MaxPooling1D(8)(sa2)

    c3 = Conv1D(64, 3, activation='relu', padding='same')(p2)

    u2 = UpSampling1D(8)(c3)
    concat2 = concatenate([u2, sa2])
    c6 = Conv1D(32, 9, activation='relu', padding='same', name='opt2')(concat2)

    u3 = UpSampling1D(32)(c6)
    concat3 = concatenate([u3, sa1])
    c7 = Conv1D(16, 33, activation='relu', padding='same')(concat3)

    # Output Layer
    outputs = Conv1D(1, 1, activation='linear', padding='same')(c7)
    model = Model(inputs, outputs)
    return model


# Define the diffusion model as a standard Keras model without condition input
class DiffusionModel:
    def __init__(self, signal_length=4096, T=1000):
        self.signal_length = signal_length
        self.T = T
        # self.unet = UNet1D(input_shape=(signal_length, 1))
        self.unet = UNet1D(input_shape=(signal_length, 1))
        self.betas = np.linspace(1e-4, 0.02, T)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)
        self.loss_fn = MeanSquaredError()
        self.optimizer = Adam(learning_rate=1e-4)

    def compile(self, optimizer=None, loss=None):
        if optimizer is not None:
            self.optimizer = optimizer
        if loss is not None:
            self.loss_fn = loss
        self.unet.compile(optimizer=self.optimizer, loss=self.loss_fn)

    def generate_sample(self, num_samples=1, num_steps=50):
        # num_samples: 要生成的样本数量
        # num_steps: 反向扩散的步骤数量
        x = np.random.normal(size=(num_samples, self.signal_length, 1))  # 初始化输入噪声数据，长度为信号长度4096
        for t in reversed(range(1, num_steps)):
            alpha_t = self.alphas[t]
            alpha_hat_t = self.alphas_cumprod[t]
            beta_t = self.betas[t]
            predicted_noise = self.unet.predict(x, verbose=0)  # 使用UNet模型预测生成的噪声
            if t > 1:
                noise = np.random.normal(size=x.shape)  # 生成随机噪声
            else:
                noise = np.zeros_like(x)  # 最后一步不需要噪声
            x = 1 / np.sqrt(alpha_t) * (x - (1 - alpha_t) / np.sqrt(1 - alpha_hat_t) * predicted_noise) + np.sqrt(
                beta_t) * noise
        x = np.clip((x + 1) / 2, 0, 1)  # 将像素值缩放到 [0, 1] 范围内
        return x

    def add_noise(self, x_0, t):
        noise = np.random.normal(size=x_0.shape)
        alpha_t = self.alphas_cumprod[t]
        x_t = np.sqrt(alpha_t) * x_0 + np.sqrt(1 - alpha_t) * noise
        return x_t, noise

    def train_on_batch(self, x_0, reference_features=None):
        t = np.random.randint(0, self.T)
        x_t, noise = self.add_noise(x_0, t)
        # Train the model with generated noise and reference-guided feature loss
        loss = self.unet.train_on_batch(x_t, noise)

        '''if reference_features is not None:
            # Extract encoder features for reference-guided loss
            gen_features_1, gen_features_2 = self.extract_encoder_features(x_0)
            ref_features = reference_features  # Assuming reference features are provided
            # Compute feature embedding loss
            loss += covariance_loss(gen_features_1, ref_features)
            loss += covariance_loss(gen_features_2, ref_features)
            loss += covariance_loss(gen_features_2, gen_features_1)'''
        return loss

    def extract_encoder_features(self, x):
        # Extract encoder features from the diffusion model
        # intermediate_model_1 = Model(inputs=self.unet.input, outputs=self.unet.get_layer('conv1d_3').output)
        intermediate_model_1 = Model(inputs=self.unet.input, outputs=self.unet.get_layer('opt1').output)
        features_1 = intermediate_model_1.predict(x, verbose=0)
        intermediate_model_2 = Model(inputs=self.unet.input, outputs=self.unet.get_layer('opt2').output)
        features_2 = intermediate_model_2.predict(x, verbose=0)
        return features_1, features_2
