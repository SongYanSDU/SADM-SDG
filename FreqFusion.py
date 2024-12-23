import numpy as np
import tensorflow as tf
import random as python_random

seed = 42
np.random.seed(seed)
python_random.seed(seed)
tf.random.set_seed(seed)


class freq_fusion(tf.keras.layers.Layer):
    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.p = p
        self.alpha = alpha
        self.eps = eps

    def mixup(self, x, y):
        C = tf.shape(x)[-1]
        # Use TensorFlow operations to generate beta-distributed random numbers
        lmda1 = tf.random.uniform(shape=(1, C), minval=0.2, maxval=0.8, dtype=x.dtype)
        lmda1 = tf.math.pow(lmda1, 1.0 / self.alpha)  # Adjust this formula as needed for your beta distribution
        # lmda2 = tf.random.uniform(shape=(1, C), minval=0.6, maxval=1, dtype=x.dtype)
        # lmda2 = tf.math.pow(lmda2, 1.0 / self.alpha)
        # tf.print(x.shape)
        # 傅里叶变换
        x_fft = tf.signal.fft(tf.cast(x, tf.complex64))
        y_fft = tf.signal.fft(tf.cast(y, tf.complex64))
        # 提取实部和虚部
        x_real, x_imag = tf.math.real(x_fft), tf.math.imag(x_fft)
        y_real, y_imag = tf.math.real(y_fft), tf.math.imag(y_fft)

        real_mixed = x_real * lmda1 + y_real * (1 - lmda1)
        imag_mixed = x_imag * lmda1 + y_imag * (1 - lmda1)
        # 重新组合实部和虚部，进行傅里叶反变换
        x_fft_mix = tf.complex(real_mixed, imag_mixed)
        x_ifft = tf.signal.ifft(x_fft_mix)
        # xx = tf.abs(x_ifft)
        xx = tf.cast(tf.math.real(x_ifft), tf.float32)
        return xx

    def call(self, inputs):
        x, y = inputs
        mixed_x = self.mixup(x, y)
        # mixed_y = self.mixup(y, x)
        return mixed_x

    def get_config(self):
        config = super(freq_fusion, self).get_config()
        config.update({
            'p': self.p,
            'alpha': self.alpha,
            'eps': self.eps
        })
        return config


class FreqFusion(tf.keras.layers.Layer):
    def __init__(self, alpha=0.1, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.eps = eps

    def build(self, input_shape):
        # Define learnable parameters for real and imaginary parts fusion
        B, L, C = input_shape[0]  # Batch size, length, channel
        # Initialize lmda1 and lmda2 as learnable parameters, initialized to values in (0, 1)
        self.lmda1 = self.add_weight(shape=(1, C),  # You can adjust the shape to control flexibility
            initializer=tf.keras.initializers.RandomUniform(minval=0, maxval=1),
            trainable=True,
            name="lmda1")
        self.lmda2 = self.add_weight(shape=(1, C),  # You can adjust the shape to control flexibility
            initializer=tf.keras.initializers.RandomUniform(minval=0, maxval=1),
            trainable=True,
            name="lmda1")

    def mixup(self, x, y):
        # Fourier transform
        x_fft = tf.signal.fft(tf.cast(x, tf.complex64))
        y_fft = tf.signal.fft(tf.cast(y, tf.complex64))
        # Extract real and imaginary parts
        x_real, x_imag = tf.math.real(x_fft), tf.math.imag(x_fft)
        y_real, y_imag = tf.math.real(y_fft), tf.math.imag(y_fft)
        # Use trainable parameters lmda1 and lmda2 for real and imaginary parts fusion
        lmda1 = self.lmda1
        lmda2 = self.lmda2
        real_mixed = x_real * lmda1 + y_real * (1 - lmda1)
        imag_mixed = x_imag * lmda2 + y_imag * (1 - lmda2)
        # Re-combine real and imaginary parts and perform inverse Fourier transform
        x_fft_mix = tf.complex(real_mixed, imag_mixed)
        x_ifft = tf.signal.ifft(x_fft_mix)
        xx = tf.abs(x_ifft)
        # xx = tf.cast(tf.math.real(x_ifft), tf.float32)
        return xx

    def call(self, inputs):
        x, y = inputs
        mixed_x = self.mixup(x, y)
        return mixed_x

    def get_config(self):
        config = super(FreqFusion, self).get_config()
        config.update({
            'alpha': self.alpha,
            'eps': self.eps
        })
        return config

class FreqFusion0(tf.keras.layers.Layer):
    def __init__(self, alpha=0.1, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.eps = eps

    def build(self, input_shape):
        # Define learnable parameters for real and imaginary parts fusion
        B, L, C = input_shape[0]  # Batch size, length, channel
        # Initialize lmda1 and lmda2 as learnable parameters, initialized to values in (0, 1)
        self.lmda1 = self.add_weight(shape=(1, C),  # You can adjust the shape to control flexibility
                                     initializer=tf.keras.initializers.RandomUniform(minval=0, maxval=1),
                                     trainable=True,
                                     name="lmda1")
        self.lmda2 = self.add_weight(shape=(1, C),  # You can adjust the shape to control flexibility
                                     initializer=tf.keras.initializers.RandomUniform(minval=0, maxval=1),
                                     trainable=True,
                                     name="lmda2")

    def mixup(self, x, y):
        # Fourier transform
        x_fft = tf.signal.fft(tf.cast(x, tf.complex64))
        y_fft = tf.signal.fft(tf.cast(y, tf.complex64))
        # Extract real and imaginary parts
        x_real, x_imag = tf.math.real(x_fft), tf.math.imag(x_fft)
        y_real, y_imag = tf.math.real(y_fft), tf.math.imag(y_fft)
        # Use trainable parameters lmda1 and lmda2 for real and imaginary parts fusion
        lmda1 = self.lmda1
        lmda2 = self.lmda2
        real_mixed = x_real * lmda1 + y_real * (1 - lmda1)
        imag_mixed = x_imag * lmda2 + y_imag * (1 - lmda2)
        # Re-combine real and imaginary parts and perform inverse Fourier transform
        x_fft_mix = tf.complex(real_mixed, imag_mixed)
        x_ifft = tf.signal.ifft(x_fft_mix)
        xx = tf.abs(x_ifft)
        # xx = tf.cast(tf.math.real(x_ifft), tf.float32)
        return xx

    def call(self, inputs):
        x, y = inputs
        mixed_x = self.mixup(x, y)
        return mixed_x

    def get_config(self):
        config = super(FreqFusion0, self).get_config()
        config.update({
            'alpha': self.alpha,
            'eps': self.eps
        })
        return config