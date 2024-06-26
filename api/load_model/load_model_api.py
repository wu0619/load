import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

from keras.models import load_model
from keras.activations import sigmoid
from keras.layers import Conv1D, Layer, GlobalAveragePooling2D, Reshape
from keras.saving import register_keras_serializable

"""
running in tensorflow 2.14
"""
print(tf.__version__)
print(tf.config.list_physical_devices())
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.set_visible_devices(gpus[1], 'GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)


# custom keras layer: ECALayer
@register_keras_serializable('ECALayer')
class ECALayer(Layer):
    def __init__(self, gamma=2, b=1, **kwargs):
        super(ECALayer, self).__init__(**kwargs)
        self.gamma = gamma
        self.b = b

    def build(self, input_shape):
        c = input_shape[-1]
        self.t = max(1, int(abs((tf.math.log(float(c)) / tf.math.log(2.0) + self.b) / self.gamma)))
        self.conv = Conv1D(filters=1, kernel_size=self.t, padding='same', use_bias=False)
        super(ECALayer, self).build(input_shape)

    def call(self, inputs):
        # Global Average Pooling over the spatial dimensions to produce a (batch_size, 1, channels) tensor
        x = GlobalAveragePooling2D()(inputs)
        x = Reshape((1, -1))(x)
        x = self.conv(x)
        x = sigmoid(x)
        x = tf.squeeze(x, axis=1)  # Squeeze to make it (batch_size, channels)

        # Multiply weights across channels
        return inputs * x[:, tf.newaxis, tf.newaxis, :]

    def get_config(self):
        config = super(ECALayer, self).get_config()
        config.update({
            'gamma': self.gamma,
            'b': self.b
        })
        return config

"""
transform an 1d array into (1, self.h, self.m, self.c) 3d matrix
"""
class ImageCoder():
    def __init__(self) -> None:
        self.h = 8
        self.m = 12
        self.c = 3
    
    """
    data: an 1d ndarray
    """
    def encode(self, data: np.ndarray):
        training_sets = []
        for i in data:
            training_reshaped = np.array(i).reshape(1, self.c, self.h, self.m)
            training_sets.append(training_reshaped)
        training_sets = np.transpose(training_reshaped, (0, 2, 3, 1))
        return training_sets
    
    """
    Use predictions from previous steps to add new inputs to rolling predictions
    """
    def image_shift(self, original_input, new_input):
        input = np.transpose(original_input, (0, 3, 1, 2))
        output = []
        output.append(input[0])
        output = np.array(output).flatten()
        output = np.concatenate([output, new_input.flatten()], axis=0)[-len(output):]
        image = output.reshape(1, self.c, self.h, self.m)
        image = np.transpose(image, (0, 2, 3, 1))
        return image

class LoadModel():
    def __init__(self, modelPath, scalerPath) -> None:
        self.scaler = joblib.load(scalerPath)
        self.model = load_model(
            modelPath,
            custom_objects={ "ECALayer": ECALayer },
            compile=False
        )
        self.encoder = ImageCoder()
    
    def predict(self, data):
        input_data_scaled = self.scaler.transform(np.array(data).reshape(-1, 1)).reshape(1, -1)
        input_data = self.encoder.encode(input_data_scaled)
        n_rolling = 48
        final_ouputs = []
        for inputs in input_data: 
            output_roll = []
            new_window_input = np.copy(inputs).reshape(1, *inputs.shape)
            for _ in range(n_rolling):
                y_pred_roll = self.model.predict(new_window_input, verbose=0)
                output_roll.append(y_pred_roll)
                new_window_input = self.encoder.image_shift(new_window_input, y_pred_roll)
            output_roll = np.array(output_roll).flatten()
            final_ouputs.append(output_roll)
        return np.array(self.scaler.inverse_transform(final_ouputs))
        
