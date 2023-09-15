import numpy as np

import tensorflow as tf
from keras import backend as K
import tensorflow.keras as keras
from typing import List

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from keras.utils.generic_utils import get_custom_objects

class Fetcher:
    def __init__(self, model, preprocess, metric: str, layer: str = 'logit', batchsize: int = 1):
        self.model = model
        input = model.input   
        self.preprocess = preprocess
        self.numLabels = model.layers[-1].output_shape[1]
        self.batchsize = batchsize
       
        label_tensor = tf.placeholder(tf.float32, shape=(None, 1))
        loss = keras.losses.categorical_crossentropy(label_tensor, model.layers[-1].output)
        losses = tf.split(loss, batchsize)

        self.functor1 = K.function([input] + [K.learning_phase()], [layer.output for layer in model.layers])
        self.functor2 = K.function([input, label_tensor] + [K.learning_phase()], losses)

        self.fetchFunction = self.build_fetch_function(gradient=False, layer=layer)
        
    def act_fetch_function(self, input_batches: List, label_batches: List, layer):
        if len(input_batches) == 0:
            return None, None, None, None

        input_batches = self.preprocess(input_batches)

        if layer == 'logit':
            layerIndex = -1
        elif layer == 'penultimate':
            layerIndex = -2
        else:
            layerIndex = -1

        label_batches = np.expand_dims(label_batches, axis=1)
        label_batches = to_categorical(label_batches, self.numLabels)

        initialLen = len(input_batches)
        if initialLen < self.batchsize:
            for i in range(initialLen, self.batchsize):
                input_batches = np.append(input_batches, [input_batches[-1]], axis=0)
                label_batches = np.append(label_batches, [label_batches[-1]], axis=0)

        layer_outputs = self.functor1([input_batches, 0]) # activation
        
        activation = [layer_outputs[layerIndex]]
        if layer == 'all':
            activation = layer_outputs
        metadata = layer_outputs[-1]
        prediction = np.argmax(layer_outputs[-1], axis=1)

        loss = self.functor2([input_batches, label_batches, 0])

        activation = [act[:initialLen] for act in activation]
        metadata = metadata[:initialLen]
        prediction = prediction[:initialLen]
        loss = loss[:initialLen]

        # Return the prediction outputs
        return activation, metadata, prediction, loss

    def build_fetch_function(self, gradient=True, temperature=False, layer='logit'):
        def func(input_batches, label_batches):
            """The fetch function."""
            return self.act_fetch_function(
                    input_batches,
                    label_batches,
                    layer
                )
        return func

