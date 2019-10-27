from typing import NamedTuple, Dict, List

import tensorflow as tf
from keras import backend as K
from tensorflow.python.keras import Model, layers

SubModelDict = Dict[str, Model]


class ModularModels(NamedTuple):
    """Contains references to the sub models used in the online, target and exploration networks"""
    online: SubModelDict
    target: SubModelDict
    explore: SubModelDict


class ModularHeadModels(NamedTuple):
    """Contains references to the sub models used in the online and target networks"""
    online: SubModelDict
    target: SubModelDict


class SubModelWeightReferences(NamedTuple):
    """Contains references to the parameter indices and shapes of sub-models by name, used for changing specific
    parameters"""
    model_part: str
    var_indices: List[int]
    var_shapes: List[tuple]


class LayerNorm(layers.Layer):
    """ Layer Normalization in the style of https://arxiv.org/abs/1607.06450 """

    def __init__(self, initial_bias=0.1, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = 1e-6
        self.scale = None
        self.bias = None
        self.initial_bias = initial_bias

    def build(self, input_shape):
        self.scale = self.add_weight(shape=(int(input_shape[-1]),),
                                     trainable=True,
                                     initializer=tf.ones_initializer,
                                     name='{}_scale'.format(self.name))
        self.bias = self.add_weight(shape=(int(input_shape[-1]),),
                                    trainable=True,
                                    initializer=tf.constant_initializer(value=self.initial_bias),
                                    name='{}_bias'.format(self.name))
        self.built = True

    def call(self, x, mask=None):
        mean = K.mean(x, axis=-1, keepdims=True)
        variance = K.mean(K.square(x - mean), axis=-1, keepdims=True)
        std = K.sqrt(variance + self.epsilon)
        x = (x - mean) / std
        return x * self.scale + self.bias
        # return x

    def compute_output_shape(self, input_shape):
        return input_shape
