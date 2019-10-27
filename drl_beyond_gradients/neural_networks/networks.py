from typing import NamedTuple, Optional, List, Any

import tensorflow as tf
from keras.activations import softmax
from tensorflow.python.keras import layers, Sequential
from keras import backend as K, initializers
import numpy as np

from drl_beyond_gradients.neural_networks.shared import LayerNorm
from tensorflow.contrib.layers import layer_norm as layer_norm_alternative


class ConvLayerConfig(NamedTuple):
    stride: int  # assumed equal for all filter dimensions
    filter_size: int  # assumed equal for all filter dimensions
    nr_filters: int
    activation: str
    batch_norm: bool


class RGBAtariEncoder(object):

    def __init__(self, image_shape: tuple,
                 **kwargs
                 ):

        conv_layers = [
            ConvLayerConfig(stride=4, filter_size=8, nr_filters=32, activation='relu', batch_norm=False),
            ConvLayerConfig(stride=2, filter_size=4, nr_filters=64, activation='relu', batch_norm=False),
            ConvLayerConfig(stride=1, filter_size=3, nr_filters=64, activation='relu', batch_norm=False),
        ]

        rgb = layers.Input(shape=image_shape, name='rgb_input', dtype=tf.uint8)
        t = layers.Lambda(lambda x: K.cast(x, dtype='float32') / 255.)(rgb)

        for cl in conv_layers:
            t = layers.Conv2D(filters=cl.nr_filters,
                              kernel_size=(cl.filter_size, cl.filter_size),
                              strides=(cl.stride, cl.stride),
                              activation=cl.activation,
                              )(t)

        encoded = layers.Reshape(target_shape=(np.prod(t.shape[1:]),))(t)

        self.model = tf.keras.Model(inputs=[rgb], outputs=[encoded])
        self.embedding_size = 3136

    def __call__(self, image, *args, **kwargs):
        return self.model(inputs=image, *args, **kwargs)


class RGBDecoder(object):

    def __init__(self, image_shape: tuple, embedding_size: int = 16,
                 conv_layers: Optional[List[ConvLayerConfig]] = None,
                 l2_param_penalty: float = 0.00
                 ):

        pn = tf.keras.regularizers.l2(l2_param_penalty) if l2_param_penalty > 0 else None

        if conv_layers is None:
            conv_layers = [
                ConvLayerConfig(stride=2, filter_size=3, nr_filters=8, activation='elu', batch_norm=True),
                ConvLayerConfig(stride=2, filter_size=3, nr_filters=int(image_shape[-1]), activation='elu',
                                batch_norm=True),
            ]

        img_s = [int(x) for x in image_shape[:2]]
        for cl in conv_layers:
            img_s = [s / cl.stride for s in img_s]
        initial_shape = (int(img_s[0]), int(img_s[1]), 1)
        assert np.allclose(initial_shape[:2], img_s[:2]), 'eventual size divided by strides should be an integer'

        encoding = layers.Input(shape=(embedding_size,), name='embedding_input', dtype=tf.float32)

        e = layers.Dense(units=np.prod(initial_shape), activation='elu')(encoding)
        e = layers.Reshape(target_shape=initial_shape)(e)

        for cl in conv_layers:
            e = layers.Conv2DTranspose(filters=cl.nr_filters, kernel_size=(cl.filter_size, cl.filter_size),
                                       strides=(cl.stride, cl.stride), data_format='channels_last', padding='same',
                                       activation=cl.activation,
                                       kernel_regularizer=pn
                                       )(e)
            if cl.batch_norm:
                e = layers.BatchNormalization()(e)
        rgb_norm = e

        assert rgb_norm.shape[1:] == image_shape

        self.model = tf.keras.Model(inputs=[encoding], outputs=[rgb_norm])

    def __call__(self, embedding, *args, **kwargs):
        return self.model(inputs=embedding, *args, **kwargs)


class QNet(object):

    def __init__(self, layer_sizes: List[int], layer_activations: List[Any], state_shape: tuple, action_shape: tuple,
                 shared_layers: int, layer_and_batch_norm: bool, l2_param_penalty: float = 0.00):
        self.layer_and_batch_norm = layer_and_batch_norm
        self.shared_layers = shared_layers
        self.action_shape = action_shape
        self.state_shape = state_shape
        self.layer_activations = layer_activations
        self.layer_sizes = layer_sizes
        self.model: tf.keras.Model = None
        self.pn = tf.keras.regularizers.l2(l2_param_penalty) if l2_param_penalty > 0 else None

    def layer_with_layer_norm(self, x, layer_index, branch_name, ln_bias=0.1, initializers=None):
        layer_name = f'{branch_name}_{layer_index}'
        bi = 'zeros' if initializers is None else initializers
        ki = 'glorot_uniform' if initializers is None else initializers
        x = layers.Dense(units=self.layer_sizes[layer_index],
                         use_bias=True,
                         name=f'{layer_name}_W',
                         kernel_regularizer=self.pn,
                         bias_initializer=bi, kernel_initializer=ki
                         )(x)
        if self.layer_and_batch_norm:
            x = LayerNorm(name=layer_name, initial_bias=ln_bias)(x)
        x = layers.Activation(self.layer_activations[layer_index])(x)
        return x

    def __call__(self, inputs, *args, **kwargs):
        return self.model(inputs=inputs, *args, **kwargs)


class DQNNetwork(QNet):

    def __init__(self, layer_sizes: List[int], layer_activations: List[Any], state_shape: tuple,
                 action_shape: tuple, shared_layers: int, layer_and_batch_norm: bool, l2_param_penalty: float = 0.00):
        super().__init__(layer_sizes, layer_activations, state_shape, action_shape, shared_layers, layer_and_batch_norm,
                         l2_param_penalty=l2_param_penalty)

        assert len(action_shape) == 1
        number_of_actions = action_shape[0]
        state = tf.keras.Input(shape=state_shape, name='observation_input')
        t = state
        idx = 0

        while idx < len(layer_sizes):
            t = self.layer_with_layer_norm(t, idx, 'shared')
            idx += 1

        q = layers.Dense(units=number_of_actions, use_bias=True, activation=None, name='q_prediction')(t)

        self.model = tf.keras.Model(inputs=[state], outputs=[q])


class DQNPolNetwork(QNet):

    def __init__(self, layer_sizes: List[int], layer_activations: List[Any], state_shape: tuple,
                 action_shape: tuple, layer_and_batch_norm: bool, l2_param_penalty: float = 0.00, dueling: bool = False,
                 **kwargs):
        number_of_actions = action_shape[0]

        super().__init__(layer_sizes, layer_activations, state_shape, action_shape, 0, layer_and_batch_norm,
                         l2_param_penalty=l2_param_penalty)

        assert len(action_shape) == 1

        state = tf.keras.Input(shape=state_shape, name='observation_input')
        shared = state
        shared_sg = layers.Lambda(lambda x: tf.keras.backend.stop_gradient(x))(shared)

        for idx in range(len(layer_sizes)):
            # shared = self.layer_with_layer_norm(shared, idx, 'shared')
            shared = layers.Dense(units=self.layer_sizes[idx], kernel_regularizer=self.pn)(shared)
            shared = layers.Lambda(lambda x: layer_norm_alternative(x, center=True, scale=True))(shared)
            shared = layers.Activation(self.layer_activations[idx])(shared)

        q_head = layers.Dense(units=number_of_actions, use_bias=True, activation=None,
                              name='q_prediction', kernel_regularizer=self.pn, bias_regularizer=self.pn)(shared)
        q_n_head = layers.Dense(units=number_of_actions, use_bias=True, activation=None,
                                name='q_noise_prediction', kernel_regularizer=self.pn, bias_regularizer=self.pn)(shared)

        # if dueling:
        #     state_value = state
        #     for idx in range(len(layer_sizes)):
        #         state_value = layers.Dense(units=self.layer_sizes[idx], kernel_regularizer=self.pn)(state_value)
        #         state_value = layers.Lambda(lambda x: layer_norm_alternative(x, center=True, scale=True))(state_value)
        #         state_value = layers.Activation(self.layer_activations[idx])(state_value)
        #
        #     v_head = layers.Dense(units=1, use_bias=True, activation=None,
        #                           name='v_prediction', kernel_regularizer=self.pn, bias_regularizer=self.pn)(shared)
        #     a_head_centered = layers.Lambda(lambda q: q - tf.expand_dims(tf.reduce_mean(q, 1), 1))(q_head)
        #     q_head = layers.Lambda(lambda va: va[0] + va[1])((v_head, a_head_centered))

        p_head = layers.Dense(units=number_of_actions, use_bias=True, activation=None,
                              name='policy', kernel_regularizer=self.pn, bias_regularizer=self.pn)(
            shared_sg)  # TODO DIVERGE
        pn_head = layers.Dense(units=number_of_actions, use_bias=True, activation=None,
                               name='noise_policy')(shared_sg)  # TODO DIVERGE

        self.model = tf.keras.Model(inputs=[state], outputs=[q_head, q_n_head, p_head, pn_head])


class DQNRacingPolNetwork(QNet):

    def __init__(self, layer_sizes: List[int], layer_activations: List[Any], state_shape: tuple,
                 action_shape: tuple, layer_and_batch_norm: bool, l2_param_penalty: float = 0.00,
                 **kwargs):

        number_of_actions = action_shape[0]

        super().__init__(layer_sizes, layer_activations, state_shape, action_shape, 0, layer_and_batch_norm,
                         l2_param_penalty=l2_param_penalty)

        assert len(action_shape) == 1

        state = tf.keras.Input(shape=state_shape, name='observation_input')
        shared = state

        for idx in range(len(layer_sizes)):
            shared = self.layer_with_layer_norm(shared, idx, 'shared')

        shared_nog = layers.Lambda(lambda x: tf.keras.backend.stop_gradient(x))(shared)

        q_head = layers.Dense(units=number_of_actions, use_bias=True, activation=None,
                              name='q_prediction', kernel_regularizer=self.pn, bias_regularizer=self.pn)(shared)
        q_n_head = layers.Dense(units=number_of_actions, use_bias=True, activation=None,
                                name='q_noise_prediction', kernel_regularizer=self.pn, bias_regularizer=self.pn)(shared)

        p_head = layers.Dense(units=number_of_actions, use_bias=True, activation=None,
                              name='policy', kernel_regularizer=self.pn, bias_regularizer=self.pn)(shared_nog)  # TODO DIVERGE
        pn_head = layers.Dense(units=number_of_actions, use_bias=True, activation=None,
                               name='noise_policy')(shared_nog)

        self.model = tf.keras.Model(inputs=[state], outputs=[q_head, q_n_head, p_head, pn_head])



class DDPGQNetwork(QNet):

    def __init__(self, layer_sizes: List[int], layer_activations: List[Any], state_shape: tuple,
                 action_shape: tuple, layer_and_batch_norm: bool, l2_param_penalty: float = 0.00,
                 **kwargs):

        super().__init__(layer_sizes, layer_activations, state_shape, action_shape, 0, layer_and_batch_norm,
                         l2_param_penalty)

        final_init = initializers.uniform(minval=-3e-3, maxval=3e-3)
        hidden_init = initializers.VarianceScaling(scale=1/3, mode='fan_in', distribution='uniform', seed=None)

        state = tf.keras.Input(shape=state_shape, name='state_input')
        action = tf.keras.Input(shape=action_shape, name='action_input')

        h = layers.Concatenate()([state, action])

        for i in range(len(layer_sizes)):
            h = self.layer_with_layer_norm(h, i, 'Q', ln_bias=0., initializers=hidden_init)

        q = layers.Dense(units=1,
                         bias_initializer=final_init, #keras bug, simply delete the partition bit from the tf code
                         kernel_initializer=final_init)(h)

        self.model = tf.keras.Model(inputs=[state, action], outputs=[q])


class DDPGPolicyNetwork(QNet):

    def __init__(self, layer_sizes: List[int], layer_activations: List[Any], state_shape: tuple,
                 action_shape: tuple, layer_and_batch_norm: bool, l2_param_penalty: float = 0.00,
                 **kwargs):

        super().__init__(layer_sizes, layer_activations, state_shape, action_shape, 0, layer_and_batch_norm,
                         l2_param_penalty)

        hidden_init = initializers.VarianceScaling(scale=1/3, mode='fan_in', distribution='uniform', seed=None)
        # hidden_init = None
        final_init = initializers.RandomUniform(minval=-3e-3, maxval=3e-3, seed=None)

        state = tf.keras.Input(shape=state_shape, name='state_input')
        h = state

        for i in range(len(layer_sizes)):
            h = self.layer_with_layer_norm(h, i, 'policy', ln_bias=0., initializers=hidden_init)

        ap = layers.Dense(units=action_shape[0], activation='tanh', name='policy_final',
                          bias_initializer=final_init,
                          kernel_initializer=final_init)(h)
        h = state

        for i in range(len(layer_sizes)):
            h = self.layer_with_layer_norm(h, i, 'noise_policy', ln_bias=0., initializers=hidden_init)

        np = layers.Dense(units=action_shape[0], activation='tanh', name='noise_policy_final',
                          bias_initializer=final_init,
                          kernel_initializer=final_init)(h)

        self.model = tf.keras.Model(inputs=[state], outputs=[ap, np])


class CMAESPolNet(object):

    def __init__(self, state_shape: tuple, action_shape: tuple,        **kwargs):
        assert len(action_shape) == 1
        number_of_actions = action_shape[0]

        state_input = tf.keras.Input(shape=state_shape, name='observation_input')
        state = layers.Lambda(lambda x: tf.keras.backend.stop_gradient(x))(state_input)

        state = layers.Dense(units=64, activation='relu')(state)
        p_head = layers.Dense(units=number_of_actions, use_bias=True, activation=None,
                              name='cmaes_policy')(state)

        self.model = tf.keras.Model(inputs=[state_input], outputs=[p_head])

    def __call__(self, inputs, *args, **kwargs):
        return self.model(inputs=inputs, *args, **kwargs)