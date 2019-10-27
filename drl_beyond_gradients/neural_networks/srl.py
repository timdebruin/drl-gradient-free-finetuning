import warnings
from typing import Optional, NamedTuple, Any

import numpy as np
import tensorflow as tf
from keras.losses import mean_squared_error
from tensorflow.python.keras import backend, Model, layers, Sequential

from drl_beyond_gradients.neural_networks.networks import RGBDecoder


class MDPTensors(NamedTuple):
    value_function_loss: Any
    observation: Any
    next_observation: Any
    state_representation: Any
    next_state_representation: Any
    action: Any
    reward: Any
    terminal: Any


class SRLObjectives(NamedTuple):
    value_function: Any
    reward_prediction: Any
    auto_encoding: Any
    forward_dynamics: Any
    inverse_dynamics: Any
    slowness: Any
    diversity: Any


class SRLObjective(object):

    def __init__(self, tensors: MDPTensors, name: str, scale: Optional[float]) -> None:
        """
        An SRL loss objective will be made if scale != 0. If scale < 0 the objective will not be used to train the 
        representation, but the objective will be trained and the loss values will be logged for diagnostics. 
        """
        self.tensors = tensors

        scale = scale if scale is not None else 1.0
        self.stop_gradient = scale < 0.
        self.scale = backend.variable(np.abs(scale), dtype=tf.float32, name=f'srl_{name}_scale')
        self.scale = tf.stop_gradient(self.scale)
        warnings.warn('preventing gradients to the objective scales, fix for using gradient norm')

        self.model = None
        self.loss = None if np.abs(scale) > 0 else backend.zeros(shape=(1,))
        self.name = name

    def optional_gradient_stop(self, *args):
        f = (lambda x: layers.Lambda(lambda y: tf.keras.backend.stop_gradient(y))(x)) if self.stop_gradient else (
            lambda x: x)
        return tuple([f(a) for a in args])


class ValueFunction(SRLObjective):

    def __init__(self, tensors: MDPTensors, scale: Optional[float], **kwargs):
        super().__init__(tensors, 'value_function', scale)

        if self.loss is None:
            initial_loss = tf.keras.Input(shape=self.tensors.value_function_loss.shape,
                                          name='unscaled_value_function_loss')
            loss, = self.optional_gradient_stop(initial_loss)
            scaled_loss = layers.Lambda(lambda x: x * self.scale)(loss)

            self.model = Model(inputs=[initial_loss], outputs=[scaled_loss])
            self.loss = self.model(tensors.value_function_loss)


class RewardPrediction(SRLObjective):

    def __init__(self, tensors: MDPTensors, scale=None, **kwargs):
        super().__init__(tensors, name='reward_prediction', scale=scale)

        if self.loss is None:
            state_rep = tf.keras.Input(shape=self.tensors.state_representation.shape[1:],
                                       name='state_representation_input')
            act_in = tf.keras.Input(shape=self.tensors.action.shape[1:], name='action_input')
            rewards = tf.keras.Input(shape=self.tensors.reward.shape[1:], name='rewards_input')

            if kwargs['discrete_actions']:
                act = layers.Lambda(lambda x: tf.cast(x, tf.int32))(act_in)
                act = layers.Lambda(lambda x: tf.one_hot(x, depth=kwargs['n_actions'], dtype=tf.float32))(act)
            else:
                act = act_in

            merged = layers.concatenate([state_rep, act])
            merged, = self.optional_gradient_stop(merged)

            x = layers.Dense(32, activation='elu')(merged)
            pred = layers.Dense(1, activation=None)(x)
            mse = layers.Lambda(lambda x: mean_squared_error(x[0], x[1]))((rewards, pred))
            mse = layers.Lambda(lambda x: backend.mean(x))(mse)
            scaled_mse = layers.Lambda(lambda x: x * self.scale)(mse)

            self.model = Model(inputs=[state_rep, act_in, rewards], outputs=[scaled_mse])
            self.loss = self.model([self.tensors.state_representation, self.tensors.action, self.tensors.reward])


class ForwardDynamicsPrediction(SRLObjective):

    def __init__(self, tensors: MDPTensors, scale=None, **kwargs):
        super().__init__(tensors, name='forward_dynamics_prediction', scale=scale)

        if self.loss is None:
            state_rep = tf.keras.Input(shape=self.tensors.state_representation.shape[1:],
                                       name='state_representation_input')
            act_in = tf.keras.Input(shape=self.tensors.action.shape[1:], name='action_input')
            next_state_rep = tf.keras.Input(shape=self.tensors.next_state_representation.shape[1:],
                                            name='next_state_representation')

            if kwargs['discrete_actions']:
                act = layers.Lambda(lambda x: tf.cast(x, tf.int32))(act_in)
                act = layers.Lambda(lambda x: tf.one_hot(x, depth=kwargs['n_actions'], dtype=tf.float32))(act)
            else:
                act = act_in

            merged = layers.concatenate([state_rep, act])
            merged, = self.optional_gradient_stop(merged)

            x = layers.Dense(64, activation='elu')(merged)
            pred = layers.Dense(self.tensors.next_state_representation.shape[1], activation=None)(x)
            mse = layers.Lambda(lambda x: mean_squared_error(x[0], x[1]))((next_state_rep, pred))
            mse = layers.Lambda(lambda x: backend.mean(x))(mse)
            scaled_mse = layers.Lambda(lambda x: x * self.scale)(mse)

            self.model = Model(inputs=[state_rep, act_in, next_state_rep], outputs=[scaled_mse])
            self.loss = self.model([self.tensors.state_representation, self.tensors.action,
                                    self.tensors.next_state_representation])


class InverseDynamicsPrediction(SRLObjective):

    def __init__(self, tensors: MDPTensors, scale=None, **kwargs):
        super().__init__(tensors, name='inverse_dynamics_prediction', scale=scale)

        if self.loss is None:
            state_rep = tf.keras.Input(shape=self.tensors.state_representation.shape[1:],
                                       name='state_representation_input')
            act_in = tf.keras.Input(shape=self.tensors.action.shape[1:], name='action_input')
            next_state_rep = tf.keras.Input(shape=self.tensors.next_state_representation.shape[1:],
                                            name='next_state_representation')

            merged = layers.concatenate([state_rep, next_state_rep])
            merged, = self.optional_gradient_stop(merged)

            x = layers.Dense(64, activation='elu')(merged)
            if kwargs['discrete_actions']:
                pred = layers.Dense(kwargs['n_actions'], activation=None)(x)
                act = layers.Lambda(lambda x: tf.cast(x, tf.int32))(act_in)
                loss = layers.Lambda(lambda x:
                                     tf.nn.sparse_softmax_cross_entropy_with_logits(
                                         labels=x[1], logits=x[0]))([pred, act])
            else:
                pred = layers.Dense(self.tensors.action.shape[1], activation='tanh')(x)
                loss = layers.Lambda(lambda x: mean_squared_error(x[0], x[1]))((act_in, pred))

            loss = layers.Lambda(lambda x: backend.mean(x))(loss)
            scaled_loss = layers.Lambda(lambda x: x * self.scale)(loss)

            self.model = Model(inputs=[state_rep, act_in, next_state_rep], outputs=[scaled_loss])
            self.loss = self.model([self.tensors.state_representation, self.tensors.action,
                                    self.tensors.next_state_representation])


class SlownessLoss(SRLObjective):

    def __init__(self, tensors: MDPTensors, scale=None, **kwargs):
        super().__init__(tensors, name='slowness_loss', scale=scale)

        if self.loss is None:
            state_rep = self.tensors.state_representation
            next_state_rep = self.tensors.next_state_representation

            if self.stop_gradient:
                state_rep = tf.stop_gradient(state_rep)
                next_state_rep = tf.stop_gradient(next_state_rep)

            two_norm = tf.norm(state_rep - next_state_rep, ord=2, axis=-1, keepdims=False)
            self.loss = self.scale * tf.reduce_mean(two_norm)


class DiversityLoss(SRLObjective):

    def __init__(self, tensors: MDPTensors, scale=None, **kwargs):
        super().__init__(tensors, name='diversity_loss', scale=scale)

        if self.loss is None:
            state_rep = self.tensors.state_representation
            batch_size = kwargs['batch_size']

            if self.stop_gradient:
                state_rep = tf.stop_gradient(state_rep)

            differences = []
            for bi in range(batch_size - 1):
                differences.append(tf.exp(-tf.norm(tf.slice(state_rep, [bi, 0], [
                    1, -1]) - tf.slice(state_rep, [bi + 1, 0], [1, -1]), 1)))

            self.loss = self.scale * tf.add_n(differences) / (
                    batch_size - 1)


class AutoEncodingPrediction(SRLObjective):

    def __init__(self, tensors: MDPTensors, scale=None, **kwargs):
        super().__init__(tensors, name='auto_encoding_prediction', scale=scale)

        if self.loss is None:

            state_rep_in = tf.keras.Input(shape=self.tensors.state_representation.shape[1:],
                                          name='state_representation_input')

            state_rep, = self.optional_gradient_stop(state_rep_in)

            ob_shape = self.tensors.observation.shape[1:]
            if len(ob_shape) == 3:
                decoder = RGBDecoder(image_shape=ob_shape,
                                     embedding_size=self.tensors.state_representation.shape[1])
                observation = tf.keras.Input(shape=ob_shape, name='observation_input',
                                             dtype=tf.uint8)
                target = layers.Lambda(lambda x: backend.cast(x, dtype='float32') / 127.5 - 1.)(observation)
            else:
                assert len(ob_shape) == 1
                decoder = Sequential()
                decoder.add(layers.Dense(32, activation='elu'))
                decoder.add(layers.Dense(ob_shape[0], activation='tanh'))
                observation = tf.keras.Input(shape=ob_shape, name='observation_input',
                                             dtype=tf.float32)
                target = observation

            decoded = decoder(state_rep)
            mse = layers.Lambda(lambda x: mean_squared_error(x[0], x[1]))((target, decoded))
            mse = layers.Lambda(lambda x: backend.mean(x))(mse)
            scaled_mse = layers.Lambda(lambda x: x * self.scale)(mse)

            self.model = Model(inputs=[observation, state_rep_in], outputs=[scaled_mse])
            self.loss = self.model([self.tensors.observation, self.tensors.state_representation])


def get_gradient_op(tensors: MDPTensors, objective_initial_scales: SRLObjectives,
                    optimizer: tf.train.Optimizer, gradient_clip: Optional[float],
                    **kwargs):
    objectives: SRLObjectives = SRLObjectives(
        value_function=ValueFunction(tensors, objective_initial_scales.value_function, **kwargs),
        reward_prediction=RewardPrediction(tensors, objective_initial_scales.reward_prediction, **kwargs),
        auto_encoding=AutoEncodingPrediction(tensors, objective_initial_scales.auto_encoding, **kwargs),
        forward_dynamics=ForwardDynamicsPrediction(tensors, objective_initial_scales.forward_dynamics, **kwargs),
        inverse_dynamics=InverseDynamicsPrediction(tensors, objective_initial_scales.inverse_dynamics, **kwargs),
        slowness=SlownessLoss(tensors, objective_initial_scales.slowness, **kwargs),
        diversity=DiversityLoss(tensors, objective_initial_scales.diversity, **kwargs),
    )

    active_objectives = [o for o in objectives if o is not None and backend.get_value(o.scale) > 0]
    total_loss = backend.mean(backend.stack([o.loss for o in active_objectives]))

    if gradient_clip is not None:
        gradients = optimizer.compute_gradients(total_loss)
        for i, (grad, var) in enumerate(gradients):
            if grad is not None:
                gradients[i] = (tf.clip_by_norm(grad, gradient_clip), var)
        return optimizer.apply_gradients(gradients)
    else:
        return optimizer.minimize(total_loss)
