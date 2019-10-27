import collections
import os
import warnings
from typing import List, Optional

import cma
import numpy as np
import tensorflow as tf
from cor_control_benchmarks.control_benchmark import ControlBenchmark
from keras import Model
from keras.backend import batch_get_value, batch_set_value
from tensorflow.python import keras
from tensorflow.python.framework.errors_impl import InvalidArgumentError
from tensorflow.python.keras import layers

from drl_beyond_gradients.common.experience_buffer import BaseExperienceBuffer
from drl_beyond_gradients.common.helpers import batch_index
from drl_beyond_gradients.common.policy import AbstractPolicy
from drl_beyond_gradients.neural_networks.networks import DQNPolNetwork, DQNRacingPolNetwork, CMAESPolNet
from drl_beyond_gradients.neural_networks.shared import ModularHeadModels
from drl_beyond_gradients.neural_networks.srl import MDPTensors, SRLObjectives, get_gradient_op


class NNPolicy(AbstractPolicy):

    def __init__(self,
                 environment: ControlBenchmark,
                 experience_buffer: BaseExperienceBuffer,
                 tensorflow_session: tf.Session,
                 rl_grad_clip=1.0,
                 param_grad_clip=10.,
                 policy_head=True,
                 dueling=False,
                 optimizer: str = 'adam',
                 adam_epsilon: float = 1e-8,
                 l2_param_penalty: float = 0.00,
                 gamma: float = 0.95,
                 lr: float = 0.001,
                 batch_size=128,
                 parameter_exploration_update_speed=1.01,
                 use_plappert_distance=False,
                 state_encoder_class: Optional[keras.Model] = None,
                 srl_vf: float = 1., srl_ae: float = 0., srl_rp: float = 0., srl_fd: float = 0., srl_id: float = 0.,
                 srl_sf: float = 0., srl_di: float = 0.,
                 cmaes_dir='',
                 **kwargs):

        super().__init__(environment=environment, experience_buffer=experience_buffer)
        self.cmaes_dir = cmaes_dir
        self.parameter_exploration_update_speed = parameter_exploration_update_speed
        self.gamma = gamma
        self.tensorflow_session = tensorflow_session
        self.batch_size = batch_size
        state_shape = environment.state_shape if state_encoder_class is None else (3136,)

        self.separate_cmaes_policy = 'NoFrameskip' in self.environment.name

        self.action_shape = environment.discrete_action_shape
        self.action_feed_shape = (-1,)
        self.action_input = tf.keras.Input(shape=(), dtype=np.int32, name='action_placeholder')

        layer_sizes = [512]
        activations = ['relu']
        layer_and_batch_norm = True

        def create_model() -> Model:
            netbuild = DQNRacingPolNetwork if 'Racing' in self.environment.name else DQNPolNetwork
            return netbuild(layer_sizes=layer_sizes, layer_activations=activations,
                            state_shape=state_shape,
                            action_shape=self.action_shape, layer_and_batch_norm=layer_and_batch_norm,
                            l2_param_penalty=l2_param_penalty, dueling=dueling,
                            ).model

        self.observation_input = tf.keras.Input(shape=self.environment.state_shape, name='observation')
        self.next_observation_input = tf.keras.Input(shape=self.environment.state_shape, name='next_observation')
        self.reward_input = tf.keras.Input(shape=(), name='reward')
        self.terminal_input = tf.keras.Input(shape=(), name='terminal')

        self.p_continue = self.gamma * (1 - self.terminal_input)

        identity_encoder = keras.Sequential()
        identity_encoder.add(layers.Lambda(lambda x: x + 0, input_shape=self.environment.state_shape))

        self.nets = ModularHeadModels(
            online={
                'encoder': state_encoder_class().model if state_encoder_class is not None else identity_encoder,
                'rl': create_model()},
            target={
                'encoder': state_encoder_class().model if state_encoder_class is not None else identity_encoder,
                'rl': create_model()},
        )

        self.cmaes_net = CMAESPolNet(state_shape=state_shape, action_shape=self.action_shape)
        self.cmaes_net_copy = CMAESPolNet(state_shape=state_shape, action_shape=self.action_shape)

        for sub_model in ('encoder', 'rl'):
            self.nets.target[sub_model].set_weights(self.nets.online[sub_model].get_weights())

        self.online_state = self.nets.online['encoder'](self.observation_input)
        self.online_next_state = self.nets.online['encoder'](self.next_observation_input)
        self.target_next_state = self.nets.target['encoder'](self.next_observation_input)

        self.frozen_parameter_equalize_ops = []
        for src, t in zip(self.nets.online['encoder'].variables + self.nets.online['rl'].variables,
                          self.nets.target['encoder'].variables + self.nets.target['rl'].variables):
            self.frozen_parameter_equalize_ops.append(tf.assign(t, src))

        q_online, q_n_online, p_logits_online, pn_logits_online = self.nets.online['rl']([self.online_state])
        q_online_next, _, _, _ = self.nets.online['rl']([self.online_next_state])
        q_target_next, q_n_target_next, p_logits_target_next, _ = self.nets.target['rl']([self.target_next_state])

        p_online = tf.nn.softmax(p_logits_online)
        p_explore = tf.nn.softmax(pn_logits_online)
        p_target_next = tf.nn.softmax(p_logits_target_next)

        if policy_head:
            print('using explicit policy head')
            self.policy_action = tf.argmax(p_online, -1, output_type=tf.int32)
            self.perturbed_policy_action = tf.argmax(p_explore, -1, output_type=tf.int32)
            p_next = p_target_next
            self.p_loss = tf.losses.softmax_cross_entropy(
                onehot_labels=tf.one_hot(indices=tf.argmax(tf.stop_gradient(q_online), axis=-1),
                                         depth=self.action_shape[0]),
                logits=p_logits_online
            )
            self._policy_weights = [w for w in self.nets.online['rl'].weights if
                                    'policy' in w.name and 'noise' not in w.name]
            self._perturbed_policy_weights = [w for w in self.nets.online['rl'].weights if 'noise_policy' in w.name]
            self._policy_parameter_shapes = [w.shape for w in self._perturbed_policy_weights]
        else:
            print('Using max of Q as deterministic policy')
            self.policy_action = tf.argmax(q_online, -1, output_type=tf.int32)
            self.perturbed_policy_action = tf.argmax(q_n_online, -1, output_type=tf.int32)
            p_next = q_target_next
            self.p_loss = None
            self._policy_weights = [w for w in self.nets.online['rl'].weights if
                                    'q_prediction' in w.name]
            self._perturbed_policy_weights = [w for w in self.nets.online['rl'].weights if
                                              'q_noise_prediction' in w.name]
            self._policy_parameter_shapes = [w.shape for w in self._perturbed_policy_weights]

        if self.separate_cmaes_policy:
            self.cmeas_logits = self.cmaes_net([self.online_state])
            self.cmaes_action = tf.argmax(self.cmeas_logits, -1, output_type=tf.int32)
            self.cmaes_copy_action = tf.argmax(self.cmaes_net_copy([self.online_state]), -1, output_type=tf.int32)
            self.cmeas_loss = keras.losses.kullback_leibler_divergence(
                y_true=tf.stop_gradient(p_online),
                y_pred=tf.nn.softmax(self.cmeas_logits))
            cmaes_optim = tf.train.AdamOptimizer(learning_rate=1e-3)
            self.cmaes_train_op = cmaes_optim.minimize(self.cmeas_loss)
            self._cmaes_weights = [w for w in self.cmaes_net.model.weights if 'cmaes_policy' in w.name]
            self._cmaes_parameter_shapes = [w.shape for w in self._cmaes_weights]
        else:
            self.cmaes_action = self.perturbed_policy_action
            self._cmaes_weights = self._perturbed_policy_weights
            self._cmaes_parameter_shapes = self._policy_parameter_shapes

        self.q_values_policy = batch_index(values=q_online, indices=self.policy_action)
        self.q_values_action = batch_index(values=q_online, indices=self.action_input)

        self.v_target_next = batch_index(values=q_target_next,
                                         indices=tf.argmax(p_next, -1, output_type=tf.int32))

        self.target = tf.stop_gradient(self.reward_input + self.p_continue * self.v_target_next)

        self.tde = self.q_values_action - self.target
        huber_loss = tf.reduce_mean(
            tf.where(
                tf.abs(self.tde) < rl_grad_clip,
                tf.square(self.tde) * 0.5,
                rl_grad_clip * (tf.abs(self.tde) - 0.5 * rl_grad_clip)
            )
        )
        self.rl_loss = huber_loss
        if self.p_loss is not None:
            self.rl_loss += self.p_loss

        if optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=adam_epsilon)
        elif optimizer == 'sgd':
            optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.)

        self.srlq_update_op = get_gradient_op(
            tensors=MDPTensors(
                value_function_loss=self.rl_loss,
                observation=self.observation_input,
                next_observation=self.next_observation_input,
                state_representation=self.online_state,
                next_state_representation=self.online_next_state,
                action=self.action_input,
                reward=self.reward_input,
                terminal=self.terminal_input,
            ),
            objective_initial_scales=SRLObjectives(
                value_function=srl_vf,
                reward_prediction=srl_rp,
                auto_encoding=srl_ae,
                forward_dynamics=srl_fd,
                inverse_dynamics=srl_id,
                slowness=srl_sf,
                diversity=srl_di,
            ),
            session=tensorflow_session,
            optimizer=optimizer,
            batch_size=self.batch_size,
            gradient_clip=param_grad_clip,
            discrete_actions=hasattr(self.environment, 'discrete_action_shape'),
            n_actions=self.action_shape[0]
        )

        self.desired_exploration_epsilon = 0.2
        self.exploration_sigma = 0.2
        self.expl_dist_list = []

        if use_plappert_distance:
            self.exploration_distance_op = tf.reduce_mean(
                tf.reduce_sum(p_online * (tf.log(p_online) - tf.log(p_explore)), axis=-1))

            self.exploration_threshold = lambda: -np.log(
                1. - self.desired_exploration_epsilon + self.desired_exploration_epsilon / float(self.action_shape[0]))

        else:
            self.exploration_distance_op = tf.reduce_mean(tf.cast(
                tf.not_equal(self.policy_action, self.perturbed_policy_action), tf.float32))

            self.exploration_threshold = lambda: self.desired_exploration_epsilon

        self.diagnostics = collections.defaultdict(list)

        self.initialize_tf_variables()

        # cma-es training
        self.es: Optional[cma.CMAEvolutionStrategy] = None
        self.tested_cma_es_params: List[np.ndarray] = []
        self.tested_cma_es_params_fvals: List[float] = []
        self.cma_es_params_to_try: List[np.ndarray] = []
        self.current_behavior_params: Optional[np.ndarray] = None
        self.cma_es_asked: bool = False

    def __call__(self, *args, **kwargs) -> np.ndarray:
        """evaluate the policy action"""
        super().__call__(*args, **kwargs)
        tf.keras.backend.set_learning_phase(0)

        if self.step % 10000 == 0:
            self.tensorflow_session.run(self.frozen_parameter_equalize_ops)

        use_cmaes = kwargs.get('cmaes', False)
        expl_op = self.cmaes_action if use_cmaes else self.perturbed_policy_action

        expl, pol, v_value, distance = self.tensorflow_session.run([expl_op, self.policy_action,
                                                                    self.q_values_policy, self.exploration_distance_op],
                                                                   feed_dict={self.observation_input: self._state})

        action = expl if kwargs.get('explore', False) else pol

        return np.squeeze(action)

    def train(self, **kwargs):
        """Performs a single training step"""

        tf.keras.backend.set_learning_phase(1)

        batch = self.experience_buffer.sample_batch(batch_size=self.batch_size)
        feed_dict = {
            self.observation_input: batch.experiences.state,
            self.next_observation_input: batch.experiences.next_state,
            self.action_input: batch.experiences.action.reshape(
                self.action_feed_shape),
            self.reward_input: batch.experiences.reward,
            self.terminal_input: batch.experiences.terminal
        }

        self.tensorflow_session.run(self.srlq_update_op, feed_dict=feed_dict)

    def pre_train_cmaes(self):
        if not self.separate_cmaes_policy:
            return
        tf.keras.backend.set_learning_phase(1)

        last_score = -1
        score = 0

        while score > last_score:
            scores = []
            last_score = score
            for _ in range(len(self.experience_buffer) // self.batch_size):
                batch = self.experience_buffer.sample_batch(batch_size=self.batch_size)
                feed_dict = {self.observation_input: batch.experiences.state}

                _, p, c = self.tensorflow_session.run([self.cmaes_train_op, self.policy_action, self.cmaes_action],
                                                      feed_dict)
                scores.append(np.sum(p == c) / np.size(p))
            score = np.mean(scores)
            print(score)

    @property
    def policy_weights(self):
        return batch_get_value(self._policy_weights)

    @policy_weights.setter
    def policy_weights(self, value):
        assert len(value) == len(self.policy_weights)
        tuples = []
        for sw, w in zip(self._policy_weights, value):
            tuples.append((sw, w))
        batch_set_value(tuples)

    @property
    def perturbed_policy_weights(self):
        return batch_get_value(self._perturbed_policy_weights)

    @perturbed_policy_weights.setter
    def perturbed_policy_weights(self, value):
        assert len(value) == len(self._perturbed_policy_weights)
        tuples = []
        for sw, w in zip(self._perturbed_policy_weights, value):
            tuples.append((sw, w))
        batch_set_value(tuples)

    @property
    def cmaes_weights(self):
        return batch_get_value(self._cmaes_weights)

    @cmaes_weights.setter
    def cmaes_weights(self, value):
        assert len(value) == len(self._cmaes_weights)
        tuples = []
        for sw, w in zip(self._cmaes_weights, value):
            tuples.append((sw, w))
        batch_set_value(tuples)

    @property
    def flat_perturbed_policy_weights(self):
        return np.concatenate([np.reshape(w, (-1,)) for w in self.perturbed_policy_weights])

    @flat_perturbed_policy_weights.setter
    def flat_perturbed_policy_weights(self, value):
        start = 0
        un_flattened = []

        for ws in self._policy_parameter_shapes:
            length = np.prod(ws)
            w = value[start:start + length]
            start += length
            un_flattened.append(w.reshape(ws))

        self.perturbed_policy_weights = un_flattened

    @property
    def flat_cmaes_weights(self):
        return np.concatenate([np.reshape(w, (-1,)) for w in self.cmaes_weights])

    @flat_cmaes_weights.setter
    def flat_cmaes_weights(self, value):
        start = 0
        un_flattened = []

        for ws in self._cmaes_parameter_shapes:
            length = np.prod(ws)
            w = value[start:start + length]
            start += length
            un_flattened.append(w.reshape(ws))

        self.cmaes_weights = un_flattened

    def initialize_tf_variables(self):
        self.tensorflow_session.run(tf.global_variables_initializer())

    def exploration_policy_update(self):
        self.perturbed_policy_weights = self.policy_weights
        self.flat_perturbed_policy_weights = np.random.normal(
            self.flat_perturbed_policy_weights, scale=self.exploration_sigma)

    # def update_sigma(self):
    #     m = self.parameter_exploration_update_speed if np.mean(self.expl_dist_list) < self.exploration_threshold() \
    #         else 1 / self.parameter_exploration_update_speed
    #     self.expl_dist_list.clear()
    #     self.exploration_sigma *= m

    def sample_cma_es(self, start_from, save_cmaes_params=True):
        if self.es is None:
            cma_es_params = self.flat_cmaes_weights
            if 'zero' in start_from:
                cma_es_params.fill(0.)
                initial_sigma = 1
            else:
                initial_sigma = self.exploration_sigma
            self.es = cma.CMAEvolutionStrategy(x0=cma_es_params, sigma0=initial_sigma)

        if len(self.cma_es_params_to_try) == 0:
            if len(self.tested_cma_es_params) > 0:
                if self.cma_es_asked:
                    self._save_es_generation(save_cmaes_params)
                    self.es.tell(self.tested_cma_es_params, self.tested_cma_es_params_fvals)
                    self.tested_cma_es_params.clear()
                    self.tested_cma_es_params_fvals.clear()
                    self.cma_es_asked = False
            self.cma_es_params_to_try = self.es.ask()
            self.cma_es_asked = True

        self.flat_cmaes_weights = self.cma_es_params_to_try.pop()

    def reset_cmaes(self) -> None:
        self.es = None
        self.cma_es_asked = False
        self.tested_cma_es_params.clear()
        self.tested_cma_es_params_fvals.clear()

    def finished_episode_with_score(self, last_reward_sum) -> None:
        if self.es:
            self.tested_cma_es_params.append(self.flat_cmaes_weights)
            self.tested_cma_es_params_fvals.append(-1 * last_reward_sum)

    def _save_es_generation(self, save_cmaes_params=True):
        r = self.es.result
        if save_cmaes_params:
            np.savez_compressed(f'{self.cmaes_dir}evalutaions_iter_{r.iterations}.npz',
                                fvalues=np.array(self.tested_cma_es_params_fvals),
                                params=np.array(self.tested_cma_es_params))
        else:
            np.savez_compressed(f'{self.cmaes_dir}evalutaions_iter_{r.iterations}.npz',
                                fvalues=np.array(self.tested_cma_es_params_fvals))

    def save_params_to_dir(self, save_dir, current=False):
        os.makedirs(save_dir, exist_ok=True)

        for model in self.nets.online.values():
            if any([np.isnan(x).any() for x in model.get_weights()]):
                warnings.warn('Not saving network parameters as NaN values are present')
                return

        for module_name, model in self.nets.online.items():
            print(f'saving to weights to {save_dir}{module_name}/')
            model.save_weights(f'{save_dir}{module_name}/params.ckpt')

        w = self.flat_cmaes_weights
        if current and self.es is not None:
            self.flat_cmaes_weights = self.es.result.xfavorite
            self.exploration_sigma = self.es.sigma
        if self.separate_cmaes_policy:
            self.cmaes_net.model.save_weights(f'{save_dir}cmaes_policy/params.ckpt')
        else:
            self.nets.online['rl'].save_weights(f'{save_dir}cmaes_policy/params.ckpt')
        self.flat_cmaes_weights = w

        np.save(f'{save_dir}/exploration_sigma.npy', self.exploration_sigma)

    def load_params_form_dir(self, load_dir):
        try:
            for net in self.nets:
                for module_name, model in net.items():
                    model.load_weights(f'{load_dir}{module_name}/params.ckpt')
            self.exploration_sigma = float(np.load(f'{load_dir}/exploration_sigma.npy'))
        except InvalidArgumentError:  # There used to just be the RL net
            for net in self.nets:
                net['rl'].load_weights(load_dir)

        try:
            self.cmaes_net.model.load_weights(f'{load_dir}cmaes_policy/params.ckpt')
        except Exception:
            print('COULD NOT FIND CMAES NET PARAMS')

    def adjust_sigma_to_match_epsilon(self, cma_es_start_epsilon, max_steps=20000):
        last10 = collections.deque(maxlen=10)
        self.desired_exploration_epsilon = cma_es_start_epsilon

        step = 0
        while not (sum([x > 1 for x in last10]) >= 4 and sum([x < 1 for x in last10]) >= 4) and step < max_steps:
            step += 1
            self.perturbed_policy_weights = self.policy_weights
            self.flat_perturbed_policy_weights = np.random.normal(
                self.flat_perturbed_policy_weights, scale=self.exploration_sigma)
            dist = []
            for _ in range(4):
                batch = self.experience_buffer.sample_batch(batch_size=2 * self.batch_size)
                dist.append(self.tensorflow_session.run(self.exploration_distance_op,
                                                        {self.observation_input: batch.experiences.state}))
            m = 1.01 if np.mean(dist) < self.exploration_threshold() else 1 / 1.01
            self.exploration_sigma *= m
            last10.append(m)

    def cmaes_init_sigma(self, cma_es_start_epsilon, max_steps=20000):
        self.exploration_sigma = 0.2
        if not self.separate_cmaes_policy:
            self.adjust_sigma_to_match_epsilon(cma_es_start_epsilon, max_steps)
            return

        self.cmaes_net_copy.model.set_weights(self.cmaes_net.model.get_weights())
        w0 = np.copy(self.flat_cmaes_weights)

        last10 = collections.deque(maxlen=10)

        step = 0
        while not (sum([x > 1 for x in last10]) >= 4 and sum([x < 1 for x in last10]) >= 4) and step < max_steps:
            step += 1
            self.flat_cmaes_weights = np.random.normal(w0, scale=self.exploration_sigma)
            dist = []
            for _ in range(4):
                batch = self.experience_buffer.sample_batch(batch_size=2 * self.batch_size)

                p1, p2 = self.tensorflow_session.run([self.cmaes_action, self.cmaes_copy_action],
                                                     {self.observation_input: batch.experiences.state})
                dist.append(1 - (np.sum(p1 == p2) / np.size(p1)))
            m = 1.01 if np.mean(dist) < cma_es_start_epsilon else 1 / 1.01
            self.exploration_sigma *= m
            last10.append(m)
            print(f'{self.exploration_sigma}  {sum([x > 1 for x in last10])}  {sum([x < 1 for x in last10])}')
