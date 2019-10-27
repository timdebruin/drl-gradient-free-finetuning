import collections
import os
import warnings
from typing import List, Optional

import cma
import numpy as np
import tensorflow as tf
from cor_control_benchmarks.control_benchmark import ControlBenchmark
from keras.backend import batch_get_value, batch_set_value
from tensorflow.python import keras
from tensorflow.python.framework.errors_impl import InvalidArgumentError
from tensorflow.python.keras import layers

from drl_beyond_gradients.common.experience_buffer import BaseExperienceBuffer
from drl_beyond_gradients.common.policy import AbstractPolicy
from drl_beyond_gradients.neural_networks.networks import DDPGQNetwork, DDPGPolicyNetwork
from drl_beyond_gradients.neural_networks.shared import ModularHeadModels
from drl_beyond_gradients.neural_networks.srl import MDPTensors, SRLObjectives, get_gradient_op


class NNPolicy(AbstractPolicy):

    def __init__(self, environment: ControlBenchmark, experience_buffer: BaseExperienceBuffer,
                 tensorflow_session: tf.Session,
                 rl_grad_clip=1.0,
                 param_grad_clip=10.,
                 optimizer: str = 'adam',
                 adam_epsilon: float = 1e-8,
                 tau: float = 0.001,
                 l2_param_penalty: float = 0.00,
                 gamma: float = 0.95,
                 lr: float = 0.001,
                 batch_size=128,
                 parameter_exploration_update_speed=1.01,
                 state_encoder_class: Optional[keras.Model] = None,
                 srl_vf: float = 1., srl_ae: float = 0., srl_rp: float = 0., srl_fd: float = 0., srl_id: float = 0.,
                 srl_sf: float = 0., srl_di: float = 0.,
                 cmaes_dir='',
                 **kwargs):

        super().__init__(environment=environment, experience_buffer=experience_buffer)

        self.tau = tau
        self.separate_cmaes_policy = False

        self.cmaes_dir = cmaes_dir
        self.parameter_exploration_update_speed = parameter_exploration_update_speed
        self.gamma = gamma
        self.tensorflow_session = tensorflow_session
        self.batch_size = batch_size
        if state_encoder_class is None:
            state_shape = environment.state_shape
        else:
            raise NotImplementedError

        self.action_feed_shape = -1, self.environment.action_shape[0]
        self.action_shape = environment.action_shape
        self.action_input = tf.keras.Input(shape=self.action_shape, name='action_placeholder')
        self.action_srl = self.action_input

        self.observation_input = tf.keras.Input(shape=self.environment.state_shape, name='observation')
        self.next_observation_input = tf.keras.Input(shape=self.environment.state_shape, name='next_observation')
        self.reward_input = tf.keras.Input(shape=(), name='reward')
        self.terminal_input = tf.keras.Input(shape=(), name='terminal')

        self.p_continue = self.gamma * (1 - self.terminal_input)

        identity_encoder = keras.Sequential()
        identity_encoder.add(layers.Lambda(lambda x: x + 0, input_shape=self.environment.state_shape))

        layer_sizes = [64, 64]
        nonlinearities = ['tanh', 'tanh']
        ln = True

        self.nets = ModularHeadModels(
            online={
                'encoder': identity_encoder,
                'q': DDPGQNetwork(layer_sizes=layer_sizes, layer_activations=nonlinearities, layer_and_batch_norm=ln,
                                  l2_param_penalty=l2_param_penalty, state_shape=state_shape, action_shape=self.action_shape).model,
                'pol': DDPGPolicyNetwork(layer_sizes=layer_sizes, layer_activations=nonlinearities,
                                         layer_and_batch_norm=ln, l2_param_penalty=0., state_shape=state_shape,
                                         action_shape=self.action_shape).model
            },
            target={
                'encoder': state_encoder_class().model if state_encoder_class is not None else identity_encoder,
                'q': DDPGQNetwork(layer_sizes=layer_sizes, layer_activations=nonlinearities, layer_and_batch_norm=ln,
                                  l2_param_penalty=l2_param_penalty, state_shape=state_shape, action_shape=self.action_shape).model,
                'pol': DDPGPolicyNetwork(layer_sizes=layer_sizes, layer_activations=nonlinearities,
                                         layer_and_batch_norm=ln, l2_param_penalty=0., state_shape=state_shape,
                                         action_shape=self.action_shape).model
            }
        )

        # initialize the target network to be the same as the online network
        for sub_model in ('encoder', 'q', 'pol'):
            self.nets.target[sub_model].set_weights(self.nets.online[sub_model].get_weights())

        self.online_state = self.nets.online['encoder'](self.observation_input)
        self.online_next_state = self.nets.online['encoder'](self.next_observation_input)
        self.target_next_state = self.nets.target['encoder'](self.next_observation_input)

        self.diag_state_sparsity = tf.nn.zero_fraction(self.online_state)
        self.diag_next_state_sparsity = tf.nn.zero_fraction(self.target_next_state)

        self.frozen_parameter_equalize_ops = []
        for src, t in zip(
                self.nets.online['encoder'].variables + self.nets.online['q'].variables + self.nets.online[
                    'pol'].variables,
                self.nets.target['encoder'].variables + self.nets.target['q'].variables + self.nets.target[
                    'pol'].variables):
            self.frozen_parameter_equalize_ops.append(tf.assign(t, (1. - self.tau) * t + self.tau * src))

        q_online = self.nets.online['q']([self.online_state, self.action_input])
        p_online, p_explore = self.nets.online['pol']([self.online_state])
        p_target_next, _ = self.nets.target['pol']([self.target_next_state])
        v_target_next = self.nets.target['q']([self.target_next_state, p_target_next])

        self.policy_action = p_online
        self.perturbed_policy_action = p_explore

        self._policy_weights = [w for w in self.nets.online['pol'].weights if
                                'policy' in w.name and 'noise' not in w.name]
        self._perturbed_policy_weights = [w for w in self.nets.online['pol'].weights if 'noise_policy' in w.name]
        self._policy_parameter_shapes = [w.shape for w in self._perturbed_policy_weights]

        self.q_values_policy = self.nets.online['q']([self.online_state, p_online])
        self.q_values_action = q_online

        self.v_target_next = v_target_next

        self.target = tf.stop_gradient(self.reward_input + self.p_continue * self.v_target_next)

        tde = self.q_values_action - self.target
        huber_loss = tf.reduce_mean(
            tf.where(
                tf.abs(tde) < rl_grad_clip,
                tf.square(tde) * 0.5,
                rl_grad_clip * (tf.abs(tde) - 0.5 * rl_grad_clip)
            )
        )
        self.initial_train_step = True
        self.rl_loss = huber_loss

        self.tde = (self.q_values_action - self.target)

        if optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=adam_epsilon)
            policy_optimizer = tf.train.AdamOptimizer(learning_rate=lr / 10, epsilon=adam_epsilon)
        elif optimizer == 'sgd':
            optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.)
            policy_optimizer = tf.train.MomentumOptimizer(learning_rate=lr / 10, momentum=0.)
        else:
            raise ValueError

        policy_loss = -tf.reduce_mean(self.q_values_policy)

        gradients = policy_optimizer.compute_gradients(policy_loss, var_list=self.nets.online['pol'].variables)
        for i, (grad, var) in enumerate(gradients):
            if grad is not None and param_grad_clip is not None:
                gradients[i] = (tf.clip_by_norm(grad, param_grad_clip), var)
        self.pol_train_op = policy_optimizer.apply_gradients(gradients)

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
            discrete_actions=False,
        )

        self.desired_exploration_epsilon = 0.2
        self.exploration_sigma = 0.2
        self.expl_dist_list = []

        self.exploration_distance_op = tf.sqrt(tf.reduce_mean((self.policy_action - self.perturbed_policy_action) ** 2))

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

        expl_op = self.perturbed_policy_action

        expl, pol, v_value, distance = self.tensorflow_session.run([expl_op, self.policy_action,
                                                                    self.q_values_policy, self.exploration_distance_op],
                                                                   feed_dict={self.observation_input: self._state})

        self.diagnostics['empirical epsilon'].append(distance)

        action = expl if kwargs.get('explore', False) else pol

        self.expl_dist_list.append(distance)
        self.diagnostics['observed_exploration_sigma'].append(distance)

        v = v_value[0]
        try:
            v = v[0]
        except IndexError:
            pass

        for op in ('max', 'min'):
            if len(self.diagnostics[f'predicted Q {op}']) == 0:
                self.diagnostics[f'predicted Q {op}'].append(v)
            else:
                if v > self.diagnostics[f'predicted Q {op}'][0] and op == 'max':
                    self.diagnostics[f'predicted Q {op}'][0] = v
                elif v < self.diagnostics[f'predicted Q {op}'][0] and op == 'min':
                    self.diagnostics[f'predicted Q {op}'][0] = v
        self.diagnostics[f'predicted Q mean'].append(v)

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

        self.tensorflow_session.run([self.srlq_update_op, self.pol_train_op], feed_dict)
        self.tensorflow_session.run(self.frozen_parameter_equalize_ops)

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

    def initialize_tf_variables(self):
        self.tensorflow_session.run(tf.global_variables_initializer())

    def exploration_policy_update(self):
        self.perturbed_policy_weights = self.policy_weights
        self.flat_perturbed_policy_weights = np.random.normal(
            self.flat_perturbed_policy_weights, scale=self.exploration_sigma)

    def update_sigma(self):
        m = self.parameter_exploration_update_speed if np.mean(self.expl_dist_list) < self.exploration_threshold() \
            else 1 / self.parameter_exploration_update_speed
        self.expl_dist_list.clear()
        self.exploration_sigma *= m

    def sample_cma_es(self, start_from, save_cmaes_params=True):
        if self.es is None:
            cma_es_params = self.flat_perturbed_policy_weights
            if 'zero' in start_from:
                cma_es_params.fill(0.)

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

        self.flat_perturbed_policy_weights = self.cma_es_params_to_try.pop()

    def reset_cmaes(self) -> None:
        self.es = None
        self.cma_es_asked = False
        self.tested_cma_es_params.clear()
        self.tested_cma_es_params_fvals.clear()

    def finished_episode_with_score(self, last_reward_sum) -> None:
        if self.es:
            self.tested_cma_es_params.append(self.flat_perturbed_policy_weights)
            self.tested_cma_es_params_fvals.append(-1 * last_reward_sum)

    def _save_es_generation(self, save_cmaes_params):
        r = self.es.result
        if save_cmaes_params:
            np.savez_compressed(f'{self.cmaes_dir}evalutaions_iter_{r.iterations}.npz',
                                fvalues=np.array(self.tested_cma_es_params_fvals),
                                params=np.array(self.tested_cma_es_params))
        else:
            np.savez_compressed(f'{self.cmaes_dir}evalutaions_iter_{r.iterations}.npz',
                                fvalues=np.array(self.tested_cma_es_params_fvals))

    def get_diagnostics_dict(self):
        self.run_once_per_report_diagnostics()
        d = {}
        for k, v in self.diagnostics.items():
            if len(v) > 0:
                d[k] = np.mean(np.array(v))
                v.clear()
        return d

    def run_once_per_report_diagnostics(self):
        self.diagnostics['sigma'].append(self.exploration_sigma)

    def save_params_to_dir(self, save_dir, current=False):
        os.makedirs(save_dir, exist_ok=True)

        for model in self.nets.online.values():
            if any([np.isnan(x).any() for x in model.get_weights()]):
                warnings.warn('Not saving network parameters as NaN values are present')
                return

        for module_name, model in self.nets.online.items():
            print(f'saving to weights to {save_dir}{module_name}/')
            model.save_weights(f'{save_dir}{module_name}/params.ckpt')

        w = self.flat_perturbed_policy_weights
        if current and self.es is not None:
            self.flat_perturbed_policy_weights = self.es.result.xfavorite
            self.exploration_sigma = self.es.sigma
        self.nets.online['pol'].save_weights(f'{save_dir}cmaes_policy/params.ckpt')
        self.flat_perturbed_policy_weights = w

        np.save(f'{save_dir}/exploration_sigma.npy', self.exploration_sigma)

    def load_params_form_dir(self, load_dir):
        try:
            for net in self.nets:
                for module_name, model in net.items():
                    model.load_weights(f'{load_dir}{module_name}/params.ckpt')
            self.exploration_sigma = float(np.load(f'{load_dir}/exploration_sigma.npy'))
        except InvalidArgumentError:  # There used to just be the RL net
            for net in self.nets:
                net['pol'].load_weights(load_dir)

        try:
            self.nets.online['pol'].load_weights(f'{load_dir}cmaes_policy/params.ckpt')
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
        self.exploration_sigma = 0.1
        self.adjust_sigma_to_match_epsilon(cma_es_start_epsilon=cma_es_start_epsilon, max_steps=max_steps)
