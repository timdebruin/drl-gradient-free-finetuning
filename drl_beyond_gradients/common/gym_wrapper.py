import warnings
from typing import Tuple, Any

import baselines
from cv2 import resize

import numpy as np
import cor_control_benchmarks as cb
import gym

from baselines.common.atari_wrappers import wrap_deepmind, make_atari


def shift_and_scale_from_box(box):
    # normalized = (vector - shift) / scale, de - normalized = vector * scale + shift
    scale = (box.high - box.low) / 2
    shift = (box.high + box.low) / 2
    return shift, scale


class GymRacingBenchmark(cb.control_benchmark.ControlBenchmark):
    """Wrapper for the gym benchmark to make it consistent with the ControlBenchmark interface"""

    def __init__(self, sampling_time: float = None, max_seconds: float = None,
                 do_not_normalize: bool = False, **kwargs):
        warnings.warn(f'The following arguments are ignored: {kwargs}')

        self.max_seconds = max_seconds
        self.do_not_normalize = do_not_normalize
        self.warned_that_state_is_not_normalized = False
        self._name = 'CarRacing-v0'
        self._env = gym.make(self._name)

        # fix annoying OpenAI bug
        self._env.reset()
        self._env.render()

        a_shift, a_scale = shift_and_scale_from_box(self._env.action_space)
        s_shift, s_scale = shift_and_scale_from_box(self._env.observation_space)
        s_shift = s_shift[:self.state_shape[0], :self.state_shape[1], :self.state_shape[2]]
        s_scale = s_scale[:self.state_shape[0], :self.state_shape[1], :self.state_shape[2]]

        fps = 50 if sampling_time is None else int(1 / sampling_time)
        if fps != 50:
            warnings.warn(f"CarRacing was designed for 50 FPS, changing that to {fps}", UserWarning)
            self._env.env.FPS = fps
        ms = max_seconds
        if not max_seconds:
            ms = self._env._max_episode_steps / fps
        if max_seconds and max_seconds * fps > self._env._max_episode_steps:
            ms = self._env._max_episode_steps / fps
            warnings.warn(
                f"CarRacing uses at most {self._env._max_episode_steps} steps, which means max_seconds will be "
                f"reduced from {max_seconds} to {ms}.", UserWarning)

        super().__init__(state_names=['2d pixel bank' for _ in range(self.state_shape[0])],
                         action_names=['steering', 'accelerator', 'break'],
                         state_shift=s_shift,
                         state_scale=s_scale,
                         action_shift=a_shift,
                         action_scale=a_scale,
                         initial_states=None,
                         sampling_time=1 / fps,
                         max_seconds=ms,
                         target_state=np.zeros((self.state_shape[0],)),
                         target_action=np.zeros((self.action_shape[0],)),
                         state_penalty_weights=np.zeros((self.state_shape[0],)),
                         action_penalty_weights=np.zeros((self.action_shape[0],)),
                         binary_reward_state_tolerance=np.zeros((self.state_shape[0],)),
                         binary_reward_action_tolerance=np.zeros((self.action_shape[0],)),
                         domain_bound_handling=[cb.DomainBound.IGNORE for _ in range(self.state_shape[0])],
                         reward_type=cb.RewardType.BINARY,
                         do_not_normalize=do_not_normalize)
        self.step_counter = 0
        self._u = None
        self._state, self._reward, self._terminal, self._info = None, None, None, None

    def reset(self) -> np.ndarray:
        """Reset the environment to one of the initial states.
        :return the initial state after reset"""
        self.step_counter = 0
        self._reset_log()
        self._state = self._env.reset()
        return self.state

    def reset_to_specific_state(self, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Any]:
        """ Take a step from the current state using the specified action. Action is assumed to be in [-1,1] unless
        the environment was created with do_not_normalize = True.
        :param action: The action to take in the current state for sample_time seconds
        :return: A tuple with (next_state, reward, terminal, additional_info)"""
        if self.do_not_normalize:
            action = self.normalize_action(np.array(action))
            assert -1 <= action.all() <= 1, 'Action was out of the allowed range'
        else:
            action = np.array(action)
            assert -1 <= action.all() <= 1, 'Actions should be normalized between -1 and 1'
        self._u = self.denormalize_action(action)

        self._step_log_pre()

        with warnings.catch_warnings():
            self._state, self._reward, self._terminal, self._info = self._env.step(action)
        self.step_counter += 1

        self._step_log_post(self._reward, self._terminal)

        return self.state, self.reward, self._terminal, self._info

    @property
    def name(self) -> str:
        """Return an identifier that describes the benchmark for fair comparisons."""
        return self._name

    @property
    def action_shape(self):
        return self._env.action_space.shape

    @property
    def state_shape(self):
        # rgbs = self._env.observation_space.shape
        # print(*rgbs)
        return (84, 84, 1)  # black and white

    @property
    def not_normalized_state_domain(self):
        return {
            'min': self._env.observation_space.low,
            'max': self._env.observation_space.high,
        }

    @property
    def not_normalized_action_domain(self):
        return {
            'min': self._env.action_space.low,
            'max': self._env.action_space.high,
        }

    @property
    def state(self) -> np.ndarray:
        """Return either the normalized state or the true state, depending on whether normalization is used"""
        if not self.warned_that_state_is_not_normalized:
            warnings.warn('The RGB uint8 state is not normalized by default to save memory, consider normalizing it '
                          'later. Explicitly calling normalized_state (instead of state) will return the '
                          'normalized state')
            self.warned_that_state_is_not_normalized = True

        rs = resize(
            np.mean(self.true_state, axis=-1, keepdims=True).astype(np.uint8),
            (84, 84)).reshape((84, 84, 1))
        return rs

    @property
    def reward(self) -> float:
        """Obtain the reward based on the current state and the action that resulted in the transition to that state."""
        return self._reward

    @property
    def max_steps_passed(self):
        """Returns True if the maximum episode length of the benchmark has expired."""
        return self.step_counter * self.sampling_time >= self.max_seconds

    def _f(self):
        """Calculate the state at the next time step based on the equations of motion.
        Uses two steps of the Runge–Kutta method (https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods)."""
        raise ReferenceError('_f not used for gym environments')

    def _eom(self, state_action: np.ndarray) -> np.ndarray:
        """Calculates the benchmark specific equations of motion for the state-action vector."""
        raise ReferenceError('eom not used for gym environments')

    def _state_bounds_check(self) -> bool:
        """Check whether the state has exited the normalized domain,
        correct the state based on the strategy per state component given in domain_bound_handling,
        return whether the domain violation means that the episode should be terminated."""
        raise ReferenceError('state bounds check not used for gym environments')

    def _derivative_dimension(self, state_dimension: int) -> int:
        """ Return the index in the state vector of the derivative of the state_dimension index,
        return -1 if the derivative of the given state component is not in the state vector.
        :param state_dimension: the index in the state of the component that the derivative should be of
        :return: the index of the state vector component that contains the derivative, or -1 if the
        derivative is not in the state vector"""
        raise ReferenceError('_derivative_dimension not used for gym environments')


class DiscreteGymRacingBenchmark(GymRacingBenchmark):

    def __init__(self, **kwargs):
        self.action_maps = [  # steer, gas, brake
            np.array([0., .5, 0.]),
            np.array([-1., 0.2, 0.]),
            np.array([-.5, .5, 0.]),
            np.array([.5, .5, 0.]),
            np.array([1., 0.2, 0.]),
            np.array([0., .8, 0.]),
            np.array([0., 0., .8]),
        ]
        super().__init__(**kwargs)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Any]:
        action = int(action)
        action = self.action_maps[action]
        return super().step(action)

    @property
    def discrete_action_shape(self):
        return tuple([len(self.action_maps)])


class DiscretePendulumBenchmark(cb.PendulumBenchmark):

    def __init__(self, **kwargs):
        self.action_maps = [
            np.array([-1.]),
            np.array([1.0]),
            np.array([0.]),
        ]
        super().__init__(**kwargs)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Any]:
        action = int(action)
        action = self.action_maps[action]
        return super().step(action)

    @property
    def discrete_action_shape(self):
        return tuple([len(self.action_maps)])

    @property
    def name(self):
        return f'Discretized: {super().name}'


class GymBenchmark(cb.control_benchmark.ControlBenchmark):
    """Wrapper for the gym benchmark to make it consistent with the ControlBenchmark interface"""

    def __init__(self, name: str, sampling_time: float = None, max_seconds: float = None,
                 do_not_normalize: bool = False, **kwargs):
        warnings.warn(f'The following arguments are ignored: {kwargs}')

        self.max_seconds = max_seconds
        self.do_not_normalize = do_not_normalize
        self.warned_that_state_is_not_normalized = False
        self._name = name
        if not (hasattr(self, '_env') and self._env is not None):
            self._env = gym.make(self._name)

        try:
            a_shift, a_scale = shift_and_scale_from_box(self._env.action_space)
            a_ref = np.zeros((self.action_shape[0],))
        except AttributeError:
            a_shift = [0]
            a_scale = [0]
            a_ref = np.zeros(1)
        s_shift, s_scale = shift_and_scale_from_box(self._env.observation_space)

        super().__init__(state_names=['observation' for _ in range(self.state_shape[0])],
                         action_names=['steering', 'accelerator', 'break'],
                         state_shift=s_shift,
                         state_scale=s_scale,
                         action_shift=a_shift,
                         action_scale=a_scale,
                         initial_states=None,
                         sampling_time=sampling_time,
                         max_seconds=max_seconds,
                         target_state=np.zeros((self.state_shape[0],)),
                         target_action=a_ref,
                         state_penalty_weights=np.zeros((self.state_shape[0],)),
                         action_penalty_weights=a_ref,
                         binary_reward_state_tolerance=np.zeros((self.state_shape[0],)),
                         binary_reward_action_tolerance=a_ref,
                         domain_bound_handling=[cb.DomainBound.IGNORE for _ in range(self.state_shape[0])],
                         reward_type=cb.RewardType.BINARY,
                         do_not_normalize=do_not_normalize)
        self.step_counter = 0
        self._u = None
        self._state, self._reward, self._terminal, self._info = None, None, None, None

    def reset(self) -> np.ndarray:
        """Reset the environment to one of the initial states.
        :return the initial state after reset"""
        self.step_counter = 0
        self._reset_log()
        self._state = self._env.reset()
        return self.state

    def reset_to_specific_state(self, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Any]:
        """ Take a step from the current state using the specified action. Action is assumed to be in [-1,1] unless
        the environment was created with do_not_normalize = True.
        :param action: The action to take in the current state for sample_time seconds
        :return: A tuple with (next_state, reward, terminal, additional_info)"""
        if not isinstance(action, int):
            if self.do_not_normalize:
                action = self.normalize_action(np.array(action))
                assert -1 <= action.all() <= 1, 'Action was out of the allowed range'
            else:
                action = np.array(action)
                assert -1 <= action.all() <= 1, 'Actions should be normalized between -1 and 1'
            self._u = self.denormalize_action(action)
        else:
            self._u = action

        self._step_log_pre()

        with warnings.catch_warnings():
            self._state, self._reward, self._terminal, self._info = self._env.step(action)
        self.step_counter += 1

        self._step_log_post(self._reward, self._terminal)

        return self.state, self.reward, self._terminal, self._info

    @property
    def name(self) -> str:
        """Return an identifier that describes the benchmark for fair comparisons."""
        return self._name

    @property
    def action_shape(self):
        return self._env.action_space.shape

    @property
    def state_shape(self):
        return self._env.observation_space.shape

    @property
    def not_normalized_state_domain(self):
        return {
            'min': self._env.observation_space.low,
            'max': self._env.observation_space.high,
        }

    @property
    def not_normalized_action_domain(self):
        return {
            'min': self._env.action_space.low,
            'max': self._env.action_space.high,
        }

    @property
    def reward(self) -> float:
        """Obtain the reward based on the current state and the action that resulted in the transition to that state."""
        return self._reward

    @property
    def max_steps_passed(self):
        """Returns True if the maximum episode length of the benchmark has expired."""
        return self.step_counter * self.sampling_time >= self.max_seconds

    def _f(self):
        """Calculate the state at the next time step based on the equations of motion.
        Uses two steps of the Runge–Kutta method (https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods)."""
        raise ReferenceError('_f not used for gym environments')

    def _eom(self, state_action: np.ndarray) -> np.ndarray:
        """Calculates the benchmark specific equations of motion for the state-action vector."""
        raise ReferenceError('eom not used for gym environments')

    def _state_bounds_check(self) -> bool:
        """Check whether the state has exited the normalized domain,
        correct the state based on the strategy per state component given in domain_bound_handling,
        return whether the domain violation means that the episode should be terminated."""
        raise ReferenceError('state bounds check not used for gym environments')

    def _derivative_dimension(self, state_dimension: int) -> int:
        """ Return the index in the state vector of the derivative of the state_dimension index,
        return -1 if the derivative of the given state component is not in the state vector.
        :param state_dimension: the index in the state of the component that the derivative should be of
        :return: the index of the state vector component that contains the derivative, or -1 if the
        derivative is not in the state vector"""
        raise ReferenceError('_derivative_dimension not used for gym environments')


class AtariBenchmark(GymBenchmark):

    def __init__(self, name: str, **kwargs):
        self._raw_env = make_atari(name)
        self._env = wrap_deepmind(self._raw_env, episode_life=True, clip_rewards=True, frame_stack=True, scale=False)
        super().__init__(name, do_not_normalize=True, **kwargs)

    def reset(self):
        self.step_counter = 0
        self._reset_log()
        self._state = self._env.reset()
        return self._state

    def step(self, action: np.ndarray):
        self._u = action

        # self._step_log_pre()

        self._state, self._reward, self._terminal, self._info = self._env.step(action)
        self.step_counter += 1

        self._step_log_post(self._reward, self._terminal)

        return self._state, self._reward, self._terminal, self._info

    @property
    def true_state(self):
        return None

    @property
    def discrete_action_shape(self):
        return tuple([self._env.action_space.n])

    @property
    def max_steps_passed(self):
        return False
