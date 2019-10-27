import copy
import random
from typing import NamedTuple, List, Union, Optional

from baselines.common.atari_wrappers import LazyFrames
import numpy as np


class Experience(NamedTuple):
    """Sample of one experience, each component has size [?]"""
    state: Union[np.ndarray, LazyFrames]
    action: np.ndarray
    next_state: Union[np.ndarray, LazyFrames]
    reward: Union[float, np.ndarray]
    terminal: Union[bool, np.ndarray]


class ExperienceBatch(NamedTuple):
    """Contains a batch of experiences of size [B, ?] where B is the batch size,
    or size [T, B, ?] where T is the sequence length. The indices of the buffer are of size
     [B] or size {T, B]."""
    experiences: Experience
    buffer_indices: List[Union[int, List[int]]]


class BaseExperienceBuffer(object):
    """An experience buffer with basic functionality,
    implements uniform sampling of batches of experiences from a FIFO buffer,
    subclasses might implement more advanced behavior."""

    def __init__(self, max_buffer_size: int, normalize_reward_on_first_batch_sample: False) -> None:
        """ Create the experience buffer
        :param max_buffer_size: Maximum number of experiences to store in the buffer"""

        self.reward_norm = None if normalize_reward_on_first_batch_sample else 1.0
        self.max_buffer_size = max_buffer_size
        self._data: List[Experience] = []
        self._buffer_index: int = 0

    def __len__(self):
        return len(self._data)

    def add_experience(self, experience: Experience) -> None:
        """ Add an experience to the buffer"""

        if len(self._data) < self.max_buffer_size:
            self._data.append(experience)
        else:
            self._data[self._buffer_index] = experience
        self._buffer_index = (self._buffer_index + 1) % self.max_buffer_size

    def sample_batch(self, batch_size: int, sequence_length: int = 1) -> ExperienceBatch:
        """ Sample a batch of experiences from the buffer
        :param batch_size: the number of experiences (or sequences of experiences) to return
        :param sequence_length: the number of subsequent experiences to return (when 1, the size of the returned
        experiences is [B, ?], when larger than 1, the size is: [T, B, ?].
        :return: The experience batch.
        """

        if sequence_length > 1:
            raise NotImplementedError
        else:
            return self._batch_with_indices([random.randint(0, len(self._data) - 1) for _ in range(batch_size)])

    def _batch_with_indices(self, indices: Union[List[int], List[List[int]]]) -> Optional[ExperienceBatch]:
        indices_shape = np.array(indices).squeeze().shape

        if len(indices) == 0:
            return None
        if type(indices[0]) == list:
            raise NotImplementedError
        else:
            states, actions, next_states, rewards, terminals = [], [], [], [], []
            if self.reward_norm is None:
                self.determine_reward_norm()
            for index in indices:
                sample = copy.deepcopy(self._data[index]) if isinstance(self._data[index].state, LazyFrames) \
                    else self._data[index]
                # sample = copy.copy(self._data[index])
                # sample = self._data[index]
                states.append(np.array(sample.state, copy=False))
                actions.append(np.array(sample.action, copy=False))
                next_states.append(np.array(sample.next_state, copy=False))
                rewards.append(sample.reward/self.reward_norm)
                terminals.append(sample.terminal)

            return ExperienceBatch(
                buffer_indices=indices,
                experiences=Experience(
                    state=np.array(states),
                    action=np.array(actions),
                    next_state=np.array(next_states),
                    reward=np.array(rewards).reshape(indices_shape),
                    terminal=np.array(terminals).reshape(indices_shape)
                )
            )

    def determine_reward_norm(self):
        """Set the reward normalization to the largest reward observed so far"""
        largest_abs = 0
        for datum in self._data:
            largest_abs = max(np.abs(datum.reward), largest_abs)
        if largest_abs > 0:
            self.reward_norm = largest_abs
        else:
            self.reward_norm = 1
