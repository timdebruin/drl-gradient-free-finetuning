from typing import Optional
import numpy as np
from baselines.common.atari_wrappers import LazyFrames

from cor_control_benchmarks.control_benchmark import ControlBenchmark

from drl_beyond_gradients.common.experience_buffer import BaseExperienceBuffer


class AbstractPolicy(object):
    """Base class with some shared policy methods to ensure consistency."""

    def __init__(self,
                 environment: Optional[ControlBenchmark], experience_buffer: Optional[BaseExperienceBuffer]) -> None:
        """Initialize the environment by giving it some common components needed for learning
        :param environment: optional benchmark, used for online learning strategies
        :param experience_buffer: optional experience buffer, used for offline learning
        """
        self.experience_buffer = experience_buffer
        self.environment = environment
        self._state: Optional[np.ndarray] = None  # the state for which to calculate the policy action

    def __call__(self, *args, **kwargs):
        """Get the policy action for the given state"""
        assert len(args) == 1, 'Policy expects one unnamed argument: the state.'
        s = args[0]
        if isinstance(s, LazyFrames):
            s = np.array(s)
        assert isinstance(s, np.ndarray), 'Policy expects the state as an numpy ndarray'
        self._state = s[None]
        self.step = kwargs.get('step', -1)

    def train(self, **kwargs):
        """Train the policy"""
        raise NotImplementedError

    def finished_episode_with_score(self, last_reward_sum):
        """Tell the policy that the last episode was finished and what the score was.
        Used for episode based exploration."""
        pass
