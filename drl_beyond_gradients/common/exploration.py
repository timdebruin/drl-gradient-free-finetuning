import random
from typing import Optional

import numpy as np


class ExplorationPolicy(object):
    """Functions to apply exploration to actions. Actions are assumed to be in the range [-1, 1]."""

    def __init__(self) -> None:
        pass

    def apply(self, policy_action: np.ndarray) -> np.ndarray:
        """ Apply the exploration policy to the policy action
        :param policy_action: the action before exploration
        :return: the action with exploration applied
        """
        raise NotImplementedError()


class EpsilonGreedy(ExplorationPolicy):
    """Explore in a temporally uncorrelated way by overwriting the policy action with one of the possible discrete
    actions (when given) or a vector of the same size as the policy action with components sampled uniformly in [-1, 1].
    This second option is for continuous actions."""

    def __init__(self, epsilon: float, repeat: int = 0, discrete_actions: Optional[int] = None) -> None:
        """ Create an epsilon greedy exploration policy
        :param epsilon: the probability of returning a random action rather than the policy action
        :param repeat: when choosing a random action, perform the same action for the next repeat time steps
        :param discrete_actions: if given, the policy is discrete and a number between 0 and _discrete_actions - 1 is
        sampled uniformly at random when exploring. When not given, the policy is continuous and the action domain is
        sampled uniformly.
        """
        super().__init__()
        self.repeat = repeat
        self._repeat_count = repeat
        self._last_action: Optional[np.ndarray] = None
        self._discrete_actions = discrete_actions
        assert 0. <= epsilon <= 1.
        self.epsilon = epsilon

    def apply(self, policy_action: np.ndarray) -> np.ndarray:
        """ Apply the exploration policy to the policy action
        :param policy_action: the action before exploration
        :return: the action with exploration applied
        """
        if self._repeat_count < self.repeat:
            self._repeat_count += 1
            return self._last_action

        if random.random() < self.epsilon / max(self.repeat, 1):
            self._repeat_count = 0
            if not self._discrete_actions:
                action = np.random.uniform(-1, 1, policy_action.size)
            else:
                action = np.array(random.randrange(self._discrete_actions))
            self._last_action = action
            return action
        return policy_action

    def __call__(self, *args, **kwargs):
        policy_action = args[0]
        return self.apply(policy_action)
