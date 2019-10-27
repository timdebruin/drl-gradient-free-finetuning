import copy
import glob
import os
import random
import tempfile
import warnings

import cor_control_benchmarks as cb
import math
import numpy as np
import tensorflow as tf
import tqdm
from baselines.common.atari_wrappers import LazyFrames
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds

from drl_beyond_gradients.common.experience_buffer import BaseExperienceBuffer, Experience
from drl_beyond_gradients.common.exploration import EpsilonGreedy
from drl_beyond_gradients.common.gym_wrapper import DiscreteGymRacingBenchmark, AtariBenchmark
from drl_beyond_gradients.common.helpers import any_in_str
from drl_beyond_gradients.neural_networks.networks import RGBAtariEncoder

ex = Experiment()
ex.captured_out_filter = apply_backspaces_and_linefeeds
# ex.observers.append(MongoObserver.create()) # fill out to collect data on the experiments
# (https://sacred.readthedocs.io/en/latest/observers.html)


@ex.config
def config():
    """Used by sacred to log the important variables that determine the method performance. """
    data_dir = tempfile.gettempdir()  # provide to save results in a persistent location
    # experiment
    name = 'test'

    temp = 0
    while os.path.isdir(f'{data_dir}/{name}/{temp}/'):
        temp += 1
    save_dir = f'{data_dir}/{name}/{temp}/'
    del temp
    cmaes_dir = f'{save_dir}cmaes/'

    # benchmark specifications
    benchmark = 'magman'  # one of {magman, gym_car_racing_discrete, 'atari_enduro', 'atari_freeway'}
    reward_type = 'ABSOLUTE'  # for the magman benchmark: 'ABSOLUTE', 'QUADRATIC' or 'BINARY'
    duration = 2000  # steps for Atari benchmarks episodes for other

    # network specifications
    network = 'DDPG'  # one of {NAF, DQN, DQNF}
    layer_and_batch_norm = True
    l2_param_penalty = 0.00
    policy_head = True

    # exploration
    exploration_type = 'parameter_noise'  # one of {'epsilon_greedy', 'parameter_noise'}
    plappert_distance = False  # use the original distance metric for DQN ( https://arxiv.org/abs/1706.01905 )
    initial_exploration = 0.5  # epsilon for e-greedy
    final_exploration = 0.2  # epsilon for e-greedy
    eg_action_repeat = 4  # when using e-greedy repeat actions for this many time steps (prob of initiating a random
    # action  is reduced to keep the same overall epsilon)

    # sgd training
    sgd_warm_start_duration = 10  # number of time units to complete using a random policy before training the network
    optimizer = 'adam'  # one of {adam, sgd)
    adam_epsilon = 1e-8  # 1e-8 is the default, https://arxiv.org/pdf/1707.06887.pdf recommends 0.01/batch_size
    batch_size = 128
    normalize_reward = True
    steps_between_updates = 4  # for DQN: number of environment steps between gradient updates (=1 for DDPG)
    learning_rate = 1e-4
    buffer_size = int(1e6)
    gamma = 0.99
    param_grad_clip = None

    # srl  weights for different losses that shape the state representation
    srl_vf = 1.0  # value function (RL) loss
    srl_ae = 0.0  # auto-encoding loss
    srl_rp = 0.0  # reward prediction loss
    srl_fd = 0.0  # forward (embedding) dynamics loss
    srl_id = 0.0  # inverse dynamics loss
    srl_sf = 0.0  # slowness loss
    srl_di = 0.0  # diversity loss

    # cma-es training
    cma_es_start_from = 'last'  # one of best, last, zero
    cma_es_start_time = 2500  # steps for Atari benchmarks episodes for other
    cma_es_start_exploration = 0.1  # Initial exploration intensity for CMA-ES
    cma_es_episodes_per_evaluation = 1  # average the undiscounted return over this many rollouts
    cmaes_save_params = True  # optionally disable to reduce memory usage during small scale tests

    # resume run
    load_name = ''  # if not an empty string, load previously saved train state from this directory

    if load_name:
        base_load_dir = random.choice(glob.glob(os.path.expanduser(f"{data_dir}{load_name}/*/")))
    else:
        base_load_dir = None


@ex.capture
def get_environment(benchmark: str, reward_type: str) -> cb.control_benchmark.ControlBenchmark:
    if type(reward_type) == str:
        reward_type = {
            'ABSOLUTE': cb.RewardType.ABSOLUTE,
            'QUADRATIC': cb.RewardType.QUADRATIC,
            'BINARY': cb.RewardType.BINARY
        }[reward_type]

    env_class = {
        'pendulum': lambda reward_type: cb.PendulumBenchmark(
            max_seconds=2., reward_type=reward_type, max_voltage=3,
            state_penalty_weights=np.array([5., 0.01]), action_penalty_weights=np.array([0.1])
        ),
        'magman': lambda reward_type: cb.MagmanBenchmark(
            max_seconds=2.0, reward_type=reward_type,
        ),
        'gym_car_racing_discrete': DiscreteGymRacingBenchmark,
        'atari_enduro': lambda reward_type: AtariBenchmark('EnduroNoFrameskip-v4'),
        'atari_freeway': lambda reward_type: AtariBenchmark('FreewayNoFrameskip-v4'),
    }[benchmark]

    return env_class(reward_type=reward_type)


@ex.capture
def get_action_space_exploration(env, exploration_type, network, initial_exploration, eg_action_repeat):
    if exploration_type == 'parameter_noise':
        try:
            da = env.discrete_action_shape[0]
        except AttributeError:
            da = None
        action_space_exploration = EpsilonGreedy(
            epsilon=0.01, repeat=0, discrete_actions=da)  # as in https://arxiv.org/abs/1706.01905: some eg exploration
        # during parameter noise exploration

    elif exploration_type == 'epsilon_greedy' and network == 'DQN' or network == 'DQNP':
        action_space_exploration = EpsilonGreedy(epsilon=initial_exploration, repeat=eg_action_repeat,
                                                 discrete_actions=env.discrete_action_shape[0])
    elif exploration_type == 'epsilon_greedy' and network == 'NAF':
        action_space_exploration = EpsilonGreedy(epsilon=initial_exploration, repeat=eg_action_repeat,
                                                 discrete_actions=None)
    else:
        raise NotImplementedError
    return action_space_exploration


@ex.capture
def get_policy(env, experience_buffer, sess,
               network, benchmark, cmaes_dir,
               l2_param_penalty, policy_head, gamma,
               srl_vf, srl_ae, srl_rp, srl_fd, srl_id, srl_sf, srl_di,
               plappert_distance, batch_size, learning_rate,
               optimizer, param_grad_clip
               ):
    if 'DDPG' in network:
        from drl_beyond_gradients.neural_networks.ddpg_policy import NNPolicy
    elif 'DQN' in network:
        from drl_beyond_gradients.neural_networks.dqn_policy import NNPolicy
    else:
        raise ValueError(f'network architecture unkown: {network}')

    if any_in_str(['gym_car_racing', 'atari'], benchmark):
        warnings.simplefilter("ignore", DeprecationWarning)  # Car racing deprecation spam
        state_encoder_class = lambda: RGBAtariEncoder(env.state_shape, l2_param_penalty=l2_param_penalty)
    else:
        state_encoder_class = None

    return NNPolicy(environment=env, experience_buffer=experience_buffer, tensorflow_session=sess,
                    network=network, l2_param_penalty=l2_param_penalty, policy_head=policy_head,
                    optimizer=optimizer, param_grad_clip=param_grad_clip,
                    lr=learning_rate, gamma=gamma, batch_size=batch_size,
                    use_plappert_distance=plappert_distance,
                    state_encoder_class=state_encoder_class,
                    srl_vf=srl_vf, srl_ae=srl_ae, srl_rp=srl_rp, srl_fd=srl_fd, srl_id=srl_id, srl_sf=srl_sf,
                    srl_di=srl_di,
                    cmaes_dir=cmaes_dir)


@ex.automain
def my_main(
        save_dir, cmaes_dir,
        benchmark, duration,
        network,
        exploration_type, initial_exploration, final_exploration,
        steps_between_updates, buffer_size, gamma, sgd_warm_start_duration,
        normalize_reward,
        cma_es_start_time, cma_es_start_from,
        cmaes_save_params,
        base_load_dir, cma_es_start_exploration, cma_es_episodes_per_evaluation):
    """
    Main experiment function. This function is called and the parameters are automatically filled out by
    sacred from the config above, when running python training_run.py  to update parameter values use:
    python training_run.py with 'parameter=value'
    """

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(cmaes_dir, exist_ok=True)

    env = get_environment()
    experience_buffer = BaseExperienceBuffer(buffer_size, normalize_reward_on_first_batch_sample=normalize_reward)
    diagnostics = cb.Diagnostics(benchmark=env, log=cb.LogType.REWARD_SUM, gamma=gamma)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    policy = get_policy(env, experience_buffer, sess)
    action_space_exploration = get_action_space_exploration(env)

    time_unit_name = 'step' if 'atari' in benchmark else 'episode'
    start_time = 0 if base_load_dir is None else cma_es_start_time
    episode = -1 if time_unit_name == 'step' else start_time
    step = start_time if time_unit_name == 'step' else 0
    time_unit = step if time_unit_name == 'step' else episode

    current_cmaes_eval_scores = []
    best_cmaes_score = 0

    def increment_episode():
        nonlocal episode, time_unit
        episode += 1
        if time_unit_name == 'episode':
            time_unit = episode
            progress_bar.update()

    def increment_step():
        nonlocal step, time_unit
        step += 1
        if time_unit_name == 'step':
            time_unit = step
            progress_bar.update()

    def update_exploration():
        if 'atari' in benchmark:
            current_exploration_intensity = max(0.1, 1 - step / (1e6 / 0.9))
            policy.desired_exploration_epsilon = current_exploration_intensity
        else:
            current_exploration_intensity = (1 - (time_unit / duration)) * initial_exploration + (
                    time_unit / duration) * final_exploration
        if time_unit < cma_es_start_time:
            if exploration_type == 'epsilon_greedy':
                action_space_exploration.epsilon = current_exploration_intensity
            else:
                if time_unit > sgd_warm_start_duration:
                    policy.adjust_sigma_to_match_epsilon(
                        current_exploration_intensity, max_steps=1 if network == 'DDPG' else 100)
                    # short magman episodes, longer episodes for other benchmarks

        if time_unit < cma_es_start_time:
            if exploration_type == 'parameter_noise':
                policy.exploration_policy_update()
        else:
            if len(current_cmaes_eval_scores) == 0:
                policy.sample_cma_es(cma_es_start_from, save_cmaes_params=cmaes_save_params)

        return exploration_type == 'parameter_noise' or time_unit >= cma_es_start_time

    def start_cmaes():
        if 'best' in cma_es_start_from:
            policy.load_params_form_dir(save_dir)
            # asuming the best result was obtained during exploration
            policy.policy_weights = policy.perturbed_policy_weights

        print(f'Updating sigma to approximate epsilon = {cma_es_start_exploration}')
        print(f'Ensuring sufficient experiences are in the buffer')
        stm = 1e5 if policy.separate_cmaes_policy else 1e4
        t = True
        while len(experience_buffer._data) < min(experience_buffer.max_buffer_size, stm):
            s = env.reset() if t else sp
            a = action_space_exploration(policy(s, explore=True, cmaes=False))
            sp, r, t, _ = env.step(a)
            experience_buffer.add_experience(Experience(state=s, action=a, next_state=sp, reward=r, terminal=t))

        if policy.separate_cmaes_policy:
            print('pre-training cma-es policy')
            policy.pre_train_cmaes()

        if 'zero' in cma_es_start_from:
            if 'DDPG' in network:
                policy.exploration_sigma = cma_es_start_exploration
        else:
            print('Adjusting sigma')
            policy.exploration_sigma = 0.1
            policy.cmaes_init_sigma(cma_es_start_exploration)
        nonlocal cma_es_started
        cma_es_started = True
        action_space_exploration.epsilon = 0

    progress_bar = tqdm.tqdm(range(start_time, duration),
                             unit=time_unit_name,
                             smoothing=0.1, mininterval=1.)
    cma_es_started = False

    def fully_random_action():
        try:
            action = random.randint(0, env.discrete_action_shape[0] - 1)
        except AttributeError:
            action = np.random.uniform(-1, 1, env.action_shape[0])
        return action

    def possibly_restore_policy():
        if base_load_dir is not None:
            avail_start_eps = [int(x.split('/')[-2].split('_')[-1]) for x in glob.glob(f'{base_load_dir}before*/')]
            load_dir = f'{base_load_dir}before_time_{max([x for x in avail_start_eps if x <= cma_es_start_time])}/'
            print(f'Loading previous network from: {load_dir}')
            policy.load_params_form_dir(load_dir)
            policy.save_params_to_dir(save_dir)

    with sess, progress_bar:
        possibly_restore_policy()
        while not (time_unit > duration):
            increment_episode()

            if time_unit >= cma_es_start_time and time_unit > 0 and not cma_es_started:
                start_cmaes()
            param_explore = update_exploration()

            terminal = False
            state = env.reset()
            while not terminal:
                increment_step()

                s = copy.deepcopy(state) if (
                        isinstance(state, LazyFrames) and time_unit < cma_es_start_time) else state
                if time_unit < sgd_warm_start_duration:
                    action = fully_random_action()
                else:
                    action = action_space_exploration(policy(s, explore=param_explore,
                                                             step=step, cmaes=time_unit > cma_es_start_time))

                next_state, reward, terminal, _ = env.step(action)

                if time_unit < cma_es_start_time:
                    experience_buffer.add_experience(Experience(
                        state=state, action=action, next_state=next_state, reward=reward,
                        terminal=True if terminal and not env.max_steps_passed else False
                    ))

                state = next_state

                if sgd_warm_start_duration < time_unit < cma_es_start_time and step % steps_between_updates == 0:
                    policy.train(min_return=diagnostics.lowest_observed_return,
                                 max_return=diagnostics.highest_observed_return)

            with open(f'{save_dir}scores.csv', 'a+') as f:
                f.write(f'{step}, {episode}, {diagnostics.last_reward_sum}\n')

            if time_unit >= cma_es_start_time:
                current_cmaes_eval_scores.append(diagnostics.last_reward_sum)
                if len(current_cmaes_eval_scores) == cma_es_episodes_per_evaluation:
                    policy.finished_episode_with_score(np.mean(current_cmaes_eval_scores))
                    current_cmaes_eval_scores.clear()

            progress_bar.set_postfix({
                'best reward sum': diagnostics.best_reward_sum,
                'last_reward_sum': f'{diagnostics.last_reward_sum:4.3f}'
            })

            if diagnostics.last_episode_was_best:
                policy.save_params_to_dir(f'{save_dir}')
                if time_unit_name == 'step':
                    epk = 1000000 * math.ceil(time_unit / 1e6)
                else:
                    epk = 500 * math.ceil(episode / 500)
                policy.save_params_to_dir(f'{save_dir}before_time_{epk}/')

    return diagnostics.best_reward_sum
