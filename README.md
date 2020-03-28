# Fine-tuning Deep RL with Gradient-Free Optimization

This code accompanies the paper "Fine-tuning Deep RL with Gradient-Free Optimization" by *Tim de Bruin, Jens Kober, Robert Babuska and karl Tuyls.*

Compared to the code that was used to obtain the experimental results, this version has been refactored and simplified. It should allow for reproducing the results, but has not been extensively tested.

## Installation
The code has been tested with python 3.6+ and tensorflow 1.13. 

First install [tensorflow](https://www.tensorflow.org/install/pip) and [baselines](https://github.com/openai/baselines#installation)
The rest of the dependencies can be installed by running:

```bash
pip install -r requirements.txt
``` 

Still from the base directory, add the directory to the python path:
```bash
export PYTHONPATH=$PYTHONPATH:$PWD
``` 
Now, the code can be run with:
```bash
python drl_beyond_gradients/training_run.py 
```


**known issue**:
The following error might be encountered:
```
/lib/python3.6/site-packages/tensorflow/python/keras/engine/base_layer_utils.py", line 127, in <lambda>
    shape, dtype=dtype, partition_info=partition_info)
TypeError: __call__() got an unexpected keyword argument 'partition_info'
```
This seems to be an error in keras and can be fixed by opening the file (base_layer_utils.py) and changing line 127 to:
```python
init_val = lambda: initializer(  # pylint: disable=g-long-lambda
          shape, dtype=dtype)
```
by removing the partition_info argument.


## Running the experiments from the paper
The experiments use sacred for the settings (and logging, although that has been removed in this simplified version). The available arguments can be seen in [training_run.py](drl_beyond_gradients/training_run.py).

Default arguments can be overwritten (and their values shown)  using the following syntax:

```bash
python drl_beyond_gradients/training_run.py --print_config with 'parameter1=new_value' 'parameter2="new string value"
```

To reproduce the experiments from the paper use (variations on) the following commands:

**CarRacing:**
```bash
python drl_beyond_gradients/training_run.py --print_config with 'cma_es_start_time=2500' 'cma_es_start_exploration=0.25' 'param_grad_clip=10.' 'policy_head=True' 'benchmark="gym_car_racing_discrete"' 'duration=10000' 'sgd_warm_start_duration=200' 'network="DQN"' 'layer_and_batch_norm=True' 'l2_param_penalty=0.01' 'exploration_type="parameter_noise"'  'initial_exploration=0.10' 'final_exploration=0.10' 'plappert_distance=False' 'normalize_reward=False' 'batch_size=32'  'optimizer="adam"' 'adam_epsilon=1e-4' 'learning_rate=1e-4' 'buffer_size=1000000' 'gamma=0.99' 'srl_vf = 1.0' 'srl_ae=0.' 'srl_rp=0' 'srl_fd=0' 'srl_id=0' 'cma_es_start_from="best"'  'eg_action_repeat=0'
```

**Atari:**
```bash
python drl_beyond_gradients/training_run.py --print_config with 'benchmark="atari_enduro"' 'cma_es_start_exploration=0.5' 'param_grad_clip=10.' 'policy_head=True'  'duration=250000000' 'sgd_warm_start_duration=50000' 'network="DQN"' 'layer_and_batch_norm=True' 'l2_param_penalty=0.01' 'exploration_type="parameter_noise"'  'initial_exploration=0.10' 'final_exploration=0.10' 'plappert_distance=False' 'normalize_reward=False' 'batch_size=32'  'optimizer="adam"' 'adam_epsilon=1e-4' 'learning_rate=1e-4' 'buffer_size=1000000' 'gamma=0.99' 'srl_vf = 1.0' 'srl_ae=0.' 'srl_rp=0.' 'srl_fd=0.' 'srl_id=0.' 'cma_es_start_from="best"' 'cma_es_start_time=40000000' 'eg_action_repeat=0'
```
**MagMan:**
```bash
 python drl_beyond_gradients/training_run.py --print_config with 'cma_es_start_time=50' 'cma_es_start_exploration=0.50' 'cma_es_start_from="best"' 'cmaes_save_params=False' 'learning_rate=1e-3' 'benchmark="magman"' 'duration=2000' 'network="DDPG"' 'initial_exploration=0.2' 'final_exploration=0.2' 'sgd_warm_start_duration=1' 'batch_size=64' 'buffer_size=100000' 'gamma=0.95' 'param_grad_clip=10' 'srl_ae=0.' 'srl_rp=0.' 'srl_fd=0.' 'srl_id=0.' 'reward_type="ABSOLUTE"'
```
 
 
 
