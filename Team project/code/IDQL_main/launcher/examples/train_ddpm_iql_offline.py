import os

import numpy as np
from absl import app, flags

import sys
sys.path.append('/content/gdrive/MyDrive/IDQL_main/examples/states')

from train_diffusion_offline import call_main
from launcher.hyperparameters import set_hyperparameters


FLAGS = flags.FLAGS
flags.DEFINE_integer('variant', 0, 'Logging interval.')


def main(_):
    constant_parameters = dict(project='offline_schedule_final',
                               experiment_name='ddpm_iql',
                               max_steps=100001, #Actor takes two steps per critic step
                               batch_size=512,
                               eval_episodes=50,
                               log_interval=100,
                               eval_interval=300000,
                               save_video = False,
                               filter_threshold=None,
                               take_top=None,
                               online_max_steps = 0,
                               unsquash_actions=False,
                               normalize_returns=True,
                               training_time_inference_params=dict(
                                N = 64,
                                clip_sampler = True,
                                M = 0,),
                               rl_config=dict(
                                   model_cls='DDPMIQLLearner',
                                   actor_lr=3e-4,
                                   critic_lr=3e-4,
                                   value_lr=3e-4,
                                   T=5,
                                   N=64,
                                   M=0,
                                   actor_dropout_rate=0.1,
                                   actor_num_blocks=3,
                                   decay_steps=int(3e6),
                                   actor_layer_norm=True,
                                   value_layer_norm=True,
                                   actor_tau=0.001,
                                   beta_schedule='vp',
                               ))

    sweep_parameters = dict(
                            seed=list(range(10)),
                            env_name=['antmaze-umaze'],
                            # env_name=['walker2d-medium-v2', 'walker2d-medium-replay-v2', 'walker2d-medium-expert-v2',  
                            # 'halfcheetah-medium-v2', 'halfcheetah-medium-replay-v2', 'halfcheetah-medium-expert-v2',
                            # 'hopper-medium-v2', 'hopper-medium-replay-v2', 'hopper-medium-expert-v2', 
                            # 'antmaze-umaze-v2', 'antmaze-umaze-diverse-v2', 'antmaze-medium-diverse-v2', 
                            # 'antmaze-medium-play-v2', 'antmaze-large-diverse-v2', 'antmaze-large-play-v2',],
                            )
    

    variants = [constant_parameters]
    name_keys = ['experiment_name', 'env_name']
    variants = set_hyperparameters(sweep_parameters, variants, name_keys)

    inference_sweep_parameters = dict(
                            N = [16, 64, 256],
                            clip_sampler = [True], 
                            M = [0],
                            )
    
    inference_variants = [{}]
    name_keys = []
    inference_variants = set_hyperparameters(inference_sweep_parameters, inference_variants)

    filtered_variants = []
    for variant in variants:
        if 'T' not in variant:
            variant['T'] = 1000
        
        variant['rl_config']['T'] = variant.get('T', None)
        variant['rl_config']['beta_schedule'] = variant.get('beta_schedule', 'vp') # 设置默认的 beta_schedule 为 'vp'
        variant['inference_variants'] = inference_variants
            
        if 'antmaze' in variant['env_name']:
            variant['rl_config']['critic_hyperparam'] = 0.9
        else:
            variant['rl_config']['critic_hyperparam'] = 0.7

        filtered_variants.append(variant)


    # print(len(filtered_variants))
    variant = filtered_variants[FLAGS.variant]
    # print(FLAGS.variant)
    call_main(variant)


if __name__ == '__main__':
    app.run(main)