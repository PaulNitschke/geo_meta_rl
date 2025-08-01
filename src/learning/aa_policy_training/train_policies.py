import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env

from src.learning.aa_policy_training.utils import GarageToGymWrapper

# Training setup

def train_and_save_pi_and_buffer(task_env, 
                          save_dir,
                          seed:int,
                          n_envs:int,
                          N_steps:int,
                          batch_size:int=182):
    """Trains a policy in a Garage task_env via SAC and saves the policy and the replay buffer."""
    task_0_gym=GarageToGymWrapper(task_env=task_env)

    vec_env = make_vec_env(lambda: task_0_gym, n_envs=n_envs, seed=seed)
    pi = SAC("MlpPolicy", vec_env, verbose=1, seed=seed, batch_size=batch_size)
    pi.learn(total_timesteps=N_steps)
    
    pi.save(save_dir+"/policy")
    pi.save_replay_buffer(save_dir+ "/replay_buffer")

def train_and_save_pis_and_buffers(tasks: list,
                   save_dir:str,
                   args)->list:
    """Trains policies for each task in the list and saves them along with their replay buffers. Returns the directions of the tasks."""
    dirs=[]
    for idx_task, task in enumerate(tasks):
        train_and_save_pi_and_buffer(task_env=task, 
                            save_dir=f"{save_dir}/task_{idx_task}", 
                            seed=args.seed, 
                            n_envs=args.n_envs, 
                            N_steps=args.n_steps_train_pis)
        dirs.append(f"{save_dir}/task_{idx_task}")
    return dirs