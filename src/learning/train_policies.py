from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env

from src.utils import GarageToGymWrapper

# Training setup

def train_and_save_policy(task_env, 
                          task_name,
                          seed:int,
                          n_envs:int,
                          N_steps:int,
                          batch_size:int=182):
    """Trains a policy in a Garage task_env via SAC and saves the policy and the replay buffer."""
    task_0_gym=GarageToGymWrapper(task_env=task_env)

    vec_env = make_vec_env(lambda: task_0_gym, n_envs=n_envs, seed=seed)
    pi = SAC("MlpPolicy", vec_env, verbose=1, seed=seed, batch_size=batch_size)
    pi.learn(total_timesteps=N_steps)
    
    pi.save(task_name)
    pi.save_replay_buffer(task_name+ "_replay_buffer")