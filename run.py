from stable_baselines3 import PPO, SAC, TD3
import tqdm as tqdm
from stable_baselines3.common.callbacks import EvalCallback
from sme_rl.sme import SME 

algorithm = PPO
total_timesteps = 1_000_000
eval_every = 10_000
seed = 0

n_state_channel = 8
n_action_channel = 4
episode_length = 100
reward_every = 1
min_reward = 0.
policy_complexity = 1
kill_threshold = 0.


env = SME(
    n_state_channels=n_state_channel,
    n_action_channels=n_action_channel,
    episode_length=episode_length,
    reward_every=reward_every,
    min_reward=min_reward,
    kill_threshold=kill_threshold,
    policy_complexity=policy_complexity,
    seed=seed
)

eval_env = SME(
    n_state_channels=n_state_channel,
    n_action_channels=n_action_channel,
    episode_length=episode_length,
    reward_every=reward_every,
    min_reward=min_reward,
    kill_threshold=kill_threshold,
    policy_complexity=policy_complexity,
    seed=seed
)

eval_callback = EvalCallback(
    eval_env, 
    eval_freq=eval_every,
    deterministic=True, 
    render=False
)

model = algorithm(
    "MlpPolicy",
    env,
    verbose=0,
    seed=seed
)

model.learn(
    total_timesteps=total_timesteps,
    callback=eval_callback,
    progress_bar=True
)

eval_env.eval(model, channel_min=-1., channel_max=2., wd_weight=0.5, probes=10_000)
