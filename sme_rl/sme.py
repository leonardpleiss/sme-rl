import gymnasium
from gymnasium import spaces
import numpy as np
import torch as th
from pathlib import Path
from .utils import DeepUniformNetwork, WaveTransition, seed_everything, save_npz, get_eval_samples

class SME(gymnasium.Env):
    def __init__(
            self,
            n_state_channels: int,
            n_action_channels: int,
            episode_length: int = 100,
            reward_every: int = 1,
            min_reward: float = 0.,
            kill_threshold: float = 0.,
            policy_complexity: int = 1,
            seed: int = 1,
            same_initial_state: bool = False,
            render_mode = None
        ):
        
        seed_everything(seed)

        super().__init__()
        
        assert isinstance(n_state_channels, int)
        assert isinstance(n_action_channels, int)
        assert isinstance(episode_length, int)

        assert n_state_channels > 0
        assert n_action_channels > 0
        assert episode_length > 0

        self.n_state_channels = n_state_channels
        self.n_action_channels = n_action_channels
        self.episode_length = episode_length
        self.render_mode = render_mode
        self.reward_every = reward_every
        self.min_reward = min_reward
        self.kill_threshold = kill_threshold

        self.step_count = 0
        self.cum_reward = 0.
        self.n_state_channels_wrapped = n_state_channels + 2 # state channels + stepcount
        self.same_initial_state = same_initial_state

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1, self.n_state_channels_wrapped),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1, self.n_action_channels),
            dtype=np.float32
        )
        
        self.initial_state_raw = th.rand(
            size=(1, self.n_state_channels),
            dtype=th.float32
        )
        
        self.initial_state = th.cat([
            self.initial_state_raw,
            th.tensor([0.0, 0.0], dtype=th.float32).unsqueeze(0)
        ], dim=1)

        self.optimal_state_raw = self.initial_state_raw.clone()
        self.optimal_state = self.initial_state.clone()
        
        self.transition_fct = WaveTransition(
            state_dim=n_state_channels,
            action_dim=n_action_channels,
        )
        
        self.optimal_policy = DeepUniformNetwork(
            n_in=n_state_channels,
            m_out=n_action_channels,
            complexity=policy_complexity,
        )

        config = {
            "n_state_channels": n_state_channels,
            "n_action_channels": n_action_channels,
            "episode_length": episode_length,
            "reward_every": reward_every,
            "min_reward": min_reward,
            "kill_threshold": kill_threshold,
            "policy_complexity": policy_complexity,
            "seed": seed,
            "same_initial_state": same_initial_state,
        }

        print("\nInitialized with:")
        for key, value in config.items():
            print(f"  - {key:<25}: {value}")


    def reset(self, *, seed=None, options=None):

        super().reset(seed=seed)

        if self.same_initial_state:
            self.state = self.initial_state.clone()
            self.state_raw = self.initial_state_raw.clone()

        else:
            self.state_raw = th.rand(
                size=(1, self.n_state_channels),
                dtype=th.float32
            )
            
            self.state = th.cat([
                self.state_raw,
                th.tensor([0.0, 0.0], dtype=th.float32).unsqueeze(0)
            ], dim=1)

        self.optimal_state_raw = self.state_raw.clone()
        self.optimal_state = self.state.clone()

        self.step_count = 0
        self.cum_reward = 0.

        info = {}

        return self.state.cpu().numpy(), info

    def step(self, action: np.ndarray):

        action_tensor = th.from_numpy(action).float()
        
        with th.no_grad():

            optimal_action = self.optimal_policy(self.state_raw)

            # Calculate next states
            self.step_count += 1
            norm_step_count = self.step_count / self.episode_length
            
            current_state = self.state.clone() 
            self.state, self.state_raw = self._wrapped_transition(current_state, action_tensor, norm_step_count)
            self.optimal_state, self.optimal_state_raw = self._wrapped_transition(current_state, optimal_action, norm_step_count)

            raw_reward = 1 - (th.abs(action_tensor - optimal_action).mean(axis=1))
            raw_reward = th.clamp((raw_reward - 0.75) * 4.0, min=0.0)
                  
            step_reward = (raw_reward > (self.min_reward)) * raw_reward

            self.cum_reward += step_reward

        # Termination conditions
        terminated = step_reward < self.kill_threshold
        truncated = self.step_count >= self.episode_length
        
        # Get final reward
        if (self.step_count % self.reward_every == 0) or truncated:
            reward = self.cum_reward
            self.cum_reward = 0.

        else:
            reward = 0.0

        info = {}

        return self.state.detach().cpu().numpy(), reward, terminated, truncated, info

    def _wrapped_transition(self, state, action, norm_step_count):

        norm_step_count = th.tensor([norm_step_count], dtype=th.float32).unsqueeze(1)
        norm_cum_reward = th.tensor([self.cum_reward / self.reward_every], dtype=th.float32).unsqueeze(1)

        raw_state = state[:, :-2].float() # Strip step_counter and cum_rewards

        new_state_raw = self.transition_fct(raw_state, action)

        new_state =  th.cat([
            new_state_raw,
            norm_step_count,
            norm_cum_reward,
        ], dim=1)

        return new_state, new_state_raw 

    def render(self):
        if self.render_mode == "human":
            print(f"State: {self.state.tolist()}")

    def eval(
            self,
            model,
            storepath=None,
            channel_min=0.,
            channel_max=1.,
            probes=10000,
            wd_weight=.5,
            interface="sb3"
        ):
        
        with th.no_grad():

            states = get_eval_samples(n_samples=probes, n_dims=self.n_state_channels, min_val=channel_min, max_val=channel_max, wd_weight=wd_weight)

            zeros = np.zeros((probes, 2), dtype=states.dtype)
            padded_states_np = np.concatenate([states, zeros], axis=1)
            
            opt_a = self.optimal_policy(th.tensor(states).float())

            a = np.array([model.predict(obs.reshape(1, -1), deterministic=True)[0] for obs in padded_states_np])
            a = a.reshape(probes, -1)

            raw_reward = (1 - np.abs(opt_a - a)).mean(axis=1)
            raw_reward = th.clamp((raw_reward - 0.75) * 4.0, min=0.0)

            if storepath is not None:

                storepath = Path(storepath)
                
                save_npz(
                    storepath,
                    states=states,
                    opt_a=opt_a,
                    a=a,
                    reward=raw_reward,
                )

        return states, opt_a, a, raw_reward
    
    def close(self):
        pass