import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import warnings
import os
import copy
import time
import random
import argparse
import multiprocessing as mp
import matplotlib.pyplot as plt
import logging
from scipy.signal import savgol_filter
import gym
import d4rl


# ============================== 1. Core network modules ==============================
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x)) * self.max_action
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.q1_fc1 = nn.Linear(state_dim + action_dim, 256)
        self.q1_fc2 = nn.Linear(256, 256)
        self.q1_fc3 = nn.Linear(256, 1)
        self.q2_fc1 = nn.Linear(state_dim + action_dim, 256)
        self.q2_fc2 = nn.Linear(256, 256)
        self.q2_fc3 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        q1 = F.relu(self.q1_fc1(sa))
        q1 = F.relu(self.q1_fc2(q1))
        q1 = self.q1_fc3(q1)
        q2 = F.relu(self.q2_fc1(sa))
        q2 = F.relu(self.q2_fc2(q2))
        q2 = self.q2_fc3(q2)
        return q1, q2


class ValueNet(nn.Module):
    def __init__(self, state_dim):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class RewardNet(nn.Module):
    """Reward model R(s,a): outputs a scalar in [0, 1] for lower-policy training"""
    def __init__(self, state_dim, action_dim):
        super(RewardNet, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(sa))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))


# ============================== 2. Default configuration ==============================
def get_config(seed=42, env_id='hopper-medium-v2', device_str=None):
    if device_str is None:
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    return {
        'seed': seed,
        'max_timesteps': int(1e6),  # count lower-policy updates only
        'eval_freq': 5000,
        'eval_episodes': 10,
        'batch_size': 256,
        'actor_lr': 3e-4,
        'critic_lr': 3e-4,
        'discount': 0.99,
        'tau': 0.005,
        'alpha': 2.5,
        'beta': 2.0,
        'k': 6,
        'buffer_max_size': int(2e6),
        'policy_freq': 2,
        'policy_noise': 0.2,
        'noise_clip': 0.5,
        'max_action': 1.0,
        'env_id': env_id,
        'max_ep_len': 1000,
        'print_loss_freq': 5000,
        'normalize_states': True,
        'norm_eps': 1e-3,
        'smooth_window': 21,
        'smooth_polyorder': 3,
        'device_str': device_str,
        'delay_step': 50,
        'lower_update_steps': 10000,
        'upper_update_steps': 10000,
        'upper_first': False,
        'value_lr': 1e-4,
        'reward_lr': 1e-4,
        'kd_workers': 4,
        'use_amp': True,
        'save_eval_checkpoints': True,
        'save_training_checkpoint': True,
        'save_final_model': True,
        'resume': True,

    }


# ============================== 3. Utilities ==============================
def set_seed(seed: int, logger):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    logger.info(f"Random seed: {seed}")


def env_reset(env, seed=None):
    if seed is None:
        out = env.reset()
        return out[0] if isinstance(out, tuple) else out
    try:
        out = env.reset(seed=seed)
        return out[0] if isinstance(out, tuple) else out
    except TypeError:
        pass
    try:
        env.seed(seed)
    except Exception:
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    out = env.reset()
    return out[0] if isinstance(out, tuple) else out


def env_step(env, action):
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        return obs, reward, bool(terminated or truncated), info
    else:
        obs, reward, done, info = out
        return obs, reward, bool(done), info


def compute_discounted_returns(rewards, terminals, gamma=0.99):
    returns = np.zeros_like(rewards, dtype=np.float32)
    current_return = 0.0
    for i in reversed(range(len(rewards))):
        if terminals[i]:
            current_return = 0.0
        current_return = rewards[i] + gamma * current_return
        returns[i] = current_return
    return returns


def norm_batch(x, eps=1e-8):
    x_min = x.min(dim=1, keepdim=True)[0]
    x_max = x.max(dim=1, keepdim=True)[0]
    return (x - x_min) / (x_max - x_min + eps)


def load_d4rl_sparse_data(env_id: str, delay_step, discount, logger, dense_reward=False, norm_reward=False):
    env = gym.make(env_id)
    dataset = env.get_dataset()  # use get_dataset to keep the timeouts field

    states = dataset["observations"].astype(np.float32)
    actions = dataset["actions"].astype(np.float32)
    next_states = dataset["next_observations"].astype(np.float32)
    terminals = dataset["terminals"].astype(np.float32)
    original_rewards = dataset["rewards"].astype(np.float32)
    timeouts = dataset.get("timeouts", np.zeros_like(terminals, dtype=np.float32)).astype(np.float32)
    # episode boundary = environment terminal or timeout truncation
    ep_end = ((terminals + timeouts) > 0).astype(np.float32)
    n_timeouts = int(timeouts.sum())
    logger.info(f"Loaded D4RL dataset: {env_id} | terminals={int(terminals.sum())} timeouts={n_timeouts}")

    if dense_reward:
        rewards = original_rewards
        logger.info(f"  [dense-reward mode]")
    else:
        mode_str = " [reward normalization]" if norm_reward else ""
        logger.info(f"  (delay={delay_step}){mode_str}")
        rewards = np.zeros_like(original_rewards, dtype=np.float32)
        accumulated, step_counter = 0.0, 0
        for i in range(len(original_rewards)):
            accumulated += original_rewards[i]
            step_counter += 1
            if step_counter == delay_step or ep_end[i]:
                rewards[i] = accumulated / step_counter if norm_reward else accumulated
                accumulated, step_counter = 0.0, 0
            else:
                rewards[i] = 0.0
        nonzero = np.count_nonzero(rewards)
        logger.info(f"  sparse conversion done | active steps {nonzero}/{len(rewards)} | sparsity {1 - nonzero / len(rewards):.1%}")

    real_q = compute_discounted_returns(rewards, ep_end, discount)

    data = {
        'states': states, 'actions': actions, 'next_states': next_states,
        'rewards': rewards, 'dones': ep_end, 'real_q_values': real_q,
        'state_dim': states.shape[1], 'action_dim': actions.shape[1]
    }
    env.close()
    return data


def smooth_curve(y, w, o):
    if len(y) < 3:
        return y
    if len(y) < w:
        w = len(y) if len(y) % 2 == 1 else len(y) - 1
        if w < 3:
            w = 3
    try:
        return savgol_filter(y, w, o)
    except Exception:
        return np.convolve(y, np.ones(w) / w, 'same')


# ============================== 4. ReplayBuffer (CPU storage) ==============================
class ReplayBuffer:
    def __init__(self, state_dim, action_dim, device, max_size=int(2e6)):
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0
        self.device = device
        self.state = torch.zeros((self.max_size, state_dim), dtype=torch.float32)
        self.action = torch.zeros((self.max_size, action_dim), dtype=torch.float32)
        self.next_state = torch.zeros((self.max_size, state_dim), dtype=torch.float32)
        self.reward = torch.zeros((self.max_size, 1), dtype=torch.float32)
        self.not_done = torch.zeros((self.max_size, 1), dtype=torch.float32)
        self.real_q = torch.zeros((self.max_size, 1), dtype=torch.float32)

    def __len__(self):
        return self.size

    def add_batch(self, states, actions, next_states, rewards, dones, real_q_values):
        batch_size = len(states)
        if self.ptr + batch_size > self.max_size:
            split = self.max_size - self.ptr
            self._add_part(states[:split], actions[:split], next_states[:split],
                           rewards[:split], dones[:split], real_q_values[:split])
            self._add_part(states[split:], actions[split:], next_states[split:],
                           rewards[split:], dones[split:], real_q_values[split:])
        else:
            self._add_part(states, actions, next_states, rewards, dones, real_q_values)

    def _add_part(self, states, actions, next_states, rewards, dones, real_q_values):
        n = len(states)
        indices = np.arange(self.ptr, self.ptr + n) % self.max_size
        self.state[indices] = torch.as_tensor(states)
        self.action[indices] = torch.as_tensor(actions)
        self.next_state[indices] = torch.as_tensor(next_states)
        self.reward[indices] = torch.as_tensor(rewards).reshape(-1, 1)
        self.not_done[indices] = 1.0 - torch.as_tensor(dones).reshape(-1, 1)
        self.real_q[indices] = torch.as_tensor(real_q_values).reshape(-1, 1)
        self.ptr = (self.ptr + n) % self.max_size
        self.size = min(self.size + n, self.max_size)

    def sample(self, batch_size):
        ind = torch.randint(0, self.size, (batch_size,))
        return (
            self.state[ind].to(self.device), self.action[ind].to(self.device),
            self.reward[ind].to(self.device), self.next_state[ind].to(self.device),
            self.not_done[ind].to(self.device), self.real_q[ind].to(self.device),
        )

    def normalize_states(self, eps=1e-3):
        mean = self.state[:self.size].mean(0, keepdim=True)
        std = self.state[:self.size].std(0, keepdim=True) + eps
        self.state[:self.size] = (self.state[:self.size] - mean) / std
        self.next_state[:self.size] = (self.next_state[:self.size] - mean) / std
        return mean.squeeze(0).numpy(), std.squeeze(0).numpy()


# ============================== 5. TD3+NC lower policy ==============================
class TD3NC(object):
    def __init__(self, state_dim, action_dim, max_action, device, cfg):
        self.device = torch.device(device)
        self.total_it = 0

        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg['actor_lr'])

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg['critic_lr'])

        self.action_dim = action_dim
        self.max_action = max_action
        self.discount = cfg['discount']
        self.tau = cfg['tau']
        self.policy_noise = cfg['policy_noise']
        self.noise_clip = cfg['noise_clip']
        self.policy_freq = cfg['policy_freq']
        self.alpha = cfg['alpha']
        self.k = cfg['k']
        self.beta = cfg['beta']

        self.use_amp = cfg['use_amp'] and str(self.device) != 'cpu'
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        self.kd_data_gpu = None       # full (state, action) data [N, D] for brute-force GPU k-NN
        self.real_q_gpu = None

        self.models = {
            "actor": self.actor, "critic": self.critic,
            "actor_target": self.actor_target, "critic_target": self.critic_target,
            "actor_optimizer": self.actor_optimizer, "critic_optimizer": self.critic_optimizer
        }

    @torch.no_grad()
    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().numpy().flatten()

    def train(self, replay_buffer, reward_net, batch_size=256):
        self.total_it += 1
        tb_statics = dict()
        state, action, reward, next_state, not_done, _ = replay_buffer.sample(batch_size)

        with torch.no_grad():
            if reward_net is None:
                train_reward = reward
            else:
                train_reward = reward_net(state, action)
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = train_reward + not_done * self.discount * torch.min(target_Q1, target_Q2)

        # Critic forward (AMP or regular precision)
        if self.use_amp:
            with torch.cuda.amp.autocast():
                current_Q1, current_Q2 = self.critic(state, action)
                critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            self.critic_optimizer.zero_grad()
            self.scaler.scale(critic_loss).backward()
            self.scaler.unscale_(self.critic_optimizer)
            # grad clip removed
            self.scaler.step(self.critic_optimizer)
            self.scaler.update()
        else:
            current_Q1, current_Q2 = self.critic(state, action)
            critic_loss = F.smooth_l1_loss(current_Q1, target_Q) + F.smooth_l1_loss(current_Q2, target_Q)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            # grad clip removed
            self.critic_optimizer.step()

        tb_statics["critic_loss"] = critic_loss.item()

        # Actor update with neighbor search
        if self.total_it % self.policy_freq == 0:
            pi = self.actor(state)

            # Brute-force GPU k-NN with torch.cdist and topk instead of scipy KDTree
            key = torch.cat([self.beta * state, pi], dim=1)
            dist = torch.cdist(key, self.kd_data_gpu)           # [B, N] L2
            distances_th, idx_th = torch.topk(dist, k=self.k, dim=1, largest=False)

            # Use GPU indices to gather neighbor actions and q_scores directly, avoiding extra CPU-to-GPU copies
            neighbor_actions_th = self.kd_data_gpu[idx_th][:, :, -self.action_dim:]
            q_scores_th = self.real_q_gpu[idx_th]

            dist_norm = norm_batch(1.0 / (distances_th + 1e-8))
            q_norm = norm_batch(q_scores_th)
            best_idx = torch.argmax(0.6 * q_norm + 0.4 * dist_norm, dim=1)
            best_act = neighbor_actions_th[torch.arange(batch_size), best_idx]

            if self.use_amp:
                with torch.cuda.amp.autocast():
                    Q1, _ = self.critic(state, pi)
                    lmbda = self.alpha / Q1.abs().mean().detach()
                    actor_loss = -lmbda * Q1.mean()
                    dc_loss = F.mse_loss(pi, best_act)
                    combined_loss = actor_loss + dc_loss

                self.actor_optimizer.zero_grad()
                self.scaler.scale(combined_loss).backward()
                self.scaler.unscale_(self.actor_optimizer)
                # grad clip removed
                self.scaler.step(self.actor_optimizer)
                self.scaler.update()
            else:
                Q1, _ = self.critic(state, pi)
                lmbda = self.alpha / Q1.abs().mean().detach()
                actor_loss = -lmbda * Q1.mean()
                dc_loss = F.mse_loss(pi, best_act)
                combined_loss = actor_loss + dc_loss

                self.actor_optimizer.zero_grad()
                combined_loss.backward()
                # grad clip removed
                self.actor_optimizer.step()

            tb_statics.update({
                "dc_loss": dc_loss.item(), "actor_loss": actor_loss.item(),
                "combined_loss": combined_loss.item(), "Q_value": Q1.mean().item()
            })

            # Soft-update target networks
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        return tb_statics

    def set_kd_data(self, kd_data, real_q_values):
        # Keep full data on GPU and use torch.cdist plus topk during training for native GPU brute-force search
        self.kd_data_gpu = torch.from_numpy(kd_data).float().to(self.device)
        self.real_q_gpu = torch.from_numpy(real_q_values).float().to(self.device)

    def save(self, path, full=False):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if full:
            torch.save({k: v.state_dict() for k, v in self.models.items()}, path)
        else:
            torch.save({"actor": self.actor.state_dict(),
                        "critic": self.critic.state_dict()}, path)

    def checkpoint_state(self):
        state = {k: v.state_dict() for k, v in self.models.items()}
        state["total_it"] = self.total_it
        if self.scaler is not None:
            state["scaler"] = self.scaler.state_dict()
        return state

    def load_checkpoint_state(self, state):
        for k, v in self.models.items():
            if k in state:
                v.load_state_dict(state[k])
        self.total_it = int(state.get("total_it", self.total_it))
        if self.scaler is not None and "scaler" in state:
            self.scaler.load_state_dict(state["scaler"])


# ============================== 6. Upper-level trainer ==============================
class UpperLayerTrainer:
    def __init__(self, state_dim, action_dim, device, cfg):
        self.device = torch.device(device)
        self.value_net = ValueNet(state_dim).to(self.device)
        self.reward_net = RewardNet(state_dim, action_dim).to(self.device)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=cfg['value_lr'])
        self.reward_optimizer = optim.Adam(self.reward_net.parameters(), lr=cfg['reward_lr'])
        self.mse_loss = nn.MSELoss()

        self.use_amp = cfg['use_amp'] and str(self.device) != 'cpu'
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

    def train_step(self, replay_buffer, batch_size=256):
        s1, a1, _, _, _, g1 = replay_buffer.sample(batch_size)
        s2, a2, _, _, _, g2 = replay_buffer.sample(batch_size)

        if self.use_amp:
            with torch.cuda.amp.autocast():
                v1, v2 = self.value_net(s1), self.value_net(s2)
                r1, r2 = self.reward_net(s1, a1), self.reward_net(s2, a2)
                value_loss = self.mse_loss(v1, g1) + self.mse_loss(v2, g2)
                contrast_loss = -torch.mean((v1 - v2) * (r1 - r2))
                total_loss = value_loss + contrast_loss

            self.value_optimizer.zero_grad()
            self.reward_optimizer.zero_grad()
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.value_optimizer)
            self.scaler.unscale_(self.reward_optimizer)
            self.scaler.step(self.value_optimizer)
            self.scaler.step(self.reward_optimizer)
            self.scaler.update()
        else:
            v1, v2 = self.value_net(s1), self.value_net(s2)
            r1, r2 = self.reward_net(s1, a1), self.reward_net(s2, a2)
            value_loss = self.mse_loss(v1, g1) + self.mse_loss(v2, g2)
            contrast_loss = -torch.mean((v1 - v2) * (r1 - r2))
            total_loss = value_loss + contrast_loss

            self.value_optimizer.zero_grad()
            self.reward_optimizer.zero_grad()
            total_loss.backward()
            self.value_optimizer.step()
            self.reward_optimizer.step()

        return {
            "value_loss": value_loss.item(),
            "reward_contrast_loss": contrast_loss.item(),
            "total_upper_loss": total_loss.item()
        }

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "value_net": self.value_net.state_dict(),
            "reward_net": self.reward_net.state_dict(),
        }, path)

    def checkpoint_state(self):
        state = {
            "value_net": self.value_net.state_dict(),
            "reward_net": self.reward_net.state_dict(),
            "value_optimizer": self.value_optimizer.state_dict(),
            "reward_optimizer": self.reward_optimizer.state_dict(),
        }
        if self.scaler is not None:
            state["scaler"] = self.scaler.state_dict()
        return state

    def load_checkpoint_state(self, state):
        if "value_net" in state:
            self.value_net.load_state_dict(state["value_net"])
        if "reward_net" in state:
            self.reward_net.load_state_dict(state["reward_net"])
        if "value_optimizer" in state:
            self.value_optimizer.load_state_dict(state["value_optimizer"])
        if "reward_optimizer" in state:
            self.reward_optimizer.load_state_dict(state["reward_optimizer"])
        if self.scaler is not None and "scaler" in state:
            self.scaler.load_state_dict(state["scaler"])


# ============================== 7. Single-seed experiment ==============================
@torch.no_grad()
def evaluate_policy(policy, env, eval_episodes, max_ep_len, seed, mean=0, std=1):
    policy.actor.eval()
    total_raw, total_norm = [], []
    for ep in range(eval_episodes):
        s = env_reset(env, seed + ep * 100)
        ep_r = 0.0
        steps = 0
        while steps < max_ep_len:
            s_norm = (s - mean) / std
            a = policy.select_action(s_norm)
            s, r, done, _ = env_step(env, a)
            ep_r += r
            steps += 1
            if done:
                break
        total_raw.append(ep_r)
        if hasattr(env, "get_normalized_score"):
            total_norm.append(env.get_normalized_score(ep_r) * 100)
    policy.actor.train()
    return np.mean(total_raw), np.mean(total_norm) if total_norm else np.nan, np.std(total_norm) if total_norm else 0.0


def run_experiment(seed, save_dir, cfg_override=None):
    """Run the full training pipeline for one random seed and return the CSV path."""
    cfg = get_config(seed)
    if cfg_override:
        cfg.update(cfg_override)   # cfg_override is already an independent copy and will not modify external state

    # Independent output path for each seed
    model_dir = os.path.join(save_dir, f'seed{seed}')
    csv_prefix = 'td3_nc' if cfg.get('lower_only') else 'ours'
    csv_path = os.path.join(save_dir, f'{csv_prefix}_seed{seed}.csv')
    checkpoint_path = os.path.join(model_dir, 'training_checkpoint.pth')

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s [Seed {seed}] - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(save_dir, f'train_seed{seed}.log'), encoding='utf-8')
        ]
    )
    logger = logging.getLogger(f'seed{seed}')

    warnings.filterwarnings('ignore')
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    start_time = time.time()
    set_seed(seed, logger)
    os.makedirs(model_dir, exist_ok=True)
    device = torch.device(cfg['device_str'])
    logger.info(f"Device: {device} | AMP: {cfg['use_amp'] and str(device) != 'cpu'}")
    logger.info(f"Hyperparameters: alpha={cfg['alpha']} beta={cfg['beta']} K={cfg['k']} "
               f"actor_lr={cfg['actor_lr']} critic_lr={cfg['critic_lr']} "
               f"lower={cfg['lower_update_steps']} upper={cfg['upper_update_steps']}")

    # 1. Load D4RL dataset
    data = load_d4rl_sparse_data(cfg['env_id'], cfg['delay_step'], cfg['discount'], logger,
                                  dense_reward=cfg.get('dense_reward', False),
                                  norm_reward=cfg.get('norm_reward', False))
    state_dim, action_dim = data['state_dim'], data['action_dim']

    # 2. ReplayBuffer
    rb = ReplayBuffer(state_dim, action_dim, device, cfg['buffer_max_size'])
    rb.add_batch(data['states'], data['actions'], data['next_states'],
                 data['rewards'], data['dones'], data['real_q_values'])

    # If g-as-reward is enabled, replace buffer rewards with Monte Carlo discounted return g
    if cfg.get('g_as_reward'):
        g_vals = data['real_q_values'].copy()
        if cfg.get('norm_g_reward'):
            g_vals = (g_vals - g_vals.mean()) / (g_vals.std() + 1e-8)
        rb.reward[:rb.size] = torch.as_tensor(g_vals).reshape(-1, 1)
        logger.info(f"g-as-reward: reward has been replaced by MC return g "
                    f"(norm={cfg.get('norm_g_reward', False)}, mean={g_vals.mean():.3f}, std={g_vals.std():.3f})")

    # 3. Normalization + neighbor-ranking data
    mean, std = 0.0, 1.0
    if cfg['normalize_states']:
        mean, std = rb.normalize_states(cfg['norm_eps'])
        np.savez(os.path.join(model_dir, 'norm_stats.npz'), mean=mean, std=std)
        train_s = data['states']
        train_a = data['actions']
        s_norm = (train_s - mean) / std
        kd_data = np.hstack([cfg['beta'] * s_norm, train_a]).astype(np.float32)
        kd_q = data['real_q_values']
    else:
        train_s = data['states']
        train_a = data['actions']
        kd_data = np.hstack([cfg['beta'] * train_s, train_a]).astype(np.float32)
        kd_q = data['real_q_values']

    # 4. Initialize models
    lower_agent = TD3NC(state_dim, action_dim, cfg['max_action'], cfg['device_str'], cfg)
    lower_agent.set_kd_data(kd_data, kd_q)
    upper_trainer = UpperLayerTrainer(state_dim, action_dim, device, cfg)

    # 5. Record container
    records = []  # one row per evaluation point
    best_norm = -np.inf
    loss_sum = {
        'critic_loss': 0.0,
        'actor_loss': 0.0,
        'dc_loss': 0.0,
        'combined_loss': 0.0,
        'Q_value': 0.0,
    }
    loss_cnt = 0
    total_global_step = 0

    def _rng_state():
        state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    def _restore_rng_state(state):
        if not state:
            return
        if "python" in state:
            random.setstate(state["python"])
        if "numpy" in state:
            np.random.set_state(state["numpy"])
        if "torch" in state:
            torch.set_rng_state(state["torch"])
        if torch.cuda.is_available() and "cuda" in state:
            torch.cuda.set_rng_state_all(state["cuda"])

    def _save_training_checkpoint():
        tmp_path = checkpoint_path + ".tmp"
        torch.save({
            "seed": seed,
            "total_global_step": total_global_step,
            "records": records,
            "best_norm": best_norm,
            "loss_sum": loss_sum,
            "loss_cnt": loss_cnt,
            "lower_agent": lower_agent.checkpoint_state(),
            "upper_trainer": upper_trainer.checkpoint_state(),
            "rng_state": _rng_state(),
        }, tmp_path)
        os.replace(tmp_path, checkpoint_path)

    if cfg.get('resume') and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        lower_agent.load_checkpoint_state(ckpt.get("lower_agent", {}))
        upper_trainer.load_checkpoint_state(ckpt.get("upper_trainer", {}))
        records = ckpt.get("records", records)
        best_norm = ckpt.get("best_norm", best_norm)
        loss_sum = ckpt.get("loss_sum", loss_sum)
        loss_cnt = int(ckpt.get("loss_cnt", loss_cnt))
        total_global_step = int(ckpt.get("total_global_step", total_global_step))
        _restore_rng_state(ckpt.get("rng_state"))
        logger.info(f"Resumed from checkpoint: {checkpoint_path} | step={total_global_step} | records={len(records)}")

    eval_env = gym.make(cfg['env_id'])

    logger.info(f"=== Start training (seed={seed}) ===")
    order = "upper-to-lower" if cfg.get('upper_first') else "lower-to-upper"
    logger.info(f"Training order: {order} | Lower {cfg['lower_update_steps']} steps <-> Upper {cfg['upper_update_steps']} steps")

    def _lower_block():
        nonlocal total_global_step, loss_sum, loss_cnt, best_norm
        for _ in range(cfg['lower_update_steps']):
            if total_global_step >= cfg['max_timesteps']:
                break
            reward_net = None if cfg.get('lower_only') else upper_trainer.reward_net
            loss_dict = lower_agent.train(rb, reward_net, cfg['batch_size'])
            for k in loss_dict:
                if k in loss_sum:
                    loss_sum[k] += loss_dict[k]
            loss_cnt += 1
            total_global_step += 1

            if total_global_step % cfg['print_loss_freq'] == 0:
                logger.info(f"[Lower] Step {total_global_step} | "
                           f"Critic {loss_sum['critic_loss']/loss_cnt:.4f} | "
                           f"Actor {loss_sum['actor_loss']/max(loss_cnt,1):.4f}")

            if total_global_step % cfg['eval_freq'] == 0:
                mean_raw, mean_norm, std_norm = evaluate_policy(
                    lower_agent, eval_env, cfg['eval_episodes'],
                    cfg['max_ep_len'], seed, mean, std)
                logger.info(f"[Eval] Step {total_global_step}: Raw={mean_raw:.2f} Norm={mean_norm:.2f}+-{std_norm:.2f}")
                records.append({
                    'step': total_global_step,
                    'raw_reward': mean_raw,
                    'norm_score': mean_norm,
                    'norm_std': std_norm,
                })
                pd.DataFrame(records).to_csv(csv_path, index=False)
                if cfg.get('save_training_checkpoint', True):
                    _save_training_checkpoint()
                if mean_norm > best_norm:
                    best_norm = mean_norm
                    lower_agent.save(os.path.join(model_dir, 'lower_best.pth'))
                    upper_trainer.save(os.path.join(model_dir, 'upper_best.pth'))

    def _upper_block():
        nonlocal total_global_step
        for _ in range(cfg['upper_update_steps']):
            upper_trainer.train_step(rb, cfg['batch_size'])
            # Do not increment total_global_step; count lower-policy updates only

    while total_global_step < cfg['max_timesteps']:
        if cfg.get('upper_first'):
            if not cfg.get('lower_only'):
                _upper_block()
            _lower_block()
        else:
            _lower_block()
            if not cfg.get('lower_only'):
                _upper_block()

    eval_env.close()

    # 6. Save CSV
    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    logger.info(f"Results saved to {csv_path}")

    # Save final model
    if cfg.get('save_final_model', True):
        lower_agent.save(os.path.join(model_dir, 'lower_final.pth'), full=True)
        upper_trainer.save(os.path.join(model_dir, 'upper_final.pth'))
    if cfg.get('save_training_checkpoint', True):
        _save_training_checkpoint()

    elapsed = (time.time() - start_time) / 60
    logger.info(f"Seed {seed} finished; elapsed {elapsed:.1f}min best score {best_norm:.2f}")

    return csv_path, best_norm


# ============================== 8. D4RL plotting ==============================
def load_curve(csv_dir, seeds, smooth_window=21, smooth_polyorder=3):
    """Load multi-seed curves from a CSV directory and return (steps, mean_smooth, std_smooth).
    For multiple seeds, std is computed across seeds. For one seed, std is read from the CSV norm_std column."""
    all_curves = []
    # First look up requested seeds; otherwise auto-discover all seed CSV files in the directory
    found = False
    for seed in seeds:
        csv_path = os.path.join(csv_dir, f'ours_seed{seed}.csv')
        if not os.path.exists(csv_path):
            csv_path = os.path.join(csv_dir, f'td3_nc_seed{seed}.csv')
        if os.path.exists(csv_path):
            found = True
            df = pd.read_csv(csv_path)
            all_curves.append(df[['step', 'norm_score', 'norm_std']].values)
    if not found:
        import glob
        patterns = ['ours_seed*.csv', 'td3_nc_seed*.csv']
        for p in sorted(sum((glob.glob(os.path.join(csv_dir, pat)) for pat in patterns), [])):
            df = pd.read_csv(p)
            all_curves.append(df[['step', 'norm_score', 'norm_std']].values)
            print(f"[INFO] Auto-discovered {p}")

    if not all_curves:
        return None, None, None

    # Automatically normalize old total_global_step format around 2e6 to the newer lower_step scale around 1e6
    for i, c in enumerate(all_curves):
        if c[:, 0].max() > 1.5e6:
            all_curves[i] = c.copy()
            all_curves[i][:, 0] /= 2
    common_steps = all_curves[0][:, 0]
    norm_interp = []
    std_interp = []
    for curve in all_curves:
        steps, scores, stds = curve[:, 0], curve[:, 1], curve[:, 2]
        norm_interp.append(np.interp(common_steps, steps, scores))
        std_interp.append(np.interp(common_steps, steps, stds))

    mean_arr = np.array(norm_interp)
    std_arr = np.array(std_interp)

    mean_curve = np.mean(mean_arr, axis=0)
    if len(all_curves) > 1:
        std_curve = np.std(mean_arr, axis=0)  # cross-seed standard deviation
    else:
        std_curve = std_arr[0]  # single seed: use evaluation-episode standard deviation

    mean_smooth = smooth_curve(mean_curve, smooth_window, smooth_polyorder)
    std_smooth = smooth_curve(std_curve, smooth_window, smooth_polyorder)
    return common_steps, mean_smooth, std_smooth


def plot_d4rl_benchmark(csv_dir, save_path, env_id, seeds=[0, 1, 2, 3, 4],
                        smooth_window=21, smooth_polyorder=3, label=None):
    """Standard D4RL plot for a single experiment."""
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 12,
        'axes.linewidth': 1.5, 'axes.labelsize': 14, 'axes.titlesize': 14,
        'xtick.major.width': 1.5, 'ytick.major.width': 1.5,
        'xtick.labelsize': 11, 'ytick.labelsize': 11,
        'figure.dpi': 150, 'savefig.dpi': 150,
        'legend.fontsize': 12,
    })

    t, mean_smooth, std_smooth = load_curve(csv_dir, seeds, smooth_window, smooth_polyorder)
    if t is None:
        print("[ERROR] No CSV data found; cannot plot")
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.fill_between(t, mean_smooth - 0.5 * std_smooth, mean_smooth + 0.5 * std_smooth,
                     alpha=0.2, color='tab:blue')
    ax.plot(t, mean_smooth, linewidth=2.5, color='tab:blue',
            label=label or ('TD3+NC' if cfg.get('lower_only') else 'Ours'))

    env_name = env_id.replace('-', ' ').title()
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Normalized Score')
    ax.set_title(f'D4RL {env_name}')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, t[-1])
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Benchmark plot saved to {save_path}")


def plot_comparison(configs, save_path, env_id, seeds=[0, 1, 2, 3, 4],
                    smooth_window=21, smooth_polyorder=3):
    """
    Multi-hyperparameter comparison plot. Each config is shown as a mean curve with a standard-deviation band.
    configs: [(csv_dir, label), ...]
    """
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 11,
        'axes.linewidth': 1.5, 'axes.labelsize': 13, 'axes.titlesize': 13,
        'xtick.major.width': 1.5, 'ytick.major.width': 1.5,
        'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'figure.dpi': 150, 'savefig.dpi': 150,
        'legend.fontsize': 10,
    })

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red',
              'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    all_steps = []

    for i, (csv_dir, label) in enumerate(configs):
        t, mean_smooth, std_smooth = load_curve(csv_dir, seeds, smooth_window, smooth_polyorder)
        if t is None:
            print(f"[WARNING] {label} ({csv_dir}) has no data; skipped")
            continue

        color = colors[i % len(colors)]
        ax.fill_between(t, mean_smooth - 0.5 * std_smooth, mean_smooth + 0.5 * std_smooth,
                         alpha=0.12, color=color)
        ax.plot(t, mean_smooth, linewidth=2.2, color=color, label=label)
        all_steps.append(t[-1])

    env_name = env_id.replace('-', ' ').title()
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Normalized Score')
    ax.set_title(f'D4RL {env_name} - Hyperparameter Comparison')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(all_steps) if all_steps else 1)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved to {save_path}")


# ============================== 9. Main entry point ==============================
def worker_wrapper(args):
    """multiprocessing worker: receive (seed, save_dir, config override, GPU ID) and run one experiment"""
    seed, save_dir, cfg_override, gpu_id = args
    cfg_override = dict(cfg_override)  # copy to avoid modifying other workers
    if gpu_id is not None:
        cfg_override['device_str'] = f'cuda:{gpu_id}'
    csv_path, best = run_experiment(seed, save_dir, cfg_override)
    return seed, csv_path, best


def main():
    parser = argparse.ArgumentParser(description='Ours and lower-only TD3+NC on D4RL')
    parser.add_argument('--seed', type=int, default=None,
                        help='Single-seed mode: specify the random seed')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4],
                        help='multi-seed mode: seed list (default: 0 1 2 3 4)')
    parser.add_argument('--parallel', type=int, default=1,
                        help='number of parallel processes (default: 1; for multi-seed runs, use no more than the number of GPUs)')
    parser.add_argument('--env', type=str, default='hopper-medium-v2',
                        help='D4RL environment ID')
    parser.add_argument('--save-dir', type=str, default='results/ours',
                        help='result save directory')
    parser.add_argument('--plot-only', action='store_true',
                        help='plot from existing CSV files only; do not train')
    parser.add_argument('--plot-path', '--save-fig', type=str, default=None, dest='plot_path',
                        help='benchmark plot save path')
    parser.add_argument('--compare', type=str, nargs='+', default=None,
                        help='comparison mode: multiple experiment directories separated by spaces')
    parser.add_argument('--labels', type=str, nargs='+', default=None,
                        help='comparison mode: legend labels matching --compare')
    # Tunable hyperparameters
    parser.add_argument('--alpha', type=float, default=None, help='TD3+NC alpha')
    parser.add_argument('--beta', type=float, default=None, help='KD-Tree state weight')
    parser.add_argument('--k', type=int, default=None, help='number of KD-Tree nearest neighbors')
    parser.add_argument('--actor-lr', type=float, default=None, help='actor learning rate')
    parser.add_argument('--critic-lr', type=float, default=None, help='critic learning rate')
    parser.add_argument('--value-lr', type=float, default=None, help='value-network learning rate')
    parser.add_argument('--reward-lr', type=float, default=None, help='reward-network learning rate')
    parser.add_argument('--discount', type=float, default=None, help='discount factor gamma')
    parser.add_argument('--tau', type=float, default=None, help='soft-update coefficient')
    parser.add_argument('--lower-steps', type=int, default=None, help='lower-policy update steps per phase')
    parser.add_argument('--upper-steps', type=int, default=None, help='upper-level update steps per phase')
    parser.add_argument('--upper-first', action='store_true', help='train upper level first (default: lower level first)')
    parser.add_argument('--delay-step', type=int, default=None, help='delayed sparse-reward interval')
    parser.add_argument('--max-timesteps', type=int, default=None, help='maximum training steps')
    parser.add_argument('--policy-freq', type=int, default=None, help='delayed actor update frequency')
    parser.add_argument('--policy-noise', type=float, default=None, help='target-policy smoothing noise')
    parser.add_argument('--noise-clip', type=float, default=None, help='noise clipping range')
    parser.add_argument('--batch-size', type=int, default=None, help='batch size')
    parser.add_argument('--grad-clip', type=float, default=None, help='gradient clipping threshold')
    parser.add_argument('--eval-episodes', type=int, default=None, help='number of evaluation episodes (default: 10)')
    parser.add_argument('--dense-reward', action='store_true', help='use original dense rewards without sparsification')
    parser.add_argument('--lower-only', action='store_true', help='TD3+NC mode: train only the lower policy without the upper value/reward layer')
    parser.add_argument('--no-amp', action='store_true', help='disable AMP mixed-precision training')
    parser.add_argument('--resume', action='store_true', help='resume from training_checkpoint.pth in the seed directory')
    parser.add_argument('--no-training-checkpoint', action='store_true', help='do not save training checkpoints')
    parser.add_argument('--no-eval-checkpoints', action='store_true', help='do not save model checkpoints at evaluation points')
    parser.add_argument('--no-final-model', action='store_true', help='do not save the final model')
    parser.add_argument('--norm-reward', action='store_true', help='normalize sparse rewards by the accumulation length (divide by delay_step)')
    parser.add_argument('--dataset', type=str, default='d4rl', choices=['d4rl'],
                        help='dataset type')
    parser.add_argument('--g-as-reward', action='store_true', help='use Monte Carlo discounted return g directly as lower-policy reward, skipping the upper level')
    parser.add_argument('--norm-g-reward', action='store_true', help='globally normalize g rewards (zero mean and unit std)')
    args = parser.parse_args()

    # Collect non-default CLI hyperparameters
    cli_hparams = {}
    for key in ['alpha', 'beta', 'k', 'actor_lr', 'critic_lr', 'value_lr', 'reward_lr',
                'discount', 'tau', 'delay_step', 'max_timesteps', 'policy_freq', 'policy_noise',
                'noise_clip', 'batch_size', 'grad_clip', 'eval_episodes']:
        val = getattr(args, key)
        if val is not None:
            cli_hparams[key] = val
    if args.lower_steps is not None:
        cli_hparams['lower_update_steps'] = args.lower_steps
    if args.upper_steps is not None:
        cli_hparams['upper_update_steps'] = args.upper_steps
    if args.upper_first:
        cli_hparams['upper_first'] = True
    if args.dense_reward:
        cli_hparams['dense_reward'] = True
    if args.lower_only:
        cli_hparams['lower_only'] = True
    if args.no_amp:
        cli_hparams['use_amp'] = False
    if args.resume:
        cli_hparams['resume'] = True
    if args.no_training_checkpoint:
        cli_hparams['save_training_checkpoint'] = False
    if args.no_eval_checkpoints:
        cli_hparams['save_eval_checkpoints'] = False
    if args.no_final_model:
        cli_hparams['save_final_model'] = False
    if args.norm_reward:
        cli_hparams['norm_reward'] = True
    if args.dataset:
        cli_hparams['dataset'] = args.dataset
    if args.g_as_reward:
        cli_hparams['g_as_reward'] = True
        cli_hparams['lower_only'] = True  # g is used directly as reward, so the upper level is not needed
    if args.norm_g_reward:
        cli_hparams['norm_g_reward'] = True

    # Comparison-plot mode, usable alone or with --plot-only
    if args.compare:
        plot_path = args.plot_path or 'results/comparison.png'
        if args.labels:
            configs = list(zip(args.compare, args.labels))
        else:
            configs = [(d, os.path.basename(d)) for d in args.compare]
        print(f"Comparison mode: {[(d, l) for d, l in configs]}")
        plot_comparison(configs, plot_path, args.env, args.seeds)
        if args.plot_only or args.seed is None:
            return

    # Plot-only mode for a single group
    if args.plot_only:
        plot_path = args.plot_path or os.path.join(args.save_dir, 'd4rl_benchmark.png')
        plot_d4rl_benchmark(args.save_dir, plot_path, args.env, args.seeds)
        return

    cli_hparams['env_id'] = args.env

    os.makedirs(args.save_dir, exist_ok=True)

    # Single-seed mode
    if args.seed is not None:
        csv_path, best = run_experiment(args.seed, args.save_dir, cli_hparams)
        print(f"\nDone! Seed {args.seed} best score: {best:.2f}")
        print(f"CSV: {csv_path}")
        plot_path = args.plot_path or os.path.join(args.save_dir, f'benchmark_seed{args.seed}.png')
        plot_d4rl_benchmark(args.save_dir, plot_path, args.env, [args.seed])
        return

    # Multi-seed mode
    seeds = args.seeds
    print(f"=== Multi-seed training ===")
    gpu_count = torch.cuda.device_count()
    print(f"Seeds: {seeds}")
    print(f"parallel workers: {args.parallel}")
    print(f"available GPUs: {gpu_count}")
    print(f"GPU assignment: " + ", ".join(f"seed{s}->GPU{s % gpu_count}" if gpu_count > 0 else f"seed{s}->CPU" for s in seeds))
    print(f"save directory: {args.save_dir}")
    if cli_hparams:
        print(f"CLI Hyperparameters: {cli_hparams}")

    if args.parallel > 1:
        print(f"Launching {len(seeds)} experiments({args.parallel} concurrent processes)...")
        mp.set_start_method('spawn', force=True)
        tasks = [(s, args.save_dir, cli_hparams, s % gpu_count if gpu_count > 0 else None) for s in seeds]
        with mp.Pool(processes=min(args.parallel, len(seeds))) as pool:
            results = pool.map(worker_wrapper, tasks)
    else:
        results = []
        for s in seeds:
            print(f"\n{'='*60}")
            print(f"Starting seed {s}")
            print(f"{'='*60}")
            csv_path, best = run_experiment(s, args.save_dir, cli_hparams)
            results.append((s, csv_path, best))

    # Summary
    print(f"\n{'='*60}")
    print("All experiments completed!")
    best_scores = {}
    for s, csv_path, best in results:
        print(f"  Seed {s}: best={best:.2f}  csv={csv_path}")
        best_scores[s] = best
    print(f"  Mean +- Std: {np.mean(list(best_scores.values())):.2f} +- {np.std(list(best_scores.values())):.2f}")

    # Save summary
    summary = pd.DataFrame([{'seed': s, 'best_norm_score': v} for s, v in best_scores.items()])
    summary.to_csv(os.path.join(args.save_dir, 'summary.csv'), index=False)

    # Draw D4RL benchmark plot
    plot_path = args.plot_path or os.path.join(args.save_dir, 'd4rl_benchmark.png')
    plot_d4rl_benchmark(args.save_dir, plot_path, args.env, seeds)


if __name__ == "__main__":
    # Suppress matplotlib DISPLAY warnings in headless server environments
    import matplotlib
    matplotlib.use('Agg')
    main()
