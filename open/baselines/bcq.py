# BCQ: Off-Policy Deep Reinforcement Learning without Exploration
# https://arxiv.org/pdf/1812.02900.pdf
# Adapted for D4RL + sparse reward support
import copy
import csv
import os
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import d4rl
import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from sparse_utils import sparsify_reward


@dataclass
class TrainConfig:
    # Experiment
    device: str = "cuda"
    env: str = "hopper-medium-v2"
    seed: int = 0
    eval_freq: int = int(5e3)
    n_episodes: int = 10
    max_timesteps: int = int(1e6)
    checkpoints_path: Optional[str] = None
    save_dir: Optional[str] = None
    load_model: str = ""
    # BCQ
    buffer_size: int = 2_000_000
    batch_size: int = 256
    discount: float = 0.99
    tau: float = 0.005
    beta: float = 0.2  # VAE KL weight
    lmbda: float = 0.75  # Conservative Q weight
    phi: float = 0.05  # Perturbation range
    latent_dim: int = 32
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    vae_lr: float = 3e-4
    normalize: bool = True
    normalize_reward: bool = False
    delay_step: int = 0  # delayed sparse-reward interval; 0 means dense reward
    norm_reward: bool = False


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size, device):
        self.max_size = max_size; self.device = device
        self.ptr = 0; self.size = 0
        self.state = torch.zeros((max_size, state_dim), dtype=torch.float32)
        self.action = torch.zeros((max_size, action_dim), dtype=torch.float32)
        self.next_state = torch.zeros((max_size, state_dim), dtype=torch.float32)
        self.reward = torch.zeros((max_size, 1), dtype=torch.float32)
        self.not_done = torch.zeros((max_size, 1), dtype=torch.float32)

    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        n = len(data["observations"])
        self.state[:n] = torch.tensor(data["observations"], dtype=torch.float32)
        self.action[:n] = torch.tensor(data["actions"], dtype=torch.float32)
        self.next_state[:n] = torch.tensor(data["next_observations"], dtype=torch.float32)
        self.reward[:n] = torch.tensor(data["rewards"], dtype=torch.float32).view(-1, 1)
        self.not_done[:n] = 1.0 - torch.tensor(data["terminals"], dtype=torch.float32).view(-1, 1)
        self.size = n; self.ptr = n

    def sample(self, batch_size):
        ind = torch.randint(0, self.size, (batch_size,))
        return (self.state[ind], self.action[ind], self.reward[ind],
                self.next_state[ind], self.not_done[ind])

    def normalize_states(self, eps=1e-3):
        mean = self.state[:self.size].mean(0, keepdim=True)
        std = self.state[:self.size].std(0, keepdim=True) + eps
        self.state[:self.size] = (self.state[:self.size] - mean) / std
        self.next_state[:self.size] = (self.next_state[:self.size] - mean) / std
        return mean, std


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, out_dim))
    def forward(self, x): return self.net(x)


class VAE(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim=32, hidden=256, max_action=1.0):
        super().__init__()
        self.max_action = max_action; self.latent_dim = latent_dim
        self.encoder = MLP(state_dim + action_dim, 2 * latent_dim, hidden)
        self.decoder = MLP(state_dim + latent_dim, action_dim, hidden)
        self.register_buffer("latent_mean", torch.zeros(1, latent_dim))
        self.register_buffer("latent_std", torch.ones(1, latent_dim))

    def encode(self, s, a):
        x = torch.cat([s, a], dim=-1)
        mu, log_std = self.encoder(x).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, min=-4, max=15)
        return mu, torch.exp(log_std)

    def decode(self, s, z=None):
        if z is None:
            z = torch.normal(self.latent_mean, self.latent_std).repeat(s.shape[0], 1).to(s.device)
        return torch.tanh(self.decoder(torch.cat([s, z], dim=-1))) * self.max_action

    def forward(self, s, a):
        mu, std = self.encode(s, a)
        z = mu + std * torch.randn_like(std)
        a_recon = self.decode(s, z)
        return a_recon, mu, std


class PerturbationNet(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim=32, hidden=256, max_action=1.0, phi=0.05):
        super().__init__()
        self.max_action = max_action; self.phi = phi
        self.net = MLP(state_dim + action_dim + latent_dim, action_dim, hidden)
        self.zdim = latent_dim

    def forward(self, s, a):
        z = torch.normal(0, 1, (s.shape[0], self.zdim)).to(s.device)
        delta = self.phi * self.max_action * torch.tanh(self.net(torch.cat([s, a, z], dim=-1)))
        return torch.clamp(a + delta, -self.max_action, self.max_action)


class CriticTwin(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        self.q1 = MLP(state_dim + action_dim, 1, hidden)
        self.q2 = MLP(state_dim + action_dim, 1, hidden)
    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1); return self.q1(x), self.q2(x)
    def min_q(self, s, a):
        q1, q2 = self(s, a); return torch.min(q1, q2)


class BCQ:
    def __init__(self, state_dim, action_dim, max_action, device, config: TrainConfig):
        self.device = device; self.total_it = 0; self.max_action = max_action
        self.cfg = config; self.latent_dim = config.latent_dim

        self.vae = VAE(state_dim, action_dim, config.latent_dim, max_action=max_action).to(device)
        self.perturb = PerturbationNet(state_dim, action_dim, config.latent_dim,
                                        max_action=max_action, phi=config.phi).to(device)
        self.critic = CriticTwin(state_dim, action_dim).to(device)
        self.critic_tgt = copy.deepcopy(self.critic)

        self.vae_opt = optim.Adam(self.vae.parameters(), lr=config.vae_lr)
        self.perturb_opt = optim.Adam(self.perturb.parameters(), lr=config.actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        self.actor = self  # compatibility with the eval interface

    def select_action(self, state):
        with torch.no_grad():
            a = self.vae.decode(state)
            a = self.perturb(state, a)
        return a

    def train(self, batch):
        self.total_it += 1
        s, a, r, ns, not_done = [b.to(self.device) for b in batch]

        # VAE
        a_recon, mu, std = self.vae(s, a)
        recon_loss = F.mse_loss(a_recon, a)
        kl_loss = -0.5 * (1 + torch.log(std.pow(2) + 1e-6) - mu.pow(2) - std.pow(2)).mean()
        vae_loss = recon_loss + self.cfg.beta * kl_loss
        self.vae_opt.zero_grad(); vae_loss.backward(); self.vae_opt.step()

        # Critic
        with torch.no_grad():
            a_next = self.vae.decode(ns)
            a_next_p = self.perturb(ns, a_next)
            target_q = r + not_done * self.cfg.discount * self.critic_tgt.min_q(ns, a_next_p)
        q1, q2 = self.critic(s, a)
        critic_loss = (1 - self.cfg.lmbda) * (F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q))
        critic_loss += self.cfg.lmbda * F.mse_loss(torch.min(q1, q2), torch.zeros_like(q1))
        self.critic_opt.zero_grad(); critic_loss.backward(); self.critic_opt.step()

        # soft update target
        for p, tp in zip(self.critic.parameters(), self.critic_tgt.parameters()):
            tp.data.mul_(1 - self.cfg.tau).add_(self.cfg.tau * p.data)

        # Perturbation
        a_base = self.vae.decode(s)
        a_pert = self.perturb(s, a_base)
        perturb_loss = -self.critic.min_q(s, a_pert).mean()
        self.perturb_opt.zero_grad(); perturb_loss.backward(); self.perturb_opt.step()

        return {"vae_loss": vae_loss.item(), "critic_loss": critic_loss.item(),
                "perturb_loss": perturb_loss.item(), "total_it": self.total_it}


def eval_actor(env, actor, device, n_episodes, seed):
    actor.vae.eval(); actor.perturb.eval(); actor.critic.eval()
    env.seed(seed)
    scores = []
    for _ in range(n_episodes):
        s, done = env.reset(), False
        ret = 0.0
        while not done:
            s_t = torch.tensor(s.reshape(1, -1), dtype=torch.float32, device=device)
            a = actor.select_action(s_t).cpu().numpy().flatten()
            out = env.step(a)
            s, r, done = out[0], out[1], out[2] or out[3] if len(out) >= 4 else out[2]
            ret += r
        scores.append(ret)
    actor.vae.train(); actor.perturb.train(); actor.critic.train()
    return np.mean(scores)


def train(config: TrainConfig):
    set_seed(config.seed)
    env = gym.make(config.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    dataset = d4rl.qlearning_dataset(env)

    if config.delay_step > 0:
        sparsify_reward(dataset, config.delay_step, config.norm_reward)

    if config.normalize:
        state_mean = dataset["observations"].mean(0)
        state_std = dataset["observations"].std(0) + 1e-3
    else:
        state_mean, state_std = 0.0, 1.0

    dataset["observations"] = (dataset["observations"] - state_mean) / state_std
    dataset["next_observations"] = (dataset["next_observations"] - state_mean) / state_std

    replay_buffer = ReplayBuffer(state_dim, action_dim, config.buffer_size, config.device)
    replay_buffer.load_d4rl_dataset(dataset)

    orig_env = env
    class WrappedEnv:
        def seed(self, s): orig_env.seed(s)
        def reset(self):
            s, _ = orig_env.reset() if hasattr(orig_env, 'reset') else (orig_env.reset(), {})
            return (s - state_mean) / state_std
        def step(self, a):
            out = orig_env.step(a)
            s, r = out[0], out[1]
            done = out[2] or out[3] if len(out) >= 4 else out[2]
            return (s - state_mean) / state_std, r, done, {}
    env = WrappedEnv()

    trainer = BCQ(state_dim, action_dim, max_action, config.device, config)
    if config.save_dir is not None:
        os.makedirs(config.save_dir, exist_ok=True)
        csv_path = os.path.join(config.save_dir, f"bcq_seed{config.seed}.csv")
        csv_fh = open(csv_path, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_fh)
        csv_writer.writerow(["step", "norm_score"])
    else:
        csv_fh = None
        csv_writer = None

    print("---------------------------------------")
    print(f"Training BCQ, Env: {config.env}, Seed: {config.seed}")
    print(f"delay_step: {config.delay_step}, norm_reward: {config.norm_reward}")
    print("---------------------------------------")

    evaluations = []
    for t in range(int(config.max_timesteps)):
        batch = replay_buffer.sample(config.batch_size)
        log_dict = trainer.train(batch)
        if (t + 1) % config.eval_freq == 0:
            print(f"Time steps: {t + 1}")
            eval_score = eval_actor(env, trainer, config.device, config.n_episodes, config.seed)
            normalized = gym.make(config.env).get_normalized_score(eval_score) * 100.0
            evaluations.append(normalized)
            if csv_writer is not None:
                csv_writer.writerow([t + 1, normalized])
                csv_fh.flush()
            print(f"  Score: {eval_score:.1f}, D4RL: {normalized:.1f}")
            print("---------------------------------------")
            if config.checkpoints_path is not None:
                torch.save({"vae": trainer.vae.state_dict(),
                            "perturb": trainer.perturb.state_dict(),
                            "critic": trainer.critic.state_dict()},
                           os.path.join(config.checkpoints_path, f"checkpoint_{t}.pt"))

    if csv_fh is not None:
        csv_fh.close()
    return evaluations


def main():
    config = pyrallis.parse(config_class=TrainConfig)
    train(config)


if __name__ == "__main__":
    main()
