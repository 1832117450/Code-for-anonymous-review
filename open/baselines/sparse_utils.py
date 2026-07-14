"""Delayed sparse-reward utilities for adding --delay-step support to CORL algorithms"""
import numpy as np


def sparsify_reward(dataset: dict, delay_step: int, norm_reward: bool = False):
    """Convert dense rewards to delayed sparse rewards in place by modifying dataset["rewards"].

    Args:
        dataset: dict returned by d4rl qlearning_dataset, containing rewards, terminals, and timeouts
        delay_step: reward delay interval; accumulated reward is emitted every delay_step steps
        norm_reward: whether to normalize accumulated rewards by delay_step
    """
    rewards = dataset["rewards"].copy()
    terminals = dataset["terminals"].astype(bool)
    timeouts = dataset.get("timeouts", np.zeros_like(rewards, dtype=np.float32)).astype(bool)

    N = len(rewards)
    sparse_rewards = np.zeros(N, dtype=np.float32)
    accumulated, step_counter = 0.0, 0

    for i in range(N):
        accumulated += rewards[i]
        step_counter += 1
        if step_counter == delay_step or terminals[i] or timeouts[i]:
            sparse_rewards[i] = accumulated / step_counter if norm_reward else accumulated
            accumulated, step_counter = 0.0, 0

    dataset["rewards"] = sparse_rewards
    nonzero = np.count_nonzero(sparse_rewards)
    print(f"  Sparsify rewards: delay={delay_step} | active steps {nonzero}/{N} ({nonzero/N*100:.1f}%)"
          f" | norm={norm_reward} | mean_sparse={sparse_rewards[sparse_rewards!=0].mean():.2f}")
    return dataset
