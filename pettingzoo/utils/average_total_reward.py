"""
平均总奖励计算模块。

这个模块提供了计算环境中智能体平均总奖励的功能。
主要用于评估智能体的性能和环境的难度。

主要功能：
1. 计算多个回合的平均总奖励
2. 支持并行和AEC环境
3. 提供自定义策略支持
"""

from typing import Any, Callable, Dict, Optional, Union

import numpy as np

from pettingzoo.utils.env import AECEnv, ParallelEnv


def average_total_reward(
    env: Union[AECEnv, ParallelEnv],
    max_episodes: int = 100,
    max_steps: Optional[int] = None,
    *,
    policy: Optional[Callable[[Any, Dict], Any]] = None,
) -> float:
    """计算环境中所有智能体的平均总奖励。

    这个函数运行多个回合的游戏，并计算所有智能体获得的总奖励的平均值。
    如果没有提供策略函数，将使用随机动作。

    参数：
        env (Union[AECEnv, ParallelEnv]): 要评估的环境
        max_episodes (int): 要运行的最大回合数，默认为100
        max_steps (Optional[int]): 每个回合的最大步数，默认为None（无限制）
        policy (Optional[Callable]): 策略函数，接受观察和动作空间作为输入，返回动作

    返回：
        float: 所有智能体在所有回合中的平均总奖励

    示例：
        >>> env = your_environment()
        >>> avg_reward = average_total_reward(env, max_episodes=50)
        >>> print(f"平均总奖励: {avg_reward}")
    """
    if policy is None:

        def policy(observation: Any, action_space: Any) -> Any:
            """默认的随机策略。

            参数：
                observation: 环境观察
                action_space: 动作空间

            返回：
                Any: 随机选择的动作
            """
            return action_space.sample()

    total_reward = 0
    total_steps = 0
    episodes = 0

    for episode in range(max_episodes):
        observations, infos = env.reset()
        episode_reward = 0
        steps = 0

        while True:
            if isinstance(env, ParallelEnv):
                # 并行环境处理
                actions = {
                    agent: policy(observations[agent], env.action_space(agent))
                    for agent in env.agents
                }
                observations, rewards, terminations, truncations, infos = env.step(actions)
                episode_reward += sum(rewards.values())
            else:
                # AEC环境处理
                for agent in env.agent_iter():
                    observation, reward, termination, truncation, info = env.last()
                    episode_reward += reward

                    if termination or truncation:
                        action = None
                    else:
                        action = policy(observation, env.action_space(agent))
                    env.step(action)

            steps += 1
            if max_steps is not None and steps >= max_steps:
                break

            if isinstance(env, ParallelEnv):
                if all(terminations.values()) or all(truncations.values()):
                    break
            else:
                if all(env.terminations.values()) or all(env.truncations.values()):
                    break

        total_reward += episode_reward
        total_steps += steps
        episodes += 1

    average_reward = total_reward / episodes

    return average_reward
