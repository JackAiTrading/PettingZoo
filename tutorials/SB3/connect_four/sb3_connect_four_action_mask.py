"""使用 Stable-Baselines3 在四子连珠环境中训练智能体，使用无效动作掩码。

有关 PettingZoo 中无效动作掩码的信息，请参见 https://pettingzoo.farama.org/api/aec/#action-masking
有关 SB3 中无效动作掩码的更多信息，请参见 https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html

作者: Elliot (https://github.com/elliottower)
"""
import glob
import os
import time

import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker

import pettingzoo.utils
from pettingzoo.classic import connect_four_v3


class SB3ActionMaskWrapper(pettingzoo.utils.BaseWrapper):
    """包装器，允许 PettingZoo 环境与 SB3 非法动作掩码一起使用。"""

    def reset(self, seed=None, options=None):
        """类似 Gymnasium 的重置函数，为每个智能体分配相同的观察/动作空间。

        这是必需的，因为 SB3 是为单智能体强化学习设计的，不期望观察/动作空间是函数
        """
        super().reset(seed, options)

        # 从观察空间中去除动作掩码
        self.observation_space = super().observation_space(self.possible_agents[0])[
            "observation"
        ]
        self.action_space = super().action_space(self.possible_agents[0])

        # 返回初始观察和信息（PettingZoo AEC 环境默认不返回）
        return self.observe(self.agent_selection), {}

    def step(self, action):
        """类似 Gymnasium 的步进函数，返回观察、奖励、终止、截断、信息。

        观察是针对下一个智能体的（用于确定下一个动作），而其余项目是针对刚刚行动的智能体的（用于理解刚刚发生了什么）。
        """
        current_agent = self.agent_selection

        super().step(action)

        next_agent = self.agent_selection
        return (
            self.observe(next_agent),
            self._cumulative_rewards[current_agent],
            self.terminations[current_agent],
            self.truncations[current_agent],
            self.infos[current_agent],
        )

    def observe(self, agent):
        """仅返回原始观察，移除动作掩码。"""
        return super().observe(agent)["observation"]

    def action_mask(self):
        """用于访问动作掩码的独立函数。"""
        return super().observe(self.agent_selection)["action_mask"]


def mask_fn(env):
    # 在这个函数中执行任何您想要的操作来返回当前环境的动作掩码
    # 在这个例子中，我们假设环境有一个我们可以依赖的有用方法。
    return env.action_mask()


def train_action_mask(env_fn, steps=10_000, seed=0, **env_kwargs):
    """训练一个模型在零和游戏环境中作为每个智能体进行游戏，使用无效动作掩码。"""
    env = env_fn.env(**env_kwargs)

    print(f"开始在 {str(env.metadata['name'])} 上训练。")

    # 自定义包装器，将 PettingZoo 环境转换为可与 SB3 动作掩码一起工作
    env = SB3ActionMaskWrapper(env)

    env.reset(seed=seed)  # 必须调用 reset() 以重新定义空间

    env = ActionMasker(env, mask_fn)  # 包装以启用掩码（SB3 函数）
    # MaskablePPO 的行为与 SB3 的 PPO 相同，除非环境被包装
    # 使用 ActionMasker。如果检测到包装器，掩码会自动
    # 被检索并在学习时使用。注意 MaskablePPO 不接受
    # 新的 action_mask_fn 关键字参数，这与早期草稿不同。
    model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1)
    model.set_random_seed(seed)
    model.learn(total_timesteps=steps)

    model.save(f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}")

    print("模型已保存。")

    print(f"完成在 {str(env.unwrapped.metadata['name'])} 上的训练。\n")

    env.close()


def eval_action_mask(env_fn, num_games=100, render_mode=None, **env_kwargs):
    # 评估训练过的智能体与随机智能体的对抗
    env = env_fn.env(render_mode=render_mode, **env_kwargs)

    print(
        f"开始评估与随机智能体的对抗。训练过的智能体将作为 {env.possible_agents[1]} 进行游戏。"
    )

    try:
        latest_policy = max(
            glob.glob(f"{env.metadata['name']}*.zip"), key=os.path.getctime
        )
    except ValueError:
        print("策略未找到。")
        exit(0)

    model = MaskablePPO.load(latest_policy)

    scores = {agent: 0 for agent in env.possible_agents}
    total_rewards = {agent: 0 for agent in env.possible_agents}
    round_rewards = []

    for i in range(num_games):
        env.reset(seed=i)
        env.action_space(env.possible_agents[0]).seed(i)

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            # 分离观察和动作掩码
            observation, action_mask = obs.values()

            if termination or truncation:
                # 如果有赢家，记录下来，否则不改变分数（平局）
                if (
                    env.rewards[env.possible_agents[0]]
                    != env.rewards[env.possible_agents[1]]
                ):
                    winner = max(env.rewards, key=env.rewards.get)
                    scores[winner] += env.rewards[
                        winner
                    ]  # 只记录最大奖励（游戏赢家）
                # 同时跟踪负面和正面奖励（惩罚非法移动）
                for a in env.possible_agents:
                    total_rewards[a] += env.rewards[a]
                # 按回合列出奖励，供参考
                round_rewards.append(env.rewards)
                break
            else:
                if agent == env.possible_agents[0]:
                    act = env.action_space(agent).sample(action_mask)
                else:
                    # 注意：PettingZoo 期望整数动作 # TODO：是否将国际象棋更改为将动作转换为整数类型？
                    act = int(
                        model.predict(
                            observation, action_masks=action_mask, deterministic=True
                        )[0]
                    )
            env.step(act)
    env.close()

    # 避免除以零
    if sum(scores.values()) == 0:
        winrate = 0
    else:
        winrate = scores[env.possible_agents[1]] / sum(scores.values())
    print("每轮奖励： ", round_rewards)
    print("总奖励（包括负面奖励）： ", total_rewards)
    print("胜率： ", winrate)
    print("最终得分： ", scores)
    return round_rewards, total_rewards, winrate, scores


if __name__ == "__main__":
    if gym.__version__ > "0.29.1":
        raise ImportError(
            f"此脚本需要 gymnasium 版本 0.29.1 或更低，但您有版本 {gym.__version__}。"
        )

    env_fn = connect_four_v3

    env_kwargs = {}

    # 评估/训练超参数说明：
    # 10k 步骤：胜率：0.76，损失量级为 1e-03
    # 20k 步骤：胜率：0.86，损失量级为 1e-04
    # 40k 步骤：胜率：0.86，损失量级为 7e-06

    # 训练模型与自身对抗（在笔记本 CPU 上大约需要 20 秒）
    train_action_mask(env_fn, steps=20_480, seed=0, **env_kwargs)

    # 对抗随机智能体评估 100 场游戏（胜率应该约为 80%）
    eval_action_mask(env_fn, num_games=100, render_mode=None, **env_kwargs)

    # 观看与随机智能体的两场游戏
    eval_action_mask(env_fn, num_games=2, render_mode="human", **env_kwargs)
