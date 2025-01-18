"""
手动控制策略模块。

这个模块实现了一个手动控制策略，允许用户通过键盘来控制活塞球游戏中的活塞。
用户可以使用键盘上的特定按键来选择和移动活塞。

键盘控制：
- 'a': 选择左边的活塞
- 'd': 选择右边的活塞
- 'w': 向上移动当前选中的活塞
- 's': 向下移动当前选中的活塞
"""

import numpy as np
import pygame


class ManualPolicy:
    """手动控制策略类。

    这个类实现了一个手动控制策略，允许用户通过键盘来控制活塞。

    属性：
        agent_selection (int): 当前选中的活塞索引
        env: 环境实例
        agent_id (int): 智能体标识符
        agent: 智能体实例
        show_obs (bool): 是否显示当前观察
        default_action (int): 默认动作
        action_mapping (dict): 动作映射
    """

    def __init__(self, env, agent_id: int = 0, show_obs: bool = False):
        """初始化手动控制策略。

        参数：
            env: 游戏环境实例
            agent_id (int, 可选): 初始选中的活塞ID
            show_obs (bool, 可选): 是否显示当前观察
        """
        self.env = env
        self.agent_id = agent_id
        self.agent = self.env.agents[self.agent_id]
        self.agent_selection = agent_id
        self.show_obs = show_obs
        self.default_action = 0
        self.action_mapping = dict()
        self.action_mapping[pygame.K_w] = 1.0
        self.action_mapping[pygame.K_s] = -1.0
        pygame.key.set_repeat(500, 100)

    def __call__(self, observation, agent):
        """执行手动控制策略。

        根据用户的键盘输入返回相应的动作。

        参数：
            observation: 当前环境的观察（未使用）
            agent: 智能体标识符

        返回：
            int: 动作编号
                0: 不动作
                1: 向上移动
                2: 向下移动
        """
        # 仅在我们是正确的智能体时触发
        assert (
            agent == self.agent
        ), f"手动策略仅适用于智能体：{self.agent}，但收到了 {agent} 的标签。"

        # 获取所有按键事件
        for event in pygame.event.get():
            # 如果是按键事件
            if event.type == pygame.KEYDOWN:
                # 选择活塞
                if event.key == pygame.K_a:  # 左移
                    self.agent_selection = max(0, self.agent_selection - 1)
                elif event.key == pygame.K_d:  # 右移
                    self.agent_selection = min(
                        self.env.n_pistons - 1, self.agent_selection + 1
                    )

                # 移动活塞
                if event.key == pygame.K_w:  # 向上
                    return 1
                elif event.key == pygame.K_s:  # 向下
                    return 2

                # 按 ESC 退出
                if event.key == pygame.K_ESCAPE:
                    exit()

                # 按退格键重置
                elif event.key == pygame.K_BACKSPACE:
                    self.env.reset()

        return self.default_action  # 默认动作：不移动

    @property
    def available_agents(self):
        """获取可用的智能体列表。

        返回：
            list: 智能体标识符列表
        """
        return self.env.agent_name_mapping


def manual_control(**kwargs):
    """手动控制环境入口函数。

    创建环境实例并运行手动控制循环。

    参数：
        **kwargs: 传递给环境的参数

    返回：
        env: 环境实例
    """
    from pettingzoo.butterfly import pistonball_v6

    env = pistonball_v6.env(render_mode="human")
    env.reset()

    clock = pygame.time.Clock()
    manual_policy = pistonball_v6.ManualPolicy(env)

    for agent in env.agent_iter():
        clock.tick(env.metadata["render_fps"])

        observation, reward, termination, truncation, info = env.last()

        if agent == manual_policy.agent:
            action = manual_policy(observation, agent)
        else:
            action = env.action_space(agent).sample()

        action = np.array(action)
        action = action.reshape(
            1,
        )
        env.step(action)

        if termination or truncation:
            env.reset()


if __name__ == "__main__":
    manual_control()
