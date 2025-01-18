"""
手动控制策略模块。

这个模块提供了一个手动控制智能体的策略类。
主要用于调试和演示环境功能。

主要功能：
1. 提供键盘控制接口
2. 支持多个智能体的手动控制
3. 实现基本的动作映射

键盘控制映射：
- A-L: 控制前12个活塞
- 分号: 控制第13个活塞
- 引号: 控制第14个活塞
- 回车: 控制第15个活塞
"""

import numpy as np
import pygame

from pettingzoo.butterfly.pistonball.pistonball import WINDOW_HEIGHT, WINDOW_WIDTH


class ManualPolicy:
    """手动控制策略类。

    这个类实现了一个基于键盘输入的手动控制策略。
    用户可以使用键盘来控制智能体的行为。

    属性:
        agent_id_mapping (dict): 智能体ID到控制键的映射
    """

    def __init__(self, env, agent_id: int = 0, show_obs: bool = False):
        """初始化手动控制策略。

        设置智能体ID到控制键的映射关系。

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

        # 设置智能体到按键的映射
        self.agent_id_mapping = {
            "piston_0": pygame.K_a,
            "piston_1": pygame.K_s,
            "piston_2": pygame.K_d,
            "piston_3": pygame.K_f,
            "piston_4": pygame.K_g,
            "piston_5": pygame.K_h,
            "piston_6": pygame.K_j,
            "piston_7": pygame.K_k,
            "piston_8": pygame.K_l,
            "piston_9": pygame.K_SEMICOLON,
            "piston_10": pygame.K_QUOTE,
            "piston_11": pygame.K_RETURN,
            "piston_12": pygame.K_BACKSPACE,
            "piston_13": pygame.K_DELETE,
            "piston_14": pygame.K_END,
        }

    def __call__(self, observation, agent):
        """执行手动控制策略。

        根据键盘输入决定智能体的动作。

        参数：
            observation: 环境观察，包含当前状态信息
            agent (str): 智能体名称，用于确定控制键

        返回：
            int: 选择的动作
                1: 向上移动活塞
                0: 保持活塞位置不变
                -1: 向下移动活塞

        注意：
            如果用户关闭游戏窗口，将抛出异常
        """
        # 处理Pygame事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise Exception("用户关闭了游戏窗口")
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

        # 获取当前按键状态
        keys = pygame.key.get_pressed()

        # 检查特定智能体的控制键
        if agent in self.agent_id_mapping:
            key = self.agent_id_mapping[agent]
            if keys[key]:
                return 1  # 向上移动活塞

        return self.default_action  # 默认保持位置不变

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
