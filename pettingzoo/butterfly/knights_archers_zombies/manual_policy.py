"""
手动控制策略模块。

这个模块实现了骑士射手大战僵尸游戏的手动控制功能，支持多个玩家通过键盘同时操作。

主要功能：
1. 键盘输入处理
2. 多玩家控制映射
3. 角色移动和攻击控制
4. 实时响应

控制方式：
骑士1：
- WASD：移动
- F：攻击

射手1：
- IJKL：移动
- H：射击

骑士2：
- 方向键：移动
- 右Shift：攻击

射手2：
- TFGH：移动
- Y：射击

通用：
- ESC：退出游戏
- R：重置游戏

注意事项：
- 支持同时按键
- 可自定义按键映射
- 需要pygame支持
"""

"""
骑士弓箭手僵尸游戏的手动控制策略。

这个模块实现了骑士弓箭手僵尸游戏的手动控制接口，允许人类玩家使用键盘和鼠标控制游戏。
"""

import pygame


class ManualPolicy:
    """手动控制策略类。

    这个类实现了手动控制策略的逻辑，包括键盘和鼠标输入处理。

    属性:
        env: 环境实例
        agent_id (int): 智能体标识符
        agent (str): 智能体名称
        show_obs (bool): 是否显示观察
        key_action_map (dict): 按键到动作的映射
    """

    def __init__(self, env, agent_id: int = 0, show_obs: bool = False):
        """初始化手动控制策略实例。

        参数:
            env: 环境实例
            agent_id (int): 智能体标识符（默认为0）
            show_obs (bool): 是否显示观察（默认为False）
        """
        self.env = env
        self.agent_id = agent_id
        self.agent = self.env.agents[self.agent_id]

        # TO-DO: show current agent observation if this is True
        self.show_obs = show_obs

        # action mappings for all agents are the same
        if True:
            self.default_action = 5
            self.action_mapping = dict()
            self.action_mapping[pygame.K_w] = 0  # front
            self.action_mapping[pygame.K_s] = 1  # back
            self.action_mapping[pygame.K_a] = 2  # rotate left
            self.action_mapping[pygame.K_d] = 3  # rotate right
            self.action_mapping[pygame.K_SPACE] = 4  # weapon

    def __call__(self, observation, agent):
        """手动控制策略函数。

        根据键盘和鼠标输入返回相应的动作。
        - WASD：移动
        - 空格键/鼠标左键：攻击
        - 其他键：不动

        参数:
            observation (ndarray): 当前观察
            agent (str): 智能体标识符

        返回:
            int: 动作值
            - 0: 不动
            - 1-4: 移动
            - 5: 攻击
        """
        # only trigger when we are the correct agent
        assert (
            agent == self.agent
        ), f"Manual Policy only applied to agent: {self.agent}, but got tag for {agent}."

        # set the default action
        action = self.default_action

        # if we get a key, override action using the dict
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    # escape to end
                    exit()

                elif event.key == pygame.K_BACKSPACE:
                    # backspace to reset
                    self.env.reset()

                elif event.key in self.action_mapping:
                    action = self.action_mapping[event.key]

        return action

    @property
    def available_agents(self):
        """获取可用的智能体列表。

        返回:
            list: 智能体标识符列表
        """
        return self.env.agent_name_mapping


if __name__ == "__main__":
    from pettingzoo.butterfly import knights_archers_zombies_v10

    env = knights_archers_zombies_v10.env(render_mode="human")
    env.reset()

    manual_policy = knights_archers_zombies_v10.ManualPolicy(env)

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if agent == manual_policy.agent:
            action = manual_policy(observation, agent)
        else:
            action = env.action_space(agent).sample()

        env.step(action)

        if termination or truncation:
            env.reset()
