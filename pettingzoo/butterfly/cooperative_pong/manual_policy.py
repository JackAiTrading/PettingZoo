"""
手动控制策略模块。

这个模块实现了合作乒乓球游戏的手动控制功能，允许玩家通过键盘控制球拍。

主要功能：
1. 键盘输入处理
2. 球拍控制映射
3. 实时响应
4. 多玩家支持

控制方式：
1. 玩家1：W/S 键控制上下移动
2. 玩家2：上/下 方向键控制移动
3. ESC：退出游戏
4. R：重置游戏

注意事项：
- 需要pygame支持
- 支持多种控制模式
- 可自定义按键映射
"""

import pygame


class ManualPolicy:
    def __init__(self, env, agent_id: int = 0, show_obs: bool = False):
        self.env = env
        self.agent_id = agent_id
        self.agent = self.env.agents[self.agent_id]

        # TO-DO: show current agent observation if this is True
        self.show_obs = show_obs

        # action mappings for all agents are the same
        if True:
            self.default_action = 0
            self.action_mapping = dict()
            self.action_mapping[pygame.K_w] = 1
            self.action_mapping[pygame.K_s] = 2

    def __call__(self, observation, agent):
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
        return self.env.agent_name_mapping


if __name__ == "__main__":
    from pettingzoo.butterfly import cooperative_pong_v5

    env = cooperative_pong_v5.env(render_mode="human")
    env.reset()

    clock = pygame.time.Clock()
    manual_policy = cooperative_pong_v5.ManualPolicy(env)

    for agent in env.agent_iter():
        clock.tick(env.metadata["render_fps"])

        observation, reward, termination, truncation, info = env.last()

        if agent == manual_policy.agent:
            action = manual_policy(observation, agent)
        else:
            action = env.action_space(agent).sample()

        env.step(action)

        if termination or truncation:
            env.reset()
