import numpy as np
import pygame


class ManualPolicy:
    def __init__(self, env, agent_id: int = 0, show_obs: bool = False):
        self.env = env
        self.agent_id = agent_id
        self.agent = self.env.agents[self.agent_id]

        # 待办：如果这个值为 True，显示当前智能体的观察
        self.show_obs = show_obs

        # 所有智能体的动作映射都是相同的
        if True:
            self.default_action = 0
            self.action_mapping = dict()
            self.action_mapping[pygame.K_w] = 1.0
            self.action_mapping[pygame.K_s] = -1.0

    def __call__(self, observation, agent):
        # 仅在我们是正确的智能体时触发
        assert (
            agent == self.agent
        ), f"手动策略仅适用于智能体：{self.agent}，但收到了 {agent} 的标签。"

        # 设置默认动作
        action = self.default_action

        # 如果我们收到一个按键，使用字典覆盖动作
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    # 按 ESC 退出
                    exit()

                elif event.key == pygame.K_BACKSPACE:
                    # 按退格键重置
                    self.env.reset()

                elif event.key in self.action_mapping:
                    action = self.action_mapping[event.key]

        return action

    @property
    def available_agents(self):
        return self.env.agent_name_mapping


if __name__ == "__main__":
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
