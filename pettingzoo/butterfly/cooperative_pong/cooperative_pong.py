# noqa: D212, D415
"""
# 合作乒乓球

```{figure} butterfly_cooperative_pong.gif
:width: 200px
:name: cooperative_pong
```

此环境是<a href='..'>蝴蝶环境</a>的一部分。请先阅读该页面以获取一般信息。

| 导入                 | `from pettingzoo.butterfly import cooperative_pong_v5` |
|---------------------|------------------------------------------------------|
| 动作                | 离散                                                  |
| 并行 API            | 是                                                   |
| 手动控制            | 是                                                   |
| 智能体              | `agents= ['paddle_0', 'paddle_1']`                   |
| 智能体数量          | 2                                                    |
| 动作形状            | Discrete(3)                                          |
| 动作值范围          | [0, 1]                                               |
| 观察形状            | (280, 480, 3)                                        |
| 观察值范围          | [0, 255]                                             |
| 状态形状            | (560, 960, 3)                                        |
| 状态值范围          | (0, 255)                                             |


合作乒乓球是一个简单的乒乓球游戏，目标是让球尽可能长时间地保持在场内。当球从屏幕左边或右边界出界时，游戏结束。游戏中有两个智能体（球拍），一个沿着左边界移动，另一个沿着右边界移动。球的所有碰撞都是弹性的。每次重置时，球都会从屏幕中心以随机方向开始移动。为了让学习更具挑战性，默认情况下右侧球拍是分层蛋糕形状的。
每个智能体的观察空间是屏幕的一半。智能体有两个可能的动作（向上/向下移动）。如果球保持在界内，每个智能体在每个时间步都会收到 `max_reward / max_cycles`（默认 0.11）的奖励。否则，每个智能体会收到 `off_screen_penalty`（默认 -10）的惩罚，游戏结束。


### 手动控制

使用 'W' 和 'S' 键移动左侧球拍。使用 '上' 和 '下' 方向键移动右侧球拍。

### 参数

``` python
cooperative_pong_v5.env(ball_speed=9, left_paddle_speed=12,
right_paddle_speed=12, cake_paddle=True, max_cycles=900, bounce_randomness=False, max_reward=100, off_screen_penalty=-10)
```

`ball_speed`：球的速度（以像素为单位）

`left_paddle_speed`：左侧球拍的速度（以像素为单位）

`right_paddle_speed`：右侧球拍的速度（以像素为单位）

`cake_paddle`：如果为 True，右侧球拍将呈现 4 层婚礼蛋糕的形状

`max_cycles`：经过 max_cycles 步后，所有智能体将返回完成状态

`bounce_randomness`：如果为 True，球与球拍的每次碰撞都会在球的方向上添加一个小的随机角度，球的速度保持不变。

`max_reward`：在 max_cycles 时间步内给予每个智能体的总奖励

`off_screen_penalty`：当球出界时给予每个智能体的负奖励惩罚


### 版本历史

* v5：修复了球瞬移的错误
* v4：添加了 max_reward 和 off_screen_penalty 参数并更改了默认值，修复了球偶尔瞬移的故障，重新设计了奖励机制 (1.14.0)
* v3：将观察空间更改为包含整个屏幕 (1.10.0)
* v2：其他修复 (1.4.0)
* v1：修复了 `dones` 计算中的错误 (1.3.1)
* v0：初始版本发布 (1.0.0)

"""

import gymnasium
import numpy as np
import pygame
from gymnasium.utils import EzPickle, seeding

from pettingzoo import AECEnv
from pettingzoo.butterfly.cooperative_pong.ball import Ball
from pettingzoo.butterfly.cooperative_pong.cake_paddle import CakePaddle
from pettingzoo.butterfly.cooperative_pong.manual_policy import ManualPolicy
from pettingzoo.butterfly.cooperative_pong.paddle import Paddle
from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import AgentSelector
from pettingzoo.utils.conversions import parallel_wrapper_fn

FPS = 15


__all__ = ["ManualPolicy", "env", "raw_env", "parallel_env"]


def deg_to_rad(deg):
    """将角度转换为弧度"""
    return deg * np.pi / 180


def get_flat_shape(width, height, kernel_window_length=2):
    """获取展平后的形状"""
    return int(width * height / (kernel_window_length * kernel_window_length))


def original_obs_shape(screen_width, screen_height, kernel_window_length=2):
    """获取原始观察空间的形状"""
    return (
        int(screen_height * 2 / kernel_window_length),
        int(screen_width * 2 / kernel_window_length),
        3,
    )


class CooperativePong:
    def __init__(
            self,
            randomizer,
            ball_speed=9,  # 球的速度
            left_paddle_speed=12,  # 左侧球拍速度
            right_paddle_speed=12,  # 右侧球拍速度
            cake_paddle=True,  # 是否使用蛋糕形状的球拍
            max_cycles=900,  # 最大周期数
            bounce_randomness=False,  # 是否添加碰撞随机性
            max_reward=100,  # 最大奖励
            off_screen_penalty=-10,  # 出界惩罚
            render_mode=None,  # 渲染模式
            render_ratio=2,  # 渲染比例
            kernel_window_length=2,  # 核窗口长度
            render_fps=15,  # 渲染帧率
        ):
        super().__init__()

        pygame.init()
        self.num_agents = 2

        self.render_ratio = render_ratio
        self.kernel_window_length = kernel_window_length

        # 显示屏幕
        self.s_width, self.s_height = 960 // render_ratio, 560 // render_ratio
        self.area = pygame.Rect(0, 0, self.s_width, self.s_height)
        self.max_reward = max_reward
        self.off_screen_penalty = off_screen_penalty

        # 定义动作和观察空间
        self.action_space = [
            gymnasium.spaces.Discrete(3) for _ in range(self.num_agents)
        ]
        original_shape = original_obs_shape(
            self.s_width, self.s_height, kernel_window_length=kernel_window_length
        )
        original_color_shape = (original_shape[0], original_shape[1], 3)
        self.observation_space = [
            gymnasium.spaces.Box(
                low=0, high=255, shape=(original_color_shape), dtype=np.uint8
            )
            for _ in range(self.num_agents)
        ]
        # 定义环境的全局空间或状态
        self.state_space = gymnasium.spaces.Box(
            low=0, high=255, shape=((self.s_height, self.s_width, 3)), dtype=np.uint8
        )

        self.render_mode = render_mode
        self.screen = None

        # 设置速度
        self.speed = [ball_speed, left_paddle_speed, right_paddle_speed]

        self.max_cycles = max_cycles

        # 球拍
        self.p0 = Paddle((20 // render_ratio, 80 // render_ratio), left_paddle_speed)
        if cake_paddle:
            self.p1 = CakePaddle(right_paddle_speed, render_ratio=render_ratio)
        else:
            self.p1 = Paddle(
                (20 // render_ratio, 100 // render_ratio), right_paddle_speed
            )

        self.agents = ["paddle_0", "paddle_1"]  # 智能体列表

        # 球
        self.ball = Ball(
            randomizer,
            (20 // render_ratio, 20 // render_ratio),
            ball_speed,
            bounce_randomness,
        )
        self.randomizer = randomizer

        self.reinit()

        self.render_fps = render_fps
        if self.render_mode == "human":
            self.clock = pygame.time.Clock()

    def reinit(self):
        self.rewards = dict(zip(self.agents, [0.0] * len(self.agents)))
        self.terminations = dict(zip(self.agents, [False] * len(self.agents)))
        self.truncations = dict(zip(self.agents, [False] * len(self.agents)))
        self.infos = dict(zip(self.agents, [{}] * len(self.agents)))
        self.score = 0

    def reset(self, seed=None, options=None):
        # 重置球和球拍的初始条件
        self.ball.rect.center = self.area.center
        # 设置方向为 [0, 2*np.pi) 之间的角度
        angle = get_valid_angle(self.randomizer)
        # angle = deg_to_rad(89)
        self.ball.speed = [
            int(self.ball.speed_val * np.cos(angle)),
            int(self.ball.speed_val * np.sin(angle)),
        ]

        self.p0.rect.midleft = self.area.midleft
        self.p1.rect.midright = self.area.midright
        self.p0.reset()
        self.p1.reset()
        self.p0.speed = self.speed[1]
        self.p1.speed = self.speed[2]

        self.terminate = False
        self.truncate = False

        self.num_frames = 0

        self.reinit()

        # Pygame 表面，即使在 render_mode == None 时，也需要从像素值中获取观察值
        # 观察
        if self.render_mode != "human":
            self.screen = pygame.Surface((self.s_width, self.s_height))

        self.render()

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "您正在调用 render 方法，而没有指定任何渲染模式。"
            )
            return

        if self.screen is None:
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode((self.s_width, self.s_height))
                pygame.display.set_caption("合作乒乓球")
        self.draw()

        observation = np.array(pygame.surfarray.pixels3d(self.screen))
        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.render_fps)
        return (
            np.transpose(observation, axes=(1, 0, 2))
            if self.render_mode == "rgb_array"
            else None
        )

    def observe(self):
        observation = np.array(pygame.surfarray.pixels3d(self.screen))
        observation = np.rot90(
            observation, k=3
        )  # 现在观察值的排列方式为 H, W 作为行和列
        observation = np.fliplr(observation)  # 按正确的顺序排列
        return observation

    def state(self):
        """返回环境的全局状态。"""
        state = pygame.surfarray.pixels3d(self.screen).copy()
        state = np.rot90(state, k=3)
        state = np.fliplr(state)
        return state

    def draw(self):
        pygame.draw.rect(self.screen, (0, 0, 0), self.area)
        self.p0.draw(self.screen)
        self.p1.draw(self.screen)
        self.ball.draw(self.screen)

    def step(self, action, agent):
        # 根据动作更新 p0, p1
        # 动作：0：不做任何事情，
        # 动作：1：p[i] 向上移动
        # 动作：2：p[i] 向下移动
        if agent == self.agents[0]:
            self.rewards = {a: 0 for a in self.agents}
            self.p0.update(self.area, action)
        elif agent == self.agents[1]:
            self.p1.update(self.area, action)

            # 如果没有终止，则执行其余操作
            if not self.terminate:
                # 更新球的位置
                self.terminate = self.ball.update2(self.area, self.p0, self.p1)

                # 执行最后一个智能体移动后的其他操作
                # 奖励是球在场内停留的时间长度
                reward = 0
                # 球出界
                if self.terminate:
                    reward = self.off_screen_penalty
                    self.score += reward
                if not self.terminate:
                    self.num_frames += 1
                    reward = self.max_reward / self.max_cycles
                    self.score += reward
                    self.truncate = self.num_frames >= self.max_cycles

                for ag in self.agents:
                    self.rewards[ag] = reward
                    self.terminations[ag] = self.terminate
                    self.truncations[ag] = self.truncate
                    self.infos[ag] = {}

        self.render()


def env(**kwargs):
    env = raw_env(**kwargs)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


parallel_env = parallel_wrapper_fn(env)


class raw_env(AECEnv, EzPickle):
    # class env(MultiAgentEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],  # 渲染模式
        "name": "cooperative_pong_v5",  # 环境名称
        "is_parallelizable": True,  # 是否可并行
        "render_fps": FPS,  # 渲染帧率
        "has_manual_policy": True,  # 是否有手动策略
    }

    def __init__(self, **kwargs):
        EzPickle.__init__(self, **kwargs)
        self._kwargs = kwargs

        self._seed()

        self.agents = self.env.agents[:]
        self.possible_agents = self.agents[:]
        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        # 空间
        self.action_spaces = dict(zip(self.agents, self.env.action_space))
        self.observation_spaces = dict(zip(self.agents, self.env.observation_space))
        self.state_space = self.env.state_space
        # 字典
        self.observations = {}
        self.rewards = self.env.rewards
        self.terminations = self.env.terminations
        self.truncations = self.env.truncations
        self.infos = self.env.infos

        self.score = self.env.score

        self.render_mode = self.env.render_mode
        self.screen = None

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    # def convert_to_dict(self, list_of_list):
    #     return dict(zip(self.agents, list_of_list))

    def _seed(self, seed=None):
        self.randomizer, seed = seeding.np_random(seed)
        self.env = CooperativePong(self.randomizer, **self._kwargs)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._seed(seed=seed)
        self.env.reset()
        self.agents = self.possible_agents[:]
        self.agent_selection = self._agent_selector.reset()
        self.rewards = self.env.rewards
        self._cumulative_rewards = {a: 0 for a in self.agents}
        self.terminations = self.env.terminations
        self.truncations = self.env.truncations
        self.infos = self.env.infos

    def observe(self, agent):
        obs = self.env.observe()
        return obs

    def state(self):
        state = self.env.state()
        return state

    def close(self):
        self.env.close()

    def render(self):
        return self.env.render()

    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return
        agent = self.agent_selection
        if not self.action_spaces[agent].contains(action):
            raise Exception(
                "智能体 {} 的动作必须在 Discrete({})."
                "它目前是 {}".format(agent, self.action_spaces[agent].n, action)
            )

        self.env.step(action, agent)
        # 选择下一个智能体并观察
        self.agent_selection = self._agent_selector.next()
        self.rewards = self.env.rewards
        self.terminations = self.env.terminations
        self.truncations = self.env.truncations
        self.infos = self.env.infos

        self.score = self.env.score

        self._cumulative_rewards[agent] = 0
        self._accumulate_rewards()


# 这个环境最初是由 Ananth Hari 在另一个仓库中完整创建的，
# 后来由 J K Terry 添加到这里（这就是为什么在 git 历史中显示他们是创建者的原因）
