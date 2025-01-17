# noqa: D212, D415
"""
# 活塞球（Pistonball）

```{figure} butterfly_pistonball.gif
:width: 200px
:name: pistonball
```

此环境是<a href='..'>蝴蝶环境</a>的一部分。请先阅读该页面以获取一般信息。

| 导入                 | `from pettingzoo.butterfly import pistonball_v6`     |
|---------------------|------------------------------------------------------|
| 动作                | 二者之一                                              |
| 并行 API            | 是                                                   |
| 手动控制            | 是                                                   |
| 智能体              | `agents= ['piston_0', 'piston_1', ..., 'piston_19']` |
| 智能体数量          | 20                                                   |
| 动作形状            | (1,)                                                 |
| 动作值范围          | [-1, 1]                                              |
| 观察形状            | (457, 120, 3)                                        |
| 观察值范围          | (0, 255)                                             |
| 状态形状            | (560, 880, 3)                                        |
| 状态值范围          | (0, 255)                                             |


这是一个简单的基于物理的合作游戏，目标是通过激活垂直移动的活塞将球移动到游戏边界的左墙。每个活塞智能体的观察是一个 RGB 图像，包含智能体旁边的两个活塞（或墙）以及它们上方的空间。每个活塞都可以在任何给定时间被操作。在离散模式下，动作空间为 0 表示向下移动，1 表示保持静止，2 表示向上移动。在连续模式下，[-1, 1] 范围内的值与活塞升高或降低的幅度成比例。连续动作按因子 4 进行缩放，因此在离散和连续动作空间中，动作 1 将使活塞向上移动 4 像素，-1 将使活塞向下移动 4 像素。

相应地，活塞必须学习高度协调的涌现行为以实现环境的最优策略。每个智能体获得的奖励是球整体向左移动的距离与球在靠近活塞时向左移动的距离（即活塞贡献的移动）的组合。当球的任何部分直接位于活塞下方时，该活塞被认为靠近球。平衡这些局部和全局奖励之间的比例似乎对学习这个环境至关重要，因此这是一个环境参数。应用的局部奖励是球的 x 位置变化的 0.5 倍。此外，全局奖励是 x 位置的变化除以起始位置，乘以 100，再加上 `time_penalty`（默认 -0.1）。对于每个活塞，奖励是 `local_ratio` * local_reward + (1-`local_ratio`) * global_reward。局部奖励应用于球周围的活塞，而全局奖励提供给所有活塞。

活塞球使用 chipmunk 物理引擎，因此物理效果与愤怒的小鸟游戏中的物理效果差不多真实。

按键 *a* 和 *d* 控制选择哪个活塞移动（最初选择最右边的活塞），按键 *w* 和 *s* 在垂直方向上移动活塞。


### 参数

``` python
pistonball_v6.env(n_pistons=20, time_penalty=-0.1, continuous=True,
random_drop=True, random_rotate=True, ball_mass=0.75, ball_friction=0.3,
ball_elasticity=1.5, max_cycles=125)
```

`n_pistons`: 环境中的活塞（智能体）数量。

`time_penalty`: 每个时间步给每个活塞添加的奖励量。值越高意味着越重视让球穿过屏幕以终止游戏。

`continuous`: 如果为真，活塞动作是 -1 到 1 之间的实数，该值会被添加到活塞高度。如果为假，则动作是向上或向下移动一个单位的离散值。

`random_drop`: 如果为真，球将在随机的 x 值处初始生成。如果为假，球将始终在 x=800 处生成。

`random_rotate`: 如果为真，球将以随机角动量生成。

`ball_mass`: 设置球物理对象的质量。

`ball_friction`: 设置球物理对象的摩擦力。

`ball_elasticity`: 设置球物理对象的弹性。

`max_cycles`: 经过 max_cycles 步后，所有智能体将返回完成状态。


### 版本历史

* v6: 修复球从左墙反弹的问题。
* v5: 由于物理引擎不精确导致球进入左列不再给予额外奖励
* v4: 更改了 `max_cycles` 和 `continuous` 的默认参数，升级 PyMunk 版本 (1.6.0)
* v3: 重构，添加了活塞数量参数，微小的视觉更改 (1.5.0)
* v2: 各种修复，升级 PyGame 和 PyMunk 版本 (1.4.0)
* v1: 修复连续模式 (1.0.1)
* v0: 初始版本发布 (1.0.0)

"""

import math

import gymnasium
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
from gymnasium.utils import EzPickle, seeding

from pettingzoo import AECEnv
from pettingzoo.butterfly.pistonball.manual_policy import ManualPolicy
from pettingzoo.utils import AgentSelector, wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn

_image_library = {}

FPS = 20

__all__ = ["ManualPolicy", "env", "parallel_env", "raw_env"]


def get_image(path):
    from os import path as os_path

    cwd = os_path.dirname(__file__)
    image = pygame.image.load(cwd + "/" + path)
    sfc = pygame.Surface(image.get_size(), flags=pygame.SRCALPHA)
    sfc.blit(image, (0, 0))
    return sfc


def env(**kwargs):
    env = raw_env(**kwargs)
    if env.continuous:
        env = wrappers.ClipOutOfBoundsWrapper(env)
    else:
        env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


parallel_env = parallel_wrapper_fn(env)


class raw_env(AECEnv, EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array"],  # 渲染模式：人类可视化和RGB数组
        "name": "pistonball_v6",  # 环境名称
        "is_parallelizable": True,  # 是否可并行化
        "render_fps": FPS,  # 渲染帧率
        "has_manual_policy": True,  # 是否有手动策略
    }

    def __init__(
        self,
        n_pistons=20,  # 活塞数量
        time_penalty=-0.1,  # 时间惩罚
        continuous=True,  # 是否为连续动作空间
        random_drop=True,  # 是否随机投放球
        random_rotate=True,  # 是否随机旋转球
        ball_mass=0.75,  # 球的质量
        ball_friction=0.3,  # 球的摩擦力
        ball_elasticity=1.5,  # 球的弹性
        max_cycles=125,  # 最大周期数
        render_mode=None,  # 渲染模式
    ):
        EzPickle.__init__(
            self,
            n_pistons=n_pistons,
            time_penalty=time_penalty,
            continuous=continuous,
            random_drop=random_drop,
            random_rotate=random_rotate,
            ball_mass=ball_mass,
            ball_friction=ball_friction,
            ball_elasticity=ball_elasticity,
            max_cycles=max_cycles,
            render_mode=render_mode,
        )
        self.dt = 1.0 / FPS
        self.n_pistons = n_pistons
        self.piston_head_height = 11
        self.piston_width = 40
        self.piston_height = 40
        self.piston_body_height = 23
        self.piston_radius = 5
        self.wall_width = 40
        self.ball_radius = 40
        self.screen_width = (2 * self.wall_width) + (self.piston_width * self.n_pistons)
        self.screen_height = 560
        y_high = self.screen_height - self.wall_width - self.piston_body_height
        y_low = self.wall_width
        obs_height = y_high - y_low

        assert (
            self.piston_width == self.wall_width
        ), "Wall width and piston width must be equal for observation calculation"
        assert self.n_pistons > 1, "n_pistons must be greater than 1"

        self.agents = ["piston_" + str(r) for r in range(self.n_pistons)]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.n_pistons))))
        self._agent_selector = AgentSelector(self.agents)

        self.observation_spaces = dict(
            zip(
                self.agents,
                [
                    gymnasium.spaces.Box(
                        low=0,
                        high=255,
                        shape=(obs_height, self.piston_width * 3, 3),
                        dtype=np.uint8,
                    )
                ]
                * self.n_pistons,
            )
        )
        self.continuous = continuous
        if self.continuous:
            self.action_spaces = dict(
                zip(
                    self.agents,
                    [gymnasium.spaces.Box(low=-1, high=1, shape=(1,))] * self.n_pistons,
                )
            )
        else:
            self.action_spaces = dict(
                zip(self.agents, [gymnasium.spaces.Discrete(3)] * self.n_pistons)
            )
        self.state_space = gymnasium.spaces.Box(
            low=0,
            high=255,
            shape=(self.screen_height, self.screen_width, 3),
            dtype=np.uint8,
        )

        pygame.init()
        pymunk.pygame_util.positive_y_is_up = False

        self.render_mode = render_mode
        self.renderOn = False
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.max_cycles = max_cycles

        self.piston_sprite = get_image("piston.png")
        self.piston_body_sprite = get_image("piston_body.png")
        self.background = get_image("background.png")
        self.random_drop = random_drop
        self.random_rotate = random_rotate

        self.pistonList = []
        self.pistonRewards = []  # 跟踪个体奖励
        self.recentFrameLimit = (
            20  # 定义"最近"在帧数上的含义
        )
        self.recentPistons = set()  # 最近接触球的活塞集合
        self.time_penalty = time_penalty
        # TODO: 这是一个糟糕的想法，这个逻辑应该在某个时候被移除
        self.local_ratio = 0
        self.ball_mass = ball_mass
        self.ball_friction = ball_friction
        self.ball_elasticity = ball_elasticity

        self.terminate = False
        self.truncate = False

        self.pixels_per_position = 4
        self.n_piston_positions = 16

        self.screen.fill((0, 0, 0))
        self.draw_background()
        # self.screen.blit(self.background, (0, 0))

        self.render_rect = pygame.Rect(
            self.wall_width,  # 左边界
            self.wall_width,  # 上边界
            self.screen_width - (2 * self.wall_width),  # 宽度
            self.screen_height
            - (2 * self.wall_width)
            - self.piston_body_height,  # 高度
        )

        # 当球超出边界时绘制背景图像。球的半径是 40
        self.valid_ball_position_rect = pygame.Rect(
            self.render_rect.left + self.ball_radius,  # 左边界
            self.render_rect.top + self.ball_radius,  # 上边界
            self.render_rect.width - (2 * self.ball_radius),  # 宽度
            self.render_rect.height - (2 * self.ball_radius),  # 高度
        )

        self.frames = 0

        self._seed()

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def observe(self, agent):
        observation = pygame.surfarray.pixels3d(self.screen)
        i = self.agent_name_mapping[agent]
        # 设置 x 边界以包含 40px 左边和 40px 右边的活塞
        x_high = self.wall_width + self.piston_width * (i + 2)
        x_low = self.wall_width + self.piston_width * (i - 1)
        y_high = self.screen_height - self.wall_width - self.piston_body_height
        y_low = self.wall_width
        cropped = np.array(observation[x_low:x_high, y_low:y_high, :])
        observation = np.rot90(cropped, k=3)
        observation = np.fliplr(observation)
        return observation

    def state(self):
        """返回全局环境的观察。"""
        state = pygame.surfarray.pixels3d(self.screen).copy()
        state = np.rot90(state, k=3)
        state = np.fliplr(state)
        return state

    def enable_render(self):
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Pistonball")

        self.renderOn = True
        # self.screen.blit(self.background, (0, 0))
        self.draw_background()
        self.draw()

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    def add_walls(self):
        top_left = (self.wall_width, self.wall_width)
        top_right = (self.screen_width - self.wall_width, self.wall_width)
        bot_left = (self.wall_width, self.screen_height - self.wall_width)
        bot_right = (
            self.screen_width - self.wall_width,
            self.screen_height - self.wall_width,
        )
        walls = [
            pymunk.Segment(self.space.static_body, top_left, top_right, 1),  # 顶部墙
            pymunk.Segment(self.space.static_body, top_left, bot_left, 1),  # 左边界
            pymunk.Segment(
                self.space.static_body, bot_left, bot_right, 1
            ),  # 底部边界
            pymunk.Segment(self.space.static_body, top_right, bot_right, 1),  # 右边界
        ]
        for wall in walls:
            wall.friction = 0.64
            self.space.add(wall)

    def add_ball(self, x, y, b_mass, b_friction, b_elasticity):
        mass = b_mass
        radius = 40
        inertia = pymunk.moment_for_circle(mass, 0, radius, (0, 0))
        body = pymunk.Body(mass, inertia)
        body.position = x, y
        # 每秒弧度
        if self.random_rotate:
            body.angular_velocity = self.np_random.uniform(-6 * math.pi, 6 * math.pi)
        shape = pymunk.Circle(body, radius, (0, 0))
        shape.friction = b_friction
        shape.elasticity = b_elasticity
        self.space.add(body, shape)
        return body

    def add_piston(self, space, x, y):
        piston = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        piston.position = x, y
        segment = pymunk.Segment(
            piston,
            (0, 0),
            (self.piston_width - (2 * self.piston_radius), 0),
            self.piston_radius,
        )
        segment.friction = 0.64
        segment.color = pygame.color.THECOLORS["blue"]
        space.add(piston, segment)
        return piston

    def move_piston(self, piston, v):
        def cap(y):
            maximum_piston_y = (
                self.screen_height
                - self.wall_width
                - (self.piston_height - self.piston_head_height)
            )
            if y > maximum_piston_y:
                y = maximum_piston_y
            elif y < maximum_piston_y - (
                self.n_piston_positions * self.pixels_per_position
            ):
                y = maximum_piston_y - (
                    self.n_piston_positions * self.pixels_per_position
                )
            return y

        piston.position = (
            piston.position[0],
            cap(piston.position[1] - v * self.pixels_per_position),
        )

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._seed(seed)
        self.space = pymunk.Space(threaded=False)
        self.add_walls()
        # self.space.threads = 2
        self.space.gravity = (0.0, 750.0)
        self.space.collision_bias = 0.0001
        self.space.iterations = 10  # 10 是 PyMunk 的默认值

        self.pistonList = []
        maximum_piston_y = (
            self.screen_height
            - self.wall_width
            - (self.piston_height - self.piston_head_height)
        )
        for i in range(self.n_pistons):
            # 将可能的 y 位移乘以 0.5，以使用可能位置的下半部分
            possible_y_displacements = np.arange(
                0,
                0.5 * self.pixels_per_position * self.n_piston_positions,
                self.pixels_per_position,
            )
            piston = self.add_piston(
                self.space,
                self.wall_width
                + self.piston_radius
                + self.piston_width * i,  # x 位置
                maximum_piston_y
                # y 位置
                - self.np_random.choice(possible_y_displacements),
            )
            piston.velociy = 0
            self.pistonList.append(piston)

        self.horizontal_offset = 0
        self.vertical_offset = 0
        horizontal_offset_range = 30
        vertical_offset_range = 15
        if self.random_drop:
            self.vertical_offset = self.np_random.integers(
                -vertical_offset_range, vertical_offset_range + 1
            )
            self.horizontal_offset = self.np_random.integers(
                -horizontal_offset_range, horizontal_offset_range + 1
            )
        ball_x = (
            self.screen_width
            - self.wall_width
            - self.ball_radius
            - horizontal_offset_range
            + self.horizontal_offset
        )
        ball_y = (
            self.screen_height
            - self.wall_width
            - self.piston_body_height
            - self.ball_radius
            - (0.5 * self.pixels_per_position * self.n_piston_positions)
            - vertical_offset_range
            + self.vertical_offset
        )

        # 确保球始终在左边界的右边
        ball_x = max(ball_x, self.wall_width + self.ball_radius + 1)

        self.ball = self.add_ball(
            ball_x, ball_y, self.ball_mass, self.ball_friction, self.ball_elasticity
        )
        self.ball.angle = 0
        self.ball.velocity = (0, 0)
        if self.random_rotate:
            self.ball.angular_velocity = self.np_random.uniform(
                -6 * math.pi, 6 * math.pi
            )

        self.lastX = int(self.ball.position[0] - self.ball_radius)
        self.distance = self.lastX - self.wall_width

        self.draw_background()
        self.draw()

        self.agents = self.possible_agents[:]

        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()

        self.terminate = False
        self.truncate = False
        self.rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self._cumulative_rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self.terminations = dict(zip(self.agents, [False for _ in self.agents]))
        self.truncations = dict(zip(self.agents, [False for _ in self.agents]))
        self.infos = dict(zip(self.agents, [{} for _ in self.agents]))

        self.frames = 0

    def draw_background(self):
        outer_walls = pygame.Rect(
            0,  # 左边界
            0,  # 上边界
            self.screen_width,  # 宽度
            self.screen_height,  # 高度
        )
        outer_wall_color = (58, 64, 65)
        pygame.draw.rect(self.screen, outer_wall_color, outer_walls)
        inner_walls = pygame.Rect(
            self.wall_width / 2,  # 左边界
            self.wall_width / 2,  # 上边界
            self.screen_width - self.wall_width,  # 宽度
            self.screen_height - self.wall_width,  # 高度
        )
        inner_wall_color = (68, 76, 77)
        pygame.draw.rect(self.screen, inner_wall_color, inner_walls)
        self.draw_pistons()

    def draw_pistons(self):
        piston_color = (65, 159, 221)
        x_pos = self.wall_width
        for piston in self.pistonList:
            self.screen.blit(
                self.piston_body_sprite,
                (x_pos, self.screen_height - self.wall_width - self.piston_body_height),
            )
            # 高度是活塞的蓝色部分的大小。6 是活塞底部的高度（灰色部分）
            height = (
                self.screen_height
                - self.wall_width
                - self.piston_body_height
                - (piston.position[1] + self.piston_radius)
                + (self.piston_body_height - 6)
            )
            body_rect = pygame.Rect(
                piston.position[0]
                + self.piston_radius
                + 1,  # +1 以匹配活塞图形
                piston.position[1] + self.piston_radius + 1,
                18,
                height,
            )
            pygame.draw.rect(self.screen, piston_color, body_rect)
            x_pos += self.piston_width

    def draw(self):
        if self.render_mode is None:
            return
        # 重新绘制背景图像，当球超出有效位置时
        if not self.valid_ball_position_rect.collidepoint(self.ball.position):
            # self.screen.blit(self.background, (0, 0))
            self.draw_background()

        ball_x = int(self.ball.position[0])
        ball_y = int(self.ball.position[1])

        color = (255, 255, 255)
        pygame.draw.rect(self.screen, color, self.render_rect)
        color = (65, 159, 221)
        pygame.draw.circle(self.screen, color, (ball_x, ball_y), self.ball_radius)

        line_end_x = ball_x + (self.ball_radius - 1) * np.cos(self.ball.angle)
        line_end_y = ball_y + (self.ball_radius - 1) * np.sin(self.ball.angle)
        color = (58, 64, 65)
        pygame.draw.line(
            self.screen, color, (ball_x, ball_y), (line_end_x, line_end_y), 3
        )  # 39 因为它总是会在 40 处粘住

        for piston in self.pistonList:
            self.screen.blit(
                self.piston_sprite,
                (
                    piston.position[0] - self.piston_radius,
                    piston.position[1] - self.piston_radius,
                ),
            )
        self.draw_pistons()

    def get_nearby_pistons(self):
        # 第一个活塞是最左边的
        nearby_pistons = []
        ball_pos = int(self.ball.position[0] - self.ball_radius)
        closest = abs(self.pistonList[0].position.x - ball_pos)
        closest_piston_index = 0
        for i in range(self.n_pistons):
            next_distance = abs(self.pistonList[i].position.x - ball_pos)
            if next_distance < closest:
                closest = next_distance
                closest_piston_index = i

        if closest_piston_index > 0:
            nearby_pistons.append(closest_piston_index - 1)
        nearby_pistons.append(closest_piston_index)
        if closest_piston_index < self.n_pistons - 1:
            nearby_pistons.append(closest_piston_index + 1)

        return nearby_pistons

    def get_local_reward(self, prev_position, curr_position):
        local_reward = 0.5 * (prev_position - curr_position)
        return local_reward

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        if self.render_mode == "human" and not self.renderOn:
            # sets self.renderOn to true and initializes display
            self.enable_render()

        self.draw_background()
        self.draw()

        observation = np.array(pygame.surfarray.pixels3d(self.screen))
        if self.render_mode == "human":
            pygame.display.flip()
        return (
            np.transpose(observation, axes=(1, 0, 2))
            if self.render_mode == "rgb_array"
            else None
        )

    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return

        action = np.asarray(action)
        agent = self.agent_selection
        if self.continuous:
            # 动作是一个 1 项 numpy 数组，move_piston 期望一个标量
            self.move_piston(self.pistonList[self.agent_name_mapping[agent]], action[0])
        else:
            self.move_piston(
                self.pistonList[self.agent_name_mapping[agent]], action - 1
            )

        self.space.step(self.dt)
        if self._agent_selector.is_last():
            ball_min_x = int(self.ball.position[0] - self.ball_radius)
            ball_next_x = (
                self.ball.position[0]
                - self.ball_radius
                + self.ball.velocity[0] * self.dt
            )
            if ball_next_x <= self.wall_width + 1:
                self.terminate = True
            # 确保球不能穿过墙
            ball_min_x = max(self.wall_width, ball_min_x)
            self.draw()
            local_reward = self.get_local_reward(self.lastX, ball_min_x)
            # 相反的顺序是因为从右向左移动
            global_reward = (100 / self.distance) * (self.lastX - ball_min_x)
            if not self.terminate:
                global_reward += self.time_penalty
            total_reward = [
                global_reward * (1 - self.local_ratio)
            ] * self.n_pistons  # 从全局奖励开始
            local_pistons_to_reward = self.get_nearby_pistons()
            for index in local_pistons_to_reward:
                total_reward[index] += local_reward * self.local_ratio
            self.rewards = dict(zip(self.agents, total_reward))
            self.lastX = ball_min_x
            self.frames += 1
        else:
            self._clear_rewards()

        self.truncate = self.frames >= self.max_cycles
        # 清除最近活塞的列表，以便下一个奖励周期
        if self.frames % self.recentFrameLimit == 0:
            self.recentPistons = set()
        if self._agent_selector.is_last():
            self.terminations = dict(
                zip(self.agents, [self.terminate for _ in self.agents])
            )
            self.truncations = dict(
                zip(self.agents, [self.truncate for _ in self.agents])
            )

        self.agent_selection = self._agent_selector.next()
        self._cumulative_rewards[agent] = 0
        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()


# 游戏美术由 J K Terry 创作
