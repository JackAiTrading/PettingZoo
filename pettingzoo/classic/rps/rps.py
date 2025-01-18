# noqa: D212, D415
"""
# 石头剪刀布游戏环境

这个模块实现了标准的石头剪刀布游戏，支持两个玩家同时出拳。
游戏规则：石头胜剪刀，剪刀胜布，布胜石头。

```{figure} classic_rps.gif
:width: 140px
:name: rps
```

这个环境是经典环境的一部分。请先阅读<a href='..'>经典环境</a>页面获取更多信息。

| 导入             | `from pettingzoo.classic import rps_v2` |
|--------------------|-----------------------------------------|
| 动作类型            | 离散                                |
| 并行API           | 是                                     |
| 手动控制          | 否                                      |
| 智能体             | `agents= ['player_0', 'player_1']`      |
| 智能体数量          | 2                                       |
| 动作形状          | 离散(3)                             |
| 动作值            | 离散(3)                             |
| 观察形状          | 离散(4)                             |
| 观察值            | 离散(4)                             |


石头剪刀布是两个玩家同时出拳的游戏。如果两个玩家出相同的拳，则为平局。如果两个玩家出不同的拳，则胜负由以下规则决定：石头胜剪刀，剪刀胜布，布胜石头。

这个游戏可以通过添加新的动作对来扩展。添加新的动作对可以使游戏更加平衡。这样一来，游戏的动作数量将为奇数，每个动作将胜过其他动作的一半，同时被其他动作的一半击败。这个游戏最常见的扩展是[石头剪刀布蜥蜴史波克](http://www.samkass.com/theories/RPSSL.html)，在这个版本中，只添加了一个新的动作对。

### 参数

```python
rps_v2.env(num_actions=3, max_cycles=15)
```

`num_actions`: 游戏中的动作数量。默认值为3，表示标准的石头剪刀布游戏。这个参数必须是大于3的奇数。如果这个值为5，则游戏将扩展为石头剪刀布蜥蜴史波克。

`max_cycles`: 游戏的最大回合数。超过这个回合数后，所有玩家将被视为完成游戏。

### 观察空间

#### 石头剪刀布

如果动作数量为3，则游戏为标准的石头剪刀布游戏。观察空间为一个标量值，可能的值为4个。由于两个玩家同时出拳，因此观察空间在两个玩家都出拳之前为None。因此，3表示没有出拳。石头、剪刀和布分别用0、1和2表示。

| 值  | 观察 |
| :----: | :---------:  |
| 0      | 石头         |
| 1      | 剪刀        |
| 2      | 布          |
| 3      | None         |

#### 扩展游戏

如果动作数量大于3，则观察空间为一个标量值，可能的值为1+n个，其中n是动作数量。观察空间在两个玩家都出拳之前为None，最大可能值为1+n，表示没有出拳。额外的动作按照从0开始的顺序编码。如果动作数量为5，则游戏扩展为石头剪刀布蜥蜴史波克。下表显示了一个可能的观察空间。

| 值  | 观察 |
| :----: | :---------:  |
| 0      | 石头         |
| 1      | 剪刀        |
| 2      | 布          |
| 3      | 蜥蜴        |
| 4      | 史波克      |
| 5      | 动作6       |
| 6      | 动作7       |
| 7      | None         |

### 动作空间

#### 石头剪刀布

动作空间为一个标量值，可能的值为3个。石头、剪刀和布分别用0、1和2表示。

| 值  | 动作 |
| :----: | :---------:  |
| 0      | 石头         |
| 1      | 剪刀        |
| 2      | 布          |

#### 扩展游戏

动作空间为一个标量值，可能的值为n个，其中n是动作数量。动作按照从0开始的顺序编码。下表显示了一个可能的动作空间。

| 值  | 动作 |
| :----: | :---------:  |
| 0      | 石头         |
| 1      | 剪刀        |
| 2      | 布          |
| 3      | 蜥蜴        |
| 4      | 史波克      |
| 5      | 动作6       |
| 6      | 动作7       |

### 奖励

| 胜者 | 负者 |
| :----: | :---: |
| +1     | -1    |

如果游戏结束时为平局，则两个玩家都将获得0奖励。

### 版本历史

* v2: 合并RPS和石头剪刀布蜥蜴史波克环境，添加num_actions和max_cycles参数 (1.9.0)
* v1: 更新所有环境的版本号，因为采用了新的智能体迭代方案 (1.4.0)
* v0: 初始版本发布 (1.0.0)

"""
from __future__ import annotations

import os

import gymnasium
import numpy as np
import pygame
from gymnasium.spaces import Discrete
from gymnasium.utils import EzPickle

from pettingzoo import AECEnv
from pettingzoo.utils import AgentSelector, wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn


def get_image(path):
    from os import path as os_path

    import pygame

    cwd = os_path.dirname(__file__)
    image = pygame.image.load(cwd + "/" + path)
    sfc = pygame.Surface(image.get_size(), flags=pygame.SRCALPHA)
    sfc.blit(image, (0, 0))
    return sfc


def get_font(path, size):
    from os import path as os_path

    cwd = os_path.dirname(__file__)
    font = pygame.font.Font((cwd + "/" + path), size)
    return font


def env(**kwargs):
    env = raw_env(**kwargs)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


parallel_env = parallel_wrapper_fn(env)


class raw_env(AECEnv, EzPickle):
    """石头剪刀布游戏环境。

    这个环境实现了标准的石头剪刀布游戏，支持两个玩家同时出拳。

    属性:
        metadata (dict): 环境的元数据，包括版本信息
        possible_agents (list): 可能的智能体列表，包括玩家1和玩家2
        action_spaces (dict): 每个玩家的动作空间
        observation_spaces (dict): 每个玩家的观察空间
        num_moves (int): 游戏总回合数
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "rps_v2",
        "is_parallelizable": True,
        "render_fps": 2,
    }

    def __init__(
        self,
        num_actions: int | None = 3,
        max_cycles: int | None = 15,
        render_mode: str | None = None,
        screen_height: int | None = 800,
    ):
        EzPickle.__init__(self, num_actions, max_cycles, render_mode, screen_height)
        super().__init__()
        self.max_cycles = max_cycles

        # 动作数量必须是奇数且大于3
        assert num_actions > 2, "动作数量必须大于或等于3。"
        assert num_actions % 2 != 0, "动作数量必须是奇数。"
        self._moves = ["ROCK", "PAPER", "SCISSORS"]
        if num_actions > 3:
            # 扩展到蜥蜴和史波克
            self._moves.extend(("SPOCK", "LIZARD"))
            for action in range(num_actions - 5):
                self._moves.append("ACTION_" f"{action + 6}")
        # None是最后一个可能的动作，以满足离散动作空间
        self._moves.append("None")
        self._none = num_actions

        self.agents = ["player_" + str(r) for r in range(2)]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.num_agents))))
        self.action_spaces = {agent: Discrete(num_actions) for agent in self.agents}
        self.observation_spaces = {
            agent: Discrete(1 + num_actions) for agent in self.agents
        }

        self.render_mode = render_mode
        self.screen_height = screen_height
        self.screen = None

        if self.render_mode == "human":
            self.clock = pygame.time.Clock()

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "您正在调用render方法，但没有指定任何渲染模式。"
            )
            return

        def offset(i, size, offset=0):
            if i == 0:
                return -(size) - offset
            else:
                return offset

        screen_height = self.screen_height
        screen_width = int(screen_height * 5 / 14)

        # 加载所有必要的图像
        paper = get_image(os.path.join("img", "Paper.png"))
        rock = get_image(os.path.join("img", "Rock.png"))
        scissors = get_image(os.path.join("img", "Scissors.png"))
        spock = get_image(os.path.join("img", "Spock.png"))
        lizard = get_image(os.path.join("img", "Lizard.png"))

        # 缩放历史图像
        paper = pygame.transform.scale(
            paper, (int(screen_height / 9), int(screen_height / 9 * (14 / 12)))
        )
        rock = pygame.transform.scale(
            rock, (int(screen_height / 9), int(screen_height / 9 * (10 / 13)))
        )
        scissors = pygame.transform.scale(
            scissors, (int(screen_height / 9), int(screen_height / 9 * (14 / 13)))
        )
        spock = pygame.transform.scale(
            spock, (int(screen_height / 9), int(screen_height / 9))
        )
        lizard = pygame.transform.scale(
            lizard, (int(screen_height / 9 * (9 / 18)), int(screen_height / 9))
        )

        # 设置背景颜色
        bg = (255, 255, 255)
        self.screen.fill(bg)

        # 设置字体属性
        black = (0, 0, 0)
        font = get_font(
            (os.path.join("font", "Minecraft.ttf")), int(screen_height / 25)
        )

        for i, move in enumerate(self.history[0:10]):
            # 绘制历史记录
            if self._moves[move] == "ROCK":
                self.screen.blit(
                    rock,
                    (
                        (screen_width / 2)
                        + offset((i) % 2, screen_height / 9, screen_height * 7 / 126),
                        (screen_height * 7 / 24)
                        + ((screen_height / 7) * np.floor(i / 2)),
                    ),
                )
            elif self._moves[move] == "PAPER":
                self.screen.blit(
                    paper,
                    (
                        (screen_width / 2)
                        + offset((i) % 2, screen_height / 9, screen_height * 7 / 126),
                        (screen_height * 7 / 24)
                        + ((screen_height / 7) * np.floor(i / 2)),
                    ),
                )
            elif self._moves[move] == "SCISSORS":
                self.screen.blit(
                    scissors,
                    (
                        (screen_width / 2)
                        + offset((i) % 2, screen_height / 9, screen_height * 7 / 126),
                        (screen_height * 7 / 24)
                        + ((screen_height / 7) * np.floor(i / 2)),
                    ),
                )
            elif self._moves[move] == "SPOCK":
                self.screen.blit(
                    spock,
                    (
                        (screen_width / 2)
                        + offset(
                            (i + 1) % 2, screen_height / 9, screen_height * 7 / 126
                        ),
                        (screen_height * 7 / 24)
                        + ((screen_height / 7) * np.floor(i / 2)),
                    ),
                )
            elif self._moves[move] == "LIZARD":
                self.screen.blit(
                    lizard,
                    (
                        (screen_width / 2)
                        + offset(
                            (i + 1) % 2, screen_height / 9, screen_height * 7 / 126
                        ),
                        (screen_height * 7 / 24)
                        + ((screen_height / 7) * np.floor(i / 2)),
                    ),
                )

        # 缩放当前游戏图像
        paper = pygame.transform.scale(
            paper, (int(screen_height / 7), int(screen_height / 7 * (14 / 12)))
        )
        rock = pygame.transform.scale(
            rock, (int(screen_height / 7), int(screen_height / 7 * (10 / 13)))
        )
        scissors = pygame.transform.scale(
            scissors, (int(screen_height / 7), int(screen_height / 7 * (14 / 13)))
        )
        spock = pygame.transform.scale(
            spock, (int(screen_height / 7), int(screen_height / 7))
        )
        lizard = pygame.transform.scale(
            lizard, (int(screen_height / 7 * (9 / 18)), int(screen_height / 7))
        )

        if len(self.agents) > 1:
            for i in range(0, 2):
                # 绘制每个玩家的文本
                text = font.render("玩家 " + str(i + 1), True, black)
                textRect = text.get_rect()
                textRect.center = (
                    (screen_width / 2) + offset(i, 0, screen_width * 11 / 40),
                    screen_height / 40,
                )
                self.screen.blit(text, textRect)

                # 绘制每个玩家的动作
                if self._moves[self.state[self.agents[i]]] == "ROCK":
                    self.screen.blit(
                        rock,
                        (
                            (screen_width / 2)
                            + offset(i, screen_height / 7, screen_height / 42),
                            screen_height / 12,
                        ),
                    )
                elif self._moves[self.state[self.agents[i]]] == "PAPER":
                    self.screen.blit(
                        paper,
                        (
                            (screen_width / 2)
                            + offset(i, screen_height / 7, screen_height / 42),
                            screen_height / 12,
                        ),
                    )
                elif self._moves[self.state[self.agents[i]]] == "SCISSORS":
                    self.screen.blit(
                        scissors,
                        (
                            (screen_width / 2)
                            + offset(i, screen_height / 7, screen_height / 42),
                            screen_height / 12,
                        ),
                    )
                elif self._moves[self.state[self.agents[i]]] == "SPOCK":
                    self.screen.blit(
                        spock,
                        (
                            (screen_width / 2)
                            + offset(i, screen_height / 7, screen_height / 42),
                            screen_height / 12,
                        ),
                    )
                elif self._moves[self.state[self.agents[i]]] == "LIZARD":
                    self.screen.blit(
                        lizard,
                        (
                            (screen_width / 2)
                            + offset(i, screen_height / 7, screen_height / 42),
                            screen_height / 12,
                        ),
                    )

        if self.render_mode == "human":
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])

        observation = np.array(pygame.surfarray.pixels3d(self.screen))

        return (
            np.transpose(observation, axes=(1, 0, 2))
            if self.render_mode == "rgb_array"
            else None
        )

    def observe(self, agent):
        # 每个玩家的观察是另一个玩家的上一轮动作
        return np.array(self.observations[agent])

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.next()
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self.state = {agent: self._none for agent in self.agents}
        self.observations = {agent: self._none for agent in self.agents}

        self.history = [-1] * (2 * 5)

        self.num_moves = 0

        screen_height = self.screen_height
        screen_width = int(screen_height * 5 / 14)

        if self.screen is None:
            pygame.init()

        if self.render_mode == "human":
            self.screen = pygame.display.set_mode((screen_width, screen_height))
            pygame.display.set_caption("石头剪刀布")
        else:
            self.screen = pygame.Surface((screen_width, screen_height))

    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return

        agent = self.agent_selection

        self.state[self.agent_selection] = action

        # 收集奖励，如果这是最后一个玩家动作
        if self._agent_selector.is_last():
            # 相同动作 => 每个玩家奖励为0
            if self.state[self.agents[0]] == self.state[self.agents[1]]:
                rewards = (0, 0)
            else:
                # 相同动作奇偶性 => 动作值较低的玩家获胜
                if (self.state[self.agents[0]] + self.state[self.agents[1]]) % 2 == 0:
                    if self.state[self.agents[0]] > self.state[self.agents[1]]:
                        rewards = (-1, 1)
                    else:
                        rewards = (1, -1)
                # 不同动作奇偶性 => 动作值较高的玩家获胜
                else:
                    if self.state[self.agents[0]] > self.state[self.agents[1]]:
                        rewards = (1, -1)
                    else:
                        rewards = (-1, 1)
            self.rewards[self.agents[0]], self.rewards[self.agents[1]] = rewards

            self.num_moves += 1

            self.truncations = {
                agent: self.num_moves >= self.max_cycles for agent in self.agents
            }
            for i in self.agents:
                self.observations[i] = self.state[
                    self.agents[1 - self.agent_name_mapping[i]]
                ]

            if self.render_mode == "human":
                self.render()

            # 记录历史
            self.history[2:] = self.history[:-2]
            self.history[0] = self.state[self.agents[0]]
            self.history[1] = self.state[self.agents[1]]

        else:
            self.state[self.agents[1 - self.agent_name_mapping[agent]]] = self._none

            self._clear_rewards()

            if self.render_mode == "human":
                self.render()

        self._cumulative_rewards[self.agent_selection] = 0
        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()
