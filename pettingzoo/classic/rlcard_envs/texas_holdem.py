"""
德州扑克游戏环境。

这个模块实现了标准的德州扑克游戏，支持多个玩家参与。
游戏规则：每个玩家先获得两张手牌，然后通过下注和社区牌来组成最佳的五张牌组合。

这个环境是经典环境的一部分。请先阅读该页面的通用信息。

| 导入             | `from pettingzoo.classic import texas_holdem_v4` |
|--------------------|--------------------------------------------------|
| 动作类型          | 离散                                             |
| 并行API          | 是                                               |
| 手动控制         | 否                                               |
| 智能体           | `agents= ['player_0', 'player_1']`               |
| 智能体数量       | 2                                                |
| 动作形状         | Discrete(4)                                      |
| 动作值           | Discrete(4)                                      |
| 观察形状         | (72,)                                            |
| 观察值           | [0, 1]                                           |


## 参数

``` python
texas_holdem_v4.env(num_players=2)
```

`num_players`: 设置游戏中的玩家数量。最少为2名玩家。

### 观察空间

观察是一个字典，包含一个 `'observation'` 元素（即下面描述的常规RL观察）和一个 `'action_mask'`（包含合法动作，在合法动作掩码部分描述）。

主要观察空间是一个包含72个布尔整数的向量。前52个条目描述了当前玩家的手牌和任何社区牌，如下所示：

|  索引   | 描述                                                      |
|:-------:|----------------------------------------------------------|
|  0 - 12 | 黑桃<br>_`0`: A, `1`: 2, ..., `12`: K_                   |
| 13 - 25 | 红心<br>_`13`: A, `14`: 2, ..., `25`: K_                 |
| 26 - 38 | 方块<br>_`26`: A, `27`: 2, ..., `38`: K_                 |
| 39 - 51 | 梅花<br>_`39`: A, `40`: 2, ..., `51`: K_                 |
| 52 - 56 | 第1轮下注筹码<br>_`52`: 0, `53`: 1, ..., `56`: 4_       |
| 57 - 61 | 第2轮下注筹码<br>_`57`: 0, `58`: 1, ..., `61`: 4_       |
| 62 - 66 | 第3轮下注筹码<br>_`62`: 0, `63`: 1, ..., `66`: 4_       |
| 67 - 71 | 第4轮下注筹码<br>_`67`: 0, `68`: 1, ..., `71`: 4_       |

#### 合法动作掩码

当前智能体可用的合法动作可以在字典观察的 `action_mask` 元素中找到。`action_mask` 是一个二进制向量，其中每个索引表示该动作是否合法。除了当前轮到的智能体外，其他所有智能体的 `action_mask` 都是零。采取非法动作会导致游戏结束，非法行动的智能体获得-1的奖励，其他所有智能体获得0的奖励。

### 动作空间

| 动作ID  | 动作   |
|:-------:|--------|
|    0    | 跟注   |
|    1    | 加注   |
|    2    | 弃牌   |
|    3    | 过牌   |

### 奖励

| 赢家           | 输家           |
| :------------: | :------------: |
| +下注筹码/2    | -下注筹码/2    |

### 版本历史

* v4: 升级到 RLCard 1.0.3 (1.11.0)
* v3: 修复了任意调用 observe() 的错误 (1.8.0)
* v2: 升级 RLCard 版本，修复错误，观察中的合法动作掩码取代了信息中的非法动作列表 (1.5.0)
* v1: 升级 RLCard 版本，修复观察空间，采用新的智能体迭代方案 (1.4.0)
* v0: 初始版本发布 (1.0.0)

"""

from __future__ import annotations

import os

import gymnasium
import numpy as np
import pygame
from gymnasium.utils import EzPickle

from pettingzoo.classic.rlcard_envs.rlcard_base import RLCardBase
from pettingzoo.utils import wrappers

# 像素艺术来自 Mariia Khmelnytska (https://www.123rf.com/photo_104453049_stock-vector-pixel-art-playing-cards-standart-deck-vector-set.html)

def get_image(path):
    """获取指定路径的图像。

    参数:
        path (str): 图像文件路径

    返回:
        pygame.Surface: 加载的图像
    """
    from os import path as os_path

    cwd = os_path.dirname(__file__)
    image = pygame.image.load(cwd + "/" + path)
    return image


def get_font(path, size):
    """获取指定路径和大小的字体。

    参数:
        path (str): 字体文件路径
        size (int): 字体大小

    返回:
        pygame.font.Font: 加载的字体
    """
    from os import path as os_path

    cwd = os_path.dirname(__file__)
    font = pygame.font.Font((cwd + "/" + path), size)
    return font


def env(**kwargs):
    """创建德州扑克环境的包装器。

    返回:
        AECEnv: 包装后的环境
    """
    env = raw_env(**kwargs)
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class raw_env(RLCardBase, EzPickle):
    """德州扑克游戏的主要环境类。

    这个环境实现了标准的德州扑克游戏，支持多个玩家参与。

    属性:
        metadata (dict): 环境的元数据，包括版本信息和渲染模式
        render_mode (str): 渲染模式
        screen_height (int): 屏幕高度
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "texas_holdem_v4",
        "is_parallelizable": False,
        "render_fps": 1,
    }

    def __init__(
        self,
        num_players: int = 2,
        render_mode: str | None = None,
        screen_height: int | None = 1000,
    ):
        """初始化德州扑克环境。

        参数:
            num_players (int): 玩家数量，默认为2
            render_mode (str): 渲染模式，可以是 "human" 或 "rgb_array"
            screen_height (int): 屏幕高度，默认为1000
        """
        EzPickle.__init__(self, num_players, render_mode, screen_height)
        super().__init__("limit-holdem", num_players, (72,))
        self.render_mode = render_mode
        self.screen_height = screen_height

        if self.render_mode == "human":
            self.clock = pygame.time.Clock()

    def step(self, action):
        """执行一步游戏。

        参数:
            action (int): 玩家的动作，可以是跟注(0)、加注(1)、弃牌(2)或过牌(3)

        返回:
            observations (dict): 每个玩家的观察
            rewards (dict): 每个玩家的奖励
            terminations (dict): 每个玩家的终止状态
            truncations (dict): 每个玩家的截断状态
            infos (dict): 每个玩家的额外信息
        """
        super().step(action)

        if self.render_mode == "human":
            self.render()

    def render(self):
        """渲染当前游戏状态。

        根据render_mode的不同，可以：
        - "human": 在窗口中显示游戏界面
        - "rgb_array": 返回RGB数组形式的游戏界面
        """
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        def calculate_width(self, screen_width, i):
            return int(
                (
                    screen_width
                    / (np.ceil(len(self.possible_agents) / 2) + 1)
                    * np.ceil((i + 1) / 2)
                )
                + (tile_size * 33 / 616)
            )

        def calculate_offset(hand, j, tile_size):
            return int(
                (len(hand) * (tile_size * 23 / 56)) - ((j) * (tile_size * 23 / 28))
            )

        def calculate_height(screen_height, divisor, multiplier, tile_size, offset):
            return int(multiplier * screen_height / divisor + tile_size * offset)

        screen_height = self.screen_height
        screen_width = int(
            screen_height * (1 / 20)
            + np.ceil(len(self.possible_agents) / 2) * (screen_height * 12 / 20)
        )

        if self.screen is None:
            pygame.init()

            if self.render_mode == "human":
                self.screen = pygame.display.set_mode((screen_width, screen_height))
                pygame.display.set_caption("Texas Hold'em")
            else:
                self.screen = pygame.Surface((screen_width, screen_height))

        # Setup dimensions for card size and setup for colors
        tile_size = screen_height * 2 / 10

        bg_color = (7, 99, 36)
        white = (255, 255, 255)
        self.screen.fill(bg_color)

        chips = {
            0: {"value": 10000, "img": "ChipOrange.png", "number": 0},
            1: {"value": 5000, "img": "ChipPink.png", "number": 0},
            2: {"value": 1000, "img": "ChipYellow.png", "number": 0},
            3: {"value": 100, "img": "ChipBlack.png", "number": 0},
            4: {"value": 50, "img": "ChipBlue.png", "number": 0},
            5: {"value": 25, "img": "ChipGreen.png", "number": 0},
            6: {"value": 10, "img": "ChipLightBlue.png", "number": 0},
            7: {"value": 5, "img": "ChipRed.png", "number": 0},
            8: {"value": 1, "img": "ChipWhite.png", "number": 0},
        }

        # Load and blit all images for each card in each player's hand
        for i, player in enumerate(self.possible_agents):
            state = self.env.game.get_state(self._name_to_int(player))
            for j, card in enumerate(state["hand"]):
                # Load specified card
                card_img = get_image(os.path.join("img", card + ".png"))
                card_img = pygame.transform.scale(
                    card_img, (int(tile_size * (142 / 197)), int(tile_size))
                )
                # Players with even id go above public cards
                if i % 2 == 0:
                    self.screen.blit(
                        card_img,
                        (
                            (
                                calculate_width(self, screen_width, i)
                                - calculate_offset(state["hand"], j, tile_size)
                                - tile_size
                                * (8 / 10)
                                * (1 - np.ceil(i / 2))
                                * (0 if len(self.possible_agents) == 2 else 1)
                            ),
                            calculate_height(screen_height, 4, 1, tile_size, -1),
                        ),
                    )
                # Players with odd id go below public cards
                else:
                    self.screen.blit(
                        card_img,
                        (
                            (
                                calculate_width(self, screen_width, i)
                                - calculate_offset(state["hand"], j, tile_size)
                                - tile_size
                                * (8 / 10)
                                * (1 - np.ceil((i - 1) / 2))
                                * (0 if len(self.possible_agents) == 2 else 1)
                            ),
                            calculate_height(screen_height, 4, 3, tile_size, 0),
                        ),
                    )

            # Load and blit text for player name
            font = get_font(os.path.join("font", "Minecraft.ttf"), 36)
            text = font.render("Player " + str(i + 1), True, white)
            textRect = text.get_rect()
            if i % 2 == 0:
                textRect.center = (
                    (
                        screen_width
                        / (np.ceil(len(self.possible_agents) / 2) + 1)
                        * np.ceil((i + 1) / 2)
                        - tile_size
                        * (8 / 10)
                        * (1 - np.ceil(i / 2))
                        * (0 if len(self.possible_agents) == 2 else 1)
                    ),
                    calculate_height(screen_height, 4, 1, tile_size, -(22 / 20)),
                )
            else:
                textRect.center = (
                    (
                        screen_width
                        / (np.ceil(len(self.possible_agents) / 2) + 1)
                        * np.ceil((i + 1) / 2)
                        - tile_size
                        * (8 / 10)
                        * (1 - np.ceil((i - 1) / 2))
                        * (0 if len(self.possible_agents) == 2 else 1)
                    ),
                    calculate_height(screen_height, 4, 3, tile_size, (23 / 20)),
                )
            self.screen.blit(text, textRect)

            # Load and blit number of poker chips for each player
            font = get_font(os.path.join("font", "Minecraft.ttf"), 24)
            text = font.render(str(state["my_chips"]), True, white)
            textRect = text.get_rect()

            # Calculate number of each chip
            total = state["my_chips"]
            height = 0
            for key in chips:
                num = total / chips[key]["value"]
                chips[key]["number"] = int(num)
                total %= chips[key]["value"]

                chip_img = get_image(os.path.join("img", chips[key]["img"]))
                chip_img = pygame.transform.scale(
                    chip_img, (int(tile_size / 2), int(tile_size * 16 / 45))
                )

                # Blit poker chip img
                for j in range(0, int(chips[key]["number"])):
                    if i % 2 == 0:
                        self.screen.blit(
                            chip_img,
                            (
                                (
                                    calculate_width(self, screen_width, i)
                                    + tile_size
                                    * (8 / 10)
                                    * (
                                        1
                                        if len(self.possible_agents) == 2
                                        else np.ceil(i / 2)
                                    )
                                ),
                                calculate_height(screen_height, 4, 1, tile_size, -1 / 2)
                                - ((j + height) * tile_size / 15),
                            ),
                        )
                    else:
                        self.screen.blit(
                            chip_img,
                            (
                                (
                                    calculate_width(self, screen_width, i)
                                    + tile_size
                                    * (8 / 10)
                                    * (
                                        1
                                        if len(self.possible_agents) == 2
                                        else np.ceil((i - 1) / 2)
                                    )
                                ),
                                calculate_height(screen_height, 4, 3, tile_size, 1 / 2)
                                - ((j + height) * tile_size / 15),
                            ),
                        )
                height += chips[key]["number"]

            # Blit text number
            if i % 2 == 0:
                textRect.center = (
                    (
                        calculate_width(self, screen_width, i)
                        + (tile_size * (5 / 20))
                        + tile_size
                        * (8 / 10)
                        * (1 if len(self.possible_agents) == 2 else np.ceil(i / 2))
                    ),
                    calculate_height(screen_height, 4, 1, tile_size, -1 / 2)
                    - ((height + 1) * tile_size / 15),
                )
            else:
                textRect.center = (
                    (
                        calculate_width(self, screen_width, i)
                        + (tile_size * (5 / 20))
                        + tile_size
                        * (8 / 10)
                        * (
                            1
                            if len(self.possible_agents) == 2
                            else np.ceil((i - 1) / 2)
                        )
                    ),
                    calculate_height(screen_height, 4, 3, tile_size, 1 / 2)
                    - ((height + 1) * tile_size / 15),
                )
            self.screen.blit(text, textRect)

        # Load and blit public cards
        for i, card in enumerate(state["public_cards"]):
            card_img = get_image(os.path.join("img", card + ".png"))
            card_img = pygame.transform.scale(
                card_img, (int(tile_size * (142 / 197)), int(tile_size))
            )
            if len(state["public_cards"]) <= 3:
                self.screen.blit(
                    card_img,
                    (
                        (
                            (
                                ((screen_width / 2) + (tile_size * 31 / 616))
                                - calculate_offset(state["public_cards"], i, tile_size)
                            ),
                            calculate_height(screen_height, 2, 1, tile_size, -(1 / 2)),
                        )
                    ),
                )
            else:
                if i <= 2:
                    self.screen.blit(
                        card_img,
                        (
                            (
                                ((screen_width / 2) + (tile_size * 31 / 616))
                                - calculate_offset(
                                    state["public_cards"][:3], i, tile_size
                                )
                            ),
                            calculate_height(
                                screen_height, 2, 1, tile_size, -21 / 20
                            ),
                        ),
                    )
                else:
                    self.screen.blit(
                        card_img,
                        (
                            (
                                ((screen_width / 2) + (tile_size * 31 / 616))
                                - calculate_offset(
                                    state["public_cards"][3:], i - 3, tile_size
                                )
                            ),
                            calculate_height(
                                screen_height, 2, 1, tile_size, 1 / 20
                            ),
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
