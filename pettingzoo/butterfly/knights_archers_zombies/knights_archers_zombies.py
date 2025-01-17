# noqa: D212, D415
"""
# 骑士、弓箭手与僵尸（'KAZ'）

```{figure} butterfly_knights_archers_zombies.gif
:width: 200px
:name: knights_archers_zombies
```

此环境是<a href='..'>蝴蝶环境</a>的一部分。请先阅读该页面以获取一般信息。

| 导入                | `from pettingzoo.butterfly import knights_archers_zombies_v10` |
|---------------------|----------------------------------------------------------------|
| 动作                | 离散                                                           |
| 并行 API            | 是                                                            |
| 手动控制            | 是                                                            |
| 智能体              | `agents= ['archer_0', 'archer_1', 'knight_0', 'knight_1']`     |
| 智能体数量          | 4                                                              |
| 动作形状            | (1,)                                                           |
| 动作值范围          | [0, 5]                                                         |
| 观察形状            | (512, 512, 3)                                                  |
| 观察值范围          | (0, 255)                                                       |
| 状态形状            | (720, 1280, 3)                                                 |
| 状态值范围          | (0, 255)                                                       |


僵尸从屏幕上边缘以不可预测的路径向下边缘行走。玩家控制的智能体是骑士和弓箭手（默认 2 名骑士和 2 名弓箭手），初始位置在屏幕底部边缘。每个智能体都可以顺时针或逆时针旋转，并可以向前或向后移动。每个智能体也可以攻击来消灭僵尸。当骑士攻击时，会在其当前朝向方向前方挥舞锤子形成一个弧形。当弓箭手攻击时，会沿着弓箭手的朝向方向射出一支直线箭。当所有智能体死亡（与僵尸碰撞）或僵尸到达屏幕底部边缘时，游戏结束。当骑士的锤子击中并杀死僵尸时，获得 1 分。当弓箭手的箭击中并杀死僵尸时，获得 1 分。

这个环境有两种可能的观察类型：向量化和基于图像的。

#### 向量化（默认）
向环境传递参数 `vector_state=True`。

对于每个智能体，观察是一个 (N+1)x5 的数组，其中 `N = num_archers + num_knights + num_swords + max_arrows + max_zombies`。
> 注意 `num_swords = num_knights`

观察的行排序如下所示：
```
[
[当前智能体],
[弓箭手 1],
...,
[弓箭手 N],
[骑士 1],
...
[骑士 M],
[剑 1],
...
[剑 M],
[箭 1],
...
[箭 max_arrows],
[僵尸 1],
...
[僵尸 max_zombies]
]
```

总共有 N+1 行。没有实体的行将全为 0，但实体的排序不会改变。

**向量分解**

这里解释了观察中每一行的含义。所有距离都归一化到 [0, 1]。
注意对于位置，[0, 0] 是图像的左上角。向下是正 y，向左是正 x。

对于 `当前智能体` 的向量：
- 第一个值没有意义，始终为 0。
- 接下来的四个值是当前智能体的位置和角度。
  - 前两个值是位置值，分别归一化到图像的宽度和高度。
  - 最后两个值是智能体的朝向，表示为单位向量。

对于其他所有实体：
- 矩阵的每一行（这是一个宽度为 5 的向量）的分解如下：
  - 第一个值是实体与当前智能体之间的绝对距离。
  - 接下来的四个值是每个实体相对于当前智能体的相对位置和绝对角度。
    - 前两个值是相对于当前智能体的位置值。
    - 最后两个值是实体的角度，表示为相对于世界的方向单位向量。

**类型掩码**

有一个选项可以在每个行向量前添加类型掩码。这可以通过传递 `use_typemasks=True` 作为 kwarg 来启用。

类型掩码是一个宽度为 6 的向量，看起来像这样：
```
[0., 0., 0., 1., 0., 0.]
```

每个值分别对应：
```
[僵尸, 弓箭手, 骑士, 剑, 箭, 当前智能体]
```

如果那里没有实体，整个类型掩码（以及整个状态向量）将为 0。

因此，设置 `use_typemask=True` 会导致观察成为 (N+1)x11 的向量。

**序列空间**（实验性）

还有一个选项可以向环境传递 `sequence_space=True` 作为 kwarg。这只是从观察和状态向量中移除所有不存在的实体。注意这仍然是**实验性的**，因为状态和观察大小不再是常量。特别是，`N` 现在是一个可变数字。

#### 基于图像
向环境传递参数 `vector_state=False`。

每个智能体将环境观察为其周围的一个正方形区域，自己的身体在正方形的中心。观察表示为智能体周围的 512x512 像素图像，换句话说，是智能体周围 16x16 大小的空间。

### 手动控制

使用 'W'、'A'、'S' 和 'D' 键移动弓箭手。使用 'F' 键射箭。使用 'Q' 和 'E' 键旋转弓箭手。
按 'X' 键生成一个新的弓箭手。

使用 'I'、'J'、'K' 和 'L' 键移动骑士。使用 ';' 键挥剑。使用 'U' 和 'O' 键旋转骑士。
按 'M' 键生成一个新的骑士。


### 参数

``` python
knights_archers_zombies_v10.env(
  spawn_rate=20,
  num_archers=2,
  num_knights=2,
  max_zombies=10,
  max_arrows=10,
  killable_knights=True,
  killable_archers=True,
  pad_observation=True,
  line_death=False,
  max_cycles=900,
  vector_state=True,
  use_typemasks=False,
  sequence_space=False,
)
```

`spawn_rate`：在生成新僵尸之前的周期数。数字越小意味着僵尸生成的速率越高。

`num_archers`：初始生成的弓箭手智能体数量。

`num_knights`：初始生成的骑士智能体数量。

`max_zombies`：同时存在的最大僵尸数量。

`max_arrows`：同时存在的最大箭数量。

`killable_knights`：如果设置为 False，骑士智能体不会被僵尸杀死。

`killable_archers`：如果设置为 False，弓箭手智能体不会被僵尸杀死。

`pad_observation`：如果智能体靠近环境边缘，它们的观察无法形成 40x40 的网格。如果设置为 True，观察将用黑色填充。

`line_death`：如果设置为 False，智能体触碰上边界或下边界时不会死亡。如果为 True，智能体一触碰上边界或下边界就会死亡。

`vector_state`：是否使用向量化状态，如果设置为 `False`，将提供基于图像的观察。

`use_typemasks`：仅在设置 `vector_state=True` 时相关，向向量添加类型掩码。

`sequence_space`：**实验性**，仅在设置 `vector_state=True` 时相关，移除向量状态中不存在的实体。


### 版本历史

* v10：添加可向量化的状态空间 (1.17.0)
* v9：代码重写和多项修复 (1.16.0)
* v8：代码清理和多个错误修复 (1.14.0)
* v7：修复了与回合结束崩溃相关的小错误 (1.6.0)
* v6：修复了奖励结构 (1.5.2)
* v5：移除了黑色死亡参数 (1.5.0)
* v4：修复了观察和渲染问题 (1.4.2)
* v3：其他错误修复，升级了 PyGame 和 PyMunk 版本 (1.4.0)
* v2：修复了 `dones` 计算中的错误 (1.3.1)
* v1：修复了所有环境处理过早死亡的方式 (1.3.0)
* v0：初始版本发布 (1.0.0)

"""

import os
import sys
from itertools import repeat

import gymnasium
import numpy as np
import pygame
import pygame.gfxdraw
from gymnasium.spaces import Box, Discrete, Sequence
from gymnasium.utils import EzPickle, seeding

from pettingzoo import AECEnv
from pettingzoo.butterfly.knights_archers_zombies.manual_policy import ManualPolicy
from pettingzoo.butterfly.knights_archers_zombies.src import constants as const
from pettingzoo.butterfly.knights_archers_zombies.src.img import get_image
from pettingzoo.butterfly.knights_archers_zombies.src.players import Archer, Knight
from pettingzoo.butterfly.knights_archers_zombies.src.weapons import Arrow, Sword
from pettingzoo.butterfly.knights_archers_zombies.src.zombie import Zombie
from pettingzoo.utils import AgentSelector, wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn

sys.dont_write_bytecode = True


__all__ = ["ManualPolicy", "env", "parallel_env", "raw_env"]


def env(**kwargs):
    env = raw_env(**kwargs)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


parallel_env = parallel_wrapper_fn(env)


class raw_env(AECEnv, EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "knights_archers_zombies_v10",
        "is_parallelizable": True,
        "render_fps": const.FPS,
        "has_manual_policy": True,
    }

    def __init__(
        self,
        spawn_rate=20,
        num_archers=2,
        num_knights=2,
        max_zombies=10,
        max_arrows=10,
        killable_knights=True,
        killable_archers=True,
        pad_observation=True,
        line_death=False,
        max_cycles=900,
        vector_state=True,
        use_typemasks=False,
        sequence_space=False,
        render_mode=None,
    ):
        EzPickle.__init__(
            self,
            spawn_rate=spawn_rate,
            num_archers=num_archers,
            num_knights=num_knights,
            max_zombies=max_zombies,
            max_arrows=max_arrows,
            killable_knights=killable_knights,
            killable_archers=killable_archers,
            pad_observation=pad_observation,
            line_death=line_death,
            max_cycles=max_cycles,
            vector_state=vector_state,
            use_typemasks=use_typemasks,
            sequence_space=sequence_space,
            render_mode=render_mode,
        )
        # variable state space
        self.sequence_space = sequence_space
        if self.sequence_space:
            assert vector_state, "vector_state must be True if sequence_space is True."

            assert (
                use_typemasks
            ), "use_typemasks should be True if sequence_space is True"

        # whether we want RGB state or vector state
        self.vector_state = vector_state
        # agents + zombies + weapons
        self.num_tracked = (
            num_archers + num_knights + max_zombies + num_knights + max_arrows
        )
        self.use_typemasks = True if sequence_space else use_typemasks
        self.typemask_width = 6
        self.vector_width = 4 + self.typemask_width if use_typemasks else 4

        # Game Status
        self.frames = 0
        self.render_mode = render_mode
        self.screen = None

        # Game Constants
        self._seed()
        self.spawn_rate = spawn_rate
        self.max_cycles = max_cycles
        self.pad_observation = pad_observation
        self.killable_knights = killable_knights
        self.killable_archers = killable_archers
        self.line_death = line_death
        self.num_archers = num_archers
        self.num_knights = num_knights
        self.max_zombies = max_zombies
        self.max_arrows = max_arrows

        # Represents agents to remove at end of cycle
        self.kill_list = []
        self.agent_list = []
        self.agents = []
        self.dead_agents = []

        self.agent_name_mapping = {}
        a_count = 0
        for i in range(self.num_archers):
            a_name = "archer_" + str(i)
            self.agents.append(a_name)
            self.agent_name_mapping[a_name] = a_count
            a_count += 1
        for i in range(self.num_knights):
            k_name = "knight_" + str(i)
            self.agents.append(k_name)
            self.agent_name_mapping[k_name] = a_count
            a_count += 1

        shape = (
            [512, 512, 3]
            if not self.vector_state
            else [self.num_tracked + 1, self.vector_width + 1]
        )
        low = 0 if not self.vector_state else -1.0
        high = 255 if not self.vector_state else 1.0
        dtype = np.uint8 if not self.vector_state else np.float64
        if not self.sequence_space:
            obs_space = Box(low=low, high=high, shape=shape, dtype=dtype)
            self.observation_spaces = dict(
                zip(
                    self.agents,
                    [obs_space for _ in enumerate(self.agents)],
                )
            )
        else:
            box_space = Box(low=low, high=high, shape=[shape[-1]], dtype=dtype)
            obs_space = Sequence(space=box_space, stack=True)
            self.observation_spaces = dict(
                zip(
                    self.agents,
                    [obs_space for _ in enumerate(self.agents)],
                )
            )

        self.action_spaces = dict(
            zip(self.agents, [Discrete(6) for _ in enumerate(self.agents)])
        )

        shape = (
            [const.SCREEN_HEIGHT, const.SCREEN_WIDTH, 3]
            if not self.vector_state
            else [self.num_tracked, self.vector_width]
        )
        low = 0 if not self.vector_state else -1.0
        high = 255 if not self.vector_state else 1.0
        dtype = np.uint8 if not self.vector_state else np.float64
        self.state_space = Box(
            low=low,
            high=high,
            shape=shape,
            dtype=dtype,
        )
        self.possible_agents = self.agents

        if self.render_mode == "human":
            self.clock = pygame.time.Clock()

        self.left_wall = get_image(os.path.join("img", "left_wall.png"))
        self.right_wall = get_image(os.path.join("img", "right_wall.png"))
        self.right_wall_rect = self.right_wall.get_rect()
        self.right_wall_rect.left = const.SCREEN_WIDTH - self.right_wall_rect.width
        self.floor_patch1 = get_image(os.path.join("img", "patch1.png"))
        self.floor_patch2 = get_image(os.path.join("img", "patch2.png"))
        self.floor_patch3 = get_image(os.path.join("img", "patch3.png"))
        self.floor_patch4 = get_image(os.path.join("img", "patch4.png"))

        self._agent_selector = AgentSelector(self.agents)
        self.reinit()

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    # Spawn Zombies at Random Location at every 100 iterations
    def spawn_zombie(self):
        if len(self.zombie_list) < self.max_zombies:
            self.zombie_spawn_rate += 1
            zombie = Zombie(self.np_random)

            if self.zombie_spawn_rate >= self.spawn_rate:
                zombie.rect.x = self.np_random.integers(0, const.SCREEN_WIDTH)
                zombie.rect.y = 5

                self.zombie_list.add(zombie)
                self.zombie_spawn_rate = 0

    # actuate weapons
    def action_weapon(self, action, agent):
        if action == 5:
            if agent.is_knight:
                if agent.weapon_timeout > const.SWORD_TIMEOUT:
                    # make sure that the current knight doesn't have a sword already
                    if len(agent.weapons) == 0:
                        agent.weapons.add(Sword(agent))

            if agent.is_archer:
                if agent.weapon_timeout > const.ARROW_TIMEOUT:
                    # make sure that the screen has less arrows than allowable
                    if self.num_active_arrows < self.max_arrows:
                        agent.weapons.add(Arrow(agent))

    # move weapons
    def update_weapons(self):
        for agent in self.agent_list:
            for weapon in list(agent.weapons):
                weapon.update()

                if not weapon.is_active:
                    agent.weapons.remove(weapon)

    @property
    def num_active_arrows(self):
        num_arrows = 0
        for agent in self.agent_list:
            if agent.is_archer:
                num_arrows += len(agent.weapons)
        return num_arrows

    @property
    def num_active_swords(self):
        num_swords = 0
        for agent in self.agent_list:
            if agent.is_knight:
                num_swords += len(agent.weapons)
        return num_swords

    # Zombie Kills the Knight (also remove the sword)
    def zombit_hit_knight(self):
        for zombie in self.zombie_list:
            zombie_knight_list = pygame.sprite.spritecollide(
                zombie, self.knight_list, True
            )

            for knight in zombie_knight_list:
                knight.alive = False
                knight.weapons.empty()

                if knight.agent_name not in self.kill_list:
                    self.kill_list.append(knight.agent_name)

                self.knight_list.remove(knight)

    # Zombie Kills the Archer
    def zombie_hit_archer(self):
        for zombie in self.zombie_list:
            zombie_archer_list = pygame.sprite.spritecollide(
                zombie, self.archer_list, True
            )

            for archer in zombie_archer_list:
                archer.alive = False
                self.archer_list.remove(archer)
                if archer.agent_name not in self.kill_list:
                    self.kill_list.append(archer.agent_name)

    # Zombie Kills the Sword
    def sword_hit(self):
        for knight in self.knight_list:
            for sword in knight.weapons:
                zombie_sword_list = pygame.sprite.spritecollide(
                    sword, self.zombie_list, True
                )

                for zombie in zombie_sword_list:
                    self.zombie_list.remove(zombie)
                    sword.knight.score += 1

    # Zombie Kills the Arrow
    def arrow_hit(self):
        for agent in self.agent_list:
            if agent.is_archer:
                for arrow in list(agent.weapons):
                    zombie_arrow_list = pygame.sprite.spritecollide(
                        arrow, self.zombie_list, True
                    )

                    # For each zombie hit, remove the arrow, zombie and add to the score
                    for zombie in zombie_arrow_list:
                        agent.weapons.remove(arrow)
                        self.zombie_list.remove(zombie)
                        arrow.archer.score += 1

    # Zombie reaches the End of the Screen
    def zombie_endscreen(self, run, zombie_list):
        for zombie in zombie_list:
            if zombie.rect.y > const.SCREEN_HEIGHT - const.ZOMBIE_Y_SPEED:
                run = False
        return run

    # Zombie Kills all Players
    def zombie_all_players(self, run, knight_list, archer_list):
        if not knight_list and not archer_list:
            run = False
        return run

    def observe(self, agent):
        if not self.vector_state:
            screen = pygame.surfarray.pixels3d(self.screen)

            i = self.agent_name_mapping[agent]
            agent_obj = self.agent_list[i]
            agent_position = (agent_obj.rect.x, agent_obj.rect.y)

            if not agent_obj.alive:
                cropped = np.zeros((512, 512, 3), dtype=np.uint8)
            else:
                min_x = agent_position[0] - 256
                max_x = agent_position[0] + 256
                min_y = agent_position[1] - 256
                max_y = agent_position[1] + 256
                lower_y_bound = max(min_y, 0)
                upper_y_bound = min(max_y, const.SCREEN_HEIGHT)
                lower_x_bound = max(min_x, 0)
                upper_x_bound = min(max_x, const.SCREEN_WIDTH)
                startx = lower_x_bound - min_x
                starty = lower_y_bound - min_y
                endx = 512 + upper_x_bound - max_x
                endy = 512 + upper_y_bound - max_y
                cropped = np.zeros_like(self.observation_spaces[agent].low)
                cropped[startx:endx, starty:endy, :] = screen[
                    lower_x_bound:upper_x_bound, lower_y_bound:upper_y_bound, :
                ]

            return np.swapaxes(cropped, 1, 0)

        else:
            # get the agent
            agent = self.agent_list[self.agent_name_mapping[agent]]

            # get the agent position
            agent_state = agent.vector_state
            agent_pos = np.expand_dims(agent_state[0:2], axis=0)

            # get vector state of everything
            vector_state = self.get_vector_state()
            state = vector_state[:, -4:]
            is_dead = np.sum(np.abs(state), axis=1) == 0.0
            all_ids = vector_state[:, :-4]
            all_pos = state[:, 0:2]
            all_ang = state[:, 2:4]

            # get relative positions
            rel_pos = all_pos - agent_pos

            # get norm of relative distance
            norm_pos = np.linalg.norm(rel_pos, axis=1, keepdims=True) / np.sqrt(2)

            # kill dead things
            all_ids[is_dead] *= 0
            all_ang[is_dead] *= 0
            rel_pos[is_dead] *= 0
            norm_pos[is_dead] *= 0

            # combine the typemasks, positions and angles
            state = np.concatenate([all_ids, norm_pos, rel_pos, all_ang], axis=-1)

            # get the agent state as absolute vector
            # typemask is one longer to also include norm_pos
            if self.use_typemasks:
                typemask = np.zeros(self.typemask_width + 1)
                typemask[-2] = 1.0
            else:
                typemask = np.array([0.0])
            agent_state = agent.vector_state
            agent_state = np.concatenate([typemask, agent_state], axis=0)
            agent_state = np.expand_dims(agent_state, axis=0)

            # prepend agent state to the observation
            state = np.concatenate([agent_state, state], axis=0)
            if self.sequence_space:
                # remove pure zero rows if using sequence space
                state = state[~np.all(state == 0, axis=-1)]

            return state

    def state(self):
        """Returns an observation of the global environment."""
        if not self.vector_state:
            state = pygame.surfarray.pixels3d(self.screen).copy()
            state = np.rot90(state, k=3)
            state = np.fliplr(state)
        else:
            state = self.get_vector_state()

        return state

    def get_vector_state(self):
        state = []
        typemask = np.array([])

        # handle agents
        for agent_name in self.possible_agents:
            if agent_name not in self.dead_agents:
                agent = self.agent_list[self.agent_name_mapping[agent_name]]

                if self.use_typemasks:
                    typemask = np.zeros(self.typemask_width)
                    if agent.is_archer:
                        typemask[1] = 1.0
                    elif agent.is_knight:
                        typemask[2] = 1.0

                vector = np.concatenate((typemask, agent.vector_state), axis=0)
                state.append(vector)
            else:
                state.append(np.zeros(self.vector_width))

        # handle swords
        for agent in self.agent_list:
            if agent.is_knight:
                for sword in agent.weapons:
                    if self.use_typemasks:
                        typemask = np.zeros(self.typemask_width)
                        typemask[4] = 1.0

                    vector = np.concatenate((typemask, sword.vector_state), axis=0)
                    state.append(vector)

        # handle empty swords
        state.extend(
            repeat(
                np.zeros(self.vector_width),
                self.num_knights - self.num_active_swords,
            )
        )

        # handle arrows
        for agent in self.agent_list:
            if agent.is_archer:
                for arrow in agent.weapons:
                    if self.use_typemasks:
                        typemask = np.zeros(self.typemask_width)
                        typemask[3] = 1.0

                    vector = np.concatenate((typemask, arrow.vector_state), axis=0)
                    state.append(vector)

        # handle empty arrows
        state.extend(
            repeat(
                np.zeros(self.vector_width),
                self.max_arrows - self.num_active_arrows,
            )
        )

        # handle zombies
        for zombie in self.zombie_list:
            if self.use_typemasks:
                typemask = np.zeros(self.typemask_width)
                typemask[0] = 1.0

            vector = np.concatenate((typemask, zombie.vector_state), axis=0)
            state.append(vector)

        # handle empty zombies
        state.extend(
            repeat(
                np.zeros(self.vector_width),
                self.max_zombies - len(self.zombie_list),
            )
        )

        return np.stack(state, axis=0)

    def step(self, action):
        # check if the particular agent is done
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return

        # agent_list : list of agent instance indexed by number
        # agent_name_mapping: dict of {str, idx} for agent index and name
        # agent_selection : str representing the agent name
        # agent: agent instance
        agent = self.agent_list[self.agent_name_mapping[self.agent_selection]]

        # cumulative rewards from previous iterations should be cleared
        self._cumulative_rewards[self.agent_selection] = 0
        agent.score = 0

        # this is... so whacky... but all actions here are index with 1 so... ok
        action = action + 1
        out_of_bounds = agent.update(action)

        # check for out of bounds death
        if self.line_death and out_of_bounds:
            agent.alive = False
            if agent in self.archer_list:
                self.archer_list.remove(agent)
            else:
                agent.weapons.empty()
                self.knight_list.remove(agent)
            self.kill_list.append(agent.agent_name)

        # actuate the weapon if necessary
        self.action_weapon(action, agent)

        # Do these things once per cycle
        if self._agent_selector.is_last():
            # Update the weapons
            self.update_weapons()

            # Zombie Kills the Sword
            self.sword_hit()

            # Zombie Kills the Arrow
            self.arrow_hit()

            # Zombie Kills the Archer
            if self.killable_archers:
                self.zombie_hit_archer()

            # Zombie Kills the Knight
            if self.killable_knights:
                self.zombit_hit_knight()

            # update some zombies
            for zombie in self.zombie_list:
                zombie.update()

            # Spawning Zombies at Random Location at every 100 iterations
            self.spawn_zombie()

            if self.screen is not None:
                self.draw()

            self.check_game_end()
            self.frames += 1

        terminate = not self.run
        truncate = self.frames >= self.max_cycles
        self.terminations = {a: terminate for a in self.agents}
        self.truncations = {a: truncate for a in self.agents}

        # manage the kill list
        if self._agent_selector.is_last():
            # start iterating on only the living agents
            _live_agents = self.agents[:]
            for k in self.kill_list:
                # kill the agent
                _live_agents.remove(k)
                # set the termination for this agent for one round
                self.terminations[k] = True
                # add that we know this guy is dead
                self.dead_agents.append(k)

            # reset the kill list
            self.kill_list = []

            # reinit the agent selector with existing agents
            self._agent_selector.reinit(_live_agents)

        # if there still exist agents, get the next one
        if len(self._agent_selector.agent_order):
            self.agent_selection = self._agent_selector.next()

        self._clear_rewards()
        next_agent = self.agent_list[self.agent_name_mapping[self.agent_selection]]
        self.rewards[self.agent_selection] = next_agent.score

        self._accumulate_rewards()
        self._deads_step_first()

        if self.render_mode == "human":
            self.render()

    def draw(self):
        self.screen.fill((66, 40, 53))
        self.screen.blit(self.left_wall, self.left_wall.get_rect())
        self.screen.blit(self.right_wall, self.right_wall_rect)
        self.screen.blit(self.floor_patch1, (500, 500))
        self.screen.blit(self.floor_patch2, (900, 30))
        self.screen.blit(self.floor_patch3, (150, 430))
        self.screen.blit(self.floor_patch4, (300, 50))
        self.screen.blit(self.floor_patch1, (1000, 250))

        # draw all the sprites
        self.zombie_list.draw(self.screen)
        for agent in self.agent_list:
            agent.weapons.draw(self.screen)
        self.archer_list.draw(self.screen)
        self.knight_list.draw(self.screen)

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        if self.screen is None:
            pygame.init()

            if self.render_mode == "human":
                self.screen = pygame.display.set_mode(
                    [const.SCREEN_WIDTH, const.SCREEN_HEIGHT]
                )
                pygame.display.set_caption("Knights, Archers, Zombies")
            elif self.render_mode == "rgb_array":
                self.screen = pygame.Surface((const.SCREEN_WIDTH, const.SCREEN_HEIGHT))

        self.draw()

        observation = np.array(pygame.surfarray.pixels3d(self.screen))
        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        return (
            np.transpose(observation, axes=(1, 0, 2))
            if self.render_mode == "rgb_array"
            else None
        )

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    def check_game_end(self):
        # Zombie reaches the End of the Screen
        self.run = self.zombie_endscreen(self.run, self.zombie_list)

        # Zombie Kills all Players
        self.run = self.zombie_all_players(self.run, self.knight_list, self.archer_list)

    def reinit(self):
        # Dictionaries for holding new players and their weapons
        self.archer_dict = {}
        self.knight_dict = {}

        # Game Variables
        self.score = 0
        self.run = True
        self.zombie_spawn_rate = 0
        self.knight_player_num = self.archer_player_num = 0

        # Creating Sprite Groups
        self.zombie_list = pygame.sprite.Group()
        self.archer_list = pygame.sprite.Group()
        self.knight_list = pygame.sprite.Group()

        # agent_list is a list of instances
        # agents is s list of strings
        self.agent_list = []
        self.agents = []
        self.dead_agents = []

        for i in range(self.num_archers):
            name = "archer_" + str(i)
            self.archer_dict[f"archer{self.archer_player_num}"] = Archer(
                agent_name=name
            )
            self.archer_dict[f"archer{self.archer_player_num}"].offset(i * 50, 0)
            self.archer_list.add(self.archer_dict[f"archer{self.archer_player_num}"])
            self.agent_list.append(self.archer_dict[f"archer{self.archer_player_num}"])
            if i != self.num_archers - 1:
                self.archer_player_num += 1

        for i in range(self.num_knights):
            name = "knight_" + str(i)
            self.knight_dict[f"knight{self.knight_player_num}"] = Knight(
                agent_name=name
            )
            self.knight_dict[f"knight{self.knight_player_num}"].offset(i * 50, 0)
            self.knight_list.add(self.knight_dict[f"knight{self.knight_player_num}"])
            self.agent_list.append(self.knight_dict[f"knight{self.knight_player_num}"])
            if i != self.num_knights - 1:
                self.knight_player_num += 1

        self.agent_name_mapping = {}
        a_count = 0
        for i in range(self.num_archers):
            a_name = "archer_" + str(i)
            self.agents.append(a_name)
            self.agent_name_mapping[a_name] = a_count
            a_count += 1
        for i in range(self.num_knights):
            k_name = "knight_" + str(i)
            self.agents.append(k_name)
            self.agent_name_mapping[k_name] = a_count
            a_count += 1

        if self.render_mode is not None:
            self.render()
        else:
            self.screen = pygame.Surface((const.SCREEN_WIDTH, const.SCREEN_HEIGHT))
        self.frames = 0

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._seed(seed=seed)
        self.agents = self.possible_agents
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()
        self.rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self._cumulative_rewards = {a: 0 for a in self.agents}
        self.terminations = dict(zip(self.agents, [False for _ in self.agents]))
        self.truncations = dict(zip(self.agents, [False for _ in self.agents]))
        self.infos = dict(zip(self.agents, [{} for _ in self.agents]))
        self.reinit()


# The original code for this game, that was added by J K Terry, was
# created by Dipam Patel in a different repository (hence the git history)

# Game art purchased from https://finalbossblues.itch.io/time-fantasy-monsters
# and https://finalbossblues.itch.io/icons
