---
title: Atari 环境
firstpage:
---

# Atari 环境

```{toctree}
:hidden:
atari/basketball_pong
atari/boxing
atari/combat_plane
atari/combat_tank
atari/double_dunk
atari/entombed_competitive
atari/entombed_cooperative
atari/flag_capture
atari/foozpong
atari/ice_hockey
atari/joust
atari/mario_bros
atari/maze_craze
atari/othello
atari/pong
atari/quadrapong
atari/space_invaders
atari/space_war
atari/surround
atari/tennis
atari/video_checkers
atari/volleyball_pong
atari/warlords
atari/wizard_of_wor
```

Atari 环境基于[街机学习环境（ALE）](https://github.com/mgbellemare/Arcade-Learning-Environment)。这个环境在现代强化学习的发展中起到了重要作用，因此我们希望我们的[多智能体版本](https://github.com/Farama-Foundation/Multi-Agent-ALE)能在多智能体强化学习的发展中发挥作用。

```{raw} html
    :file: atari/list.html
```

### 安装

这组环境的特定依赖项可以通过以下命令安装：

````bash
pip install 'pettingzoo[atari]'
````

使用 [AutoROM](https://github.com/Farama-Foundation/AutoROM) 安装 ROM，或使用 `rom_path` 参数指定 Atari ROM 的路径（参见[通用参数](#common-parameters)）。

### 使用方法

要启动一个带有随机智能体的[太空入侵者](/environments/atari/space_invaders/)环境：
```python
from pettingzoo.atari import space_invaders_v2

env = space_invaders_v2.env(render_mode="human")
env.reset(seed=42)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        action = env.action_space(agent).sample() # 这里是您插入策略的地方

    env.step(action)
env.close()
```

### 游戏概述

大多数游戏有两个玩家，除了 Warlords 和一些 Pong 变体有四个玩家。

### 环境细节

ALE 已经被广泛研究，发现了一些值得注意的问题：

* **确定性**：Atari 主机是确定性的，因此理论上智能体可以记住精确的动作序列来最大化最终得分。这并不理想，所以我们建议使用 [SuperSuit](https://github.com/Farama-Foundation/SuperSuit) 的 `sticky_actions` 包装器（示例见下文）。这是 *"Machado et al. (2018)，"回顾街机学习环境：通用智能体的评估协议和开放问题"* 推荐的方法。
* **帧闪烁**：由于硬件限制，Atari 游戏通常不会每帧都渲染所有精灵。相反，精灵（如 Joust 中的骑士）有时每隔一帧渲染一次，甚至（在 Wizard of Wor 中）每三帧渲染一次。处理这个问题的标准方法是计算前两个观察值的逐像素最大值（实现方法见下文）。

### 预处理

我们建议使用 [supersuit](https://github.com/Farama-Foundation/SuperSuit) 库进行预处理。这组环境的特定依赖项可以通过以下命令安装：

````bash
pip install supersuit
````

以下是 Atari 预处理的示例用法：

```python
import supersuit
from pettingzoo.atari import space_invaders_v2

env = space_invaders_v2.env()

# 按照 openai baseline 的 MaxAndSKip 包装器，
# 对最后 2 帧取最大值来处理帧闪烁问题
env = supersuit.max_observation_v0(env, 2)

# repeat_action_probability 设置为 0.25 以引入非确定性
env = supersuit.sticky_actions_v0(env, repeat_action_probability=0.25)

# 跳过帧以加快处理速度并减少控制
# 要与 gym 兼容，使用 frame_skip(env, (2,5))
env = supersuit.frame_skip_v0(env, 4)

# 缩小观察值以加快处理速度
env = supersuit.resize_v1(env, 84, 84)

# 允许智能体看到屏幕上的所有内容，尽管 Atari 存在屏幕闪烁问题
env = supersuit.frame_stack_v1(env, 4)
```

### 通用参数

所有 Atari 环境都有以下环境参数：

```python
# 以太空入侵者为例，但可以替换为任何 Atari 游戏
from pettingzoo.atari import space_invaders_v2

space_invaders_v2.env(obs_type='rgb_image', full_action_space=True, max_cycles=100000, auto_rom_install_path=None)
```

`obs_type`：此参数有三个可能的值：

* 'rgb_image'（默认）- 产生一个类似人类玩家看到的 RGB 图像。
* 'grayscale_image' - 产生一个灰度图像。
* 'ram' - 产生 Atari 主机 RAM 的 1024 位观察值。

`full_action_space`：将此选项设置为 True 会将动作空间设置为完整的 18 个动作。设置为 `False`（默认）会移除重复的动作，只保留唯一的动作。

`max_cycles`：游戏终止前的帧数（每个智能体可以采取的步数）。

`auto_rom_install_path`：使用 [Farama-Foundation/AutoROM](https://github.com/Farama-Foundation/AutoROM) 工具安装的 AutoROM 安装路径。
这是您安装 AutoROM 时指定的路径。例如，如果您使用拳击 Atari 环境，
那么库会在 `/auto_rom_install_path/ROM/boxing/boxing.bin` 路径查找 ROM。
如果未指定（值为 `None`），则库会在默认的 AutoROM 路径查找已安装的 ROM。

### 引用

街机学习环境的多人游戏在以下论文中引入：

```
@article{terry2020arcade,
  Title = {Multiplayer Support for the Arcade Learning Environment},
  Author = {Terry, J K and Black, Benjamin},
  journal={arXiv preprint arXiv:2009.09341},
  year={2020}
}
```

街机学习环境最初在以下论文中引入：

```
@Article{bellemare13arcade,
  author = { {Bellemare}, M.~G. and {Naddaf}, Y. and {Veness}, J. and {Bowling}, M.},
  title = {The Arcade Learning Environment: An Evaluation Platform for General Agents},
  journal = {Journal of Artificial Intelligence Research},
  year = "2013",
  month = "jun",
  volume = "47",
  pages = "253--279",
}
```

街机学习环境的各种扩展在以下论文中引入：

```
@article{machado2018revisiting,
  title={Revisiting the arcade learning environment: Evaluation protocols and open problems for general agents},
  author={Machado, Marlos C and Bellemare, Marc G and Talvitie, Erik and Veness, Joel and Hausknecht, Matthew and Bowling, Michael},
  journal={Journal of Artificial Intelligence Research},
  volume={61},
  pages={523--562},
  year={2018}
}
