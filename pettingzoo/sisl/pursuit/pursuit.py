# noqa: D212, D415
"""
# 追捕游戏

```{figure} sisl_pursuit.gif
:width: 140px
:name: pursuit
```

此环境是 <a href='..'>SISL 环境</a>的一部分。请先阅读该页面以了解基本信息。

| 导入                  | `from pettingzoo.sisl import pursuit_v4`               |
|----------------------|--------------------------------------------------------|
| 动作空间             | 离散                                                    |
| 并行 API            | 支持                                                    |
| 手动控制             | 支持                                                    |
| 智能体               | `agents= ['pursuer_0', 'pursuer_1', ..., 'pursuer_7']` |
| 智能体数量           | 8 (+/-)                                                |
| 动作形状             | (5)                                                    |
| 动作值域             | Discrete(5)                                            |
| 观察空间形状         | (7, 7, 3)                                              |
| 观察值域             | [0, 30]                                                |


默认情况下，30个蓝色逃避者和8个红色追捕者被放置在一个16 x 16的网格中，中心有一个白色障碍物。逃避者随机移动，追捕者由玩家控制。每当追捕者完全包围一个逃避者时，每个参与包围的追捕者都会获得5点奖励，
且该逃避者会从环境中移除。追捕者每次接触到逃避者也会获得0.01点奖励。追捕者具有上、下、左、右和停留这5个离散动作。每个追捕者观察以自己为中心的7 x 7网格区域，用红色追捕者周围的橙色方框表示。
当所有逃避者都被捕获或者游戏进行了500个周期后，环境终止。注意，该环境已经应用了PettingZoo论文第4.1节中描述的奖励优化。

观察空间的完整形状为`(obs_range, obs_range, 3)`，其中第一个通道用1表示墙壁的位置，第二个通道表示每个坐标中盟友的数量，第三个通道表示每个坐标中对手的数量。

### 手动控制

使用'J'和'K'键选择不同的追捕者。被选中的追捕者可以用方向键移动。


### 参数

``` python
pursuit_v4.env(max_cycles=500, x_size=16, y_size=16, shared_reward=True, n_evaders=30,
n_pursuers=8,obs_range=7, n_catch=2, freeze_evaders=False, tag_reward=0.01,
catch_reward=5.0, urgency_reward=-0.1, surround=True, constraint_window=1.0)
```

`x_size, y_size`: 环境世界空间的大小

`shared_reward`: 是否在所有智能体之间分配奖励

`n_evaders`: 逃避者的数量

`n_pursuers`: 追捕者的数量

`obs_range`: 智能体观察的周围区域大小

`n_catch`: 捕获一个逃避者所需的追捕者数量

`freeze_evaders`: 控制逃避者是否可以移动

`tag_reward`: "标记"或接触单个逃避者的奖励

`term_pursuit`: 追捕者成功捕获逃避者时的额外奖励

`urgency_reward`: 每一步给予智能体的奖励

`surround`: 控制逃避者是在被包围时移除，还是在n_catch个追捕者位于逃避者位置时移除

`constraint_window`: 智能体可以随机生成的区域大小（从中心计算，按比例单位）。默认为1.0，表示可以在地图任何位置生成。值为0表示所有智能体都在中心生成。

`max_cycles`: 在max_cycles步后所有智能体将返回完成状态


### 版本历史

* v4: 更改了奖励共享机制，修复了一个收集bug，在渲染中添加了智能体计数 (1.14.0)
* v3: 修复了观察空间bug (1.5.0)
* v2: 修复了各种bug (1.4.0)
* v1: 各种修复和环境参数变更 (1.3.1)
* v0: 初始版本发布 (1.0.0)

"""

import numpy as np
import pygame
from gymnasium.utils import EzPickle

from pettingzoo import AECEnv
from pettingzoo.sisl.pursuit.manual_policy import ManualPolicy
from pettingzoo.sisl.pursuit.pursuit_base import Pursuit as _env
from pettingzoo.utils import AgentSelector, wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn

__all__ = ["ManualPolicy", "env", "parallel_env", "raw_env"]


def env(**kwargs):
    """创建并返回一个带有包装器的追捕环境"""
    env = raw_env(**kwargs)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


parallel_env = parallel_wrapper_fn(env)


class raw_env(AECEnv, EzPickle):
    """原始追捕环境类"""
    metadata = {
        "render_modes": ["human", "rgb_array"],  # 渲染模式：人类观看和RGB数组
        "name": "pursuit_v4",  # 环境名称
        "is_parallelizable": True,  # 是否可并行化
        "render_fps": 5,  # 渲染帧率
        "has_manual_policy": True,  # 是否支持手动策略
    }

    def __init__(self, *args, **kwargs):
        """初始化环境

        参数：
            *args：位置参数
            **kwargs：关键字参数，包括render_mode等
        """
        EzPickle.__init__(self, *args, **kwargs)
        self.env = _env(*args, **kwargs)
        self.render_mode = kwargs.get("render_mode")
        pygame.init()
        self.agents = ["pursuer_" + str(a) for a in range(self.env.num_agents)]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.num_agents))))
        self._agent_selector = AgentSelector(self.agents)
        # 空间
        self.n_act_agents = self.env.act_dims[0]
        self.action_spaces = dict(zip(self.agents, self.env.action_space))
        self.observation_spaces = dict(zip(self.agents, self.env.observation_space))
        self.steps = 0
        self.closed = False

    def reset(self, seed=None, options=None):
        """重置环境

        参数：
            seed：随机种子
            options：重置选项
        """
        if seed is not None:
            self.env._seed(seed=seed)
        self.steps = 0
        self.agents = self.possible_agents[:]
        self.rewards = dict(zip(self.agents, [(0) for _ in self.agents]))
        self._cumulative_rewards = dict(zip(self.agents, [(0) for _ in self.agents]))
        self.terminations = dict(zip(self.agents, [False for _ in self.agents]))
        self.truncations = dict(zip(self.agents, [False for _ in self.agents]))
        self.infos = dict(zip(self.agents, [{} for _ in self.agents]))
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()
        self.env.reset()

    def close(self):
        """关闭环境"""
        if not self.closed:
            self.closed = True
            self.env.close()

    def render(self):
        """渲染环境"""
        if not self.closed:
            return self.env.render()

    def step(self, action):
        """执行一步动作

        参数：
            action：要执行的动作
        """
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return
        agent = self.agent_selection
        self.env.step(
            action, self.agent_name_mapping[agent], self._agent_selector.is_last()
        )
        for k in self.terminations:
            if self.env.frames >= self.env.max_cycles:
                self.truncations[k] = True
            else:
                self.terminations[k] = self.env.is_terminal
        for k in self.agents:
            self.rewards[k] = self.env.latest_reward_state[self.agent_name_mapping[k]]
        self.steps += 1

        self._cumulative_rewards[self.agent_selection] = 0
        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()

    def observe(self, agent):
        """获取指定智能体的观察

        参数：
            agent：智能体名称
        """
        o = self.env.safely_observe(self.agent_name_mapping[agent])
        return np.swapaxes(o, 2, 0)

    def observation_space(self, agent: str):
        """返回指定智能体的观察空间

        参数：
            agent：智能体名称
        """
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        """返回指定智能体的动作空间

        参数：
            agent：智能体名称
        """
        return self.action_spaces[agent]
