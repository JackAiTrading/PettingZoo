# noqa: D212, D415
"""
# 多行走者

```{figure} sisl_multiwalker.gif
:width: 140px
:name: multiwalker
```

此环境是<a href='..'>SISL环境</a>的一部分。请先阅读该页面以获取一般信息。

| 导入                 | `from pettingzoo.sisl import multiwalker_v9`   |
|----------------------|------------------------------------------------|
| 动作空间            | 连续                                           |
| 并行API             | 是                                             |
| 手动控制            | 否                                             |
| 智能体              | `agents= ['walker_0', 'walker_1', 'walker_2']` |
| 智能体数量          | 3                                              |
| 动作形状            | (4,)                                           |
| 动作值范围          | (-1, 1)                                        |
| 观察形状            | (31,)                                          |
| 观察值范围          | [-inf,inf]                                     |


在这个环境中，双足机器人试图携带放置在它们顶部的包裹向右移动。默认情况下，机器人数量设置为3。

每个行走者获得的奖励等于包裹相对于前一时间步的位置变化，乘以`forward_reward`缩放因子。最大可达到的总奖励取决于地形长度；作为参考，对于长度为75的地形，在最优策略下的总奖励约为300。

如果包裹掉落，或者包裹超出地形的左边缘，环境就会结束。默认情况下，如果任何行走者摔倒，环境也会结束。在所有这些情况下，每个行走者都会收到-100的奖励。如果包裹从地形的右边缘掉落，环境也会结束，此时奖励为0。

当行走者摔倒时，它会额外受到-10的惩罚。如果`terminate_on_fall = False`，则当行走者摔倒时环境不会结束，只有当包裹掉落时才会结束。如果`remove_on_fall = True`，摔倒的行走者会从环境中移除。智能体还会收到一个小的形状奖励，即头部角度变化的-5倍，以保持头部水平。

如果选择了`shared_reward`（默认为True），智能体的个人奖励会被平均，得到一个单一的平均奖励，该奖励会返回给所有智能体。

每个行走者在其两条腿的两个关节上施加力，形成一个由4个元素组成的连续动作空间。每个行走者通过31个元素的向量进行观察，该向量包含关于环境的模拟噪声激光雷达数据和关于相邻行走者的信息。环境的持续时间默认上限为500帧（可以通过`max_cycles`设置控制）。



### 观察空间

每个智能体接收的观察由其腿部和关节的各种物理属性组成，以及来自机器人正前方和下方空间的激光雷达读数。观察还包括关于相邻行走者和包裹的信息。邻居和包裹的观察具有正态分布的信号噪声，由`position_noise`和`angle_noise`控制。对于没有邻居的行走者，关于邻居位置的观察为零。



此表列举了观察空间：

| 索引: [起始, 结束) | 描述                                                       |   值范围    |
|:-----------------:|------------------------------------------------------------|:---------------:|
|          0          | 躯体角度                |  [0, 2*pi]  |
|          1          | 躯体角速度                                               | [-inf, inf] |
|          2          | X方向速度                                                | [-1, 1]     |
|          3          | Y方向速度                                                | [-1, 1]     |
|          4          | 髋关节1角度                                              | [-inf, inf] |
|          5          | 髋关节1速度                                              | [-inf, inf] |
|          6          | 膝关节1角度                                              | [-inf, inf] |
|          7          | 膝关节1速度                                              | [-inf, inf] |
|          8          | 腿1地面接触标志                                          |   {0, 1}    |
|          9          | 髋关节1角度                                              | [-inf, inf] |
|         10          | 髋关节2速度                                              | [-inf, inf] |
|         11          | 膝关节2角度                                              | [-inf, inf] |
|         12          | 膝关节2速度                                              | [-inf, inf] |
|         13          | 腿2地面接触标志                                          |   {0, 1}    |
|        14-23        | 激光雷达传感器读数                                       | [-inf, inf] |
|         24          | 左侧邻居相对X位置（最左边行走者为0.0）（有噪声）        | [-inf, inf] |
|         25          | 左侧邻居相对Y位置（最左边行走者为0.0）（有噪声）        | [-inf, inf] |
|         26          | 右侧邻居相对X位置（最右边行走者为0.0）（有噪声）        | [-inf, inf] |
|         27          | 右侧邻居相对Y位置（最右边行走者为0.0）（有噪声）        | [-inf, inf] |
|         28          | 行走者相对于包裹的X位置（左边缘为0，右边缘为1）（有噪声）| [-inf, inf] |
|         29          | 行走者相对于包裹的Y位置（有噪声）                        | [-inf, inf] |
|         30          | 包裹角度（有噪声）                                        | [-inf, inf] |

### 参数

``` python
multiwalker_v9.env(n_walkers=3, position_noise=1e-3, angle_noise=1e-3, forward_reward=1.0, terminate_reward=-100.0, fall_reward=-10.0, shared_reward=True,
terminate_on_fall=True, remove_on_fall=True, terrain_length=200, max_cycles=500)
```



`n_walkers`: 环境中双足行走者智能体的数量

`position_noise`: 应用于邻居和包裹位置观察的噪声

`angle_noise`: 应用于邻居和包裹旋转观察的噪声

`forward_reward`: 获得的奖励是`forward_reward`乘以包裹位置的变化

`fall_reward`: 当智能体摔倒时应用的奖励

`shared_reward`: 奖励是在所有智能体之间分配还是单独分配

`terminate_reward`: 如果行走者未能将包裹携带到地形右边缘，每个行走者获得的奖励

`terminate_on_fall`: 如果为`True`（默认），一个行走者摔倒会导致所有智能体结束，并且他们都会收到额外的`terminate_reward`。如果为`False`，则只有摔倒的智能体会收到`fall_reward`，其他智能体不会结束，即环境继续运行。

`remove_on_fall`: 当行走者摔倒时将其移除（仅在`terminate_on_fall`为False时有效）

`terrain_length`: 地形长度（以步数计）

`max_cycles`: 在max_cycles步后所有智能体都将返回done


### 版本历史
* v8: 替换了local_ratio，修复了奖励，将地形长度作为参数，并更新了文档 (1.15.0)
* v7: 修复了行走者碰撞问题 (1.8.2)
* v6: 修复了观察空间并大幅提高了代码质量 (1.5.0)
* v5: 修复了奖励结构，添加了参数 (1.4.2)
* v4: 修复了各种bug (1.4.0)
* v3: 修复了观察空间 (1.3.3)
* v2: 各种修复和环境参数更改 (1.3.1)
* v1: 修复了所有环境如何处理提前结束的问题 (1.3.0)
* v0: 初始版本发布 (1.0.0)

"""

import numpy as np
from gymnasium.utils import EzPickle

from pettingzoo import AECEnv
from pettingzoo.sisl.multiwalker.multiwalker_base import FPS
from pettingzoo.sisl.multiwalker.multiwalker_base import MultiWalkerEnv as _env
from pettingzoo.utils import AgentSelector, wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn


def env(**kwargs):
    """创建并返回一个带有包装器的多行走者环境"""
    env = raw_env(**kwargs)
    env = wrappers.ClipOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


parallel_env = parallel_wrapper_fn(env)


class raw_env(AECEnv, EzPickle):
    """原始多行走者环境类"""
    metadata = {
        "render_modes": ["human", "rgb_array"],  # 渲染模式：人类观看和RGB数组
        "name": "multiwalker_v9",                # 环境名称
        "is_parallelizable": True,               # 是否可并行化
        "render_fps": FPS,                       # 渲染帧率
    }

    def __init__(self, *args, **kwargs):
        """初始化环境

        参数：
            *args：位置参数
            **kwargs：关键字参数
        """
        EzPickle.__init__(self, *args, **kwargs)
        self.env = _env(*args, **kwargs)
        self.render_mode = self.env.render_mode
        self.agents = ["walker_" + str(r) for r in range(self.env.n_walkers)]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(
            zip(self.agents, list(range(self.env.n_walkers)))
        )
        self._agent_selector = AgentSelector(self.agents)
        # 空间定义
        self.action_spaces = dict(zip(self.agents, self.env.action_space))
        self.observation_spaces = dict(zip(self.agents, self.env.observation_space))
        self.state_space = self.env.state_space
        self.steps = 0

    def observation_space(self, agent):
        """返回指定智能体的观察空间"""
        return self.observation_spaces[agent]

    def action_space(self, agent):
        """返回指定智能体的动作空间"""
        return self.action_spaces[agent]

    def convert_to_dict(self, list_of_list):
        """将列表转换为以智能体为键的字典"""
        return dict(zip(self.agents, list_of_list))

    def reset(self, seed=None, options=None):
        """重置环境

        参数：
            seed：随机种子
            options：重置选项
        """
        if seed is not None:
            self.env._seed(seed=seed)
        self.env.reset()
        self.steps = 0
        self.agents = self.possible_agents[:]
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()
        self._cumulative_rewards = dict(zip(self.agents, [(0) for _ in self.agents]))
        self.rewards = dict(zip(self.agents, [(0) for _ in self.agents]))
        self.terminations = dict(zip(self.agents, [False for _ in self.agents]))
        self.truncations = dict(zip(self.agents, [False for _ in self.agents]))
        self.infos = dict(zip(self.agents, [{} for _ in self.agents]))

    def close(self):
        """关闭环境"""
        self.env.close()

    def render(self):
        """渲染环境"""
        return self.env.render()

    def state(self):
        """返回环境的状态"""
        return self.env.state()

    def observe(self, agent):
        """返回指定智能体的观察"""
        return self.env.observe(self.agent_name_mapping[agent])

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
        action = np.array(action, dtype=np.float32)
        is_last = self._agent_selector.is_last()
        self.env.step(action, self.agent_name_mapping[agent], is_last)
        if is_last:
            last_rewards = self.env.get_last_rewards()
            for agent in self.rewards:
                self.rewards[agent] = last_rewards[self.agent_name_mapping[agent]]
            for agent in self.terminations:
                self.terminations[agent] = self.env.get_last_dones()[
                    self.agent_name_mapping[agent]
                ]
            self.agent_name_mapping = {
                agent: i
                for i, (agent, done) in enumerate(
                    zip(self.possible_agents, self.env.get_last_dones())
                )
            }
            iter_agents = self.agents[:]
            for agent in self.terminations:
                if self.terminations[agent] or self.truncations[agent]:
                    iter_agents.remove(agent)
            self._agent_selector.reinit(iter_agents)
        else:
            self._clear_rewards()
        if self._agent_selector.agent_order:
            self.agent_selection = self._agent_selector.next()

        if self.env.frames >= self.env.max_cycles:
            self.terminations = dict(zip(self.agents, [True for _ in self.agents]))

        self._cumulative_rewards[agent] = 0
        self._accumulate_rewards()
        self._deads_step_first()
        self.steps += 1

        if self.render_mode == "human":
            self.render()
