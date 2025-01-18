# noqa: D212, D415
"""
水世界环境。

这个环境模拟了一个水下多智能体探索场景，智能体需要在充满食物和
毒物的水域中生存，通过合作来获取资源并避免危险。

主要特点：
1. 连续控制
2. 资源竞争
3. 危险规避
4. 团队合作

环境规则：
1. 基本设置
   - 智能体
   - 食物点
   - 毒物点
   - 活动空间

2. 交互规则
   - 连续移动
   - 碰撞检测
   - 资源获取
   - 伤害计算

3. 智能体行为
   - 目标追踪
   - 危险避免
   - 能量管理
   - 团队协作

4. 终止条件
   - 能量耗尽
   - 中毒死亡
   - 目标完成
   - 时间结束

环境参数：
- 观察空间：局部感知范围
- 动作空间：连续运动控制
- 奖励：资源获取和生存时间
- 最大步数：由场景设置决定

环境特色：
1. 物理系统
   - 流体动力
   - 惯性效应
   - 阻力模拟
   - 碰撞响应

2. 资源系统
   - 食物分布
   - 毒物生成
   - 能量转换
   - 资源再生

3. 感知系统
   - 视野范围
   - 传感器噪声
   - 信息共享
   - 危险预警

4. 评估系统
   - 生存能力
   - 资源效率
   - 避险技巧
   - 团队协作

注意事项：
- 能量管理
- 风险评估
- 路径规划
- 团队配合
"""
"""
# 水世界

```{figure} sisl_waterworld.gif
:width: 140px
:name: waterworld
```

此环境是 <a href='..'>SISL 环境</a>的一部分。请先阅读该页面以了解基本信息。

| 导入                  | `from pettingzoo.sisl import waterworld_v4`            |
|----------------------|--------------------------------------------------------|
| 动作空间             | 连续                                                    |
| 并行 API            | 支持                                                    |
| 手动控制             | 不支持                                                  |
| 智能体               | `agents= ['pursuer_0', 'pursuer_1', ..., 'pursuer_4']` |
| 智能体数量           | 5                                                      |
| 动作形状             | (2,)                                                   |
| 动作值域             | [-0.01, 0.01]                                          |
| 观察空间形状         | (242,)                                                 |
| 观察值域             | [-√2, 2*√2]                                            |


水世界是一个古细菌在环境中导航和生存的模拟。这些被称为追捕者的古细菌试图消耗食物同时避开毒物。水世界中的智能体是追捕者，而食物和毒物属于环境。毒物的半径是追捕者半径的0.75倍，而食物的半径是追捕者半径的2倍。根据输入参数，
可能需要多个追捕者合作才能消耗食物，这创造了一个既合作又竞争的动态环境。同样，奖励可以全局分配给所有追捕者，也可以局部应用于特定追捕者。环境是一个连续的2D空间，每个追捕者都有一个位置，其x和y值均在[0,1]范围内。智能体不能
超越最小和最大x和y值的边界。智能体通过选择一个推力向量来添加到它们当前的速度上来行动。每个追捕者都有一些均匀分布的传感器，可以读取追捕者附近物体的速度和方向。这些信息在观察空间中报告，可用于在环境中导航。

### 观察空间

每个智能体的观察形状是一个长度大于4的向量，其具体长度取决于环境的输入参数。向量的完整大小是每个传感器的特征数乘以传感器数量，再加上两个元素，分别表示追捕者是否与食物或毒物发生碰撞。默认情况下，启用`speed_features`时每个传感器的特征数为8，
关闭`speed_features`时为5。因此，启用`speed_features`时，观察形状的完整形式为`(8 × n_sensors) + 2`。观察向量的元素值在[-1, 1]范围内。

例如，默认情况下有5个智能体（紫色）、5个食物目标（绿色）和10个毒物目标（红色）。每个智能体有30个范围有限的传感器，用黑线表示，用于检测邻近实体（食物和毒物目标），从而得到一个包含242个关于环境计算值的向量作为观察空间。
这些值表示每个传感器在古细菌上感知到的距离和速度。在其范围内没有感知到任何物体的传感器报告速度为0，距离为1。

这已从参考环境中修复，以防止物品漂浮到屏幕外并永远丢失。

这个表格列举了启用`speed_features = True`时的观察空间：

|        索引：[开始, 结束)         | 描述                                       |   值域     |
| :--------------------------------: | ------------------------------------------ | :---------: |
|           0 到 n_sensors           | 每个传感器的障碍物距离                     |   [0, 1]   |
|    n_sensors 到 (2 * n_sensors)    | 每个传感器的边界距离                       |   [0, 1]   |
| (2 * n_sensors) 到 (3 * n_sensors) | 每个传感器的食物距离                       |   [0, 1]   |
| (3 * n_sensors) 到 (4 * n_sensors) | 每个传感器的食物速度                       | [-2*√2, 2*√2] |
| (4 * n_sensors) 到 (5 * n_sensors) | 每个传感器的毒物距离                       |   [0, 1]   |
| (5 * n_sensors) 到 (6 * n_sensors) | 每个传感器的毒物速度                       | [-2*√2, 2*√2] |
| (6 * n_sensors) 到 (7 * n_sensors) | 每个传感器的追捕者距离                     |   [0, 1]   |
| (7 * n_sensors) 到 (8 * n_sensors) | 每个传感器的追捕者速度                     | [-2*√2, 2*√2] |
|           8 * n_sensors            | 表示智能体是否与食物碰撞                   |   {0, 1}    |
|        (8 * n_sensors) + 1         | 表示智能体是否与毒物碰撞                   |   {0, 1}    |

这个表格列举了`speed_features = False`时的观察空间：

|        索引：[开始, 结束)        | 描述                                       | 值域    |
| :-------------------------------: | ------------------------------------------ | :-----: |
|           0 - n_sensors           | 每个传感器的障碍物距离                     | [0, 1] |
|    n_sensors - (2 * n_sensors)    | 每个传感器的边界距离                       | [0, 1] |
| (2 * n_sensors) - (3 * n_sensors) | 每个传感器的食物距离                       | [0, 1] |
| (3 * n_sensors) - (4 * n_sensors) | 每个传感器的毒物距离                       | [0, 1] |
| (4 * n_sensors) - (5 * n_sensors) | 每个传感器的追捕者距离                     | [0, 1] |
|          (5 * n_sensors)          | 表示智能体是否与食物碰撞                   | {0, 1}  |
|        (5 * n_sensors) + 1        | 表示智能体是否与毒物碰撞                   | {0, 1}  |

### 动作空间

智能体具有一个连续的动作空间，表示为一个2元素向量，对应于水平和垂直推力。值的范围取决于`pursuer_max_accel`。动作值必须在`[-pursuer_max_accel, pursuer_max_accel]`范围内。如果这个动作向量的幅度超过`pursuer_max_accel`，
它将被缩放到`pursuer_max_accel`。这个速度向量会被添加到古细菌的当前速度上。

**智能体动作空间：** `[水平推力, 垂直推力]`

### 奖励

当多个智能体（取决于`n_coop`）一起捕获食物时，每个智能体都会收到`food_reward`奖励（食物不会被销毁）。他们会因接触食物而获得`encounter_reward`的塑形奖励，因接触毒物而获得`poison_reward`奖励，并且每个动作都会获得
`thrust_penalty x ||action||`奖励，其中`||action||`是动作速度的欧几里得范数。所有这些奖励都基于`local_ratio`进行分配，其中按`local_ratio`缩放的奖励（局部奖励）应用于产生奖励的智能体，而按智能体数量平均的奖励（全局奖励）
按`(1 - local_ratio)`缩放并应用于每个智能体。默认情况下，环境运行500帧。

### 参数

``` python
waterworld_v4.env(n_pursuers=5, n_evaders=5, n_poisons=10, n_coop=2, n_sensors=20,
sensor_range=0.2,radius=0.015, obstacle_radius=0.2, n_obstacles=1,
obstacle_coord=[(0.5, 0.5)], pursuer_max_accel=0.01, evader_speed=0.01,
poison_speed=0.01, poison_reward=-1.0, food_reward=10.0, encounter_reward=0.01,
thrust_penalty=-0.5, local_ratio=1.0, speed_features=True, max_cycles=500)
```

`n_pursuers`: 追捕古细菌（智能体）的数量

`n_evaders`: 食物对象的数量

`n_poisons`: 毒物对象的数量

`n_coop`: 需要同时接触食物才能消耗它的追捕古细菌（智能体）数量

`n_sensors`: 所有追捕古细菌（智能体）上的传感器数量

`sensor_range`: 所有追捕古细菌（智能体）上传感器树突的长度

`radius`: 古细菌基础半径。追捕者：radius，食物：2 x radius，毒物：3/4 x radius

`obstacle_radius`: 障碍物对象的半径

`obstacle_coord`: 障碍物对象的坐标。可以设置为`None`以使用随机位置

`pursuer_max_accel`: 追捕古细菌的最大加速度（最大动作大小）

`pursuer_speed`: 追捕者（智能体）的最大速度

`evader_speed`: 食物速度

`poison_speed`: 毒物速度

`poison_reward`: 追捕者消耗毒物对象的奖励（通常为负）

`food_reward`: 追捕者消耗食物对象的奖励

`encounter_reward`: 追捕者与食物对象碰撞的奖励

`thrust_penalty`: 用于惩罚大动作的负奖励的缩放因子

`local_ratio`: 在所有智能体之间局部分配与全局分配的奖励比例

`speed_features`: 切换追捕古细菌（智能体）传感器是否检测其他对象和古细菌的速度

`max_cycles`: 在max_cycles步后所有智能体将返回完成状态

* v4: 主要重构 (1.22.0)
* v3: 重构和主要bug修复 (1.5.0)
* v2: 修复各种bug (1.4.0)
* v1: 各种修复和环境参数变更 (1.3.1)
* v0: 初始版本发布 (1.0.0)

"""

from gymnasium.utils import EzPickle

from pettingzoo import AECEnv
from pettingzoo.sisl.waterworld.waterworld_base import FPS
from pettingzoo.sisl.waterworld.waterworld_base import WaterworldBase as _env
from pettingzoo.utils import AgentSelector, wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn


def env(**kwargs):
    env = raw_env(**kwargs)
    env = wrappers.ClipOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


parallel_env = parallel_wrapper_fn(env)


class raw_env(AECEnv, EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "waterworld_v4",
        "is_parallelizable": True,
        "render_fps": FPS,
    }

    def __init__(self, *args, **kwargs):
        EzPickle.__init__(self, *args, **kwargs)
        AECEnv.__init__(self)
        self.env = _env(*args, **kwargs)

        self.agents = ["pursuer_" + str(r) for r in range(self.env.num_agents)]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.num_agents))))
        self._agent_selector = AgentSelector(self.agents)

        # spaces
        self.action_spaces = dict(zip(self.agents, self.env.action_space))
        self.observation_spaces = dict(zip(self.agents, self.env.observation_space))
        self.has_reset = False

        self.render_mode = self.env.render_mode

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def convert_to_dict(self, list_of_list):
        return dict(zip(self.agents, list_of_list))

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.env._seed(seed=seed)
        self.has_reset = True
        self.env.reset()
        self.agents = self.possible_agents[:]
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()
        self.rewards = dict(zip(self.agents, [(0) for _ in self.agents]))
        self._cumulative_rewards = dict(zip(self.agents, [(0) for _ in self.agents]))
        self.terminations = dict(zip(self.agents, [False for _ in self.agents]))
        self.truncations = dict(zip(self.agents, [False for _ in self.agents]))
        self.infos = dict(zip(self.agents, [{} for _ in self.agents]))

    def close(self):
        if self.has_reset:
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
        is_last = self._agent_selector.is_last()
        self.env.step(action, self.agent_name_mapping[agent], is_last)

        for r in self.rewards:
            self.rewards[r] = self.env.control_rewards[self.agent_name_mapping[r]]
        if is_last:
            for r in self.rewards:
                self.rewards[r] += self.env.last_rewards[self.agent_name_mapping[r]]

        if self.env.frames >= self.env.max_cycles:
            self.truncations = dict(zip(self.agents, [True for _ in self.agents]))
        else:
            self.terminations = dict(zip(self.agents, self.env.last_dones))

        self._cumulative_rewards[self.agent_selection] = 0
        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()

    def observe(self, agent):
        return self.env.observe(self.agent_name_mapping[agent])
