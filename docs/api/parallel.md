---
title: 并行
---


# 并行 API

除了主要的 API 外，我们还有一个次要的并行 API，用于所有智能体具有同时动作和观察的环境。支持并行 API 的环境可以通过 `<game>.parallel_env()` 创建。这个 API 基于*部分可观察随机游戏*（POSGs）范式，其细节类似于 [RLlib 的多智能体环境规范](https://docs.ray.io/en/latest/rllib-env.html#multi-agent-and-hierarchical)，但我们允许智能体之间有不同的观察和动作空间。

有关与 AEC API 的比较，请参见[关于 AEC](https://pettingzoo.farama.org/api/aec/#about-aec)。更多信息，请参见[*PettingZoo：多智能体强化学习的标准 API*](https://arxiv.org/pdf/2009.14471.pdf)。

[PettingZoo 包装器](/api/wrappers/pz_wrappers/)可用于在并行和 AEC 环境之间转换，但有一些限制（例如，AEC 环境必须只在每个循环结束时更新一次）。

## 示例

[PettingZoo Butterfly](/environments/butterfly/) 提供了并行环境的标准示例，如 [Pistonball](/environments/butterfly/pistonball)。

我们提供了创建两个自定义并行环境的教程：[石头剪刀布（并行）](https://pettingzoo.farama.org/content/environment_creation/#example-custom-parallel-environment)和一个简单的[网格世界环境](/tutorials/custom_environment/2-environment-logic/)。

## 用法

可以按以下方式与并行环境交互：

``` python
from pettingzoo.butterfly import pistonball_v6
parallel_env = pistonball_v6.parallel_env(render_mode="human")
observations, infos = parallel_env.reset(seed=42)

while parallel_env.agents:
    # 这里是你插入策略的地方
    actions = {agent: parallel_env.action_space(agent).sample() for agent in parallel_env.agents}

    observations, rewards, terminations, truncations, infos = parallel_env.step(actions)
parallel_env.close()
```

## ParallelEnv

```{eval-rst}
.. currentmodule:: pettingzoo.utils.env

.. autoclass:: ParallelEnv

    .. py:attribute:: agents

        所有当前智能体的名称列表，通常是整数。这些可能会随着环境的进展而改变（即可以添加或删除智能体）。

        :type: list[AgentID]

    .. py:attribute:: num_agents

        agents 列表的长度。

        :type: int

    .. py:attribute:: possible_agents

        环境可能生成的所有可能智能体的列表。等同于观察和动作空间中的智能体列表。这不能通过游戏或重置来改变。

        :type: list[AgentID]

    .. py:attribute:: max_num_agents

        possible_agents 列表的长度。

        :type: int

    .. py:attribute:: observation_spaces

        每个智能体的观察空间字典，以名称为键。这不能通过游戏或重置来改变。

        :type: Dict[AgentID, gym.spaces.Space]

    .. py:attribute:: action_spaces

        每个智能体的动作空间字典，以名称为键。这不能通过游戏或重置来改变。

        :type: Dict[AgentID, gym.spaces.Space]

    .. automethod:: step

        执行所有智能体的动作。

        参数:
            actions (dict): 一个字典，包含每个智能体的动作，以智能体名称为键。

        返回:
            tuple: 包含以下元素的元组：
                - observations (dict): 每个智能体的新观察的字典
                - rewards (dict): 每个智能体的奖励的字典
                - terminations (dict): 每个智能体的终止状态的字典
                - truncations (dict): 每个智能体的截断状态的字典
                - infos (dict): 每个智能体的附加信息的字典

    .. automethod:: reset

        重置环境到初始状态，并返回第一个观察。

        参数:
            seed (int, optional): 随机数生成器的种子。
            options (dict, optional): 用于自定义重置行为的其他选项。

        返回:
            tuple: 包含以下元素的元组：
                - observations (dict): 每个智能体的初始观察的字典
                - infos (dict): 每个智能体的附加信息的字典

    .. automethod:: render

        渲染环境。

    .. automethod:: close

        关闭环境，清理任何打开的资源。

    .. automethod:: state

        返回环境的全局状态。

    .. automethod:: observation_space

        返回指定智能体的观察空间。

        参数:
            agent (str): 智能体的名称

        返回:
            gym.spaces.Space: 智能体的观察空间

    .. automethod:: action_space

        返回指定智能体的动作空间。

        参数:
            agent (str): 智能体的名称

        返回:
            gym.spaces.Space: 智能体的动作空间

```
