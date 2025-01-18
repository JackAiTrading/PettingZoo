---
title: 工具
---


# 工具

PettingZoo 有一系列辅助工具，为与环境交互提供额外的功能。

注意：另请参见 [PettingZoo 包装器](/api/wrappers/pz_wrappers/)，它们提供了用于自定义环境的额外功能。

### 平均总奖励

```{eval-rst}
.. currentmodule:: pettingzoo.utils

.. automodule:: pettingzoo.utils.average_total_reward
   :members:
   :undoc-members:
```

环境的平均总奖励（如文档中所示）是在所有回合中对所有智能体在所有步骤中的奖励求和，然后对回合求平均。

这个值对于建立最简单的基准很重要：随机策略。

``` python
from pettingzoo.utils import average_total_reward
from pettingzoo.butterfly import pistonball_v6
env = pistonball_v6.env()
average_total_reward(env, max_episodes=100, max_steps=10000000000)
```

其中 `max_episodes` 和 `max_steps` 都限制了评估的总次数（当达到第一个限制时评估停止）

### 观察保存

```{eval-rst}
.. currentmodule:: pettingzoo.utils

.. automodule:: pettingzoo.utils.save_observation
   :members:
   :undoc-members:
```

如果游戏中的智能体进行的观察是图像，那么这些观察可以保存为图像文件。此函数接收环境和指定的智能体作为参数。如果没有指定 `agent`，则选择环境当前选定的智能体。如果将 `all_agents` 设置为 `True`，则保存环境中所有智能体的观察。默认情况下，图像保存在当前工作目录下与环境名称匹配的文件夹中。保存的图像将与观察智能体的名称匹配。如果传入 `save_dir`，将创建一个新文件夹来保存图像。如果需要，可以在训练/评估期间调用此函数，这就是为什么在使用它之前必须重置环境的原因。

``` python
from pettingzoo.utils import save_observation
from pettingzoo.butterfly import pistonball_v6
env = pistonball_v6.env()
env.reset(seed=42)
save_observation(env, agent=None, all_agents=False)
```

### 捕获标准输出

基类，用于 [CaptureStdoutWrapper](https://pettingzoo.farama.org/api/wrappers/pz_wrappers/#pettingzoo.utils.wrappers.CaptureStdoutWrapper)。将系统标准输出捕获为变量中的字符串值。


```{eval-rst}
.. currentmodule:: pettingzoo.utils

.. automodule:: pettingzoo.utils.capture_stdout
   :members:
   :undoc-members:
```

### 智能体选择器

智能体选择器工具允许在 AEC 环境中轻松循环智能体。它可以随时重置或用新的顺序重新初始化，允许改变回合顺序或处理动态数量的智能体（参见 [Knights-Archers-Zombies](https://pettingzoo.farama.org/environments/butterfly/knights_archers_zombies/) 以了解生成/销毁智能体的示例）。

注意：虽然许多 PettingZoo 环境在内部使用 AgentSelector 来管理智能体循环，但它不应该在与环境交互时在外部使用。相反，应使用 `for agent in env.agent_iter()` （参见 [AEC API 用法](https://pettingzoo.farama.org/api/aec/#usage)）。

```{eval-rst}
.. currentmodule:: pettingzoo.utils

.. automodule:: pettingzoo.utils.agent_selector
   :members:
   :exclude-members: mqueue
```

### 环境日志记录器

环境日志记录器提供了环境的常见警告和错误的功能，并允许自定义消息。它在 [PettingZoo 包装器](/api/wrappers/pz_wrappers/) 中被内部使用。

```{eval-rst}

.. currentmodule:: pettingzoo.utils
.. autoclass:: pettingzoo.utils.env_logger.EnvLogger
   :members:
   :undoc-members:
