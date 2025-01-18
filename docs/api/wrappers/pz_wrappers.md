---
title: PettingZoo 包装器
---

# PettingZoo 包装器

PettingZoo 包含以下类型的包装器：
* [转换包装器](#conversion-wrappers)：用于在 [AEC](/api/aec/) 和 [并行](/api/parallel/) API 之间转换环境的包装器
* [实用包装器](#utility-wrappers)：一组提供便利可重用逻辑的包装器，例如强制执行回合顺序或裁剪超出范围的动作。

## 转换包装器

### AEC 转并行

```{eval-rst}
.. currentmodule:: pettingzoo.utils.conversions

.. automodule:: pettingzoo.utils.conversions
   :members: aec_to_parallel
   :undoc-members:
```

可以使用下面显示的 `aec_to_parallel` 包装器将环境从 AEC 环境转换为并行环境。请注意，此包装器对底层环境做出以下假设：

1. 环境按循环步进，即按顺序遍历每个活动智能体。
2. 环境只在循环结束时更新智能体的观察。

PettingZoo 中的大多数并行环境只在循环结束时分配奖励。在这些环境中，AEC API 和并行 API 的奖励方案是等效的。如果 AEC 环境确实在循环内分配奖励，那么奖励将在 AEC 环境和并行环境中的不同时间步分配。特别是，AEC 环境将从智能体一次步进到下一次步进时分配所有奖励，而并行环境将从第一个智能体步进到最后一个智能体步进时分配所有奖励。

要将 AEC 环境转换为并行环境：
``` python
from pettingzoo.utils.conversions import aec_to_parallel
from pettingzoo.butterfly import pistonball_v6
env = pistonball_v6.env()
env = aec_to_parallel(env)
```

### 并行转 AEC

```{eval-rst}
.. currentmodule:: pettingzoo.utils.conversions

.. automodule:: pettingzoo.utils.conversions
   :members: parallel_to_aec
   :undoc-members:
```

任何并行环境都可以使用 `parallel_to_aec` 包装器高效地转换为 AEC 环境。

要将并行环境转换为 AEC 环境：
``` python
from pettingzoo.utils import parallel_to_aec
from pettingzoo.butterfly import pistonball_v6
env = pistonball_v6.parallel_env()
env = parallel_to_aec(env)
```

## 实用包装器

我们希望我们的 pettingzoo 环境既易于使用又易于实现。为了结合这两点，我们有一组简单的包装器，提供输入验证和其他便利的可重用逻辑。

您可以按照以下示例类似的方式将这些包装器应用到您的环境：

包装 AEC 环境：
```python
from pettingzoo.utils import TerminateIllegalWrapper
from pettingzoo.classic import tictactoe_v3
env = tictactoe_v3.env()
env = TerminateIllegalWrapper(env, illegal_reward=-1)

env.reset()
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    if termination or truncation:
        action = None
    else:
        action = env.action_space(agent).sample()  # 这里是您插入策略的地方
    env.step(action)
env.close()
```
注意：大多数 AEC 环境在初始化时都包含 TerminateIllegalWrapper，所以这段代码不会改变环境的行为。

包装并行环境：
```python
from pettingzoo.utils import BaseParallelWrapper
from pettingzoo.butterfly import pistonball_v6

parallel_env = pistonball_v6.parallel_env(render_mode="human")
parallel_env = BaseParallelWrapper(parallel_env)

observations, infos = parallel_env.reset()

while parallel_env.agents:
    actions = {agent: parallel_env.action_space(agent).sample() for agent in parallel_env.agents}  # 这里是您插入策略的地方
    observations, rewards, terminations, truncations, infos = parallel_env.step(actions)
```

```{eval-rst}
.. warning::

    目前包含的 PettingZoo 包装器不支持并行环境，要使用它们，您必须将环境转换为 AEC，应用包装器，然后再转换回并行。
```
```python
from pettingzoo.utils import ClipOutOfBoundsWrapper
from pettingzoo.sisl import multiwalker_v9
from pettingzoo.utils import aec_to_parallel

parallel_env = multiwalker_v9.env(render_mode="human")
parallel_env = ClipOutOfBoundsWrapper(parallel_env)
parallel_env = aec_to_parallel(parallel_env)

observations, infos = parallel_env.reset()

while parallel_env.agents:
    actions = {agent: parallel_env.action_space(agent).sample() for agent in parallel_env.agents}  # 这里是您插入策略的地方
    observations, rewards, terminations, truncations, infos = parallel_env.step(actions)
```

```{eval-rst}
.. currentmodule:: pettingzoo.utils.wrappers

.. autoclass:: BaseWrapper
.. autoclass:: TerminateIllegalWrapper
.. autoclass:: CaptureStdoutWrapper
.. autoclass:: AssertOutOfBoundsWrapper
.. autoclass:: ClipOutOfBoundsWrapper
.. autoclass:: OrderEnforcingWrapper

```
