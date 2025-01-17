---
title: 环境创建
---
# 环境创建

本文档概述了创建新环境以及 PettingZoo 中包含的相关有用的包装器、工具和测试，这些都是为创建新环境而设计的。

我们将通过创建一个简单的石头剪刀布环境来演示，包括 [AEC](/api/aec/) 和 [Parallel](/api/parallel/) 环境的示例代码。

查看我们的[自定义环境教程](/tutorials/custom_environment/index)，了解创建自定义环境的完整流程，包括复杂的环境逻辑和非法动作掩码。

## 示例自定义环境

这是一个经过仔细注释的 PettingZoo 石头剪刀布环境版本。

```{eval-rst}
.. literalinclude:: ../code_examples/aec_rps.py
   :language: python
```

要与你的自定义 AEC 环境交互，使用以下代码：

```{eval-rst}
.. literalinclude:: ../code_examples/aec_rps_usage.py
   :language: python
```

## 示例自定义并行环境

```{eval-rst}
.. literalinclude:: ../code_examples/parallel_rps.py
   :language: python
```

要与你的自定义并行环境交互，使用以下代码：

```{eval-rst}
.. literalinclude:: ../code_examples/parallel_rps_usage.py
   :language: python
```

## 使用包装器

包装器是一个环境转换，它接收一个环境作为输入，并输出一个与输入环境相似但应用了某些转换或验证的新环境。PettingZoo 提供了[包装器来转换环境](/api/pz_wrappers)，在 AEC API 和并行 API 之间来回转换，以及一组简单的[实用包装器](/api/pz_wrappers)，提供输入验证和其他方便的可重用逻辑。PettingZoo 还通过 SuperSuit 配套包（`pip install supersuit`）包含了[包装器](/api/supersuit_wrappers)。

```python
from pettingzoo.butterfly import pistonball_v6
from pettingzoo.utils import ClipOutOfBoundsWrapper

env = pistonball_v6.env()
wrapped_env = ClipOutOfBoundsWrapper(env)
# 包装的环境在使用前必须重置
wrapped_env.reset()
```

## 开发者工具

utils 目录包含一些对调试环境有帮助的函数。这些在 API 文档中有记录。

utils 目录还包含一些只对开发新环境有帮助的类。这些记录如下。

### 智能体选择器

`AgentSelector` 类循环遍历智能体

它可以用如下方式循环遍历智能体列表：

```python
from pettingzoo.utils import AgentSelector
agents = ["agent_1", "agent_2", "agent_3"]
selector = AgentSelector(agents)
agent_selection = selector.reset()
# agent_selection 将是 "agent_1"
for i in range(100):
    agent_selection = selector.next()
    # 将依次选择 "agent_2", "agent_3", "agent_1", "agent_2", "agent_3", ...
```

### 已弃用模块

DeprecatedModule 在 PettingZoo 中用于帮助引导用户远离旧的过时环境版本，转向新版本。如果你想创建类似的版本控制系统，这可能会有帮助。

例如，当用户尝试导入 `knights_archers_zombies_v0` 环境时，他们导入以下变量（定义在 `pettingzoo/butterfly/__init__.py` 中）：
``` python
from pettingzoo.utils.deprecated_module import DeprecatedModule
knights_archers_zombies_v0 = DeprecatedModule("knights_archers_zombies", "v0", "v10")
```
这个声明告诉用户 `knights_archers_zombies_v0` 已经弃用，应该使用 `knights_archers_zombies_v10` 代替。具体来说，它会给出以下错误：
``` python notest
from pettingzoo.butterfly import knights_archers_zombies_v0
knights_archers_zombies_v0.env()
# pettingzoo.utils.deprecated_module.DeprecatedEnv: knights_archers_zombies_v0 现在已弃用，请使用 knights_archers_zombies_v10 代替
```
