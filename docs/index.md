---
hide-toc: true
firstpage:
lastpage:
---

```{toctree}
:hidden:
:caption: 介绍

content/basic_usage
content/environment_creation
content/environment_tests
```

```{toctree}
:hidden:
:caption: API

api/aec
api/parallel
api/wrappers
api/utils
```

```{toctree}
:hidden:
:caption: 环境

environments/atari
environments/butterfly
environments/classic
environments/mpe
environments/sisl
environments/third_party_envs
```

```{toctree}
:hidden:
:caption: 教程

tutorials/custom_environment/index
tutorials/cleanrl/index
tutorials/tianshou/index
tutorials/rllib/index
tutorials/langchain/index
tutorials/sb3/index
tutorials/agilerl/index
```

```{toctree}
:hidden:
:caption: 开发

Github <https://github.com/Farama-Foundation/PettingZoo>
release_notes/index
参与文档编写 <https://github.com/Farama-Foundation/PettingZoo/tree/master/docs/>
```

```{project-logo} _static/img/pettingzoo-text.png
:alt: PettingZoo 标志
```

```{project-heading}
多智能体强化学习的 API 标准。
```

```{figure} _static/img/environments-demo.gif
    :width: 480px
    :name: PettingZoo 环境
```

**PettingZoo 是一个简单、符合 Python 风格的接口，能够表示通用的多智能体强化学习（MARL）问题。**
PettingZoo 包含各种参考环境、有用的工具，以及用于创建自定义环境的工具。

[AEC API](/api/aec/) 支持顺序回合制环境，而 [Parallel API](/api/parallel/) 支持同时行动的环境。

与环境的交互接口类似于 [Gymnasium](https://gymnasium.farama.org)：

```python
from pettingzoo.butterfly import knights_archers_zombies_v10
env = knights_archers_zombies_v10.env(render_mode="human")
env.reset(seed=42)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        # 这里是你插入策略的地方
        action = env.action_space(agent).sample()

    env.step(action)
```
