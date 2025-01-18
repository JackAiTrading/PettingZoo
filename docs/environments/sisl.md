---
title: SISL 环境
firstpage:
---

# SISL 环境

```{toctree}
:hidden:
sisl/multiwalker
sisl/pursuit
sisl/waterworld
```

```{raw} html
    :file: sisl/list.html
```

SISL 环境是一组三个合作性多智能体基准环境，由 SISL（斯坦福智能系统实验室）创建，作为"使用深度强化学习的合作多智能体控制"的一部分发布。代码最初发布在：[https://github.com/sisl/MADRL](https://github.com/sisl/MADRL)

### 安装

这组环境的特定依赖项可以通过以下命令安装：

````bash
pip install 'pettingzoo[sisl]'
````

### 使用方法
要启动一个带有随机智能体的[水世界](/environments/sisl/waterworld/)环境：

```python
from pettingzoo.sisl import waterworld_v4
env = waterworld_v4.env(render_mode='human')

env.reset()
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        action = env.action_space(agent).sample() # 这里是您插入策略的地方

    env.step(action)
env.close()
```

请注意，我们对所有包含的环境进行了重大的错误修复。因此，我们不建议直接将这些环境的结果与原始论文中的结果进行比较。

如果您使用这些环境，请同时引用：

```
@inproceedings{gupta2017cooperative,
  title={Cooperative multi-agent control using deep reinforcement learning},
  author={Gupta, Jayesh K and Egorov, Maxim and Kochenderfer, Mykel},
  booktitle={International Conference on Autonomous Agents and Multiagent Systems},
  pages={66--83},
  year={2017},
  organization={Springer}
}
