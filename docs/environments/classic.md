---
title: 经典环境
firstpage:
---

# 经典环境

```{toctree}
:hidden:
classic/chess
classic/connect_four
classic/gin_rummy
classic/go
classic/hanabi
classic/leduc_holdem
classic/rps
classic/texas_holdem_no_limit
classic/texas_holdem
classic/tictactoe
```

```{raw} html
    :file: classic/list.html
```

经典环境实现了流行的回合制人类游戏，主要是竞争性的。

### 安装

这组环境的特定依赖项可以通过以下命令安装：

````bash
pip install 'pettingzoo[classic]'
````

### 使用方法

要启动一个带有随机智能体的[四子棋](/environments/classic/connect_four/)环境：
``` python
from pettingzoo.classic import connect_four_v3

env = connect_four_v3.env(render_mode="human")
env.reset(seed=42)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        mask = observation["action_mask"]
        action = env.action_space(agent).sample(mask)  # 这里是您插入策略的地方

    env.step(action)
env.close()
```

经典环境与本库中的其他环境有一些不同：

* 目前所有经典环境都不需要任何环境参数。
* 所有经典环境都仅通过终端打印进行渲染。
* 大多数环境只在游戏结束时给出奖励，当智能体获胜时奖励为 1，失败时为 -1。
* 许多经典环境的动作空间中包含非法动作。这些环境会在观察值中包含当前时刻的合法动作信息。这是通过字典类型的观察值实现的，其中 `observation` 元素是观察值，`action_mask` 元素是一个二进制向量，如果动作合法则对应位置为 1。请注意，`action_mask` 观察值只在智能体即将采取行动时才会有非零值。
* 在包含非法动作的环境中，执行非法动作的智能体会获得与失败相同的奖励，其他智能体获得 0 奖励，然后游戏结束。

许多经典环境都基于 [RLCard](https://github.com/datamllab/rlcard)。如果您在研究中使用这些库，请引用它们：

```
@article{zha2019rlcard,
  title={RLCard: A Toolkit for Reinforcement Learning in Card Games},
  author={Zha, Daochen and Lai, Kwei-Herng and Cao, Yuanpu and Huang, Songyi and Wei, Ruzhe and Guo, Junyu and Hu, Xia},
  journal={arXiv preprint arXiv:1910.04376},
  year={2019}
}
```
