---
title: 蝴蝶环境
firstpage:
---

# 蝴蝶环境

```{toctree}
:hidden:
butterfly/cooperative_pong
butterfly/knights_archers_zombies
butterfly/pistonball
```

```{raw} html
    :file: butterfly/list.html
```

蝴蝶环境是由 Farama 创建的具有挑战性的场景，使用 Pygame 实现，具有类似 Atari 的视觉空间。

所有环境都需要高度的协调，并且需要学习涌现行为才能达到最优策略。因此，这些环境目前在学习上具有很大的挑战性。

环境可以通过其各自文档中指定的参数进行高度配置：
[协作乒乓球](/environments/butterfly/cooperative_pong/)、
[骑士射手大战僵尸](/environments/butterfly/knights_archers_zombies/)、
[活塞球](/environments/butterfly/pistonball/)。

### 安装
这组环境的特定依赖项可以通过以下命令安装：

````bash
pip install 'pettingzoo[butterfly]'
````

### 使用方法

要启动一个带有随机智能体的[活塞球](/environments/butterfly/pistonball/)环境：
```python
from pettingzoo.butterfly import pistonball_v6

env = pistonball_v6.parallel_env(render_mode="human")
observations, infos = env.reset()

while env.agents:
    # 这里是您插入策略的地方
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}

    observations, rewards, terminations, truncations, infos = env.step(actions)
env.close()
```

要启动一个带有交互式用户输入的[骑士射手大战僵尸](/environments/butterfly/knights_archers_zombies/)环境（参见 [manual_policy.py](https://github.com/Farama-Foundation/PettingZoo/blob/master/pettingzoo/butterfly/knights_archers_zombies/manual_policy.py)）：
```python
import pygame
from pettingzoo.butterfly import knights_archers_zombies_v10

env = knights_archers_zombies_v10.env(render_mode="human")
env.reset(seed=42)

manual_policy = knights_archers_zombies_v10.ManualPolicy(env)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    elif agent == manual_policy.agent:
        # 获取用户输入（控制键为 WASD 和空格）
        action = manual_policy(observation, agent)
    else:
        # 这里是您插入策略的地方（用于非玩家智能体）
        action = env.action_space(agent).sample()

    env.step(action)
env.close()
```
