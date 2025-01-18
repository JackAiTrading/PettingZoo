---
title: MPE 环境
firstpage:
---

# MPE 环境

```{toctree}
:hidden:
mpe/simple
mpe/simple_adversary
mpe/simple_crypto
mpe/simple_push
mpe/simple_reference
mpe/simple_speaker_listener
mpe/simple_spread
mpe/simple_tag
mpe/simple_world_comm
```

```{raw} html
    :file: mpe/list.html
```

多粒子环境（MPE）是一组面向通信的环境，其中粒子智能体可以（有时）移动、通信、相互看见、相互推动，并与固定的地标互动。

这些环境来自 [OpenAI 的 MPE](https://github.com/openai/multiagent-particle-envs) 代码库，进行了一些小的修复，主要是将动作空间默认设置为离散的，使奖励保持一致，并清理了某些环境的观察空间。

### 安装

这组环境的特定依赖项可以通过以下命令安装：

````bash
pip install 'pettingzoo[mpe]'
````

### 使用方法
要启动一个带有随机智能体的[简单标记](/environments/mpe/simple_tag/)环境：

```python
from pettingzoo.mpe import simple_tag_v3
env = simple_tag_v3.env(render_mode='human')

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

### 环境类型

简单对抗（Simple Adversary）、简单加密（Simple Crypto）、简单推动（Simple Push）、简单标记（Simple Tag）和简单世界通信（Simple World Comm）环境是对抗性的（"好"智能体获得奖励意味着"对手"智能体受到惩罚，反之亦然，尽管并不总是完全零和的）。在大多数这些环境中，有绿色渲染的"好"智能体和红色渲染的"对手"团队。

简单参考（Simple Reference）、简单说话者-听众（Simple Speaker Listener）和简单扩散（Simple Spread）环境本质上是合作性的（智能体必须一起工作以实现目标，并根据自己的成功和其他智能体的成功获得混合奖励）。

### 关键概念

* **地标**：地标是环境中无法控制的静态圆形特征。在一些环境中，如 Simple，它们是影响智能体奖励的目标地点，奖励取决于智能体与它们的距离。在其他环境中，它们可以是阻碍智能体移动的障碍物。这些在每个环境的文档中都有详细描述。

* **可见性**：当一个智能体对另一个智能体可见时，后者的观察包含前者的相对位置（在 Simple World Comm 和 Simple Tag 中，还包括前者的速度）。如果一个智能体暂时隐藏（只在 Simple World Comm 中可能），那么该智能体的位置和速度被设置为零。

* **通信**：在某些环境中，某些智能体可以作为其动作的一部分广播消息（详见动作空间），该消息将传输给每个允许看到该消息的智能体。在 Simple Crypto 中，这个消息用于表示 Bob 和 Eve 已重构消息。

* **颜色**：由于所有智能体都被渲染为圆形，智能体只能通过颜色被人类识别，所以在大多数环境中都描述了智能体的颜色。智能体本身不会观察到颜色。

* **距离**：地标和智能体通常在地图上从 -1 到 1 之间均匀随机放置。这意味着它们通常相距 1-2 个单位。在考虑奖励的规模（通常依赖于距离）和包含相对和绝对位置的观察空间时，记住这一点很重要。

### 终止

游戏在执行完由环境参数 `max_cycles` 指定的周期数后终止。所有环境的默认值都是 25 个周期，与原始 OpenAI 源代码相同。

### 观察空间

智能体的观察空间是一个向量，通常由智能体的位置和速度、其他智能体的相对位置和速度、地标的相对位置、地标和智能体的类型，以及从其他智能体接收到的通信组成。具体形式在环境文档中有详细说明。

如果一个智能体无法看到或观察到第二个智能体的通信，那么第二个智能体不会包含在第一个智能体的观察空间中，这导致在某些环境中不同智能体具有不同大小的观察空间。

### 动作空间

注意：[OpenAI 的 MPE](https://github.com/openai/multiagent-particle-envs) 默认使用连续动作空间。

离散动作空间（默认）：

动作空间是一个离散动作空间，表示智能体可以执行的移动和通信组合。可以移动的智能体可以在 4 个基本方向之间选择或者不动。可以通信的智能体可以在 2 到 10 个环境相关的通信选项之间选择，这些选项会向所有能听到的智能体广播消息。

连续动作空间（通过 continuous_actions=True 设置）：

动作空间是一个连续动作空间，表示智能体可以执行的移动和通信。可以移动的智能体可以在四个基本方向的每个方向上输入 0.0 到 1.0 之间的速度，其中相对的速度（例如左和右）会被相加。可以通信的智能体可以在他们有权访问的每个环境通信通道上输出一个连续值。

### 渲染

渲染在一个窗口中显示场景，如果智能体漫游超出其边界，窗口会自动增长。通信在场景底部渲染。`render()` 方法还会返回渲染区域的像素图。

### 引用

MPE 环境最初在以下工作中描述：

```
@article{mordatch2017emergence,
  title={Emergence of Grounded Compositional Language in Multi-Agent Populations},
  author={Mordatch, Igor and Abbeel, Pieter},
  journal={arXiv preprint arXiv:1703.04908},
  year={2017}
}
```

但首次作为以下工作的一部分发布：

```
@article{lowe2017multi,
  title={Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments},
  author={Lowe, Ryan and Wu, Yi and Tamar, Aviv and Harb, Jean and Abbeel, Pieter and Mordatch, Igor},
  journal={Neural Information Processing Systems (NIPS)},
  year={2017}
}
```

如果您在研究中使用这些环境，请引用其中一个或两个。
