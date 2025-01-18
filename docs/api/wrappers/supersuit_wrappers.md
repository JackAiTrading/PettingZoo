---
title: SuperSuit 包装器
---

# SuperSuit 包装器

[SuperSuit](https://github.com/Farama-Foundation/SuperSuit) 配套包（`pip install supersuit`）包含一系列可以应用于 [AEC](/api/aec/) 和 [并行](/api/parallel/) 环境的预处理函数。

将 [space invaders](/environments/atari/space_invaders/) 转换为灰度观察空间并堆叠最后 4 帧：

``` python
from pettingzoo.atari import space_invaders_v2
from supersuit import color_reduction_v0, frame_stack_v1

env = space_invaders_v2.env()

env = frame_stack_v1(color_reduction_v0(env, 'full'), 4)
```

## 包含的函数

SuperSuit 包含以下包装器：

```{eval-rst}
.. py:function:: clip_reward_v0(env, lower_bound=-1, upper_bound=1)

  将奖励裁剪到 lower_bound 和 upper_bound 之间。这是处理具有显著幅度差异的奖励的常用方法，特别是在 Atari 环境中。

.. py:function:: clip_actions_v0(env)

  将 Box 动作裁剪到动作空间的高低边界之内。这是应用于具有连续动作空间的环境的标准转换，以保持传递给环境的动作在指定边界内。

.. py:function:: color_reduction_v0(env, mode='full')

  简化图形环境（形状为 (x,y,3)）中的颜色信息。`mode='full'` 完全将观察灰度化。这可能会消耗大量计算资源。参数 'R'、'G' 或 'B' 仅从观察中获取相应的 R、G 或 B 颜色通道。这种方式更快，通常也足够使用。

.. py:function:: dtype_v0(env, dtype)

  将观察重新转换为特定的 dtype。许多图形游戏返回 `uint8` 观察，而神经网络通常需要 `float16` 或 `float32`。`dtype` 可以是 NumPy 接受的任何 dtype 参数（例如 np.dtype 类或字符串）。

.. py:function:: flatten_v0(env)

  将观察展平为一维数组。

.. py:function:: frame_skip_v0(env, num_frames)

  通过重复应用旧动作来跳过 `num_frames` 帧。被跳过的观察会被忽略。被跳过的奖励会被累积。与 Gymnasium Atari 的 frameskip 参数类似，`num_frames` 也可以是一个元组 `(min_skip, max_skip)`，表示可能的跳过长度范围，从中随机选择（仅在单智能体环境中）。

.. py:function:: delay_observations_v0(env, delay)

  将观察延迟 `delay` 帧。在执行 `delay` 帧之前，观察全为零。与 frame_skip 一起，这是实现高 FPS 游戏反应时间的首选方式。

.. py:function:: sticky_actions_v0(env, repeat_action_probability)

  为旧动作分配一个"粘性"概率，使其不按请求更新。这是为了防止智能体在高度确定性的游戏（如 Atari）中学习预定义的动作模式。注意，粘性是累积的，所以一个动作连续两回合粘性的概率是 repeat_action_probability^2，以此类推。这是根据 *"Machado et al. (2018)，"重新审视街机学习环境：通用智能体的评估协议和开放问题"* 向 Atari 添加随机性的推荐方式。

.. py:function:: frame_stack_v1(env, num_frames=4)

  堆叠最近的帧。对于通过普通向量（一维数组）观察的向量游戏，输出只是连接成更长的一维数组。二维或三维数组被堆叠成更高的三维数组。在游戏开始时，尚不存在的帧用 0 填充。`num_frames=1` 相当于不使用此函数。

.. py:function:: max_observation_v0(env, memory)

  结果观察变为 `memory` 数量的先前帧的最大值。这对于 Atari 环境很重要，因为由于游戏机和 CRT 电视的特性，许多游戏的元素是间歇性闪烁而不是持续显示的。OpenAI baselines 的 MaxAndSkip Atari 包装器相当于设置 `memory=2` 然后 `frame_skip` 为 4。

.. py:function:: normalize_obs_v0(env, env_min=0, env_max=1)

  根据观察空间中定义的已知最小和最大观察值，将观察线性缩放到 `env_min`（默认 0）到 `env_max`（默认 1）的范围。仅适用于具有 float32 或 float64 dtype 和有限边界的 Box 观察。如果您想要规范化其他类型，可以先应用 dtype 包装器将类型转换为 float32 或 float64。

.. py:function:: reshape_v0(env, shape)

  将观察重塑为给定形状。

.. py:function:: resize_v1(env, x_size, y_size, linear_interp=False)

  使用默认的区域插值来上采样或下采样观察图像。通过设置 `linear_interp=True` 也可以使用线性插值（它更快，更适合上采样）。此包装器仅适用于二维或三维观察，并且仅在观察是图像时有意义。

.. py:function:: nan_noop_v0(env)

  如果某一步的动作是 NaN 值，以下包装器将触发警告并执行无操作动作来代替。无操作动作作为参数在 `step(action, no_op_action)` 函数中接受。

.. py:function:: nan_zeros_v0(env)

  如果某一步的动作是 NaN 值，以下包装器将触发警告并执行零动作来代替。

.. py:function:: nan_random_v0(env)

  如果某一步的动作是 NaN 值，以下包装器将触发警告并执行随机动作来代替。随机动作将从动作掩码中获取。

.. py:function:: scale_actions_v0(env, scale)

  通过 __init__() 中的 `scale` 参数缩放动作空间的高低边界。此外，在调用 step() 时按相同的值缩放任何动作。

```

## 仅包含的多智能体函数

```{eval-rst}
.. py:function:: agent_indicator_v0(env, type_only=False)

  将智能体 ID 的指示器添加到观察中，仅支持离散和一维、二维和三维 box。对于一维空间，智能体 ID 被转换为一个独热向量并附加到观察中（根据需要增加观察空间的大小）。二维和三维空间被视为图像（通道在最后），ID 被转换为 *n* 个额外通道，代表 ID 的通道全为 1，其他通道全为 0（一种独热编码）。这允许像参数共享这样的 MADRL 方法为异构智能体学习策略，因为策略可以知道它在对哪个智能体采取行动。设置 `type_only` 参数将智能体名称解析为 `<type>_<n>`，并使附加的独热向量仅标识类型，而不是特定的智能体名称。这对于环境中有许多智能体但只有少数类型的智能体的游戏很有用。MADRL 的智能体指示首次在 *使用深度强化学习的合作多智能体控制* 中引入。

.. py:function:: black_death_v2(env)

  不是移除死亡的动作，而是将观察和奖励设为 0 并忽略动作。这可以简化处理智能体死亡机制。"black death"这个名字不是来自瘟疫，而是因为当你死亡时你会看到一个黑色图像（一个填充了零的图像）。

.. py:function:: pad_action_space_v0(env)

  根据 *参数共享在深度强化学习中出人意料地有用* 中提出的算法，将所有智能体的动作空间填充到与最大的相同。这使得需要所有智能体具有同质动作空间的 MARL 方法能够在具有异质动作空间的环境中工作。填充区域内的离散动作将被设置为零，Box 动作将被裁剪到原始空间。

.. py:function:: pad_observations_v0(env)

  根据 *参数共享在深度强化学习中出人意料地有用* 中提出的算法，用 0 将观察填充到任何智能体的最大观察形状。这使得需要所有智能体具有同质观察的 MARL 方法能够在具有异质观察的环境中工作。目前支持 Discrete 和 Box 观察空间。
```

## 引用

如果您在研究中使用了这个项目，请引用：

```
@article{SuperSuit,
  Title = {SuperSuit: Simple Microwrappers for Reinforcement Learning Environments},
  Author = {Terry, J K and Black, Benjamin and Hari, Ananth},
  journal={arXiv preprint arXiv:2008.08932},
  year={2020}
}
