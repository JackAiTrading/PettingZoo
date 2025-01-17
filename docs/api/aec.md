---
title: AEC
---

# AEC API

默认情况下，PettingZoo 将游戏建模为[*智能体环境循环*](https://arxiv.org/abs/2009.13051)（AEC）环境。这使得 PettingZoo 可以表示多智能体强化学习能考虑的任何类型的游戏。

更多信息，请参见[关于 AEC](#about-aec)或[*PettingZoo：多智能体强化学习的标准 API*](https://arxiv.org/pdf/2009.14471.pdf)。

[PettingZoo 包装器](/api/wrappers/pz_wrappers/)可用于在并行和 AEC 环境之间转换，但有一些限制（例如，AEC 环境必须只在每个循环结束时更新一次）。

## 示例
[PettingZoo Classic](/environments/classic/)为回合制游戏提供了标准的 AEC 环境示例，其中许多实现了[非法动作掩码](#action-masking)。

我们提供了一个[教程](/content/environment_creation/)来创建一个简单的石头剪刀布 AEC 环境，展示了如何使用 AEC 环境来表示同时动作的游戏。

## 用法

可以按以下方式与 AEC 环境交互：

```python
from pettingzoo.classic import rps_v2

env = rps_v2.env(render_mode="human")
env.reset(seed=42)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        action = env.action_space(agent).sample() # 这里是你插入策略的地方

    env.step(action)
env.close()
```

### 动作掩码
AEC 环境通常包含动作掩码，用于标记智能体的有效/无效动作。

使用动作掩码采样动作：
```python
from pettingzoo.classic import chess_v6

env = chess_v6.env(render_mode="human")
env.reset(seed=42)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        # 无效动作掩码是可选的，取决于环境
        if "action_mask" in info:
            mask = info["action_mask"]
        elif isinstance(observation, dict) and "action_mask" in observation:
            mask = observation["action_mask"]
        else:
            mask = None
        action = env.action_space(agent).sample(mask) # 这里是你插入策略的地方

    env.step(action)
env.close()
```

注意：动作掩码是可选的，可以使用 `observation` 或 `info` 来实现。

* [PettingZoo Classic](/environments/classic/) 环境在 `observation` 字典中存储动作掩码：
  * `mask = observation["action_mask"]`
* [Shimmy](https://shimmy.farama.org/) 的 [OpenSpiel 环境](https://shimmy.farama.org/environments/open_spiel/)在 `info` 字典中存储动作掩码：
  * `mask = info["action_mask"]`

要在自定义环境中实现动作掩码，请参见[自定义环境：动作掩码](/tutorials/custom_environment/3-action-masking/)

有关动作掩码的更多信息，请参见[策略梯度算法中无效动作掩码的深入研究](https://arxiv.org/abs/2006.14171)（Huang，2022）

## 关于 AEC
[*智能体环境循环*](https://arxiv.org/abs/2009.13051)（AEC）模型被设计为 MARL 的类 [Gym](https://github.com/openai/gym) API，支持所有可能的用例和环境类型。这包括具有以下特征的环境：
- 大量智能体（参见 [Magent2](https://magent2.farama.org/)）
- 可变数量的智能体（参见[骑士、弓箭手、僵尸](/environments/butterfly/knights_archers_zombies)）
- 任何类型的动作和观察空间（例如，[Box](https://gymnasium.farama.org/api/spaces/fundamental/#gymnasium.spaces.Box)、[Discrete](https://gymnasium.farama.org/api/spaces/fundamental/#gymnasium.spaces.Discrete)、[MultiDiscrete](https://gymnasium.farama.org/api/spaces/fundamental/#gymnasium.spaces.MultiDiscrete)、[MultiBinary](https://gymnasium.farama.org/api/spaces/fundamental/#multibinary)、[Text](https://gymnasium.farama.org/api/spaces/fundamental/#gymnasium.spaces.Text)）
- 嵌套的动作和观察空间（例如，[Dict](https://gymnasium.farama.org/api/spaces/composite/#dict)、[Tuple](https://gymnasium.farama.org/api/spaces/composite/#tuple)、[Sequence](https://gymnasium.farama.org/api/spaces/composite/#sequence)、[Graph](https://gymnasium.farama.org/api/spaces/composite/#graph)）
- 支持动作掩码（参见 [Classic](/environments/classic) 环境）
- 可以随时间变化并因智能体而异的动作和观察空间（参见 [generated_agents](https://github.com/Farama-Foundation/PettingZoo/blob/master/pettingzoo/test/example_envs/generated_agents_env_v0.py) 和 [variable_env_test](https://github.com/Farama-Foundation/PettingZoo/blob/master/test/variable_env_test.py)）
- 变化的回合顺序和不断发展的环境动态（例如，具有多个阶段的游戏，反转回合）

在 AEC 环境中，智能体按顺序行动，在采取行动之前接收更新的观察和奖励。环境在每个智能体的步骤之后更新，这是表示顺序游戏（如国际象棋）的自然方式。AEC 模型足够灵活，可以处理多智能体强化学习可以考虑的任何类型的游戏。

环境在每个智能体的步骤之后更新。智能体在其回合开始时接收更新的观察和奖励。环境在每一步之后都会更新，
这是表示顺序游戏（如国际象棋和围棋）的自然方式。

```{figure} /_static/img/aec_cycle_figure.png
    :width: 480px
    :name: 国际象棋的 AEC 图
```

这与我们的[并行 API](/api/parallel/)中表示的[*部分可观察随机游戏*](https://en.wikipedia.org/wiki/Game_theory#Stochastic_outcomes_(and_relation_to_other_fields))（POSG）模型形成对比，在 POSG 中，智能体同时行动，只能在循环结束时接收观察和奖励。
这使得表示顺序游戏变得困难，并导致竞争条件——智能体选择互斥的动作。这导致环境行为取决于智能体顺序的内部解析，如果即使一个竞争条件没有被环境捕获和处理（例如，通过打破平局），就会导致难以检测的错误。

AEC 模型类似于 DeepMind 的 [OpenSpiel](https://github.com/deepmind/open_spiel) 中使用的[*扩展式游戏*](https://en.wikipedia.org/wiki/Extensive-form_game)（EFG）模型。
EFG 将顺序游戏表示为树，明确地将每个可能的动作序列表示为从根到叶的路径。
EFG 的一个限制是其形式定义特定于博弈论，只允许在游戏结束时给出奖励，而在强化学习中，学习通常需要频繁的奖励。

EFG 可以通过添加表示环境的玩家（例如，OpenSpiel 中的[机会节点](https://openspiel.readthedocs.io/en/latest/concepts.html#the-tree-representation)）来扩展以表示随机游戏，该玩家根据给定的概率分布采取行动。然而，这要求用户在与环境交互时手动采样和应用机会节点动作，留下用户错误和潜在的随机种子问题的空间。

相比之下，AEC 环境在每个智能体步骤之后内部处理环境动态，从而产生更简单的环境心智模型，并允许任意和不断发展的环境动态（而不是静态的机会分布）。AEC 模型也更接近于计算机游戏在代码中的实现方式，可以被认为类似于游戏编程中的游戏循环。

有关 AEC 模型和 PettingZoo 设计理念的更多信息，请参见[*PettingZoo：多智能体强化学习的标准 API*](https://arxiv.org/pdf/2009.14471.pdf)。

## AECEnv

```{eval-rst}
.. currentmodule:: pettingzoo.utils.env

.. autoclass:: AECEnv

```

## 属性

```{eval-rst}

.. autoattribute:: AECEnv.agents

    所有当前智能体的名称列表，通常是整数。这些可能会随着环境的进展而改变（即可以添加或删除智能体）。

    :type: List[AgentID]

.. autoattribute:: AECEnv.num_agents

    agents 列表的长度。

.. autoattribute:: AECEnv.possible_agents

    环境可能生成的所有可能智能体的列表。等同于观察和动作空间中的智能体列表。这不能通过游戏或重置来改变。

    :type: List[AgentID]

.. autoattribute:: AECEnv.max_num_agents

    possible_agents 列表的长度。

.. autoattribute:: AECEnv.agent_selection

    环境的一个属性，对应于当前可以执行动作的选定智能体。

    :type: AgentID

.. autoattribute:: AECEnv.terminations

.. autoattribute:: AECEnv.truncations

.. autoattribute:: AECEnv.rewards

    一个字典，包含调用时每个当前智能体的奖励，以名称为键。奖励是上一步后生成的即时奖励。注意，可以从此属性中添加或删除智能体。`last()` 不直接访问此属性，而是将返回的奖励存储在内部变量中。奖励结构如下所示：

    {0:[第一个智能体的奖励], 1:[第二个智能体的奖励] ... n-1:[第n个智能体的奖励]}

    :type: Dict[AgentID, float]

.. autoattribute:: AECEnv.infos

    一个字典，包含每个当前智能体的信息，以名称为键。每个智能体的信息也是一个字典。注意，可以从此属性中添加或删除智能体。`last()` 访问此属性。返回的字典如下所示：

        infos = {0:[第一个智能体的信息], 1:[第二个智能体的信息] ... n-1:[第n个智能体的信息]}

    :type: Dict[AgentID, Dict[str, Any]]

.. autoattribute:: AECEnv.observation_spaces

    每个智能体的观察空间字典，以名称为键。这不能通过游戏或重置来改变。

    :type: Dict[AgentID, gymnasium.spaces.Space]

.. autoattribute:: AECEnv.action_spaces

    每个智能体的动作空间字典，以名称为键。这不能通过游戏或重置来改变。

    :type: Dict[AgentID, gymnasium.spaces.Space]
```

## 方法

```{eval-rst}
.. automethod:: AECEnv.step
.. automethod:: AECEnv.reset
.. automethod:: AECEnv.observe
.. automethod:: AECEnv.render
.. automethod:: AECEnv.close

```
