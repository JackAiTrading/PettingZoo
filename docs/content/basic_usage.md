---
title: API
---
# 基本用法

## 安装

安装 PettingZoo 基础库：`pip install pettingzoo`。

这不包括所有环境族的依赖项（某些环境在特定系统上可能难以安装）。

要安装某个环境族的依赖项，使用 `pip install 'pettingzoo[atari]'`，或使用 `pip install 'pettingzoo[all]'` 安装所有依赖项。

我们支持 Linux 和 macOS 上的 Python 3.8、3.9、3.10 和 3.11。我们会接受与 Windows 相关的 PR，但不官方支持它。

## 初始化环境

在 PettingZoo 中使用环境与在 Gymnasium 中使用非常相似。你可以通过以下方式初始化环境：

``` python
from pettingzoo.butterfly import pistonball_v6
env = pistonball_v6.env()
```

环境通常可以在创建时通过参数进行高度配置，例如：

``` python
from pettingzoo.butterfly import cooperative_pong_v5

cooperative_pong_v5.env(ball_speed=18, left_paddle_speed=25,
right_paddle_speed=25, cake_paddle=True, max_cycles=900, bounce_randomness=False)
```

## 与环境交互

与环境的交互接口类似于 Gymnasium：

``` python
from pettingzoo.butterfly import cooperative_pong_v5

env = cooperative_pong_v5.env(render_mode="human")
env.reset(seed=42)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        # 这里是你插入策略的地方
        action = env.action_space(agent).sample()

    env.step(action)
env.close()
```

常用的方法有：

`agent_iter(max_iter=2**63)` 返回一个迭代器，产生环境中当前的智能体。当环境中的所有智能体都完成或执行了 `max_iter` 步骤时终止。

`last(observe=True)` 返回当前可以行动的智能体的观察、奖励、完成状态和信息。返回的奖励是该智能体自上次行动以来累积的奖励。如果 `observe` 设置为 False，则不会计算观察值，而是返回 None。注意，单个智能体完成并不意味着环境完成。

`reset()` 重置环境并在首次调用时设置它以供使用。在使用任何其他方法之前必须调用此方法。

`step(action)` 接收并执行智能体在环境中的动作，自动将控制权切换到下一个智能体。

## 额外的环境 API

PettingZoo 将游戏建模为*智能体环境循环*（AEC）游戏，因此可以支持多智能体强化学习能考虑的任何游戏，允许出现奇妙的情况。因此，我们的 API 包含了你可能不需要但在需要时非常重要的低级函数和属性。它们的功能用于实现上述高级函数，所以包含它们只是代码组织的问题。

`agents`：所有当前智能体的名称列表，通常是整数。这些可能会随着环境的进展而改变（即可以添加或删除智能体）。

`num_agents`：agents 列表的长度。

`agent_selection` 环境的一个属性，对应于当前可以执行动作的选定智能体。

`observation_space(agent)` 一个函数，用于获取特定智能体的观察空间。对于特定的智能体 ID，这个空间永远不应该改变。

`action_space(agent)` 一个函数，用于获取特定智能体的动作空间。对于特定的智能体 ID，这个空间永远不应该改变。

`terminations`：一个字典，包含调用时每个当前智能体的终止状态，以名称为键。`last()` 访问此属性。注意，可以从此字典中添加或删除智能体。返回的字典如下所示：

`terminations = {0:[第一个智能体的终止状态], 1:[第二个智能体的终止状态] ... n-1:[第n个智能体的终止状态]}`

`truncations`：一个字典，包含调用时每个当前智能体的截断状态，以名称为键。`last()` 访问此属性。注意，可以从此字典中添加或删除智能体。返回的字典如下所示：

`truncations = {0:[第一个智能体的截断状态], 1:[第二个智能体的截断状态] ... n-1:[第n个智能体的截断状态]}`

`infos`：一个字典，包含每个当前智能体的信息，以名称为键。每个智能体的信息也是一个字典。注意，可以从此属性中添加或删除智能体。`last()` 访问此属性。返回的字典如下所示：

`infos = {0:[第一个智能体的信息], 1:[第二个智能体的信息] ... n-1:[第n个智能体的信息]}`

`observe(agent)`：返回智能体当前可以做出的观察。`last()` 调用此函数。

`rewards`：一个字典，包含调用时每个当前智能体的奖励，以名称为键。奖励是上一步后生成的即时奖励。注意，可以从此属性中添加或删除智能体。`last()` 不直接访问此属性，而是将返回的奖励存储在内部变量中。奖励结构如下所示：

`{0:[第一个智能体的奖励], 1:[第二个智能体的奖励] ... n-1:[第n个智能体的奖励]}`

`seed(seed=None)`：重新设置环境的随机种子。必须在 `seed()` 之后调用 `reset()`，并在 `step()` 之前调用。

`render()`：使用初始化时指定的渲染模式返回环境中的渲染帧。在渲染模式为 `'rgb_array'` 的情况下，返回一个 numpy 数组，而在 `'ansi'` 模式下返回打印的字符串。在 `human` 模式下不需要调用 `render()`。

`close()`：关闭渲染窗口。

### 可选的 API 组件

虽然基础 API 不要求，但大多数下游包装器和工具依赖于以下属性和方法，除非在特殊情况下无法添加一个或多个，否则应该将它们添加到新环境中。

`possible_agents`：环境可能生成的所有可能智能体的列表。等同于观察和动作空间中的智能体列表。这不能通过游戏或重置来改变。

`max_num_agents`：possible_agents 列表的长度。

`observation_spaces`：每个智能体的观察空间字典，以名称为键。这不能通过游戏或重置来改变。

`action_spaces`：每个智能体的动作空间字典，以名称为键。这不能通过游戏或重置来改变。

`state()`：返回环境当前状态的全局观察。不是所有环境都支持此功能。

`state_space`：环境的全局观察空间。不是所有环境都支持此功能。

## 重要用法

### 检查整个环境是否完成

当智能体终止或截断时，它会从 `agents` 中移除，所以当环境完成时 `agents` 将是一个空列表。这意味着 `not env.agents` 是环境完成的简单条件。

### 解包环境

如果你有一个包装过的环境，想要获取所有包装层下面的未包装环境（以便手动调用函数或更改环境的某些基本方面），你可以使用 `.unwrapped` 属性。如果环境已经是基础环境，`.unwrapped` 属性将只返回它自己。

``` python
from pettingzoo.butterfly import knights_archers_zombies_v10

base_env = knights_archers_zombies_v10.env().unwrapped
```

### 可变数量的智能体（死亡）

智能体可以在环境过程中死亡和生成。如果智能体死亡，则其在 `terminated` 字典中的条目被设置为 `True`，它成为下一个选定的智能体（或在另一个也终止或截断的智能体之后），并且它采取的动作必须是 `None`。在采取这个空动作之后，智能体将从 `agents` 和其他可变属性中移除。智能体生成可以通过将其附加到 `agents` 和其他可变属性（它已经在可能的智能体和动作/观察空间中），并在某个时点用 agent_iter 转换到它来完成。

### 环境作为智能体

在某些情况下，将智能体动作与环境动作分开对研究有帮助。这可以通过将环境视为智能体来实现。我们建议在 env.agents 中将环境参与者称为 `env`，并让它采取 `None` 作为动作。

## 原始环境

环境默认包装在一些轻量级包装器中，这些包装器处理错误消息并确保在不正确使用（即执行非法移动或在重置前进行步骤）时的合理行为。但是，这些会增加很小的开销。如果你想创建一个没有这些包装器的环境，可以使用每个模块中包含的 `raw_env()` 构造函数：

``` python
environment_parameters = {}  # 要传递给环境的任何参数
env = knights_archers_zombies_v10.raw_env(**environment_parameters)
```
