---
title: "LangChain：创建 LLM 智能体"
---

# LangChain：创建 LLM 智能体

本教程将演示如何使用 LangChain 创建可以与 PettingZoo 环境交互的 LLM 智能体。

本教程改编自 LangChain 的文档：[模拟环境：PettingZoo](https://python.langchain.com/en/latest/use_cases/agent_simulations/petting_zoo.html)。

> 对于 LLM 智能体的许多应用来说，环境是真实的（互联网、数据库、REPL 等）。但是，我们也可以定义智能体在模拟环境中进行交互，比如基于文本的游戏。这是一个如何使用 PettingZoo 创建简单的智能体-环境交互循环的示例。


## 环境设置
要学习本教程，您需要安装下面显示的依赖项。建议使用新创建的虚拟环境以避免依赖冲突。
```{eval-rst}
.. literalinclude:: ../../../tutorials/LangChain/requirements.txt
   :language: text
```

## 环境循环
```{eval-rst}
.. literalinclude:: ../../../tutorials/LangChain/langchain_example.py
   :pyobject: main
   :language: python
```


## Gymnasium 智能体
这里我们重现了来自 [LangChain Gymnasium 示例](https://python.langchain.com/en/latest/use_cases/agent_simulations/gymnasium.html) 的相同 `GymnasiumAgent`。如果多次重试后仍未采取有效行动，它会简单地采取随机行动。
```{eval-rst}
.. literalinclude:: ../../../tutorials/LangChain/gymnasium_agent.py
   :language: python
```

## PettingZoo 智能体
`PettingZooAgent` 将 `GymnasiumAgent` 扩展到多智能体设置。主要区别是：
- `PettingZooAgent` 接受一个 `name` 参数来在多个智能体中识别它
- `get_docs` 函数的实现方式不同，因为 `PettingZoo` 仓库结构与 `Gymnasium` 仓库结构不同

```{eval-rst}
.. literalinclude:: ../../../tutorials/LangChain/pettingzoo_agent.py
   :language: python
```

### 石头剪刀布
我们现在可以使用 `PettingZooAgent` 运行一个多智能体石头剪刀布游戏的模拟。

```{eval-rst}
.. literalinclude:: ../../../tutorials/LangChain/langchain_example.py
   :pyobject: rock_paper_scissors
   :language: python
```

```text
观察：3
奖励：0
终止：False
截断：False
返回：0

动作：1

观察：3
奖励：0
终止：False
截断：False
返回：0

动作：1

观察：1
奖励：0
终止：False
截断：False
返回：0

动作：2

观察：1
奖励：0
终止：False
截断：False
返回：0

动作：1

观察：1
奖励：1
终止：False
截断：False
返回：1

动作：0

观察：2
奖励：-1
终止：False
截断：False
返回：-1

动作：0

观察：0
奖励：0
终止：False
截断：True
返回：1

动作：None

观察：0
奖励：0
终止：False
截断：True
返回：-1

动作：None
```


## 动作掩码智能体
一些 `PettingZoo` 环境提供 `action_mask` 来告诉智能体哪些动作是有效的。`ActionMaskAgent` 继承 `PettingZooAgent` 以使用来自 `action_mask` 的信息来选择动作。

```{eval-rst}
.. literalinclude:: ../../../tutorials/LangChain/action_masking_agent.py
   :language: python
```

### 井字棋
这是一个使用 `ActionMaskAgent` 的井字棋游戏示例。
```{eval-rst}
.. literalinclude:: ../../../tutorials/LangChain/langchain_example.py
   :pyobject: tic_tac_toe
   :language: python
```

```text
观察：{'observation': array([[[0, 0],
        [0, 0],
        [0, 0]],

       [[0, 0],
        [0, 0],
        [0, 0]],

       [[0, 0],
        [0, 0],
        [0, 0]]], dtype=int8), 'action_mask': array([1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int8)}
奖励：0
终止：False
截断：False
返回：0

动作：0
     |     |
  X  |  -  |  -
_____|_____|_____
     |     |
  -  |  -  |  -
_____|_____|_____
     |     |
  -  |  -  |  -
     |     |

观察：{'observation': array([[[0, 1],
        [0, 0],
        [0, 0]],

       [[0, 0],
        [0, 0],
        [0, 0]],

       [[0, 0],
        [0, 0],
        [0, 0]]], dtype=int8), 'action_mask': array([0, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int8)}
奖励：0
终止：False
截断：False
返回：0

动作：1
     |     |
  X  |  -  |  -
_____|_____|_____
     |     |
  O  |  -  |  -
_____|_____|_____
     |     |
  -  |  -  |  -
     |     |

观察：{'observation': array([[[1, 0],
        [0, 1],
        [0, 0]],

       [[0, 0],
        [0, 0],
        [0, 0]],

       [[0, 0],
        [0, 0],
        [0, 0]]], dtype=int8), 'action_mask': array([0, 0, 1, 1, 1, 1, 1, 1, 1], dtype=int8)}
```
### 德州扑克无限注
这是一个使用 `ActionMaskAgent` 的德州扑克无限注游戏示例。
```{eval-rst}
.. literalinclude:: ../../../tutorials/LangChain/langchain_example.py
   :pyobject: texas_holdem_no_limit
   :language: python
```

```text
观察：{'observation': array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0.,
       0., 0., 2.], dtype=float32), 'action_mask': array([1, 1, 0, 1, 1], dtype=int8)}
奖励：0
终止：False
截断：False
返回：0

动作：1

观察：{'observation': array([0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
       0., 0., 2.], dtype=float32), 'action_mask': array([1, 1, 0, 1, 1], dtype=int8)}
奖励：0
终止：False
截断：False
返回：0

动作：1

观察：{'observation': array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 1., 2.], dtype=float32), 'action_mask': array([1, 1, 1, 1, 1], dtype=int8)}
奖励：0
终止：False
截断：False
返回：0

动作：1

观察：{'observation': array([0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 2., 2.], dtype=float32), 'action_mask': array([1, 1, 1, 1, 1], dtype=int8)}
奖励：0
终止：False
截断：False
返回：0

动作：0

观察：{'observation': array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1.,
       0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 2., 2.], dtype=float32), 'action_mask': array([1, 1, 1, 1, 1], dtype=int8)}
奖励：0
终止：False
截断：False
返回：0

动作：2

观察：{'observation': array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 0., 0., 0., 0.,
       0., 2., 6.], dtype=float32), 'action_mask': array([1, 1, 1, 1, 1], dtype=int8)}
奖励：0
终止：False
截断：False
返回：0

动作：2

观察：{'observation': array([0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0.,
       0., 2., 8.], dtype=float32), 'action_mask': array([1, 1, 1, 1, 1], dtype=int8)}
奖励：0
终止：False
截断：False
返回：0

动作：3

观察：{'observation': array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,
        1.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        6., 20.], dtype=float32), 'action_mask': array([1, 1, 1, 1, 1], dtype=int8)}
奖励：0
终止：False
截断：False
返回：0

动作：4

观察：{'observation': array([  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
         0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   1.,   0.,   0.,
         0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
         0.,   0.,   0.,   0.,   0.,   0.,   1.,   0.,   0.,   1.,   1.,
         0.,   0.,   1.,   0.,   0.,   0.,   0.,   0.,   8., 100.],
      dtype=float32), 'action_mask': array([1, 1, 0, 0, 0], dtype=int8)}
奖励：0
终止：False
截断：False
返回：0

动作：4
[WARNING]：非法动作，游戏终止，当前玩家输掉。

观察：{'observation': array([  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
         0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   1.,   0.,   0.,
         0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
         0.,   0.,   0.,   0.,   0.,   0.,   1.,   0.,   0.,   1.,   1.,
         0.,   0.,   1.,   0.,   0.,   0.,   0.,   0.,   8., 100.],
      dtype=float32), 'action_mask': array([1, 1, 0, 0, 0], dtype=int8)}
奖励：-1.0
终止：True
截断：True
返回：-1.0

动作：None

观察：{'observation': array([  0.,   0.,   1.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
         0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   1.,   0.,   0.,
         0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
         0.,   0.,   0.,   0.,   0.,   0.,   1.,   0.,   0.,   1.,   0.,
         0.,   0.,   0.,   0.,   1.,   0.,   0.,   0.,  20., 100.],
      dtype=float32), 'action_mask': array([1, 1, 0, 0, 0], dtype=int8)}
奖励：0
终止：True
截断：True
返回：0

动作：None

观察：{'observation': array([  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
         0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   1.,   0.,   0.,
         0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   1.,   0.,   0.,
         1.,   0.,   0.,   0.,   0.,   0.,   1.,   0.,   0.,   1.,   0.,
         0.,   0.,   0.,   0.,   0.,   0.,   0.,   0., 100., 100.],
      dtype=float32), 'action_mask': array([1, 1, 0, 0, 0], dtype=int8)}
奖励：0
终止：True
截断：True
返回：0

动作：None

观察：{'observation': array([  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   1.,   0.,   0.,
         0.,   0.,   0.,   0.,   0.,   0.,   0.,   1.,   1.,   0.,   0.,
         0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
         0.,   0.,   0.,   0.,   0.,   0.,   1.,   0.,   0.,   1.,   0.,
         0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   2., 100.],
      dtype=float32), 'action_mask': array([1, 1, 0, 0, 0], dtype=int8)}
奖励：0
终止：True
截断：True
返回：0

动作：None
```



## 完整代码

以下代码应该可以正常运行。注释旨在帮助您了解如何使用 PettingZoo 与 LangChain。如果您有任何问题，请随时在 [Discord 服务器](https://discord.gg/nhvKkYa6qX) 中提问。

```{eval-rst}

.. literalinclude:: ../../../tutorials/LangChain/langchain_example.py
   :language: python

```
