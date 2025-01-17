[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/) [![代码风格: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<p align="center">
    <img src="https://raw.githubusercontent.com/Farama-Foundation/PettingZoo/master/pettingzoo-text.png" width="500px"/>
</p>

PettingZoo 是一个用于进行多智能体强化学习研究的 Python 库，类似于 [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) 的多智能体版本。

文档网站位于 [pettingzoo.farama.org](https://pettingzoo.farama.org)，我们还有一个公共 Discord 服务器（我们也用来协调开发工作），您可以在这里加入：https://discord.gg/nhvKkYa6qX

## 环境

PettingZoo 包含以下环境系列：

* [Atari](https://pettingzoo.farama.org/environments/atari/): 多玩家 Atari 2600 游戏（合作、竞争和混合）
* [Butterfly](https://pettingzoo.farama.org/environments/butterfly): 我们开发的合作图形游戏，需要高度协调
* [Classic](https://pettingzoo.farama.org/environments/classic): 经典游戏，包括纸牌游戏、棋盘游戏等
* [MPE](https://pettingzoo.farama.org/environments/mpe): 一组简单的非图形通信任务，最初来自 https://github.com/openai/multiagent-particle-envs
* [SISL](https://pettingzoo.farama.org/environments/sisl): 3个合作环境，最初来自 https://github.com/sisl/MADRL

## 安装

安装基础 PettingZoo 库：`pip install pettingzoo`

这不包括所有环境系列的依赖项（某些环境在某些系统上可能会出现安装问题）。

要安装某个系列的依赖项，请使用 `pip install 'pettingzoo[atari]'`，或使用 `pip install 'pettingzoo[all]'` 安装所有依赖项。

我们支持在 Linux 和 macOS 上使用 Python 3.8、3.9、3.10 和 3.11。我们会接受与 Windows 相关的 PR，但不正式支持它。

注意：某些 Linux 发行版可能需要手动安装 `cmake`、`swig` 或 `zlib1g-dev`（例如，`sudo apt install cmake swig zlib1g-dev`）

## 入门

有关 PettingZoo 的介绍，请参见[基本用法](https://pettingzoo.farama.org/content/basic_usage/)。要创建新环境，请参见我们的[环境创建教程](https://pettingzoo.farama.org/tutorials/custom_environment/1-project-structure/)和[自定义环境示例](https://pettingzoo.farama.org/content/environment_creation/)。

有关使用 PettingZoo 训练 RL 模型的示例，请参见我们的教程：
* [CleanRL：实现 PPO](https://pettingzoo.farama.org/tutorials/cleanrl/implementing_PPO/)：在 [Pistonball](https://pettingzoo.farama.org/environments/butterfly/pistonball/) 环境中训练多个 PPO 智能体。
* [Tianshou：训练智能体](https://pettingzoo.farama.org/tutorials/tianshou/intermediate/)：在[井字棋](https://pettingzoo.farama.org/environments/classic/tictactoe/)环境中训练 DQN 智能体。
* [AgileRL：训练、课程和自我对弈](https://pettingzoo.farama.org/main/tutorials/agilerl/DQN/)：在[四子连珠](https://pettingzoo.farama.org/environments/classic/connect_four/)环境中使用课程学习和自我对弈训练智能体。

## API

PettingZoo 将环境建模为 [*智能体环境循环* (AEC) 游戏](https://arxiv.org/pdf/2009.14471.pdf)，以便能够在一个 API 下清晰地支持所有类型的多智能体 RL 环境，并最小化某些常见错误类型的可能性。

在 PettingZoo 中使用环境与 Gymnasium 非常相似，即通过以下方式初始化环境：

```python
from pettingzoo.butterfly import pistonball_v6
env = pistonball_v6.env()
```

可以以与 Gymnasium 非常相似的方式与环境交互：

```python
env.reset()
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    action = None if termination or truncation else env.action_space(agent).sample()  # 这里是您插入策略的地方
    env.step(action)
```

完整的 API 文档请参见 https://pettingzoo.farama.org/api/aec/

### 并行 API

在某些环境中，假设智能体同时采取行动是有效的。对于这些游戏，我们提供了一个次要 API 来允许并行操作，文档位于 https://pettingzoo.farama.org/api/parallel/

## SuperSuit

SuperSuit 是一个库，包含了 PettingZoo 和 Gymnasium 环境中常用的所有包装器（帧堆叠、观察归一化等），具有良好的 API。我们开发它是为了替代内置于 PettingZoo 中的包装器。https://github.com/Farama-Foundation/SuperSuit

## 环境版本控制

PettingZoo 为了可重现性原因保持严格的版本控制。所有环境都以"_v"结尾。当对环境进行更改时，可能会影响学习结果，因此会将版本号增加一，以避免潜在的混淆。

## 引用

要引用本项目，请使用以下格式：

```
@article{terry2021pettingzoo,
  title={Pettingzoo: Gym for multi-agent reinforcement learning},
  author={Terry, J and Black, Benjamin and Grammel, Nathaniel and Jayakumar, Mario and Hari, Ananth and Sullivan, Ryan and Santos, Luis S and Dieffendahl, Clemens and Horsch, Caroline and Perez-Vicente, Rodrigo and others},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={15032--15043},
  year={2021}
}
```

## 项目维护者

* 项目经理：[David Gerard](https://github.com/David-GERARD) - `david.gerard.23@ucl.ac.uk`。
* 项目维护还由更广泛的 Farama 团队贡献：[farama.org/team](https://farama.org/team)。
