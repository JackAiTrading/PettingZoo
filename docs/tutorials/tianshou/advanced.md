---
title: "天授（Tianshou）：命令行界面和日志记录"
---

# 天授（Tianshou）：命令行界面和日志记录

本教程是一个完整的示例，展示如何使用天授（Tianshou）在[井字棋](/environments/classic/tictactoe/)环境中训练一个[深度 Q 网络](https://tianshou.readthedocs.io/en/master/tutorials/dqn.html)（DQN）智能体。

它扩展了[训练智能体](/tutorials/tianshou/intermediate/)中的代码，添加了命令行界面（使用 [argparse](https://docs.python.org/3/library/argparse.html)）和日志记录（使用天授的[日志记录器](https://tianshou.readthedocs.io/en/master/tutorials/logger.html)）功能。

## 环境设置
要学习本教程，你需要安装下面显示的依赖项。建议使用新创建的虚拟环境以避免依赖冲突。
```{eval-rst}
.. literalinclude:: ../../../tutorials/Tianshou/requirements.txt
   :language: text
```

## 代码
以下代码应该可以直接运行。注释旨在帮助你理解如何在天授（Tianshou）中使用 PettingZoo。如果你有任何问题，请随时在 [Discord 服务器](https://discord.gg/nhvKkYa6qX)中询问。

```{eval-rst}
.. literalinclude:: ../../../tutorials/Tianshou/3_cli_and_logging.py
   :language: python
```
