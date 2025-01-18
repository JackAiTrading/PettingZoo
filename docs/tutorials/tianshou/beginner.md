---
title: "天授（Tianshou）：基础 API 使用"
---

# 天授（Tianshou）：基础 API 使用

本教程是一个简单的示例，展示如何在 PettingZoo 环境中使用[天授（Tianshou）](https://github.com/thu-ml/tianshou)。

它演示了在[石头剪刀布](/environments/classic/rps/)环境中，两个使用[随机策略](https://tianshou.readthedocs.io/en/master/_modules/tianshou/policy/random.html)的智能体之间的对局。

## 环境设置
要学习本教程，你需要安装下面显示的依赖项。建议使用新创建的虚拟环境以避免依赖冲突。
```{eval-rst}
.. literalinclude:: ../../../tutorials/Tianshou/requirements.txt
   :language: text
```

## 代码
以下代码应该可以直接运行。注释旨在帮助你理解如何在天授（Tianshou）中使用 PettingZoo。如果你有任何问题，请随时在 [Discord 服务器](https://discord.gg/nhvKkYa6qX)中询问。
```{eval-rst}
.. literalinclude:: ../../../tutorials/Tianshou/1_basic_api_usage.py
   :language: python
```
