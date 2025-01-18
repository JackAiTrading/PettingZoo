---
title: "教程：仓库结构"
---

# 教程：仓库结构

## 简介

欢迎来到四个简短教程中的第一个，这些教程将指导您完成从构思到部署创建自己的 PettingZoo 环境的过程。

我们将创建一个并行环境，这意味着每个智能体同时行动。

在考虑环境逻辑之前，我们应该先了解环境仓库的结构。

## 目录结构
环境仓库通常使用以下结构布局：

    Custom-Environment
    ├── custom-environment
        └── env
            └── custom_environment.py
        └── custom_environment_v0.py
    ├── README.md
    └── requirements.txt

- `/custom-environment/env` 是存储您的环境的地方，以及任何辅助函数（在复杂环境的情况下）。
- `/custom-environment/custom_environment_v0.py` 是导入环境的文件 - 我们使用文件名进行环境版本控制。
- `/README.md` 是用于描述您的环境的文件。
- `/requirements.txt` 是用于跟踪您的环境依赖项的文件。至少应该包含 `pettingzoo`。**请使用 `==` 对所有依赖项进行版本控制**。

### 进阶：额外的（可选）文件
上述文件结构是最小的。一个更适合部署的环境会包括：
- `/docs/` 用于文档，
- `/setup.py` 用于打包，
- `/custom-environment/__init__.py` 用于处理弃用，以及
- Github actions 用于环境测试的持续集成。

实现这些超出了本教程的范围。

## 骨架代码
您的所有环境逻辑都存储在 `/custom-environment/env` 中

```{eval-rst}
.. literalinclude:: ../../../tutorials/CustomEnvironment/tutorial1_skeleton_creation.py
   :language: python
   :caption: /custom-environment/env/custom_environment.py
```
