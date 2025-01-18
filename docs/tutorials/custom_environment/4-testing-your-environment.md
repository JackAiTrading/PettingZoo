---
title: "教程：测试您的环境"
---

# 教程：测试您的环境

## 简介

现在我们的环境已经完成，我们可以测试它以确保它按预期工作。PettingZoo 有一个内置的测试套件，可以用来测试您的环境。

## 代码

注意：这段代码可以添加到同一个文件的底部，而不需要使用任何导入，但最佳实践是将测试保存在一个单独的文件中，并使用模块化导入，如下所示。

为了简单起见，这里使用相对导入，并假设您的自定义环境在同一目录中。如果您的测试在其他位置（例如，根级别的 `/test/` 目录），建议使用绝对路径导入。

```{eval-rst}
.. literalinclude:: ../../../tutorials/CustomEnvironment/tutorial4_testing_the_environment.py
   :language: python
   :caption: /custom-environment/env/custom_environment.py
```
