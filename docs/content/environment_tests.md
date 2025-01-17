# 测试环境

PettingZoo 通过多个合规性测试来检验环境。如果你要添加新环境，我们建议你在自己的环境上运行这些测试。

## API 测试

PettingZoo 的 API 有许多特性和要求。为了确保你的环境与 API 一致，我们有 api_test。以下是一个示例：

``` python
from pettingzoo.test import api_test
from pettingzoo.butterfly import pistonball_v6
env = pistonball_v6.env()
api_test(env, num_cycles=1000, verbose_progress=False)
```

如你所见，你只需将环境传递给测试。如果有 API 问题，测试将断言或给出其他错误，如果通过则正常返回。

可选参数有：

* `num_cycles`：运行环境的周期数，并检查输出是否与 API 一致。
* `verbose_progress`：打印消息以指示测试的部分完成情况。对调试环境很有用。

## 并行 API 测试

这是 API 测试的类似版本，但用于并行环境。你可以这样使用这个测试：

``` python
from pettingzoo.test import parallel_api_test
from pettingzoo.butterfly import pistonball_v6
env = pistonball_v6.parallel_env()
parallel_api_test(env, num_cycles=1000)
```

## 种子测试

要有一个使用随机性的正确可重现环境，你需要能够在评估期间通过设置定义随机行为的随机数生成器的种子来使其确定性。种子测试检查使用常数调用 `seed()` 方法是否真的使环境具有确定性。

种子测试接受一个创建 pettingzoo 环境的函数。例如：

``` python
from pettingzoo.test import seed_test, parallel_seed_test
from pettingzoo.butterfly import pistonball_v6
env_fn = pistonball_v6.env
seed_test(env_fn, num_cycles=10)

# 或者对于并行环境
parallel_env_fn = pistonball_v6.parallel_env
parallel_seed_test(parallel_env_fn)
```

内部有两个独立的测试。

1. 环境设置种子后，两个独立的环境是否给出相同的结果？
2. 调用 seed() 然后调用 reset() 后，单个环境是否给出相同的结果？

第一个可选参数 `num_cycles` 表示运行环境多长时间来检查确定性。有些环境只在初始化很久之后才会失败测试。

第二个可选参数 `test_kept_state` 允许用户禁用第二个测试。一些基于物理的环境由于缓存等原因导致的几乎无法检测到的差异而无法通过此测试，这些差异并不重要到需要关注的程度。

### 最大周期测试

最大周期测试测试 `max_cycles` 环境参数是否存在，以及生成的环境是否实际运行正确的周期数。如果你的环境不接受 `max_cycles` 参数，你不应该运行这个测试。这个测试存在的原因是在实现 `max_cycles` 时可能会出现许多差一错误。示例测试用法如下：

``` python
from pettingzoo.test import max_cycles_test
from pettingzoo.butterfly import pistonball_v6
max_cycles_test(pistonball_v6)
```

## 渲染测试

渲染测试检查渲染 1) 不会崩溃 2) 在给定模式时产生正确类型的输出（仅支持 `'human'`、`'ansi'` 和 `'rgb_array'` 模式）。
``` python
from pettingzoo.test import render_test
from pettingzoo.butterfly import pistonball_v6
env_func = pistonball_v6.env
render_test(env_func)
```

渲染测试方法接受一个可选参数 `custom_tests`，允许在非标准模式下进行额外测试。

``` python
from pettingzoo.test import render_test
from pettingzoo.butterfly import pistonball_v6
env_func = pistonball_v6.env

custom_tests = {
    "svg": lambda render_result: isinstance(render_result, str)
}
render_test(env_func, custom_tests=custom_tests)
```

## 性能基准测试

为了确保我们不会出现性能倒退，我们有性能基准测试。这个测试只是打印出环境在 5 秒内执行的步骤和周期数。这个测试需要手动检查其输出：

``` python
from pettingzoo.test import performance_benchmark
from pettingzoo.butterfly import pistonball_v6
env = pistonball_v6.env()
performance_benchmark(env)
```

## 保存观察测试

保存观察测试是为了视觉检查具有图形观察的游戏的观察结果，以确保它们是预期的。我们发现观察是环境中的一个巨大的错误来源，所以在可能的情况下手动检查它们是很好的。这个测试只是尝试保存所有智能体的观察结果。如果失败，它只会打印一个警告。输出需要进行视觉检查以确保正确性。

``` python
from pettingzoo.test import test_save_obs
from pettingzoo.butterfly import pistonball_v6
env = pistonball_v6.env()
test_save_obs(env)
```
