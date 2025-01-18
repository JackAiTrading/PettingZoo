# 从waterworld包导入环境类型
from pettingzoo.sisl.waterworld.waterworld import env, parallel_env, raw_env

# 导出环境类型：标准环境、并行环境和原始环境
__all__ = ["env", "parallel_env", "raw_env"]
