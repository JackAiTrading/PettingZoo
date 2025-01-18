# SISL环境模块导入和注册
from pettingzoo.sisl import multiwalker_v9, pursuit_v4, waterworld_v4

# SISL环境字典，包含所有可用的SISL环境
sisl_environments = {
    "sisl/multiwalker_v9": multiwalker_v9,  # 多智能体步行者环境
    "sisl/waterworld_v4": waterworld_v4,    # 水世界环境
    "sisl/pursuit_v4": pursuit_v4,          # 追捕环境
}
