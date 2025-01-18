"""
活塞球游戏环境测试模块。

这个模块包含了活塞球游戏环境的单元测试。
主要测试环境的基本功能和边界情况。

测试内容：
1. 环境初始化
2. 动作空间和观察空间
3. 奖励计算
4. 状态转换
5. 渲染功能
"""

import numpy as np
import pytest

from pettingzoo.butterfly.pistonball.pistonball import raw_env
from pettingzoo.test.api_test import api_test
from pettingzoo.utils import parallel_to_aec_wrapper


def test_env_creation():
    """测试环境创建。
    
    验证环境是否能正确创建，并检查基本属性。
    """
    env = raw_env()
    assert env is not None
    assert len(env.possible_agents) == 15  # 默认15个活塞
    assert all(isinstance(agent, str) for agent in env.possible_agents)


def test_reset():
    """测试环境重置。
    
    验证重置功能是否正常工作。
    """
    env = raw_env()
    observations, infos = env.reset()
    
    # 检查观察和信息字典
    assert isinstance(observations, dict)
    assert isinstance(infos, dict)
    assert set(observations.keys()) == set(env.agents)
    assert set(infos.keys()) == set(env.agents)
    
    # 检查观察值
    for agent in env.agents:
        assert observations[agent].shape == (100, 120, 3)
        assert observations[agent].dtype == np.uint8


def test_step():
    """测试环境步进。
    
    验证动作执行和状态转换是否正确。
    """
    env = raw_env()
    env.reset()
    
    # 创建随机动作
    actions = {agent: env.action_spaces[agent].sample() for agent in env.agents}
    
    # 执行动作
    observations, rewards, terminations, truncations, infos = env.step(actions)
    
    # 验证返回值
    assert isinstance(observations, dict)
    assert isinstance(rewards, dict)
    assert isinstance(terminations, dict)
    assert isinstance(truncations, dict)
    assert isinstance(infos, dict)
    
    # 检查键值匹配
    assert set(observations.keys()) == set(env.agents)
    assert set(rewards.keys()) == set(env.agents)
    assert set(terminations.keys()) == set(env.agents)
    assert set(truncations.keys()) == set(env.agents)
    assert set(infos.keys()) == set(env.agents)


def test_render():
    """测试渲染功能。
    
    验证不同渲染模式是否正常工作。
    """
    # 测试rgb_array模式
    env = raw_env(render_mode="rgb_array")
    env.reset()
    render_result = env.render()
    assert isinstance(render_result, np.ndarray)
    assert render_result.shape[2] == 3  # RGB格式
    
    # 测试human模式
    env = raw_env(render_mode="human")
    env.reset()
    render_result = env.render()
    assert render_result is None  # human模式返回None
    
    # 清理资源
    env.close()


def test_reward_calculation():
    """测试奖励计算。
    
    验证奖励计算是否符合预期。
    """
    env = raw_env(time_penalty=0.1)
    env.reset()
    
    # 执行随机动作
    actions = {agent: env.action_spaces[agent].sample() for agent in env.agents}
    _, rewards, _, _, _ = env.step(actions)
    
    # 验证奖励范围
    for reward in rewards.values():
        assert isinstance(reward, float)
        assert -1.0 <= reward <= 1.0  # 考虑时间惩罚


def test_parallel_api():
    """测试并行API。
    
    验证环境是否支持并行API。
    """
    env = raw_env()
    assert hasattr(env, "step")
    assert hasattr(env, "reset")
    
    # 测试AEC转换
    aec_env = parallel_to_aec_wrapper(env)
    assert hasattr(aec_env, "step")
    assert hasattr(aec_env, "reset")


def test_api_compatibility():
    """测试API兼容性。
    
    使用PettingZoo的API测试工具验证环境。
    """
    env = raw_env()
    api_test(env)


def test_custom_parameters():
    """测试自定义参数。
    
    验证环境是否正确处理自定义参数。
    """
    # 测试不同数量的活塞
    env = raw_env(n_pistons=10)
    assert len(env.possible_agents) == 10
    
    # 测试连续动作空间
    env = raw_env(continuous=True)
    for space in env.action_spaces.values():
        assert isinstance(space, gym.spaces.Box)
    
    # 测试随机参数
    env = raw_env(random_drop=False, random_rotate=False)
    obs, _ = env.reset()
    assert obs is not None


if __name__ == "__main__":
    pytest.main([__file__])
