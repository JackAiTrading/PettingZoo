"""
雅达利基础环境模块。

这个模块实现了所有雅达利游戏环境的基础类和功能，为具体游戏提供统一的接口和实现。
包括环境初始化、状态管理、动作处理等核心功能。

主要功能：
1. 环境管理
   - 初始化游戏
   - 重置状态
   - 渲染画面
   - 关闭环境

2. 状态处理
   - 观察空间
   - 动作空间
   - 奖励计算
   - 终止判定

3. 游戏控制
   - 帧率控制
   - 按键映射
   - 多人支持
   - 难度设置

4. 辅助功能
   - 录制回放
   - 状态保存
   - 错误处理
   - 调试信息

环境参数：
1. 基本设置
   - game - 游戏ROM名称
   - num_players - 玩家数量
   - mode_num - 游戏模式
   - seed - 随机种子

2. 显示设置
   - render_mode - 渲染模式
   - full_action_space - 完整动作空间
   - max_cycles - 最大步数
   - auto_rom_install_path - ROM安装路径

3. 观察设置
   - obs_type - 观察类型
   - repeat_action_probability - 动作重复概率
   - difficulty - 游戏难度
   - restrict_actions - 动作限制

4. 其他设置
   - frameskip - 跳帧数
   - repeat_action_probability - 动作重复概率
   - full_action_space - 完整动作空间
   - max_cycles - 最大步数

注意事项：
- ROM文件依赖
- 性能优化
- 多人同步
- 状态一致性
"""

"""Atari 环境的基类。

这个模块提供了所有 Atari 游戏环境的基础实现。
它包含了通用的功能，如环境初始化、状态管理、渲染等。
"""

from pathlib import Path

import gymnasium
import multi_agent_ale_py
import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.utils import EzPickle, seeding

from pettingzoo.utils import wrappers
from pettingzoo.utils.conversions import parallel_to_aec_wrapper, parallel_wrapper_fn
from pettingzoo.utils.env import ParallelEnv

__all__ = [
    "parallel_wrapper_fn",  # 并行包装器函数
    "parallel_to_aec_wrapper",  # 并行到 AEC 的包装器
    "base_env_wrapper_fn",  # 基础环境包装器函数
    "BaseAtariEnv",  # 基础 Atari 环境
    "ParallelAtariEnv",  # 并行 Atari 环境
]


def base_env_wrapper_fn(raw_env_fn):
    """创建一个基础环境包装器。

    Args:
        raw_env_fn: 原始环境函数

    Returns:
        包装后的环境函数
    """
    def env_fn(**kwargs):
        env = raw_env_fn(**kwargs)
        env = wrappers.AssertOutOfBoundsWrapper(env)
        env = wrappers.OrderEnforcingWrapper(env)
        return env

    return env_fn


def BaseAtariEnv(**kwargs):
    """创建一个基础 Atari 环境。

    Args:
        **kwargs: 关键字参数

    Returns:
        包装后的 AEC 环境
    """
    return parallel_to_aec_wrapper(ParallelAtariEnv(**kwargs))


class ParallelAtariEnv(ParallelEnv, EzPickle):
    """并行 Atari 环境类。

    这个类实现了并行版本的 Atari 游戏环境。
    它支持多个玩家同时行动，并提供了各种观察类型和动作空间选项。
    """

    def __init__(
        self,
        game,  # 游戏名称
        num_players,  # 玩家数量
        mode_num=None,  # 模式编号
        seed=None,  # 随机种子
        obs_type="rgb_image",  # 观察类型
        full_action_space=False,  # 是否使用完整动作空间
        env_name=None,  # 环境名称
        max_cycles=100000,  # 最大周期数
        render_mode=None,  # 渲染模式
        auto_rom_install_path=None,  # ROM 自动安装路径
    ):
        """初始化并行 Atari 环境。

        Args:
            game: Atari 游戏的名称
            num_players: 玩家数量
            mode_num: 游戏模式编号，如果为 None 则使用默认模式
            seed: 随机种子
            obs_type: 观察类型，可以是 'ram'、'rgb_image' 或 'grayscale_image'
            full_action_space: 是否使用完整的动作空间
            env_name: 环境名称，如果为 None 则自动生成
            max_cycles: 最大周期数
            render_mode: 渲染模式
            auto_rom_install_path: ROM 自动安装路径
        """
        EzPickle.__init__(
            self,
            game=game,
            num_players=num_players,
            mode_num=mode_num,
            seed=seed,
            obs_type=obs_type,
            full_action_space=full_action_space,
            env_name=env_name,
            max_cycles=max_cycles,
            render_mode=render_mode,
            auto_rom_install_path=auto_rom_install_path,
        )

        assert obs_type in (
            "ram",
            "rgb_image",
            "grayscale_image",
        ), "obs_type 必须是 'ram'、'rgb_image' 或 'grayscale_image'"
        self.obs_type = obs_type
        self.full_action_space = full_action_space
        self.num_players = num_players
        self.max_cycles = max_cycles
        if env_name is None:
            env_name = "custom_" + game
        self.metadata = {
            "render_modes": ["human", "rgb_array"],
            "name": env_name,
            "render_fps": 60,
        }
        self.render_mode = render_mode

        multi_agent_ale_py.ALEInterface.setLoggerMode("error")
        self.ale = multi_agent_ale_py.ALEInterface()

        self.ale.setFloat(b"repeat_action_probability", 0.0)

        if auto_rom_install_path is None:
            start = Path(multi_agent_ale_py.__file__).parent
        else:
            start = Path(auto_rom_install_path).resolve()

        # 开始在本地目录中查找 ROM
        final = start / f"{game}.bin"
        if not final.exists():
            # 如果不存在，在 'roms' 目录中查找
            final = start / "roms" / f"{game}.bin"

        if not final.exists():
            # 使用旧的 AutoROM 安装路径作为备份
            final = start / "ROM" / game / f"{game}.bin"

        if not final.exists():
            raise OSError(
                f"ROM {game} 未安装。请使用 AutoROM 工具安装 ROM（https://github.com/Farama-Foundation/AutoROM）"
                "或使用 `rom_path` 参数指定并仔细检查 Atari ROM 的路径。"
            )

        self.rom_path = str(final)
        self.ale.loadROM(self.rom_path)

        # 获取可用的游戏模式
        all_modes = self.ale.getAvailableModes(num_players)

        if mode_num is None:
            mode = all_modes[0]
        else:
            mode = mode_num
            assert (
                mode in all_modes
            ), f"mode_num 参数错误。选择了模式 {mode_num}，但只支持 {list(all_modes)} 这些模式"

        self.mode = mode
        self.ale.setMode(self.mode)
        assert num_players == self.ale.numPlayersActive()

        # 设置动作空间
        if full_action_space:
            action_size = 18
            action_mapping = np.arange(action_size)
        else:
            action_mapping = self.ale.getMinimalActionSet()
            action_size = len(action_mapping)

        self.action_mapping = action_mapping

        # 设置观察空间
        if obs_type == "ram":
            observation_space = gymnasium.spaces.Box(
                low=0, high=255, dtype=np.uint8, shape=(128,)
            )
        else:
            (screen_width, screen_height) = self.ale.getScreenDims()
            if obs_type == "rgb_image":
                num_channels = 3
            elif obs_type == "grayscale_image":
                num_channels = 1
            observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(screen_height, screen_width, num_channels),
                dtype=np.uint8,
            )

        # 设置智能体
        player_names = ["first", "second", "third", "fourth"]
        self.agents = [f"{player_names[n]}_0" for n in range(num_players)]
        self.possible_agents = self.agents[:]

        # 为每个智能体设置动作空间和观察空间
        self.action_spaces = {
            agent: gymnasium.spaces.Discrete(action_size)
            for agent in self.possible_agents
        }
        self.observation_spaces = {
            agent: observation_space for agent in self.possible_agents
        }

        self._screen = None
        self._seed(seed)

    def _seed(self, seed=None):
        """设置随机种子。

        Args:
            seed: 随机种子值

        Returns:
            使用的随机种子列表
        """
        self.np_random, seed = seeding.np_random(seed)
        self.ale.setInt(b"random_seed", seed)
        return [seed]

    def reset(self, seed=None, options=None):
        """重置环境到初始状态。

        Args:
            seed: 随机种子
            options: 重置选项

        Returns:
            observations: 初始观察值
            infos: 额外信息
        """
        if seed is not None:
            self._seed(seed)
        self.ale.reset_game()
        self.agents = self.possible_agents[:]
        self.num_cycles = 0
        obs = self._observe()
        infos = {agent: {} for agent in self.agents}
        return obs, infos

    def observation_space(self, agent):
        """获取指定智能体的观察空间。

        Args:
            agent: 智能体名称

        Returns:
            观察空间
        """
        return self.observation_spaces[agent]

    def action_space(self, agent):
        """获取指定智能体的动作空间。

        Args:
            agent: 智能体名称

        Returns:
            动作空间
        """
        return self.action_spaces[agent]

    def _observe(self):
        """获取当前环境的观察值。

        Returns:
            每个智能体的观察值字典
        """
        obs = {}
        if self.obs_type == "ram":
            obs_list = self.ale.getRAM()
        else:
            obs_list = self.ale.getScreenRGB2() if self.obs_type == "rgb_image" else self.ale.getScreenGrayscale()
        for agent in self.agents:
            obs[agent] = obs_list
        return obs

    def step(self, action_dict):
        """执行一步环境交互。

        Args:
            action_dict: 每个智能体的动作字典

        Returns:
            observations: 新的观察值
            rewards: 奖励值
            terminations: 终止状态
            truncations: 截断状态
            infos: 额外信息
        """
        actions = []
        for agent in self.agents:
            actions.append(self.action_mapping[action_dict[agent]])

        rewards = self.ale.act(actions)
        self.num_cycles += 1

        obs = self._observe()
        done = self.ale.game_over() or self.num_cycles >= self.max_cycles
        truncations = {agent: done for agent in self.agents}
        terminations = {agent: self.ale.game_over() for agent in self.agents}
        rewards = dict(zip(self.agents, rewards))
        infos = {agent: {} for agent in self.agents}

        return obs, rewards, terminations, truncations, infos

    def render(self):
        """渲染环境。

        Returns:
            根据渲染模式返回不同的渲染结果
        """
        if self.render_mode is None:
            return

        if self.render_mode == "human" and self._screen is None:
            pygame.init()
            (screen_width, screen_height) = self.ale.getScreenDims()
            self._screen = pygame.display.set_mode((screen_width, screen_height))
            pygame.display.set_caption("Arcade Learning Environment")

        rgb_image = self.ale.getScreenRGB2()

        if self.render_mode == "rgb_array":
            return rgb_image

        elif self.render_mode == "human":
            rgb_image = np.transpose(rgb_image, (1, 0, 2))
            pygame_surface = pygame.surfarray.make_surface(rgb_image)
            self._screen.blit(pygame_surface, (0, 0))
            pygame.display.flip()
            return None

    def close(self):
        """关闭环境，释放资源。"""
        if self._screen is not None:
            pygame.quit()
            self._screen = None

    def clone_state(self):
        """克隆模拟器状态（不包括系统状态）。

        Returns:
            当前模拟器状态的副本
        
        注意：
            恢复此状态将*不会*得到相同的环境。
            要完整克隆和恢复完整状态，请参见 `{clone,restore}_full_state()`。
        """
        return self.ale.cloneState()

    def restore_state(self, state):
        """恢复模拟器状态（不包括系统状态）。

        Args:
            state: 要恢复的状态
        """
        self.ale.restoreState(state)

    def clone_full_state(self):
        """克隆模拟器状态（包括系统状态和伪随机性）。

        Returns:
            当前模拟器的完整状态副本
        
        注意：
            恢复此状态将得到相同的环境。
        """
        return self.ale.cloneSystemState()

    def restore_full_state(self, state):
        """恢复模拟器状态（包括系统状态和伪随机性）。

        Args:
            state: 要恢复的完整状态
        """
        self.ale.restoreSystemState(state)
