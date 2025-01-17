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
    "parallel_wrapper_fn",
    "parallel_to_aec_wrapper",
    "base_env_wrapper_fn",
    "BaseAtariEnv",
    "ParallelAtariEnv",
]


def base_env_wrapper_fn(raw_env_fn):
    def env_fn(**kwargs):
        env = raw_env_fn(**kwargs)
        env = wrappers.AssertOutOfBoundsWrapper(env)
        env = wrappers.OrderEnforcingWrapper(env)
        return env

    return env_fn


def BaseAtariEnv(**kwargs):
    return parallel_to_aec_wrapper(ParallelAtariEnv(**kwargs))


class ParallelAtariEnv(ParallelEnv, EzPickle):
    def __init__(
        self,
        game,
        num_players,
        mode_num=None,
        seed=None,
        obs_type="rgb_image",
        full_action_space=False,
        env_name=None,
        max_cycles=100000,
        render_mode=None,
        auto_rom_install_path=None,
    ):
        """初始化 `ParallelAtariEnv` 类。

        帧跳过（frameskip）应该是一个元组（表示要从中选择的随机范围，顶部值被排除）或一个整数。
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
        ), "obs_type 必须是 'ram' 或 'rgb_image' 或 'grayscale_image'"
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

        # 开始在本地目录中查找
        final = start / f"{game}.bin"
        if not final.exists():
            # 如果不行，在 'roms' 中查找
            final = start / "roms" / f"{game}.bin"

        if not final.exists():
            # 使用旧的 AutoROM 安装路径作为备份
            final = start / "ROM" / game / f"{game}.bin"

        if not final.exists():
            raise OSError(
                f"rom {game} 未安装。请使用 AutoROM 工具安装 roms（https://github.com/Farama-Foundation/AutoROM）"
                "或使用 `rom_path` 参数指定并仔细检查 Atari rom 的路径。"
            )

        self.rom_path = str(final)
        self.ale.loadROM(self.rom_path)

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

        if full_action_space:
            action_size = 18
            action_mapping = np.arange(action_size)
        else:
            action_mapping = self.ale.getMinimalActionSet()
            action_size = len(action_mapping)

        self.action_mapping = action_mapping

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

        player_names = ["first", "second", "third", "fourth"]
        self.agents = [f"{player_names[n]}_0" for n in range(num_players)]
        self.possible_agents = self.agents[:]

        self.action_spaces = {
            agent: gymnasium.spaces.Discrete(action_size)
            for agent in self.possible_agents
        }
        self.observation_spaces = {
            agent: observation_space for agent in self.possible_agents
        }

        self._screen = None
        self._seed(seed)

    def _seed(self, seed):
        self.np_random, seed = seeding.np_random(seed)
        self.ale.setInt(b"random_seed", seed)
        self.ale.loadROM(self.rom_path)
        self.ale.setMode(self.mode)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._seed(seed=seed)
        else:
            self.np_random, seed = seeding.np_random()
        self.ale.reset_game()
        self.agents = self.possible_agents[:]
        self.terminations = {agent: False for agent in self.possible_agents}
        self.frame = 0

        obs = self._observe()
        infos = {agent: {} for agent in self.possible_agents if agent in self.agents}
        return {agent: obs for agent in self.agents}, infos

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _observe(self):
        if self.obs_type == "ram":
            bytes = self.ale.getRAM()
            return bytes
        elif self.obs_type == "rgb_image":
            return self.ale.getScreenRGB()
        elif self.obs_type == "grayscale_image":
            return self.ale.getScreenGrayscale()

    def step(self, action_dict):
        actions = np.zeros(self.max_num_agents, dtype=np.int32)
        for i, agent in enumerate(self.possible_agents):
            if agent in action_dict:
                actions[i] = action_dict[agent]

        actions = self.action_mapping[actions]
        rewards = self.ale.act(actions)
        self.frame += 1
        truncations = {agent: self.frame >= self.max_cycles for agent in self.agents}

        if self.ale.game_over():
            terminations = {agent: True for agent in self.agents}
        else:
            lives = self.ale.allLives()
            # an inactive agent in ale gets a -1 life.
            terminations = {
                agent: int(life) < 0
                for agent, life in zip(self.possible_agents, lives)
                if agent in self.agents
            }

        obs = self._observe()
        observations = {agent: obs for agent in self.agents}
        rewards = {
            agent: rew
            for agent, rew in zip(self.possible_agents, rewards)
            if agent in self.agents
        }
        infos = {agent: {} for agent in self.possible_agents if agent in self.agents}
        self.agents = [agent for agent in self.agents if not terminations[agent]]

        if self.render_mode == "human":
            self.render()
        return observations, rewards, terminations, truncations, infos

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "您正在调用渲染方法，但没有指定任何渲染模式。"
            )
            return

        assert (
            self.render_mode in self.metadata["render_modes"]
        ), f"{self.render_mode} 不是有效的渲染模式"
        (screen_width, screen_height) = self.ale.getScreenDims()
        image = self.ale.getScreenRGB()
        if self.render_mode == "human":
            zoom_factor = 4
            if self._screen is None:
                pygame.init()
                self._screen = pygame.display.set_mode(
                    (screen_width * zoom_factor, screen_height * zoom_factor)
                )

            myImage = pygame.image.frombuffer(
                image.tobytes(), image.shape[:2][::-1], "RGB"
            )

            myImage = pygame.transform.scale(
                myImage, (screen_width * zoom_factor, screen_height * zoom_factor)
            )

            self._screen.blit(myImage, (0, 0))

            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return image

    def close(self):
        if self._screen is not None:
            pygame.quit()
            self._screen = None

    def clone_state(self):
        """克隆模拟器状态（不包括系统状态）。

        恢复此状态将*不会*得到相同的环境。
        要完整克隆和恢复完整状态，
        请参见 `{clone,restore}_full_state()`。
        """
        state_ref = self.ale.cloneState()
        state = self.ale.encodeState(state_ref)
        self.ale.deleteState(state_ref)
        return state

    def restore_state(self, state):
        """恢复模拟器状态（不包括系统状态）。"""
        state_ref = self.ale.decodeState(state)
        self.ale.restoreState(state_ref)
        self.ale.deleteState(state_ref)

    def clone_full_state(self):
        """克隆模拟器状态（包括系统状态和伪随机性）。

        恢复此状态将得到相同的环境。
        """
        state_ref = self.ale.cloneSystemState()
        state = self.ale.encodeState(state_ref)
        self.ale.deleteState(state_ref)
        return state

    def restore_full_state(self, state):
        """恢复模拟器状态（包括系统状态和伪随机性）。"""
        state_ref = self.ale.decodeState(state)
        self.ale.restoreSystemState(state_ref)
        self.ale.deleteState(state_ref)
