from collections import defaultdict
from typing import Optional

import gymnasium
import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.utils import seeding

from pettingzoo.sisl.pursuit.utils import agent_utils, two_d_maps
from pettingzoo.sisl.pursuit.utils.agent_layer import AgentLayer
from pettingzoo.sisl.pursuit.utils.controllers import (
    PursuitPolicy,
    RandomPolicy,
    SingleActionPolicy,
)


class Pursuit:
    def __init__(
        self,
        x_size: int = 16,
        y_size: int = 16,
        max_cycles: int = 500,
        shared_reward: bool = True,
        n_evaders: int = 30,
        n_pursuers: int = 8,
        obs_range: int = 7,
        n_catch: int = 2,
        freeze_evaders: bool = False,
        evader_controller: Optional[PursuitPolicy] = None,
        pursuer_controller: Optional[PursuitPolicy] = None,
        tag_reward: float = 0.01,
        catch_reward: float = 5.0,
        urgency_reward: float = -0.1,
        surround: bool = True,
        render_mode=None,
        constraint_window: float = 1.0,
    ):
        """在追捕游戏中，一组追捕者必须'标记'一组逃避者。

        必需参数：
            x_size, y_size：世界大小
            shared_reward：奖励是否在所有智能体之间共享
            n_evaders：逃避者数量
            n_pursuers：追捕者数量
            obs_range：每个智能体的视野范围

        可选参数：
            pursuer_controller：盟友追捕者的固定策略
            evader_controller：对手逃避者的固定策略
            tag_reward：'标记'单个逃避者的奖励
            max_cycles：游戏应该在多少帧后结束
            n_catch：逃避者需要被包围到什么程度才会被移除
            freeze_evaders：切换逃避者是否移动
            catch_reward：捕获逃避者的奖励
            urgency_reward：每步的紧迫感奖励
            surround：切换逃避者移除的包围条件
            constraint_window：智能体可以随机生成的窗口范围
        """
        self.x_size = x_size
        self.y_size = y_size
        self.map_matrix = two_d_maps.rectangle_map(self.x_size, self.y_size)
        self.max_cycles = max_cycles
        self._seed()

        self.shared_reward = shared_reward
        self.local_ratio = 1.0 - float(self.shared_reward)

        self.n_evaders = n_evaders
        self.n_pursuers = n_pursuers
        self.num_agents = self.n_pursuers

        self.latest_reward_state = [0 for _ in range(self.num_agents)]
        self.latest_done_state = [False for _ in range(self.num_agents)]
        self.latest_obs = [None for _ in range(self.num_agents)]

        # 默认情况下可以看到周围7个格子
        self.obs_range = obs_range
        # assert self.obs_range % 2 != 0, "obs_range应该是奇数"
        self.obs_offset = int((self.obs_range - 1) / 2)
        self.pursuers = agent_utils.create_agents(
            self.n_pursuers, self.map_matrix, self.obs_range, self.np_random
        )
        self.evaders = agent_utils.create_agents(
            self.n_evaders, self.map_matrix, self.obs_range, self.np_random
        )

        self.pursuer_layer = AgentLayer(x_size, y_size, self.pursuers)
        self.evader_layer = AgentLayer(x_size, y_size, self.evaders)

        self.n_catch = n_catch

        n_act_purs = self.pursuer_layer.get_nactions(0)
        n_act_ev = self.evader_layer.get_nactions(0)

        self.freeze_evaders = freeze_evaders

        if self.freeze_evaders:
            self.evader_controller = (
                SingleActionPolicy(4)
                if evader_controller is None
                else evader_controller
            )
            self.pursuer_controller = (
                SingleActionPolicy(4)
                if pursuer_controller is None
                else pursuer_controller
            )
        else:
            self.evader_controller = (
                RandomPolicy(n_act_purs, self.np_random)
                if evader_controller is None
                else evader_controller
            )
            self.pursuer_controller = (
                RandomPolicy(n_act_ev, self.np_random)
                if pursuer_controller is None
                else pursuer_controller
            )

        self.current_agent_layer = np.zeros((x_size, y_size), dtype=np.int32)

        self.tag_reward = tag_reward  # 标记单个逃避者的奖励

        self.catch_reward = catch_reward  # 捕获逃避者的奖励

        self.urgency_reward = urgency_reward  # 每步的紧迫感奖励

        self.ally_actions = np.zeros(n_act_purs, dtype=np.int32)  # 盟友动作
        self.opponent_actions = np.zeros(n_act_ev, dtype=np.int32)  # 对手动作

        max_agents_overlap = max(self.n_pursuers, self.n_evaders)  # 最大智能体重叠数
        obs_space = spaces.Box(
            low=0,
            high=max_agents_overlap,
            shape=(self.obs_range, self.obs_range, 3),
            dtype=np.float32,
        )
        act_space = spaces.Discrete(n_act_purs)
        self.action_space = [act_space for _ in range(self.n_pursuers)]

        self.observation_space = [obs_space for _ in range(self.n_pursuers)]
        self.act_dims = [n_act_purs for i in range(self.n_pursuers)]

        self.evaders_gone = np.array([False for i in range(self.n_evaders)])  # 逃避者是否被移除

        self.surround = surround  # 是否使用包围条件

        self.render_mode = render_mode
        self.screen = None
        self.constraint_window = constraint_window  # 生成窗口约束

        self.surround_mask = np.array([[-1, 0], [1, 0], [0, 1], [0, -1]])  # 包围掩码

        self.model_state = np.zeros((4,) + self.map_matrix.shape, dtype=np.float32)  # 模型状态
        self.pixel_scale = 30  # 像素缩放比例

        self.frames = 0  # 当前帧数
        self.reset()

    def observation_space(self, agent):
        """返回指定智能体的观察空间"""
        return self.observation_spaces[agent]

    def action_space(self, agent):
        """返回指定智能体的动作空间"""
        return self.action_spaces[agent]

    def close(self):
        """关闭游戏窗口"""
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    #################################################################
    # 以下函数是与MultiAgentSimulator的接口 #
    #################################################################

    @property
    def agents(self):
        """返回追捕者列表"""
        return self.pursuers

    def _seed(self, seed=None):
        """设置随机种子"""
        self.np_random, seed_ = seeding.np_random(seed)
        try:
            policies = [self.evader_controller, self.pursuer_controller]
            for policy in policies:
                try:
                    policy.set_rng(self.np_random)
                except AttributeError:
                    pass
        except AttributeError:
            pass

        return [seed_]

    def get_param_values(self):
        """返回所有参数值"""
        return self.__dict__

    def reset(self):
        """重置环境状态"""
        self.evaders_gone.fill(False)

        x_window_start = self.np_random.uniform(0.0, 1.0 - self.constraint_window)
        y_window_start = self.np_random.uniform(0.0, 1.0 - self.constraint_window)
        xlb, xub = int(self.x_size * x_window_start), int(
            self.x_size * (x_window_start + self.constraint_window)
        )
        ylb, yub = int(self.y_size * y_window_start), int(
            self.y_size * (y_window_start + self.constraint_window)
        )
        constraints = [[xlb, xub], [ylb, yub]]

        self.pursuers = agent_utils.create_agents(
            self.n_pursuers,
            self.map_matrix,
            self.obs_range,
            self.np_random,
            randinit=True,
            constraints=constraints,
        )
        self.pursuer_layer = AgentLayer(self.x_size, self.y_size, self.pursuers)

        self.evaders = agent_utils.create_agents(
            self.n_evaders,
            self.map_matrix,
            self.obs_range,
            self.np_random,
            randinit=True,
            constraints=constraints,
        )
        self.evader_layer = AgentLayer(self.x_size, self.y_size, self.evaders)

        self.latest_reward_state = [0 for _ in range(self.num_agents)]
        self.latest_done_state = [False for _ in range(self.num_agents)]
        self.latest_obs = [None for _ in range(self.num_agents)]

        self.model_state[0] = self.map_matrix
        self.model_state[1] = self.pursuer_layer.get_state_matrix()
        self.model_state[2] = self.evader_layer.get_state_matrix()

        self.frames = 0

        return self.safely_observe(0)

    def step(self, action, agent_id, is_last):
        """执行一步游戏"""
        agent_layer = self.pursuer_layer
        opponent_layer = self.evader_layer
        opponent_controller = self.evader_controller

        # 应用动作，改变追捕者层
        agent_layer.move_agent(agent_id, action)

        # 更新追捕者层
        self.model_state[1] = self.pursuer_layer.get_state_matrix()

        self.latest_reward_state = self.reward() / self.num_agents

        if is_last:
            # 可能改变逃避者层
            ev_remove, pr_remove, pursuers_who_remove = self.remove_agents()

            for i in range(opponent_layer.n_agents()):
                # 控制器输入应该是一个观察，但现在不重要
                a = opponent_controller.act(self.model_state)
                opponent_layer.move_agent(i, a)

            self.latest_reward_state += self.catch_reward * pursuers_who_remove
            self.latest_reward_state += self.urgency_reward
            self.frames = self.frames + 1

        # 更新剩余层
        self.model_state[0] = self.map_matrix
        self.model_state[2] = self.evader_layer.get_state_matrix()

        global_val = self.latest_reward_state.mean()
        local_val = self.latest_reward_state
        self.latest_reward_state = (
            self.local_ratio * local_val + (1 - self.local_ratio) * global_val
        )

        if self.render_mode == "human":
            self.render()

    def draw_model_state(self):
        """绘制模型状态"""
        # -1 是建筑像素标志
        x_len, y_len = self.model_state[0].shape
        for x in range(x_len):
            for y in range(y_len):
                pos = pygame.Rect(
                    self.pixel_scale * x,
                    self.pixel_scale * y,
                    self.pixel_scale,
                    self.pixel_scale,
                )
                col = (0, 0, 0)
                if self.model_state[0][x][y] == -1:
                    col = (255, 255, 255)
                pygame.draw.rect(self.screen, col, pos)

    def draw_pursuers_observations(self):
        """绘制追捕者的观察"""
        for i in range(self.pursuer_layer.n_agents()):
            x, y = self.pursuer_layer.get_position(i)
            patch = pygame.Surface(
                (self.pixel_scale * self.obs_range, self.pixel_scale * self.obs_range)
            )
            patch.set_alpha(128)
            patch.fill((255, 152, 72))
            ofst = self.obs_range / 2.0
            self.screen.blit(
                patch,
                (
                    self.pixel_scale * (x - ofst + 1 / 2),
                    self.pixel_scale * (y - ofst + 1 / 2),
                ),
            )

    def draw_pursuers(self):
        """绘制追捕者"""
        for i in range(self.pursuer_layer.n_agents()):
            x, y = self.pursuer_layer.get_position(i)
            center = (
                int(self.pixel_scale * x + self.pixel_scale / 2),
                int(self.pixel_scale * y + self.pixel_scale / 2),
            )
            col = (255, 0, 0)
            pygame.draw.circle(self.screen, col, center, int(self.pixel_scale / 3))

    def draw_evaders(self):
        """绘制逃避者"""
        for i in range(self.evader_layer.n_agents()):
            x, y = self.evader_layer.get_position(i)
            center = (
                int(self.pixel_scale * x + self.pixel_scale / 2),
                int(self.pixel_scale * y + self.pixel_scale / 2),
            )
            col = (0, 0, 255)

            pygame.draw.circle(self.screen, col, center, int(self.pixel_scale / 3))

    def draw_agent_counts(self):
        """绘制智能体数量"""
        font = pygame.font.SysFont("Comic Sans MS", self.pixel_scale * 2 // 3)

        agent_positions = defaultdict(int)
        evader_positions = defaultdict(int)

        for i in range(self.evader_layer.n_agents()):
            x, y = self.evader_layer.get_position(i)
            evader_positions[(x, y)] += 1

        for i in range(self.pursuer_layer.n_agents()):
            x, y = self.pursuer_layer.get_position(i)
            agent_positions[(x, y)] += 1

        for x, y in evader_positions:
            (pos_x, pos_y) = (
                self.pixel_scale * x + self.pixel_scale // 2,
                self.pixel_scale * y + self.pixel_scale // 2,
            )

            agent_count = evader_positions[(x, y)]
            count_text: str
            if agent_count < 1:
                count_text = ""
            elif agent_count < 10:
                count_text = str(agent_count)
            else:
                count_text = "+"

            text = font.render(count_text, False, (0, 255, 255))

            self.screen.blit(text, (pos_x, pos_y))

        for x, y in agent_positions:
            (pos_x, pos_y) = (
                self.pixel_scale * x + self.pixel_scale // 2,
                self.pixel_scale * y + self.pixel_scale // 2,
            )

            agent_count = agent_positions[(x, y)]
            count_text: str
            if agent_count < 1:
                count_text = ""
            elif agent_count < 10:
                count_text = str(agent_count)
            else:
                count_text = "+"

            text = font.render(count_text, False, (255, 255, 0))

            self.screen.blit(text, (pos_x, pos_y - self.pixel_scale // 2))

    def render(self):
        """渲染游戏"""
        if self.render_mode is None:
            gymnasium.logger.warn(
                "您正在调用渲染方法，而没有指定任何渲染模式。"
            )
            return

        if self.screen is None:
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.pixel_scale * self.x_size, self.pixel_scale * self.y_size)
                )
                pygame.display.set_caption("Pursuit")
            else:
                self.screen = pygame.Surface(
                    (self.pixel_scale * self.x_size, self.pixel_scale * self.y_size)
                )

        self.draw_model_state()

        self.draw_pursuers_observations()

        self.draw_evaders()
        self.draw_pursuers()
        self.draw_agent_counts()

        observation = pygame.surfarray.pixels3d(self.screen)
        new_observation = np.copy(observation)
        del observation
        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
        return (
            np.transpose(new_observation, axes=(1, 0, 2))
            if self.render_mode == "rgb_array"
            else None
        )

    def save_image(self, file_name):
        """保存游戏截图"""
        self.render()
        capture = pygame.surfarray.array3d(self.screen)

        xl, xh = -self.obs_offset - 1, self.x_size + self.obs_offset + 1
        yl, yh = -self.obs_offset - 1, self.y_size + self.obs_offset + 1

        window = pygame.Rect(xl, yl, xh, yh)
        subcapture = capture.subsurface(window)

        pygame.image.save(subcapture, file_name)

    def reward(self):
        """计算奖励"""
        es = self.evader_layer.get_state_matrix()  # 逃避者位置
        rewards = [
            self.tag_reward
            * np.sum(
                es[
                    np.clip(
                        self.pursuer_layer.get_position(i)[0]
                        + self.surround_mask[:, 0],
                        0,
                        self.x_size - 1,
                    ),
                    np.clip(
                        self.pursuer_layer.get_position(i)[1]
                        + self.surround_mask[:, 1],
                        0,
                        self.y_size - 1,
                    ),
                ]
            )
            for i in range(self.n_pursuers)
        ]
        return np.array(rewards)

    @property
    def is_terminal(self):
        """检查游戏是否结束"""
        # ev = self.evader_layer.get_state_matrix()  # 逃避者位置
        # if np.sum(ev) == 0.0:
        if self.evader_layer.n_agents() == 0:
            return True
        return False

    def update_ally_controller(self, controller):
        """更新盟友控制器"""
        self.ally_controller = controller

    def update_opponent_controller(self, controller):
        """更新对手控制器"""
        self.opponent_controller = controller

    def n_agents(self):
        """返回智能体数量"""
        return self.pursuer_layer.n_agents()

    def safely_observe(self, i):
        """安全地观察指定智能体"""
        agent_layer = self.pursuer_layer
        obs = self.collect_obs(agent_layer, i)
        return obs

    def collect_obs(self, agent_layer, i):
        """收集指定智能体的观察"""
        for j in range(self.n_agents()):
            if i == j:
                return self.collect_obs_by_idx(agent_layer, i)
        assert False, "bad index"

    def collect_obs_by_idx(self, agent_layer, agent_idx):
        """通过索引收集指定智能体的观察"""
        # 返回一个所有观察的平面数组
        obs = np.zeros((3, self.obs_range, self.obs_range), dtype=np.float32)
        obs[0].fill(1.0)  # 边界墙设置为-0.1？
        xp, yp = agent_layer.get_position(agent_idx)

        xlo, xhi, ylo, yhi, xolo, xohi, yolo, yohi = self.obs_clip(xp, yp)

        obs[0:3, xolo:xohi, yolo:yohi] = np.abs(self.model_state[0:3, xlo:xhi, ylo:yhi])
        return obs

    def obs_clip(self, x, y):
        """裁剪观察"""
        xld = x - self.obs_offset
        xhd = x + self.obs_offset
        yld = y - self.obs_offset
        yhd = y + self.obs_offset
        xlo, xhi, ylo, yhi = (
            np.clip(xld, 0, self.x_size - 1),
            np.clip(xhd, 0, self.x_size - 1),
            np.clip(yld, 0, self.y_size - 1),
            np.clip(yhd, 0, self.y_size - 1),
        )
        xolo, yolo = abs(np.clip(xld, -self.obs_offset, 0)), abs(
            np.clip(yld, -self.obs_offset, 0)
        )
        xohi, yohi = xolo + (xhi - xlo), yolo + (yhi - ylo)
        return xlo, xhi + 1, ylo, yhi + 1, xolo, xohi + 1, yolo, yohi + 1

    def remove_agents(self):
        """移除被捕获的智能体"""
        # 返回元组（n_evader_removed，n_pursuer_removed，purs_sur）
        # purs_sur：bool数组，哪些追捕者包围了逃避者
        n_pursuer_removed = 0
        n_evader_removed = 0
        removed_evade = []
        removed_pursuit = []

        ai = 0
        rems = 0
        xpur, ypur = np.nonzero(self.model_state[1])
        purs_sur = np.zeros(self.n_pursuers, dtype=bool)
        for i in range(self.n_evaders):
            if self.evaders_gone[i]:
                continue
            x, y = self.evader_layer.get_position(ai)
            if self.surround:
                pos_that_catch = self.surround_mask + self.evader_layer.get_position(ai)
                truths = np.array(
                    [
                        np.equal([xi, yi], pos_that_catch).all(axis=1)
                        for xi, yi in zip(xpur, ypur)
                    ]
                )
                if np.sum(truths.any(axis=0)) == self.need_to_surround(x, y):
                    removed_evade.append(ai - rems)
                    self.evaders_gone[i] = True
                    rems += 1
                    tt = truths.any(axis=1)
                    for j in range(self.n_pursuers):
                        xpp, ypp = self.pursuer_layer.get_position(j)
                        tes = np.concatenate((xpur[tt], ypur[tt])).reshape(
                            2, len(xpur[tt])
                        )
                        tem = tes.T == np.array([xpp, ypp])
                        if np.any(np.all(tem, axis=1)):
                            purs_sur[j] = True
                ai += 1
            else:
                if self.model_state[1, x, y] >= self.n_catch:
                    # 添加概率移除？
                    removed_evade.append(ai - rems)
                    self.evaders_gone[i] = True
                    rems += 1
                    for j in range(self.n_pursuers):
                        xpp, ypp = self.pursuer_layer.get_position(j)
                        if xpp == x and ypp == y:
                            purs_sur[j] = True
                ai += 1

        ai = 0
        for i in range(self.pursuer_layer.n_agents()):
            x, y = self.pursuer_layer.get_position(i)
            # 可以在这里随机移除追捕者吗？
        for ridx in removed_evade:
            self.evader_layer.remove_agent(ridx)
            n_evader_removed += 1
        for ridx in removed_pursuit:
            self.pursuer_layer.remove_agent(ridx)
            n_pursuer_removed += 1
        return n_evader_removed, n_pursuer_removed, purs_sur

    def need_to_surround(self, x, y):
        """计算需要包围的格子数量"""
        # 计算x，y位置周围需要包围的格子数量
        tosur = 4
        if x == 0 or x == (self.x_size - 1):
            tosur -= 1
        if y == 0 or y == (self.y_size - 1):
            tosur -= 1
        neighbors = self.surround_mask + np.array([x, y])
        for n in neighbors:
            xn, yn = n
            if not 0 < xn < self.x_size or not 0 < yn < self.y_size:
                continue
            if self.model_state[0][xn, yn] == -1:
                tosur -= 1
        return tosur
