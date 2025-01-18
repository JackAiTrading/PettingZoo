import numpy as np
from gymnasium import spaces

from pettingzoo.sisl._utils import Agent

#################################################################
# 实现单个2D智能体的动态
#################################################################


class DiscreteAgent(Agent):
    # 构造函数
    def __init__(
        self,
        xs,
        ys,
        map_matrix,
        randomizer,
        obs_range=3,
        n_channels=3,
        seed=1,
        flatten=False,
    ):
        # map_matrix是环境的地图（-1表示建筑物）
        # n_channels是观察通道的数量

        self.random_state = randomizer

        self.xs = xs
        self.ys = ys

        self.eactions = [
            0,  # 向左移动
            1,  # 向右移动
            2,  # 向上移动
            3,  # 向下移动
            4,  # 停留
        ]

        self.motion_range = [[-1, 0], [1, 0], [0, 1], [0, -1], [0, 0]]

        self.current_pos = np.zeros(2, dtype=np.int32)  # x和y位置
        self.last_pos = np.zeros(2, dtype=np.int32)  # 上一个位置
        self.temp_pos = np.zeros(2, dtype=np.int32)  # 临时位置

        self.map_matrix = map_matrix

        self.terminal = False  # 是否终止

        self._obs_range = obs_range

        if flatten:
            self._obs_shape = (n_channels * obs_range**2 + 1,)
        else:
            self._obs_shape = (obs_range, obs_range, 4)
            # self._obs_shape = (4, obs_range, obs_range)

    @property
    def observation_space(self):
        """返回观察空间"""
        return spaces.Box(low=-np.inf, high=np.inf, shape=self._obs_shape)

    @property
    def action_space(self):
        """返回动作空间"""
        return spaces.Discrete(5)

    # 动态函数
    def step(self, a):
        """执行一步动作

        参数：
            a：动作索引

        返回：
            当前位置
        """
        cpos = self.current_pos
        lpos = self.last_pos
        # 如果死亡或达到目标则不移动
        if self.terminal:
            return cpos
        # 如果在建筑物中，死亡并停留在那里
        if self.inbuilding(cpos[0], cpos[1]):
            self.terminal = True
            return cpos
        tpos = self.temp_pos
        tpos[0] = cpos[0]
        tpos[1] = cpos[1]

        # 转换是确定性的
        tpos += self.motion_range[a]
        x = tpos[0]
        y = tpos[1]

        # 检查边界
        if not self.inbounds(x, y):
            return cpos
        # 如果撞到建筑物，则停留
        if self.inbuilding(x, y):
            return cpos
        else:
            lpos[0] = cpos[0]
            lpos[1] = cpos[1]
            cpos[0] = x
            cpos[1] = y
            return cpos

    def get_state(self):
        """返回当前状态"""
        return self.current_pos

    # 辅助函数
    def inbounds(self, x, y):
        """检查位置是否在边界内"""
        if 0 <= x < self.xs and 0 <= y < self.ys:
            return True
        return False

    def inbuilding(self, x, y):
        """检查位置是否在建筑物内"""
        if self.map_matrix[x, y] == -1:
            return True
        return False

    def nactions(self):
        """返回动作数量"""
        return len(self.eactions)

    def set_position(self, xs, ys):
        """设置位置"""
        self.current_pos[0] = xs
        self.current_pos[1] = ys

    def current_position(self):
        """返回当前位置"""
        return self.current_pos

    def last_position(self):
        """返回上一个位置"""
        return self.last_pos
