"""
乒乓球拍类模块。

这个模块实现了乒乓球拍的基本功能和行为。
球拍可以在垂直方向上移动，并与球发生碰撞。

主要功能：
1. 球拍的移动控制
2. 碰撞检测
3. 位置更新
4. 速度限制

特性：
1. 可配置的大小和速度
2. 边界检查
3. 平滑移动
4. 碰撞响应
"""

"""
球拍类模块。

这个模块实现了合作乒乓游戏中球拍的行为，包括移动和碰撞检测。
每个球拍可以上下移动来击打球。
"""

import numpy as np


class Paddle:
    """球拍类。

    这个类实现了球拍的物理行为，包括位置更新和移动控制。

    属性:
        pos (numpy.ndarray): 球拍的位置 [x, y]
        vel (numpy.ndarray): 球拍的速度 [vx, vy]
        size (numpy.ndarray): 球拍的尺寸 [宽度, 高度]
        speed (float): 球拍的移动速度
        area (list): 游戏区域的尺寸 [宽度, 高度]
    """

    def __init__(self, pos, size, speed, area):
        """初始化球拍实例。

        参数:
            pos (list): 初始位置 [x, y]
            size (list): 球拍尺寸 [宽度, 高度]
            speed (float): 移动速度
            area (list): 游戏区域尺寸 [宽度, 高度]
        """
        self.pos = np.array(pos, dtype=np.float32)
        self.vel = np.zeros(2, dtype=np.float32)
        self.size = np.array(size, dtype=np.float32)
        self.speed = speed
        self.area = area

    def up(self):
        """向上移动球拍。"""
        self.vel[1] = -self.speed

    def down(self):
        """向下移动球拍。"""
        self.vel[1] = self.speed

    def left(self):
        """向左移动球拍。"""
        self.vel[0] = -self.speed

    def right(self):
        """向右移动球拍。"""
        self.vel[0] = self.speed

    def stop_vertical(self):
        """停止垂直方向的移动。"""
        self.vel[1] = 0

    def stop_horizontal(self):
        """停止水平方向的移动。"""
        self.vel[0] = 0

    def update(self):
        """更新球拍的位置。

        根据当前速度更新位置，并确保球拍不会移出游戏区域。
        """
        # 更新位置
        self.pos += self.vel

        # 确保球拍不会移出游戏区域
        # 水平方向
        if self.pos[0] < 0:
            self.pos[0] = 0
        elif self.pos[0] + self.size[0] > self.area[0]:
            self.pos[0] = self.area[0] - self.size[0]

        # 垂直方向
        if self.pos[1] < 0:
            self.pos[1] = 0
        elif self.pos[1] + self.size[1] > self.area[1]:
            self.pos[1] = self.area[1] - self.size[1]
