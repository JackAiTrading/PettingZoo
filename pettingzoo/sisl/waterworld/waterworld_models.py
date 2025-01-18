import numpy as np
import pygame
import pymunk
from gymnasium import spaces


class Obstacle:
    def __init__(self, x, y, pixel_scale=750, radius=0.1):
        """
        初始化静态障碍物对象。
        
        参数：
            x：x坐标
            y：y坐标
            pixel_scale：像素缩放比例
            radius：障碍物半径
        """
        self.body = pymunk.Body(0, 0, pymunk.Body.STATIC)
        self.body.position = x, y
        self.body.velocity = 0.0, 0.0

        self.shape = pymunk.Circle(self.body, pixel_scale * 0.1)
        self.shape.density = 1
        self.shape.elasticity = 1
        self.shape.custom_value = 1

        self.radius = radius * pixel_scale
        self.color = (120, 176, 178)

    def add(self, space):
        """将障碍物添加到物理空间"""
        space.add(self.body, self.shape)

    def draw(self, display, convert_coordinates):
        """在显示器上绘制障碍物"""
        pygame.draw.circle(
            display, self.color, convert_coordinates(self.body.position), self.radius
        )


class MovingObject:
    def __init__(self, x, y, pixel_scale=750, radius=0.015):
        """
        初始化移动对象的基类。
        
        参数：
            x：x坐标
            y：y坐标
            pixel_scale：像素缩放比例
            radius：对象半径
        """
        self.pixel_scale = 30 * 25
        self.body = pymunk.Body()
        self.body.position = x, y

        self.shape = pymunk.Circle(self.body, pixel_scale * radius)
        self.shape.elasticity = 1
        self.shape.density = 1
        self.shape.custom_value = 1

        self.shape.reset_position = self.reset_position
        self.shape.reset_velocity = self.reset_velocity

        self.radius = radius * pixel_scale

    def add(self, space):
        """将移动对象添加到物理空间"""
        space.add(self.body, self.shape)

    def draw(self, display, convert_coordinates):
        """在显示器上绘制移动对象"""
        pygame.draw.circle(
            display, self.color, convert_coordinates(self.body.position), self.radius
        )

    def reset_position(self, x, y):
        """重置对象位置"""
        self.body.position = x, y

    def reset_velocity(self, vx, vy):
        """重置对象速度"""
        self.body.velocity = vx, vy


class Evaders(MovingObject):
    def __init__(self, x, y, vx, vy, radius=0.03, collision_type=2, max_speed=100):
        """
        初始化逃避者（食物）对象。
        
        参数：
            x：x坐标
            y：y坐标
            vx：x方向速度
            vy：y方向速度
            radius：逃避者半径
            collision_type：碰撞类型
            max_speed：最大速度
        """
        super().__init__(x, y, radius=radius)

        self.body.velocity = vx, vy

        self.color = (145, 250, 116)  # 绿色
        self.shape.collision_type = collision_type
        self.shape.counter = 0
        self.shape.max_speed = max_speed
        self.shape.density = 0.01


class Poisons(MovingObject):
    def __init__(
        self, x, y, vx, vy, radius=0.015 * 3 / 4, collision_type=3, max_speed=100
    ):
        """
        初始化毒物对象。
        
        参数：
            x：x坐标
            y：y坐标
            vx：x方向速度
            vy：y方向速度
            radius：毒物半径
            collision_type：碰撞类型
            max_speed：最大速度
        """
        super().__init__(x, y, radius=radius)

        self.body.velocity = vx, vy

        self.color = (238, 116, 106)  # 红色
        self.shape.collision_type = collision_type
        self.shape.max_speed = max_speed


class Pursuers(MovingObject):
    def __init__(
        self,
        x,
        y,
        max_accel,
        pursuer_speed,
        radius=0.015,
        n_sensors=30,
        sensor_range=0.2,
        collision_type=1,
        speed_features=True,
    ):
        """
        初始化追捕者（智能体）对象。
        
        参数：
            x：x坐标
            y：y坐标
            max_accel：最大加速度
            pursuer_speed：追捕者速度
            radius：追捕者半径
            n_sensors：传感器数量
            sensor_range：传感器范围
            collision_type：碰撞类型
            speed_features：是否启用速度特征
        """
        super().__init__(x, y, radius=radius)

        self.color = (101, 104, 249)  # 蓝色
        self.shape.collision_type = collision_type
        self.sensor_color = (0, 0, 0)  # 黑色
        self.n_sensors = n_sensors
        self.sensor_range = sensor_range * self.pixel_scale
        self.max_accel = max_accel
        self.max_speed = pursuer_speed
        self.body.velocity = 0.0, 0.0

        self.shape.food_indicator = 0  # 如果这一步捕获了食物则为1，否则为0
        self.shape.food_touched_indicator = 0  # 如果这一步接触了食物则为1，否则为0
        self.shape.poison_indicator = 0  # 如果这一步中毒则为1，否则为0

        # 生成self.n_sensors个角度，从0到2pi均匀分布
        # 我们生成多一个角度并移除它，因为linspace[0] = 0 = 2pi = linspace[-1]
        angles = np.linspace(0.0, 2.0 * np.pi, self.n_sensors + 1)[:-1]

        # 将角度转换为x-y坐标
        sensor_vectors = np.c_[np.cos(angles), np.sin(angles)]
        self._sensors = sensor_vectors
        self.shape.custom_value = 1

        # 每个传感器的观察坐标数量
        self._sensor_obscoord = 5
        if speed_features:
            self._sensor_obscoord += 3

        self.sensor_obs_coord = self.n_sensors * self._sensor_obscoord
        self.obs_dim = self.sensor_obs_coord + 2  # +2表示是否与食物碰撞和是否与毒物碰撞

    @property
    def observation_space(self):
        """返回观察空间"""
        return spaces.Box(
            low=np.float32(-2 * np.sqrt(2)),
            high=np.float32(2 * np.sqrt(2)),
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

    @property
    def action_space(self):
        """返回动作空间"""
        return spaces.Box(
            low=np.float32(-self.max_accel),
            high=np.float32(self.max_accel),
            shape=(2,),
            dtype=np.float32,
        )

    @property
    def position(self):
        """返回当前位置"""
        assert self.body.position is not None
        return np.array([self.body.position[0], self.body.position[1]])

    @property
    def velocity(self):
        """返回当前速度"""
        assert self.body.velocity is not None
        return np.array([self.body.velocity[0], self.body.velocity[1]])

    @property
    def sensors(self):
        """返回传感器配置"""
        assert self._sensors is not None
        return self._sensors

    def draw(self, display, convert_coordinates):
        """在显示器上绘制追捕者及其传感器"""
        self.center = convert_coordinates(self.body.position)
        for sensor in self._sensors:
            start = self.center
            end = self.center + self.sensor_range * sensor
            pygame.draw.line(display, self.sensor_color, start, end, 1)

        pygame.draw.circle(display, self.color, self.center, self.radius)

    def get_sensor_barrier_readings(self):
        """获取到障碍物的距离。

        参见 https://github.com/BolunDai0216/WaterworldRevamp 
        获取详细解释。
        """
        # 获取每个传感器的端点位置
        sensor_vectors = self._sensors * self.sensor_range
        position_vec = np.array([self.body.position.x, self.body.position.y])
        sensor_endpoints = position_vec + sensor_vectors

        # 在环境的障碍物上裁剪传感器线。
        # 注意，任何被裁剪的向量可能与原始传感器的角度不同
        clipped_endpoints = np.clip(sensor_endpoints, 0.0, self.pixel_scale)

        # 提取裁剪后的传感器向量
        clipped_vectors = clipped_endpoints - position_vec

        # 找到裁剪后的传感器向量与原始传感器向量的比率
        # 用这个比率缩放向量将使向量的末端限制在障碍物处
        ratios = np.divide(
            clipped_vectors,
            sensor_vectors,
            out=np.ones_like(clipped_vectors),
            where=sensor_vectors != 0,
        )

        # 取每个传感器的最小比率，这将给出到最近障碍物的距离
        min_ratios = np.min(ratios, axis=1)

        return min_ratios

    def get_sensor_reading(
        self, object_coord, object_radius, object_velocity, object_max_velocity
    ):
        """获取到另一个对象（障碍物、追捕者、逃避者、毒物）的距离和速度。"""
        # 获取追捕者的位置和速度
        self.center = self.body.position
        _velocity = self.body.velocity

        # 在局部坐标系中获取对象的距离作为2x1的numpy数组
        distance_vec = np.array(
            [[object_coord[0] - self.center[0]], [object_coord[1] - self.center[1]]]
        )
        distance_squared = np.sum(distance_vec**2)

        # 获取相对速度作为2x1的numpy数组
        relative_speed = np.array(
            [
                [object_velocity[0] - _velocity[0]],
                [object_velocity[1] - _velocity[1]],
            ]
        )

        # 将距离投影到传感器向量上
        sensor_distances = self._sensors @ distance_vec

        # 将速度向量投影到传感器向量上
        sensor_velocities = (
            self._sensors @ relative_speed / (object_max_velocity + self.max_speed)
        )

        # 检查有效检测标准
        wrong_direction_idx = sensor_distances < 0
        out_of_range_idx = sensor_distances - object_radius > self.sensor_range
        no_intersection_idx = (
            distance_squared - sensor_distances**2 > object_radius**2
        )
        not_sensed_idx = wrong_direction_idx | out_of_range_idx | no_intersection_idx

        # 将未感知到的传感器位置读数设置为传感器范围
        sensor_distances = np.clip(sensor_distances / self.sensor_range, 0, 1)
        sensor_distances[not_sensed_idx] = 1.0

        # 将未感知到的传感器速度读数设置为零
        sensor_velocities[not_sensed_idx] = 0.0

        return sensor_distances, sensor_velocities
