import numpy as np

#################################################################
# 实现2D问题的合作智能体层
#################################################################


class AgentLayer:
    def __init__(self, xs, ys, allies, seed=1):
        """初始化AgentLayer类。

        参数：
            xs：地图的x尺寸
            ys：地图的y尺寸
            allies：盟友智能体列表
            seed：随机种子

        每个盟友智能体必须支持以下方法：
        - move(action)：移动
        - current_position()：当前位置
        - nactions()：动作数量
        - set_position(x, y)：设置位置
        """
        self.allies = allies
        self.nagents = len(allies)
        self.global_state = np.zeros((xs, ys), dtype=np.int32)

    def n_agents(self):
        """返回智能体数量"""
        return self.nagents

    def move_agent(self, agent_idx, action):
        """移动指定的智能体"""
        return self.allies[agent_idx].step(action)

    def set_position(self, agent_idx, x, y):
        """设置指定智能体的位置"""
        self.allies[agent_idx].set_position(x, y)

    def get_position(self, agent_idx):
        """返回指定智能体的位置"""
        return self.allies[agent_idx].current_position()

    def get_nactions(self, agent_idx):
        """返回指定智能体的动作数量"""
        return self.allies[agent_idx].nactions()

    def remove_agent(self, agent_idx):
        """移除指定的智能体
        
        参数：
            agent_idx：智能体索引（介于0和nagents之间）
        """
        self.allies.pop(agent_idx)
        self.nagents -= 1

    def get_state_matrix(self):
        """返回表示所有盟友位置的矩阵。

        示例：矩阵包含给定(x,y)位置的盟友数量
        0 0 0 1 0 0 0
        0 2 0 2 0 0 0
        0 0 0 0 0 0 1
        1 0 0 0 0 0 5
        """
        gs = self.global_state
        gs.fill(0)
        for ally in self.allies:
            x, y = ally.current_position()
            gs[x, y] += 1
        return gs

    def get_state(self):
        """返回所有智能体的状态"""
        pos = np.zeros(2 * len(self.allies))
        idx = 0
        for ally in self.allies:
            pos[idx : (idx + 2)] = ally.get_state()
            idx += 2
        return pos
