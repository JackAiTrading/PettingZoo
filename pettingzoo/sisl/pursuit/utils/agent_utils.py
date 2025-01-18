import numpy as np

from pettingzoo.sisl.pursuit.utils.discrete_agent import DiscreteAgent

#################################################################
# 实现多智能体深度强化学习的工具函数
#################################################################


def create_agents(
    nagents,
    map_matrix,
    obs_range,
    randomizer,
    flatten=False,
    randinit=False,
    constraints=None,
):
    """在地图（map_matrix）上初始化智能体。

    参数：
        -nagents：要放置在地图上的智能体数量
        -randinit：如果为True，将智能体放置在随机的可行位置
                  如果为False，将所有智能体放置在(0,0)
        expanded_mat：此矩阵用于生成非相邻的智能体
    """
    xs, ys = map_matrix.shape
    agents = []
    expanded_mat = np.zeros((xs + 2, ys + 2))
    for i in range(nagents):
        xinit, yinit = (0, 0)
        if randinit:
            xinit, yinit = feasible_position_exp(
                randomizer, map_matrix, expanded_mat, constraints=constraints
            )
            # 填充expanded_mat
            expanded_mat[xinit + 1, yinit + 1] = -1
            expanded_mat[xinit + 2, yinit + 1] = -1
            expanded_mat[xinit, yinit + 1] = -1
            expanded_mat[xinit + 1, yinit + 2] = -1
            expanded_mat[xinit + 1, yinit] = -1
        agent = DiscreteAgent(
            xs, ys, map_matrix, randomizer, obs_range=obs_range, flatten=flatten
        )
        agent.set_position(xinit, yinit)
        agents.append(agent)
    return agents


def feasible_position_exp(randomizer, map_matrix, expanded_mat, constraints=None):
    """返回地图（map_matrix）上的一个可行位置。"""
    xs, ys = map_matrix.shape
    while True:
        if constraints is None:
            x = randomizer.integers(0, xs)
            y = randomizer.integers(0, ys)
        else:
            xl, xu = constraints[0]
            yl, yu = constraints[1]
            x = randomizer.integers(xl, xu)
            y = randomizer.integers(yl, yu)
        if map_matrix[x, y] != -1 and expanded_mat[x + 1, y + 1] != -1:
            return (x, y)


def set_agents(agent_matrix, map_matrix):
    """设置智能体

    参数：
        agent_matrix：智能体矩阵
        map_matrix：地图矩阵
    """
    # 检查输入尺寸
    if agent_matrix.shape != map_matrix.shape:
        raise ValueError("智能体配置和地图矩阵的尺寸不匹配")

    agents = []
    xs, ys = agent_matrix.shape
    for i in range(xs):
        for j in range(ys):
            n_agents = agent_matrix[i, j]
            if n_agents > 0:
                if map_matrix[i, j] == -1:
                    raise ValueError(
                        "试图将智能体放置在建筑物中：请检查地图矩阵和智能体配置"
                    )
                agent = DiscreteAgent(xs, ys, map_matrix)
                agent.set_position(i, j)
                agents.append(agent)
    return agents
