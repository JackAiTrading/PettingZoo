"""
简单加密环境。

这个环境模拟了一个多智能体加密通信场景，包括发送者、接收者和对手，
通过加密和解密来实现安全通信，同时防止对手截获信息。

主要特点：
1. 多方博弈
2. 信息加密
3. 通信对抗
4. 策略学习

环境规则：
1. 基本设置
   - 智能体角色
   - 通信信道
   - 加密方式
   - 得分规则

2. 交互规则
   - 消息发送
   - 加密解密
   - 信息截获
   - 结果评估

3. 智能体行为
   - 信息处理
   - 加密选择
   - 对抗策略
   - 通信优化

4. 终止条件
   - 通信完成
   - 信息泄露
   - 时间耗尽
   - 目标达成

环境参数：
- 观察空间：通信状态
- 动作空间：加密和通信选择
- 奖励：通信成功和安全性
- 最大步数：由通信轮数决定

环境特色：
1. 通信系统
   - 消息编码
   - 加密机制
   - 信道模型
   - 噪声干扰

2. 加密机制
   - 密钥生成
   - 加密算法
   - 解密过程
   - 安全验证

3. 策略元素
   - 密钥选择
   - 通信时机
   - 对抗防护
   - 效率优化

4. 评估系统
   - 通信成功率
   - 信息安全性
   - 通信效率
   - 整体表现

注意事项：
- 密钥管理
- 通信策略
- 安全防护
- 效率优化
"""

# noqa: D212, D415
"""
# Simple Crypto

```{figure} mpe_simple_crypto.gif
:width: 140px
:name: simple_crypto
```

This environment is part of the <a href='..'>MPE environments</a>. Please read that page first for general information.

| Import             | `from pettingzoo.mpe import simple_crypto_v3` |
|--------------------|-----------------------------------------------|
| Actions            | Discrete/Continuous                           |
| Parallel API       | Yes                                           |
| Manual Control     | No                                            |
| Agents             | `agents= [eve_0, bob_0, alice_0]`             |
| Agents             | 2                                             |
| Action Shape       | (4)                                           |
| Action Values      | Discrete(4)/Box(0.0, 1.0, (4))                |
| Observation Shape  | (4),(8)                                       |
| Observation Values | (-inf,inf)                                    |
| State Shape        | (20,)                                         |
| State Values       | (-inf,inf)                                    |


In this environment, there are 2 good agents (Alice and Bob) and 1 adversary (Eve). Alice must sent a private 1 bit message to Bob over a public channel. Alice and Bob are rewarded +2 if Bob reconstructs the message, but are rewarded -2 if Eve reconstruct the message (that adds to 0 if both teams
reconstruct the bit). Eve is rewarded -2 based if it cannot reconstruct the signal, zero if it can. Alice and Bob have a private key (randomly generated at beginning of each episode) which they must learn to use to encrypt the message.


Alice observation space: `[message, private_key]`

Bob observation space: `[private_key, alices_comm]`

Eve observation space: `[alices_comm]`

Alice action space: `[say_0, say_1, say_2, say_3]`

Bob action space: `[say_0, say_1, say_2, say_3]`

Eve action space: `[say_0, say_1, say_2, say_3]`

For Bob and Eve, their communication is checked to be the 1 bit of information that Alice is trying to convey.

### Arguments

``` python
simple_crypto_v3.env(max_cycles=25, continuous_actions=False, dynamic_rescaling=False)
```



`max_cycles`:  number of frames (a step for each agent) until game terminates

`continuous_actions`: Whether agent action spaces are discrete(default) or continuous

`dynamic_rescaling`: Whether to rescale the size of agents and landmarks based on the screen size

"""

import numpy as np
from gymnasium.utils import EzPickle

from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
from pettingzoo.utils.conversions import parallel_wrapper_fn

"""Simple crypto environment.

Scenario:
1 speaker, 2 listeners (one of which is an adversary). Good agents rewarded for proximity to goal, and distance from
adversary to goal. Adversary is rewarded for its distance to the goal.
"""


class raw_env(SimpleEnv, EzPickle):
    def __init__(
        self,
        max_cycles=25,
        continuous_actions=False,
        render_mode=None,
        dynamic_rescaling=False,
    ):
        EzPickle.__init__(
            self,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
        )
        scenario = Scenario()
        world = scenario.make_world()
        SimpleEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            dynamic_rescaling=dynamic_rescaling,
        )
        self.metadata["name"] = "simple_crypto_v3"


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)


class CryptoAgent(Agent):
    def __init__(self):
        super().__init__()
        self.key = None


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        num_agents = 3
        num_adversaries = 1
        num_landmarks = 2
        world.dim_c = 4
        # add agents
        world.agents = [CryptoAgent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.adversary = True if i < num_adversaries else False
            agent.collide = False
            agent.speaker = True if i == 2 else False
            agent.movable = False
            base_name = (
                "eve" if agent.adversary else ("alice" if agent.speaker else "bob")
            )
            agent.name = f"{base_name}_0"
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = False
            landmark.movable = False
        return world

    def reset_world(self, world, np_random):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25, 0.25, 0.25])
            if agent.adversary:
                agent.color = np.array([0.75, 0.25, 0.25])
            agent.key = None
        # random properties for landmarks
        color_list = [np.zeros(world.dim_c) for i in world.landmarks]
        for i, color in enumerate(color_list):
            color[i] += 1
        for color, landmark in zip(color_list, world.landmarks):
            landmark.color = color
        # set goal landmark
        goal = np_random.choice(world.landmarks)

        world.agents[1].color = goal.color
        world.agents[2].key = np_random.choice(world.landmarks).color

        for agent in world.agents:
            agent.goal_a = goal

        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        return (agent.state.c, agent.goal_a.color)

    # return all agents that are not adversaries
    def good_listeners(self, world):
        return [
            agent for agent in world.agents if not agent.adversary and not agent.speaker
        ]

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        return (
            self.adversary_reward(agent, world)
            if agent.adversary
            else self.agent_reward(agent, world)
        )

    def agent_reward(self, agent, world):
        # Agents rewarded if Bob can reconstruct message, but adversary (Eve) cannot
        good_listeners = self.good_listeners(world)
        adversaries = self.adversaries(world)
        good_rew = 0
        adv_rew = 0
        for a in good_listeners:
            if (a.state.c == np.zeros(world.dim_c)).all():
                continue
            else:
                good_rew -= np.sum(np.square(a.state.c - agent.goal_a.color))
        for a in adversaries:
            if (a.state.c == np.zeros(world.dim_c)).all():
                continue
            else:
                adv_l1 = np.sum(np.square(a.state.c - agent.goal_a.color))
                adv_rew += adv_l1
        return adv_rew + good_rew

    def adversary_reward(self, agent, world):
        # Adversary (Eve) is rewarded if it can reconstruct original goal
        rew = 0
        if not (agent.state.c == np.zeros(world.dim_c)).all():
            rew -= np.sum(np.square(agent.state.c - agent.goal_a.color))
        return rew

    def observation(self, agent, world):
        # goal color
        goal_color = np.zeros(world.dim_color)
        if agent.goal_a is not None:
            goal_color = agent.goal_a.color

        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        comm = []
        for other in world.agents:
            if other is agent or (other.state.c is None) or not other.speaker:
                continue
            comm.append(other.state.c)

        key = world.agents[2].key

        # prnt = False
        # speaker
        if agent.speaker:
            # if prnt:
            #     print('speaker')
            #     print(agent.state.c)
            #     print(np.concatenate([goal_color] + [key]))
            return np.concatenate([goal_color] + [key])
        # listener
        if not agent.speaker and not agent.adversary:
            # if prnt:
            #     print('listener')
            #     print(agent.state.c)
            #     print(np.concatenate([key] + comm))
            return np.concatenate([key] + comm)
        if not agent.speaker and agent.adversary:
            # if prnt:
            #     print('adversary')
            #     print(agent.state.c)
            #     print(np.concatenate(comm))
            return np.concatenate(comm)
