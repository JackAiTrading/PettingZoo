# AgileRL：实现 DQN - 课程学习和自我对弈

```{eval-rst}
.. figure:: connect_four_self_opp.gif
   :align: center
   :height: 400px

   图1：通过自我对弈训练的四子棋智能体
```

本教程展示如何在 [connect four](https://pettingzoo.farama.org/environments/classic/connect_four/) 经典环境中训练 [DQN](https://agilerl.readthedocs.io/en/latest/api/algorithms/dqn.html) 智能体。

本教程重点介绍强化学习中的两种技术 - **课程学习**和**自我对弈**。课程学习是指在不同"课程"中通过逐渐增加难度的任务来训练智能体。想象一下你想成为国际象棋世界冠军。你不会一开始就决定通过与特级大师对弈来学习国际象棋 - 那太困难了。相反，你会与和你水平相当的人对弈，慢慢提高，然后逐渐与更强的对手对弈，直到你准备好与最优秀的选手竞争。同样的概念也适用于强化学习模型。有时，任务太难以一次性学习，所以我们必须创建一个课程来指导智能体，教它解决我们最终的困难环境。

本教程还使用了自我对弈。自我对弈是竞争性强化学习环境中使用的一种技术。智能体通过与自己的副本（对手）对弈来训练，并学会击败这个对手。然后对手会更新为这个更好版本的智能体的副本，智能体必须再次学会击败自己。这个过程不断重复，智能体通过利用自己的弱点和发现新策略来逐步改进。

在本教程中，自我对弈被视为课程中的最后一课。然而，这两种技术可以独立使用，而且在资源无限的情况下，自我对弈可以击败通过人工设计课程进行课程学习训练的智能体。Richard Sutton 的 [The Bitter Lesson](http://incompleteideas.net/IncIdeas/BitterLesson.html) 对课程学习提供了一个有趣的观点，任何从事这项任务的工程师都值得考虑。然而，与 Sutton 不同，我们并不都拥有 Deepmind 和顶级机构提供的资源，因此在决定如何解决自己的强化学习问题时必须务实。如果你想进一步讨论这个令人兴奋的研究领域，请加入 AgileRL 的 [Discord 服务器](https://discord.com/invite/eB8HyTA2ux)，让我们知道你的想法！


## 什么是 DQN？
[DQN](https://agilerl.readthedocs.io/en/latest/api/algorithms/dqn.html)（深度 Q 网络）是 Q 学习的扩展，它使用经验回放缓冲区和目标网络来提高学习稳定性。要了解更多关于 DQN 的信息，请查看 AgileRL [文档](https://agilerl.readthedocs.io/en/latest/api/algorithms/dqn.html)。

### 我可以使用它吗？

|   | 动作空间 | 观察空间 |
|---|--------------|-------------------|
|离散  | ✔️           | ✔️                |
|连续   | ❌           | ✔️                |


## 环境设置

要学习本教程，你需要安装下面显示的依赖项。建议使用新创建的虚拟环境以避免依赖冲突。
```{eval-rst}
.. literalinclude:: ../../../tutorials/AgileRL/requirements.txt
   :language: text
```

## 代码
### 在四子棋游戏中使用 DQN 进行课程学习和自我对弈
以下代码应该可以正常运行。注释旨在帮助你理解如何将 PettingZoo 与 AgileRL 一起使用。如果你有任何问题，请随时在 [Discord 服务器](https://discord.com/invite/eB8HyTA2ux)中询问。

这是一个复杂的教程，所以我们将分阶段进行。[完整代码](#full-training-code)可以在本节末尾找到。虽然本教程的大部分内容都是针对四子棋环境的特定内容，但它展示了如何将这些技术更广泛地应用于其他问题。

### 导入
导入以下包、函数和类将使我们能够运行本教程。

<details>
   <summary>导入</summary>

   ```python
   import copy
   import os
   import random
   from collections import deque
   from datetime import datetime

   import numpy as np
   import torch
   import wandb
   import yaml
   from agilerl.components.replay_buffer import ReplayBuffer
   from agilerl.hpo.mutation import Mutations
   from agilerl.hpo.tournament import TournamentSelection
   from agilerl.utils.utils import initialPopulation
   from tqdm import tqdm, trange

   from pettingzoo.classic import connect_four_v3
   ```
</details>

### 课程学习
首先，我们需要设置和修改我们的环境以启用课程学习。课程学习通过改变智能体训练的环境来实现。这可以通过改变某些动作发生时的情况来实现 - 改变环境返回的下一个观察值，或者更简单地改变奖励。首先，我们将改变奖励。默认情况下，四子棋使用以下奖励：

* 胜利 = +1
* 失败 = -1
* 继续游戏 = 0

为了帮助指导我们的智能体，我们可以为环境中的其他结果引入奖励，例如为连续放置 3 个棋子给予小奖励，或者当对手做到同样的事情时给予小惩罚。我们还可以使用奖励塑形来鼓励我们的智能体进行更多探索。在四子棋中，如果对抗随机对手，一个简单的获胜方法就是总是在同一列放置棋子。智能体可能会通过这种方式取得成功，因此不会学习其他可以帮助它击败更好对手的更复杂策略。因此，我们可以选择对垂直获胜给予稍低的奖励，而对水平或对角线获胜给予更高的奖励，以鼓励智能体尝试不同的获胜方式。一个示例奖励系统可以定义如下：

* 获胜（水平或对角线）= +1
* 获胜（垂直）= +0.8
* 三子连线 = +0.05
* 对手三子连线 = -0.05
* 失败 = -1
* 继续游戏 = 0

#### 配置文件

最好使用 YAML 配置文件来定义我们课程中的课程，并轻松更改和跟踪我们的设置。我们课程中的前三课可以定义如下：

<details>
   <summary>第 1 课</summary>

   ```{eval-rst}
   .. literalinclude:: ../../../tutorials/AgileRL/curriculums/connect_four/lesson1.yaml
      :language: yaml
   ```
</details>

<details>
   <summary>第 2 课</summary>

   ```{eval-rst}
   .. literalinclude:: ../../../tutorials/AgileRL/curriculums/connect_four/lesson2.yaml
      :language: yaml
   ```
</details>

<details>
   <summary>第 3 课</summary>

   ```{eval-rst}
   .. literalinclude:: ../../../tutorials/AgileRL/curriculums/connect_four/lesson3.yaml
      :language: yaml
   ```
</details><br>

要实现我们的课程，我们创建一个 `CurriculumEnv` 类，它作为我们的四子棋环境的包装器，使我们能够改变奖励来指导智能体的训练。这使用我们设置的配置来定义课程。

<details>
   <summary>课程环境</summary>

   ```python
   class CurriculumEnv:
      """课程学习环境的包装器，用于修改奖励。

      :param env: 要学习的环境
      :type env: PettingZoo 风格的环境
      :param lesson: 课程学习的课程设置
      :type lesson: dict
      """

      def __init__(self, env, lesson):
         self.env = env
         self.lesson = lesson
   ```
</details>

在定义课程中的不同课程时，我们可以通过修改智能体的环境观察来增加任务的难度 - 在四子棋中，我们可以提高对手的技能水平。通过逐步进行这个过程，我们可以帮助我们的智能体提高。我们也可以在课程之间改变奖励；例如，一旦我们学会击败随机对手，现在想要对抗更强的对手时，我们可能希望对所有方向的获胜给予相同的奖励。在本教程中，实现了一个 `Opponent` 类来为训练我们的智能体提供不同难度级别。

<details>
   <summary>对手</summary>

   ```python
   class Opponent:
      """四子棋对手，用于训练和/或评估。

      :param env: 要学习的环境
      :type env: PettingZoo 风格的环境
      """
   ```
</details>

### 通用设置

在我们继续本教程之前，定义和设置训练所需的其他所有内容会很有帮助。

<details>
   <summary>设置代码</summary>

   ```python
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   print("===== AgileRL 课程学习演示 =====")

   lesson_number = 1

   # 加载课程
   with open(f"./curriculums/connect_four/lesson{lesson_number}.yaml") as file:
      LESSON = yaml.safe_load(file)

   # 定义网络配置
   NET_CONFIG = {
      "arch": "cnn",  # 网络架构
      "hidden_size": [64, 64],  # Actor 隐藏层大小
      "channel_size": [128],  # CNN 通道大小
      "kernel_size": [4],  # CNN 内核大小
      "stride_size": [1],  # CNN 步幅大小
      "normalize": False,  # 将图像从 [0,255] 范围归一化为 [0,1]
   }

   # 定义初始超参数
   INIT_HP = {
      "POPULATION_SIZE": 6,
      # "ALGO": "Rainbow DQN",  # 算法
      "ALGO": "DQN",  # 算法
      "DOUBLE": True,
      # 交换图像通道维度从最后一个到第一个 [H, W, C] -> [C, H, W]
      "BATCH_SIZE": 256,  # 批大小
      "LR": 1e-4,  # 学习率
      "GAMMA": 0.99,  # 折扣因子
      "MEMORY_SIZE": 100000,  # 最大内存缓冲区大小
      "LEARN_STEP": 1,  # 学习频率
      "N_STEP": 1,  # 计算 TD 错误的步数
      "PER": False,  # 使用优先经验回放缓冲区
      "ALPHA": 0.6,  # 优先回放缓冲区参数
      "TAU": 0.01,  # 软更新目标参数
      "BETA": 0.4,  # 重要性采样系数
      "PRIOR_EPS": 0.000001,  # 最小优先级
      "NUM_ATOMS": 51,  # 支持单元数
      "V_MIN": 0.0,  # 支持最小值
      "V_MAX": 200.0,  # 支持最大值
      "WANDB": False,  # 使用 Weights and Biases 跟踪
   }

   # 定义四子棋环境
   env = connect_four_v3.env()
   env.reset()

   # 配置算法输入参数
   state_dim = [
      env.observation_space(agent)["observation"].shape for agent in env.agents
   ]
   one_hot = False
   action_dim = [env.action_space(agent).n for agent in env.agents]
   INIT_HP["DISCRETE_ACTIONS"] = True
   INIT_HP["MAX_ACTION"] = None
   INIT_HP["MIN_ACTION"] = None

   # 将环境包装在课程学习包装器中
   env = CurriculumEnv(env, LESSON)

   # 为 PyTorch 层预处理维度
   # 我们只需要担心单个智能体的状态维度
   # 我们将 6x7x2 观察值平展为智能体神经网络的输入
   state_dim = np.moveaxis(np.zeros(state_dim[0]), [-1], [-3]).shape
   action_dim = action_dim[0]

   # 创建一个准备好进行进化超参数优化的种群
   pop = initialPopulation(
      INIT_HP["ALGO"],
      state_dim,
      action_dim,
      one_hot,
      NET_CONFIG,
      INIT_HP,
      population_size=INIT_HP["POPULATION_SIZE"],
      device=device,
   )

   # 配置回放缓冲区
   field_names = ["state", "action", "reward", "next_state", "done"]
   memory = ReplayBuffer(
      action_dim=action_dim,  # 智能体动作数量
      memory_size=INIT_HP["MEMORY_SIZE"],  # 最大回放缓冲区大小
      field_names=field_names,  # 存储在内存中的字段名称
      device=device,
   )

   # 实例化一个锦标赛选择对象（用于 HPO）
   tournament = TournamentSelection(
      tournament_size=2,  #锦标赛选择大小
      elitism=True,  #锦标赛选择中的精英主义
      population_size=INIT_HP["POPULATION_SIZE"],  #种群大小
      evo_step=1,
   )  # 使用最后 N 个适应度评分进行评估

   # 实例化一个变异对象（用于 HPO）
   mutations = Mutations(
      algo=INIT_HP["ALGO"],
      no_mutation=0.2,  # 没有变异的概率
      architecture=0,  # 架构变异的概率
      new_layer_prob=0.2,  # 新层变异的概率
      parameters=0.2,  # 参数变异的概率
      activation=0,  # 激活函数变异的概率
      rl_hp=0.2,  # RL 超参数变异的概率
      rl_hp_selection=[
            "lr",
            "learn_step",
            "batch_size",
      ],  # 选择用于变异的 RL 超参数
      mutation_sd=0.1,  # 变异强度
      # 定义每个超参数的搜索空间
      min_lr=0.0001,
      max_lr=0.01,
      min_learn_step=1,
      max_learn_step=120,
      min_batch_size=8,
      max_batch_size=64,
      arch=NET_CONFIG["arch"],  # MLP 或 CNN
      rand_seed=1,
      device=device,
   )

   # 定义训练循环参数
   episodes_per_epoch = 10
   max_episodes = LESSON["max_train_episodes"]  # 总训练集
   max_steps = 500  # 每个训练集的最大步数
   evo_epochs = 20  # 进化频率
   evo_loop = 50  # 评估训练集的数量
   elite = pop[0]  # 分配一个占位符“精英”智能体
   epsilon = 1.0  # 初始 epsilon 值
   eps_end = 0.1  # 最终 epsilon 值
   eps_decay = 0.9998  # epsilon 衰减
   opp_update_counter = 0
   wb = INIT_HP["WANDB"]

   ```
</details>

作为课程的一部分，我们也可以选择用随机经验填充回放缓冲区，并对这些经验进行离线训练。

<details>
   <summary>填充回放缓冲区</summary>

   ```python
   # 执行缓冲区和智能体预热
   if LESSON["buffer_warm_up"]:
      warm_up_opponent = Opponent(env, difficulty=LESSON["warm_up_opponent"])
      memory = env.fill_replay_buffer(
            memory, warm_up_opponent
      )  # 用随机转换填充回放缓冲区
      if LESSON["agent_warm_up"] > 0:
            print("预热智能体...")
            agent = pop[0]
            # 训练随机收集的样本
            for epoch in trange(LESSON["agent_warm_up"]):
               experiences = memory.sample(agent.batch_size)
               agent.learn(experiences)
            pop = [agent.clone() for _ in pop]
            elite = agent
            print("智能体种群预热完成。")
   ```
</details>

### 自我对弈

在本教程中，我们使用自我对弈作为课程中的最后一课。通过反复改进我们的智能体并使其学会击败自己，我们可以让它发现新的策略并实现更好的性能。我们可以将预先训练的智能体的权重从以前的课程加载到种群中，如下所示：

<details>
   <summary>加载预训练权重</summary>

   ```python
   if LESSON["pretrained_path"] is not None:
      for agent in pop:
            # 加载预训练检查点
            agent.loadCheckpoint(LESSON["pretrained_path"])
            # 为新任务重新初始化优化器
            agent.lr = INIT_HP["LR"]
            agent.optimizer = torch.optim.Adam(
               agent.actor.parameters(), lr=agent.lr
            )
   ```
</details>

要训练对抗旧版本的智能体，我们创建一个对手池。在训练时，我们随机从池中选择一个对手。定期更新对手池，通过删除最旧的对手并添加最新版本的智能体的副本。这提供了训练对抗越来越困难的对手和提供对手可能采取的动作多样性的平衡。

<details>
   <summary>创建对手池</summary>

   ```python
   if LESSON["opponent"] == "self":
      # 创建初始对手池
      opponent_pool = deque(maxlen=LESSON["opponent_pool_size"])
      for _ in range(LESSON["opponent_pool_size"]):
            opp = copy.deepcopy(pop[0])
            opp.actor.load_state_dict(pop[0].actor.state_dict())
            opp.actor.eval()
            opponent_pool.append(opp)
   ```
</details>

一个示例课程配置可以定义如下：

<details>
   <summary>第 4 课</summary>

   ```{eval-rst}
   .. literalinclude:: ../../../tutorials/AgileRL/curriculums/connect_four/lesson4.yaml
      :language: yaml
   ```
</details>

也可以只通过自我对弈训练智能体，而不使用课程中的任何以前的课程。这需要大量的训练时间，但最终可能会比其他方法产生更好的性能，并且可以避免 [The Bitter Lesson](http://incompleteideas.net/IncIdeas/BitterLesson.html) 中讨论的一些错误。

### 训练循环

四子棋训练循环必须考虑到智能体只在每次与环境交互时采取动作（对手在交替回合中采取动作）。这必须在保存转换到回放缓冲区时考虑。同样，我们必须等待下一个玩家的结果才能确定转换的奖励。这不是一个真正的马尔可夫决策过程，但我们仍然可以在这些非平稳条件下训练强化学习智能体。

定期评估种群中智能体的性能（或“适应度”），并进行进化步骤。表现最好的智能体更有可能成为下一代的成员，种群中智能体的超参数和神经架构会发生变异。这种进化使我们能够在单次训练中优化超参数并最大化我们的智能体的性能。

<details>
   <summary>训练循环</summary>

   ```python
   if max_episodes > 0:
      if wb:
         wandb.init(
               # 设置 wandb 项目
               project="AgileRL",
               name="{}-EvoHPO-{}-{}Opposition-CNN-{}".format(
                  "connect_four_v3",
                  INIT_HP["ALGO"],
                  LESSON["opponent"],
                  datetime.now().strftime("%m%d%Y%H%M%S"),
               ),
               # 跟踪超参数和运行元数据
               config={
                  "algo": "Evo HPO Rainbow DQN",
                  "env": "connect_four_v3",
                  "INIT_HP": INIT_HP,
                  "lesson": LESSON,
               },
         )

   total_steps = 0
   total_episodes = 0
   pbar = trange(int(max_episodes / episodes_per_epoch))

   # 训练循环
   for idx_epi in pbar:
      turns_per_episode = []
      train_actions_hist = [0] * action_dim
      for agent in pop:  # 循环种群
            for episode in range(episodes_per_epoch):
               env.reset()  # 在每个训练集开始时重置环境
               observation, env_reward, done, truncation, _ = env.last()

               (
                  p1_state,
                  p1_state_flipped,
                  p1_action,
                  p1_next_state,
                  p1_next_state_flipped,
               ) = (None, None, None, None, None)

               if LESSON["opponent"] == "self":
                  # 如果使用自我对弈，则从对手池中随机选择对手
                  opponent = random.choice(opponent_pool)
               else:
                  # 创建所需难度的对手
                  opponent = Opponent(env, difficulty=LESSON["opponent"])

               # 随机决定智能体是否先行或后行
               if random.random() > 0.5:
                  opponent_first = False
               else:
                  opponent_first = True

               score = 0
               turns = 0  # 转数计数器

               for idx_step in range(max_steps):
                  # 玩家 0 的回合
                  p0_action_mask = observation["action_mask"]
                  p0_state = np.moveaxis(observation["observation"], [-1], [-3])
                  p0_state_flipped = np.expand_dims(np.flip(p0_state, 2), 0)
                  p0_state = np.expand_dims(p0_state, 0)

                  if opponent_first:
                        if LESSON["opponent"] == "self":
                           p0_action = opponent.getAction(
                              p0_state, 0, p0_action_mask
                           )[0]
                        elif LESSON["opponent"] == "random":
                           p0_action = opponent.getAction(
                              p0_action_mask, p1_action, LESSON["block_vert_coef"]
                           )
                        else:
                           p0_action = opponent.getAction(player=0)
                  else:
                        p0_action = agent.getAction(
                           p0_state, epsilon, p0_action_mask
                        )[
                           0
                        ]  # 获取智能体的下一个动作
                        train_actions_hist[p0_action] += 1

                  env.step(p0_action)  # 在环境中执行动作
                  observation, env_reward, done, truncation, _ = env.last()
                  p0_next_state = np.moveaxis(
                        observation["observation"], [-1], [-3]
                  )
                  p0_next_state_flipped = np.expand_dims(
                        np.flip(p0_next_state, 2), 0
                  )
                  p0_next_state = np.expand_dims(p0_next_state, 0)

                  if not opponent_first:
                        score += env_reward
                  turns += 1

                  # 检查游戏是否结束（玩家 0 获胜）
                  if done or truncation:
                        reward = env.reward(done=True, player=0)
                        memory.save2memoryVectEnvs(
                           np.concatenate(
                              (
                                    p0_state,
                                    p1_state,
                                    p0_state_flipped,
                                    p1_state_flipped,
                              )
                           ),
                           [p0_action, p1_action, 6 - p0_action, 6 - p1_action],
                           [
                              reward,
                              LESSON["rewards"]["lose"],
                              reward,
                              LESSON["rewards"]["lose"],
                           ],
                           np.concatenate(
                              (
                                    p0_next_state,
                                    p1_next_state,
                                    p0_next_state_flipped,
                                    p1_next_state_flipped,
                              )
                           ),
                           [done, done, done, done],
                        )
                  else:  # 游戏继续
                        if p1_state is not None:
                           reward = env.reward(done=False, player=1)
                           memory.save2memoryVectEnvs(
                              np.concatenate((p1_state, p1_state_flipped)),
                              [p1_action, 6 - p1_action],
                              [reward, reward],
                              np.concatenate(
                                    (p1_next_state, p1_next_state_flipped)
                              ),
                              [done, done],
                           )

                        # 玩家 1 的回合
                        p1_action_mask = observation["action_mask"]
                        p1_state = np.moveaxis(
                           observation["observation"], [-1], [-3]
                        )
                        # 交换棋子，使智能体始终从相同的角度看到棋盘
                        p1_state[[0, 1], :, :] = p1_state[[0, 1], :, :]
                        p1_state_flipped = np.expand_dims(np.flip(p1_state, 2), 0)
                        p1_state = np.expand_dims(p1_state, 0)

                        if not opponent_first:
                           if LESSON["opponent"] == "self":
                              p1_action = opponent.getAction(
                                    p1_state, 0, p1_action_mask
                              )[0]
                           elif LESSON["opponent"] == "random":
                              p1_action = opponent.getAction(
                                    p1_action_mask,
                                    p0_action,
                                    LESSON["block_vert_coef"],
                              )
                           else:
                              p1_action = opponent.getAction(player=1)
                        else:
                           p1_action = agent.getAction(
                              p1_state, epsilon, p1_action_mask
                           )[
                              0
                           ]  # 获取智能体的下一个动作
                           train_actions_hist[p1_action] += 1

                        env.step(p1_action)  # 在环境中执行动作
                        observation, env_reward, done, truncation, _ = env.last()
                        p1_next_state = np.moveaxis(
                           observation["observation"], [-1], [-3]
                        )
                        p1_next_state[[0, 1], :, :] = p1_next_state[[0, 1], :, :]
                        p1_next_state_flipped = np.expand_dims(
                           np.flip(p1_next_state, 2), 0
                        )
                        p1_next_state = np.expand_dims(p1_next_state, 0)

                        if opponent_first:
                           score += env_reward
                        turns += 1

                        # 检查游戏是否结束（玩家 1 获胜）
                        if done or truncation:
                           reward = env.reward(done=True, player=1)
                           memory.save2memoryVectEnvs(
                              np.concatenate(
                                    (
                                       p0_state,
                                       p1_state,
                                       p0_state_flipped,
                                       p1_state_flipped,
                                    )
                              ),
                              [
                                    p0_action,
                                    p1_action,
                                    6 - p0_action,
                                    6 - p1_action,
                              ],
                              [
                                    LESSON["rewards"]["lose"],
                                    reward,
                                    LESSON["rewards"]["lose"],
                                    reward,
                              ],
                              np.concatenate(
                                    (
                                       p0_next_state,
                                       p1_next_state,
                                       p0_next_state_flipped,
                                       p1_next_state_flipped,
                                    )
                              ),
                              [done, done, done, done],
                           )

                        else:  # 游戏继续
                           reward = env.reward(done=False, player=0)
                           memory.save2memoryVectEnvs(
                              np.concatenate((p0_state, p0_state_flipped)),
                              [p0_action, 6 - p0_action],
                              [reward, reward],
                              np.concatenate(
                                    (p0_next_state, p0_next_state_flipped)
                              ),
                              [done, done],
                           )

                  # 根据学习频率学习
                  if (memory.counter % agent.learn_step == 0) and (
                        len(memory) >= agent.batch_size
                  ):
                        # 采样回放缓冲区
                        # 根据智能体的 RL 算法学习
                        experiences = memory.sample(agent.batch_size)
                        agent.learn(experiences)

                  # 停止训练集，如果任何智能体终止
                  if done or truncation:
                        break

               total_steps += idx_step + 1
               total_episodes += 1
               turns_per_episode.append(turns)
               # 保存总训练集奖励
               agent.scores.append(score)

               if LESSON["opponent"] == "self":
                  if (total_episodes % LESSON["opponent_upgrade"] == 0) and (
                        (idx_epi + 1) > evo_epochs
                  ):
                        elite_opp, _, _ = tournament._elitism(pop)
                        elite_opp.actor.eval()
                        opponent_pool.append(elite_opp)
                        opp_update_counter += 1

            # 更新 epsilon 以进行探索
            epsilon = max(eps_end, epsilon * eps_decay)

      mean_turns = np.mean(turns_per_episode)

      # 现在进化种群，如果必要
      if (idx_epi + 1) % evo_epochs == 0:
            # 评估种群与随机动作
            fitnesses = []
            win_rates = []
            eval_actions_hist = [0] * action_dim  # 评估动作直方图
            eval_turns = 0  # 评估转数计数器
            for agent in pop:
               with torch.no_grad():
                  rewards = []
                  for i in range(evo_loop):
                        env.reset()  # 在每个训练集开始时重置环境
                        observation, reward, done, truncation, _ = env.last()

                        player = -1  # 跟踪当前玩家

                        # 创建所需难度的对手
                        opponent = Opponent(env, difficulty=LESSON["eval_opponent"])

                        # 随机决定智能体是否先行或后行
                        if random.random() > 0.5:
                           opponent_first = False
                        else:
                           opponent_first = True

                        score = 0

                        for idx_step in range(max_steps):
                           action_mask = observation["action_mask"]
                           if player < 0:
                              if opponent_first:
                                    if LESSON["eval_opponent"] == "random":
                                       action = opponent.getAction(action_mask)
                                    else:
                                       action = opponent.getAction(player=0)
                              else:
                                    state = np.moveaxis(
                                       observation["observation"], [-1], [-3]
                                    )
                                    state = np.expand_dims(state, 0)
                                    action = agent.getAction(state, 0, action_mask)[
                                       0
                                    ]  # 获取智能体的下一个动作
                                    eval_actions_hist[action] += 1
                           if player > 0:
                              if not opponent_first:
                                    if LESSON["eval_opponent"] == "random":
                                       action = opponent.getAction(action_mask)
                                    else:
                                       action = opponent.getAction(player=1)
                              else:
                                    state = np.moveaxis(
                                       observation["observation"], [-1], [-3]
                                    )
                                    state[[0, 1], :, :] = state[[0, 1], :, :]
                                    state = np.expand_dims(state, 0)
                                    action = agent.getAction(state, 0, action_mask)[
                                       0
                                    ]  # 获取智能体的下一个动作
                                    eval_actions_hist[action] += 1

                           env.step(action)  # 在环境中执行动作
                           observation, reward, done, truncation, _ = env.last()

                           if (player > 0 and opponent_first) or (
                              player < 0 and not opponent_first
                           ):
                              score += reward

                           eval_turns += 1

                           if done or truncation:
                              break

                           player *= -1

                        rewards.append(score)
               mean_fit = np.mean(rewards)
               agent.fitness.append(mean_fit)
               fitnesses.append(mean_fit)

            eval_turns = eval_turns / len(pop) / evo_loop

            pbar.set_postfix_str(
               f"    训练集平均奖励：{np.mean(agent.scores[-episodes_per_epoch:])}   训练集平均转数：{mean_turns}   评估集平均适应度：{np.mean(fitnesses)}   评估集最佳适应度：{np.max(fitnesses)}   评估集平均转数：{eval_turns}   总步数：{total_steps}"
            )
            pbar.update(0)

            # 格式化动作直方图以进行可视化
            train_actions_hist = [
               freq / sum(train_actions_hist) for freq in train_actions_hist
            ]
            eval_actions_hist = [
               freq / sum(eval_actions_hist) for freq in eval_actions_hist
            ]
            train_actions_dict = {
               f"train/action_{index}": action
               for index, action in enumerate(train_actions_hist)
            }
            eval_actions_dict = {
               f"eval/action_{index}": action
               for index, action in enumerate(eval_actions_hist)
            }

            if wb:
               wandb_dict = {
                  "global_step": total_steps,
                  "train/mean_score": np.mean(agent.scores[-episodes_per_epoch:]),
                  "train/mean_turns_per_game": mean_turns,
                  "train/epsilon": epsilon,
                  "train/opponent_updates": opp_update_counter,
                  "eval/mean_fitness": np.mean(fitnesses),
                  "eval/best_fitness": np.max(fitnesses),
                  "eval/mean_turns_per_game": eval_turns,
               }
               wandb_dict.update(train_actions_dict)
               wandb_dict.update(eval_actions_dict)
               wandb.log(wandb_dict)

            #锦标赛选择和种群变异
            elite, pop = tournament.select(pop)
            pop = mutations.mutation(pop)

   if max_episodes > 0:
      if wb:
         wandb.finish()

   # 保存训练好的智能体
   save_path = LESSON["save_path"]
   os.makedirs(os.path.dirname(save_path), exist_ok=True)
   elite.saveCheckpoint(save_path)
   print(f"精英智能体保存到 '{save_path}'。")
   ```
</details>

### 训练好的模型权重
训练好的模型权重位于 `PettingZoo/tutorials/AgileRL/models`。看看这些模型，与它们对弈训练，看看你是否能击败它们！

### 观看训练好的智能体对弈
以下代码允许你加载保存的 DQN 智能体，测试其性能，然后将多个训练集可视化为 GIF。

<details>
   <summary>渲染训练好的智能体</summary>

   ```{eval-rst}
   .. literalinclude:: ../../../tutorials/AgileRL/render_agilerl_dqn.py
      :language: python
   ```
</details>

### 完整训练代码

<details>
   <summary>完整训练代码</summary>

   > 请注意，在第 612 行，`max_episodes` 被设置为 10，以允许快速测试本教程代码。这个行可以被删除，下面的行可以取消注释，以使用配置文件中设置的训练集数量。

   ```{eval-rst}
   .. literalinclude:: ../../../tutorials/AgileRL/agilerl_dqn_curriculum.py
      :language: python
   ```
</details>
