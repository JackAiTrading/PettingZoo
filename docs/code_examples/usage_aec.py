from pettingzoo.atari import space_invaders_v2

env = space_invaders_v2.env(render_mode="human")
env.reset(seed=42)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        # 这里是你插入策略的地方
        action = env.action_space(agent).sample()

    env.step(action)
env.close()
