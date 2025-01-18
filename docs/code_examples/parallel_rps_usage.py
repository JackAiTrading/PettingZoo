from . import parallel_rps

env = parallel_rps.parallel_env(render_mode="human")
observations, infos = env.reset()

while env.agents:
    # 这里是你插入策略的地方
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}

    observations, rewards, terminations, truncations, infos = env.step(actions)
env.close()
