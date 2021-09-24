agent.load('dqn-agent')

with agent.eval_mode():
    for i in range(3):
        observation = env.reset()
        Return = 0
        time_step = 0
        while True:
            action = agent.act(observation)
            observation, reward, done, _  = env.step(action)
            Return += reward
            time_step += 1
            reset = t == 200
            done = 0
            agent.observe(obs, r, done, reset)
            if done or reset:
                break
        print('evaluation episode:', i, 'R:', R)