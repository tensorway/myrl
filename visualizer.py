import gym
import numpy as np
import time

def showit(env, act, time_delta=0.005, max_steps=2000):
    dons = 0
    observation = env.reset()
    total_rew = 0
    for t in range(max_steps):
            env.render()
            obser = np.expand_dims(observation, axis=0)
            action, _= act(obser)
            #print(action, _)
            if type(env.action_space) is gym.spaces.box.Box:
                observation, reward, done, info = env.step(action.squeeze(0))
            else:
                observation, reward, done, info = env.step(action.squeeze(0).squeeze(-1))
            dons += done
            total_rew += reward
            if done:
                env.close()
                break
            time.sleep(time_delta)
            print(t, "/"+str(max_steps), end='\r')
    env.close()
    return total_rew
    
