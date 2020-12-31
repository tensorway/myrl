import gym
import numpy as np
import time
import cv2

def showit(env, act, time_delta=0.005, max_steps=2000, save_file=None, fps=50):
    dons = 0
    observation = env.reset()
    total_rew = 0
    if save_file is not None:
        img = env.render(mode="rgb_array")
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(save_file ,fourcc, fps, (img.shape[1], img.shape[0]))
    for t in range(max_steps):
            if save_file is None:
                env.render()
            else:
                img = env.render(mode="rgb_array")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                out.write(img)
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
    if save_file is not None:
        out.release()
    env.close()
    return total_rew
