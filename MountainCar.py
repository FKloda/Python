import random

import gym
import numpy as np
import math
import utils
import matplotlib as plt


class StateDiscretizer:
    # predani rozmeru prostredi a spojitych stavu a jejich rozdeleni na diskretni intervaly
    # rozmer prostredi - 1,8 * 0,14 ?
    def __init__(self, ranges, states):
        self.num = (abs(ranges[0]) + abs(ranges[3])) / states

    # prirazeni stavu do spravneho intervalu
    def transform(self, obs):
        return math.floor(obs[0] / self.num)


class QLearningAgent:
    # nastaveni moznych akci - L, N, R
    # diskretizace stavu prostredi
    # definice matice uzitku Q[stavy, akce]
    # promenna na zapamatovani si minuleho stavu a minule akce
    # donastaveni dalsich parametru trenovani
    def __init__(self, actions, state_transformer, train=True):
        self.actions = actions
        self.Q = np.zeros((3, 16), dtype=int)
        self.Train = train
        self.transformer = state_transformer
        self.currAction
        self.currState

    # na zaklade stavu a akce se vybira nova akce
    # 1. najde se nejlepsi akce pro dany stav
    # 2. s malou pravd. vezme nahodnou
    # 3. updatuje se Q matice
    def act(self, observe, reward, done):
        state = transformer.transform(observe)
        rnd = random.randint(0,100)
        if (rnd <= 10):
            rnd = random.randint(0,2)



    # reset minuleho stavu a akce na konci epizody
    def reset(self):
        self.currstate = self.laststate

env = gym.make('MountainCar-v0')
print('observation space:', env.observation_space)
print('observation space low:', env.observation_space.low)
print('observation space high:', env.observation_space.high)
print('action space:', env.action_space)

obs = env.reset()
print('initial observation:', obs)

"""
action = env.action_space.sample()
obs, r, done, info = env.step(action)
print('next observation:', obs)
print('reward:', r)
print('done:', done)
print('info:', info)
"""

transformer = StateDiscretizer()
agent = QLearningAgent(env.action_space, transformer)
total_rewards = []
for i in range(1000):
    obs = env.reset()
    agent.reset()
    done = False

    r = 0
    R = 0  # celkova odmena - jen pro logovani
    t = 0  # cislo kroku - jen pro logovani

    while not done:
        action = agent.act(obs, r, done)
        obs, r, done, _ = env.step(action)
        R += r
        t += 1

    total_rewards.append(R)
agent.train = False


