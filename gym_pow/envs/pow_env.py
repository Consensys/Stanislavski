##Create your own environment https://github.com/openai/gym/blob/master/docs/creating-environments.md
import random
import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import jnius_config
jnius_config.set_classpath('.', './wittgenstein/wittgenstein-all.jar')
from jnius import autoclass
# Possible Observations miner, hashrate ratio, revenue ratio, revenue, uncle rate, total revenue, avg difficulty

class PoWEnv(gym.Env):   
    def __init__(self, slip=.4):
            self.slip = slip  # probability of 'finding' a valid block
            '''Action Space represents the actions you can take go forwards on the main chain, 
            go to the side and create a fork, or do nothing
            Actions
                0 hold 1 block
                1 publish 2 blocks in a row
                2 publish 3 blocks in a row
                3 add 1 block to private chain
                if you have already 2 blocks in a secret chain and you mine a new one you will automatically publish
            '''
            #represents number of blocks you can go forward into on the main chain
            self.min_distance = 0
            self.max_distance = 10
            self.max_secret_chain = 10
            self.low = np.array([self.min_distance,self.max_distance ])
            self.high = np.array([self.max_distance, self.max_secret_chain])
            self.observation_space = spaces.Box(self.low, self.high, dtype=np.int32)
            self.reset()

    def seed(self, seed):
            self.np_random, seed = seeding.np_random(seed)
            return [seed]

    def step(self, action, episode, epsilon):

        assert self.action_space.contains(action)

        self.byz.goNextStep()
        secretHeight = self.byz.getSecretBlockSize()

        done = False
        if self.byz.countMyBlocks()>=1000:
            done = True
        elif secretHeight >= 10: #ok, so we ask for an action but we actually ignore it. Should this be moved to wittgenstein?
            self.byz.sendMinedBlocks(1)
        elif action ==0:
            self.byz.sendMinedBlocks(0)
        elif action ==1:
            self.byz.sendMinedBlocks(1)
        elif action == 2:
            self.byz.sendMinedBlocks(2)
        elif action == 3:
            self.byz.sendMinedBlocks(3)

        distance = self.byz.getAdvance()
        secretHeight = self.byz.getSecretBlockSize()
        self.state = (distance,secretHeight)
        reward = self.getReward()

        if done:
            eth_reward = self.byz.getReward()
            ratio = self.byz.getRewardRatio()
            info = {"hp":self.slip, "msg":"last step","time":self.p.getTimeInSeconds(),"amount":eth_reward,"ratio":ratio}
        else:
            info = {"hp":self.slip, "msg":"valid","time":self.p.getTimeInSeconds(),"amount":0}

        return np.array(self.state), reward, done, info

# Should return 4 values, an Object, a float, boolean, dict

    def getReward(self):
        if self.byz.iamAhead() is False:
            newCount = -1
        elif self.byz.getSecretBlockSize() > 0:
            newCount = 1.1
        else:
            newCount = 1
        reward = newCount - self.old_count
        self.old_count = newCount
        return reward - 0.01

    def validAction(self, action,secretHeight):
        if secretHeight>= action:
            return True

        return False

    def reset(self):
        self.action_space = spaces.Discrete(4)
        self.p = autoclass('net.consensys.wittgenstein.protocols.ethpow.ETHMinerAgent').create(self.slip)
        self.p.init()
        self.byz = self.p.getByzNode()
        self.state = np.array((0,0))
        self.old_count = 0
        self.seed(1)
        return np.array(self.state)

    def render(self):
        print('\n',self.byz.minedToSend)

    def get_hashPower(self,x):
        return self.p.avgDifficulty(k)

    def get_height(self):
        h = [self.MAX_HEIGHT -self.curr_step]
        return h

    def actionSpaceSample(self):
        return np.random.choice(self.action_space)

    def maxAction(self,Q,state,actions):
        values = np.array(Q[state,a] for a in actions)
        action = np.argmax(values)
        return actions[action]
