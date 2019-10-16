##Create your own environment https://github.com/openai/gym/blob/master/docs/creating-environments.md
import gym
import jnius_config
import numpy as np
from gym import spaces
from gym.utils import seeding

jnius_config.set_classpath('.', './wittgenstein/wittgenstein-all.jar')
from jnius import autoclass
# Possible Observations miner, hashrate ratio, revenue ratio, revenue, uncle rate, total revenue, avg difficulty

class PoWEnv(gym.Env):   
    def __init__(self):
        self.slip = 0.4  # probability of 'finding' a valid block
        self.max_block = 0
        '''Action Space represents the actions you can take go forwards on the main chain, 
        go to the side and create a fork, or do nothing
        Actions
            0 hold 1 block
            1 publish 2 blocks in a row
            2 publish 3 blocks in a row
            3 add 1 block to private chain
        '''
        #represents number of blocks you can go forward into on the main chain
        self.max_unsent_blocks = 10
        low = np.array([0, 0, 1])
        high = np.array([self.max_unsent_blocks, self.max_unsent_blocks, 3])
        self.observation_space = spaces.Box(low, high, dtype=np.int32)
        self.observation_space_size = (self.max_unsent_blocks + 1) * (self.max_unsent_blocks +1 ) * 3
        self.action_space = spaces.Discrete(3)
        self.reset()

    def seed(self):
        self.np_random, seed = seeding.np_random()
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)

        # action == 0 means do nothing
        if action > 0:
            self.byz.sendMinedBlocks(action)

        lastEthReward = self.byz.getReward(500)
        myBlocks = self.byz.countMyBlocks()
        done = myBlocks >= self.max_block
        reward = self.getReward(done)

        if done:
            eth_reward = self.byz.getReward()
            ratio = self.byz.getRewardRatio()
            info = {"hp":self.slip, "time":self.p.getTimeInSeconds(),"amount":eth_reward,"ratio":ratio}
        else:
            info = {}
            self.last_event = self.byz.goNextStep()
            self.state = self.getState()

        return np.array(self.state), self.last_event, self.p.getTimeInSeconds(), myBlocks, reward, lastEthReward, done, info

    def getState(self):
        distance = self.byz.getAdvance()
        secretHeight = self.byz.getSecretBlockSize()
        return (distance, secretHeight, self.last_event)


    def getReward(self, done):
        a = self.byz.getAdvance()
        if a > 0:
            return a
        return self.byz.getTheyInRaw()

    def getReward6(self, done):
        if done:
            return self.byz.getReward()
        return 0

    def getReward5(self, done):
        return 1 if self.byz.iAmAhead() else -1

    def getReward4(self, done):
        if self.byz.getAdvance() > 0:
            return 1.1
        if self.byz.iAmAhead():
            return 1
        return -1

    def getReward3(self, done):
        if self.byz.iAmAhead():
            return 1 - self.slip
        return -self.slip

    def getReward2(self, done):
        if self.byz.getAdvance() < self.byz.getSecretBlockSize():
            return -10
        if self.byz.iAmAhead():
            return 1
        return -1

    def getReward1(self, done):
        if self.byz.iAmAhead() is False:
            newCount = -1
        elif self.byz.getSecretBlockSize() > 0:
            newCount = 1.1
        else:
            newCount = 1
        reward = newCount - self.old_count
        self.old_count = newCount
        return reward - 0.01

    def resetSlip(self, _slip):
        self.slip = _slip
        self.max_block = 20000 * self.slip
        self.reset()

    def reset(self):
        self.p = autoclass('net.consensys.wittgenstein.protocols.ethpow.ETHMinerAgent').create(self.slip)
        self.p.init()
        self.byz = self.p.getByzNode()
        self.last_event = self.byz.goNextStep()
        self.state = self.getState()
        self.old_count = 0
        self.seed()
        return np.array(self.state)

    def render(self):
        print('\n',self.byz.minedToSend)
