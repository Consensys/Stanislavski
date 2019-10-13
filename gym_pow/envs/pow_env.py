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
            self.max_unsent_blocks = 10
            low = np.array([0, 0, 1])
            high = np.array([self.max_unsent_blocks, self.max_unsent_blocks, 3])
            self.observation_space = spaces.Box(low, high, dtype=np.int32)
            self.action_space = spaces.Discrete(3)
            self.reset()

    def seed(self, seed):
            self.np_random, seed = seeding.np_random()
            return [seed]

    def step(self, action):
        assert self.action_space.contains(action)

        # action == 0 means do nothing
        if action > 0:
            self.byz.sendMinedBlocks(action)

        reward = self.getReward()

        lastEthReward = self.byz.getReward(500)
        myBlocks = self.byz.countMyBlocks()
        done = myBlocks > 10000

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

    def getReward(self):
        if self.byz.iAmAhead():
            return 1
        return -1

    def getReward4(self):
        if self.byz.getAdvance() > 0:
            return 1.1
        if self.byz.iAmAhead():
            return 1
        return -1

    def getReward3(self):
        if self.byz.iAmAhead():
            return 1 - self.slip
        return -self.slip

    def getReward2(self):
        if self.byz.getAdvance() < self.byz.getSecretBlockSize():
            return -10
        if self.byz.iAmAhead():
            return 1
        return -1

    def getReward1(self):
        if self.byz.iAmAhead() is False:
            newCount = -1
        elif self.byz.getSecretBlockSize() > 0:
            newCount = 1.1
        else:
            newCount = 1
        reward = newCount - self.old_count
        self.old_count = newCount
        return reward - 0.01

    def reset(self):
        self.p = autoclass('net.consensys.wittgenstein.protocols.ethpow.ETHMinerAgent').create(self.slip)
        self.p.init()
        self.byz = self.p.getByzNode()
        self.last_event = self.byz.goNextStep()
        self.state = self.getState()
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
