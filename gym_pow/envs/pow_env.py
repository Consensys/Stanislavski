##Create your own environment https://github.com/openai/gym/blob/master/docs/creating-environments.md
import gym
import jnius_config
import numpy as np
from gym import spaces
from gym.utils import seeding

jnius_config.set_classpath('.', './wittgenstein/wittgenstein-all.jar')
from jnius import autoclass

class PoWEnv(gym.Env):   
    def __init__(self):
        self.block_info = 1000
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
        self.seed()
        # define in reset:
        self.p = None
        self.slip = None
        self.max_block = None
        self.byz = None
        self.last_event = None
        self.state = None
        self.old_count = None
        self.resetSlip(0.4)

    def seed(self):
        self.np_random, seed = seeding.np_random()
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)

        # action == 0 means do nothing
        if action > 0:
            self.byz.sendMinedBlocks(action)

        last_eth_reward = self.byz.getReward(self.block_info)
        my_blocks = self.byz.countMyBlocks()
        done = my_blocks >= self.max_block
        reward = self.getReward(done)

        if done:
            eth_reward = self.byz.getReward()
            ratio = self.byz.getRewardRatio()
            info = {"hp":self.slip, "time":self.p.getTimeInSeconds(),"amount":eth_reward,"ratio":ratio}
        else:
            info = {}
            self.last_event = self.byz.goNextStep()
            self.state = self.getState()

        return self.state, self.last_event, self.p.getTimeInSeconds(), my_blocks, reward, last_eth_reward, done, info

    def getState(self):
        distance = self.byz.getAdvance()
        secret_distance = self.byz.getSecretAdvance()
        return secret_distance, secret_distance, self.last_event

    def getReward(self, done):
        a = self.byz.getAdvance()
        if a > 0:
            return a
        return self.byz.getLag()

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
        return self.state

    def render(self):
        print('\n',self.byz.minedToSend)
