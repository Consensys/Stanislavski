##Create your own environment https://github.com/openai/gym/blob/master/docs/creating-environments.md
import random
import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import jnius_config
jnius_config.set_classpath('.', '/Users/vanessabridge/Desktop/Projects/wittgenstein/build/libs/wittgenstein-all.jar')
from jnius import autoclass
# Possible Observations miner, hashrate ratio, revenue ratio, revenue, uncle rate, total revenue, avg difficulty

class PoWEnv(gym.Env):   
    def __init__(self, n=99, slip=.25, empty_block=2, full_block=2.5):
            self.n = n
            self.slip = slip  # probability of 'finding' a valid block
            self.empty_block = empty_block  # payout for 'empty' block, no transactions added
            self.full_block = full_block  # payout for publishing a block with transactions
            '''Action Space represents the actions you can take go forwards on the main chain, 
            go to the side and create a fork, or do nothing
            Actions
                0 Do nothing
                1 publish 1 block
                2 publish 2 blocks in a row
                3 publish 3 blocks in a row
                4 hold 1 block
                5 hold 2 blocks in a row
                6 hold 3 blocks in a row
            '''
            self.action_space = spaces.Discrete(7)
            self.p = autoclass('net.consensys.wittgenstein.protocols.ethpow.ETHMinerAgent').create(self.slip)
            self.p.init()
            self.byz= self.p.getByzantineNode()
            self.MAX_HEIGHT = 99 + self.byz.head.height
            #represents number of blocks you can go forward into on the main chain
            self.observation_space = spaces.Tuple((
                spaces.Discrete(self.byz.head.height),
                spaces.Discrete(self.byz.head.height),
                spaces.Discrete(2)))
            self.head = 0
            self.reward = 0
            self.seed(1)
            print(self.MAX_HEIGHT)
            print("byzantine node ",self.byz)
            self.p.goNextStep()
            self.p.network().printNetworkLatency() 
            self.secrete_blocks = 0

    def seed(self, seed):
            self.np_random, seed = seeding.np_random(seed)
            return [seed]

    def step(self, action):
            assert self.action_space.contains(action)
            reward = 0
            done = False
            mined = self.byz.mine10ms()
            self.miner
            while (True):
                self.p.goNextStep()
                mined = self.byz.mine10ms()
                if mined is True:
                    break
            if self.byz.head.height==self.MAX_HEIGHT:
                print(self.head)
                done = True
            elif mined is True:
                if  action == 0:
                    self.miner.append(start_private_chain())
                    reward = -1
                elif 1<= action and action <= 3:
                    reward = action*self.full_block
                    self.head -=action*1
                elif 4<= action and action <=6:
                    self.miner.append(start_private_chain())
                    self.head +=action*1
                    reward = 0
                self.p.goNextStep()#run so you can publish blocks
            return self._get_obs(), reward, done, {}
# Should return 4 values, an Object, a float, boolean, dict

    def _get_obs(self):
        return (self.head, self.miner,len(self.miner))

    def start_private_chain(self):
        if len(self.miner) == 0:
            self.secrete_blocks = 1
        else:
            self.secrete_blocks +=1
        return self.secrete_blocks

    def validAction(self, state, action):
        return True

    def reset(self):
        self.n = 99
        self.slip = 0.25  # probability of 'finding' a valid block
        self.empty_block = 2  # payout for 'empty' block, no transactions added
        self.full_block = 2.5  # payout for publishing a block with transactions
        self.action_space = spaces.Discrete(7)
        self.p = autoclass('net.consensys.wittgenstein.protocols.ethpow.ETHMinerAgent').create(self.slip)
        self.p.init()
        self.byz= self.p.getByzantineNode()
        self.MAX_HEIGHT = 99 + self.byz.head.height
        self.observation_space = spaces.Tuple((
            spaces.Discrete(self.byz.head.height),
            spaces.Discrete(self.byz.head.height),
            spaces.Discrete(2)))
        self.head = 0
        self.reward = 0
        self.seed(1)
        print(self.MAX_HEIGHT)
        print("byzantine node ",self.byz)
        self.p.goNextStep()
        self.p.network().printNetworkLatency() 
        self.secrete_blocks = 0
        

    def render(self):
        print('\n',self.head)
           

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
 