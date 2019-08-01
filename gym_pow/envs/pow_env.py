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
    def __init__(self, n=5, slip=.25, empty_block=2, full_block=2.5):
            self.n = n
            self.slip = slip  # probability of 'finding' a valid block
            self.empty_block = empty_block  # payout for 'empty' block, no transactions added
            self.full_block = full_block  # payout for publishing a block with transactions
            self.state = 0  # Start at beginning of the chain
            self.stateSpacePlus = [i for i in range(100)]
            self.action_space = {'Publish':1, 'Hold': 0}
            self.possibleActions = ['Publish','Hold']
            self.head = 0
            self.p = autoclass('net.consensys.wittgenstein.protocols.ethpow.ETHMinerAgent').create(slip)
            self.p.init()
            self.byz= self.p.getByzantineNode()
            self.MAX_HEIGHT = 99 + self.byz.head.height
            self.chain = np.zeros((2,self.MAX_HEIGHT))
            high = np.zeros(self.MAX_HEIGHT,dtype=int)
            low = np.zeros(self.MAX_HEIGHT,dtype=int)
            #self.action_space.n = len(self.action_space)
            self.observation_space =  spaces.Box(low,high, dtype=np.int64)
            self.reward = 0
            self.seed(1)
            print(self.MAX_HEIGHT)
            print("byzantine node ",self.byz)
            self.p.goNextStep()
            self.p.network().printNetworkLatency() 

    def seed(self, seed):
            self.np_random, seed = seeding.np_random(seed)
            return [seed]

    def step(self, action):
            reward = 0
            done = False
            mined = self.byz.mine10ms()
            while (True):
                self.p.goNextStep()
                mined = self.byz.mine10ms()
                if mined is True:
                    self.state =1
                    break
            #print('Height: ',self.byz.head.height)
            #print(mined)
            if self.byz.head.height==self.MAX_HEIGHT:
                print(self.head)
                done = True
            elif mined is True:
                if  self.action_space.get(action) == 'Publish' and mined is True:
                    self.state = 1 
                    reward = self.empty_block
                    self.chain[1][self.head] = 1
                    self.head +=1
                elif self.action_space.get(action) == 'Hold':
                    self.state =0
                    reward = 0
                    self.chain[0][self.head] = 1
                    self.head +=1
                      # agent slipped, reverse action taken
            else:  # 'backwards': go back to the beginning, get empty_block reward
                    #represent second chain i.e. fork
                reward = 0
                self.state = 0
                self.p.goNextStep()#self.chain[]
            
            return self.state, reward, done, {}
# Should return 4 values, an Object, a float, boolean, dict


    def reset(self):
            self.state = 0
            return self.state

    def render(self):
        for chain in self.chain:
            print('\n')
            for block in chain:
                print('-b',block)
            

    def get_hashPower(self,x):
        return self.p.avgDifficulty(k)

    def get_height(self):
        h = [self.MAX_HEIGHT -self.curr_step]
        return h

    def actionSpaceSample(self):
        return np.random.choice(self.possibleActions)

    def maxAction(self,Q,state,actions):
        values = np.array(Q[state,a] for a in actions)
        action = np.argmax(values)
        return actions[action]
 