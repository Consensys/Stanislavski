##Create your own environment https://github.com/openai/gym/blob/master/docs/creating-environments.md
import random
import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import jnius_config
jnius_config.set_classpath('.', '/Users/vanessabridge/Desktop/Projects/gym_pow/wittgenstein/wittgenstein-all.jar')
from jnius import autoclass
# Possible Observations miner, hashrate ratio, revenue ratio, revenue, uncle rate, total revenue, avg difficulty

class PoWEnv(gym.Env):   
    def __init__(self, n=99, slip=.4, empty_block=2, full_block=2.5):
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
                if you have already 2 blocks in a secret chain and you mine a new one you will automatically publish
            '''
            self.action_space = spaces.Discrete(6)
            self.p = autoclass('net.consensys.wittgenstein.protocols.ethpow.ETHMinerAgent').create(self.slip)
            self.p.init()
            self.byz= self.p.getByzantineNode()
            self.MAX_HEIGHT =100
            #represents number of blocks you can go forward into on the main chain
            self.low = 0
            self.high = 3
            self.observation_space = spaces.Discrete(4)
            self.head = 0
            self.reward = 0
            self.seed(1)
            print(self.MAX_HEIGHT)
            print("byzantine node ",self.byz)
            self.p.goNextStep()
            self.p.network().printNetworkLatency() 
            self.miner = [0,0]
            self.curr_step=0
            self.blocks_mined =0

    def seed(self, seed):
            self.np_random, seed = seeding.np_random(seed)
            return [seed]

    def step(self, action):
            assert self.action_space.contains(action)
            reward = 0
            done = False
            mined = self.byz.mine10ms()
            self.curr_step = self.byz.head.height
            #Mine until you have a valid block
            while (True):
                self.p.goNextStep()
                mined = self.byz.mine10ms()
                if mined is True:
                    self.blocks_mined +=1
                    break
            #if self.byz.head.height==self.MAX_HEIGHT:
            if self.blocks_mined>=self.MAX_HEIGHT:
                print('HEAD: ',self.byz.head.height)
                print('MAX HEIGHT: ',self.MAX_HEIGHT)
                done = True    
                self.head = self.byz.head.height           
            elif mined is True:
                #force to publish call something like p.sendALL
                if self.miner[1] == 3:
                    if self.head-self.miner[0]<3:
                        reward = 3* self.full_block
                    else:
                        reward = 1.8
                    return self._get_obs(), reward, done,{}
                if  action == 0:
                    #not sending your block
                    reward= -1
                elif self.validAction(action):
                    if action ==1: 
                        self.miner[1] -=1
                        #call function to store secret blocks
                        reward = 1*self.full_block
                    elif action ==2:
                        self.miner[1]-=2
                        #call function to store secret blocks
                        reward = 2*self.full_block
                    elif action == 3:
                        self.miner[1]-=3
                        #call function 
                    elif action == 4:
                        self.miner[1]=1
                        #add to secret block in wittgenstein
                        #reward remains 0
                    elif action == 5:
                        self.miner[1]=2
                    elif action ==6:
                        self.miner[1]=3 
                else:
                    return self._get_obs(), reward, done, {"invalid action"}
                self.p.goNextStep()#run so you can publish blocks
            return self._get_obs(), reward, done, {}
# Should return 4 values, an Object, a float, boolean, dict

    def _get_obs(self):
        print("HEAD, Type : ", self.head, type(self.head))
        print("Miner Head", self.miner)
        if(type(self.head)==list):

            return np.array([self.miner[1]])
       
        return np.array([self.miner[1]])

    def validAction(self, action):
        # If you want to publish a block you need to ensure you have at least that number in a private chain
        if action <=1 and action <=3:
            return True if self.miner[1]>=action  else False
        #If you want to add 1 block you need to check there is less than 3 blocks in the private chain
        elif action ==4:
            return True if self.miner[1]<3 else False
        #To keep 2 blocks hidden when you mined 1 and have already 1 hidden
        elif action ==5:
            return True if self.miner[1]==1 else False
        elif action ==6:
            return True if self.miner[1]==2 else False
        return False


    def reset(self):
        self.slip = 0.25  # probability of 'finding' a valid block
        self.empty_block = 2  # payout for 'empty' block, no transactions added
        self.full_block = 2.5  # payout for publishing a block with transactions
        self.action_space = spaces.Discrete(6)
        self.p = autoclass('net.consensys.wittgenstein.protocols.ethpow.ETHMinerAgent').create(self.slip)
        self.p.init()
        self.byz= self.p.getByzantineNode()
        self.MAX_HEIGHT = 100
        self.observation_space = spaces.Discrete(4)
        self.head = 0#change to actual protocol starting height
        self.reward = 0
        self.seed(1)
        print("SELF BY H: ",self.byz.head.height)
        print("RESET MAX H: ",self.MAX_HEIGHT)
        self.p.goNextStep()
        self.p.network().printNetworkLatency() 
        self.secret_blocks = 0
        self.miner =[0,0]
        self.curr_step =0
        self.blocks_mined =0
        return self._get_obs()
        

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
 
