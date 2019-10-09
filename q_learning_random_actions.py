import gym
import gym_pow
import numpy as np
import time, pickle, os
import matplotlib.pyplot as plt

env = gym.make('pow-v0')# Change to 40% hashpower

lr_rate = 0.5
gamma = 0.95

Q = np.zeros((100, env.action_space.n))
print(Q)

    
def choose_action(state,_epsilon):
    action=0
    if np.random.uniform(0, 1) <_epsilon:
        return env.action_space.sample(), True
    else:
        return np.argmax(Q[state[1], :]), False

def choose_random_action():
    action = env.action_space.sample()
    return action

def choose_honest_action():
    return 1

def learn(state, state2, reward, action, episode):
    predict = Q[state[1], action]
    if episode % 100 == 0:
        print("predicted value", predict)
    target = reward + gamma * np.max(Q[state2[1], :])
    Q[state[1], action] = Q[state[1], action] + lr_rate * (target - predict)

# Start
def start(type_of_action):
    average_payouts = []
    t = 0
    episode=0
    while True:
        episode+=1
        if episode % 10 == 0:
            epsilon = 0
        else:
            epsilon = 0.10
            if episode < 500:
                epsilon = 0.5

        state = env.reset()
        done = False
        total_payout = 0
        print("Episode",t)
        max_steps=0
        while done is False:
            #env.render()
            max_steps+=1
            rd = False
            if(type_of_action=="random"):
             
                action = choose_random_action()

            elif(type_of_action=="honest"):
                action = choose_honest_action()
            else:
                action, rd = choose_action(state,epsilon) 
            
            state2, reward, done, info = env.step(action,episode,epsilon)
            if done is True:
                print("REWARD RATIO: ",info['ratio']," epsilon ", epsilon) 

            if episode % 100 == 0: 
                print("action: ",action, rd)
                print("STATE: ",state2)
                print("----INFO---- ",info)

            learn(state, state2, reward, action, episode)
            total_payout+=info['amount']
            state = state2

            if done:
                t+=1
                break
        average_payouts.append(total_payout)

        if episode%1000==0:
            plt.plot(average_payouts[-1000:])
            plt.xlabel('Episodes in range {} {}'.format(episode,type_of_action))
            plt.ylabel('ETH Payout in an hour')
            plt.savefig('q_learning_range_%s'%episode)
            plt.clf()

'''def print_graph_periodically(period):
    plt.plot()#stored data per period
    plt.xlabel()'''

def main():
    '''start("random",1)
    start("honest",1)'''
    start("agent")


if __name__ == '__main__':
    main()
