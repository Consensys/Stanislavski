import gym
import gym_pow
#import matplotlib.pyplot as plt
import numpy as np

env = gym.make('pow-v0')


def choose_action(Q, state,_epsilon):
    if np.random.uniform(0, 1) <_epsilon:
        return choose_random_action(state), True
    else:
        # better: would be to remove the invalid actions from the Q[state[0], :]
        action =  np.argmax(Q[state[0], :])
        if state[1] >= action:
            return action, False
        else:
            return choose_random_action(state), True

def choose_random_action(state):
    while True:
        action = env.action_space.sample()
        if state[1] >= action:
            return action

def choose_honest_action():
    return 1

def learn(state, state2, reward, action, Q, lr_rate, gamma):
    predict = Q[state[0], action]
    '''if episode % 500 == 0:
        print("predicted value", predict)'''
    target = reward + gamma * np.max(Q[state2[0], :])
    Q[state[0], action] = Q[state[0], action] + lr_rate * (target - predict)

def start(type_of_action, lr_rate, gamma):
    Q = np.zeros((100, env.action_space.n))

    average_payouts = []
    episode = 0
    while episode < 300:
        episode += 1

        epsilon = 0.001
        if episode < 200:
            epsilon = 0.01
        if episode < 100:
            epsilon = 0.05
        if episode < 50:
            epsilon = 0.10
        if episode < 10:
            epsilon = 0.2
        if episode < 5:
            epsilon = 0.4
        if episode % 50 == 0:
            epsilon = 0
        if episode % 55 == 0:
            epsilon = 0.1

        state = env.reset()
        done = False
        total_payout = 0
        max_steps=0
        lastPrintBlock = 0
        while done is False:
            #env.render()
            max_steps+=1
            rd = False
            if(type_of_action=="random"):
                action = choose_random_action()
            elif(type_of_action=="honest"):
                action = choose_honest_action()
            else:
                action, rd = choose_action(Q, state,epsilon)
            
            state2, last_event, time, myBlocks, reward, lastRewardEth, done, info = env.step(action)

            if epsilon == 0 and max_steps < 200:
                rds = ""
                if rd: rds = "(random)"
                print("STEP, time", time, "myBlocks", myBlocks, "lastRewardEth",lastRewardEth, "action",action, rds, "state",state, "->", state2)

            if myBlocks % 500 == 0 and lastPrintBlock != myBlocks:
                print("BLOCKS, episode", episode, "time:", time, "myBlocks", myBlocks, "lastRewardEth",lastRewardEth,"epsilon", epsilon, "alpha", lr_rate, "gamma", gamma)
                lastPrintBlock = myBlocks

            learn(state, state2, reward, action, Q, lr_rate, gamma)
            state = state2

            if done:
                print("REWARD RATIO:",info['ratio'],"epsilon", epsilon, "episode", episode, "hp:", info['hp'], "alpha", lr_rate, "gamma", gamma)
                total_payout = info['amount']

        average_payouts.append(total_payout)

        '''if episode%1000==0 and False:
            print(Q)
            plt.plot(average_payouts[-1000:])
            plt.xlabel('Episodes in range {} {}'.format(episode,type_of_action))
            plt.ylabel('ETH Payout in an hour')
            plt.savefig('q_learning_range_%s'%episode)
            plt.clf()'''

'''def print_graph_periodically(period):
    plt.plot()#stored data per period
    plt.xlabel()'''

def main():
    for alpha in [0.05, 0.20, 0.30, 0.01]:
        for gamma in [0.60, 0.70, 0.80, 0.90, 0.99, 0.999, 0.50]:
            start("agent", alpha, gamma)

if __name__ == '__main__':
    main()
