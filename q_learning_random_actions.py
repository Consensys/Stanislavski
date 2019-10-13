import gym
#import matplotlib.pyplot as plt
import gym_pow
import numpy as np

env = gym.make('pow-v0')


def choose_action(Q, state,_epsilon):
    # if we reach the maximum size for the queue, we force the action
    #  so it does not overflow the maximum size we allocated
    if state[1] > 10: return 1, False
    if np.random.uniform(0, 1) <_epsilon:
        return choose_random_action(state), True
    else:
        # Filter invalid actions
        bestAction = 0 # do nothing, always valid
        bestScore = Q[state][0][0]
        for i in range(1, env.action_space.n - 1):
            if state[1] >= i and Q[state][0][i] > bestScore:
                bestAction = i
                bestScore = Q[state][0][i]
        return bestAction, False

def choose_random_action(state):
    # 2 thirds of our actions are sending blocks, so to better explorate the space
    #  we give a stronger weight to the 'do nothing' action
    for i in range(0, env.action_space.n-1):
        # We select randomly only valid actions.
        if state[1] < i: return i - 1
        if np.random.uniform(0, 1) < 0.5:
            return i
    return env.action_space.n - 1

def choose_honest_action():
    return 1

def learn(cur_state, next_state, reward, action, Q, alpha, gamma):
    predict = Q[cur_state, action]
    target = reward + gamma * np.max(Q[next_state])
    Q[cur_state, action] = Q[cur_state, action] + alpha * (target - predict)

def start(type_of_action, lr_rate, gamma):
    Q = np.zeros((10*10*3, env.action_space.n))

    average_payouts = []
    episode = 0
    while episode < 500:
        episode += 1

        epsilon = 0.1
        if episode < 100:
            epsilon = 0.5
        if episode < 10:
            epsilon = 1
        if episode % 2 == 0:
            epsilon = 0
        if episode % 9 == 0:
            epsilon = 0.9

        state = env.reset()
        done = False
        total_payout = 0
        staps = 0
        lastPrintBlock = 0
        while done is False:
            epsilonUsed = epsilon if lastPrintBlock < 9000 else 0
            staps += 1
            rd = False
            if type_of_action == "random":
                action = choose_random_action()
            elif type_of_action == "honest":
                action = choose_honest_action()
            else:
                action, rd = choose_action(Q, state, epsilonUsed)

            #print (state)
            state2, last_event, time, myBlocks, reward, lastRewardEth, done, info = env.step(action)

            if epsilon == 0 and staps < 200:
                rds = ""
                if rd: rds = "(random)"
                print("STEP, time", time, "myBlocks", myBlocks, "lastRewardEth",lastRewardEth, "action",action, rds, "state",state, "->", state2)

            if myBlocks % 500 == 0 and lastPrintBlock != myBlocks:
                print("BLOCKS, episode", episode, "time:", time, "myBlocks", myBlocks, "lastRewardEth",lastRewardEth,"epsilon", epsilonUsed, "alpha", lr_rate, "gamma", gamma, "state", state)
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
    for alpha in [0.2, .4, 0.05]:
        for gamma in [0.9, 0.70, 0.50, 0.99, 0.999]:
            start("agent", alpha, gamma)

if __name__ == '__main__':
    main()
