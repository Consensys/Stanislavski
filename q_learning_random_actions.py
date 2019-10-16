# coding=utf-8
import gym
#import matplotlib.pyplot as plt
import gym_pow
import numpy as np

env = gym.make('pow-v0')

def choose_action(Q, state, epsilon):
    # if we reach the maximum size for the queue, we force the action
    #  so it does not overflow the maximum size we allocated
    if state[1] > 10: return 1, False
    if epsilon > 0 and (epsilon < 1 or np.random.uniform(0, 1) < epsilon):
        return choose_random_action(state), True
    else:
        # We filter the invalid actions
        scores = Q[state][0]
        bestAction = 0 # do nothing, always valid
        bestScore =  scores[0]
        for i in range(1, env.action_space.n - 1):
            if state[1] >= i and scores[i] > bestScore:
                bestAction = i
                bestScore = scores[i]
        return bestAction, False

def choose_random_action(state):
    # 2 thirds of our actions are sending blocks, so to better explore the space
    #  we give a stronger weight to the 'do nothing' action
    for i in range(0, env.action_space.n-1):
        # We select randomly only valid actions.
        if state[1] < i: return i - 1
        if np.random.uniform(0, 1) < 0.5:
            return i
    return env.action_space.n - 1

def choose_honest_action():
    return 1

# α: The learning rate or step size determines to what extent newly acquired information
#   overrides old information.
# 𝛾: the weight for a step from a state t steps into the future is
#   calculated as 𝛾^t, where 𝛾 (the discount factor) is a number between 0 and 1
#   and has the effect of valuing rewards received earlier higher than those received later
#   (reflecting the value of a "good start").
def learn(cur_state, next_state, reward, action, Q, alpha, gamma):
    predict = Q[cur_state][0][action]

    # get the best valid choice for the new state
    next_action, _ = choose_action(Q, next_state, 0)
    actual = reward + gamma * Q[next_state][0][next_action]
    #print (next_action, next_state, reward, predict, actual)
    Q[cur_state[0],action] += alpha * (actual - predict)

def getEpsilon(max_episode, episode):
    epsilon = 0.1
    if episode < max_episode / 2:
        epsilon = 0.3
    if episode < max_episode / 4:
        epsilon = 0.6
    if episode < max_episode / 8:
        epsilon = 0.9
    if episode < 2:
        epsilon = 1
    if episode % 5  == 0:
        epsilon = 0
    return epsilon

def printState(Q):
    for a in range(1, 10):
        s1 = [a, a, 1]
        s2 = [a, a, 2]
        s3 = [a, a, 3]
        print (a, Q[s1][0], Q[s2][0], Q[s3][0])

def start(type_of_action, slip, alpha, gamma):
    max_episode = 100

    env.resetSlip(slip)
    Q = np.zeros((env.observation_space_size, env.action_space.n))

    average_payouts = []
    for episode in range(1, max_episode):
        state = env.reset()

        epsilon = 1 if type_of_action == "random" else getEpsilon(max_episode, episode)
        done = False
        steps = 0
        lastPrintBlock = 0
        while done is False:
            steps += 1

            epsilonUsed = epsilon if lastPrintBlock < env.max_block * .9 else 0

            action, rd = choose_action(Q, state, epsilonUsed)
            new_state, last_event, time, myBlocks, reward, lastRewardEth, done, info = env.step(action)

            if epsilon == 0 and steps < 200:
                rds = "(random)" if rd else ""
                print("slip", env.slip, "STEP, time", time, "myBlocks", myBlocks, "lastRewardEth",lastRewardEth, "action",action, rds, "state",state, "->", new_state, Q[state][action])

            if myBlocks % 500 == 0 and lastPrintBlock != myBlocks:
                print("slip", env.slip, "BLOCKS, episode", episode, "time:", time, "myBlocks", myBlocks, "lastRewardEth",lastRewardEth,"epsilon", epsilonUsed, "alpha", alpha, "gamma", gamma, "state", state)
                lastPrintBlock = myBlocks

            learn(state, new_state, reward, action, Q, alpha, gamma)
            state = new_state

            if done:
                print("slip", env.slip, "REWARD RATIO:",info['ratio'],"epsilon", epsilon, "episode", episode, "hp:", info['hp'], "alpha", alpha, "gamma", gamma)
                if epsilon == 0:
                    printState(Q)
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
    for alpha in [0.5,  0.3, .1, 0.05]:
        for gamma in [0.99, 0.90, 0.70, 0.5]:
            for slip in [0.6, 0.4, 0.1]:
                start("agent", slip, alpha, gamma)

if __name__ == '__main__':
    main()
