import gym
import gym_pow
import numpy as np
import time, pickle, os
import matplotlib.pyplot as plt

env = gym.make('pow-v0')# Change to 40% hashpower

epsilon = 0.2
total_episodes = 500
max_steps = 1000

lr_rate = 0.8
gamma = 1

Q = np.zeros((10, env.action_space.n))
print(Q)

    
def choose_action(state):
    action=0
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state[1], :])
    return action

def choose_random_action():
    action = env.action_space.sample()
    return action
def choose_honest_action():
    return 0

def learn(state, state2, reward, action):
    predict = Q[state[1], action]
    print("predicted value", predict)
    target = reward + gamma * np.max(Q[state2[1], :])
    Q[state[1], action] = Q[state[1], action] + lr_rate * (target - predict)

# Start
average_payouts = []
for episode in range(total_episodes):
    state = env.reset()
    t = 0
    total_payout = 0
    if(t%10==0):
        print("Episode",t)
    while t < max_steps:
        #env.render()

        action = choose_action(state)  
        #action = choose_random_action()
        #action = choose_honest_action()
        print("action: ",action)

        state2, reward, done, info = env.step(action)  
        print("STATE: ",state2)
        learn(state, state2, reward, action)
        total_payout+=reward
        state = state2

        t += 1
       
        if done:
            break
            time.sleep(0.1)
    average_payouts.append(total_payout)

plt.plot(average_payouts)                
plt.xlabel('total_episodes')
plt.ylabel('ETH payout after {} rounds'.format(max_steps))
plt.show()    
print ("Average payout after {} rounds is {}".format(max_steps, sum(average_payouts)/total_episodes))


        

print(Q)

with open("pow_qTable.pkl", 'wb') as f:
    pickle.dump(Q, f)