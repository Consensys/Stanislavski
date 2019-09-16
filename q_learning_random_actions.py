import gym
import gym_pow
import numpy as np
import time, pickle, os
import matplotlib.pyplot as plt

env = gym.make('pow-v0')# Change to 40% hashpower

epsilon = 0.5
total_episodes = 200
max_steps = 1000

lr_rate = 0.6
gamma = 1

Q = np.zeros((env.observation_space.n+1, env.action_space.n))
print(Q)

    
def choose_action(state):
    action=0
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state, :])
    return action

def learn(state, state2, reward, action):
    predict = Q[state, action]
    print("predicted value", predict)
    target = reward + gamma * np.max(Q[state2, :])
    Q[state, action] = Q[state, action] + lr_rate * (target - predict)

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
        print("action: ",action)

        state2, reward, done, info = env.step(action)  
        print("REWARD: ",reward)
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