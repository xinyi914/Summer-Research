# MC_Policy

import numpy as np
from numpy import random as rd
import random
import matplotlib.pyplot as plt
import seaborn as sns

CAPACITY = 10
TIME = 12
EPOCH = 1000000

# action
# action_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
action_list = [0,1,2,3,4,5,6,7,8,9,10]
#
action = np.array(action_list)

# random demand
def demand(time):
    if time <= 4:
        # assume average demand is 4
        return rd.poisson(lam=4,size = 1)
    elif time <= 8:
        # assume average demand is 8
        return rd.poisson(lam=8, size = 1)
    else:
        # assume average demand is 5
        return rd.poisson(lam=5,size = 1)


# the input state is the state after the take away has happened
# return reward and next state
def nextstate(s,a,demand,w1 = 1,w2 = 1):

    # the newly made food is in age 1
    a1 = s[1] # the number of sandwich in 0-30min during t~t+1
    a2 = s[2] # the number of sandwich in 31-60min during t~t+1
    counter = a1 + a2 # the number of sandwich on the counter
    # update state after meeting the demand
    reward = 0
    if demand <= a2:
        a2 = a2 - demand
    elif demand <= (a2+a1):
        a1 = a1 - (demand-a2)
        a2 = 0
    else:
        a1 = 0
        a2 = 0
        reward = reward + w1*(counter - demand)

    reward = reward - w2*a2

    if s[0] == 11:
        reward = reward - w2*(a1+a2)

    assert reward <= 0, f"rewards larger than 0, got {reward}, s is {s}, a is {a}, demand is {demand}"
    time = s[0] + 1
    a = int(a)
    a1 = int(a1)
    return reward, [time,a,a1]


def simulate(policy,epsilon=0.1):
    states = [[0,0,0]]
    actions = []
    reward = []
    for i in range(TIME):
        # state_length = len(states)
        s = states[-1]
        a = int(policy[s[0],s[1],s[2]])
        if random.random() < epsilon:
            a = int(np.random.choice(action, size=1))
        actions.append(a)
        r, sprime = nextstate(s,a,demand(i))
        if i < 11:
            states.append(sprime)
        # reward = gamma * reward + r
        reward.append(r)
    # actions.append(0)
    return reward,states,actions

rewards = []
def MC(q,qn,policy,V):
    for i in range(EPOCH):
        reward,states,actions = simulate(policy)
        # rewards.append(reward)
        # print(len(states))
        gamma = 0.5
        g = 0
        for j in range(len(states)-1,-1,-1):
            g = gamma * g + reward[j]
            s = states[j]
            a = actions[j]
            qn[s[0],s[1],s[2],a] += 1
            q[s[0],s[1],s[2],a] += (g-q[s[0],s[1],s[2],a])/qn[s[0],s[1],s[2],a]
            V[s[0],s[1],s[2]] = q[s[0], s[1], s[2]].max()
            max_a = q[s[0], s[1], s[2]].argmax()
            policy[s[0],s[1],s[2]] = max_a
        rewards.append(g)

policy = np.zeros([TIME+1,CAPACITY+1,CAPACITY+1])
q = np.zeros([TIME+1, CAPACITY+1,CAPACITY+1,CAPACITY+1])
qn = np.zeros([TIME+1, CAPACITY+1,CAPACITY+1,CAPACITY+1])
V = np.zeros([TIME+1,CAPACITY+1,CAPACITY+1])
MC(q,qn,policy,V)
np.save("trial7_alt_MC_20c_policy.npy",policy)

# result
policy = np.load("trial7_alt_MC_10c_policy.npy")

tomake = []
rewards = np.zeros(EPOCH)

for i in range(EPOCH):
    s = [0, 0, 0]
    reward = 0
    for t in range(TIME):
        tomake.append(policy[s[0], s[1], s[2]])
        d = demand(t)
        r, sprime = nextstate(s, tomake[t], d)
        s = sprime
        reward += r
    rewards[i] = reward

sns.displot(rewards,kde=True,bins=10)

plt.savefig("rewards_distribution_alt_MC_10c_1000000_12_periods_random_4")
plt.show()

mu = np.mean(rewards)
sd = np.std(rewards)
print("mean: ", mu," standard deviation: ",sd)
