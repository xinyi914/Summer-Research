# trial 5_alt
# the demand is random
# have the state with age1(0-15),age2(15-30)
# add lead time for making the sandwiches: one time period


import numpy as np
from numpy import random as rd
import random
import seaborn as sns
import matplotlib.pyplot as plt
# assume the maximum number of making the sandwiches is 20
# assume no transit time and preparation time
CAPACITY = 10
TIME = 12
EPOCH = 1000000

# action
# action_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
action_list = [0,1,2,3,4,5,6,7,8,9,10]

action = np.array(action_list)

# Q table
Q = np.zeros([TIME+1,CAPACITY+1,CAPACITY+1,CAPACITY+1])

# state: time,age1 will be in (0-15)min, age2 will be in (15-30)min
start = np.array([0,0,0])

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
def nextstate(s,a,demand):

     # the newly made food is in age 1
    a1 = s[1] # the number of sandwich in 0-15min during t~t+1
    a2 = s[2] # the number of sandwich in 15-30min during t~t+1
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
        # unmet demand
        reward = reward + (counter - demand)

    # waste
    reward = reward - a2

    # at the end, all sandwiches will be  wasted
    if s[0] == 11:
        reward = reward - (a1+a2)

    assert reward <= 0, f"rewards larger than 0, got {reward}, s is {s}, a is {a}, demand is {demand}"

    time = s[0] + 1
    a = int(a)
    a1 = int(a1)
    return reward, [time,a,a1]


def Qlearning(Q):
    alpha = 0.1
    gamma = 1.0
    epsilon = 0.1
    rsumall = []

    for i in range(EPOCH):
        path = []
        rsum = 0
        s = start
        path.append(s)

        # if i > 200000 and i < 500000:
        #     epsilon = 0.05
        # elif i > 500000 and i<1000000:
        #     epsilon = 0.01
        for t in range(TIME):
            de = demand(t)
            # get the action following the policy
            a = int(Q[s[0],s[1],s[2]].argmax())
            if random.random() < epsilon:
                a = int(np.random.choice(action, size=1))
            r,sprime= nextstate(s,a,de)
            rsum += r
            # for debugging
            check = Q[s[0],s[1],s[2],a]
            add = alpha*(r + gamma * Q[sprime[0], sprime[1], sprime[2]].max() -
                                                         Q[s[0],s[1],s[2],a])
            Q[s[0],s[1],s[2],a] += alpha*(r + gamma * Q[sprime[0], sprime[1], sprime[2]].max() -
                                                         Q[s[0],s[1],s[2],a])
            assert Q[s[0],s[1],s[2],a] <= 0, f"Q larger than 0, previous Q is {check}, the added value is {add},current Q value is {Q[s[0],s[1],s[2],a]}"
            s = sprime
            path.append(s)
        rsumall.append(rsum)
    return Q,rsumall


Qnew,rewards = Qlearning(Q)

policy = np.zeros([TIME+1,CAPACITY+1,CAPACITY+1])
for time in range(TIME+1):
    for age3 in range(CAPACITY+1):
        for age4 in range(CAPACITY+1):
            policy[time,age3,age4] = Qnew[time,age3,age4].argmax()


np.save("trial5_alt_Q_policy1_change_epsilon.npy",policy)
policy = np.load("trial5_alt_Q_policy1_change_epsilon.npy")

# result
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

sns.displot(rewards,kde=True,bins=15)
plt.savefig("rewards_distribution_alt_Q_10c_1000000_12_periods_random_change_epsilon")
plt.show()

mu = np.mean(rewards)
sd = np.std(rewards)
print("mean: ", mu," standard deviation: ",sd)
