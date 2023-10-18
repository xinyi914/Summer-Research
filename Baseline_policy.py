# trial 7_delay_reward
#implement the simple policy to compare with the simulation of random demand

# have the state with age1(0-15),age2(15-30)
# add lead time for making the sandwiches: one time period
# add lead time will cause the demand in the first time period can not be met
# change the way to simulate

import numpy as np
from numpy import random as rd
import random
import matplotlib.pyplot as plt
import seaborn as sns
# assume the maximum number of making the sandwiches is 20
# assume no transit time and preparation time

CAPACITY = 10
TIME = 12
EPOCH = 1000000

# action
# action_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
action_list = [0,1,2,3,4,5,6,7,8,9,10]

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

def nextstate(s,a,demand):

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
        reward = reward + (counter - demand)


    reward = reward - a2

    if s[0] == 11:
        reward = reward - (a1+a2)

    assert reward <= 0, f"rewards larger than 0, got {reward}, s is {s}, a is {a}, demand is {demand}"
    time = s[0] + 1
    a = int(a)
    a1 = int(a1)
    # print("reward3: ", reward)
    return reward, [time,a,a1]

reward_list = []
for i in range(EPOCH):
    s = [0, 0, 0]
    reward = 0
    for t in range(TIME):
        if t <= 4:
            a = 4
        elif t <= 8:
            a = 8
        else:
            a = 5
        # tomake.append(a)
        d = demand(t)
        r, sprime = nextstate(s, a, d)
        # print("state: ", s, " to make: ", a," reward: ",r,", demand: ",d)
        s = sprime
        reward += r
    reward_list.append(reward)
rewards = np.array(reward_list)
sns.displot(rewards,kde=True,bins=15)

# plt.plot(list(range(TIME)),tomake)
# plt.xlabel("Time")
# plt.ylabel("Number of sandwiches to make")
plt.savefig("rewards_distribution_alt_simple_10c_1000000_12_periods_random_3")
plt.show()

mu = np.mean(rewards)
sd = np.std(rewards)
print("mean: ", mu," standard deviation: ",sd)

