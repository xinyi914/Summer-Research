# trial 5
# the demand is detemrined
# only retain the last two age periods in the state
# have the state with age1(0-30),age2(31-60)
# add lead time for making the sandwiches: one time period
# add lead time will cause the demand in the first time period can not be met,
# so I added a preparation time period, start from 10:30


import numpy as np
from numpy import random as rd
import random
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

# state: time,age3(0-30), age4(31-60)
start = np.array([0,4,0])

# calculate the demand for each time
def demand(time):
    if time <= 4:
        return 4
    elif time <= 8:
        return 8
    else:
        return 5

# the input state: n1 and n2 is the number of sandwiches has been stayed 15min and 30min
# return reward and next state
def nextstate(s,a,demand):
    # waste
    reward = -s[2]

    # the newly made food is in age 1
    a4 = s[1]
    a3 = a
    # a3 = age2
    # a2 = age1
    # a1 = a
    counter = a3 + a4 # the number of sandwich on the counter
    # if s[0] == 11:
    #     reward = s[4]
    # print("demand: ",demand," action: ",a)
    # update state after meeting the demand
    if demand <= a4:
        a4 = a4 - demand

        # print("a4 reduced by ", demand)
    elif demand <= (a3+a4):
        a3 = a3 - (demand-a4)
        a4 = 0
    else:
        a4 = 0
        a3 = 0
        # print("unmet demand ")
        # may add some weights on the waste and the unmet demand
        # assert demand > counter
        reward = reward + (counter - demand)

    # calculate both met and unmet demand
    # reward = reward + (counter - demand)
    if s[0] == 11:
        reward = reward - (a3+a4)

    # else:
    #     # count the unmet/met demand into the reward
    #     reward = reward + (counter - demand)
    # counter = a1+a2+a3+a4
    assert reward <= 0, f"rewards larger than 0, got {reward}, s is {s}, a is {a}, demand is {demand}"
    time = s[0] + 1

    return reward, [time,a3,a4]


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
        k=0
        # future_action[0] is the action chosen by previous round and will be added to the state of this round
        # future_action[1] is the action chosen by this round and will be added to the state of next round
        future_action = [0,0]
        if i > 200000 and i < 500000:
            epsilon = 0.05
        elif i > 500000 and i<100000:
            epsilon = 0.01
        for t in range(TIME): # not include 2:00pm
            de = demand(t)
            # limit the number of sandwiches on the counter
            a = int(Q[s[0],s[1],s[2]].argmax())
            if random.random() < epsilon:
                a = int(np.random.choice(action, size=1))
            future_action[0] = future_action[1]
            future_action[1] = a
            r,sprime= nextstate(s,future_action[0],de)
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
            k += 1
        rsumall.append(rsum)
    return Q,rsumall


Qnew,rewards = Qlearning(Q)

policy = np.zeros([TIME+1,CAPACITY+1,CAPACITY+1])
for time in range(TIME+1):
    for age3 in range(CAPACITY+1):
        for age4 in range(CAPACITY+1):
            policy[time,age3,age4] = Qnew[time,age3,age4].argmax()

# see which state [:,age1,age2,age3,age4] has nonzero value
# print(*np.argwhere(policy != 0))

print("Q policy: ",Qnew[4,0,0,:])
plt.plot(list(range(EPOCH)),rewards)
plt.xlabel("epochs")
plt.ylabel("rewards")
plt.savefig("rewards_lead_time_10c_1000000_12_periods_various_epsilon")
plt.show()
# plot policy
plt.clf()
# plot the state at [3,4,2,5]
tomake = []
for t in range(TIME+1):
    if t == 0:
        tomake.append(policy[t,4,0])
    else:
        tomake.append(policy[t,0,0])

plt.plot(list(range(TIME+1)),tomake)
plt.xlabel("Time")
plt.ylabel("Number of sandwiches to make")
plt.savefig("policy_lead_time_10c_1000000_12_periods_various_epsilon")
plt.show()