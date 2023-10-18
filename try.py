import numpy as np
import random
import matplotlib.pyplot as plt
# assume the maximum number of making the sandwiches is 20
# assume no transit time and preparation time
CAPACITY = 20
TIME = 12
# state(age1,age2,age3,age4,# of sandwiches on the counter, time]
# state = np.array([[0,0,0,0,0,0],[0,0,0,0,0,1],[0,0,0,0,0,2],[0,0,0,0,0,3], # 11-11:45
#                   [0,0,0,0,0,4],[0,0,0,0,0,5],[0,0,0,0,0,6],[0,0,0,0,0,7], # 12-12:45
#                   [0,0,0,0,0,8],[0,0,0,0,0,9],[0,0,0,0,0,10],[0,0,0,0,0,11], # 13-13:45
#                   [0,0,0,0,0,12], # 14
#                   ])
# state need to assume the maximum number of sandwiches on the counter is 20
# state_list = []
# for a1 in range(CAPACITY+1):
#     for a2 in range(CAPACITY+1):
#         for a3 in range(CAPACITY+1):
#             for a4 in range(CAPACITY+1):
#                 for counter in range(CAPACITY+1):
#                     for time in range(TIME+1):
#                         state_list.append([a1,a2,a3,a4,counter,time])
#
# # print(state_list)
# # the number of food on the counter can be removed which is just the sum of the a1-a4
# state = np.array(state_list)

# action
action_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
action = np.array(action_list)

# Q table
Q = np.zeros([TIME+1,CAPACITY+1,CAPACITY+1,CAPACITY+1,CAPACITY+1,CAPACITY+1,CAPACITY+1])

start = np.array([0,0,0,0,0,0])

# calculate the demand for each time
def demand(time):
    if time < 4:
        return 4
    elif time < 8:
        return 8
    elif time < 12:
        return 3
#SARSA
# # reward function
# def reward(s,a,demand):
#     # the cost of waste (similar to holding cost)
#     waste = s[3]
#     # the cost of not meeting the demand FIFO
#     lack = demand-(s[4]-s[3]+a) # s[4] is the number of
#

# the input state is the state after the take away has happened
# return reward and next state
def nextstate(s,a,demand):

    reward = -s[4]

    # the newly made food is in age 1
    a4 = s[3]
    a3 = s[2]
    a2 = s[1]
    a1 = a
    counter = a1+a2+a3+a4 # the number of sandwich on the counter
    # if s[0] == 11:
    #     reward = s[4]
    # print("demand: ",demand," action: ",a)
    # update state after meeting the demand
    if demand <= a4:
        a4 = a4 - demand
        # print("a4 reduced by ", demand)
    elif demand <= a4 + a3:
        a3 = a3 - (demand - a4)
        a4 = 0
        # print("a3 reduced by ", demand-a3)
    elif demand <= a4 + a3 + a2:
        a2 = a2 - (demand-a4-a3)
        a4 = 0
        a3 = 0
        # print("a2 reduced by ", demand-a4-a3)
    elif demand <= a4 + a3 + a2 + a1:
        a1 = a1 - (demand - a4 - a3 - a2)
        a4 = 0
        a3 = 0
        a2 = 0
        # print("a1 reduced by ", demand-a4-a3-a2)

    else:
        a1 = 0
        a2 = 0
        a3 = 0
        a4 = 0
        # print("unmet demand ")
        reward = reward + (counter - demand)
    # calculate both met and unmet demand
    # reward = reward + (counter - demand)
    if s[0] == 11:
        reward = reward - (a1+a2+a3+a4)
    # else:
    #     # count the unmet/met demand into the reward
    #     reward = reward + (counter - demand)
    counter = a1+a2+a3+a4
    time = s[0] + 1

    return reward, [time,a1,a2,a3,a4,counter]


def Qlearning(Q):
    alpha = 0.5
    gamma = 1.0
    epsilon = 0.1
    rsumall = []

    for i in range(10000):
        path = []
        rsum = 0
        s = start
        path.append(s)
        k=0
        for t in range(TIME): # not include 2:00pm
            # at time i (i != 0): calculate the number of sandwiches taken away and update the state and
            de = demand(t)
            residual = 20 - s[5]
            # limit the number of sandwiches on the counter
            a = int(Q[s[0],s[1],s[2],s[3],s[4],s[5],0:(residual+1)].argmax())
            # print("a ", a)
            if random.random() < epsilon:
                a = int(np.random.choice(action[0:residual+1], size=1))
            r,sprime = nextstate(s,a,de)
            rsum += r
            # print(sprime)
            residualprime = 20-sprime[5]
            # print(residualprime)
            # print(Q[sprime[0], sprime[1], sprime[2], sprime[3], sprime[4], sprime[5]].max())
            Q[s[0],s[1],s[2],s[3],s[4],s[5],a] += alpha*(r+gamma *
                                                         Q[sprime[0], sprime[1], sprime[2], sprime[3], sprime[4], sprime[5],0:(residualprime+1)].max() -
                                                         Q[s[0],s[1],s[2],s[3],s[4],s[5],a])
            s = sprime
            path.append(s)
            k += 1
        rsumall.append(rsum)
    return Q,rsumall

start = [0,0,0,0,0,0]
Qnew,rewards = Qlearning(Q)

policy = np.zeros([TIME+1,CAPACITY+1,CAPACITY+1,CAPACITY+1,CAPACITY+1,CAPACITY+1])
for time in range(TIME+1):
    for age1 in range(CAPACITY+1):
        for age2 in range(CAPACITY+1):
            for age3 in range(CAPACITY+1):
                for age4 in range(CAPACITY+1):
                    max_sand = CAPACITY - (age1 + age2 + age3 + age4)
                    for capacity in range(max_sand+1):
                        policy[time,age1,age2,age3,age4,0:capacity] = Qnew[time,age1,age2,age3,age4,0:max_sand+1].argmax()

plt.plot(list(range(10000)),rewards)
plt.xlabel("epochs")
plt.ylabel("rewards")
plt.show()


# print("rewardss: ", rewards)