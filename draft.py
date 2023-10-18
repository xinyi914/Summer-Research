import numpy as np
from numpy import random
# t = np.array([[2,3,4,5],[2,4,3,5],[23,42,2,4]])
# c = np.zeros([2,3,4,5,5])
# c[0,0,1,2][0] = 1
# c[0,0,1,2][1] = 1
# c[0,0,1,2][2] = 2
# c[0,0,1,2][4] = 3
# a = int(c[0,0,1,2,0:2].argmax())
# action_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
# action = np.array(action_list)
# residual = 4
# # a = int(np.random.choice(action, size=1))
# # a = int(np.random.choice(action[0:residual+1], size=1))
# a = list(range(500))
a = random.poisson(lam=4,size=1)
print(a)