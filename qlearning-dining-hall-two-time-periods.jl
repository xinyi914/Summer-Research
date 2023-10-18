using Debugger
using Plots
using Distributions

# suppose that there is a maximum of m sandwiches/items that will
# be made in any 15-minute interval.
m = 10
P4 = Poisson(4)    # Poisson distribution with mean 4 per time period
P5 = Poisson(5)
P8 = Poisson(8)
# the state space is
# time of day, there are 12 15 minute intervals
# betweem 11am and 1pm. t = 1,...,12.
# but note that t=13 is the terminal state.
# num items that have age  0-15 min
# num items that have age 16-30 min
# s = [1,0,0]   # initial state at beginning of episode
# Q function is a Dict where the key is a vector that contains state
# and the value is a vector that contains the possible actions
# (with an estimated Q value for each action

function initQ()
    Q = Dict{Vector{Int64}, Vector{Float64}}()
    for t in 1:13
        for n1 in 0:m  # 0-15 min
            for n2 in 0:m  # 16-30 min
                s = [t,n1,n2]
                a = zeros(m+1)  # action is 0,1,2,...,m
                Q[s] = a
            end
        end
    end
    return Q
end

# generate demand according the time of day
# note that t = 1 means 11am
#           t = 5 means 12noon
#           t = 9 means 1pm
#           t = 13 means 2pm
function demand(t)
    if t <= 4     # 11am to 12noon
        dmd = 2
    elseif t <= 8 # 12noon to 1pm
        dmd = 8
    else          # 1pm to 2pm
        dmd = 5
    end
    return dmd
end

# take action a, observe the reward r and the next state sprime.  note
# that the action a is to make quantity a sandwiches, those sandwiches
# are not available until the next time period. in other words, it takes
# 15 minutes to make sandwiches.
function nextstate(st, a)
    s = copy(st)
    t = s[1]   # the current time period
    # generate the demand for the current time period
    dmd_rem = demand(t)
    d = dmd_rem    # save 
    # the sandwiches are selected according to FIFO
    while dmd_rem > 0
        if sum(s[2:3]) == 0  # no items left to satisfy demand
            break
        elseif s[3] > 0
            s[3] -= 1
            dmd_rem -= 1
        else
            @assert s[2] > 0
            s[2] -= 1
            dmd_rem -=1
        end
    end

    if dmd_rem > 0      # there is unmet demand for items
        @assert s[3] == 0 && s[2] == 0
        r = -1 * dmd_rem
    elseif s[3] > 0   # these items will be wasted
        @assert dmd_rem == 0
        r = -1 * s[3]
        if t == 12    # all remaining items at end of day are wasted
            r += -1 * s[2]
        end
    else
        @assert dmd_rem == 0 & s[3] == 0
        r = 0
        if t == 12
            r += -1 * s[2]
        end
    end
    sprime = [t+1, a, s[2]]   # the new items are available for the next time period
    return r, sprime, d
end

function choose_action(Q, s, eps=0.10)
    qval, idx = findmax(Q[s])
    a = idx - 1   # action is 0,...,m
    if rand() < eps
        a = rand(0:m)
        qval = Q[s][a+1]
    end
    return a, qval
end

#
function Qlearning!(Q)
    alpha = 0.1          # step size
    gamma = 1.0          # discount factor
    epsilon = 0.10       # for the epsilon-greedy policy
    #    
    sumr = zeros(1000000)
    sumd = zeros(12)
    for i in 1:1000000
        s = [1,2,0]      # initialize the state for this episode
        for t in 1:12    # loop over this episode
            # choose a from s using policy dervied from Q (e.g. epsilon-greedy)
            a, qval = choose_action(Q, s, epsilon)
            # take action a and observe reward r and next state s'
            r, sprime, d = nextstate(s, a)
            # make the update to Q
            Q[s][a+1] += alpha * (r + gamma*findmax(Q[sprime])[1] - Q[s][a+1])
            s = sprime
            sumr[i] += r
            sumd[t] += d
        end
    end
    return sumr, sumd
end

Q = initQ();
sumr, sumd = Qlearning!(Q);
plot(sumr)
sumd ./ 1000000

# extract the policy
policy = Dict{Vector{Int64}, Int64}()
for (key, value) in Q
    policy[key] = findmax(value)[2] - 1
end

# plot the policy for a few specific states
tomake = zeros(12)
for t in 1:12
    if t==1
        tomake[t] = policy[[t,4,0]]
    else
        tomake[t] = policy[[t,0,0]]
    end
end
plot(1:12, tomake, xticks=1:12)
