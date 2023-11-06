import numpy as np

# Setup Grid
grid = np.ones((6, 9))
grid[1:4, 2] = 0
grid[0:3, 7] = 0
grid[4, 5] = 0
start = (2, 0)
goal = (0, 8)
rewards = np.zeros((6, 9))
rewards[goal] = 1

# Parameters
gamma = 0.95
alpha = 0.1
epsilon = 0.1
n = 10

# Helper
rng = np.random.default_rng()

# Actions
'''
0: up
1: right
2: down
3: left
'''
actions = [0,1,2,3]

# Q-Table
# States x Actions
Q = np.zeros((grid.size, len(actions)))

# Model
# States x Actions
Model = np.zeros((grid.size, len(actions)), dtype=object)

# Epsilon-Greedy Policy
def epsilon_greedy(S):
    # rng = np.random.default_rng()
    p = rng.random()
    action = 0
    if p < epsilon: # Random Choice
        action = rng.choice(actions)
    else: # Max of Q[S] Choice
        max = np.argwhere(Q[S] == np.amax(Q[S]))
        action = rng.choice(max)[0]
    return action


def take_action(S, A):
    nextState = 0
    R = 0

    # NEED TO CHECK IF EDGE AND SKIP IF SO!!!!
    if A == 0:  # Up
        # if using 2D indices: nextState = np.unravel_index(S, grid.shape)
        nextState = S - grid.shape[1]
    elif A == 1:  # Right
        nextState = S + 1
    elif A == 2:  # Down
        nextState = S + grid.shape[1]
    elif A == 3:  # Left
        nextState = S - 1
    else:
        print('Action Not Valid')

    R = rewards.flatten()[nextState]
    return R, nextState

# DynaQ Algorithm
currentState = start
observedStates = []

while True:
    S = np.ravel_multi_index(currentState, grid.shape)
    A = epsilon_greedy(S)

    # Take action A; observe resultant reward, R and state, S'
    R, nextState = take_action(S, A)
    print(f'A: {A}')
    print(f'nextState: {nextState}')
    print(f'R: {R}')

    # Update Q-Table
    max = np.argwhere(Q[nextState] == np.amax(Q[nextState]))
    greedyAction = rng.choice(max)[0]
    Q[S, A] = Q[S, A] + alpha*(R + gamma*Q[nextState, greedyAction] - Q[S, A])

    # Update Model (assuming deterministic environment)
    Model[S, A] = (R, nextState)

    for i in range(n):
        pass

    break
