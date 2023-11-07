import numpy as np
import matplotlib.pyplot as plt

# Setup Grid
grid = np.ones((6, 9))
grid[1:4, 2] = 0
grid[0:3, 7] = 0
grid[4, 5] = 0
start = (2, 0)
goal = (0, 8)
rewards = np.zeros((6, 9))
rewards[goal] = 1

# DynaQ Algorithm Parameters
gamma = 0.95
alpha = 0.1
epsilon = 0.1
n = [0, 5, 50]

# Actions
'''
0: up
1: right
2: down
3: left
'''
actions = [0,1,2,3]

# Helper Variables
rng = np.random.default_rng(seed=42)
observedStates = []
observedStateActions = np.zeros((grid.size, len(actions)))
numSteps = 0
numEpisodes = 50
numExperiments = 30
avgNumSteps = np.empty((numExperiments, numEpisodes))
avgResults = np.empty((len(n), numEpisodes))

# Q-Table
# States x Actions
Q = np.zeros((grid.size, len(actions)))

# Model
# States x Actions
Model = np.zeros((grid.size, len(actions)), dtype=object)


# Epsilon-Greedy Policy
def epsilon_greedy(S):
    p = rng.random()
    action = 0
    if p <= epsilon: # Random Choice
        action = rng.choice(actions)
    else: # Max of Q[S] Choice
        max = np.argwhere(Q[S] == np.amax(Q[S]))
        action = rng.choice(max)[0]
    return action


def not_out_of_grid(S, A):
    notValidMoves = {
        0: [0,3],
        1: [0],
        2: [0],
        3: [0],
        4: [0],
        5: [0],
        6: [0],
        7: [0],
        8: [0,1],
        9: [3],
        18: [3],
        27: [3],
        36: [3],
        17: [1],
        26: [1],
        35: [1],
        44: [1],
        45: [2,3],
        46: [2],
        47: [2],
        48: [2],
        49: [2],
        50: [2],
        51: [2],
        52: [2],
        53: [1,2]
    }

    if S in notValidMoves.keys() and A in notValidMoves[S]:
        return False
    else:
        return True


def take_action(S, A):
    nextState = 0
    R = 0

    if not_out_of_grid(S,A):

        if A == 0:  # Up
            nextState = S - grid.shape[1]
        elif A == 1:  # Right
            nextState = S + 1
        elif A == 2:  # Down
            nextState = S + grid.shape[1]
        elif A == 3:  # Left
            nextState = S - 1
        else:
            print('Action Not Valid')

        # Obstacles
        if nextState in [11,20,29,7,16,25,41]:
            nextState = S

        R = rewards.flatten()[nextState]

    else:
        R = rewards.flatten()[S]
        nextState = S

    return R, nextState


# Run 30 experiments for n=0,5,50 with 50 episodes each experiment.
for idx, numPlanningSteps in enumerate(n):

    for experiment in range(numExperiments):

        Q = np.zeros((grid.size, len(actions)))
        stepsPerEpisode = []

        for episode in range(numEpisodes):

            # DynaQ Algorithm
            currentState = np.ravel_multi_index(start, grid.shape)
            numSteps = 0
            while True:
                if currentState == np.ravel_multi_index(goal, grid.shape):
                    # print('Reached Terminal State')
                    break
                else:
                    S = currentState
                    A = epsilon_greedy(S)
                    observedStateActions[S, A] = 1
                    if not S in observedStates:
                        observedStates.append(S)

                    # Take action A; observe resultant reward, R and state, S'
                    R, nextState = take_action(S, A)
                    # print('\n')
                    # print(f'currentState: {S}')
                    # print(f'Action: {A}')
                    # print(f'nextState: {nextState}')
                    # print(f'Reward: {R}')

                    # Update Q-Table
                    max = np.argwhere(Q[nextState] == np.amax(Q[nextState]))
                    greedyAction = rng.choice(max)[0]
                    Q[S, A] = Q[S, A] + alpha*(R + gamma*Q[nextState, greedyAction] - Q[S, A])

                    # Update Model (assuming deterministic environment)
                    Model[S, A] = (R, nextState)

                    # Planning Phase
                    for i in range(numPlanningSteps):
                        # print(f'    Planning Phase')
                        # print(f'      observedStates: {observedStates}')
                        S = rng.choice(observedStates)
                        observedActions = np.argwhere(observedStateActions[S] == np.amax(observedStateActions[S]))
                        A = rng.choice(observedActions)[0]
                        R, nextStatePlanning = Model[S, A]
                        # print(f'      S: {S}')
                        # print(f'      A: {A}')
                        # print(f'      nextState: {nextStatePlanning}')
                        # print(f'      R: {R}')

                        # Update Q-Table
                        greedyActions = np.argwhere(Q[nextStatePlanning] == np.amax(Q[nextStatePlanning]))
                        greedyAction = rng.choice(greedyActions)[0]
                        Q[S, A] = Q[S, A] + alpha * (R + gamma * Q[nextStatePlanning, greedyAction] - Q[S, A])
                    currentState = nextState
                    numSteps += 1

            stepsPerEpisode.append(numSteps)
        avgNumSteps[experiment] = stepsPerEpisode
    avgResults[idx] = np.mean(avgNumSteps, axis=0)


print(f'Q: {Q}')

# Plot Learning Curves for n=0,5,50
plt.xlabel('Episodes')
plt.ylabel('Steps per Episode')
plt.plot(range(numEpisodes), avgResults[0], label='0 planning steps')
plt.plot(range(numEpisodes), avgResults[1], label='5 planning steps')
plt.plot(range(numEpisodes), avgResults[2], label='50 planning steps')
plt.legend(loc='upper center')
plt.show()