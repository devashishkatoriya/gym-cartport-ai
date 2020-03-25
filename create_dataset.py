
# pip install numpy

from random import randint
import numpy as np
import random
import gym

# ---------------------------------------

# Initialize gym environment
env = gym.make("CartPole-v0")
env.reset()
goal_steps = 1000
score_requirement = 110


# Function to create dataset
def initial_population(n):

    scores = []
    accepted_scores = []

    X_train = []            # OBS
    Y_train = []            # Moves

    while len(Y_train) <= n:

        # time.sleep(1)

        score = 0
        game_memory = []
        prev_observation = []

        for _ in range(goal_steps):

            # choose random action (0 or 1)
            action = random.randrange(0, 2)

            # env.render()

            # apply the action
            observation, reward, done, info = env.step(action)

            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])
            prev_observation = observation
            score += reward
            if done:
                break

        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:

                # convert to one-hot
                if data[1] == 1:
                    output = [0, 1]
                elif data[1] == 0:
                    output = [1, 0]

                X_train.append(data[0])
                Y_train.append(output)

        env.reset()
        scores.append(score)

    return X_train, Y_train


# Main Function
def main():

    print('Generating Datasets...')

    n = 5000
    X_train, Y_train = initial_population(n)

    X_train_np = np.array(X_train)
    Y_train_np = np.array(Y_train)

    np.save('data_X_train.npy', X_train_np)
    np.save('data_Y_train.npy', Y_train_np)

    print('Done')

    return 0


# ---------------------------------------

if __name__ == "__main__":
    main()
