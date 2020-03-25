
# pip install gym numpy tensorflow keras

from keras.layers import Dense, Activation
from keras.models import Sequential
import keras
import numpy as np
import gym
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# ----------------------------------

# Initialize gym environment
env = gym.make("CartPole-v0")
env.reset()
goal_steps = 1000
score_requirement = 110


# Define Model
def create_model(input_size, output_size):
    model = Sequential()

    model.add(Dense(input_size, input_dim=4))
    model.add(Activation('relu'))

    model.add(Dense(32))
    model.add(Activation('relu'))

    model.add(Dense(32))
    model.add(Activation('relu'))

    model.add(Dense(output_size))
    model.add(Activation('softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


# Main Function
def main():

    no_of_games = 10

    # Create untrained model
    model = create_model(4, 2)

    # Checkpoint path to stored model
    checkpoint_path = "model/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Load previous weights
    model.load_weights(checkpoint_path)

    # Testing our model
    print('Testing...')

    scores = []
    choices = []
    for each_game in range(no_of_games):

        print('Game #', each_game+1)

        score = 0
        game_memory = []
        prev_obs = []

        # time.sleep(0.1)

        env.reset()
        for _ in range(goal_steps):

            # env.render()

            if len(prev_obs) == 0:
                action = 1              # Take right action
            else:
                Y = model.predict(np.array([prev_obs]))
                if Y[0][1] <= 0.5:
                    action = 0
                else:
                    action = 1

            choices.append(action)

            new_observation, reward, done, info = env.step(action)
            prev_obs = new_observation
            game_memory.append([new_observation, action])
            score += reward
            if done:
                break

        scores.append(score)
        print('Score:', score)

    print('Score requirement:', score_requirement)
    print('Average Score:', sum(scores)/len(scores))
    print('choice 1:{}  choice 0:{}'.format(choices.count(
        1)/len(choices), choices.count(0)/len(choices)))

    print('Done!')

    return


# ----------------------------------
if __name__ == "__main__":
    main()
