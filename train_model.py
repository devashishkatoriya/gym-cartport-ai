
# pip install pyyaml h5py numpy os tensorflow keras

from keras.layers import Dense, Activation
from keras.models import Sequential
import keras
import numpy as np
import random
import time
import gym
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# ----------------------------------


LR = 1e-3
env = gym.make("CartPole-v0")
env.reset()
goal_steps = 1000
score_requirement = 110
initial_games = 60000

t1 = time.time()


def some_random_games_first():
    # Each of these is its own game.
    for episode in range(5):

        # time.sleep(1)
        env.reset()

        for t in range(goal_steps):

            # env.render()

            action = env.action_space.sample()
            #print ('action is ', action)

            observation, reward, done, info = env.step(action)
            if done:
                break

# some_random_games_first()


def initial_population():
    # [OBS, MOVES]
    training_data = []
    scores = []
    accepted_scores = []

    for _ in range(initial_games):

        # time.sleep(1)

        score = 0
        game_memory = []
        prev_observation = []

        for _ in range(goal_steps):

            # choose random action (0 or 1)
            action = random.randrange(0, 2)

            # env.render()

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

                training_data.append([data[0], output])

        env.reset()

        scores.append(score)

    training_data_save = np.array(training_data)
    np.save('saved.npy', training_data_save)

    return training_data


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

    # Load the training dataset
    X_train = np.load('data_X_train.npy')
    Y_train = np.load('data_Y_train.npy')

    print('Train data loaded.')

    print('X_train[0]', X_train[0])

    # Create untrained model
    model = create_model(len(X_train[0]), len(Y_train[0]))

    print('Model created.')

    # Checkpoint path to store model
    checkpoint_path = "model/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Callback method that checkpoints the model's weights
    cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path)

    print('Training...')

    # Train the model
    model.fit(X_train, Y_train,
              epochs=200,
              batch_size=32,
              callbacks=[cp_callback])

    print('Done!')

    t2 = time.time()

    print('Total time taken:', (t2-t1))

    return


# ----------------------------------
if __name__ == "__main__":
    main()
