import sys
import gym
import torch
import pylab
import random
import numpy as np
from collections import deque
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from prioritized_memory import Memory

import pickle
import sys
from pymgrid.Environments.pymgrid_cspla import MicroGridEnv
from wrappers import NormalizedMicroGridEnv
from gym.wrappers import FrameStack, FlattenObservation
from torch.utils.tensorboard import SummaryWriter
import datetime
from collections import deque
from torch.optim.lr_scheduler import StepLR

current_time = datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")

EPISODES = 10

# approximate Q function using Neural Network
# state is input and Q Value of each action is output of network


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 24),
            nn.ReLU(inplace=True),
            nn.Linear(24, 24),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(24, action_size)
        )

    def forward(self, x):
        return self.fc(x)


# it uses Neural Network to approximate q function
# and prioritized experience replay memory & target q network
class DQNAgent():
    def __init__(self, state_size, action_size):
        self.render = False
        self.load_model = False

        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.lr_step_size = 10
        self.lr_gamma = 0.9
        self.memory_size = 2**15
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.explore_step = 1000
        self.epsilon_decay = 0.99995
        self.batch_size = 64
        self.train_start = 10000

        # create prioritized replay memory using SumTree
        self.memory = Memory(self.memory_size)

        # create main model and target model
        self.model = DQN(state_size, action_size)
        self.model.apply(self.weights_init)
        self.target_model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.learning_rate)
        self.scheduler = StepLR(
            self.optimizer, step_size=self.lr_step_size, gamma=self.lr_gamma)

        # initialize target model
        self.update_target_model()

        if self.load_model:
            self.model = torch.load('save_model/per_dqn')
        self.model.train()

    # weight xavier initialize
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            torch.nn.init.xavier_uniform_(m.weight)

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    # get action from model using epsilon-greedy policy
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            state = torch.from_numpy(state).float()
            q_value = self.model(state)
            _, action = torch.max(q_value, 1)
            return int(action)

    # save sample (error,<s,a,r,s'>) to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        target = self.model(torch.tensor(state).float()).data
        old_val = target[0][action]
        target_val = self.target_model(
            torch.tensor(next_state).float()).data
        if done:
            target[0][action] = reward
        else:
            target[0][action] = reward + \
                self.discount_factor * torch.max(target_val)

        error = abs(old_val - target[0][action])

        self.memory.add(error, (state, action, reward, next_state, done))

    # pick samples from prioritized replay memory (with batch_size)
    def train_model(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)

        mini_batch, idxs, is_weights = self.memory.sample(self.batch_size)
        mini_batch = np.array(mini_batch).transpose()

        states = np.vstack(mini_batch[0])
        actions = list(mini_batch[1])
        rewards = list(mini_batch[2])
        next_states = np.vstack(mini_batch[3])
        dones = mini_batch[4]

        # bool to binary
        dones = dones.astype(int)

        # Q function of current state
        states = torch.tensor(states).float()
        pred = self.model(states)

        # one-hot encoding
        a = torch.tensor(actions, dtype=torch.long).view(-1, 1)

        one_hot_action = torch.zeros(
            self.batch_size, self.action_size)
        one_hot_action.scatter_(1, a, 1)

        pred = torch.sum(pred.mul(one_hot_action), dim=1)

        # Q function of next state
        next_states = torch.tensor(next_states, dtype=torch.float)
        next_pred = self.target_model(next_states.float()).data

        rewards = torch.tensor(rewards, dtype=torch.float)
        dones = torch.tensor(dones, dtype=torch.float)

        # Q Learning: get maximum Q value at s' from target model
        target = rewards + (1 - dones) * \
            self.discount_factor * next_pred.max(1)[0]

        errors = torch.abs(pred - target).data.numpy()

        # update priority
        for i in range(self.batch_size):
            idx = idxs[i]
            self.memory.update(idx, errors[i])

        self.optimizer.zero_grad()

        # MSE Loss function
        loss = (torch.tensor(is_weights).float() *
                F.mse_loss(pred, target)).mean()
        loss.backward()

        # and train
        self.optimizer.step()
        return loss.item()


if __name__ == "__main__":
    with open('building_1.pkl', 'rb') as f:
        building_1 = pickle.load(f)
    with open('building_2.pkl', 'rb') as f:
        building_2 = pickle.load(f)
    with open('building_3.pkl', 'rb') as f:
        building_3 = pickle.load(f)

    buildings = [building_1, building_2, building_3]
    perfect_train_scores = [4068.5, 13568.92, 15345.97]

    building_choice = 2
    similarity_stop = 0.3
    scores_list = deque(maxlen=5)

    perfect_train_score = perfect_train_scores[building_choice]
    building = buildings[building_choice]

    env = MicroGridEnv(
        env_config={'microgrid': building, "testing": False})
    env = NormalizedMicroGridEnv(env)
    #env = FlattenObservation(FrameStack(env, 24))

    test_env = MicroGridEnv(
        env_config={'microgrid': building, "testing": True})
    test_env = NormalizedMicroGridEnv(test_env)

    writer = SummaryWriter(comment=current_time)

    state_size = env.observation_space.low.size
    action_size = env.action_space.n

    model = DQN(state_size, action_size)

    agent = DQNAgent(state_size, action_size)
    scores, episodes, steps = [], [], 0

    for e in range(EPISODES):
        done = False
        score = 0

        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            if agent.render:
                env.render()

            # get action for the current state and go one step in environment
            action = agent.get_action(state)

            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            # save the sample <s, a, r, s'> to the replay memory
            agent.append_sample(state, action, reward, next_state, done)
            # every time step do the training
            if agent.memory.tree.n_entries >= agent.train_start:
                loss = agent.train_model()
                writer.add_scalar('Loss', loss, steps)
                steps += 1

            score += reward
            state = next_state

            if done:
                # every episode update the target model to be same with model
                agent.update_target_model()

                # every episode, plot the play time
                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./save_graph/per_dqn.png")

                print("episode:", e, "  score:", score, "  memory length:",
                      agent.memory.tree.n_entries, "  epsilon:", agent.epsilon)

                agent.scheduler.step()  # should be val loss
                writer.add_scalar(
                    'Learning rate', agent.scheduler.get_last_lr()[0], e)

                writer.add_scalar('Training total building cost', score, e)
                torch.save(agent.model, "./save_model/per_dqn")
                torch.save(agent.model, "./save_model/per_dqn_" +
                           current_time + "_" + str(e).zfill(5))

                test_done = False
                test_score = 0

                state = test_env.reset()
                state = np.reshape(state, [1, state_size])

                with torch.no_grad():
                    while not test_done:
                        state = torch.from_numpy(state).float().cpu()
                        q_value = agent.model(state)
                        _, action = torch.max(q_value, 1)
                        action = int(action)
                        next_state, reward, test_done, info = test_env.step(action)
                        next_state = np.reshape(next_state, [1, state_size])
                        test_score += reward
                        state = next_state

                        if test_done:
                            writer.add_scalar(
                                'Testing total building cost', test_score, e)

                similarity = abs(score - perfect_train_score) / \
                    perfect_train_score
                print("Similarity to perfect score", similarity)
                if similarity <= similarity_stop:
                    print("Reached similarity stop of", similarity_stop)
                    sys.exit()
                if len(scores_list) >= scores_list.maxlen:
                    mean = np.mean(scores_list)
                    similarity = abs(score - mean) / abs(mean)
                    print("Similarity to stable score", similarity)
                    if similarity <= similarity_stop:
                        print("Reached stable score")
                        sys.exit()
                scores_list.append(score)
