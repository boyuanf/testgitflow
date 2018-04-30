""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
""" Change the gradient ascent of the orignial code to gradient decent, also make the calculation follow the formula"""

import numpy as np
import pickle
import gym
import time

# hyperparameters
H = 200  # number of hidden layer neurons
batch_size = 10  # every how many episodes to do a param update?
learning_rate = 1e-3
gamma = 0.99  # discount factor for reward
decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2
resume = True  # resume from previous checkpoint?
render = True


# model initialization
D = 80 * 80  # input dimensionality: 80x80 grid
if resume:
    model = pickle.load(open('save.p', 'rb'), encoding="iso-8859-1")
    #model = pickle.load(open('save.p', 'rb'))
else:
    model = {}
    # np.random.randn: Return a sample (or samples) from the "standard normal" distribution.
    model['W1'] = np.random.randn(H, D) / np.sqrt(
        D)  # "Xavier" initialization: setting the variance of W[l] to sqrt(1/n[l-1]) and mean to 0
    model['W2'] = np.random.randn(1, H) / np.sqrt(H)

grad_buffer = {k: np.zeros_like(v) for k, v in model.items()}  # update buffers that add up gradients over a batch
rmsprop_cache = {k: np.zeros_like(v) for k, v in model.items()}  # rmsprop memory


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))  # sigmoid "squashing" function to interval [0,1]


def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2, only take the first color channel
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()  # flatten to 1D array


def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    """ Explained in 'More general advantage functions' section """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0:
            running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def policy_forward(x):
    W1 = model['W1']
    W2 = model['W2']
    Z1 = np.dot(model['W1'], x)  # x shape:  (6400,1)
    A1 = Z1
    A1[A1 < 0] = 0  # ReLU nonlinearity, A1 shape:  (200,)
    Z2 = np.dot(model['W2'], A1)  # Z2 shape (1)
    A2 = sigmoid(Z2)
    return A1, A2, Z1  # return probability of taking action 2, and hidden state

def policy_backward(A1, Z1, dZ2, X):
    m = dZ2.shape[1]
    W2 = model["W2"]

    dW2 = np.dot(dZ2, A1.T)
    dZ1 = np.dot(W2.T, dZ2)
    dZ1_tmp = np.dot(W2.T, dZ2)
    dZ1[Z1 <= 0] = 0  # backpro prelu, here different from the NG's formula, but seems Ok too
    dW1 = np.dot(dZ1, X.T)
    return {'W1': dW1, 'W2': dW2, 'dZ1': dZ1}

env = gym.make("Pong-v0")
observation = env.reset()
prev_x = None  # used in computing the difference frame
xs, A1s, Z1s, dZ2s, drs = [], [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 0
while True:
    if render:
        env.render()
        time.sleep(0.01)

    # preprocess the observation, set input to network to be difference image
    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(D)
    prev_x = cur_x
    x = x.reshape((x.shape[0], 1))

    # forward the policy network and sample an action from the returned probability
    A1, A2, Z1 = policy_forward(x)  # only return the result of a single input
    action = 2 if np.random.uniform() < A2 else 3  # roll the dice! np.random.uniform(): Draw samples from a uniform distribution.

    # record various intermediates (needed later for backprop)
    xs.append(x)  # observation
    Z1s.append(Z1)  # hidden state
    A1s.append(A1)  # hidden state
    # print(len(A1s))
    y = 1 if action == 2 else 0  # a "fake label"
    # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)
    dZ2s.append(A2 - y)

    # step the environment and get new measurements
    observation, reward, done, info = env.step(action)
    reward_sum += reward

    drs.append(reward)  # record reward (has to be done after we call step() to get reward for previous action)

    if done:  # an episode finished
        episode_number += 1

        # stack together all inputs, hidden states, action gradients, and rewards for this episode
        epx = np.hstack(xs)
        epZ1 = np.hstack(Z1s)
        epA1 = np.hstack(A1s)
        epdZ2 = np.hstack(dZ2s)
        epr = np.vstack(drs)

        # compute the discounted reward backwards through time
        discounted_epr = discount_rewards(epr)
        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        epdZ2 *= discounted_epr.T  # modulate the gradient with advantage (PG magic happens right here.)
        grad = policy_backward(epA1, epZ1, epdZ2, epx)

        xs, A1s, Z1s, dZ2s, drs = [], [], [], [], []  # reset array memory

        for k in model:
            grad_buffer[k] += grad[k]  # accumulate grad over batch

        # perform rmsprop parameter update every batch_size episodes
        if episode_number % batch_size == 0:
            for k, v in model.items():
                g = grad_buffer[k]  # gradient
                rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g ** 2
                model[k] -= learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)  # w = w - a * dw
                grad_buffer[k] = np.zeros_like(v)  # reset batch gradient buffer

        # boring book-keeping
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
        if episode_number % 100 == 0:
            pickle.dump(model, open('save.p', 'wb'))

        # save current ep_num to the file
        with open('ep_num', 'a') as file:
           log_str = 'ep: ' + str(episode_number) + '\t' + 'reward: ' + str(reward_sum) + '\t' + \
                     'reward_mean: ' + str(running_reward) + '\n'
           file.write(log_str)
        file.closed

        reward_sum = 0
        observation = env.reset()  # reset env
        prev_x = None

    if reward != 0:  # Pong has either +1 or -1 reward exactly when game ends.
        print(('ep %d: game finished, reward: %f' % (episode_number, reward)) + ('' if reward == -1 else ' !!!!!!!!'))
