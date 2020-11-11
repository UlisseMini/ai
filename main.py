import gym
import time
import sys
import numpy as np
import pickle
from gym.spaces import Box, Discrete

DEFAULT_ENV = 'CartPole-v0'

# activation function
def relu(z):
    return np.maximum(0, z)


# NOTE: If you change this, make sure you deal with
# low values with cont action values.
activ = relu


class Net:
    def __init__(self, weights, biases):
        assert len(weights) == len(biases)

        self.weights = weights
        self.biases  = biases


    @classmethod
    def random(cls, layers):
        biases = [
            np.random.randn(x)
            for x in layers[1:]
        ]

        weights = [
            np.random.randn(y, x)
            for x, y in zip(layers[:-1], layers[1:])
        ]
        return cls(weights, biases)


    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)


    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)


    def forward(self, a):
        for w, b in zip(self.weights, self.biases):
            a = activ(np.dot(w, a) + b)

        return a


    def evaluate(self, env, render=False):
        total_reward = 0

        obs = env.reset()
        done = False
        while not done:
            if render:
                env.render()
                time.sleep(1/60)

            a = self.forward(obs)
            if isinstance(env.action_space, Box):
                # Since relu's min output is 0, we can't
                # reach low. say low is -2, if we subtract 2
                # from the result then we can access the low value.
                action = a + env.action_space.low
            else:
                action = np.argmax(a)

            obs, reward, done, _ = env.step(action)
            total_reward += reward
            # for mountaincar
            # if obs[0] >= env.goal_position:
            #     total_reward += 100


        return total_reward



    def train(self, env, generations, npop=50, sigma=0.01, alpha=0.01):
        """
        sigma is noise standard deviation
        alpha is learning rate
        """
        for gen in range(generations):
            w_noise = [np.random.randn(npop, *w.shape) for w in self.weights]
            b_noise = [np.random.randn(npop, *b.shape) for b in self.biases]
            assert len(w_noise) == len(b_noise)

            R = np.zeros(npop)
            for j in range(npop):
                # mutate weights and biases
                m_weights = [w + w_noise[i][j] for i, w in enumerate(self.weights)]
                m_biases = [b + b_noise[i][j]  for i, b in enumerate(self.biases)]

                mutation = Net(m_weights, m_biases)
                R[j] = mutation.evaluate(env)

            # standardize the rewards to have a gaussian distribution
            A = (R - np.mean(R)) / (np.std(R) or 1.0)

            # weight the noise based on rewards
            w_delta = []
            b_delta = []
            for i in range(len(w_noise)):
                w_weighted_avg = np.zeros(self.weights[i].shape)
                b_weighted_avg = np.zeros(self.biases[i].shape)

                for j in range(npop):
                    b_weighted_avg += b_noise[i][j] * A[j]
                    w_weighted_avg += w_noise[i][j] * A[j]

                b_weighted_avg /= npop
                w_weighted_avg /= npop


                w_delta.append(w_weighted_avg) # <-- should be self.weights[i].shape
                b_delta.append(b_weighted_avg) # <-- should be self.biases[i].shape



            # preform the parameter update.
            # print(w_delta[0].shape)
            # print(b_delta[0].shape)
            self.weights = [w + wd for w, wd in zip(self.weights, w_delta)]
            self.biases  = [b + bd for b, bd in zip(self.biases, b_delta)]


            # print current best reward
            reward = self.evaluate(env)
            print(f'gen {gen} reward: {reward}')



def main():
    env_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ENV
    save_file = f'net-{env_id}.pkl'
    with gym.make(env_id) as env:
        layers = (*env.observation_space.shape, 16)
        if isinstance(env.action_space, Box):
            layers = (*layers, *env.action_space.shape)
        else:
            layers = (*layers, env.action_space.n)

        try:
            net = Net.load(save_file)
            print(f'Loaded model from {save_file}')
        except FileNotFoundError:
            print('Initialized random network')
            net = Net.random(layers)


        try:
            net.train(env, 100)
            print('Saved model')
            net.save(save_file)
        except KeyboardInterrupt:
            pass

        print('Training done!')
        while True:
            reward = net.evaluate(env, render=True)
            print('reward', reward)


if __name__ == '__main__':
    main()
