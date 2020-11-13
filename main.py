import argparse
import gym
import time
import sys
import numpy as np
import pickle
from gym.spaces import Box, Discrete

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
        # perhaps reconstructing layers is a bit hackish /shrug
        self.layers  = [weights[0].shape[1]] + [len(b) for b in biases]


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
            net = pickle.load(f)
        # keep compat with old networks if we update init.
        return Net(net.weights, net.biases)


    def forward(self, a):
        for w, b in zip(self.weights, self.biases):
            a = activ(np.dot(w, a) + b)

        return a


    def evaluate(self, env, render=False, sleep=1/60):
        total_reward = 0

        obs = env.reset()
        done = False
        while not done:
            if render:
                env.render()
                time.sleep(sleep)

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



    def train(self, env, generations, interactive=True, npop=50, sigma=0.5, alpha=0.01):
        """
        sigma is noise standard deviation
        alpha is learning rate
        """
        for gen in range(generations):
            w_noise = [np.random.randn(npop, *w.shape)*sigma for w in self.weights]
            b_noise = [np.random.randn(npop, *b.shape)*sigma for b in self.biases]
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
            if interactive:
                reward = self.evaluate(env, render=(gen % 10 == 0), sleep=0)
                print(f'gen {gen} reward: {reward}')




def space_to_n(space):
    if isinstance(space, Box):
        # todo: flatten if multi-dimensional
        assert len(space.shape) == 1, 'no multidimensional observations'
        return space.shape[0]
    elif isinstance(space, Discrete):
        return space.n
    else:
        raise NotImplemented



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env',    help='the gym enviorment to use', default='CartPole-v0')
    parser.add_argument('--load',   help='load a model from a file', dest='model')
    parser.add_argument('--eval',   help='evaluate network', default=True, action='store_false')
    parser.add_argument('--train',  help='train network',    default=True, action='store_false')
    parser.add_argument('--npop',   help='population count',         type=int,   default=50)
    parser.add_argument('--sigma',  help='noise standard deviation', type=float, default=0.5)
    parser.add_argument('--alpha',  help='learning rate',            type=float, default=0.01)
    parser.add_argument('--layers', help='hidden layers', nargs='+', type=int,   default=[16])
    parser.add_argument('--gen',    help='number of generations',    type=int,   default=100)

    args = parser.parse_args()

    with gym.make(args.env) as env:
        input_layer = space_to_n(env.observation_space)
        output_layer = space_to_n(env.action_space)
        layers = (input_layer, *args.layers, output_layer)

        if args.model:
            net = Net.load(args.model)
            # ensure compat with env
            assert net.layers[0]  == input_layer,  f'got input layer {net.layers[0]} want {input_layer}'
            assert net.layers[-1] == output_layer, f'got output layer {net.layers[-1]} want {output_layer}'
        else:
            net = Net.random(layers)


        if args.train:
            print('Training network...')
            try:
                net.train(
                    env, args.gen, interactive=True,
                    npop=args.npop, sigma=args.sigma, alpha=args.alpha,
                )
            except KeyboardInterrupt:
                pass

        if args.eval:
            print('Evaluating network...')
            try:
                while True:
                    print('reward', net.evaluate(env, render=True))
            except KeyboardInterrupt:
                pass

        print('\nDone!')



if __name__ == '__main__':
    main()
