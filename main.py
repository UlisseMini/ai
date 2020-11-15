import argparse
import gym
import time
import sys
import os
import numpy as np
import pickle
from gym.spaces import Box, Discrete

# default network perameters
DP = {
    'sigma': 0.5,
    'alpha': 0.5,
    'npop': 50,
    'show_every': 10,
}

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


    # this is terrible, don't do this, don't write code like this
    # I really should find a better way to do this :l
    @classmethod
    def from_params(cls, params, layers):
        total_biases = sum(layers[1:])
        total_weights = sum(a*b for a,b in zip(layers[:-1], layers[1:]))

        biases_arr = params[:total_biases]
        weights_arr = params[total_biases:total_weights]

        biases = []
        weights = []

        offset = 0
        for l in range(len(layers) - 1):
            size = layers[l+1]
            biases.append(biases_arr[offset:offset+size])
            offset += size

        w_shapes = list(zip(layers[1:], layers[:-1]))
        for l in range(len(layers) - 1):
            size = w_shapes[l][0]*w_shapes[l][1]
            weights.append(
                params[offset:offset+size].reshape(w_shapes[l])
            )
            offset += size

        return cls(weights, biases)


    def params(self):
        return np.concatenate(self.biases + [w.flatten() for w in self.weights])


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



    def update_step(self, env, npop, sigma, alpha):
        params = self.params()

        noise = np.random.randn(npop, params.size)
        R = np.zeros(npop)
        for j in range(npop):
            m_net = Net.from_params(params + sigma*noise[j], self.layers)
            R[j] = m_net.evaluate(env)

        A = (R - R.mean()) / (R.std() or 1.0)

        delta  = np.dot(noise.T, A) / (npop*sigma)
        params = params + delta*alpha

        new = Net.from_params(params, self.layers)
        self.biases  = new.biases
        self.weights = new.weights



    def train(
            self, env, generations,
            interactive=True, show_every=DP['show_every'],
            npop=DP['npop'], sigma=DP['sigma'], alpha=DP['alpha']
    ):
        """
        sigma is noise standard deviation
        alpha is learning rate
        """
        for gen in range(generations):
            self.update_step(env, npop, sigma, alpha)

            # print current best reward
            if interactive:
                reward = self.evaluate(env, render=(gen % show_every == 0), sleep=0)
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
    parser.add_argument('--save',   help='save file to use')
    parser.add_argument('--eval',   help='evaluate network', default=False, action='store_true')
    parser.add_argument('--train',  help='train network',    default=False, action='store_true')
    parser.add_argument('--npop',   help='population count',         type=int,   default=DP['npop'])
    parser.add_argument('--sigma',  help='noise standard deviation', type=float, default=DP['sigma'])
    parser.add_argument('--alpha',  help='learning rate',            type=float, default=DP['alpha'])
    parser.add_argument('--layers', help='hidden layers', nargs='+', type=int,   default=[16])
    parser.add_argument('--gen',    help='number of generations',    type=int,   default=100)
    parser.add_argument(
        '--show-every',
        help='how many generations between rendering network in training',
        type=int,
        default=DP['show_every'],
    )

    args = parser.parse_args()

    save_file = args.save or f'{args.env}-{"x".join(map(str, args.layers))}.pkl'

    if not args.train and not args.eval:
        # neither supplied, set both to True
        args.train = True
        args.eval = True

    with gym.make(args.env) as env:
        input_layer = space_to_n(env.observation_space)
        output_layer = space_to_n(env.action_space)
        layers = (input_layer, *args.layers, output_layer)

        if os.path.exists(save_file):
            net = Net.load(save_file)
            # ensure compat with env
            assert net.layers[0]  == input_layer,  f'got input layer {net.layers[0]} want {input_layer}'
            assert net.layers[-1] == output_layer, f'got output layer {net.layers[-1]} want {output_layer}'
            print(f'Loaded network from {save_file}')
        else:
            net = Net.random(layers)
            print('Initialized random network')


        if args.train:
            print('Training network...')
            try:
                net.train(
                    env, args.gen,
                    interactive=True, show_every=args.show_every,
                    npop=args.npop, sigma=args.sigma, alpha=args.alpha,
                )
            # AssertionError raised by some envs when interrupted.
            except (KeyboardInterrupt, AssertionError):
                pass
            finally:
                net.save(save_file)
                print(f'Saved network to {save_file}')

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
