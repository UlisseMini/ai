from main import Net
import os
from gym.spaces import Box, Discrete
import gym

TESTS = [
    ('CartPole-v0', 200.0, 50, (16,)),
    ('Acrobot-v1', -200.0, 50, (16,)),
    ('Pendulum-v0', -1000, 100, (16,)),
]

def test(args):
    env_id, want_reward, generations, layers = args
    print(f'START({env_id})')

    with gym.make(env_id) as env:
        layers = (*env.observation_space.shape, *layers)
        if isinstance(env.action_space, Box):
            layers = (*layers, *env.action_space.shape)
        else:
            layers = (*layers, env.action_space.n)

        net = Net.random(layers)
        net.train(env, generations, render=False, print_stats=False)
        n = 10 # times to run evaluation before taking average
        reward = sum(net.evaluate(env) for _ in range(n)) / n

        if reward < want_reward:
            print(f'FAILED({env_id}): want {reward} < {want_reward}')
        else:
            print(f'SOLVED({env_id}) reward: {reward} >= {want_reward}')


def main():
    for t in TESTS: test(t)


if __name__ == '__main__':
    main()
