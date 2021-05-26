import numpy as np
# import matplotlib.pyplot as plt

rng = np.random.default_rng(seed=42)

def train(f, starting_params, npop=50, nbest=20, iterations=10):
    params = len(starting_params)

    # strating covariance matrix and means
    cov = np.identity(params)
    mean = starting_params

    for i in range(iterations):
        # Sample solutions from a multivariate normal distribution
        pop = rng.multivariate_normal(mean, cov, size=npop)
        pop_best = np.array(list(sorted(pop, key=lambda x: -f(x))))[:nbest]

        # Estimate the new covariance matrix and the mean of each parameter.
        # The reason we use total sample mean instead of best sample mean
        # is so we estimate the covariance of the best guys, ie. what mutation
        # helped them, or what direction is promising for exploration.
        mean = np.array([pop[:,i].mean() for i in range(params)])
        for i in range(params):
            for j in range(params):
                cov[i][j] = (1/nbest) * np.sum((pop_best[:,i] - mean[i])*(pop_best[:,j] - mean[j]))

        mean = np.array([pop_best[:,i].mean() for i in range(params)])
        yield pop_best[0]


if __name__ == '__main__':
    solution = np.array([0.5, 0.1])
    def f(w):
        return -np.sum((w - solution)**2)

    for sol in train(f, np.array([0, 0])):
        print(f(sol), sol)

