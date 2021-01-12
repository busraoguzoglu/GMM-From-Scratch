import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import random


class GaussianMixtureModel:

    def __init__(self, dataset, k):
        # Initialization of necessary variables
        self.dataset = dataset
        self.data = pd.DataFrame(dataset)
        self.rows, self.columns = self.data.shape   # Rows: N
        np.seterr(divide='ignore')

        self.k = k
        self.pi = np.zeros(k)
        for i in range(k):
            self.pi[i] = 1/self.k

        self.mu = []
        for i in range(k):
            self.mu.append(np.array([0.0, 0.0]))
        # Random initialization
        for i in range(0, 2):
            self.mu[i][0] = random.uniform(0, 1)
            self.mu[i][1] = random.uniform(0, 1)

        self.sigma = []
        for i in range(k):
            self.sigma.append(np.array([[15.0, 10.0],
                                        [10.0, 10.0]]))

        self.gamma = np.zeros((self.rows, self.k))
        self.conditional_distribution = np.zeros((self.rows, self.k))

        # Evaluating initial value of log likelihood
        self.loglikelihood = self.get_loglikelihood()

    def e_step(self):

        # Find gamma
        # For k distributions
        for k in range(self.k):
            gaussian_distribution = multivariate_normal(self.mu[k], self.sigma[k])
            # Getting the probability density function
            self.conditional_distribution[:, k] = gaussian_distribution.pdf(self.dataset)  # Assigning to kth column

        numerator = self.pi*self.conditional_distribution
        denominator = np.zeros((self.rows, 1))
        # Sum over k
        for r in range(self.rows):
            for k in range(self.k):
                denominator[r] += numerator[r][k]
        self.gamma = numerator / denominator


    def m_step(self):

        Nk = np.zeros(self.k)
        for i in range(self.rows):  # N
            for k in range(self.k):
                Nk[k] += self.gamma[i][k]

        # Find new pi
        for k in range(self.k):
            self.pi[k] = Nk[k]/self.rows

        # Find new mu
        for k in range(self.k):
            self.mu[k] = (self.dataset * self.gamma[:, [k]]).sum(0)/Nk[k]

        # Find new sigma
        for k in range(self.k):
            self.sigma[k] = np.cov(self.dataset.transpose(),
                                   aweights=(self.gamma[:, [k]]/Nk[k]).flatten())


    def get_loglikelihood(self):

        # Compute value of log likelihood.
        conditional_distribution = self.conditional_distribution*self.pi

        var1 = np.zeros((self.rows, 1))
        for i in range(self.rows):
            for k in range(self.k):
                var1[i] += conditional_distribution[i][k]

        lnvar1 = np.log(var1)

        loglikelihood = 0
        for i in range(self.rows):
            loglikelihood += lnvar1[i]

        return loglikelihood

    def fit(self):
        difference = 1000
        iterations = 0
        while difference > 0.001:
            self.e_step()
            self.m_step()
            newloglikelihood = self.get_loglikelihood()
            difference = newloglikelihood - self.loglikelihood
            self.loglikelihood = newloglikelihood
            iterations += 1

        print('Fit in', iterations, 'iterations.')

    def get_predictions(self):
        return np.argmax(self.gamma, axis=1)


def main():

    dataset = np.load('dataset.npy')
    k = 3

    model = GaussianMixtureModel(dataset, k)
    model.fit()

    labels = model.get_predictions()
    data = pd.DataFrame(dataset)

    data['labels'] = labels
    d0 = data[data['labels'] == 0]
    d1 = data[data['labels'] == 1]
    d2 = data[data['labels'] == 2]

    # Show the estimated values for mean and covariance:
    print('means:', model.mu)
    print('covariances:', model.sigma)

    plt.scatter(d0[0], d0[1], c='magenta')
    plt.scatter(d1[0], d1[1], c='purple')
    plt.scatter(d2[0], d2[1], c='pink')

    plt.show()



if __name__ == '__main__':
    main()
