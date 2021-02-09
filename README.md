# GMM-From-Scratch
Simple implementation of Gaussian Mixture Model for clustering unlabeled data using Expectation-Maximization (EM) algorithm. This work is implemented as a homework of CMPE544 Pattern Recognition Course of Boğaziçi University.

Gaussian Mixture Model, or shortly GMM, is an unsupervised learning approach based on clustering. In this approach, it is thought that some of the examples that we have comes from one Gaussian, and some others come from other Gaussians. We get the 'mixture' of these Gaussians to fit the model to our data. EM or Expectation Maximization algorithm, helps us to fit our model to mixture of Gaussians despite not knowing which distribution each of the examples are from.

There are two steps of EM algorithm, namely E step and M step. These are implemented as seperate functions inside the class, and they are called from another function iteratively, until the convergence criteria is reached. This criteria is checked by getting the difference between current and previous values of log likelihood, and stopping when it is smaller than a set value, which is decided according to the dataset.
