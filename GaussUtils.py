import numpy as np
import collections


class GenerateGaussianData:

    def __init__(self,
                 random=np.random.RandomState(0),
                 D=2, # dimension of data. 2 is for plotting...
                 N=1000, # expected number of samples
                 M=2, # number of Gaussians
                 center_variance=0.5):

        self.random = random
        self.D = D
        self.N = N
        self.M = M
        self.center_variance = center_variance

        # prior to each Gaussian
        self.prior = np.random.uniform(size=(self.M,))
        self.prior = self.prior / np.sum(self.prior)
        print("Actual Prior = {}".format(self.prior))

        # Let's generate some data according to GJ93:
        self.X = []  # The data
        self.labels = []

        # create centers
        self.centers = self.random.normal(size=(self.M, self.D))

        # create data
        for n in range(self.N):
            c = self.random.choice(self.M, p=self.prior)
            self.X.append(self.random.normal(self.centers[c], size=(1, D), scale=self.center_variance))
            self.labels.append(c)

        self.X = np.vstack(self.X)

        # Check prior
        self.empirical_prior = collections.Counter(self.labels)
        print("empirical_prior={}".format(self.empirical_prior))

