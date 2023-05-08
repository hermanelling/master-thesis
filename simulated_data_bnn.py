# Follows implementation by: https://github.com/nitarshan/bayes-by-backprop/blob/master/Weight%20Uncertainty%20in%20Neural%20Networks.ipynb



import math
import random
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


import torchvision.transforms as T

from tensorboardX import SummaryWriter
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from tqdm import tqdm, trange
import random
random.seed(1)

import warnings

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

from sklearn.model_selection import train_test_split

writer = SummaryWriter()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOADER_KWARGS = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
print(torch.cuda.is_available())

def simulate_from_bi_normal(mean, sigma, n, class_number, r_seed=1):
    np.random.seed(r_seed)
    # Step 1
    U = np.random.standard_normal(size=(n, 2))
    
    sigma_root = np.linalg.cholesky(sigma)
    sigma_check = sigma_root @ sigma_root.T
    if not np.allclose(sigma_check, sigma):
        print(f"Sigma_check is not equal sigma")
    
    # Step 2
    W = U @ sigma_root
    
    # Step 3
    y_1 = np.array(W[:, 0])
    y_2 = np.array(W[:, 1])
    
    # Step 4
    y_1 = np.array([float(y_1i + mean[0]) for y_1i in np.array(W[:, 0])])
    y_2 = np.array([float(y_2i + mean[1]) for y_2i in np.array(W[:, 1])])
    class_array = np.full(n, class_number)
    
    return np.column_stack((y_1, y_2, class_array))


# Class 1
mean_k = np.matrix([[-2.], [2.]])
sigma_k = np.matrix([
    [0.5, 0],
    [0, 0.5]])

# Class 2
mean_l = np.matrix([[2.], [2.]])
sigma_l = np.matrix([
    [0.5, 0],
    [0, 0.5]])

# Class 3
mean_m = np.matrix([[2.], [-2.]])
sigma_m = np.matrix([
    [0.5, 0],
    [0, 0.5]])

# Class 4
mean_n = np.matrix([[-2.], [-2.]])
sigma_n = np.matrix([
    [0.5, 0],
    [0, 0.5]])

minmaxscaler = MinMaxScaler(feature_range=(0, 1))
n_samples_200 = int(200)
k = simulate_from_bi_normal(mean_k, sigma_k, n_samples_200, 0, r_seed=1)
l = simulate_from_bi_normal(mean_l, sigma_l, n_samples_200, 1, r_seed=2)
m = simulate_from_bi_normal(mean_m, sigma_m, n_samples_200, 2, r_seed=3)
n = simulate_from_bi_normal(mean_n, sigma_n, n_samples_200, 3, r_seed=4)

# Add to one matrix and maxmin-scale:
data_4_200_unscaled = np.concatenate((k, l, m, n), axis=0)
minmaxscaler.fit(data_4_200_unscaled) 
data_4_200 = minmaxscaler.transform(data_4_200_unscaled)


minmaxscaler = MinMaxScaler(feature_range=(0, 1))

X = data_4_200_unscaled[:, :2]
y = data_4_200_unscaled[:, 2]


minmaxscaler.fit(X)
X = minmaxscaler.transform(X)
X, y = torch.from_numpy(X).float(), torch.from_numpy(y).long() 

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=10, stratify=y)
X_train.shape, y_train.shape


BATCH_SIZE = 10
TEST_BATCH_SIZE = 5


TRAIN_SIZE = len(X_train)
TEST_SIZE = len(X_test)


X_train = torch.split(X_train, BATCH_SIZE)
X_test = torch.split(X_test, TEST_BATCH_SIZE)
y_train = torch.split(y_train, BATCH_SIZE)
y_test = torch.split(y_test, TEST_BATCH_SIZE)


NUM_BATCHES = len(X_train)
NUM_TEST_BATCHES = len(X_test)

CLASSES = 4
TRAIN_EPOCHS = 10 # Endre
SAMPLES = int(len(X)*0.8) # Endre
TEST_SAMPLES = int(len(X)*0.2)

assert (TRAIN_SIZE % BATCH_SIZE) == 0
assert (TEST_SIZE % TEST_BATCH_SIZE) == 0



# Reparameterized Gaussian
class Gaussian(object):
    def __init__(self, mu, rho):
        super().__init__()
        self.mu = mu
        self.rho = rho
        self.normal = torch.distributions.Normal(0,1)
    
    @property
    def sigma(self):
        return torch.log1p(torch.exp(self.rho))
    
    def sample(self):
        epsilon = self.normal.sample(self.rho.size()).to(DEVICE)
        return self.mu + self.sigma * epsilon
    
    def log_prob(self, input):
        return (-math.log(math.sqrt(2 * math.pi))
                - torch.log(self.sigma)
                - ((input - self.mu) ** 2) / (2 * self.sigma ** 2)).sum()
    
    
# Use a scaled mixture of two Gaussians for the prior distribution on the weights
# These prior parameters are fixed and will not change during training, we therefore
# dont need to use reparameterized Gaussian here:
class ScaleMixtureGaussian(object):
    def __init__(self, pi, sigma1, sigma2):
        super().__init__()
        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.gaussian1 = torch.distributions.Normal(0,sigma1)
        self.gaussian2 = torch.distributions.Normal(0,sigma2)
    
    def log_prob(self, input):
        prob1 = torch.exp(self.gaussian1.log_prob(input))
        prob2 = torch.exp(self.gaussian2.log_prob(input))
        return (torch.log(self.pi * prob1 + (1-self.pi) * prob2)).sum()

PI = 0.5
SIGMA_1 = torch.FloatTensor([math.exp(-0)])
SIGMA_2 = torch.FloatTensor([math.exp(-6)])


# Single bayesian network layer:
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Weight parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5,-4))
        self.weight = Gaussian(self.weight_mu, self.weight_rho)
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5,-4))
        self.bias = Gaussian(self.bias_mu, self.bias_rho)
        # Prior distributions
        self.weight_prior = ScaleMixtureGaussian(PI, SIGMA_1, SIGMA_2)
        self.bias_prior = ScaleMixtureGaussian(PI, SIGMA_1, SIGMA_2)
        self.log_prior = 0
        self.log_variational_posterior = 0
        
    def forward(self, input, sample=False, calculate_log_probs=False):
        if self.training or sample:
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            weight = self.weight.mu
            bias = self.bias.mu
        if self.training or calculate_log_probs:
            self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
            self.log_variational_posterior = self.weight.log_prob(weight) + self.bias.log_prob(bias)
        else:
            self.log_prior, self.log_variational_posterior = 0, 0

        return F.linear(input, weight, bias)
    
# Bayesian Neural Network consisting of two 2 fully connected layers:
class BayesianNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Flere nodes?
        self.l1 = BayesianLinear(2, 100)
        self.l2 = BayesianLinear(100, 100)
        self.l3 = BayesianLinear(100, 4)
    
    def forward(self, x, sample=False, return_pre=False):
        x = x.view(-1, 2)
        x = F.relu(self.l1(x, sample))
        x = F.relu(self.l2(x, sample))
        x_pre = self.l3(x, sample)
        x = F.softmax(x_pre, dim=1)
        if return_pre:
            return x, x_pre
        else:
            return x
    
    def log_prior(self):
        return self.l1.log_prior \
               + self.l2.log_prior \
               + self.l3.log_prior
    
    def log_variational_posterior(self):
        return self.l1.log_variational_posterior \
               + self.l2.log_variational_posterior \
               + self.l3.log_variational_posterior
    
    def sample_elbo(self, input, target, samples=SAMPLES):
        outputs = torch.zeros(samples, BATCH_SIZE, CLASSES)
        log_priors = torch.zeros(samples)
        log_variational_posteriors = torch.zeros(samples)
        for i in range(samples):
            outputs[i] = self(input, sample=True)
            log_priors[i] = self.log_prior()
            log_variational_posteriors[i] = self.log_variational_posterior()
        log_prior = log_priors.mean()
        log_variational_posterior = log_variational_posteriors.mean()
        negative_log_likelihood = F.nll_loss(outputs.mean(0), target, size_average=False)
        loss = (log_variational_posterior - log_prior)/NUM_BATCHES + negative_log_likelihood
        return loss

net_200 = BayesianNetwork()


# Training:
def train(network, optimizer, epoch):
    network.train()
    for batch_idx, data, target in zip(range(len(X_train)), X_train, y_train):
        network.zero_grad()
        loss = network.sample_elbo(data, target)
        loss.backward()
        optimizer.step()

optimizer = optim.Adam(net_200.parameters())
print("Start")
for epoch in range(TRAIN_EPOCHS):
    train(net_200, optimizer, epoch)
    print(f"Epoch {epoch+1}/{TRAIN_EPOCHS}")
print("Ferdig")
