# Follows implementation by: https://github.com/nitarshan/bayes-by-backprop/blob/master/Weight%20Uncertainty%20in%20Neural%20Networks.ipynb


import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split


from tensorboardX import SummaryWriter
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from tqdm import tqdm, trange

from sklearn.preprocessing import MinMaxScaler
minmaxscaler = MinMaxScaler(feature_range=(0, 1))


writer = SummaryWriter()
sns.set()
sns.set_style("dark")
sns.set_palette("muted")
sns.set_color_codes("muted")


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOADER_KWARGS = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
print(torch.cuda.is_available())



breast_cancer_data = pd.read_csv("breast_cancer_data.csv").drop(columns=["id", "Unnamed: 32"])

X = breast_cancer_data.iloc[:, 1:].to_numpy()
y = breast_cancer_data.iloc[:, 0].replace({"M": 0, "B": 1}).to_numpy()

minmaxscaler.fit(X)
X = minmaxscaler.transform(X)


X, y = torch.from_numpy(X).float(), torch.from_numpy(y).long()


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=10, stratify=y)
X_test.shape, y_train.shape

X_train = X_train[:450, :]
y_train = y_train[:450]

X_test = X_test[:110, :]
y_test = X_test[:110]

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

CLASSES = 2
TRAIN_EPOCHS = 10 # Endre
SAMPLES = int(len(X)*0.8)
TEST_SAMPLES = int(len(X)*0.2)

assert (TRAIN_SIZE % BATCH_SIZE) == 0
assert (TEST_SIZE % TEST_BATCH_SIZE) == 0



PI = 0.5
SIGMA_1 = torch.FloatTensor([math.exp(-0)])
SIGMA_2 = torch.FloatTensor([math.exp(-6)])


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
    

class BayesianNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Flere nodes?
        self.l1 = BayesianLinear(30, 100)
        self.l2 = BayesianLinear(100, 100)
        self.l3 = BayesianLinear(100, 2)
    
    def forward(self, x, sample=False, return_pre=False):
        x = x.view(-1, 30)
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

cancer_net = BayesianNetwork()


# Training:
def train(net, optimizer, epoch):
    net.train()
    for batch_idx, data, target in zip(range(len(X_train)), X_train, y_train):
        net.zero_grad()
        #print(data)
        loss = cancer_net.sample_elbo(X_train[0], y_train[0])
        loss.backward()
        optimizer.step()

optimizer = optim.Adam(cancer_net.parameters())
print("Start")
for epoch in range(TRAIN_EPOCHS):
    train(cancer_net, optimizer, epoch)
    print(f"Epoch {epoch+1}/{TRAIN_EPOCHS}")
print("Ferdig")