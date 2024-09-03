import torch
from torch import nn
from torch.distributions import Laplace
import torch.nn.functional as F
from sklearn.decomposition import FastICA
from mcc import calc_disent_metrics


bs = 4096
obs_dim = 3
hid_dim = 3
A = 1.6  # -0.8
B = 3.7
C = 0.  # 1.55
theta = 1.55
loc = 1.1
scale = .1

#
distribution = Laplace(loc, scale)
# distribution = torch.distributions.Uniform(-1, 1)

# Sample data (batch size 10, 10 features)
S = distribution.sample(torch.Size([bs, hid_dim]))

X = S.clone()
X[:, 0] = S[:, 0]
X[:, 1] = S[:, 1] + A * F.leaky_relu(S[:, 0], negative_slope=.25)
X[:, 2] = S[:, 2] + theta * S[:, 1] + (theta * A + B) * F.leaky_relu(S[:, 0], negative_slope=.25)

# mixing = torch.tensor(
#     [
#         [1, 0, 0],
#         [A, 1, 0],
#         [C, B, 1],
#     ]
# )

# W = mixing[1:, :]

# X = S @ mixing.T

X_train = X  # [:, 1:]#S@W.T # same as X[:,1:]


ica = FastICA(n_components=obs_dim,       random_state=0,        whiten="unit-variance")
S_hat = ica.fit_transform(X_train.cpu())

results = calc_disent_metrics(S, S_hat)


# resolve the permutations
permuted_mixing = ica.mixing_[:, results["munkres_sort_idx"].astype(int) ]

# normalize to get 1 at epsilon -> Y
permuted_scaled_mixing = permuted_mixing/ permuted_mixing.diagonal()

print(f"Estimated vs true treatment effect: {permuted_scaled_mixing[-1, 1]}, {theta}")
