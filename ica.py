import torch
import torch.nn.functional as F
from sklearn.decomposition import FastICA
from torch.distributions import Laplace

from mcc import calc_disent_metrics


def generate_ica_data(source_dim=3, batch_size=4096, theta=1.55):
    A = 1.6  # -0.8
    B = 3.7
    loc = 1.1
    scale = .1
    distribution = Laplace(loc, scale)
    # distribution = torch.distributions.Uniform(-1, 1)

    S = distribution.sample(torch.Size([batch_size, source_dim]))
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

    return S, X, theta


# S, X, theta = generate_ica_data()


def ica_treatment_effect_estimation(X, S, random_state=0, whiten="unit-variance"):
    ica = FastICA(n_components=X.shape[1], random_state=random_state, whiten=whiten)
    S_hat = ica.fit_transform(X)
    results = calc_disent_metrics(S, S_hat)
    print(results["permutation_disentanglement_score"])
    # resolve the permutations
    permuted_mixing = ica.mixing_[:, results["munkres_sort_idx"].astype(int)]
    # normalize to get 1 at epsilon -> Y
    permuted_scaled_mixing = permuted_mixing / permuted_mixing.diagonal()

    treatment_effect_estimate = permuted_scaled_mixing[-1, -2]

    return treatment_effect_estimate


# treatment_effect_estimate = ica_treatment_effect_estimation(X, S)

# print(f"Estimated vs true treatment effect: {treatment_effect_estimate}, {theta}")
