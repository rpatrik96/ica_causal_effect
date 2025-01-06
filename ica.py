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
    from warnings import catch_warnings, filterwarnings
    from sklearn.exceptions import ConvergenceWarning

    tol = 1e-4  # Initial tolerance
    max_tol = 4e-2  # Maximum tolerance to try
    
    for attempt in range(10):
        with catch_warnings(record=True) as w:
            # filterwarnings('error')
            
            ica = FastICA(n_components=X.shape[1], random_state=random_state+attempt, 
                        whiten=whiten, tol=tol)
            S_hat = ica.fit_transform(X)
            
            if len(w) > 0:
                print(f"warning at {attempt=}")
                # Increase tolerance for next attempt
                tol = min(tol * 2, max_tol)
                if tol >= max_tol:  # Stop if max tolerance reached
                    # Run one final time with max tolerance
                    ica = FastICA(n_components=X.shape[1], random_state=random_state,
                                whiten=whiten, tol=max_tol)
                    S_hat = ica.fit_transform(X)
                    break
            else:
                print(f"success at {attempt=}")
                break

    results = calc_disent_metrics(S, S_hat)
    # resolve the permutations
    permuted_mixing = ica.mixing_[:, results["munkres_sort_idx"].astype(int)]
    # normalize to get 1 at epsilon -> Y
    permuted_scaled_mixing = permuted_mixing / permuted_mixing.diagonal()

    treatment_effect_estimate = permuted_scaled_mixing[-1, -2]

    return treatment_effect_estimate, results["permutation_disentanglement_score"]


# treatment_effect_estimate = ica_treatment_effect_estimation(X, S)

# print(f"Estimated vs true treatment effect: {treatment_effect_estimate}, {theta}")
