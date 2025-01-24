import numpy as np
import torch
import torch.nn.functional as F
from bartpy2.sklearnmodel import SklearnModel
from sklearn.decomposition import FastICA
from sklearn.linear_model import LinearRegression
from torch.distributions import Laplace

from mcc import calc_disent_metrics


def generate_ica_data(n_covariates=1, n_treatments=1, batch_size=4096, slope=1., sparse_prob=0.4):
    # Create sparse matrix of shape (n_treatments x n_covariates)
    binary_mask = torch.bernoulli(torch.ones(n_treatments, n_covariates) * sparse_prob)
    random_coeffs = torch.randn(n_treatments, n_covariates)
    A_covariates = binary_mask * random_coeffs

    theta = torch.tensor([1.55, 0.65, -2.45, 1.75, -1.35])[:n_treatments]  # Vector of thetas matching n_treatments
    B = torch.randn(n_covariates)  # Base effects on outcome per covariate

    loc = 4.1
    scale = .5
    distribution = Laplace(loc, scale)

    source_dim = n_covariates + n_treatments + 1  # +1 for outcome
    S = distribution.sample(torch.Size([batch_size, source_dim]))
    X = S.clone()

    # Define activation function based on use_nonlinear flag
    activation = lambda x: F.leaky_relu(x, negative_slope=.25) if slope != 1. else x

    # Covariates remain independent
    X[:, :n_covariates] = S[:, :n_covariates]

    # Treatments depend on all covariates
    treatment_indices = torch.arange(n_covariates, n_covariates + n_treatments)
    X[:, treatment_indices] = S[:, treatment_indices]
    # Modified to use the sparse matrix multiplication
    covariate_effects = activation(S[:, :n_covariates]) @ A_covariates.t()
    X[:, treatment_indices] += covariate_effects

    # Outcome depends on all covariates and treatments
    X[:, -1] = S[:, -1]

    # Add treatment effects 
    X[:, -1] += (theta * X[:, treatment_indices]).sum(dim=1)

    # Add covariate effects 

    # Use the sum of all sparse connections for the outcome
    X[:, -1] += (B * activation(S[:, :n_covariates])).sum(dim=1)

    return S, X, theta


def ica_treatment_effect_estimation(X, S, random_state=0, whiten="unit-variance", check_convergence=False,
                                    n_treatments=1, verbose=False):
    from warnings import catch_warnings

    tol = 1e-4  # Initial tolerance
    max_tol = 1e-2  # Maximum tolerance to try

    for attempt in range(10):
        with catch_warnings(record=True) as w:
            # filterwarnings('error')

            ica = FastICA(n_components=X.shape[1], random_state=random_state + attempt, max_iter=1000,
                          whiten=whiten, tol=tol)
            S_hat = ica.fit_transform(X)

            if len(w) > 0 and check_convergence is True:
                if verbose:
                    print(f"warning at {attempt=}")
                # Increase tolerance for next attempt
                tol = min(tol * 2, max_tol)
                if tol >= max_tol:  # Stop if max tolerance reached
                    return None, None
            else:
                if verbose:
                    print(f"success at {attempt=}")
                break

    results = calc_disent_metrics(S, S_hat)
    # resolve the permutations
    permuted_mixing = ica.mixing_[:, results["munkres_sort_idx"].astype(int)]
    # normalize to get 1 at epsilon -> Y
    permuted_scaled_mixing = permuted_mixing / permuted_mixing.diagonal()

    n_covariates = X.shape[1] - 1 - n_treatments  # Assuming  and 1 outcome
    treatment_effect_estimate = permuted_scaled_mixing[-1, n_covariates:-1]

    return treatment_effect_estimate, results["permutation_disentanglement_score"]


def main():
    sample_sizes = [100, 200, 500, 1000, 2000, 5000]
    n_dims = [10, 20, 50]
    n_treatments = [1, 2, 5]
    n_seeds = 20

    # Initialize dictionary to store results
    results_dict = {
        'sample_sizes': [],
        'n_covariates': [],
        'n_treatments': [],
        'true_params': [],
        'treatment_effects': [],
        'treatment_effects_iv': [],
        'mccs': []
    }

    for n_samples in sample_sizes:
        for n_covariates in n_dims:
            for n_treatment in n_treatments:

                S, X, true_params = generate_ica_data(batch_size=n_samples,
                                                      n_covariates=n_covariates,
                                                      n_treatments=n_treatment,
                                                      slope=1.,
                                                      sparse_prob=0.4)

                treatment_indices = torch.arange(n_covariates, n_covariates + n_treatment).numpy()
                T = X[:, treatment_indices]
                X_cov = X[:, :n_covariates]
                Y = X[:, -1].reshape(-1, )

                import pandas as pd

                # Create a pandas dataframe with separate columns for Y, each column of T, and each column of X
                data = {
                    'Y': Y,
                    **{f'T{k}': T[:, k] for k in range(T.shape[1])},
                    **{f'X{l}': X_cov[:, l] for l in range(X_cov.shape[1])}
                }

                T_names = [f'T{k}' for k in range(T.shape[1])]

                iv_df = pd.DataFrame(data)

                formula = 'Y ~ 1 + ' + ' + '.join([f'T{k}' for k in range(T.shape[1])]) + ' + ' + ' + '.join(
                    [f'X{l}' for l in range(X_cov.shape[1])])

                for seed in range(n_seeds):
                    treatment_effects, mcc = ica_treatment_effect_estimation(X, S,
                                                                             random_state=seed,
                                                                             check_convergence=False,
                                                                             n_treatments=n_treatment)

                    # Fit the IV regression model
                    from linearmodels.iv import IV2SLS
                    iv_model = IV2SLS.from_formula(formula, iv_df).fit()
                    treatment_effects_iv = iv_model.params[T_names]

                    # Store results in dictionary
                    results_dict['sample_sizes'].append(n_samples)
                    results_dict['n_covariates'].append(n_covariates)
                    results_dict['n_treatments'].append(n_treatment)
                    results_dict['true_params'].append(true_params)
                    results_dict['treatment_effects'].append(treatment_effects)
                    results_dict['treatment_effects_iv'].append(treatment_effects_iv)
                    results_dict['mccs'].append(mcc)

    import matplotlib.pyplot as plt
    import numpy as np

    # Create plots for each sample size
    for n_samples in set(results_dict['sample_sizes']):
        plt.figure(figsize=(10, 6))

        # Plot curves for each number of treatments
        for n_treatment in set(results_dict['n_treatments']):
            # Filter results for this sample size and number of treatments
            indices = [i for i, (s, t) in enumerate(zip(results_dict['sample_sizes'], results_dict['n_treatments']))
                       if s == n_samples and t == n_treatment]

            dimensions = [results_dict['n_covariates'][i] for i in indices]
            true_params = [results_dict['true_params'][i] for i in indices]
            est_params_ica = [results_dict['treatment_effects'][i] for i in indices]
            est_params_iv = [results_dict['treatment_effects_iv'][i] for i in indices]

            # Calculate MSE for each dimension
            mse = {dim: [] for dim in set(dimensions)}
            mse_iv = {dim: [] for dim in set(dimensions)}
            for dim, true_param, est_param, est_param_iv in zip(dimensions, true_params, est_params_ica, est_params_iv):
                if est_param is not None:  # Handle cases where estimation failed
                    errors = [(est - true) ** 2 for est, true in zip(est_param, true_param)]
                    mse[dim].append(np.mean(errors))
                else:
                    mse[dim].append(np.nan)

                if est_param_iv is not None:  # Handle cases where estimation failed
                    errors_iv = [(est - true) ** 2 for est, true in zip(est_param_iv, true_param)]
                    mse_iv[dim].append(np.mean(errors_iv))
                else:
                    mse_iv[dim].append(np.nan)

            for dim, errors in mse.items():
                mse[dim] = np.array(mse[dim])
                mse_iv[dim] = np.array(mse_iv[dim])

            plt.errorbar(mse.keys(), np.mean(list(mse.values()), axis=1), yerr=np.std(list(mse.values()), axis=1),
                         fmt='o-', capsize=5,
                         label=f'n_treatments={n_treatment}')

            plt.errorbar(mse_iv.keys(), np.mean(list(mse_iv.values()), axis=1),
                         yerr=np.std(list(mse_iv.values()), axis=1),
                         fmt='o-', capsize=5,
                         label=f'n_treatments={n_treatment}_IV')

        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Number of Dimensions (Covariates)')
        plt.ylabel('Mean Squared Error')
        plt.title(f'Treatment Effect MSE vs Dimensions\n(n_samples={n_samples})')
        plt.grid(True)
        plt.legend()

        plt.savefig(f'ica_iv_mse_vs_dim_n{n_samples}.svg')
        plt.close()


def main_nonlinear():
    sample_sizes = [100, 200, 500, 1000, 2000, 5000]
    n_dims = [10, 20, 50]
    slopes = [0, .1, .2, .5, 1.]
    n_seeds = 20

    # Initialize dictionary to store results
    results_dict = {
        'sample_sizes': [],
        'n_covariates': [],
        'n_treatments': [],
        'true_params': [],
        'treatment_effects': [],
        'slopes': [],
        'mccs': []
    }

    for n_samples in sample_sizes:
        for n_covariates in n_dims:
            for slope in slopes:

                S, X, true_params = generate_ica_data(batch_size=n_samples,
                                                      n_covariates=n_covariates,
                                                      n_treatments=1,
                                                      slope=slope,
                                                      sparse_prob=0.4)

                for seed in range(n_seeds):
                    treatment_effects, mcc = ica_treatment_effect_estimation(X, S,
                                                                             random_state=seed,
                                                                             check_convergence=False,
                                                                             n_treatments=1)

                    # Store results in dictionary
                    results_dict['sample_sizes'].append(n_samples)
                    results_dict['n_covariates'].append(n_covariates)
                    results_dict['true_params'].append(true_params)
                    results_dict['treatment_effects'].append(treatment_effects)
                    results_dict['slopes'].append(slope)
                    results_dict['mccs'].append(mcc)

    import matplotlib.pyplot as plt
    import numpy as np

    # Create plots for each sample size
    for n_samples in set(results_dict['sample_sizes']):
        plt.figure(figsize=(10, 6))

        # Plot curves for each number of slope
        for slope in set(results_dict['slopes']):
            # Filter results for this sample size and number of treatments
            indices = [i for i, (s, t) in enumerate(zip(results_dict['sample_sizes'], results_dict['slopes']))
                       if s == n_samples and t == slope]

            dimensions = [results_dict['n_covariates'][i] for i in indices]
            true_params = [results_dict['true_params'][i] for i in indices]
            est_params_ica = [results_dict['treatment_effects'][i] for i in indices]

            # Calculate MSE for each dimension
            mse = {dim: [] for dim in set(dimensions)}

            for dim, true_param, est_param in zip(dimensions, true_params, est_params_ica):
                if est_param is not None:  # Handle cases where estimation failed
                    errors = [(est - true) ** 2 for est, true in zip(est_param, true_param)]
                    mse[dim].append(np.mean(errors))
                else:
                    mse[dim].append(np.nan)

            for dim, errors in mse.items():
                mse[dim] = np.array(mse[dim])

            plt.errorbar(mse.keys(), np.mean(list(mse.values()), axis=1), yerr=np.std(list(mse.values()), axis=1),
                         fmt='o-', capsize=5,
                         label=f'slope={slope}')

        # plt.xscale('log')
        # plt.yscale('log')
        plt.xlabel('Number of Dimensions (Covariates)')
        plt.ylabel('Mean Squared Error')
        plt.title(f'Treatment Effect MSE vs Dimensions\n(n_samples={n_samples})')
        plt.grid(True)
        plt.legend()

        plt.savefig(f'ica_nonlinear_mse_vs_dim_n{n_samples}.svg')
        plt.close()


def bart_treatment_effect_estimation(X, n_covariates, n_treatment):
    treatment_indices = torch.arange(n_covariates, n_covariates + n_treatment)
    # Train the BART model
    bart_model = SklearnModel(
        n_trees=(n_covariates + n_treatment),
        n_chains=3,
        n_burn=2000,
        n_samples=5000
    )
    # bart_model = ResidualBART(base_estimator=LinearRegression())
    # input is all X and T
    bart_model.fit(X[:, :-1].numpy(), X[:, -1].numpy())

    # Create a range of treatment values
    treatment_range = np.linspace(X[:, treatment_indices].min(axis=0)[0], X[:, treatment_indices].max(axis=0)[0], 100)
    X_pred = torch.zeros((100, n_covariates + n_treatment))
    X_pred[:, :n_covariates] = X[:, :n_covariates].mean()
    X_pred[:, treatment_indices] = torch.from_numpy(treatment_range).reshape(-1, n_treatment).float()
    # Predict outcomes for the treatment range
    predicted_Y = bart_model.predict(X_pred.numpy())

    # Fit a linear regression model to estimate the slope of predicted_Y vs treatment
    model = LinearRegression()
    model.fit(treatment_range, predicted_Y)
    # Extract the slope from the model
    treatment_effects = model.coef_

    return treatment_effects


if __name__ == "__main__":
    print("Running multiple treatment effect estimation with ICA...")
    main()

    print("Running treatment effect estimation with ICA in nonlinear PLR...")
    main_nonlinear()
