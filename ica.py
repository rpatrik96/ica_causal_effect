import numpy as np
import torch
import torch.nn.functional as F
from bartpy2.sklearnmodel import SklearnModel
from causallib.estimation import Standardization, IPW
from sklearn.decomposition import FastICA
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from torch.distributions import Laplace

from mcc import calc_disent_metrics


def generate_ica_data(n_covariates=1, n_treatments=1, batch_size=4096, use_nonlinear=True, sparse_prob=0.4):
    # Create sparse matrix of shape (n_treatments x n_covariates)
    binary_mask = torch.bernoulli(torch.ones(n_treatments, n_covariates) * sparse_prob)
    random_coeffs = torch.randn(n_treatments, n_covariates)
    A_covariates = binary_mask * random_coeffs

    # A_treatments = torch.tensor([1.55, 1.75, 1.85, 2.0, 1.7])[:n_treatments]  # Coefficients for treatments
    theta = torch.tensor([1.55, 0.65, -2.45, 1.75, -1.35])[:n_treatments]  # Vector of thetas matching n_treatments
    B = torch.randn(n_covariates)  # Base effects on outcome per covariate

    loc = 4.1
    scale = .5
    distribution = Laplace(loc, scale)
    # distribution = torch.distributions.Uniform(-1, 1)

    source_dim = n_covariates + n_treatments + 1  # +1 for outcome
    S = distribution.sample(torch.Size([batch_size, source_dim]))
    X = S.clone()

    # Define activation function based on use_nonlinear flag
    activation = lambda x: F.leaky_relu(x, negative_slope=.25) if use_nonlinear else x

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


# S, X, theta = generate_ica_data()


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

    # print(permuted_scaled_mixing[-1, :])
    if n_treatments ==1:
        treatment_effect_estimate = treatment_effect_estimate[0]

    return treatment_effect_estimate, results["permutation_disentanglement_score"]


def main():
    sample_sizes = [100, 200, 500, 1000, 2000, 5000]
    n_dims = [10, 20, 50]
    n_treatments = [1]#, 2, 5]
    n_seeds = 20

    # Initialize dictionary to store results
    results_dict = {
        'sample_sizes': [],
        'n_covariates': [],
        'n_treatments': [],
        'true_params': [],
        'treatment_effects': [],
        'mccs': []
    }

    for n_samples in sample_sizes:
        for n_covariates in n_dims:
            for n_treatment in n_treatments:

                # n_treatment = 2
                # n_samples = 500
                # n_dims = 10
                S, X, true_params = generate_ica_data(batch_size=n_samples,
                                                      n_covariates=n_covariates,
                                                      n_treatments=n_treatment,
                                                      use_nonlinear=False,
                                                      sparse_prob=0.4)

                # from econml.dml import LinearDML
                # from sklearn.linear_model import LassoCV
                # from sklearn.ensemble import RandomForestRegressor
                #
                # treatment_indices = torch.arange(n_covariates, n_covariates + n_treatment)
                #
                # est = LinearDML(model_y=RandomForestRegressor(),
                #                 model_t=RandomForestRegressor(),
                #                 linear_first_stages=True)
                # est.fit(Y=X[:, -1].reshape(1,-1),T= X[:, treatment_indices], X=X[:, :n_covariates], W=None)
                # treatment_effect = est.effect(X=X[:, :n_covariates], T0=X[:, treatment_indices], T1=torch.zeros_like(X[:, treatment_indices]).numpy())
                #
                # print(treatment_effect, true_params)
                # exit(0)

                # from sklearn.linear_model import LinearRegression
                # import numpy as np
                #
                # treatment_indices = torch.arange(n_covariates, n_covariates + n_treatment)
                #
                # # Step 1: Fit initial outcome model Q(T, X)
                # outcome_model = LinearRegression()
                #
                # outcome_model.fit(X[:, :-1].numpy(), X[:, -1].numpy())
                # Q_hat = outcome_model.predict(X[:, :-1].numpy())
                #
                # # Step 2: Fit treatment model g(T | X)
                # treatment_model = LinearRegression()
                # T = X[:, treatment_indices].numpy()
                # Y = X[:, -1].numpy()
                # treatment_model.fit(X[:, :n_covariates].numpy(), T)
                # T_hat = treatment_model.predict(X[:, :n_covariates].numpy())
                #
                #
                #
                # # Step 3: Compute clever covariate h(T | X)
                # h = 1. / (T - T_hat)
                #
                # # Step 4: Update outcome model
                # epsilon = np.sum(h * (Y - Q_hat)) / np.sum(h ** 2)
                # Q_star = Q_hat + epsilon * h
                #
                # # Step 5: Estimate the causal effect
                # causal_effect = np.mean(Q_star)
                # print(f"{true_params} vs {causal_effect=}")
                #
                # tmle = TMLE(
                #     Standardization(GradientBoostingRegressor()),
                #     IPW(GradientBoostingClassifier()),
                # )
                #
                # tmle.fit(X[:, :n_covariates].numpy(), T, Y)

                # bart_treatment_effects = bart_treatment_effect_estimation(X, n_covariates, n_treatment)

                # print(f"{true_params=} vs {bart_treatment_effects=}")

                # exit(0)

                for seed in range(n_seeds):
                    treatment_effects, mcc = ica_treatment_effect_estimation(X, S,
                                                                             random_state=seed,
                                                                             check_convergence=False,
                                                                             n_treatments=n_treatment)

                    # Store results in dictionary
                    results_dict['sample_sizes'].append(n_samples)
                    results_dict['n_covariates'].append(n_covariates)
                    results_dict['n_treatments'].append(n_treatment)
                    results_dict['true_params'].append(true_params)
                    results_dict['treatment_effects'].append(treatment_effects)
                    results_dict['mccs'].append(mcc)

                    # print(f"\nResults for n_samples={n_samples}, n_covariates={n_covariates}, n_treatments={n_treatment}, seed={seed}")
    # print(f"True treatment effect: {true_params}")
    # print(f"Est treatment effect: {treatment_effects}")
    # print(f"Mean correlation coefficient: {mcc}")

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
            est_params = [results_dict['treatment_effects'][i] for i in indices]

            # Calculate MSE for each dimension
            mse = {dim: [] for dim in set(dimensions)}
            for dim, true_param, est_param in zip(dimensions, true_params, est_params):
                if est_param is not None:  # Handle cases where estimation failed
                    errors = [(est - true) ** 2 for est, true in zip(est_param, true_param)]
                    mse[dim].append(np.mean(errors))
                else:
                    mse[dim].append(np.nan)

            for dim, errors in mse.items():
                mse[dim] = np.array(mse[dim])

            plt.errorbar(mse.keys(), np.mean(list(mse.values()), axis=1), yerr=np.std(list(mse.values()), axis=1),
                         fmt='o-', capsize=5,
                         label=f'n_treatments={n_treatment}')

        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Number of Dimensions (Covariates)')
        plt.ylabel('Mean Squared Error')
        plt.title(f'ICA Treatment Effect MSE vs Dimensions\n(n_samples={n_samples})')
        plt.grid(True)
        plt.legend()

        plt.savefig(f'ica_mse_vs_dim_n{n_samples}.svg')
        plt.close()


# from bartpy2.extensions.baseestimator import ResidualBART
# from sklearn.linear_model import LinearRegression

from causallib.estimation import TMLE


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
    main()
