import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from bartpy2.sklearnmodel import SklearnModel
from sklearn.decomposition import FastICA
from sklearn.linear_model import LinearRegression
from torch.distributions import Laplace
from tueplots import bundles

from mcc import calc_disent_metrics
from plot_utils import plot_typography




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
                                    n_treatments=1, verbose=False, fun="logcosh"):
    from warnings import catch_warnings

    tol = 1e-4  # Initial tolerance
    max_tol = 1e-2  # Maximum tolerance to try

    for attempt in range(10):
        with catch_warnings(record=True) as w:
            # filterwarnings('error')

            ica = FastICA(n_components=X.shape[1], random_state=random_state + attempt, max_iter=1000,
                          whiten=whiten, tol=tol,fun=fun)
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
    import matplotlib.pyplot as plt
    plt.rcParams.update(bundles.icml2022(usetex=True))
    plot_typography()

    sample_sizes = [100, 200, 500, 1000, 2000, 5000]
    n_dims = [2,5,10,20, 50]
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
                    errors = [np.linalg.norm(est - true) for est, true in zip(est_param, true_param)]
                    mse[dim].append(np.mean(errors))
                else:
                    mse[dim].append(np.nan)

                if est_param_iv is not None:  # Handle cases where estimation failed
                    errors_iv = [np.linalg.norm(est - true) for est, true in zip(est_param_iv, true_param)]
                    mse_iv[dim].append(np.mean(errors_iv))
                else:
                    mse_iv[dim].append(np.nan)

            for dim in sorted(mse.keys()):
                mse[dim] = np.array(mse[dim])
                mse_iv[dim] = np.array(mse_iv[dim])

            # Plot ICA error bars
            sorted_mse_keys = sorted(mse.keys())
            ica_handle = plt.errorbar(sorted_mse_keys, np.mean([mse[key] for key in sorted_mse_keys], axis=1), 
                                      yerr=np.std([mse[key] for key in sorted_mse_keys], axis=1),
                                      fmt='o-', capsize=5, label=f'{n_treatment} (ICA)')

            # Plot IV error bars
            sorted_mse_iv_keys = sorted(mse_iv.keys())
            iv_handle = plt.errorbar(sorted_mse_iv_keys, np.mean([mse_iv[key] for key in sorted_mse_iv_keys], axis=1),
                                     yerr=np.std([mse_iv[key] for key in sorted_mse_iv_keys], axis=1),
                                     fmt='o-', capsize=5, label=f'{n_treatment}')


        plt.legend(loc='lower center', ncol=int(n_treatment/2), bbox_to_anchor=(0.5, -0.15))

        # plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$\dim X$')
        plt.ylabel(r'$\Vert\theta-\hat{\theta} \Vert_2$')
        plt.grid(True, which="both", linestyle='-.', linewidth=0.5)
        plt.legend()
        plt.xticks(ticks=dimensions, labels=[int(dim) for dim in dimensions])

        plt.savefig(f'ica_iv_mse_vs_dim_n{n_samples}.svg')
        plt.close()


def main_nonlinear():
    import matplotlib.pyplot as plt
    sample_sizes = [100, 200, 500, 1000, 2000, 5000]
    n_dims = [2,5,10, 20, 50]
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

    plt.rcParams.update(bundles.icml2022(usetex=True))
    plot_typography()

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
                    errors = [np.linalg.norm(est - true) for est, true in zip(est_param, true_param)]
                    mse[dim].append(np.mean(errors))
                else:
                    mse[dim].append(np.nan)


            for dim in sorted(mse.keys()):
                mse[dim] = np.array(mse[dim])


            sorted_keys = sorted(mse.keys())
            plt.errorbar(sorted_keys, np.mean([mse[key] for key in sorted_keys], axis=1), 
                         yerr=np.std([mse[key] for key in sorted_keys], axis=1),
                         fmt='o-', capsize=5,
                         label=f'{slope}')

        plt.rcParams.update(bundles.icml2022(usetex=True))
        plot_typography()
        plt.yscale('log')
        plt.xlabel(r'$\dim X$')
        plt.ylabel(r'$\Vert\theta-\hat{\theta} \Vert_2$')
        plt.grid(True, which="both", linestyle='-.', linewidth=0.5)
        plt.tight_layout()
        plt.xticks(ticks=dimensions, labels=[int(dim) for dim in dimensions])
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



def main_fun():
    import matplotlib.pyplot as plt
    plt.rcParams.update(bundles.icml2022(usetex=True))
    plot_typography()

    n_samples = 5000
    n_covariates = 50
    n_treatment = 1
    n_seeds = 20

    # Initialize dictionary to store results
    results_dict = {
        'sample_sizes': [],
        'n_covariates': [],
        'n_treatments': [],
        'true_params': [],
        'treatment_effects': [],
        'treatment_effects_iv': [],
        'mccs': [],
        'fun_options' : []
    }



    S, X, true_params = generate_ica_data(batch_size=n_samples,
                                            n_covariates=n_covariates,
                                            n_treatments=n_treatment,
                                            slope=1.,
                                            sparse_prob=0.4)
    fun_options = ["logcosh", "exp", "cube"]

    for fun in fun_options:
        for seed in range(n_seeds):
            treatment_effects, mcc = ica_treatment_effect_estimation(X, S,
                                                                     random_state=seed,
                                                                     check_convergence=False,
                                                                     n_treatments=n_treatment,
                                                                     fun=fun)

            # Store results in dictionary
            results_dict['sample_sizes'].append(n_samples)
            results_dict['n_covariates'].append(n_covariates)
            results_dict['n_treatments'].append(n_treatment)
            results_dict['true_params'].append(true_params)
            results_dict['treatment_effects'].append(treatment_effects)
            results_dict['mccs'].append(mcc)
            results_dict['fun_options'].append(fun)

    import matplotlib.pyplot as plt
    import numpy as np
    plt.figure(figsize=(10, 6))


    # Process data based on 'fun' and calculate MSE
    mse_by_fun = {fun: [] for fun in set(fun_options)}
    for true_param, est_param, fun_label in zip(results_dict['true_params'], results_dict['treatment_effects'], results_dict['fun_options']):
        if est_param is not None:  # Handle cases where estimation failed
            errors = [np.linalg.norm(est - true) for est, true in zip(est_param, true_param)]
            mse_by_fun[fun_label].append(np.mean(errors))
        else:
            mse_by_fun[fun_label].append(np.nan)

    # Create a bar plot for each 'fun' on the x-axis
    bar_width = 0.2
    bar_positions = np.arange(len(fun_options))

    # Calculate mean and standard deviation for each 'fun'
    means = [np.mean(mse_by_fun[fun]) for fun in fun_options]
    std_devs = [np.std(mse_by_fun[fun]) for fun in fun_options]

    plt.bar(
        bar_positions,
        means,
        yerr=std_devs,
        width=bar_width,
        capsize=5,
        tick_label=fun_options,
        label=f'{n_treatment} (ICA)'
    )
    plt.xlabel('FastICA objective function')

    # plt.legend(loc='lower center', ncol=int(n_treatment/2), bbox_to_anchor=(0.5, -0.15))

    # # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlabel(r'$\dim X$')
    plt.ylabel(r'$\Vert\theta-\hat{\theta} \Vert_2$')
    # plt.grid(True, which="both", linestyle='-.', linewidth=0.5)
    # plt.legend()
    # plt.xticks(ticks=dimensions, labels=[int(dim) for dim in dimensions])

    plt.savefig(f'ica_mse_fun.svg')
    plt.close()


def main_sparsity():
    import matplotlib.pyplot as plt
    plt.rcParams.update(bundles.icml2022(usetex=True))
    plot_typography()

    n_samples = 5000
    n_covariates = 50
    n_treatment = 1
    n_seeds = 20
    sparsities = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

    # Initialize dictionary to store results
    results_dict = {
        'sample_sizes': [],
        'n_covariates': [],
        'n_treatments': [],
        'true_params': [],
        'treatment_effects': [],
        'treatment_effects_iv': [],
        'mccs': [],
        'sparsities' : []
    }

    for sparsity in sparsities:
        S, X, true_params = generate_ica_data(batch_size=n_samples,
                                              n_covariates=n_covariates,
                                              n_treatments=n_treatment,
                                              slope=1.,
                                              sparse_prob=sparsity)

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
            results_dict['sparsities'].append(sparsity)

    import matplotlib.pyplot as plt
    import numpy as np
    plt.figure(figsize=(10, 6))


    # Process data based on 'sparsity' and calculate MSE
    mse_by_sparsity = {sparsity: [] for sparsity in set(results_dict['sparsities'])}
    for true_param, est_param, sparsity_label in zip(results_dict['true_params'], results_dict['treatment_effects'], results_dict['sparsities']):
        if est_param is not None:  # Handle cases where estimation failed
            errors = [np.linalg.norm(est - true) for est, true in zip(est_param, true_param)]
            mse_by_sparsity[sparsity_label].append(np.mean(errors))
        else:
            mse_by_sparsity[sparsity_label].append(np.nan)

    # Calculate mean and standard deviation for each 'sparsity'
    sorted_sparsities = sorted(set(results_dict['sparsities']))
    means = [np.mean(mse_by_sparsity[sparsity]) for sparsity in sorted_sparsities]
    std_devs = [np.std(mse_by_sparsity[sparsity]) for sparsity in sorted_sparsities]

    # Create an error bar plot for each 'sparsity' on the x-axis
    bar_positions = np.arange(len(sorted_sparsities))

    plt.errorbar(
        bar_positions,
        means,
        yerr=std_devs,
        fmt='o',
        capsize=5,
        label=f'{n_treatment} (ICA)'
    )
    plt.xticks(bar_positions, sorted_sparsities)
    plt.xlabel(r'Sparsity of $\mathrm{\mathbf{A}}$')

    # plt.legend(loc='lower center', ncol=int(n_treatment/2), bbox_to_anchor=(0.5, -0.15))

    # # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlabel(r'$\dim X$')
    plt.ylabel(r'$\Vert\theta-\hat{\theta} \Vert_2$')
    # plt.grid(True, which="both", linestyle='-.', linewidth=0.5)
    # plt.legend()
    # plt.xticks(ticks=dimensions, labels=[int(dim) for dim in dimensions])

    plt.savefig(f'ica_mse_vs_dim_sparsity.svg')
    plt.close()


if __name__ == "__main__":

    # main_fun()
    main_sparsity()

    print("Running multiple treatment effect estimation with ICA...")
    # main()

    print("Running treatment effect estimation with ICA in nonlinear PLR...")
    # main_nonlinear()


