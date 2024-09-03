import os
import sys

import matplotlib

matplotlib.use('Agg')
import numpy as np
from sklearn.linear_model import Lasso
from main_estimation import all_together_cross_fitting
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import joblib
import argparse
from joblib import delayed, Parallel


def experiment(x, eta, epsilon, treatment_effect, treatment_support, treatment_coef, outcome_support, outcome_coef,
               eta_second_moment, eta_third_moment, lambda_reg):
    # Generate price as a function of co-variates
    treatment = np.dot(x[:, treatment_support], treatment_coef) + eta
    # Generate demand as a function of price and co-variates
    outcome = treatment_effect * treatment + np.dot(x[:, outcome_support], outcome_coef) + epsilon
    model_treatment = Lasso(alpha=lambda_reg)
    model_outcome = Lasso(alpha=lambda_reg)
    return all_together_cross_fitting(x, treatment, outcome, eta_second_moment, eta_third_moment,
                                      model_treatment=model_treatment, model_outcome=model_outcome)


def main(args):
    os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'
    parser = argparse.ArgumentParser(
        description="Second order orthogonal ML!")
    parser.add_argument("--n_samples", dest="n_samples",
                        type=int, help='n_samples', default=5000)
    parser.add_argument("--n_dim", dest="n_dim",
                        type=int, help='n_dim', default=1000)
    parser.add_argument("--n_experiments", dest="n_experiments",
                        type=int, help='n_experiments', default=2000)
    parser.add_argument("--support_size", dest="support_size",
                        type=int, help='support_size', default=400)
    parser.add_argument("--seed", dest="seed",
                        type=int, help='seed', default=12143)
    parser.add_argument("--sigma_q", dest="sigma_q",
                        type=int, help='sigma_q', default=1)
    parser.add_argument("--output_dir", dest="output_dir", type=str, default=".")
    opts = parser.parse_args(args)

    np.random.seed(opts.seed)

    '''
    We will work with a sparse linear model with high dimensional co-variates
    '''
    # Number of (price, demand, co-variate) samples
    n_samples = opts.n_samples
    # Dimension of co-variates
    n_dim = opts.n_dim
    # How many experiments to run to see the distribution of the recovered coefficient between price and demand
    n_experiments = opts.n_experiments
    support_size = opts.support_size
    print("Support size of sparse functions: {}".format(support_size))

    '''
    True parameters
    '''

    # Support and coefficients for treatment as function of co-variates
    treatment_support = np.random.choice(range(n_dim), size=support_size, replace=False)
    treatment_coef = np.random.uniform(0, 5, size=support_size)
    print("Support of treatment as function of co-variates: {}".format(treatment_support))
    print("Coefficients of treatment as function of co-variates: {}".format(treatment_coef))

    # Distribution of residuals of treatment
    discounts = np.array([0, -.5, -2., -4.])
    probs = np.array([.65, .2, .1, .05])
    mean_discount = np.dot(discounts, probs)
    eta_sample = lambda x: np.array(
        [discounts[i] - mean_discount for i in np.argmax(np.random.multinomial(1, probs, x), axis=1)])
    # Calculate moments of the residual distribution
    eta_second_moment = np.dot(probs, (discounts - mean_discount) ** 2)
    eta_third_moment = np.dot(probs, (discounts - mean_discount) ** 3)
    eta_fourth_moment = np.dot(probs, (discounts - mean_discount) ** 4)
    print("Second Moment of Eta: {:.2f}".format(eta_second_moment))
    print("Third Moment of Eta: {:.2f}".format(eta_third_moment))
    print("Non-Gaussianity Criterion, E[eta^4] - 3 E[eta^2]^2: {:.2f}".format(
        eta_fourth_moment - 3 * eta_second_moment ** 2))

    # Support and coefficients for outcome as function of co-variates
    outcome_support = treatment_support  # np.random.choice(range(n_dim), size=support_size, replace=False)
    outcome_coef = np.random.uniform(0, 5, size=support_size)
    print("Support of outcome as function of co-variates: {}".format(outcome_support))
    print("Coefficients of outcome as function of co-variates: {}".format(outcome_coef))

    # Distribution of outcome residuals
    sigma_q = opts.sigma_q
    epsilon_sample = lambda x: np.random.uniform(-sigma_q, sigma_q, size=x)

    treatment_effect = 3.0

    true_coef_p = np.zeros(n_dim)
    true_coef_p[treatment_support] = treatment_coef
    true_coef_q = np.zeros(n_dim)
    true_coef_q[outcome_support] = outcome_coef
    true_coef_q[treatment_support] += treatment_effect * treatment_coef
    print(true_coef_q[outcome_support])
    '''
    Run  the experiments.
    '''
    # Coefficients recovered by orthogonal ML
    ortho_rec_tau = []
    first_stage_mse = []
    lambda_reg = np.sqrt(np.log(n_dim) / (n_samples))
    results = Parallel(n_jobs=-1, verbose=1)(delayed(experiment)(
        np.random.normal(size=(n_samples, n_dim)),
        eta_sample(n_samples),
        epsilon_sample(n_samples),
        treatment_effect, treatment_support, treatment_coef, outcome_support, outcome_coef, eta_second_moment,
        eta_third_moment, lambda_reg
    ) for t in range(n_experiments))

    ortho_rec_tau = [[ortho_ml, robust_ortho_ml, robust_ortho_est_ml, robust_ortho_est_split_ml] for
                     ortho_ml, robust_ortho_ml, robust_ortho_est_ml, robust_ortho_est_split_ml, _, _ in results]
    first_stage_mse = [[np.linalg.norm(true_coef_p - coef_p), np.linalg.norm(true_coef_q - coef_q)] for
                       _, _, _, _, coef_p, coef_q in results]

    print("Done with experiments!")

    def plot_estimates(estimate_list, true_tau, title="Histogram of estimates"):
        # the histogram of the data
        n, bins, patches = plt.hist(estimate_list, 40, normed=1, facecolor='green', alpha=0.75)
        sigma = np.std(estimate_list)
        mu = np.mean(estimate_list)
        # add a 'best fit' line
        y = mlab.normpdf(bins, mu, sigma)
        l = plt.plot(bins, y, 'r--', linewidth=1)
        plt.plot([treatment_effect, treatment_effect], [0, np.max(y)], 'b--', label='true effect')
        plt.title("{}. mean: {:.2f}, sigma: {:.2f}".format(title, mu, sigma))
        plt.legend()
        return np.abs(true_tau - mu), sigma

    '''
    Plotting histograms
    '''
    plt.figure(figsize=(25, 5))
    plt.subplot(1, 4, 1)
    bias_ortho, sigma_ortho = plot_estimates(np.array(ortho_rec_tau)[:, 0].flatten(), treatment_effect,
                                             title="Orthogonal estimates")
    plt.subplot(1, 4, 2)
    plot_estimates(np.array(ortho_rec_tau)[:, 1].flatten(), treatment_effect, title="Second order orthogonal")
    plt.subplot(1, 4, 3)
    plot_estimates(np.array(ortho_rec_tau)[:, 2].flatten(), treatment_effect,
                   title="Second order orthogonal with estimates")
    plt.subplot(1, 4, 4)
    bias_second, sigma_second = plot_estimates(np.array(ortho_rec_tau)[:, 3].flatten(), treatment_effect,
                                               title="Second order orthogonal with estimates on third sample")
    plt.tight_layout()
    plt.savefig(os.path.join(opts.output_dir,
                             'recovered_coefficients_from_each_method_n_samples_{}_n_dim_{}_n_exp_{}_support_{}_sigma_q_{}.png'.format(
                                 n_samples, n_dim, n_experiments, support_size, sigma_q)), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(opts.output_dir,
                             'recovered_coefficients_from_each_method_n_samples_{}_n_dim_{}_n_exp_{}_support_{}_sigma_q_{}.pdf'.format(
                                 n_samples, n_dim, n_experiments, support_size, sigma_q)), dpi=300, bbox_inches='tight')

    print("Ortho ML MSE: {}".format(bias_ortho ** 2 + sigma_ortho ** 2))
    print("Second Order ML MSE: {}".format(bias_second ** 2 + sigma_ortho ** 2))

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.title("Model_p error")
    plt.hist(np.array(first_stage_mse)[:, 0].flatten())
    plt.subplot(1, 2, 2)
    plt.hist(np.array(first_stage_mse)[:, 1].flatten())
    plt.title("Model_q error")
    plt.savefig(os.path.join(opts.output_dir,
                             'model_errors_n_samples_{}_n_dim_{}_n_exp_{}_support_{}_sigma_q_{}.png'.format(n_samples,
                                                                                                            n_dim,
                                                                                                            n_experiments,
                                                                                                            support_size,
                                                                                                            sigma_q)),
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(opts.output_dir,
                             'model_errors_n_samples_{}_n_dim_{}_n_exp_{}_support_{}_sigma_q_{}.pdf'.format(n_samples,
                                                                                                            n_dim,
                                                                                                            n_experiments,
                                                                                                            support_size,
                                                                                                            sigma_q)),
                dpi=300, bbox_inches='tight')

    joblib.dump(ortho_rec_tau, os.path.join(opts.output_dir,
                                            'recovered_coefficients_from_each_method_n_samples_{}_n_dim_{}_n_exp_{}_support_{}_sigma_q_{}'.format(
                                                n_samples, n_dim, n_experiments, support_size, sigma_q)))
    joblib.dump(first_stage_mse, os.path.join(opts.output_dir,
                                              'model_errors_n_samples_{}_n_dim_{}_n_exp_{}_support_{}_sigma_q_{}'.format(
                                                  n_samples, n_dim, n_experiments, support_size, sigma_q)))


if __name__ == "__main__":
    main(sys.argv[1:])
