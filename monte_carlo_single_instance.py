import os
import sys

import matplotlib

from plot_utils import plot_method_comparison, plot_and_save_model_errors, plot_error_vs_support, \
    plot_error_bars_from_density_estimate

matplotlib.use('Agg')
import numpy as np
from sklearn.linear_model import Lasso
from main_estimation import all_together_cross_fitting
import argparse
from joblib import delayed, Parallel

from ica import ica_treatment_effect_estimation


def experiment(x, eta, epsilon, treatment_effect, treatment_support, treatment_coef, outcome_support, outcome_coef,
               eta_second_moment, eta_third_moment, lambda_reg, check_convergence=False):
    # Generate price as a function of co-variates
    treatment = np.dot(x[:, treatment_support], treatment_coef) + eta
    # Generate demand as a function of price and co-variates
    outcome = treatment_effect * treatment + np.dot(x[:, outcome_support], outcome_coef) + epsilon
    model_treatment = Lasso(alpha=lambda_reg)
    model_outcome = Lasso(alpha=lambda_reg)

    assert (treatment_support == outcome_support).all()
    # print((x[:, treatment_support], treatment, outcome).shape)
    # ica_treatment_effect_estimate = ica_treatment_effect_estimation((x[:, treatment_support], treatment, outcome), (x[:, treatment_support], eta, epsilon))
    # ica_treatment_effect_estimate = ica_treatment_effect_estimation(np.hstack((np.dot(x[:, treatment_support], np.ones_like(treatment_coef)).reshape(-1, 1), treatment.reshape(-1, 1), outcome.reshape(-1, 1))), np.hstack((np.dot(x[:, treatment_support], np.ones_like(treatment_coef)).reshape(-1, 1), eta.reshape(-1,1), epsilon.reshape(-1,1))))
    # ica_treatment_effect_estimate = ica_treatment_effect_estimation(np.hstack((np.dot(x[:, treatment_support], np.ones_like(treatment_coef)).reshape(-1, 1), treatment.reshape(-1, 1), outcome.reshape(-1, 1))), np.hstack((np.dot(x[:, treatment_support], np.ones_like(treatment_coef)).reshape(-1, 1), eta.reshape(-1,1), epsilon.reshape(-1,1))))
    ica_treatment_effect_estimate, ica_mcc = ica_treatment_effect_estimation(
        np.hstack((x[:, treatment_support], treatment.reshape(-1, 1), outcome.reshape(-1, 1))),
        np.hstack((x[:, treatment_support], eta.reshape(-1, 1), epsilon.reshape(-1, 1))), check_convergence=check_convergence)
    # ica_treatment_effect_estimate, ica_mcc = ica_treatment_effect_estimation(np.hstack(( treatment.reshape(-1, 1), outcome.reshape(-1, 1))), np.hstack(( eta.reshape(-1,1), epsilon.reshape(-1,1))))


    print(f"Estimated vs true treatment effect: {ica_treatment_effect_estimate}, {treatment_effect}")

    return *all_together_cross_fitting(x, treatment, outcome, eta_second_moment, eta_third_moment,
                                      model_treatment=model_treatment, model_outcome=model_outcome), ica_treatment_effect_estimate, ica_mcc


def main(args):
    os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'
    parser = argparse.ArgumentParser(
        description="Second order orthogonal ML!")
    parser.add_argument("--n_samples", dest="n_samples",
                        type=int, help='n_samples', default=500)
    parser.add_argument("--n_dim", dest="n_dim",
                        type=int, help='n_dim', default=50)
    parser.add_argument("--n_experiments", dest="n_experiments",
                        type=int, help='n_experiments', default=20)
    # parser.add_argument("--support_size", dest="support_size",
    #                     type=int, help='support_size', default=10)
    parser.add_argument("--seed", dest="seed",
                        type=int, help='seed', default=12143)
    parser.add_argument("--sigma_outcome", dest="sigma_outcome",
                        type=int, help='sigma_outcome', default=1)
    parser.add_argument("--covariate_pdf", dest="covariate_pdf",
                        type=str, help='pdf of covariates', default="uniform")
    parser.add_argument("--output_dir", dest="output_dir", type=str, default="./figures")
    parser.add_argument("--check_convergence", dest="check_convergence", 
                        type=bool, help='check convergence', default=False)

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
    
    # Run experiments for different support sizes
    support_sizes = [2, 5, 10, 20, 50]
    data_samples = [100, 200, 500, 1000, 2000, 5000]
    for n_samples in data_samples:
        print(f"\nRunning experiments with sample size: {n_samples}")
        all_results = []

        for support_size in support_sizes:
            print(f"\nRunning experiments with support size: {support_size}")
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
            sigma_outcome = opts.sigma_outcome
            epsilon_sample = lambda x: np.random.uniform(-sigma_outcome, sigma_outcome, size=x)

            treatment_effect = 3.0

            true_coef_treatment = np.zeros(n_dim)
            true_coef_treatment[treatment_support] = treatment_coef
            true_coef_outcome = np.zeros(n_dim)
            true_coef_outcome[outcome_support] = outcome_coef
            true_coef_outcome[treatment_support] += treatment_effect * treatment_coef
            print(true_coef_outcome[outcome_support])
            '''
            Run the experiments.
            '''

            if opts.covariate_pdf == "gauss":
                x_sample = lambda n_samples, n_dim : np.random.normal(size=(n_samples, n_dim))
            elif opts.covariate_pdf == "uniform":
                x_sample = lambda n_samples, n_dim : np.random.uniform(size=(n_samples, n_dim))

            # Coefficients recovered by orthogonal ML
            lambda_reg = np.sqrt(np.log(n_dim) / (n_samples))
            results = [r for r in Parallel(n_jobs=-1, verbose=1)(delayed(experiment)(
                x_sample(n_samples, n_dim),
                eta_sample(n_samples),
                epsilon_sample(n_samples),
                treatment_effect, treatment_support, treatment_coef, outcome_support, outcome_coef, eta_second_moment,
                eta_third_moment, lambda_reg
            ) for _ in range(n_experiments)) if (opts.check_convergence is False or r[-1] is not None)]

            ortho_rec_tau = [[ortho_ml, robust_ortho_ml, robust_ortho_est_ml, robust_ortho_est_split_ml, ica_treatment_effect_estimate] for
                             ortho_ml, robust_ortho_ml, robust_ortho_est_ml, robust_ortho_est_split_ml, _, _, ica_treatment_effect_estimate, _ in results]
            first_stage_mse = [[np.linalg.norm(true_coef_treatment - coef_treatment), np.linalg.norm(true_coef_outcome - coef_outcome), np.linalg.norm(ica_treatment_effect_estimate-treatment_effect), ica_mcc] for
                               _, _, _, _, coef_treatment, coef_outcome, ica_treatment_effect_estimate, ica_mcc in results]

            all_results.append({
                'n_samples': n_samples,
                'support_size': support_size,
                'ortho_rec_tau': ortho_rec_tau,
                'first_stage_mse': first_stage_mse
            })

            
            print(f"Done with experiments for support size {support_size}!")

            '''
            Plotting histograms
            '''

            biases, sigmas = plot_method_comparison(ortho_rec_tau, treatment_effect, opts.output_dir, n_samples, n_dim, n_experiments, support_size,
                                   sigma_outcome)
            all_results[-1]['biases'] = biases
            all_results[-1]['sigmas'] = sigmas

            plot_and_save_model_errors(first_stage_mse, ortho_rec_tau, opts.output_dir, n_samples, n_dim, n_experiments, support_size,
                                       sigma_outcome)


        plot_error_bars_from_density_estimate(all_results, n_dim, n_experiments, n_samples, opts)

    print("\nDone with all experiments!")


if __name__ == "__main__":
    main(sys.argv[1:])
