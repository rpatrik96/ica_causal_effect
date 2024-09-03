import numpy as np
from sklearn.cross_validation import KFold
from sklearn.linear_model import LassoCV


def all_together(x, treatment, outcome, eta_second_moment, eta_third_moment,
                 model_treatment=LassoCV(alphas=[0.01, 0.1, 0.3, 0.5, 0.9, 5, 10, 20, 100]),
                 model_outcome=LassoCV(alphas=[0.01, 0.1, 0.3, 0.5, 0.9, 5, 10, 20, 100])):
    # Split the data in half, train and test
    split_size = x.shape[0] // 2
    x_train, treatment_train, outcome_train = x[:split_size], treatment[:split_size], outcome[:split_size]
    x_test, treatment_test, outcome_test = x[split_size:], treatment[split_size:], outcome[split_size:]

    # Fit with LassoCV the treatment as a function of x and the outcome as
    # a function of x, using only the train fold
    # p = g0
    # q = f0+theta g0
    model_treatment.fit(x_train, treatment_train)
    model_outcome.fit(x_train, outcome_train)

    # Then compute residuals p-g(x) and q-q(x) on test fold
    res_treatment = (treatment_test - model_treatment.predict(x_test)).flatten()
    res_outcome = (outcome_test - model_outcome.predict(x_test)).flatten()

    ''' ORTHO ML '''
    # Compute coefficient by OLS on residuals
    ortho_ml = np.sum(np.multiply(res_treatment, res_outcome)) / np.sum(np.multiply(res_treatment, res_treatment))

    ''' ROBUST ORTHO ML with KNOWN MOMENTS '''
    # Compute for each sample the quantity:
    #
    #          (Z_i-f(X_i))^3 - 3*(sigma^2)*(Z_i-f(X_i)) - cube_p 
    #
    # The coefficient is a simple division: 
    #
    #       E_n{ (Y-m(X)) * ((Z-f(X))^3-3(sigma^2)(Z-f(X))) }
    #   -----------------------------------------------------------------
    #   E_n{ (Z-f(x)) * ((Z-f(x))^3 - 3 * (sigma^2) * (Z-f(x)) - cube_p)}
    #
    mult_p = res_treatment ** 3 - 3 * eta_second_moment * res_treatment - eta_third_moment
    robust_ortho_ml = np.mean(res_outcome * mult_p) / np.mean(res_treatment * mult_p)

    ''' ROBUST ORTHO ML with ESTIMATED MOMENTS '''
    # Estimate the moments from the residuals of the first fold
    eta_residual = treatment_train - model_treatment.predict(x_train)
    eta_second_moment_est = np.mean(eta_residual ** 2)
    eta_third_moment_est = np.mean(eta_residual ** 3) - 3 * np.mean(eta_residual) * np.mean(eta_residual ** 2)
    # Estimate the treatment effect from the second fold
    mult_p_est = res_treatment ** 3 - 3 * eta_second_moment_est * res_treatment - eta_third_moment_est
    robust_ortho_est_ml = np.mean(res_outcome * mult_p_est) / np.mean(res_treatment * mult_p_est)

    ''' ROBUST ORTHO ML with ESTIMATED MOMENTS on THIRD SPLIT '''
    # Estimate the moments from the residuals of the first fold
    test_split = x_test.shape[0] // 2
    eta_residual = res_treatment[:test_split]
    eta_second_moment_est = np.mean(eta_residual ** 2)
    eta_third_moment_est = np.mean(eta_residual ** 3) - 3 * np.mean(eta_residual) * np.mean(eta_residual ** 2)
    # Estimate the treatment effect from the second fold
    res_treatment_second = res_treatment[test_split:]
    res_outcome_second = res_outcome[test_split:]
    mult_p_est = res_treatment_second ** 3 - 3 * eta_second_moment_est * res_treatment_second - eta_third_moment_est
    robust_ortho_est_split_ml = np.mean(res_outcome_second * mult_p_est) / np.mean(res_treatment_second * mult_p_est)

    return ortho_ml, robust_ortho_ml, robust_ortho_est_ml, robust_ortho_est_split_ml


def all_together_cross_fitting(x, p, q, second_p, cube_p,
                               model_treatment=LassoCV(alphas=[0.01, 0.1, 0.3, 0.5, 0.9, 5, 10, 20, 100]),
                               model_outcome=LassoCV(alphas=[0.01, 0.1, 0.3, 0.5, 0.9, 5, 10, 20, 100])):
    res_p = np.zeros(x.shape[0])
    res_q = np.zeros(x.shape[0])
    mult_p = np.zeros(x.shape[0])
    mult_p_est = np.zeros(x.shape[0])
    mult_p_est_split = np.zeros(x.shape[0])

    kf = KFold(x.shape[0], n_folds=2)
    for train_index, test_index in kf:
        # Split the data in half, train and test
        x_train, p_train, q_train = x[train_index], p[train_index], q[train_index]
        x_test, p_test, q_test = x[test_index], p[test_index], q[test_index]

        # Fit with LassoCV the treatment as a function of x and the outcome as
        # a function of x, using only the train fold
        model_treatment.fit(x_train, p_train)
        model_outcome.fit(x_train, q_train)

        # Then compute residuals p-g(x) and q-q(x) on test fold
        res_p[test_index] = (p_test - model_treatment.predict(x_test)).flatten()
        res_q[test_index] = (q_test - model_outcome.predict(x_test)).flatten()

        # Estimate multipliers for robust orthogonal methods 

        # 1. Multiplier with known moments
        mult_p[test_index] = res_p[test_index] ** 3 - 3 * second_p * res_p[test_index] - cube_p

        # 2. Multiplier with estimated moments on training data
        res_p_first = p_train - model_treatment.predict(x_train)
        second_p_est = np.mean(res_p_first ** 2)
        cube_p_est = np.mean(res_p_first ** 3) - 3 * np.mean(res_p_first) * np.mean(res_p_first ** 2)
        # Estimate the treatment effect from the second fold
        mult_p_est[test_index] = res_p[test_index] ** 3 - 3 * second_p_est * res_p[test_index] - cube_p_est

        # 3. Multiplier with estimated moments on further split and cross-fit of test data
        nested_kf = KFold(len(test_index), n_folds=2)
        for nested_train_index, nested_test_index in nested_kf:
            res_p_first = res_p[test_index[nested_train_index]]
            second_p_est = np.mean(res_p_first ** 2)
            cube_p_est = np.mean(res_p_first ** 3) - 3 * np.mean(res_p_first) * np.mean(res_p_first ** 2)
            res_p_second = res_p[test_index[nested_test_index]]
            mult_p_est_split[
                test_index[nested_test_index]] = res_p_second ** 3 - 3 * second_p_est * res_p_second - cube_p_est

    ''' ORTHO ML '''
    # Compute coefficient by OLS on residuals
    ortho_ml = np.mean(res_q * res_p) / np.mean(res_p * res_p)

    ''' ROBUST ORTHO ML with KNOWN MOMENTS '''
    # Compute for each sample the quantity:
    #
    #          (Z_i-f(X_i))^3 - 3*(sigma^2)*(Z_i-f(X_i)) - cube_p 
    #
    # The coefficient is a simple division: 
    #
    #       E_n{ (Y-m(X)) * ((Z-f(X))^3-3(sigma^2)(Z-f(X))) }
    #   -----------------------------------------------------------------
    #   E_n{ (Z-f(x)) * ((Z-f(x))^3 - 3 * (sigma^2) * (Z-f(x)) - cube_p)}
    #
    robust_ortho_ml = np.mean(res_q * mult_p) / np.mean(res_p * mult_p)

    ''' ROBUST ORTHO ML with ESTIMATED MOMENTS '''
    robust_ortho_est_ml = np.mean(res_q * mult_p_est) / np.mean(res_p * mult_p_est)

    ''' ROBUST ORTHO ML with ESTIMATED MOMENTS on THIRD SPLIT '''
    robust_ortho_est_split_ml = np.mean(res_q * mult_p_est_split) / np.mean(res_p * mult_p_est_split)

    return ortho_ml, robust_ortho_ml, robust_ortho_est_ml, robust_ortho_est_split_ml, model_treatment.coef_, model_outcome.coef_
