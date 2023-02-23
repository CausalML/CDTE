import numpy as np
from scipy.special import expit, erf, erfinv
from sklearn import clone
from nuisance import get_evar_line_search

#####################
# Experiments utils #
#####################
def exponential_dgp(n, p, cond_mean_func, propensity_func, random_state=None):
    np.random.seed(random_state)
    X = np.random.uniform(-1, 1, size=(n, p))
    A = np.random.binomial(1, expit(propensity_func(X)))
    cond_mean = cond_mean_func(X, A)
    Y = np.random.exponential(cond_mean)
    return X, A, Y

def exp_true_effect(X, cond_mean_func, tau):
    d = {}
    mean_te = cond_mean_func(X, 1) - cond_mean_func(X, 0)
    d["quantile"] = -np.log(1-tau)*mean_te
    d["superquantile_right"] = (-np.log(1-tau)+1)*mean_te
    d["superquantile_left"] = ((1-tau)/tau*np.log(1-tau)+1)*mean_te
    return d

def run_simulation(dgp, model, random_state, **dgp_args):
    X, A, Y = dgp(**dgp_args, random_state=random_state)
    # Estimation
    model_iter = clone(model, safe=False)
    model_iter.random_state = random_state
    model_iter.fit(X, A, Y)
    return model_iter

def lognormal_dgp(n, p, cond_mean_func, cond_std_func, propensity_func, random_state=None, upper_lim_q=None):
    np.random.seed(random_state)
    X = np.random.uniform(0, 1, size=(n, p))
    A = np.random.binomial(1, expit(propensity_func(X)))
    cond_mean = cond_mean_func(X, A)
    cond_std = cond_std_func(X, A)
    Y = np.random.lognormal(cond_mean, cond_std)
    if upper_lim_q is not None:
        upper_lim = np.exp(cond_mean+cond_std*np.sqrt(2)*erfinv(2*upper_lim_q-1))
        # Truncate lognormal
        for i in range(len(Y)):
            while Y[i] >= upper_lim[i]:
                Y[i] = np.random.lognormal(cond_mean[i])
    return X, A, Y

def lognormal_true_effect(X, cond_mean_func, cond_std_func, tau):
    d = {}
    quantile_func = lambda cond_mean, cond_std: np.exp(cond_mean+cond_std*np.sqrt(2)*erfinv(2*tau-1))
    d["quantile"] = quantile_func(cond_mean_func(X, 1), cond_std_func(X, 1)) \
                    - quantile_func(cond_mean_func(X, 0), cond_std_func(X, 0))
    super_right_func = lambda cond_mean, cond_std: 0.5*np.exp(cond_mean+cond_std**2/2)*(
        1+erf(cond_std/np.sqrt(2)-erfinv(2*tau-1)))/(1-tau)
    d["superquantile_right"] = super_right_func(cond_mean_func(X, 1), cond_std_func(X, 1)) \
                                - super_right_func(cond_mean_func(X, 0), cond_std_func(X, 0))
    super_left_func = lambda cond_mean, cond_std: 0.5*np.exp(cond_mean+cond_std**2/2)*(
        1-erf(cond_std/np.sqrt(2)-erfinv(2*tau-1)))/tau
    d["superquantile_left"] = super_left_func(cond_mean_func(X, 1), cond_std_func(X, 1)) \
                                - super_left_func(cond_mean_func(X, 0), cond_std_func(X, 0))
    # Approximation for reasonable truncation and small cond_std
    evar_func = lambda cond_mean, cond_std: np.exp(cond_mean+cond_std**2/2+cond_std*np.sqrt(-2*np.log(1-tau)))
    d["evar"] = evar_func(cond_mean_func(X, 1), cond_std_func(X, 1)) \
                - evar_func(cond_mean_func(X, 0), cond_std_func(X, 0))
    return d


