import numpy as np
from scipy.special import expit, erf, erfinv
from sklearn import clone

#####################
# Experiments utils #
#####################
def exponential_dgp(n, p, cond_mean_func, propensity_func, random_state=None):
    np.random.seed(random_state)
    #X = np.random.normal(size=(n, p)).clip(min=-2)
    X = np.random.uniform(-1, 1, size=(n, p))
    A = np.random.binomial(1, expit(propensity_func(X)))
    cond_mean = cond_mean_func(X, A)
    Y = np.random.exponential(cond_mean)
    """
    # Truncate exponential b/c in the real world values are truncated
    upper_limit = np.quantile(Y, 0.995)
    for i in range(len(Y)):
        while Y[i] >= upper_limit:
            Y[i] = np.random.exponential(cond_mean[i])
    """
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

def lognormal_dgp(n, p, cond_mean_func, cond_std_func, propensity_func, random_state=None):
    np.random.seed(random_state)
    #X = np.random.normal(size=(n, p)).clip(min=-2)
    X = np.random.uniform(0, 1, size=(n, p))
    A = np.random.binomial(1, expit(propensity_func(X)))
    cond_mean = cond_mean_func(X, A)
    cond_std = cond_std_func(X, A)
    Y = np.random.lognormal(cond_mean, cond_std)
    """
    # Truncate exponential b/c in the real world values are truncated
    upper_limit = np.quantile(Y, 0.995)
    for i in range(len(Y)):
        while Y[i] >= upper_limit:
            Y[i] = np.random.exponential(cond_mean[i])
    """
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
    return d


