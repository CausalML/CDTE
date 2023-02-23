import numpy as np
import warnings
from sklearn.gaussian_process.kernels import RBF

#######################
# Quantile Regressors #
#######################
class KernelQuantileRegressor:
    
    def __init__(self, kernel, tau):
        self.kernel = kernel
        self.tau = tau
        
    def fit(self, X, Y):
        self.sorted_Y_idx = np.argsort(Y)
        self.sorted_Y = Y[self.sorted_Y_idx]
        self.kernel.fit(X[self.sorted_Y_idx], Y[self.sorted_Y_idx])
        return self
    
    def predict(self, X):
        preds = np.empty(X.shape[0])
        sorted_weights = self.kernel.predict(X)
        for i, x in enumerate(X):
            quantile_idx = np.where((np.cumsum(sorted_weights[i]) >= self.tau) == True)[0][0]
            preds[i] = self.sorted_Y[quantile_idx]
        return preds

##################
# Kernel Methods #
##################
class RFKernel:
    
    def __init__(self, rf):
        self.rf=rf
        
    def fit(self, X, Y):
        self.rf.fit(X, Y)
        self.train_leaf_map = self.rf.apply(X)
        
    def predict(self, X):
        weights = np.empty((X.shape[0], self.train_leaf_map.shape[0]))
        leaf_map = self.rf.apply(X)
        for i, x in enumerate(X):
            P=(self.train_leaf_map==leaf_map[[i]])
            weights[i]=(1.*P/P.sum(axis=0)).mean(axis=1)
        return weights

class RBFKernel:
    def __init__(self, scale=1):
        self.kernel = RBF(length_scale=scale)
        
    def fit(self, X, Y):
        self.X_train = X
        return self
        
    def predict(self, X):
        weights = self.kernel(X, self.X_train)
        # Normalize weights
        norm_weights = weights/weights.sum(axis=1).reshape(-1, 1)
        return norm_weights

############################
# Superquantile regressors #
############################
class KernelSuperquantileRegressor:
    
    def __init__(self, kernel, tau, tail='left'):
        self.kernel = kernel
        self.tau = tau
        if tail not in ["left", "right"]:
            raise ValueError(
                f"The 'tail' parameter can only take values in ['left', 'right']. Got '{tail}' instead.")
        self.tail = tail
        
    def fit(self, X, Y):
        self.sorted_Y_idx = np.argsort(Y)
        self.sorted_Y = Y[self.sorted_Y_idx]
        self.kernel.fit(X[self.sorted_Y_idx], Y[self.sorted_Y_idx])
        return self
    
    def predict(self, X):
        preds = np.empty(X.shape[0])
        sorted_weights = self.kernel.predict(X)
        for i, x in enumerate(X):
            if self.tail == "right":
                idx_tail = np.where((np.cumsum(sorted_weights[i]) >= self.tau) == True)[0]
                preds[i] = np.sum(self.sorted_Y[idx_tail] * sorted_weights[i][idx_tail])/(1-self.tau)
            else:
                idx_tail = np.where((np.cumsum(sorted_weights[i]) <= self.tau) == True)[0]
                preds[i] = np.sum(self.sorted_Y[idx_tail] * sorted_weights[i][idx_tail])/self.tau
        return preds

###################
# EVaR regressors #
###################

### Misc utils ###
def get_evar_grad(y, beta, tau, weights=None):
    W_0 = np.average(np.exp(y/beta), weights=weights)
    W_1 =  np.average(y*np.exp(y/beta), weights=weights)
    grad = np.log(W_0)-np.log(1-tau) - W_1/(beta*W_0)
    return grad

def get_evar_derivatives(y, beta, tau, weights=None):
    W_0 = np.average(np.exp(y/beta), weights=weights)
    W_1 =  np.average(y*np.exp(y/beta), weights=weights)
    W_2 =  np.average(y**2*np.exp(y/beta), weights=weights)
    d1 = np.log(W_0)-np.log(1-tau) - W_1/(beta*W_0)
    d2 = W_2/beta**3/W_0 - W_1**2/beta**3/W_0**2 
    return d1, d2

def small_step_beta(y, tau, beta_init, fail_step=0.01, weights=None):
    grad = get_evar_grad(y, beta_init, tau, weights=weights)
    return (beta_init - fail_step if grad > 0 else beta_init + fail_step)

def get_evar_objective(y, beta, tau, weights=None):
    exp_mean = np.average(np.exp(y/beta), weights=weights)
    return beta*(np.log(exp_mean)-np.log(1-tau))

### Beta optimizers ###
def get_evar_line_search(y, tau, beta_init=0, tol=0.001, weights=None):
    delta_inv = 1/-np.log(1-tau)
    beta_min = 0.01
    beta_max = delta_inv
    if beta_init == 0:
        beta_init = beta_max / 2
    beta_new = beta_init
    beta = 0
    steps = 0
    while np.abs(beta_new - beta) > tol:
        steps += 1
        beta = beta_new
        grad = get_evar_grad(y, beta_new, tau, weights=weights)
        if grad > 0:
            beta_new = (beta_min + beta) / 2
            beta_max = beta
        else:
            beta_new = (beta_max + beta) / 2
            beta_min = beta
    beta_star = beta_new
    return beta_star, get_evar_objective(y, beta_star, tau, weights=weights)

def line_search_opt(y, tau, beta_init=0, tol=0.001, weights=None):
    beta_star, evar = get_evar_line_search(y, tau, beta_init=beta_init, tol=tol, weights=weights)
    lambda_star = beta_star*(np.log(np.average(np.exp(y/beta_star), weights=weights))-1)
    return evar, beta_star, lambda_star

def newton_opt(y, tau, beta_init, tol=0.001, fail_step=0.02, weights=None):
    delta_inv = 1/-np.log(1-tau)
    beta_new = beta_init
    beta = 0
    for i in range(3):
        for t in range(10):
            if np.abs(beta_new - beta) < tol:
                break
            beta = beta_new
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    d1, d2 = get_evar_derivatives(y, beta, tau, weights=weights)
                    beta_new = beta - d1/d2
                except Warning as e:
                    beta_new = 0
            #Guardrails
            if beta_new <= 0 or beta_new > delta_inv or np.isnan(beta_new):
                # Redefine beta_new and beta_init
                beta_new = small_step_beta(y, tau, beta_init, fail_step=fail_step, weights=weights)
                beta_init = beta_new
        if not np.abs(beta_new - beta) < tol:
            beta_new = small_step_beta(y, tau, beta_init, fail_step=fail_step, weights=weights)
            beta_init = beta_new
        else:
            break
    beta_star = beta_new
    lambda_star = beta_star*(np.log(np.average(np.exp(y/beta_star), weights=weights))-1)
    evar = get_evar_objective(y, beta_star, tau, weights=weights)
    return evar, beta_star, lambda_star

class KernelEVaRRegressor:
    def __init__(self, kernel, tau):
        self.kernel = kernel
        self.tau = tau
        
    def fit(self, X, Y):
        self.scale = np.max(Y)
        self.Y_train = Y / self.scale
        self.kernel.fit(X, self.Y_train)
        # Get beta_init
        self.beta_init, _ = get_evar_line_search(self.Y_train, self.tau)
        return self
        
    def predict(self, X):
        """Return EVaR, beta and lambda"""
        preds = np.empty(shape=(X.shape[0], 3))
        weights = self.kernel.predict(X)
        for i, x in enumerate(X):
            evar, beta_star, lambda_star = line_search_opt(self.Y_train, self.tau, self.beta_init, weights=weights[i])
            preds[i] = self.scale*np.array([evar, beta_star, lambda_star])
        return preds