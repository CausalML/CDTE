import numpy as np
from skgarden import RandomForestQuantileRegressor
from sklearn.gaussian_process.kernels import RBF

#######################
# Quantile Regressors #
#######################
class RandomForestQuantileRegressorWrapper(RandomForestQuantileRegressor):
    
    def __init__(self, 
                 n_estimators=10,
                 criterion='mse',
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0,
                 max_features='auto',
                 max_leaf_nodes=None,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False, 
                 tau=0.5):
        self.tau = tau
        super().__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start
            )
    
    def predict(self, X):
        return super().predict(X, quantile=self.tau*100)


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