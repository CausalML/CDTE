import numpy as np
from sklearn import clone
from sklearn.model_selection import KFold

from utils import (
    _crossfit, _crossfit_nested, 
    CQTE_Nuisance_Model, CQTE_Plugin_Model, CQTE_Nested_Nuisance_Model,
    CKLTRE_Nuisance_Model, CKLTRE_Plugin_Model
    )

#####################
# Main CDTE classes #
#####################
class CQTE:
    def __init__(self, 
                 propensity_model, 
                 quantile_model, 
                 cond_density_kernel,
                 cond_density_model,
                 cqte_model, 
                 tau=0.5, 
                 proj_idx=None, 
                 min_propensity=1e-6,
                 cv=5,
                 random_state=None):
        self.nuisance_model = CQTE_Nuisance_Model(propensity_model, quantile_model)
        self.plugin_model = CQTE_Plugin_Model(quantile_model, quantile_model)
        self.nested_nuisance_model = CQTE_Nested_Nuisance_Model(cond_density_model)
        self.cond_density_kernel = cond_density_kernel
        self.nested_outcome_func = lambda nuis, X, A, Y: self.cond_density_kernel(Y-(nuis[:, 1]*(1-A) + nuis[:, 2]*A))
        self.cqte_model = clone(cqte_model)
        self.plugin_model_proj = clone(cqte_model)
        self.tau = tau
        self.proj_idx = proj_idx
        # TODO: integrate min_propensity
        self.min_propensity = min_propensity
        self.cv = cv
        self.random_state = random_state
    
    def fit(self, X, A, Y):
        # Get folds
        folds = list(KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state).split(X))
        # Fit propensity, quantile models, conditional densities via crossfit
        nuisances, *_ = _crossfit_nested(self.nuisance_model, 
                          self.nested_nuisance_model, 
                          self.nested_outcome_func,
                          folds, X, A, Y)
        # Get pseudo-outcomes
        psi = self._get_pseudo_outcomes(X, A, Y, nuisances)
        # Fit final regression model
        self.cqte_model.fit(X[:, self.proj_idx] if self.proj_idx is not None else X, psi)
        self.plugin_model.fit(X, A, Y)
        self.plugin_model_proj.fit(X[:, self.proj_idx] if self.proj_idx is not None else X, 
                                   nuisances[:, 2] - nuisances[:, 1])
        return self
    
    def effect(self, X):
        return self.cqte_model.predict(X)
    
    def _get_pseudo_outcomes(self, X, A, Y, nuisances):
        q_A = nuisances[:, 1]*(1-A) + nuisances[:, 2]*A
        f_A = nuisances[:, 3]*(1-A) + nuisances[:, 4]*A
        psi = nuisances[:, 2] - nuisances[:, 1] + 1/(nuisances[:, 0] - 1 + A)*\
            (self.tau - (Y<=q_A)*1)/f_A
        return psi

class CSQTE:
    def __init__(self, 
                 propensity_model, 
                 quantile_model, 
                 superquantile_model,
                 csqte_model,
                 nested_quantiles=True,
                 tau=0.5, 
                 tail = "left",
                 proj_idx=None, 
                 min_propensity=1e-6,
                 cv=5,
                 random_state=None):
        self.nuisance_model = CQTE_Nuisance_Model(propensity_model, quantile_model)
        self.nested_nuisance_model = CQTE_Nested_Nuisance_Model(superquantile_model)
        self.csqte_model = clone(csqte_model, safe=False)
        self.plugin_model = CQTE_Plugin_Model(superquantile_model, superquantile_model)
        self.plugin_model_proj = clone(csqte_model, safe=False)
        self.tau = tau
        if tail not in ["left", "right"]:
            raise ValueError(f"The 'tail' parameter can only take values in ['left', 'right']. Got '{tail}' instead.")
        self.tail = tail
        self.nested_quantiles = nested_quantiles
        if nested_quantiles:
            if tail == "left":
                self.nested_outcome_func = lambda nuis, X, A, Y: 1/tau*(Y * (Y <= (nuis[:, 1]*(1-A) + nuis[:, 2]*A))*1)
            else:
                self.nested_outcome_func = lambda nuis, X, A, Y: 1/(1-tau)*(Y * (Y >= (nuis[:, 1]*(1-A) + nuis[:, 2]*A))*1)
        else:
            self.nested_outcome_func = lambda nuis, X, A, Y: Y
        self.proj_idx = proj_idx
        # TODO: integrate min_propensity
        self.min_propensity = min_propensity
        self.cv = cv
        self.random_state = random_state
    
    def fit(self, X, A, Y):
        # TODO: check inputs
        # Get folds
        folds = list(KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state).split(X))
        # Fit propensity, quantile models, conditional densities via crossfit
        nuisances, *_ = _crossfit_nested(self.nuisance_model, 
                          self.nested_nuisance_model, 
                          self.nested_outcome_func,
                          folds, X, A, Y)
        # Get pseudo-outcomes
        psi = self._get_pseudo_outcomes(X, A, Y, nuisances)
        # Fit final regression model
        self.csqte_model.fit(X[:, self.proj_idx] if self.proj_idx is not None else X, psi)
        if self.nested_quantiles:
            if self.tail == "left":
                Y_bar = (Y * ((Y <= (nuisances[:, 1]*(1-A) + nuisances[:, 2]*A))*1))
            else:
                Y_bar = (Y * ((Y >= (nuisances[:, 1]*(1-A) + nuisances[:, 2]*A))*1))
            self.plugin_model.fit(X, A, 1/self.tau*Y_bar if self.tail=='left' else 1/(1-self.tau)*Y_bar)
        else:
            self.plugin_model.fit(X, A, Y)
        self.plugin_model_proj.fit(X[:, self.proj_idx] if self.proj_idx is not None else X, 
                           nuisances[:, 4] - nuisances[:, 3])
        return self
    
    def effect(self, X):
        return self.csqte_model.predict(X)
    
    def _get_pseudo_outcomes(self, X, A, Y, nuisances):
        q_A = nuisances[:, 1]*(1-A) + nuisances[:, 2]*A
        mu_A = nuisances[:, 3]*(1-A) + nuisances[:, 4]*A
        if self.tail == "left":
            q_A_ind = (Y <= q_A)*1
            psi = nuisances[:, 4] - nuisances[:, 3] + (1/self.tau)*(1/(nuisances[:, 0] - 1 + A))* (
                Y*q_A_ind - self.tau*mu_A +
                q_A*(self.tau - q_A_ind))
        elif self.tail == "right":
            q_A_ind = (Y >= q_A)*1
            psi = nuisances[:, 4] - nuisances[:, 3] + 1/(1-self.tau)*1/(nuisances[:, 0] - 1 + A)* (
                Y*q_A_ind - (1-self.tau)*mu_A + q_A*((1-self.tau) - q_A_ind))
        return psi

class CKLRTE:
    def __init__(self,
                 propensity_model,
                 evar_model,
                 cklrte_model,
                 tau=0.5,
                 proj_idx=None,
                 min_propensity=1e-5,
                 cv=5,
                 random_state=None):
        self.nuisance_model = CKLTRE_Nuisance_Model(propensity_model, evar_model)
        self.plugin_model = CKLTRE_Plugin_Model(evar_model)
        self.cklrte_model = cklrte_model
        self.plugin_model_proj = clone(cklrte_model)
        self.tau = tau
        self.delta = -np.log(1-self.tau)
        self.proj_idx = proj_idx
        self.min_propensity = min_propensity
        self.cv = cv
        self.random_state = random_state
        
    def fit(self, X, A, Y):
        if self.cv > 1:
            # Get folds
            folds = list(KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state).split(X))
            # Fit propensity, quantile models, conditional densities via crossfit
            nuisances, *_ = _crossfit(self.nuisance_model,
                              folds, X, A, Y)
        else:
            nuisances = self.nuisance_model.fit(X, A, Y).predict(X)
        # Get pseudo-outcomes
        psi = self._get_pseudo_outcomes(X, A, Y, nuisances)
        # Fit final regression model
        self.cklrte_model.fit(X[:, self.proj_idx] if self.proj_idx is not None else X, psi)
        self.plugin_model.fit(X, A, Y)
        self.plugin_model_proj.fit(X[:, self.proj_idx] if self.proj_idx is not None else X, 
                                   nuisances[:, 2] - nuisances[:, 1])
        return self
    
    def effect(self, X):
        return self.cklrte_model.predict(X)
    
    def _get_pseudo_outcomes(self, X, A, Y, nuisances):
        evar_A = nuisances[:, 1]*(1-A) + nuisances[:, 2]*A
        beta_A = nuisances[:, 3]*(1-A) + nuisances[:, 4]*A
        lambda_A = nuisances[:, 5]*(1-A) + nuisances[:, 6]*A
        m_A = self.delta*beta_A + lambda_A + beta_A*(np.exp((Y-lambda_A)/beta_A-1))
        psi = nuisances[:, 2] - nuisances[:, 1] + 1/(nuisances[:, 0] - 1 + A)*\
            (m_A-evar_A)
        return psi