import os
import numpy as np
import pandas as pd
from sklearn import clone
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, LogLocator
from matplotlib.ticker import ScalarFormatter

def _cdte_crossfit(model, nested_model, nested_outcome_func, folds, X, A, Y):
    
    model_list = []
    nested_model_list = []
    fitted_inds = []
    
    for idx, (train_idxs, test_idxs) in enumerate(folds):
        model_list.append(clone(model, safe=False))
        nested_model_list.append(clone(nested_model, safe=False))
        fitted_inds = np.concatenate((fitted_inds, test_idxs))
        
        model_list[idx].fit(X[train_idxs], A[train_idxs], Y[train_idxs])
        nuisance_temp = model_list[idx].predict(X[test_idxs])
        nested_outcome = nested_outcome_func(model_list[idx].predict(X[train_idxs]), 
                                             X[train_idxs], A[train_idxs], Y[train_idxs])
        nested_model_list[idx].fit(X[train_idxs], A[train_idxs], nested_outcome)
        nuisance_temp = np.hstack((
            nuisance_temp, nested_model_list[idx].predict(X[test_idxs])
            ))
        if idx == 0:
            nuisances = np.full((X.shape[0], nuisance_temp.shape[1]), np.nan)
            
        nuisances[test_idxs] = nuisance_temp
    return nuisances, model_list, nested_model_list

def exp_kernel_generator(h=1):
    return lambda x: 1/h * np.exp(-x**2/h**2/2)

##################
# Wrapper models #
##################
class CQTE_Nuisance_Model:
    def __init__(self, 
                 propensity_model, 
                 quantile_model):
        self.propensity_model = clone(propensity_model, safe=False)
        self.quantile_models = [clone(quantile_model,safe=False), clone(quantile_model, safe=False)]
    
    def fit(self, X, A, Y):
        self.propensity_model.fit(X, A)
        self.quantile_models[0].fit(X[A==0], Y[A==0])
        self.quantile_models[1].fit(X[A==1], Y[A==1])
    
    def predict(self, X):
        predictions = np.hstack((
            self.propensity_model.predict_proba(X)[:, [1]],
            self.quantile_models[0].predict(X).reshape(-1, 1),
            self.quantile_models[1].predict(X).reshape(-1, 1),
        ))
        return predictions

class CQTE_Nested_Nuisance_Model:
    def __init__(self, 
                 nested_model):
        self.nested_models = [clone(nested_model, safe=False), clone(nested_model, safe=False)]
    
    def fit(self, X, A, Y):
        self.nested_models[0].fit(X[A==0], Y[A==0])
        self.nested_models[1].fit(X[A==1], Y[A==1])
    
    def predict(self, X):
        predictions = np.hstack((
            self.nested_models[0].predict(X).reshape(-1, 1),
            self.nested_models[1].predict(X).reshape(-1, 1),
        ))
        return predictions

class CQTE_Plugin_Model:

    def __init__(self, model0, model1):
        self.model0 = clone(model0, safe=False)
        self.model1 = clone(model1, safe=False)
        
    def fit(self, X, A, Y):
        self.model0.fit(X[A==0], Y[A==0])
        self.model1.fit(X[A==1], Y[A==1])
        
    def predict(self, X):
        return self.model1.predict(X) - self.model0.predict(X)

#######################
# Serialization utils #
#######################
CSQTE_PREDS_FNAME_TEMPLATE = "results/CSQTE_preds_n_iter_{n_iter}_n_{n}_p_{p}_tau_{tau}_tail_{tail}_nuis_{nuis}_final_stage_{final_stage}_dgp_{dgp}.csv"
CSQTE_COEFS_FNAME_TEMPLATE = "results/CSQTE_coefs_n_iter_{n_iter}_n_{n}_p_{p}_tau_{tau}_tail_{tail}_nuis_{nuis}_final_stage_{final_stage}_dgp_{dgp}.csv"

def write_results_to_file(trained_models, 
                          X_test, true_effects, true_coefs,
                          n, p, tau, tail, nuis, final_stage, dgp):
    if not os.path.exists("results"):
        os.makedirs("results")
    n_iter = len(trained_models)
    ##### Predictions
    preds = np.vstack(
    (np.array([trained_models[i].effect(X_test) for i in range(n_iter)]),
    np.array([trained_models[i].plugin_model.predict(X_test) for i in range(n_iter)]),
    np.array([trained_models[i].plugin_model_proj.predict(X_test) for i in range(n_iter)]),
    true_effects[f"superquantile_{tail}"],
    )
    )
    mse = np.array([mean_squared_error(true_effects[f"superquantile_{tail}"], 
        pred) for pred in preds]).reshape(-1, 1)
    model_names = np.hstack((
        np.repeat("CSQTE", n_iter), 
        np.repeat("plugin", n_iter), 
        np.repeat("plugin_final", n_iter),
        ["true_effect"]))
    preds = np.hstack((
        model_names.reshape(-1, 1),
        preds,
        mse
    ))
    colnames = ["model"] + [f"pred{i}" for i in range(X_test.shape[0])] + ["MSE"]
    preds_fname = CSQTE_PREDS_FNAME_TEMPLATE.format(
        n_iter=n_iter,
        n=n,
        p=p,
        tau=tau,
        tail=tail,
        nuis=nuis,
        final_stage=final_stage,
        dgp=dgp)
    pd.DataFrame(preds, columns=colnames).to_csv(preds_fname, index=False)
    ##### Coefs
    if final_stage == "OLS":
        coefs = np.hstack((
        np.array([trained_models[i].csqte_model.coef_ for i in range(n_iter)]),
        np.array([trained_models[i].csqte_model.coef_stderr_ for i in range(n_iter)]),
        np.array([trained_models[i].csqte_model.coef__interval()[0] for i in range(n_iter)]),
        np.array([trained_models[i].csqte_model.coef__interval()[1] for i in range(n_iter)])
        ))
        coverage = [((true_coefs >= coefs[i, 2*p:3*p]) & (true_coefs <= coefs[i, 3*p:4*p]))*1
                for i in range(n_iter)]
        coefs = np.hstack((coefs, coverage))
        coefs_plugin = np.hstack((
            np.array([trained_models[i].plugin_model_proj.coef_ for i in range(n_iter)]),
            np.array([trained_models[i].plugin_model_proj.coef_stderr_ for i in range(n_iter)]),
            np.array([trained_models[i].plugin_model_proj.coef__interval()[0] for i in range(n_iter)]),
            np.array([trained_models[i].plugin_model_proj.coef__interval()[1] for i in range(n_iter)])
        ))
        coverage = [((true_coefs >= coefs_plugin[i, 2*p:3*p]) & (true_coefs <= coefs_plugin[i, 3*p:4*p]))*1
                for i in range(n_iter)]
        coefs_plugin = np.hstack((coefs_plugin, coverage))
        model_names = np.hstack((
            np.repeat("CSQTE", n_iter), 
            np.repeat("plugin", n_iter), 
            ["true_coefs"]))
        true_coef_aug = np.zeros(5*p)
        true_coef_aug[:p] = true_coefs
        coefs = np.hstack((
            model_names.reshape(-1, 1),
            np.vstack((coefs, coefs_plugin, true_coef_aug))
        ))

        colnames = ["model"] + \
                    [f"coef{i}" for i in range(p)] +\
                    [f"stderr{i}" for i in range(p)] +\
                    [f"coef_lower{i}" for i in range(p)] +\
                    [f"coef_upper{i}" for i in range(p)] +\
                    [f"coverage{i}" for i in range(p)]

        coefs_fname = CSQTE_COEFS_FNAME_TEMPLATE.format(
            n_iter=n_iter,
            n=n,
            p=p,
            tau=tau,
            tail=tail,
            nuis=nuis,
            final_stage=final_stage,
            dgp=dgp)
        pd.DataFrame(coefs, columns=colnames).to_csv(coefs_fname, index=False)

def load_results_from_file(n_iter, n, p, tau, tail, nuis, final_stage, dgp):
    preds_fname = CSQTE_PREDS_FNAME_TEMPLATE.format(
        n_iter=n_iter,
        n=n,
        p=p,
        tau=tau,
        tail=tail,
        nuis=nuis,
        final_stage=final_stage,
        dgp=dgp)
    coefs_fname = CSQTE_COEFS_FNAME_TEMPLATE.format(
        n_iter=n_iter,
        n=n,
        p=p,
        tau=tau,
        tail=tail,
        nuis=nuis,
        final_stage=final_stage,
        dgp=dgp)
    preds = pd.read_csv(preds_fname)
    d = {
        "CSQTE_MSE": preds[preds.model=="CSQTE"]["MSE"].values,
        "plugin_MSE": preds[preds.model=="plugin"]["MSE"].values,
        "plugin_final_MSE": preds[preds.model=="plugin_final"]["MSE"].values,
        }
    if final_stage == "OLS":
        coefs = pd.read_csv(coefs_fname)
        d.update({
            "CSQTE_coverage": coefs[coefs.model=="CSQTE"][[f"coverage{i}" for i in range(p)]].values,
            "plugin_coverage": coefs[coefs.model=="plugin"][[f"coverage{i}" for i in range(p)]].values
            })
    return d

##################
# Plotting Utils #
##################
def ggplot_style_log(figsize, log_y=False):
    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    
    # Give plot a gray background like ggplot.
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.size'] = 16
    ax.set_facecolor('#EBEBEB')
    # Remove border around plot.
    [ax.spines[side].set_visible(False) for side in ax.spines]
    # Style the grid.
    ax.grid(which='major', color='white', linewidth=1.2)
    ax.grid(which='minor', color='white', linewidth=0.6)
    # Show the minor ticks and grid.
    ax.minorticks_on()
    # Now hide the minor ticks (but leave the gridlines).
    ax.tick_params(which='minor', bottom=False, left=False)
    if log_y:
        ax.loglog()
        locmaj_y = LogLocator(base=10.0, subs=(1, 3), numticks=12)
        ax.yaxis.set_major_locator(locmaj_y)
        locmin_y = LogLocator(base=10.0, subs=(10**(-0.25), 10**0.25),numticks=12)
        ax.yaxis.set_minor_locator(locmin_y)
    else:
        ax.set_xscale('log')
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))        
    locmin_x = LogLocator(base=10.0, subs=(10**0.5,),numticks=12)
    ax.xaxis.set_minor_locator(locmin_x)
    for axis in [ax.xaxis, ax.yaxis]:
        formatter = ScalarFormatter()
        formatter.set_scientific(False)
        axis.set_major_formatter(formatter)
    return ax

def ggplot_style_grid(figsize):
    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    
    # Give plot a gray background like ggplot.
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.size'] = 16
    ax.set_facecolor('#EBEBEB')
    # Remove border around plot.
    [ax.spines[side].set_visible(False) for side in ax.spines]
    # Style the grid.
    ax.grid(which='major', color='white', linewidth=1.2)
    ax.grid(which='minor', color='white', linewidth=0.6)
    # Show the minor ticks and grid.
    ax.minorticks_on()
    # Now hide the minor ticks (but leave the gridlines).
    ax.tick_params(which='minor', bottom=False, left=False)
    for axis in [ax.xaxis, ax.yaxis]:
        formatter = ScalarFormatter()
        formatter.set_scientific(True)
        axis.set_major_formatter(formatter)
    return ax

class PlottingSuite:

    def __init__(self, n_iter, ns, p, tau, tail, nuis, final_stage, dgp):
        # Read data from files
        self.n_iter = n_iter
        self.ns = ns
        self.nuis = nuis
        self.final_stage = final_stage
        self.save_prefix = f"CSQTE_n_iter_{n_iter}_p_{p}_tau_{tau}_tail_{tail}_nuis_{nuis}_final_stage_{final_stage}_dgp_{dgp}.pdf"
        self.results = {
           n: load_results_from_file(n_iter, n, p, tau, tail, nuis, final_stage, dgp)
           for n in ns
        }
    
    def plot_mse(self, show_plugin=True, show_plugin_final=True, log_yscale=False, show=True, save=False):
        figure(figsize=(6, 4), dpi=100)
        model_mean = np.array([self.results[n]["CSQTE_MSE"] for n in self.ns]).mean(axis=1)
        model_mean_sd = np.array([self.results[n]["CSQTE_MSE"] for n in self.ns]).std(axis=1) / np.sqrt(self.n_iter)
        result = [(model_mean, model_mean_sd)]
        plt.plot(self.ns, model_mean, label="CSQTE")
        plt.fill_between(self.ns, model_mean - model_mean_sd, model_mean + model_mean_sd, alpha=0.3)
        if show_plugin:
            plugin_mean = np.array([self.results[n]["plugin_MSE"] for n in self.ns]).mean(axis=1)
            plugin_sd = np.array([self.results[n]["plugin_MSE"] for n in self.ns]).std(axis=1) / np.sqrt(self.n_iter)
            plt.plot(self.ns, plugin_mean, label="Plugin")
            plt.fill_between(self.ns, plugin_mean - plugin_sd, plugin_mean + plugin_sd, alpha=0.3)
            result += [(plugin_mean, plugin_sd)]
        if show_plugin_final:
            plugin_final_mean = np.array([self.results[n]["plugin_final_MSE"] for n in self.ns]).mean(axis=1)
            plugin_final_sd = np.array([self.results[n]["plugin_final_MSE"] for n in self.ns]).std(axis=1) / np.sqrt(self.n_iter)
            plt.plot(self.ns, plugin_final_mean, label=f"Plugin+{self.final_stage}")
            plt.fill_between(self.ns, plugin_final_mean - plugin_final_sd, plugin_final_mean + plugin_final_sd, alpha=0.3)
            result += [(plugin_final_mean, plugin_final_sd)]
        plt.xlabel("n")
        plt.ylabel("MSE")
        if log_yscale:
            plt.yscale("log")
        plt.xscale("log")
        #plt.minorticks_off()
        plt.legend()
        if save:
            plt.savefig(f"results/MSE_{self.save_prefix}", dpi=200)
        if show:
            plt.show()
        return result

    def plot_coverage(self, coef_idx=0, show_plugin_final=True, show=True, save=False):
        ns = self.ns
        if self.final_stage == "OLS":
            figure(figsize=(6, 4), dpi=100)
            coverage_mean = np.array([self.results[n]["CSQTE_coverage"][:, coef_idx] for n in self.ns]).mean(axis=1)
            coverage_std = np.array([self.results[n]["CSQTE_coverage"][:, coef_idx] for n in self.ns]).std(axis=1) / np.sqrt(self.n_iter)
            result = [(coverage_mean, coverage_std)]
            plt.plot(self.ns, coverage_mean, label="CSQTE")
            plt.fill_between(self.ns, coverage_mean - coverage_std, coverage_mean + coverage_std, alpha=0.3)
            if show_plugin_final:
                plugin_coverage_mean = np.array([self.results[n]["plugin_coverage"][:, coef_idx] for n in self.ns]).mean(axis=1)
                plugin_coverage_std = np.array([self.results[n]["plugin_coverage"][:, coef_idx] for n in self.ns]).std(axis=1) / np.sqrt(self.n_iter)
                plt.plot(self.ns, plugin_coverage_mean, label="Plugin")
                plt.fill_between(self.ns, plugin_coverage_mean - plugin_coverage_std, plugin_coverage_mean + plugin_coverage_std, alpha=0.3)
                result += [(plugin_coverage_mean, plugin_coverage_std)]
            plt.xscale("log")
            plt.minorticks_off()
            plt.xlabel("n")
            plt.ylabel("95% CI Coverage")
            plt.legend()
            if save:
                plt.savefig(f"results/Coverage_{self.save_prefix}", dpi=200)
            if show:
                plt.show()
            return result
        else:
            return None