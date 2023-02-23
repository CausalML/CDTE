# Reproduces Figure 5 from "Robust and Agnostic Learning of Conditional Distributional Treatment Effects"

# Generic imports 
import numpy as np
import os
from tqdm import tqdm 
from econml.sklearn_extensions.linear_model import StatsModelsLinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from joblib import Parallel, delayed
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

# Custom imports
from utils import write_results_to_file, PlottingSuite, ggplot_style_log
from nuisance import ( 
    RFKernel, RBFKernel, KernelEVaRRegressor
)
from dgp import lognormal_dgp, lognormal_true_effect, run_simulation
from cdte import CKLRTE

# Get true OLS coefs
def get_true_OLS_coefs(n, p, cond_mean_func, cond_std_func, tau):
    X = np.random.uniform(0, 1, size=(n, p))
    true_effects = lognormal_true_effect(X, cond_mean_func, cond_std_func, tau)
    ols = StatsModelsLinearRegression(cov_type="nonrobust")
    ols.fit(X, true_effects[f"evar"])
    return ols.coef_

# DGP Definition
cond_mean_func = lambda X, A: X[:, 0] + X[:, 1]*A
cond_std_func = lambda X, A: 0.2
propensity_func = lambda X: 6*(X[:, 0]-0.5)

dgp="lognormal"

np.random.seed(1)
p = 10
tau = 0.75
n_test = 500
X_test = np.random.uniform(0, 1, size=(n_test, p))
true_effects = lognormal_true_effect(X_test, cond_mean_func, cond_std_func, tau)
true_coefs = get_true_OLS_coefs(500000, p, cond_mean_func, cond_std_func, tau)

# Define nuisances and last stages to loop over
nuiss = ["ERF", "ERBF"] # EVaR random forest, EVaR_RBF
final_stages = ["OLS", "RF"]

# Define experiment parameters
n_iter = 100
ns = [100, 200, 400, 800, 1600, 3200, 6400, 12800]

for nuis in nuiss:
    for final_stage in final_stages:
        print(f"Running CKLRTE experiments with nuisance {nuis} and final stage {final_stage}...")
        for n in tqdm(ns):
            ##############
            # Propensity #
            ##############
            propensity_model = LogisticRegression(solver="saga", penalty="elasticnet", l1_ratio=0.5)
            #####################
            # Quantile Nuisance #
            #####################
            if nuis == "ERF":
                evar_model = KernelEVaRRegressor(
                                kernel=RFKernel(
                                    rf=RandomForestRegressor(n_estimators=50, min_samples_leaf=0.05)),
                                    tau=tau
                            )
            elif nuis == "ERBF":
                # RBF scale via Silverman's rule of thumb
                RBF_scale = 0.9*n**(-1/(4+p))
                evar_model = KernelEVaRRegressor(
                        kernel=RBFKernel(scale=RBF_scale),
                        tau=tau)
            #####################
            # CKLRTE final model #
            #####################
            if final_stage == "OLS":
                cklrte_model = StatsModelsLinearRegression(cov_type="HC1")
            elif final_stage == "RF":
                cklrte_model = RandomForestRegressor(n_estimators=50, min_samples_leaf=0.05)
            ##################
            # CKLRTE Estimator #
            ##################
            CKLRTE_est = CKLRTE(
                            propensity_model = propensity_model,
                            evar_model = evar_model,
                            cklrte_model = cklrte_model,
                            tau=tau,
                            cv=1)
            trained_models = Parallel(n_jobs=-2, backend='loky', verbose=1)(
                                        delayed(run_simulation)(lognormal_dgp, CKLRTE_est, 
                                                                random_state,
                                                                n=n, p=p, 
                                                                cond_mean_func=cond_mean_func, 
                                                                cond_std_func=cond_std_func,
                                                                propensity_func=propensity_func,
                                                                upper_lim_q=0.99
                                                            ) for random_state in np.arange(n_iter))
            write_results_to_file(
                trained_models, 
                X_test, true_effects, true_coefs,
                n, p, tau, tail=None, nuis=nuis, final_stage=final_stage, dgp=dgp, cdte_name="CKLRTE")

for nuis in nuiss:
    if not os.path.exists("plots"):
        os.makedirs("plots")
    #############################
    # Process results from file #
    #############################
    final_stage = "OLS"
    ps = PlottingSuite(n_iter, ns, p, tau, None, nuis, final_stage, dgp, cdte_name="CKLRTE")
    mse_ols = ps.plot_mse(show_plugin_final=True, log_yscale=True, show=False)
    coverage_ols = ps.plot_coverage(coef_idx=1, show_plugin_final=True, show=False)
    final_stage = "RF"
    ps = PlottingSuite(n_iter, ns, p, tau, None, nuis, final_stage, dgp, cdte_name="CKLRTE")
    mse_rf = ps.plot_mse(show_plugin_final=True, log_yscale=True, show=False)
    #############
    # MSE Plots #
    #############
    plugin_all = True
    ax = ggplot_style_log(figsize=(7, 4), log_y=True)
    # CKLRTE + OLS
    plt.plot(ns, mse_ols[0][0], label="CKLRTE+OLS", color="C0", zorder=5)
    plt.fill_between(ns, mse_ols[0][0] - mse_ols[0][1], mse_ols[0][0] + mse_ols[0][1], alpha=0.3, color="C0", zorder=4)
    # CKLRTE + RF
    plt.plot(ns, mse_rf[0][0], label="CKLRTE+RF", color="C3", zorder=5)
    plt.fill_between(ns, mse_rf[0][0] - mse_rf[0][1], mse_rf[0][0] + mse_rf[0][1], alpha=0.3, color="C3", zorder=4)
    # Plugin
    plt.plot(ns, mse_ols[1][0], label="Plugin", color="black", ls="--", zorder=5)
    plt.fill_between(ns, mse_ols[1][0] - mse_ols[1][1], mse_ols[1][0] + mse_ols[1][1], alpha=0.3, color="gray", zorder=4)
    if plugin_all:
        # Plugin + OLS
        plt.plot(ns, mse_ols[2][0], label="Plugin+OLS", color="C0", ls='--', zorder=5)
        plt.fill_between(ns, mse_ols[2][0] - mse_ols[2][1], mse_ols[2][0] + mse_ols[2][1], alpha=0.3, color="C0")
        # Plugin + RF
        plt.plot(ns, mse_rf[2][0], label="Plugin+RF", color="C3", ls='--', zorder=5)
        plt.fill_between(ns, mse_rf[2][0] - mse_rf[2][1], mse_rf[2][0] + mse_rf[2][1], alpha=0.3, color="C3", zorder=4)
    ax.set_xlabel("n")
    ax.set_ylabel("Mean Squared Error")
    plt.legend(loc=(1.02, 0.44))
    ax.set_yticks([], minor=True, style='plain')
    plt.savefig(f"plots/MSE_CKLRTE_tau_{tau}_nuis_{nuis}_plugin_all_{plugin_all}.pdf", bbox_inches="tight", dpi=100)
    ##################
    # Coverage Plots #
    ##################
    ax = ggplot_style_log(figsize=(7, 4), log_y=False)
    plt.plot(ns, coverage_ols[0][0], label="CKLRTE+OLS", zorder=5)
    plt.fill_between(ns, coverage_ols[0][0] - coverage_ols[0][1], coverage_ols[0][0] + coverage_ols[0][1], alpha=0.3, zorder=4)
    plt.plot(ns, coverage_ols[1][0], label="Plugin+OLS", zorder=5, color="black", ls="--")
    plt.fill_between(ns, coverage_ols[1][0] - coverage_ols[1][1], coverage_ols[1][0] + coverage_ols[1][1], 
                    alpha=0.3, zorder=4, color="gray")
    ax.set_xlabel("n")
    ax.set_ylabel("Coverage")
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1], minor=False)
    ax.set_yticks([], minor=True, style='plain')
    plt.legend(loc=(1.02, 0.77))
    plt.savefig(f"plots/Coverage_CKLRTE_tau_{tau}_nuis_{nuis}.pdf", bbox_inches="tight", dpi=100)