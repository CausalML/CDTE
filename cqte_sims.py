# Reproduces Figure 4 from "Robust and Agnostic Learning of Conditional Distributional Treatment Effects"

# Generic imports 
import numpy as np
import os
from tqdm import tqdm 
from econml.sklearn_extensions.linear_model import StatsModelsLinearRegression
from sklearn.linear_model import LogisticRegression, QuantileRegressor
from sklearn_quantile import RandomForestQuantileRegressor
from sklearn.ensemble import RandomForestRegressor
from joblib import Parallel, delayed
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

# Custom imports
from utils import write_results_to_file, PlottingSuite, ggplot_style_log
from nuisance import ( 
    RFKernel, RBFKernel, KernelQuantileRegressor
)
from dgp import lognormal_dgp, lognormal_true_effect, run_simulation
from cdte import CQTE

# Get true OLS coefs
def get_true_OLS_coefs(n, p, cond_mean_func, cond_std_func, tau):
    X = np.random.uniform(0, 1, size=(n, p))
    true_effects = lognormal_true_effect(X, cond_mean_func, cond_std_func, tau)
    ols = StatsModelsLinearRegression(cov_type="nonrobust")
    ols.fit(X, true_effects[f"quantile"])
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

def exp_kernel_generator(h=1):
    return lambda x: 1/h * np.exp(-x**2/h**2/2)


# Define nuisances and last stages to loop over
nuiss = ["QRF_RBF", "LQR", "QRF_RBF"]
final_stages = ["OLS", "RF"]

# Define experiment parameters
n_iter = 100
ns = [100, 200, 400, 800, 1600, 3200, 6400, 12800]

for nuis in nuiss:
    for final_stage in final_stages:
        print(f"Running CQTE experiments with nuisance {nuis} and final stage {final_stage}...")
        for n in tqdm(ns):
            ##############
            # Propensity #
            ##############
            propensity_model = LogisticRegression(solver="saga", penalty="elasticnet", l1_ratio=0.5)
            #####################
            # Quantile Nuisance #
            #####################
            if nuis == "QRF":
                quantile_model = RandomForestQuantileRegressor(
                    n_estimators=50,
                    min_samples_leaf=0.05,
                    q=tau)
            elif nuis == "LQR":
                quantile_model = QuantileRegressor(quantile=tau, alpha=0.01, solver='highs')
            elif nuis == "QRF_RBF":
                # RBF scale via Silverman's rule of thumb
                RBF_scale = 0.9*n**(-1/(4+p))
                quantile_model = KernelQuantileRegressor(
                        kernel=RBFKernel(scale=RBF_scale),
                        tau=tau)
            ####################
            # Density Nuisance #
            ####################
            cond_density_kernel = exp_kernel_generator(h=1)
            #cond_density_model = Lasso(alpha=0.05)
            cond_density_model = RandomForestRegressor(n_estimators=50, 
                                                        max_depth=5, 
                                                        min_samples_leaf=5,
                                                        min_samples_split=10)
            #####################
            # CQTE final model #
            #####################
            if final_stage == "OLS":
                cqte_model = StatsModelsLinearRegression(cov_type="HC1")
            elif final_stage == "RF":
                cqte_model = RandomForestRegressor(n_estimators=50, min_samples_leaf=0.05)
            #####################
            # CQTE Estimator #
            #####################
            CQTE_est = CQTE(
                propensity_model=propensity_model, 
                quantile_model=quantile_model,
                cond_density_kernel=cond_density_kernel,
                cond_density_model=cond_density_model,
                cqte_model=cqte_model, 
                tau=tau)
            trained_models = Parallel(n_jobs=-2, backend='loky', verbose=1)(
                                        delayed(run_simulation)(lognormal_dgp, CQTE_est, 
                                                                random_state,
                                                                n=n, p=p, 
                                                                cond_mean_func=cond_mean_func, 
                                                                cond_std_func=cond_std_func,
                                                                propensity_func=propensity_func,
                                                            ) for random_state in np.arange(n_iter))
            write_results_to_file(
                trained_models, 
                X_test, true_effects, true_coefs,
                n, p, tau, tail=None, nuis=nuis, final_stage=final_stage, dgp=dgp, cdte_name="CQTE")

# Make plots, one per nuisance class
for nuis in nuiss:
    if not os.path.exists("plots"):
        os.makedirs("plots")
    #############################
    # Process results from file #
    #############################
    final_stage = "OLS"
    ps = PlottingSuite(n_iter, ns, p, tau, None, nuis, final_stage, dgp, cdte_name="CQTE")
    mse_ols = ps.plot_mse(show_plugin_final=True, log_yscale=True, show=False)
    coverage_ols = ps.plot_coverage(coef_idx=1, show_plugin_final=True, show=False)
    final_stage = "RF"
    ps = PlottingSuite(n_iter, ns, p, tau, None, nuis, final_stage, dgp, cdte_name="CQTE")
    mse_rf = ps.plot_mse(show_plugin_final=True, log_yscale=True, show=False)
    #############
    # MSE Plots #
    #############
    plugin_all = True
    ax = ggplot_style_log(figsize=(7, 4), log_y=True)
    # CQTE + OLS
    plt.plot(ns, mse_ols[0][0], label="CQTE+OLS", color="C0", zorder=5)
    plt.fill_between(ns, mse_ols[0][0] - mse_ols[0][1], mse_ols[0][0] + mse_ols[0][1], alpha=0.3, color="C0", zorder=4)
    # CQTE + RF
    plt.plot(ns, mse_rf[0][0], label="CQTE+RF", color="C3", zorder=5)
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
    plt.savefig(f"plots/MSE_CQTE_tau_{tau}_nuis_{nuis}_plugin_all_{plugin_all}.pdf", bbox_inches="tight", dpi=100)
    ##################
    # Coverage Plots #
    ##################
    ax = ggplot_style_log(figsize=(7, 4), log_y=False)
    plt.plot(ns, coverage_ols[0][0], label="CQTE+OLS", zorder=5)
    plt.fill_between(ns, coverage_ols[0][0] - coverage_ols[0][1], coverage_ols[0][0] + coverage_ols[0][1], alpha=0.3, zorder=4)
    plt.plot(ns, coverage_ols[1][0], label="Plugin+OLS", zorder=5, color="black", ls="--")
    plt.fill_between(ns, coverage_ols[1][0] - coverage_ols[1][1], coverage_ols[1][0] + coverage_ols[1][1], 
                    alpha=0.3, zorder=4, color="gray")
    ax.set_xlabel("n")
    ax.set_ylabel("Coverage")
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1], minor=False)
    ax.set_yticks([], minor=True, style='plain')
    plt.legend(loc=(1.02, 0.77))
    plt.savefig(f"plots/Coverage_CQTE_tau_{tau}_nuis_{nuis}.pdf", bbox_inches="tight", dpi=100)