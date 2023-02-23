# Reproduces Figures 3 and 6 from "Robust and Agnostic Learning of Conditional Distributional Treatment Effects"

import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from sklearn_quantile import RandomForestQuantileRegressor
from econml.dr import DRLearner
from econml.sklearn_extensions.linear_model import StatsModelsLinearRegression
from doubleml.datasets import fetch_401K
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from nuisance import ( 
    RFKernel, KernelSuperquantileRegressor
)
from cdte import CSQTE
from utils import ggplot_style_grid

###################
# Data Processing #
###################
# Get 401K data
df = fetch_401K(return_type='DataFrame')
# Select features
# X1: age (int)
# X2: inc -> income (int)
# X3: educ -> education, in #years completed (int)
# X4: fsize -> family size (int)
# X5: marr -> marrital status (binary)
# X6: two_earn -> two earners (binary)
# X7: db -> defined benefit pension status (binary)
# X8: pira -> IRA participation
# X9: hown -> home ownership
# A: e401 -> 401 (k) eligibility (binary)
# Y: net_tfa -> net financial assets (float)
feat_names = ['age', 'inc', 'educ', 'fsize', 'marr', 'twoearn', 'db', 'pira', 'hown']
X = df[feat_names].values
A = df["e401"].values
Y = df["net_tfa"].values

##################
# RF Final Stage #
##################
# Set hyperparameters as in Chernozhukov et al. (https://docs.doubleml.org/stable/examples/py_double_ml_pension.html)
random_state = 12345
n_estimators = 100
max_depth = 7
max_features = 3
min_samples_leaf = 10
#Propensity model
propensity_model = RandomForestClassifier(
                        n_estimators=n_estimators, 
                        max_depth=max_depth, 
                        max_features=max_features, 
                        min_samples_leaf=min_samples_leaf, 
                        n_jobs=-2)
# CSQTE model
csqte_model = RandomForestRegressor(
                        n_estimators=n_estimators, 
                        max_depth=max_depth, 
                        max_features=max_features, 
                        min_samples_leaf=min_samples_leaf, 
                        n_jobs=-2)
# Left 25% tail
tau=0.25
tail="left"
quantile_model = RandomForestQuantileRegressor(n_estimators=n_estimators, 
                        max_depth=max_depth, 
                        max_features=max_features, 
                        min_samples_leaf=min_samples_leaf, 
                        n_jobs=-2,
                        q=tau)
superquantile_model = KernelSuperquantileRegressor(
                        kernel=RFKernel(
                            RandomForestRegressor(
                                n_estimators=n_estimators, 
                                max_depth=max_depth, 
                                max_features=max_features, 
                                min_samples_leaf=min_samples_leaf, 
                                n_jobs=-2)
                                ),
                        tau=tau,
                        tail=tail)
CSQTE_est_left = CSQTE(
    propensity_model=propensity_model,
    quantile_model=quantile_model,
    superquantile_model=superquantile_model,
    csqte_model=csqte_model, 
    nested_quantiles=False,
    tau=tau,
    tail=tail,
    random_state=random_state)
CSQTE_est_left.fit(X, A, Y)
# Right 75% tail
tau=0.75
tail="right"
quantile_model = RandomForestQuantileRegressor(n_estimators=n_estimators, 
                        max_depth=max_depth, 
                        max_features=max_features, 
                        min_samples_leaf=min_samples_leaf, 
                        n_jobs=-2,
                        q=tau)
superquantile_model = KernelSuperquantileRegressor(
                        kernel=RFKernel(
                            RandomForestRegressor(
                                n_estimators=n_estimators, 
                                max_depth=max_depth, 
                                max_features=max_features, 
                                min_samples_leaf=min_samples_leaf, 
                                n_jobs=-2)
                                ),
                        tau=tau,
                        tail=tail)
CSQTE_est_right = CSQTE(
    propensity_model=propensity_model,
    quantile_model=quantile_model,
    superquantile_model=superquantile_model,
    csqte_model=csqte_model, 
    nested_quantiles=False,
    tau=tau,
    tail=tail,
    random_state=random_state)
CSQTE_est_right.fit(X, A, Y)
# CATEs
model_regression = RandomForestRegressor(
                        n_estimators=n_estimators, 
                        max_depth=max_depth, 
                        max_features=max_features, 
                        min_samples_leaf=min_samples_leaf, 
                        n_jobs=-2)
CATE_est = DRLearner(
    model_propensity=propensity_model, 
    model_regression=model_regression, 
    model_final=csqte_model, 
    cv=5)
CATE_est.fit(Y, A, X=X)

# Feature importances
if not os.path.exists("plots"):
        os.makedirs("plots")
ax = ggplot_style_grid(figsize=(4, 6))

importances = CSQTE_est_left.csqte_model.feature_importances_
indices = np.argsort(importances)
plt.barh(4*np.arange(len(indices)), importances[indices], align='center', label="CSQTE,\nbottom 25%", zorder=5, color="C0")

importances = CATE_est.fitted_models_final[0].feature_importances_
indices = np.argsort(importances)
plt.barh(4*np.arange(len(indices))+1, importances[indices], align='center', label="CATE", zorder=5, color="darkgray")

importances = CSQTE_est_right.csqte_model.feature_importances_
indices = np.argsort(importances)
plt.barh(4*np.arange(len(indices))+2, importances[indices], align='center', label="CSQTE,\ntop 25%", zorder=5, color="C3")

pretty_feat_names = ["Age", "Income", "Education", "Family size", "Marital status", "Two earners", "Has pension", "Has IRA", "Owns home"]
plt.title('Feature Importances')
plt.yticks(4*np.arange(len(indices))+1, [pretty_feat_names[i] for i in indices])
plt.xlabel('Relative Importance')
plt.ylabel("Features")
plt.legend(prop={'size': 12})
plt.savefig(f"plots/401k_RF_feat_importances.pdf", bbox_inches="tight", dpi=100)

# Plot histogram
ax = ggplot_style_grid(figsize=(4, 4))
effect_bins = np.arange(-20000, 60000, 2500)
plt.hist(CSQTE_est_left.csqte_model.predict(X), bins=effect_bins, histtype="step", label="CSQTE,\n bottom 25%", zorder=5, color="C0", 
         density=True, lw=1.2)
plt.hist(CATE_est.fitted_models_final[0].predict(X), bins=effect_bins, histtype="step", label="CATE", zorder=6, color="black", ls='--',
        density=True, lw=1.2)
plt.hist(CSQTE_est_right.csqte_model.predict(X), bins=effect_bins, histtype="step", label="CSQTE,\n top 25%", zorder=7, color="C3",
        density=True, lw=1.2)
plt.xlabel("Effect")
plt.ylabel("Density")
plt.legend(prop={'size': 12})
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.savefig(f"plots/401k_hist.pdf", bbox_inches="tight", dpi=100)

##################
# OLS Projection #
##################
random_state = 12345
csqte_model = StatsModelsLinearRegression()
# Left 25% tail
tau=0.25
tail="left"
quantile_model = RandomForestQuantileRegressor(n_estimators=n_estimators, 
                        max_depth=max_depth, 
                        max_features=max_features, 
                        min_samples_leaf=min_samples_leaf, 
                        n_jobs=-2,
                        q=tau)
superquantile_model = KernelSuperquantileRegressor(
                        kernel=RFKernel(
                            RandomForestRegressor(n_estimators=n_estimators, 
                                max_depth=max_depth, 
                                max_features=max_features, 
                                min_samples_leaf=min_samples_leaf, 
                                n_jobs=-2)
                                ),
                        tau=tau,
                        tail=tail)
CSQTE_est_left_proj = CSQTE(
    propensity_model=propensity_model,
    quantile_model=quantile_model,
    superquantile_model=superquantile_model,
    csqte_model=csqte_model, 
    nested_quantiles=False,
    tau=tau,
    tail=tail,
    proj_idx=[0, 1, 2],
    random_state=random_state
)
CSQTE_est_left_proj.fit(X, A, Y)
# Right 75% tail
tau=0.75
tail="right"
quantile_model = RandomForestQuantileRegressor(
                        n_estimators=n_estimators, 
                        max_depth=max_depth, 
                        max_features=max_features, 
                        min_samples_leaf=min_samples_leaf, 
                        n_jobs=-2,
                        q=tau)
superquantile_model = KernelSuperquantileRegressor(
                        kernel=RFKernel(
                            RandomForestRegressor(
                                n_estimators=n_estimators, 
                                max_depth=max_depth, 
                                max_features=max_features, 
                                min_samples_leaf=min_samples_leaf, 
                                n_jobs=-2)
                            ),
                        tau=tau,
                        tail=tail)
CSQTE_est_right_proj = CSQTE(
    propensity_model=propensity_model,
    quantile_model=quantile_model,
    superquantile_model=superquantile_model,
    csqte_model=csqte_model, 
    nested_quantiles=False,
    tau=tau,
    tail=tail,
    proj_idx=[0, 1, 2],
    random_state=random_state
)
CSQTE_est_right_proj.fit(X, A, Y)
# CATEs
model_regression = RandomForestRegressor( 
                        n_estimators=n_estimators, 
                        max_depth=max_depth, 
                        max_features=max_features, 
                        min_samples_leaf=min_samples_leaf, 
                        n_jobs=-2,
                        random_state=random_state)
CATE_est_proj = DRLearner(
                        model_propensity=propensity_model, 
                        model_regression=model_regression, 
                        model_final=csqte_model, 
                        cv=5)
CATE_est_proj.fit(Y, A, X=X[:, [0, 1, 2]], W=X)
# Make table
d = {"": ["Intercept"] + feat_names[:3]}
models = [
    ("Bottom 25%", CSQTE_est_left_proj.csqte_model), 
    ("CATE", CATE_est_proj.fitted_models_final[0]), 
    ("Top 25%", CSQTE_est_right_proj.csqte_model)]
for model_name, model in models:
    coefs = []
    coefs.append(
        f"{model.intercept_:0.2f} [{model.intercept__interval()[0]:0.2f}, {model.intercept__interval()[1]:0.2f}]")
    for i, c in enumerate(model.coef_):
        coefs.append(
        f"{c:0.2f} [{model.coef__interval()[0][i]:0.2f}, {model.coef__interval()[1][i]:0.2f}]")
    d[model_name] = coefs
pd.DataFrame(d).to_csv("plots/401k_OLS_proj_coefs.csv", index=False)


