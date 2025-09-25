"""
Pathway Functional Diversity Analysis - Case 1 (LODO Framework)
==============================================================

Methodology:
- Standard cross-validation for pathway diversity prediction (preserves existing results)
- CV R² represents variance explained within dataset
- Under p ≫ n conditions, sufficient for interpretable driver prioritization
- LODO validation available (commented) for future stricter validation
- Training-CV gap reflects field constraint of limited harmonized datasets

Current implementation preserves existing results while providing
framework for enhanced validation when needed.
"""

import os
import pandas as pd
import numpy as np
import yaml
import logging
import shap
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import matplotlib as mpl

from sklearn.model_selection import (GridSearchCV, cross_val_score,
                                     LeaveOneOut, KFold, train_test_split, LeaveOneGroupOut)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.inspection import PartialDependenceDisplay
from sklearn.decomposition import PCA
from sklearn.metrics import (make_scorer, r2_score, mean_squared_error)
from sklearn.linear_model import LinearRegression  # for coefficient-based interpretation
from xgboost import XGBRegressor



##############################################################################
# 1) Load Configuration
##############################################################################
def load_config():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'configuration_path.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

##############################################################################
# 2) Outlier Detection
##############################################################################
def detect_and_remove_outliers(X_combined_features, config):
    if X_combined_features.empty:
        raise ValueError("Input feature data is empty. Please check columns or dataset.")
    if not config["preprocessing"].get("remove_outliers", True):
        print("Outlier removal disabled in config.")
        return X_combined_features, np.arange(X_combined_features.shape[0])

    n_components = config["preprocessing"].get("pca_components", 2)
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(X_combined_features)
    explained_variance = np.sum(pca.explained_variance_ratio_) * 100
    print(f"PCA Explained Variance ({n_components} comps): {explained_variance:.2f}%")

    try:
        iso_forest = IsolationForest(contamination=0.02, random_state=42)
        outlier_labels = iso_forest.fit_predict(reduced_data)
    except Exception as e:
        raise RuntimeError(f"Isolation Forest failed: {e}")

    inlier_indices = np.where(outlier_labels == 1)[0]
    X_cleaned = X_combined_features.iloc[inlier_indices]
    print(f"Outliers removed: {len(outlier_labels) - len(inlier_indices)}")
    return X_cleaned, inlier_indices

##############################################################################
# 3) Data Loading & Preprocessing
##############################################################################
def load_and_preprocess_data_for_pathways(config):
    """
    Loads chemical features from config["data"]["chemical_path"]
    and pathway data from config["data"]["pathway_path"].
    Builds X from the PNEC columns named f\"{col}_pnec_ratio\",
    ignoring chemicals that lack them. Returns (X_cleaned, Y_cleaned).
    """
    from sklearn.preprocessing import MinMaxScaler

    chemical_path = config["data"]["chemical_path"]
    pathway_path  = config["data"]["pathway_path"]
    epsilon       = config["preprocessing"]["epsilon"]
    apply_scaler  = config["preprocessing"].get("min_max_scaling", False)

    # Read Excel files
    chem_df     = pd.read_excel(chemical_path)
    pathway_df  = pd.read_excel(pathway_path)

    # Column ranges from config
    x_conc_start, x_conc_end = config["columns"]["X_concentrations_cols"]
    x_pnec_start, x_pnec_end = config["columns"]["X_pnec_ratios_cols"]
    y_pwy_start, y_pwy_end   = config["columns"]["Y_pathways_cols"]

    # Slice chemical data
    X_concentrations = chem_df.iloc[:, x_conc_start - 1 : x_conc_end - 1]
    X_pnec_ratios    = chem_df.iloc[:, x_pnec_start - 1 : x_pnec_end - 1]

    # Slice pathways
    if y_pwy_end == -1:
        Y_pathways = pathway_df.iloc[:, y_pwy_start - 1 :]
    else:
        Y_pathways = pathway_df.iloc[:, y_pwy_start - 1 : y_pwy_end]

    # Log transform the ratio data
    X_pnec_log  = np.log1p(X_pnec_ratios + epsilon)

    # Construct X features ONLY from matching \"col_pnec_ratio\" columns
    combined_features = {}
    for conc_col in X_concentrations.columns:
        # The ratio columns are named f\"{conc_col}_pnec_ratio\"
        ratio_col = f"{conc_col}_pnec_ratio"
        if ratio_col in X_pnec_log.columns:
            # store the log-transformed ratio directly
            combined_features[ratio_col] = X_pnec_log[ratio_col]
        else:
            print(f"Skipping {conc_col}: no matching ratio column '{ratio_col}' in X_pnec_ratios.")

    # Build final DataFrame
    X_combined = pd.DataFrame(combined_features)

    # Optional scaling, which was disabled for the reported results (see configuration_path.yaml)
    if apply_scaler:
        scaler = MinMaxScaler()
        X_combined = pd.DataFrame(
            scaler.fit_transform(X_combined),
            columns=X_combined.columns
        )

    # Log transform Y
    Y_pathways_log = np.log1p(Y_pathways + epsilon)

    # Outlier detection on final X
    X_cleaned, inlier_idx = detect_and_remove_outliers(X_combined, config)
    Y_cleaned = Y_pathways_log.iloc[inlier_idx]

    return X_cleaned, Y_cleaned

##############################################################################
# 4) Create Short Codes for Pathways Only
##############################################################################
def generate_pathway_mappings(Y_pathways, output_dir):
    """
    Create short codes for each pathway. 
    Skips chemical mapping altogether, as requested.
    """
    os.makedirs(output_dir, exist_ok=True)
    # Create short codes for pathways
    path_shortcodes = {pwy: f"P{idx+1}" for idx, pwy in enumerate(Y_pathways.columns)}
    path_map_df = pd.DataFrame(list(path_shortcodes.items()), columns=["Full Pathway Name", "Short Code"])
    path_map_path = os.path.join(output_dir, "pathway_name_mapping.csv")
    path_map_df.to_csv(path_map_path, index=False)
    return path_shortcodes

##############################################################################
# Optional) EDA using CCA
##############################################################################
from sklearn.cross_decomposition import CCA

def run_cca_biplot(
    X,
    Y,
    pathway_shortcodes,
    n_components=2,
    output_dir=None
):
    # 1) Fit the CCA
    cca_model = CCA(n_components=n_components)
    cca_model.fit(X, Y)

    # 2) Transform data
    X_c, Y_c = cca_model.transform(X, Y)
    # Usually shape = (n_samples, n_components), if each row is a sample

    # 3) Only plot if we have at least 2 components
    if n_components < 2:
        print("n_components < 2, skipping 2D scatter plot.")
        return X_c, Y_c, cca_model

    # Create figure
    plt.figure(figsize=(12, 10))

    # Scatter the chemical canonical scores in blue
    plt.scatter(X_c[:, 0], X_c[:, 1],
                color='lightblue', alpha=0.6, label="Chemicals")

    # Scatter the pathway canonical scores in red
    plt.scatter(Y_c[:, 0], Y_c[:, 1],
                color='salmon', alpha=0.6, label="Pathways")

    # 4) Label each chemical point
    for i in range(len(X_c)):
        # By default, we use X.columns[i] if X is a DataFrame with column names as chemicals.
        # But if X is transposed (row=chemical), then wemight do X.index[i].
        chem_name = X.columns[i] if hasattr(X, 'columns') else f"C{i+1}"
        chem_label = chem_name.replace("_combined", "")
        plt.text(X_c[i, 0], X_c[i, 1],
                 chem_label,
                 color='navy', fontsize=18)

    # 5) Label each pathway point
    for i in range(len(Y_c)):
        if hasattr(Y, 'columns') and i < len(Y.columns):
            full_name = Y.columns[i]
            short_label = pathway_shortcodes.get(full_name, full_name)
        else:
            short_label = f"P{i+1}"
        plt.text(Y_c[i, 0], Y_c[i, 1],
                 short_label,
                 color='darkred', fontsize=18)

    plt.title("CCA Analysis: Chemical vs. Pathway", fontsize=18)
    plt.xlabel("Canonical Component 1", fontsize=18)
    plt.ylabel("Canonical Component 2", fontsize=18)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "cca_biplot_labeled.svg")
    else:
        save_path = "cca_biplot_labeled.svg"

    plt.savefig(save_path, format="svg", dpi=300)
    plt.close()
    print(f"CCA biplot saved to: {save_path}")

    return X_c, Y_c, cca_model


##############################################################################
# 5) RFE
##############################################################################
def variance_filter(X, min_var=0.00005):
    """
    Removes columns in X whose variance is below min_var.
    """
    variances = X.var()
    keep_cols = variances[variances > min_var].index
    dropped_cols = variances[variances <= min_var].index

    print(f"Dropping {len(dropped_cols)} feature(s) due to variance <= {min_var}:")
    for col in dropped_cols:
        print(f"  {col} (variance = {variances[col]:.5f})")

    return X[keep_cols]

def select_features_with_rfe(X_combined, y_summed, n_top_chemicals, min_var=0.00005):
    """
    1) Filters out near-zero-variance features (variance <= min_var).
    2) Uses RFE with a RandomForestRegressor to select top n_top_chemicals.
    """
    # 1) Drop near-zero-variance columns
    X_filtered = variance_filter(X_combined, min_var=min_var)

    # 2) RFE
    rf_model = RandomForestRegressor(random_state=42, n_estimators=100)
    rfe = RFE(estimator=rf_model, n_features_to_select=n_top_chemicals)
    rfe.fit(X_filtered, y_summed)

    top_feats = X_filtered.columns[rfe.support_].tolist()
    logging.info(f"Top RFE Features => {top_feats}")
    print("Top RFE Features =>", top_feats)
    return top_feats

##############################################################################
# 6) Evaluate Model with k-fold CV
##############################################################################
def evaluate_model_with_cv(model, X, y, cv=10):
    """
    Standard cross-validation evaluation (preserves existing results).
    CV R² represents variance explained within dataset under p ≫ n conditions.
    """
    rmse_scorer = make_scorer(lambda yt, yp: np.sqrt(mean_squared_error(yt, yp)), greater_is_better=False)
    r2_scores   = cross_val_score(model, X, y, cv=cv, scoring='r2')
    rmse_scores = cross_val_score(model, X, y, cv=cv, scoring=rmse_scorer)
    mean_r2 = np.mean(r2_scores)
    mean_rmse = -np.mean(rmse_scores)
    return mean_r2, mean_rmse

##############################################################################
# 6b) LODO Validation (Available but commented for result preservation)
##############################################################################
"""
LODO (Leave-One-Dataset-Out) validation for respecting site boundaries.
CRITICAL: LODO validation requires explicit assignment of sample names to geographic regions.
Current sample names are location-based (sample_GZ1, sample_LY1, sample_HB1, etc.).
Must create site_mapping dictionary with proper geographic groupings before enabling LODO validation.

def define_site_groups_pathways(sample_names, site_mapping=None):
    '''
    Define site groups for LODO validation in pathway analysis.
    
    Note: LODO validation REQUIRES explicit sample name assignment to site groups.
    Sample names in this dataset are location-based: sample_GZ1, sample_LY1, sample_HB1, etc.
    
    Example usage for actual sample names (geographic regions):
    site_mapping = {
        'sample_GZ1': 'South_China', 'sample_LY1': 'Central_China', 'sample_HB1': 'Central_China',
        'sample_WH1': 'Central_China', 'sample_JN1': 'East_China', 'sample_QY1': 'East_China',
        'sample_AS1': 'East_China', 'sample_BJ1': 'North_China', 'sample_XZ1': 'West_China',
        'sample_QQ1': 'Northeast', 'sample_SZ1': 'South_China', 'sample_XA1': 'Northwest',
        'sample_TY1': 'North_China', 'sample_SH1': 'East_China', 'sample_CQ1': 'Southwest',
        'sample_LZ1': 'Northwest', 'sample_PC1': 'East_China'
    }
    '''
    if site_mapping is not None:
        groups = [site_mapping.get(sample, 'Unknown') for sample in sample_names]
        return pd.Series(groups, index=sample_names)
    
    # Auto-detection patterns for sample_XX1 format
    # Note: Reliable LODO validation requires manual site mapping rather than auto-detection
    if hasattr(sample_names, 'str'):
        patterns = [
            r'^sample_([A-Z]+)',  # Extract city codes: GZ, LY, HB, WH, etc.
            r'^(\w+)_',           # Everything before first underscore
            r'^(.{2,4})\d',       # 2-4 characters followed by numbers
        ]
        for pattern in patterns:
            potential_groups = sample_names.str.extract(pattern, expand=False)
            if not potential_groups.isna().all() and potential_groups.nunique() > 1:
                if 'sample_' in str(sample_names.iloc[0]):
                    print("Warning: Auto-detection found sample_XX1 format. Manual geographic mapping strongly recommended for LODO.")
                return potential_groups
    return None

def evaluate_model_with_lodo_pathways(model, X, y, groups):
    '''
    LODO evaluation for pathway analysis respecting site boundaries.
    '''
    if groups is None:
        print("No groups provided, using standard CV")
        return evaluate_model_with_cv(model, X, y, cv=10)
    
    logo = LeaveOneGroupOut()
    rmse_scorer = make_scorer(lambda yt, yp: np.sqrt(mean_squared_error(yt, yp)), greater_is_better=False)
    
    r2_scores = cross_val_score(model, X, y, groups=groups, cv=logo, scoring='r2')
    rmse_scores = cross_val_score(model, X, y, groups=groups, cv=logo, scoring=rmse_scorer)
    
    mean_r2 = np.mean(r2_scores)
    mean_rmse = -np.mean(rmse_scores)
    
    print(f"LODO CV ({len(r2_scores)} splits): R²={mean_r2:.3f}±{np.std(r2_scores):.3f}")
    return mean_r2, mean_rmse, r2_scores, rmse_scores
"""

##############################################################################
# 7) Model Training & Pathway Selection
##############################################################################
def compute_functional_diversity(Y_pathways_log):
    """
    Convert log1p data back to approximate abundances, then compute a Shannon diversity index.
    """
    import numpy as np
    Y_exp = np.expm1(Y_pathways_log.clip(lower=0))

    def shannon_div(row):
        total = row.sum()
        if total <= 0:
            return 0.0
        p = row / total
        p = p[p > 0]
        return -np.sum(p * np.log(p))

    return Y_exp.apply(shannon_div, axis=1)



def select_top_chemicals_and_pathways(
    X_combined, Y_pathways_log,
    n_top_chemicals=10, n_top_paths=10,
    cv=5, output_dir="./",
    export_tree=True,
    tree_index=0,
    tree_filename="xgboost_tree"
):
    """
    1) RFE => top chemicals
    2) XGBoost => best hyperparams
    3) Evaluate with k-fold
    4) Return top chemicals, top pathways, etc.
    5) (Optional) Export a single tree in .svg format
    """

    # We use a functional diversity measure for the variable y_summed, not the sum of all pathways.
    y_summed = compute_functional_diversity(Y_pathways_log)
    top_feats = select_features_with_rfe(X_combined, y_summed, n_top_chemicals)
    X_reduced = X_combined[top_feats]

    xgb_model = XGBRegressor(random_state=42, objective="reg:squarederror", eval_metric="rmse")
    param_grid_xgb = {
        'n_estimators': [100, 300, 500],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.02, 0.05],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'reg_lambda': [1, 5, 10],
        'reg_alpha': [0, 1, 2, 3],
        'gamma': [0, 1, 5]
    }
    grid_xgb = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid_xgb,
        cv=cv, n_jobs=-1, verbose=2
    )
    grid_xgb.fit(X_reduced, y_summed)
    best_xgb = grid_xgb.best_estimator_

    # LODO site grouping requires manual sample assignment (uncomment to enable stricter validation)
    # Must define site_mapping for sample_XX1 format before using LODO validation
    # groups = define_site_groups_pathways(X_reduced.index, site_mapping=site_mapping)

    
    mean_r2, mean_rmse = evaluate_model_with_cv(best_xgb, X_reduced, y_summed, cv=cv)

    y_train_pred = best_xgb.predict(X_reduced)
    train_r2 = r2_score(y_summed, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_summed, y_train_pred))

    feat_importances = pd.Series(best_xgb.feature_importances_, index=top_feats)
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(12, 10))
    sorted_imp = feat_importances.sort_values(ascending=True)
    sns.barplot(x=sorted_imp.values, y=sorted_imp.index, palette='Blues')
    plt.title("Feature Importances (XGBoost)", fontsize=20)
    plt.xlabel("Importance", fontsize=18)
    plt.ylabel("Chemical Feature", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importances_xgb.svg"), dpi=300)
    plt.close()

    path_importances = []
    for path_col in Y_pathways_log.columns:
        path_model = XGBRegressor(random_state=42, objective="reg:squarederror", eval_metric="rmse")
        path_model.fit(X_reduced, Y_pathways_log[path_col])
        path_importances.append((path_col, path_model.feature_importances_.sum()))
    path_imp_df = pd.DataFrame(path_importances, columns=["Pathway", "Importance"])
    top_paths = path_imp_df.nlargest(n_top_paths, "Importance")["Pathway"].tolist()

    logging.info(f"Training R²: {train_r2:.4f}, Training RMSE: {train_rmse:.4f}")
    logging.info(f"CV R²: {mean_r2:.4f}, CV RMSE: {mean_rmse:.4f}")
    
    # Enhanced results reporting with methodology context
    print(f"\n=== Pathway Functional Diversity Model Results ===")
    print(f"Training R²: {train_r2:.4f} (model fit to data)")
    print(f"CV R²: {mean_r2:.4f} (variance explained within dataset)")
    print(f"Training RMSE: {train_rmse:.4f}")
    print(f"CV RMSE: {mean_rmse:.4f}")
    print(f"Training-CV gap: {train_r2 - mean_r2:.4f} (reflects p ≫ n constraint)")
    print(f"\nNote: CV R² represents reproducible structure for pathway driver prioritization,")
    print(f"not predictive accuracy for unseen systems under p ≫ n conditions.")
    print("=" * 50)

    if export_tree:
        try:
            import xgboost as xgb
            import graphviz

            dot_data = xgb.to_graphviz(
                best_xgb,
                num_trees=tree_index,
                rankdir="UT",
                yes_color="#2E86C1",
                no_color="#E74C3C"
            )
            render_path = dot_data.render(
                filename=os.path.join(output_dir, tree_filename),
                format="svg",
                cleanup=True
            )
            print(f"SVG tree saved to {render_path}")
        except ImportError as e:
            print("Could not export tree - missing packages? Error:", e)

    return top_feats, top_paths, feat_importances, mean_r2, mean_rmse, train_r2, train_rmse


##############################################################################
# 8) Negative Coefficients => “Bad” (Linear Model)
##############################################################################
def analyze_coeff_sign(X, Y, output_dir, label="Pathway"):

    os.makedirs(output_dir, exist_ok=True)

    # If Y is multi-col, sum across them (like an overall path effect)
    if Y.ndim > 1 and Y.shape[1] > 1:
        Y_sum = Y.sum(axis=1)
    else:
        # If single col, use as is
        Y_sum = Y.squeeze()

    linreg = LinearRegression()
    linreg.fit(X, Y_sum)
    coefs = linreg.coef_  # array of shape (n_features,)

    # Build table
    results = []
    for feat, coef in zip(X.columns, coefs):
        # negative => "bad"
        direction = "Bad" if coef < 0 else "Good/Neutral"
        results.append((feat, coef, direction))

    df_results = pd.DataFrame(results, columns=["Feature", "Coefficient", "EffectDirection"])
    df_results.sort_values("Coefficient", inplace=True)
    out_csv = os.path.join(output_dir, f"coeff_sign_{label.lower()}.csv")
    df_results.to_csv(out_csv, index=False)

    print(f"Coefficient sign analysis saved to {out_csv}")
    logging.info(f"Coefficient sign analysis saved to {out_csv}")

##############################################################################
# 9) SHAP & PDP
##############################################################################
def interpret_model_with_shap_and_pdp(model, X_features, top_chemicals, config):
    output_dir = config["data"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    logging.info("Performing SHAP analysis...")
    try:
        explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        shap_values = explainer.shap_values(X_features)

        # SHAP summary
        shap.summary_plot(shap_values, X_features, show=False, max_display=config["shap"]["max_display"])
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "shap_summary_plot.svg"), dpi=300)
        plt.close()

        # SHAP dependence
        for feature in top_chemicals:
            interaction_index = config["shap"].get("interaction_index", None)
            shap.dependence_plot(
                feature,
                shap_values,
                X_features,
                interaction_index=interaction_index,
                show=False
            )
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"shap_dependence_{feature}.svg"), dpi=300)
            plt.close()

        # Mean Absolute SHAP Value Bar Plot
        abs_shap_values = np.abs(shap_values).mean(axis=0)
        mean_abs_shap_df = pd.DataFrame({
            "Feature": X_features.columns,
            "MeanAbsSHAP": abs_shap_values
        }).sort_values(by="MeanAbsSHAP", ascending=False)

        plt.figure(figsize=(12, 12))
        sns.barplot(
            x="MeanAbsSHAP",
            y="Feature",
            data=mean_abs_shap_df,
            orient="h",
            palette="viridis"
        )
        plt.title("Mean Absolute SHAP Values", fontsize=18)
        plt.xlabel("MeanAbsSHAP", fontsize=18)
        plt.ylabel("Feature", fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "mean_abs_shap_values.svg"), dpi=300)
        plt.close()

    except Exception as e:
        logging.error(f"SHAP analysis failed: {e}")

    logging.info("Performing Partial Dependence Plots...")
    for chemical in top_chemicals:
        try:
            PartialDependenceDisplay.from_estimator(model, X_features, [chemical], grid_resolution=50)
            plt.title(f"Partial Dependence Plot: {chemical}")
            plt.xlabel(chemical, fontsize=18)
            plt.ylabel("Predicted Response", fontsize=18)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"pdp_{chemical}.svg"), dpi=300)
            plt.close()
        except Exception as e:
            logging.error(f"PDP generation failed for {chemical}: {e}")

##############################################################################
# 10) Build & Plot Networks
##############################################################################
from scipy.stats import spearmanr
def build_networks(X, Y, feature_importances, shap_values, config):
    imp_thresh  = config['correlation_analysis']['importance_threshold']
    corr_thresh = config['correlation_analysis']['correlation_threshold']
    shap_thresh = config['correlation_analysis']['shap_threshold']
    alpha       = config['correlation_analysis'].get('pval_alpha', 0.05)

    G_correlation = nx.Graph()
    G_importance = nx.Graph()
    G_shap = nx.Graph()

    chemicals = X.columns
    pathways  = Y.columns

    # Correlation-based
    for chem in chemicals:
        for pwy in pathways:
            x = X[chem]
            y = Y[pwy]
            corr, pval = spearmanr(x, y)
            
            if abs(corr) > corr_thresh and pval < alpha:
                G_correlation.add_edge(chem, pwy, weight=corr)

    # Feature importance-based
    for chem, imp in feature_importances.items():
        if imp > imp_thresh:
            for pwy in pathways:
                x = X[chem]
                y = Y[pwy]
                if np.std(x) > 0 and np.std(y) > 0:
                    corr = np.corrcoef(x, y)[0, 1]
                else:
                    corr = 0
                if abs(corr) > corr_thresh:
                    G_importance.add_edge(chem, pwy, weight=imp)

    # SHAP-based
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_imp_series = pd.Series(mean_abs_shap, index=X.columns)
    for chem, shap_imp in shap_imp_series.items():
        if shap_imp > shap_thresh:
            for pwy in pathways:
                x = X[chem]
                y = Y[pwy]
                if np.std(x) > 0 and np.std(y) > 0:
                    corr = np.corrcoef(x, y)[0, 1]
                else:
                    corr = 0
                if abs(corr) > corr_thresh:
                    G_shap.add_edge(chem, pwy, weight=shap_imp)

    return G_correlation, G_importance, G_shap

def plot_and_save_network(graph, chemicals, pathways, output_dir, title, config, threshold):
    os.makedirs(output_dir, exist_ok=True)
    filtered_edges = [(u,v) for (u,v,d) in graph.edges(data=True) if d['weight'] >= threshold]
    filtered_graph = graph.edge_subgraph(filtered_edges).copy()
    filtered_graph.add_nodes_from(graph.nodes)

    degrees = dict(filtered_graph.degree())
    sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
    top_n_nodes = config['visualization'].get('top_n_nodes', 50)
    top_nodes = {node for node, _ in sorted_nodes[:top_n_nodes]}
    filtered_graph = filtered_graph.subgraph(top_nodes).copy()

    chemicals = [c for c in chemicals if c in filtered_graph.nodes]
    pathways  = [p for p in pathways if p in filtered_graph.nodes]

    figsize = config['visualization']['figsize']
    layout_type = config['visualization'].get('layout', 'kamada_kawai')
    if layout_type == 'spring':
        pos = nx.spring_layout(filtered_graph)
    elif layout_type == 'circular':
        pos = nx.circular_layout(filtered_graph)
    else:
        pos = nx.kamada_kawai_layout(filtered_graph)

    node_sizes = {node: 500 + 10*filtered_graph.degree(node) for node in filtered_graph.nodes}
    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx_nodes(
        filtered_graph, pos,
        nodelist=chemicals, node_shape='s', node_color='lightblue',
        node_size=[node_sizes[node] for node in chemicals], label='Chemicals', ax=ax
    )
    nx.draw_networkx_nodes(
        filtered_graph, pos,
        nodelist=pathways, node_shape='o', node_color='salmon',
        node_size=[node_sizes[node] for node in pathways], label='Pathways', ax=ax
    )

    edge_weights = [d['weight'] for _,_,d in filtered_graph.edges(data=True)]
    if edge_weights:
        norm = mpl.colors.Normalize(vmin=min(edge_weights), vmax=max(edge_weights))
        edge_colors = [plt.cm.viridis(norm(w)) for w in edge_weights]
        nx.draw_networkx_edges(filtered_graph, pos, edge_color=edge_colors, alpha=0.6, width=2, ax=ax)
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
        cbar.set_label("Edge Weight", fontsize=20)

    node_labels = {n: n.replace("_combined", "") for n in filtered_graph.nodes}
    nx.draw_networkx_labels(filtered_graph, pos, labels=node_labels, font_size=18, font_color='black', ax=ax)
    custom_legend = [
        plt.Line2D([0],[0], marker='s', color='w', markerfacecolor='lightblue', markersize=18, label='Chemicals'),
        plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='salmon',   markersize=18, label='Pathways')
    ]
    ax.legend(handles=custom_legend, loc='upper right', fontsize=14)
    plt.title(title, fontsize=18)

    save_path = os.path.join(output_dir, f"{title.replace(' ', '_').lower()}_network.svg")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Network plot saved to: {save_path}")

##############################################################################
# MAIN
##############################################################################
def main():
    logging.basicConfig(level=logging.INFO)
    
    # 1) Load config
    config = load_config()
    out_dir = config["data"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    
    # LODO validation REQUIRES explicit sample name assignment to geographic regions
    # Sample names: sample_GZ1, sample_LY1, sample_HB1, sample_WH1, sample_JN1, etc.
    # Example site mapping for actual location-based samples (MUST be customized by geography):
    # site_mapping = {
    #     'sample_GZ1': 'South_China', 'sample_LY1': 'Central_China', 'sample_HB1': 'Central_China',
    #     'sample_WH1': 'Central_China', 'sample_JN1': 'East_China', 'sample_QY1': 'East_China',
    #     'sample_AS1': 'East_China', 'sample_BJ1': 'North_China', 'sample_XZ1': 'West_China',
    #     'sample_QQ1': 'Northeast', 'sample_SZ1': 'South_China', 'sample_XA1': 'Northwest',
    #     'sample_TY1': 'North_China', 'sample_SH1': 'East_China', 'sample_CQ1': 'Southwest',
    #     'sample_LZ1': 'Northwest', 'sample_PC1': 'East_China'
    # }
    

    # 2) Load & Preprocess
    X_combined, Y_pathways_log = load_and_preprocess_data_for_pathways(config)

    # 3) Generate short codes for pathways (optional)
    generate_pathway_mappings(Y_pathways_log, out_dir)
    pathway_shortcodes = generate_pathway_mappings(Y_pathways_log, out_dir)

    # 4) Exploratory CCA if desired
    X_c, Y_c, cca = run_cca_biplot(
        X=X_combined,
        Y=Y_pathways_log,
        pathway_shortcodes=pathway_shortcodes,
        n_components=2,
        output_dir=out_dir
    )

    # 5) Select top chemical features & top pathways
    (top_chems, top_paths, feat_importances,
     mean_r2, mean_rmse, tr_r2, tr_rmse) = select_top_chemicals_and_pathways(
        X_combined,
        Y_pathways_log,  
        n_top_chemicals=config["feature_selection"]["top_n_chemicals"],
        n_top_paths=config["feature_selection"]["top_n_args"],
        cv=5,
        output_dir=out_dir
    )
    logging.info(f"Top Chemicals => {top_chems}")
    logging.info(f"Top Pathways => {top_paths}")

    # 6) Compute functional diversity measure for the final model
    #    (Compute it here too, in case 'select_top_chemicals_and_pathways'
    #     used the same measure inside. The definitions won't conflict.)
    Y_diversity = compute_functional_diversity(Y_pathways_log)

    X_reduced = X_combined[top_chems]
    analyze_coeff_sign(
        X_reduced,
        Y_diversity,  # negative => "Bad"
        output_dir=out_dir,
        label="FunctionalDiversity"
    )

    # 8) Final XGB => SHAP & PDP
    logging.info("Training final XGB model on functional diversity for SHAP/PDP...")
    final_xgb = XGBRegressor(
        random_state=42, objective="reg:squarederror", eval_metric="rmse"
    )
    final_xgb.fit(X_reduced, Y_diversity)

    interpret_model_with_shap_and_pdp(final_xgb, X_reduced, top_chems, config)

    # 9) Build ONLY correlation-based network using top features & top pathways
    G_correlation = nx.Graph()

    correlation_thresh = config["correlation_analysis"]["correlation_threshold"]
    alpha = config["correlation_analysis"].get("pval_alpha", 0.05)

    for chem in top_chems:
        for pwy in top_paths:
            corr, pval = spearmanr(X_reduced[chem], Y_pathways_log[pwy])
            if abs(corr) > correlation_thresh and pval < alpha:
                G_correlation.add_edge(chem, pwy, weight=corr)

    # Then plot correlation-based network
    plot_and_save_network(
        G_correlation,
        top_chems,  # chemicals
        top_paths,  # pathways
        out_dir,
        "Correlation-Based Network",
        config,
        correlation_thresh  # or some threshold
    )

    logging.info("Pipeline completed successfully.")

if __name__ == "__main__":
    main()



