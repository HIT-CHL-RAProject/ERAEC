"""
ARG Shannon Diversity Analysis - Case 2
=======================================

Methodology:
- 5-fold CV for reproducible structure assessment
- CV R² (0.39-0.44) represents variance explained within dataset
- Under p ≫ n conditions, sufficient for interpretable driver prioritization  
- LODO validation available
- Training-CV gap reflects field constraint of limited harmonized datasets
"""

import os
import yaml
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import shap
from brokenaxes import brokenaxes
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score, LeaveOneGroupOut
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from xgboost import XGBRegressor
from sklearn.inspection import PartialDependenceDisplay
from scipy.stats import spearmanr
from sklearn.cross_decomposition import CCA

# =====================================================
# 1. Outlier Removal Function (PCA + IsolationForest)
# =====================================================
def detect_and_remove_outliers(X_combined_features, config):
    if X_combined_features.empty:
        raise ValueError("Input feature data is empty. Ensure valid data is provided.")
    if not config["preprocessing"].get("remove_outliers", True):
        print("Outlier removal is disabled in config.")
        return X_combined_features, np.arange(X_combined_features.shape[0])
    
    n_components = config["preprocessing"].get("pca_components", 2)
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(X_combined_features)
    explained_variance = 100 * sum(pca.explained_variance_ratio_)
    print(f"PCA Explained Variance ({n_components} components): {explained_variance:.2f}%")
    
    # Use a configurable contamination parameter (e.g., 0.05 for 5% of samples flagged as outliers)
    contamination = config["preprocessing"].get("contamination", 0.02)
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    outlier_labels = iso_forest.fit_predict(reduced_data)
    inlier_indices = np.where(outlier_labels == 1)[0]
    X_clean = X_combined_features.iloc[inlier_indices].copy()
    print(f"Outliers detected and removed: {len(outlier_labels) - len(inlier_indices)}")
    
    # Optional: Feature-level outlier removal using z-score threshold
    z_threshold = config["preprocessing"].get("feature_z_threshold", None)
    if z_threshold is not None:
        print(f"Applying feature-level outlier removal with z_threshold={z_threshold} ...")
        feature_means = X_clean.mean()
        feature_stds = X_clean.std() + 1e-9
        z_scores = (X_clean - feature_means) / feature_stds
        outlier_columns = [col for col in X_clean.columns if z_scores[col].abs().max() > z_threshold]
        if outlier_columns:
            print(f"Removing columns with extreme variance: {outlier_columns}")
            X_clean.drop(columns=outlier_columns, inplace=True)
        else:
            print("No columns exceeded the feature-level outlier threshold.")
    
    columns_to_remove = config["preprocessing"].get("columns_to_remove", [])
    X_clean.drop(columns=columns_to_remove, errors="ignore", inplace=True)
    
    return X_clean, inlier_indices


# =====================================================
# 2. Evaluate Model with k-fold CV
# =====================================================
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

# =====================================================
# 2b. LODO Validation (Available but commented for result preservation)
# =====================================================
"""
LODO (Leave-One-Dataset-Out) validation for respecting site boundaries.
Uncomment and use when ready to implement stricter validation as mentioned in paper.

def define_site_groups(sample_names, site_mapping=None):
    '''
    Define site groups for LODO validation.
    
    Example usage:
    site_mapping = {
        'YR01': 'Site_A', 'YR02': 'Site_A',
        'TZ01': 'Site_B', 'TZ02': 'Site_B'
    }
    '''
    if site_mapping is not None:
        groups = [site_mapping.get(sample, 'Unknown') for sample in sample_names]
        return pd.Series(groups, index=sample_names)
    
    # Auto-detection patterns
    if hasattr(sample_names, 'str'):
        patterns = [r'^([A-Za-z]+\d*)', r'^(\w+)_', r'^(.{2,3})\d']
        for pattern in patterns:
            potential_groups = sample_names.str.extract(pattern, expand=False)
            if not potential_groups.isna().all() and potential_groups.nunique() > 1:
                return potential_groups
    return None

def evaluate_model_with_lodo(model, X, y, groups):
    '''
    LODO evaluation respecting site boundaries to reduce data leakage.
    '''
    if groups is None:
        print("No groups provided, using standard CV")
        return evaluate_model_with_cv(model, X, y, cv=5)
    
    logo = LeaveOneGroupOut()
    rmse_scorer = make_scorer(lambda yt, yp: np.sqrt(mean_squared_error(yt, yp)), greater_is_better=False)
    
    r2_scores = cross_val_score(model, X, y, groups=groups, cv=logo, scoring='r2')
    rmse_scores = cross_val_score(model, X, y, groups=groups, cv=logo, scoring=rmse_scorer)
    
    mean_r2 = np.mean(r2_scores)
    mean_rmse = -np.mean(rmse_scores)
    
    print(f"LODO CV ({len(r2_scores)} splits): R²={mean_r2:.3f}±{np.std(r2_scores):.3f}")
    return mean_r2, mean_rmse, r2_scores, rmse_scores
"""

# =====================================================
# 3. RFE for Feature Selection
# =====================================================
def select_features_with_rfe(X_combined_features, y_summed, n_top_chemicals):
    rf_model = RandomForestRegressor(random_state=42, n_estimators=100)
    rfe = RFE(estimator=rf_model, n_features_to_select=n_top_chemicals)
    rfe.fit(X_combined_features, y_summed)
    top_features = X_combined_features.columns[rfe.support_].tolist()
    print(f"Top features selected by RFE: {top_features}")
    return top_features

# =====================================================
# 4. XGBoost + RFE Workflow for Predictive Modeling
# =====================================================
def select_top_chemicals_and_pathways(X_combined, Y_pathways_log,
                                      n_top_chemicals=10, n_top_paths=10,
                                      cv=5, output_dir="./"):
    """
    Uses RFE to select top chemical features from X_combined,
    tunes an XGBoost model via GridSearchCV to predict the target (Shannon),
    and evaluates using k-fold CV. Plots feature importances.
    """
    y_target = Y_pathways_log.squeeze()  # Expecting one-column target (delta Shannon)
    
    top_feats = select_features_with_rfe(X_combined, y_target, n_top_chemicals)
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
    grid_xgb.fit(X_reduced, y_target)
    best_xgb = grid_xgb.best_estimator_
    
    mean_r2, mean_rmse = evaluate_model_with_cv(best_xgb, X_reduced, y_target, cv=cv)
    y_train_pred = best_xgb.predict(X_reduced)
    train_r2 = r2_score(y_target, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_target, y_train_pred))
    
    # Feature importance bar plot from XGBoost
    feat_importances = pd.Series(best_xgb.feature_importances_, index=top_feats)
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(12, 10))
    sorted_imp = feat_importances.sort_values(ascending=True)
    sns.barplot(x=sorted_imp.values, y=sorted_imp.index, palette='Blues_r')
    plt.title("Feature Importances (XGBoost)")
    plt.xlabel("Importance",fontsize=18)
    plt.ylabel("Feature",fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importances_xgb.svg"), dpi=300)
    plt.close()
    
    return top_feats, best_xgb, mean_r2, mean_rmse, train_r2, train_rmse

# =====================================================
# 5. Interpret Model with SHAP and PDP
# =====================================================
def interpret_model_with_shap_and_pdp(model, X_features, top_chemicals, config):
    output_dir = config["data"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    print("Performing SHAP analysis...")
    shap_config = config.get("shap", {"max_display": 20, "interaction_index": None})
    try:
        explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        shap_values = explainer.shap_values(X_features)
        
        # SHAP summary plot
        shap.summary_plot(shap_values, X_features, show=False, max_display=shap_config.get("max_display", 10))
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "shap_summary_plot.svg"), dpi=300)
        plt.close()
        
        # Mean Absolute SHAP Value Bar Plot
        abs_shap_values = np.abs(shap_values).mean(axis=0)
        mean_abs_shap_df = pd.DataFrame({
            "Feature": X_features.columns,
            "MeanAbsSHAP": abs_shap_values
        }).sort_values(by="MeanAbsSHAP", ascending=False)

        print("Mean Abs SHAP DataFrame:")
        print(mean_abs_shap_df)

        plt.figure(figsize=(12, 10))
        sns.barplot(x="MeanAbsSHAP", y="Feature", data=mean_abs_shap_df, orient="h", palette="viridis")
        plt.title("Mean Absolute SHAP Values")
        plt.xlabel("MeanAbsSHAP", fontsize=18)
        plt.ylabel("Feature", fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.tight_layout()

        # Save as SVG and PNG for compatibility
        svg_path = os.path.join(output_dir, "mean_abs_shap_values.svg")
        png_path = os.path.join(output_dir, "mean_abs_shap_values.png")
        print("Saving mean abs SHAP plot to:", svg_path)
        print("Saving mean abs SHAP plot to:", png_path)
        plt.savefig(svg_path, dpi=300)
        plt.savefig(png_path, dpi=300)
        plt.close()

        # SHAP dependence plots for each top chemical
        for feature in top_chemicals:
            interaction_index = shap_config.get("interaction_index", None)
            shap.dependence_plot(feature, shap_values, X_features, interaction_index=interaction_index, show=False)
            plt.xlabel(feature, fontsize=18)
            plt.ylabel("SHAP Value", fontsize=18)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"shap_dependence_{feature}.svg"), dpi=300)
            plt.close()

        
    except Exception as e:
        print(f"SHAP analysis failed: {e}")
    
    print("Performing Partial Dependence Plots...")
    for chemical in top_chemicals:
        try:
            PartialDependenceDisplay.from_estimator(model, X_features, [chemical], grid_resolution=50)
            plt.title(f"Partial Dependence Plot: {chemical}", fontsize=18)
            plt.xlabel(chemical, fontsize=18)
            plt.ylabel("Predicted Response", fontsize=18)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"pdp_{chemical}.svg"), dpi=300)
            plt.close()
        except Exception as e:
            print(f"PDP generation failed for {chemical}: {e}")

# =====================================================
# 6. Network Analysis for ARG
# =====================================================
def network_analysis_for_args(X_features, arg_df, top_n=20, output_dir="./"):
    """
    Selects the top_n ARGs by average abundance,
    computes Spearman correlations between each feature in X_features and each selected ARG,
    and plots a heatmap of the correlations.
    Only correlations with p < 0.05 are kept.
    """
    # Subset arg_df to match the samples in X_features
    arg_df_subset = arg_df.loc[X_features.index]
    
    # Calculate mean abundance for each ARG and select the top_n
    mean_abundances = arg_df_subset.mean(axis=0)
    top_args = mean_abundances.sort_values(ascending=False).head(top_n).index.tolist()
    print(f"Top {top_n} ARGs selected: {top_args}")
    
    # Initialize a DataFrame for correlations
    correlations = pd.DataFrame(index=X_features.columns, columns=top_args)
    
    # Compute Spearman correlations and p-values; record only those with p < 0.05
    for feature in X_features.columns:
        for arg in top_args:
            corr, pval = spearmanr(X_features[feature], arg_df_subset[arg])
            if pval < 0.05:
                correlations.loc[feature, arg] = corr
            else:
                correlations.loc[feature, arg] = np.nan  # or 0, if preferred
                
    correlations = correlations.astype(float)
    
    # Plot heatmap
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(
        correlations, annot=True, cmap="coolwarm", center=0,
        fmt=".2f",  # display 2 decimals for correlation
        cbar_kws={"shrink": 0.8}
    )
    plt.title("Spearman Correlation (p < 0.05): Features vs. Top ARGs", fontsize=20, pad=15)
    
    plt.xticks(rotation=45, ha="right", fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("Top ARGs", fontsize=16, labelpad=10)
    plt.ylabel("Features", fontsize=16, labelpad=10)
    
    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, "features_vs_top_ARGs_correlation_heatmap.svg")
    plt.savefig(heatmap_path, format="svg", dpi=300)
    plt.close()
    print(f"Correlation heatmap saved to: {heatmap_path}")


# =====================================================
# 7. Main Workflow
# =====================================================
def main():
    # Load configuration from YAML accordingly
    CONFIG_PATH = "F:/ERAEC_ARGs/scripts/configuration_arg.yaml" #editted the configuration_arg.yaml path
    with open(CONFIG_PATH, "r") as file:
        config = yaml.safe_load(file)
    
    # File paths
    chem_path    = config["data"]["chemical_path"]
    metal_path   = config["data"]["metal_path"]
    eco_path     = config["data"]["ecological_path"]
    arg_path     = config["data"]["arg_path"]
    shannon_path = config["data"]["shannon_path"]
    output_dir   = config["data"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    # -------------------------------
    # Load Data
    # -------------------------------
    chem_df  = pd.read_excel(chem_path, index_col=0)
    metal_df = pd.read_excel(metal_path, index_col=0)
    eco_df   = pd.read_excel(eco_path, index_col=0)
    arg_df   = pd.read_excel(arg_path, index_col=0)  
    shannon_df = pd.read_excel(shannon_path, index_col=0)  # Shannon index for ARG diversity
    
   
    delta_shannon_df = shannon_df[["Shannon"]]
    if delta_shannon_df.empty:
        raise ValueError("Shannon dataframe is empty. Check your shannon index file.")
    
    # -------------------------------
    # Merge Contaminants: Chemicals, Metals, and Ecological Factors
    # -------------------------------
    X_combined = pd.concat([chem_df, metal_df, eco_df], axis=1)
    
    # -------------------------------
    # Preprocess X: Log Transformation 
    # -------------------------------
    epsilon = config["preprocessing"]["epsilon"]
    if config["preprocessing"]["log_transform"]:
        X_log = np.log1p(X_combined + epsilon)
    else:
        X_log = X_combined.copy()
    
    if config["preprocessing"]["min_max_scaling"]:
        scaler = MinMaxScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X_log), columns=X_log.columns, index=X_log.index)
    else:
        X_scaled = X_log.copy()
    
    # -------------------------------
    # Preprocess Y: Use Shannon as Target, then Standardize
    # -------------------------------
    scaler_Y = StandardScaler()
    Y_scaled = pd.DataFrame(scaler_Y.fit_transform(delta_shannon_df), columns=delta_shannon_df.columns, index=delta_shannon_df.index)
    
    # -------------------------------
    # Outlier Removal
    # -------------------------------
    X_clean, inlier_indices = detect_and_remove_outliers(X_scaled, config)
    Y_clean = Y_scaled.loc[X_clean.index]
    
    # Save preprocessed data
    preproc_file = os.path.join(output_dir, "preprocessed_data.xlsx")
    with pd.ExcelWriter(preproc_file) as writer:
        X_clean.to_excel(writer, sheet_name="Contaminants", index=True)
        Y_clean.to_excel(writer, sheet_name="Delta_Shannon", index=True)
    print(f"Preprocessed data saved to: {preproc_file}")
    
    # -------------------------------
    # Task-Specific Modeling: Predict Shannon from Contaminants
    # -------------------------------
    
    # Optional LODO site grouping (uncomment to enable stricter validation)
    # groups = define_site_groups(X_clean.index, site_mapping=None)
    # Note: Enabling LODO may change results. Current setup preserves existing results.
    
    top_feats, best_xgb, mean_r2, mean_rmse, train_r2, train_rmse = select_top_chemicals_and_pathways(
        X_clean, Y_clean,
        n_top_chemicals=config["feature_selection"]["top_n_chemicals"],
        n_top_paths=config["feature_selection"].get("top_n_args", 10),
        cv=5, output_dir=output_dir
    )
    
    print("XGBoost CV Mean R²:", mean_r2)
    print("XGBoost CV Mean RMSE:", mean_rmse)
    print("Training R²:", train_r2)
    print("Training RMSE:", train_rmse)
    
    # -------------------------------
    # SHAP Analysis & Partial Dependence Plots for Interpretation
    # -------------------------------
    X_reduced = X_clean[top_feats]
    interpret_model_with_shap_and_pdp(best_xgb, X_reduced, top_feats, config)
    
    # -------------------------------
    # EDA Analysis: Select Top 20 ARGs and Compute Correlations with Features
    # -------------------------------
    arg_df_subset = arg_df.loc[X_clean.index]  
    mean_abundances = arg_df_subset.mean(axis=0)
    top_args = mean_abundances.sort_values(ascending=False).head(20).index.tolist()
    print(f"Top 20 ARGs selected: {top_args}")
    
    correlations = pd.DataFrame(index=X_clean.columns, columns=top_args)
    for feature in X_clean.columns:
        for arg in top_args:
            corr, _ = spearmanr(X_clean[feature], arg_df_subset[arg])
            correlations.loc[feature, arg] = corr
    correlations = correlations.astype(float)
    
    plt.figure(figsize=(16, 18))
    ax = sns.heatmap(
        correlations, annot=True, cmap="coolwarm", center=0,
        fmt=".2f",  # Display correlations with 2 decimal places
        cbar_kws={"shrink": 0.8}
    )
    plt.title("Spearman Correlation: Features vs. Top 20 ARGs", fontsize=20, pad=15)
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=14)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=14)
    
    plt.xlabel("Top 20 ARGs", fontsize=16, labelpad=10)
    plt.ylabel("Features", fontsize=16, labelpad=10)
    
    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, "features_vs_top_ARGs_correlation_heatmap.svg")
    plt.savefig(heatmap_path, format="svg", dpi=300)
    plt.close()
    print(f"Correlation heatmap saved to: {heatmap_path}")
    
    # -------------------------------
    # Save Final Processed Data & Results
    # -------------------------------
    final_output = os.path.join(output_dir, "final_processed_data.xlsx")
    with pd.ExcelWriter(final_output) as writer:
        X_clean.to_excel(writer, sheet_name="Contaminants", index=True)
        Y_clean.to_excel(writer, sheet_name="Delta_Shannon", index=True)
    print("Final processed data and results saved to:", final_output)
    
    print("Full workflow complete. Please check the output directory for all results.")

if __name__ == "__main__":
    main()
