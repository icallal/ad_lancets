import numpy as np
from sklearn.metrics import (
    auc,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    matthews_corrcoef,
)
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
from matplotlib.font_manager import FontProperties
import argparse
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, mean_squared_error
import scipy.stats as st
import sys

sys.path.append("./ukb_func")
from ml_utils import concat_labels_and_probas, probas_to_results
import seaborn as sns

import ptitprince as pt

# Set font properties
plt.rcParams.update(
    {"font.size": 12, "font.weight": "bold"}
)

def _choose_plot_title(dirpath):
    """
    Choose the appropriate plot title based on the directory path.
    """
    if "apoe-only" in dirpath:
        return "APOE Only"
    elif "LDE_only" in dirpath:
        return "Full SNP Panel"
    elif "age_alone" in dirpath:
        return "Age Only"
    elif "all_demographics" in dirpath:
        if "apoe" in dirpath:
            return "Demographics + APOE"
        elif "LDE" in dirpath:
            return "Demographics + Full SNP Panel"
        else:
            return "Demographics"
    elif "demographics_and_lancet2024" in dirpath:
        if "apoe" in dirpath:
            return "Demographics + APOE + Lancet"
        elif "LDE" in dirpath:
            return "Demographics + Full SNP Panel + Lancet"
        else:
            return "Demographics + Lancet"
    else:
        return "Unknown Experiment"

def _save_plot(fig, curve_type, filepath, metric, image_format):
    """Save the plot figure to a file."""
    fname = f"{filepath}/{curve_type}_curve_{metric}_lgbm_all_expts_mean.{image_format}"
    fig.savefig(fname, facecolor="white", transparent=False, dpi=300)
    plt.close()

def _initialize_roc_plot():
    """Initialize a plot for ROC curves."""
    fig, ax = plt.subplots(figsize=(7, 6))
    mean_fpr = np.linspace(0, 1, 100)
    
    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", alpha=0.8)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", weight="bold", size=20)
    ax.set_ylabel("True Positive Rate", weight="bold", size=20)
    
    return fig, ax, mean_fpr

def mean_roc_curve(true_labels_list, predicted_probs_list):
    """Calculate the mean ROC curve from multiple ROC curves."""
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    auc_l = []
    auc_maxfpr025_l = []

    for true_labels, predicted_probs in zip(true_labels_list, predicted_probs_list):
        rocauc = roc_auc_score(true_labels, predicted_probs)
        auc_l.append(rocauc)

        rocauc_maxfpr025 = roc_auc_score(true_labels, predicted_probs, max_fpr=0.25)
        auc_maxfpr025_l.append(rocauc_maxfpr025)

        fpr, tpr, _ = roc_curve(true_labels, predicted_probs)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs, axis=0)

    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(auc_l)

    mean_auc_maxfpr025 = np.mean(auc_maxfpr025_l)
    std_auc_maxfpr025 = np.std(auc_maxfpr025_l)

    return mean_tpr, std_tpr, mean_auc, std_auc, mean_auc_maxfpr025, std_auc_maxfpr025

def multi_mean_roc_curve(experiment_paths, metric, image_format, output_dir="./plots"):
    """Plot multiple mean ROC curves for different experiments."""
    os.makedirs(output_dir, exist_ok=True)
    
    colors = [
        "#ff0000", "#ff7f00", "#ffae00", "#fff500", 
        "#a2ff00", "#00ff29", "#00ffce", "#00c9ff",
        "#2700ff", "#ab00ff"
    ]

    fig, ax, mean_fpr = _initialize_roc_plot()

    for i, exp_path in enumerate(experiment_paths):
        print(f"Processing experiment: {exp_path}")
        
        try:
            true_labels, probas = concat_labels_and_probas(exp_path)
            title = _choose_plot_title(exp_path)

            (mean_tpr, std_tpr, mean_auc, std_auc, 
             mean_auc_maxfpr025, std_auc_maxfpr025) = mean_roc_curve(true_labels, probas)

            label = f"{title}\nAUC: {mean_auc:.3f} $\pm$ {std_auc:.3f}"
            ax.plot(mean_fpr, mean_tpr, color=colors[i % len(colors)], 
                   label=label, lw=2, alpha=0.8)

            # Plot standard deviation
            tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
            tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
            ax.fill_between(mean_fpr, tpr_lower, tpr_upper, 
                           color=colors[i % len(colors)], alpha=0.2)
        except Exception as e:
            print(f"Error processing {exp_path}: {e}")
            continue

    font_prop = FontProperties(weight="bold", size=10)
    ax.legend(loc="lower right", prop=font_prop)
    plt.tight_layout()
    
    fname = f"{output_dir}/alzheimers_roc_curve_{metric}_lgbm_all_expts_mean.{image_format}"
    fig.savefig(fname, facecolor="white", transparent=False, dpi=300)
    plt.close()

def _initialize_pr_plot():
    """Initialize a plot for Precision-Recall curves."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_xlabel("Recall", weight="bold", size=20)
    ax.set_ylabel("Precision", weight="bold", size=20)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.25, 1.25])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    return fig, ax

def _mean_pr_curve(true_labels_list, predicted_probs_list):
    """Calculate the mean Precision-Recall curve."""
    precision_list = []
    recall_list = []
    ap_list = []

    for true_labels, predicted_probs in zip(true_labels_list, predicted_probs_list):
        precision, recall, _ = precision_recall_curve(true_labels, predicted_probs)
        precision_list.append(
            np.interp(np.linspace(0, 1, 100), recall[::-1], precision[::-1])
        )
        recall_list.append(np.linspace(0, 1, 100))

        ap = average_precision_score(true_labels, predicted_probs)
        ap_list.append(ap)

    mean_precision = np.mean(precision_list, axis=0)
    std_precision = np.std(precision_list, axis=0)
    mean_recall = np.mean(recall_list, axis=0)
    mean_ap = np.mean(ap_list)
    std_ap = np.std(ap_list)

    return mean_precision, std_precision, mean_recall, mean_ap, std_ap

def multi_mean_pr_curve(experiment_paths, metric, image_format, output_dir="./plots"):
    """Plot multiple mean Precision-Recall curves for different experiments."""
    os.makedirs(output_dir, exist_ok=True)
    
    colors = [
        "#ff0000", "#ff7f00", "#ffae00", "#fff500",
        "#a2ff00", "#00ff29", "#00ffce", "#00c9ff", 
        "#2700ff", "#ab00ff"
    ]

    fig, ax = _initialize_pr_plot()

    for i, exp_path in enumerate(experiment_paths):
        print(f"Processing experiment: {exp_path}")
        
        try:
            true_labels, probas = concat_labels_and_probas(exp_path)
            title = _choose_plot_title(exp_path)

            mean_precision, std_precision, mean_recall, mean_ap, std_ap = (
                _mean_pr_curve(true_labels, probas)
            )

            ax.plot(
                mean_recall, mean_precision,
                color=colors[i % len(colors)],
                label=f"{title}\nAP: {mean_ap:.3f} $\pm$ {std_ap:.3f}",
                lw=2, alpha=0.8,
            )
            ax.fill_between(
                mean_recall,
                mean_precision - std_precision,
                mean_precision + std_precision,
                color=colors[i % len(colors)], alpha=0.2,
            )
        except Exception as e:
            print(f"Error processing {exp_path}: {e}")
            continue

    ax.legend(loc="upper right", prop=FontProperties(weight="bold", size=10))
    plt.tight_layout()
    
    fname = f"{output_dir}/alzheimers_pr_curve_{metric}_lgbm_all_expts_mean.{image_format}"
    fig.savefig(fname, facecolor="white", transparent=False, dpi=300)
    plt.close()

def mcc_raincloud(experiment_paths, output_dir="./plots", image_format="pdf"):
    """Generate MCC raincloud plot for all experiments."""
    os.makedirs(output_dir, exist_ok=True)
    
    colors = [
        "#ff0000", "#ff7f00", "#ffae00", "#fff500",
        "#a2ff00", "#00ff29", "#00ffce", "#00c9ff",
        "#2700ff", "#ab00ff"
    ]

    res_l = []
    titles_l = []
    
    for exp_path in experiment_paths:
        try:
            _, test_results = probas_to_results(exp_path, youden=True)
            res_l.append(test_results)
            titles_l.append(_choose_plot_title(exp_path))
        except Exception as e:
            print(f"Error processing {exp_path}: {e}")
            continue

    # Extract MCC values for each experiment
    y = [i.mcc for i in res_l]

    # Prepare data for plotting
    data = []
    for i, category in enumerate(titles_l):
        category = category.replace(" + ", "\n+\n")
        for value in y[i]:
            data.append([category, value])
    
    data = pd.DataFrame(data, columns=["", "Matthews Correlation Coefficient"])

    fig, ax = plt.subplots(figsize=(12, 8))

    pt.RainCloud(
        x="", y="Matthews Correlation Coefficient",
        data=data, palette=colors[:len(titles_l)],
        bw=0.2, ax=ax, orient="v",
        box_linewidth=1, offset=0.2, move=0.2, width_viol=0.6,
    )

    plt.subplots_adjust(right=0.95)
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max * 1.1)

    ax.set_ylabel("MCC", fontweight="bold")
    plt.tight_layout()

    fname = f"{output_dir}/alzheimers_mcc_raincloud_plot_lgbm_vertical.{image_format}"
    fig.savefig(fname, facecolor="white", transparent=False, dpi=300)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot results for Alzheimers experiments")
    parser.add_argument("metric", type=str, help="metric (e.g., log_loss)")
    parser.add_argument("image_format", type=str, help="image format (pdf, png)")
    parser.add_argument("--output_dir", type=str, default="./plots", 
                       help="output directory for plots")

    args = parser.parse_args()

    # Your experiment paths
    experiment_paths = [
        './results/apoe-only', 
        './results/LDE_only', 
        './results_all/age_alone/none/allages/AD/lgbm',
        './results_all/all_demographics/apoe/allages/AD/lgbm',
        './results_all/all_demographics/LDE/allages/AD/lgbm',
        './results_all/demographics_and_lancet2024/none/allages/AD/lgbm',
        './results_all/demographics_and_lancet2024/apoe/allages/AD/lgbm', 
        './results_all/demographics_and_lancet2024/LDE/allages/AD/lgbm'
    ]

    print("Generating ROC curves...")
    multi_mean_roc_curve(experiment_paths, args.metric, args.image_format, args.output_dir)
    
    print("Generating PR curves...")
    multi_mean_pr_curve(experiment_paths, args.metric, args.image_format, args.output_dir)
    
    print("Generating MCC raincloud plot...")
    mcc_raincloud(experiment_paths, args.output_dir, args.image_format)
    
    print(f"All plots saved to {args.output_dir}")
